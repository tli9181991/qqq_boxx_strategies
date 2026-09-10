# Deploying the Top-6 vol-targeted book to IB paper

A t3.small that wakes up, trades one closing auction, and shuts down again.

```
08:45 ET  AWS starts the instance; IB Gateway autologs in
08:50 ET  qbs-preflight   data + signal + broker, sends nothing
15:30 ET  qbs-trade       rank, build the order list, submit MOC
15:45 ET  exchange MOC cutoff  <- the hard deadline the 15:30 job is racing
16:00 ET  the close; orders fill in the auction
16:15 ET  qbs-reconcile   fills, final positions, cancel stragglers
16:20 ET  AWS stops the instance
```

Local-time equivalents, since the timers follow US daylight saving:

| Phase | ET | HK (Mar–Nov) | HK (Nov–Mar) |
|---|---|---|---|
| Instance start + Gateway | 08:45 | 20:45 | 21:45 |
| Preflight | 08:50 | 20:50 | 21:50 |
| Rank + submit | 15:30 | 03:30 | 04:30 |
| MOC cutoff | 15:45 | 03:45 | 04:45 |
| Close | 16:00 | 04:00 | 05:00 |
| Reconcile + stop | 16:15 | 04:15 | 05:15 |

---

## Install

```bash
git clone https://github.com/tli9181991/qqq_boxx_strategies.git
cd qqq_boxx_strategies
sudo ./deploy/install.sh
```

The installer adds a 2 GB swapfile, installs `tzdata`, creates the `qbs`
service user, syncs the repo to `/opt/qbs`, builds a venv from
`requirements-live.txt`, and installs the systemd units — but **starts
nothing**. Walk through the checklist it prints.

The broker layer uses [`ib_async`](https://github.com/ib-api-reloaded/ib_async),
the maintained fork of `ib_insync`. If your existing paper-trade script imports
`ib_insync`, only the import line changes — the API is the same. Do not install
both into the same environment: they patch asyncio the same way and whichever
imports second wins.

**Needs Python 3.10+** (an `ib_async` requirement). Ubuntu 22.04 and 24.04 are
fine; Amazon Linux 2 ships 3.7, so install `python3.11` and run the installer as
`PYTHON=python3.11 sudo -E ./deploy/install.sh`.

### Why the swapfile

A t3.small has 2 GB. IB Gateway's JVM takes most of 1 GB, and the trade job
holds a ~100-ticker × 1500-row price frame plus pandas. That fits, but not with
much room. Without swap the OOM killer will eventually pick the Python process
mid-session, and it will do it silently — you would find out from a missing
auction, not from an error. `vm.swappiness=10` keeps the kernel reclaiming page
cache before it swaps the Gateway out from under itself.

---

## First week: dry run

Set this in `/etc/qbs/live.env` before enabling the timers:

```bash
QBS_DRY_RUN=1
```

Every phase then runs in full — downloads, ranks, reconciles against real IB
positions, computes the exact order list, writes it to the run log with
`dry_run=1` — and sends nothing. Compare a few sessions against `python run_backtest.py` on the
same day. When the order lists stop surprising you, delete the line and
`sudo systemctl restart qbs-trade.timer`.

You asked for fully automatic from day one, so nothing forces this. It costs
you a few days and it is the only chance to see the order list before the
account acts on it.

---

## Configuration

Two files, both in `/etc/qbs`. The environment always wins over the JSON, so a
one-off override never means editing a tracked file.

| Setting | Default | What it does |
|---|---|---|
| `notional` | `100000` | Book size in dollars. **Fixed, not the account NLV** — see below. |
| `ib_port` | `4002` | Gateway paper. `4001`/`7496` are live and are refused unless `allow_live_account` is set. |
| `hold_safe_asset` | `true` | Hold BOXX as the cash leg. `false` leaves it as plain cash. |
| `max_gross_turnover` | `1.6` | Abort the session if the order list would trade >160% of the book. |
| `max_order_notional` | `40000` | Abort if any single order exceeds this. |
| `min_order_notional` | `250` | Skip rebalances too small to be worth the spread. |
| `moc_cutoff_hhmm` | `15:45` | Refuse to submit after this exchange-local time. |
| `min_universe_coverage` | `0.85` | Refuse to rank if fewer than 85% of NDX names downloaded. |

### Sizing is fixed, and does not compound

`notional` is a constant, not the account's net liquidation value. That makes
position sizes reproducible: any difference between two days' order lists is a
*signal* change, never an account-value change, which is what you want while
you are still deciding whether to trust the thing.

The cost is that it does not compound. As the paper account gains or loses, the
book stays the same dollar size until you edit the number. Revisit it monthly,
or switch to NLV-based sizing once the behaviour is boring.

---

## Operating it

```bash
# What is scheduled
systemctl list-timers 'qbs-*'

# Today's signal, no broker, no network writes
sudo -u qbs /opt/qbs/.venv/bin/python -m qbs.live.runner signal

# The same from the price cache, when the feed is down
sudo -u qbs /opt/qbs/.venv/bin/python -m qbs.live.runner signal --offline

# Force a phase now
sudo systemctl start qbs-preflight.service
journalctl -u qbs-trade -n 100 --no-pager

# The audit trail
sudo -u qbs /opt/qbs/.venv/bin/python -m qbs.live.runner report --days 20
sqlite3 -header -column /opt/qbs/var/qbs.db "SELECT * FROM portfolio_nav ORDER BY 1 DESC LIMIT 10"
jq '.last_trade' /opt/qbs/var/state.json
```

---

## The run log

Every trade, selection and end-of-day mark goes into one SQLite file,
`/opt/qbs/var/qbs.db`. `runner report` prints it; anything more specific is a
query.

| Table | One row per | Holds |
|---|---|---|
| `trade_events` | thing that happened | submissions, fills, cancellations, guard trips, skipped sessions |
| `selection_events` | (date, symbol, event) | `entry` / `exit` / `hold`, with the rank and 12-1 momentum the strategy actually used |
| `position_closes` | (date, symbol) | shares, close price, market value, unrealised P&L, target vs actual weight |
| `portfolio_nav` | date | book market value, NetLiquidation, cash, risk weight, the vol scalar |
| `signal_runs` | (date, phase) | what the signal said, kept for preflight *and* trade so you can see the intraday drift |

`trade_events` is **append-only** — a phase that ran twice really did submit
twice, and the log should say so. Everything else **upserts on its natural
key**, so re-running reconcile for a session corrects the row instead of
doubling it.

**Why `hold` rows and not just entries and exits.** The boring rows are the
ones you want later: when a position turns out badly, the question is how close
it was to being dropped, and that is only answerable if the rank was recorded
every day it survived. Those ranks come from the strategy's own rank map, not
from re-ranking afterwards — a re-derivation would have to reimplement the
eligibility and absolute-momentum filters, and a second copy of that logic is
exactly what drifts.

Some useful queries:

```sql
-- equity curve
SELECT session_date, total_market_value, net_liquidation, risk_weight, scalar
FROM portfolio_nav ORDER BY session_date;

-- how long each name was held, and at what rank it came in
SELECT symbol, session_date, event, rank, score
FROM selection_events WHERE event IN ('entry','exit') ORDER BY symbol, session_date;

-- submitted vs filled, to see auction slippage
SELECT s.symbol, s.quantity, s.price AS decided, f.price AS filled,
       (f.price - s.price) / s.price AS slip
FROM trade_events s JOIN trade_events f
  ON s.symbol = f.symbol AND s.session_date = f.session_date
WHERE s.event = 'submitted' AND f.event = 'filled';

-- every session the guards refused, and why
SELECT session_date, reason FROM trade_events WHERE event = 'guard_tripped';
```

In the notebook, `store.to_frame(db, "portfolio_nav")` hands any table straight
to pandas.

Close prices are **IB's own portfolio marks**, not a re-fetch from yfinance. At
16:15 ET the official close may not have reached a free feed yet, and a mark
that disagrees with the account it describes is worse than no mark.

As with `state.json`, nothing in this database feeds the signal. Delete it and
the next run still trades exactly the same book; you lose the history, not the
strategy.

### Kill switch

```bash
sudo -u qbs touch /opt/qbs/var/HALT     # stop trading, keep everything running
sudo -u qbs rm /opt/qbs/var/HALT        # resume
```

While that file exists, preflight and trade both refuse and exit non-zero.
Reconcile still runs, so you keep the end-of-day record. Use it whenever you
are not sure — a missed session costs nothing, because the strategy recomputes
from scratch every day.

---

## What each phase does when things go wrong

**Exit codes:** `0` fine (including "market closed"), `1` error, `2` config or
connection, `3` a safety guard refused the list.

| Symptom | What happened | What to do |
|---|---|---|
| `trade` exits 3, "GUARD TRIPPED" | The order list breached a limit. **Nothing was sent.** | Read the message. A turnover breach is nearly always a bad universe download, not a real signal. |
| `trade` logs "no bar for today" and exits 0 | Market holiday, or the feed is late. | Nothing. This is the designed behaviour. |
| `preflight` exits 2 | Gateway not up or not logged in. | It retries twice at 2-minute intervals. If it still fails you have ~6 hours to fix it before the auction. |
| `trade` fails, "past the 15:45 MOC cutoff" | The job ran long. | Nothing today. Check what was slow — usually the yfinance download. |
| Reconcile warns "signal wanted X but the book holds none of" | An MOC did not fill. | Usually a thin name. Tomorrow's run recomputes and re-sends. |

### The one thing that is safe about all of this

The strategy weights are recomputed from full price history on every run.
Nothing in `var/state.json` feeds the signal — delete the whole file and the
next run still produces exactly the right orders. So a missed session, a crash
mid-write, a manual intervention, none of them can put the live book on a
different path from the strategy. **Never "fix" a missed day by replaying it.**
Just let the next session run.

`Persistent=false` on the trade timer exists for the same reason: a maintenance
reboot at 16:30 must not fire a catch-up MOC into an auction that already
happened.

---

## AWS start/stop

The instance schedule lives outside this repo. Two EventBridge Scheduler rules
against the EC2 API, both in `America/New_York` so they track DST with the
timers:

```
qbs-start   cron(45 8 ? * MON-FRI *)   ec2:StartInstances
qbs-stop    cron(20 16 ? * MON-FRI *)  ec2:StopInstances
```

Neither knows about market holidays — the instance will boot on Thanksgiving.
That is harmless: `qbs-trade` sees no bar for today and exits 0 without
trading. You pay for the idle hours, not for a bad trade.

Set the Gateway to autologin (IBC or the Gateway's own setting) so the 08:50
preflight has something to connect to. Preflight retrying twice at 2-minute
intervals covers a slow login.

---

## Verifying live against the backtest

The live signal and the backtest run the same functions — `tests/test_live.py`
pins that with a test asserting the live path reproduces the backtest's final
weight row to 1e-12. To check by hand on any given day:

```bash
sudo -u qbs /opt/qbs/.venv/bin/python -m qbs.live.runner signal
cd /opt/qbs && sudo -u qbs .venv/bin/python run_backtest.py --offline --no-charts --no-fetch-universe
```

The `Top-6 vol-targeted` row's holdings and the signal's holdings must match.
If they ever diverge, stop trading (`touch var/HALT`) and find out why before
resuming — that is the single worst failure mode this deployment has.

---

## What this deployment does not do

- **No market-calendar library.** "Is the market open" is inferred from whether
  the feed printed a bar today. Simple and robust, but it cannot tell a holiday
  from a data outage — both skip the session.
- **No partial-fill handling within a session.** An unfilled MOC is corrected by
  the next day's run, not chased intraday.
- **No alerting.** Failures surface as failed systemd units and non-zero exits.
  Wire `OnFailure=` to something that emails you if you want to hear about it
  before the next morning.
- **No position-level stops.** The vol-target overlay sizes the book down as its
  risk rises, but a single name gapping overnight still costs a full sixth of
  the book. Resting GTC stops at the broker are the mitigation and they live
  outside this system.
