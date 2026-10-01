# Setting up the paper trader on the AWS VM (Docker)

Step by step, from a box that already has Docker and the `gnzsnz/ib-gateway`
image to a scheduled paper trader.

## First: three things about your compose file

Your `dockercompose.yml` will not work as written, for reasons worth
understanding before you fix them.

**1. `command: bash -c "pip install -r requirements.txt && python main.py"`**

There is no `main.py` in this repo, and there should not be. The app is
**three short-lived phases**, not a daemon:

```
08:50 ET  preflight   data + signal + broker + order list, sends nothing
15:30 ET  trade       rank, build, submit MOC before the 15:45 cutoff
16:15 ET  reconcile   fills, end-of-day marks, cancel stragglers
```

Each runs to completion and exits. `restart: unless-stopped` on that service
would restart the trade phase every time it finished — which, once you turn dry
run off, means resubmitting orders in a loop. The compose file here puts the
trader behind a `cli` profile so `up` can never start it, and the systemd timers
invoke it with `docker compose run --rm`.

Also: `requirements.txt` pulls JupyterLab, matplotlib and pytest. On a 2 GB
t3.small use `requirements-live.txt`, which is what the Dockerfile installs.

**2. `test_ib.py` connects to `127.0.0.1:4002`**

That works from the host, because the Gateway publishes 4002. It **fails inside
the container**, where there is no Gateway on loopback — it is a separate
container reachable as `ib-gateway`. A hardcoded loopback address passes the
test you run by hand and then fails in the place it actually matters.
`deploy/docker/test_ib.py` reads host and port from the environment so the same
script proves both paths.

**3. `ports: "4002:4002"` and `"5900:5900"` bind to every interface**

That publishes the IB API and VNC to `0.0.0.0`. The API port has no
authentication beyond IB's trusted-IP list; VNC has only a password. If your
security group ever opens those ports, anyone who finds them can trade your
account. The compose file here binds both to `127.0.0.1` and you reach them over
an SSH tunnel.

`version: '3.8'` is also obsolete in Compose v2 and just prints a warning.

---

## Step 1 — Get the code onto the box

```bash
sudo mkdir -p /opt/qbs && sudo chown "$USER:$USER" /opt/qbs
git clone https://github.com/tli9181991/qqq_boxx_strategies.git /opt/qbs
cd /opt/qbs
```

Everything below runs from `/opt/qbs`.

## Step 2 — Disk, then swap

A t3.small has 2 GB of RAM. The Gateway's JVM takes most of 1 GB, and the trade
phase holds a ~100-ticker × 1500-row price frame. It fits, but without swap the
OOM killer will eventually take the Python process mid-session — and it does
that silently. You would find out from a missing auction, not an error.

**Check the disk first.** A 2 GB swapfile needs 2 GB of free disk, and the
default 8 GB root volume plus the `ib-gateway` image (1–2 GB) does not leave it:

```bash
df -h /
docker system df        # images are usually what filled it
```

If free space is under ~2.5 GB, deal with that before going further:

```bash
# Best: grow the volume. 8 GB is too small for Gateway + this app + swap.
# 30 GB is still free-tier eligible. Modify the EBS volume in the console, then
# find your device with lsblk -- the name depends on the instance family:
#   Nitro (t3, t4g, m5...):  /dev/nvme0n1  partition /dev/nvme0n1p1
#   Xen   (t2, m4...):       /dev/xvda     partition /dev/xvda1
lsblk
sudo growpart /dev/nvme0n1 1      # <-- substitute what lsblk shows
sudo resize2fs /dev/nvme0n1p1     # <-- likewise; or: sudo xfs_growfs /
df -h /

# Or reclaim:
docker system prune -a
sudo apt-get clean && sudo journalctl --vacuum-size=100M
```

Then create the swapfile:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl -w vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
free -h        # confirm 2.0Gi of swap
```

If `fallocate` fails with **"No space left on device"**, stop and fix the disk —
and check for a partial file first, because a half-written swapfile is holding
space you need: `sudo swapoff /swapfile 2>/dev/null; sudo rm -f /swapfile`.

> **Do not run `deploy/install.sh`.** That is the native install path. It builds
> a venv and installs systemd units with the *same names* as the Docker ones, so
> whichever you install last silently wins. Everything you need is on this page.

## Step 3 — Fill in the environment file

```bash
cp deploy/docker/.env.example deploy/docker/.env
chmod 600 deploy/docker/.env
echo "QBS_UID=$(id -u)"; echo "QBS_GID=$(id -g)"     # put these in the file
nano deploy/docker/.env
```

Set `TWS_USERID`, `TWS_PASSWORD`, `VNC_PASSWORD`, and the `QBS_UID`/`QBS_GID`
you just printed. The UID matters: without it the container writes the run log
as root into your repo, and you then need sudo to read your own trade history.

**Leave `QBS_DRY_RUN=1`.** Step 9 turns it off.

`deploy/docker/.env` is gitignored — check with `git status` that it does not
appear before you ever push.

## Step 4 — Start the Gateway

Clear out anything left from an earlier attempt first. This matters more than it
looks: `container_name: ib-gateway` is fixed in the compose file, so a container
of that name from a previous run blocks the new one with *"container name
already in use"*. And `docker compose down` will **not** remove it if it came
from a different compose file — Compose only touches containers carrying its own
project label — so the removal has to be by name.

```bash
docker ps -a                                          # look before you delete
docker compose -f deploy/docker/docker-compose.yml down --remove-orphans
docker rm -f ib-gateway trading-bot 2>/dev/null || true
```

If this VM is dedicated to the trader and you would rather start from nothing:

```bash
docker rm -f $(docker ps -aq) 2>/dev/null || true     # removes EVERY container
```

Only on a box you are sure about — that takes out containers belonging to
anything else running here. It removes containers, not images, so nothing has to
be re-pulled.

Now start it:

```bash
docker compose -f deploy/docker/docker-compose.yml up -d ib-gateway
docker compose -f deploy/docker/docker-compose.yml logs -f ib-gateway
```

Wait for the login to complete — first start pulls the image and can take a few
minutes. Watch for the API listening on 4002. `Ctrl-C` stops following the log,
not the container.

If login fails, it is almost always the credentials or `TRADING_MODE`. Look at
the Gateway's own screen over VNC (step 5) rather than guessing.

## Step 5 — Watch the Gateway's screen, if you need to

VNC is bound to localhost, so tunnel to it. **From your laptop:**

```bash
ssh -N -L 5900:127.0.0.1:5900 ubuntu@YOUR_VM_IP
```

Then point a VNC client at `localhost:5900` with your `VNC_PASSWORD`. Use this
to confirm the API is enabled and to clear any dialog the Gateway is stuck on.
Close the tunnel when you are done.

## Step 6 — Prove the connection

First: **which port is the Gateway actually serving?** This image uses **4004**
inside the container, not the 4002 a host-installed Gateway uses. The image is
minimal — no `ss`, no `netstat` — so ask the kernel:

```bash
docker exec ib-gateway cat /proc/net/tcp | grep -i ' 0A '   # 0A = LISTEN
```

Column 2 is `IP:PORT` in hex. `0FA2` is 4002, `0FA4` is 4004; `00000000` means
it is listening on all interfaces (reachable from other containers), `0100007F`
means loopback only (not reachable). Ignore the `0B00007F` row — that is
Docker's internal DNS.

If yours is not 4004, set `IBG_PAPER_PORT` in `deploy/docker/.env` to whatever
it is. The compose file uses that for both the published port and the trader's
`QBS_IB_PORT`.

Then, host-side:

```bash
sudo ss -tlnp | grep 4002        # expect a LISTEN line on 127.0.0.1:4002
```

Then the test that matters — from inside the compose network, which is exactly
the path the trader takes:

```bash
docker compose -f deploy/docker/docker-compose.yml build
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps \
  --entrypoint python qbs test_ib.py
```

It must print `Connected` and a `DU…` account number; the script warns if the
account does not look like a paper one. The image already has `ib_async` pinned,
so nothing needs installing on the host for this.

### Optionally, the same test from the host

Useful only to split "the Gateway is broken" from "the compose network is
broken". Do **not** `pip install --user` — Ubuntu 24.04 and later mark the
system Python as externally managed (PEP 668) and will refuse. Use a venv:

```bash
sudo apt-get install -y python3-venv          # if it is not already there
python3 -m venv ~/.venv-qbs
~/.venv-qbs/bin/pip install -q ib_async
~/.venv-qbs/bin/python deploy/docker/test_ib.py
```

Never reach for `--break-system-packages` on this box. Ubuntu's own tooling runs
on that interpreter, and apt is not something you want to repair on a machine
that is supposed to be trading unattended.

**If the host test passes and the container test fails**, it is the hostname:
inside the network the Gateway is `ib-gateway`, never `127.0.0.1`.

## Step 7 — Check the signal, with no broker involved

> Every `run` below passes `--no-deps`, and so do the systemd units. That flag
> is not cosmetic: the trader `depends_on` the Gateway, so without it Compose
> re-evaluates the Gateway's configuration on each phase. The Gateway reads the
> whole of `.env`, so editing *any* variable there — including `QBS_DRY_RUN`,
> which only the trader uses — changes its config hash and Compose recreates the
> container, discarding a logged-in session. `--no-deps` makes that impossible,
> and costs nothing: the Gateway is already up.

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps qbs signal --offline
```

This touches neither the network nor IB — it ranks the cached price history and
prints today's target book. If this fails, the problem is the app, not the
plumbing, and you have separated the two.

Then the live-data version:

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps qbs signal
```

## Step 8 — Full preflight

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps qbs preflight
```

This is the whole path: downloads, ranks, connects to IB, reads your positions,
builds the exact order list, and **sends nothing**. Read the order table it
prints. That list is what the trade phase would submit.

Confirm the run log was written and is yours, not root's:

```bash
ls -l var/
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps qbs report
```

## Step 9 — Install the timers

```bash
sudo cp deploy/docker/systemd/qbs-*.service deploy/docker/systemd/qbs-*.timer \
  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qbs-preflight.timer qbs-trade.timer qbs-reconcile.timer
systemctl list-timers 'qbs-*'
```

`list-timers` should show the next run in **UTC** corresponding to 08:50 / 15:30
/ 16:15 New York. Check the arithmetic once — during EDT that is 12:50 / 19:30 /
20:15 UTC. If they show as 08:50 UTC, `tzdata` is missing on the host.

The timers now run with `QBS_DRY_RUN=1` from your `.env`. **Leave it that way
for a few sessions.** Compare each day's order list against the backtest:

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps qbs report --days 10
```

When the order lists stop surprising you, set `QBS_DRY_RUN=0` in
`deploy/docker/.env`. No restart is needed — each phase reads the file when it
runs.

### Add the six-stock residual-momentum paper sleeve

Keep the existing `QBS_NOTIONAL`; it remains the total account book. To fund
six residual-momentum picks at $6,000 per slot from BOXX, add:

```bash
# Keep this at its existing value (100000 in the standard deployment).
QBS_NOTIONAL=100000
QBS_RESMOM_NOTIONAL=36000
QBS_DRY_RUN=1
```

`QBS_RESMOM_NOTIONAL` is carved out of the main strategy's BOXX target: a
$100,000 book remains a $100,000 book rather than becoming a leveraged
$136,000 book. If MRVL occupies one $6,000 slot in both sleeves, its combined
target is $12,000; it is not deduplicated and spread across the other names.
The position cap widens by six on its own while the sleeve is on, so
`QBS_MAX_POSITIONS` needs no change.

**When BOXX is short.** The main strategy is vol-scaled, and on roughly half of
all sessions it wants less than $36,000 in BOXX. On those days the sleeve runs
at whatever BOXX there is (the log says so: `running the sleeve at $30,000
($5,000 a slot)`) rather than borrowing, and rather than failing the signal --
which would stop the main strategy trading too.

Run preflight and read the two labelled holding lists before allowing the paper
trade timer to send anything:

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps qbs preflight
```

Each successful preflight/trade run writes net model returns to
`var/strategy_comparison.csv`, rescoring the last ten sessions so a missed run
leaves no gap. Three lines are kept:

| strategy | what it is |
|---|---|
| `momentum` | the main book as traded: vol-scaled, so often mostly BOXX |
| `momentum6` | the main strategy's six picks in equal slots, no vol scaling |
| `resmom` | the six residual picks in equal slots |

`momentum6` against `resmom` is the like-for-like race: six stocks against six
stocks. After the trial, the normal report compounds each (after the
configured commission and slippage) and prints which six is ahead:

```bash
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps \
  qbs report --days 31
```

These are strategy-model returns, not an attempt to split the IB account's
combined P&L. That distinction matters for an overlap such as MRVL: IB holds
one aggregate position, and so does the ledger, while the comparison keeps one
$6,000 contribution in each strategy.

Set `QBS_RESMOM_NOTIONAL=0` to return to the original single strategy. This
switch does not enable a live IB account; the live-account guard remains
independent.

## Sharing an account with your own holdings

Skip this if the strategy has an account to itself — which is the arrangement to
prefer, and the only one where the broker enforces the separation rather than a
file on disk.

IB reports one position per symbol per account. If you hold 22 TSM yourself and
the strategy wants 5, the strategy sees 27, computes `5 - 27`, and **sells 17 of
your shares**. Overlap is not hypothetical: the current book holds MRVL.

`position_source` decides how the strategy recovers its own book:

| | how | breaks when |
|---|---|---|
| `ledger` | tally its own fills | a fill is never recorded |
| `baseline` | account minus a snapshot of what was yours | *you* trade those names |
| `account` | the whole account is the strategy's | anything else is in there |

**Use `ledger` for a shared account**, provided you never sell what the strategy
bought. Your own trading is unbounded and silent; a missed fill is bounded, makes
the reconcile unit fail, and is cross-checked on every run.

```bash
QBS_POSITION_SOURCE=ledger
```

Verify Compose actually forwards it to the short-lived trader container:

```bash
docker compose -f deploy/docker/docker-compose.yml config \
  | grep QBS_POSITION_SOURCE
```

It must print `QBS_POSITION_SOURCE: ledger`. The trader deliberately does not
receive the whole `.env` file because it contains the IB password, so each
non-secret trader setting has to be mapped explicitly in Compose.

**If you set `ledger` before this mapping existed**, the setting never reached
the container: every phase ran in `account` mode and nothing wrote
`var/strategy_trades.csv`. With the mapping in place the strategy now looks for
that file, and if it is missing while the account holds shares, preflight and
trade stop with a guard rather than read the book as flat and buy it twice.
Seed it once:

```bash
C="docker compose -f deploy/docker/docker-compose.yml"
$C run --rm --no-deps qbs ledger --rebuild     # the shares are the strategy's
$C run --rm --no-deps qbs ledger --start-flat  # every share is yours
$C run --rm --no-deps qbs ledger               # check: ledger vs account
```

`--rebuild` replays every fill in the run log, which was recorded in account
mode too. Older run logs hold the same fill more than once (every reconcile
re-logged the day's executions); the rebuild counts each execution once. A
ledger rebuilt before that fix can claim more than the account holds -- rebuild
it again with `ledger --rebuild --force`.

**A session the ledger missed.** IB reports only the current day's executions,
so a reconcile that did not run loses that day's fills for good. The table then
shows a rotation: names the ledger still claims but the account no longer holds,
and the names bought in their place counted as yours. Set exactly those names to
the account:

```bash
$C run --rm --no-deps qbs ledger --adopt AMAT LRCX AMD PANW TEAM
$C run --rm --no-deps qbs ledger --adopt all   # nothing in the account is yours
```

Adjustments are appended as `ADJUST` rows, so the file keeps what was recorded
and what was corrected. A later `--rebuild` discards them. If you also traded by hand in that account, check the side-by-side
table afterwards: those fills are in the log as well.

Switch it on **while the account is flat**. That is the only moment a tally
starts from a guaranteed-correct zero. Switching later leaves an empty ledger
beside a non-empty account, and the strategy then reads its own book as flat
and buys the whole thing a second time. If you have already traded, seed it
from the fills the run log has been recording all along:

```bash
$C run --rm --no-deps qbs ledger --rebuild     # reconstruct from the run log
$C run --rm --no-deps qbs ledger               # ledger vs account, side by side
```

`var/strategy_trades.csv` is appended from IB's own execution records at
reconcile — never from orders sent, so an order that did not fill leaves no row.
Rows are keyed on IB's execution id, so re-running reconcile after a failure
cannot double-count, and an order filled in two parts records both.

Every run checks the tally against the broker: the account must hold at least
what the ledger claims. If it claims more, either a fill went unrecorded or
someone sold the strategy's shares — the strategy would try to sell stock that
is not there, so the run stops instead. Whatever the account holds beyond the
tally is reported as the residual: that is your book, and the strategy leaves
it alone.

### The `baseline` alternative

Capture a baseline once, before the strategy has traded the account:

```bash
C="docker compose -f deploy/docker/docker-compose.yml"
$C run --rm --no-deps qbs baseline              # show what would be recorded
$C run --rm --no-deps qbs baseline --capture    # record it
```

Everything the account holds at that moment becomes yours, and the strategy
trades only shares above those counts. Its own position is always derived as
`account - baseline`, with the broker's number authoritative, so a missed fill
or a lost run log cannot make it drift.

The baseline is a snapshot, not a running tally, and the strategy never writes
to it. Two consequences:

- **Trade a name yourself after capturing, and the baseline is wrong.** Buy
  more and the strategy will treat the extra shares as its own and may sell
  them. Sell some and the account falls below the baseline: the strategy reads
  its own position as zero and would buy more every session, compounding. It
  refuses instead, with a guard naming the symbol.
- **Re-capturing later sweeps the strategy's positions into the baseline** and
  orphans them — it could then never sell them. A second capture needs
  `--force`, after reading the table it prints.

Fix a stale baseline by settling the account the way you want it, then
`baseline --capture --force`.

### Having the ranker skip names you hold

Netting shares lets the strategy trade a name you also hold. If you would
rather it stayed out of that name entirely and diversified away from you, add
it to `exclude_tickers` in the live JSON config, or set `QBS_EXCLUDE_TICKERS`:

It takes a list — comma or space separated, any case — not a single name:

```bash
QBS_EXCLUDE_TICKERS=MRVL,NVDA,AMD
QBS_EXCLUDE_OWN=1                    # also skip everything in the baseline
```

```json
{ "exclude_tickers": ["MRVL", "NVDA", "AMD"] }
```

A skipped name is not a lost slot — the next name down takes it, so the book
stays six wide. Excluding one name costs roughly a point of CAGR (measured on
the current history: 26.4% -> 25.4%, Calmar 1.40 -> 1.36), which is a fair
trade when you hold that name yourself and have the exposure anyway.

It does not scale. Excluding all six of the current holdings drops the book to
6.9% CAGR and 0.23 Sharpe, because a discretionary book concentrates in exactly
the names the ranker likes, so every exclusion removes one of its better
candidates. Keep the list to one or two.

An excluded name the strategy already holds is **sold** on the next run: it has
no target, and excluded names deliberately stay tradeable so the position can
be closed rather than stranded.

### Holidays and half days

The trader does not carry a holiday calendar and does not need one. The trade
phase asks the price feed instead: if there is no bar dated today, the market
is shut and it refuses to submit, logging `market closed or data late`. That
covers every holiday, and — unlike any fixed list — the unscheduled closures a
list cannot predict, such as a state funeral or a hurricane.

Half days are the one case that test cannot catch, because the market really
did open and there really is a bar. On July 3rd, the Friday after Thanksgiving
and Christmas Eve the auction is at 13:00 ET, while the trade timer fires at
15:30 — so the run would submit MOC into an auction that finished two and a
half hours earlier.

The trade phase sits those sessions out, at the top, before downloading or
ranking anything:

```
2026-11-27 is an early close (auction 13:00 America/New_York) -- sitting it
out; tomorrow's run recomputes from scratch
```

It exits 0, because a half day is a normal expected quiet day and a unit that
fails three times a year looks like a broken trader.

**Sitting out is not going to cash.** Nothing is sold; the book holds whatever
it held into the shortened session and through the holiday. The only thing
skipped is the *rebalance*, which on those days would have to clear a thin,
widely-quoted closing auction — the one place a shortened session costs real
money. Over the six years in the cache the book has averaged +0.79% on a half
day against +0.07% on every other day, so holding through them has been the
right side of the trade; the sample is twelve sessions, which is not enough to
lean on, but it is certainly no argument for being flat.

`QBS_EARLY_CLOSE_HHMM` (12:45) remains as a backstop under `--force`: forcing
a run overrides the calendar, so a shortened session can be traded deliberately
before 12:45, but no flag brings back an auction that has already happened.

The three dates are derived from the date itself, so there is no list that
expires at the end of the year. The derivation knows that when the 4th of July
or Christmas Day falls on a Saturday the holiday moves *back* onto the 3rd or
the 24th and the market is shut rather than early. For anything the rules
cannot know about, `QBS_EARLY_CLOSE_DATES` takes `YYYY-MM-DD` entries.

Missing one auction costs nothing structural: the strategy recomputes from
scratch every session, so the next run corrects the book.

### The CSV logs in `var/`

Six files, so the run log can be read with `cat` and no tooling:

| file | what | how it is written |
|---|---|---|
| `ranking_log.csv` | the top 25 of each day's ranking, with rank, score and whether it is held | appended, one block per date |
| `shadow_log.csv` | what each candidate ranking rule *would* hold, and where it disagrees with the live book | appended, one block per date |
| `watchlist_log.csv` | where a watched non-constituent would have ranked, against the book's own cutoffs | appended, one block per date |
| `trade_log.csv` | every trading event — the same rows `report` prints | rewritten from the database each run |
| `strategy_book.csv` | today's book: strategy shares beside the account's and yours | rewritten each run |
| `strategy_trades.csv` | the position ledger, when `position_source=ledger` | appended from IB's fills |

`ranking_log.csv` is the only one that accumulates history, because it has to:
what the strategy saw on a past date cannot be recovered from today's prices
once the ranking has moved on. It is keyed on the date, so the twice-daily
preflight and any re-run after a failure cannot duplicate a day. `QBS_RANKING_TOP`
sets the depth (default 25 — the book holds 6 and exits past 10, so 25 shows
the names queued behind them and a rotation becomes visible before it happens).

`shadow_log.csv` accumulates for the same reason, and exists because six years
of cached history cannot settle a question that has already been asked of it
sixty times. A candidate ranking rule is scored beside the live book every day,
holding nothing and sending no orders, so that in a year there are two return
streams to compare out of sample. The `live_held` column marks the names the
real book also holds, which makes the divergence a filter rather than a script.

The candidate under observation is `z(6-1) + w * z(turn)`, where `turn` is the
last month's return rate minus the prior quarter's — the "this name has only
just turned up" tilt. `QBS_SHADOW_WEIGHTS` sets which `w` values are scored
(default `0.5 1.25 2.0`); empty turns the log off. It is computed in `preflight`
and `signal`, never in `trade`: it is a few seconds of arithmetic that cannot
change an order, and the phase racing the MOC cutoff should not be carrying it.

`watchlist_log.csv` answers "is this stock stronger than what we hold?" for
names the book may not be able to buy. `QBS_WATCHLIST=TSM,GOOGL` scores each
name's 6-1 momentum every day and reports where it places in the constituents'
ranking, beside the score of the last name in the book and the last name inside
the band:

```
watch TSM:   rank 4 if it were a constituent, 6-1 momentum +131.2% (book needs +120.4%, band +90.5%)
watch GOOGL: rank 38, 6-1 momentum +15.7% (book needs +120.4%, band +90.5%)
```

The two lines differ because the two names do. GOOGL **is** in the index, so it
is already ranked and the log reports the rank the book acts on — the same
number `ranking_log.csv` has for it that day. TSM is not, so it is interpolated
into that ranking: the place it would take among the constituents, with nothing
else moved. The `constituent` column says which kind a row is.

Each name is placed against the constituents alone, never against the other
watched names, so adding a name to the list cannot change what any other row
reports.

A blank rank means the name was filtered out rather than ranked low — too
little history, or it lost to BOXX over the same window. That is written as a
blank rather than a large number, because "rank 99" would read as a weak name
instead of an excluded one.

Watched non-constituents are **not buyable**. They are ranked in a copy of the
universe; the book is computed from the index constituents and never sees them.
Do not try to get the same effect with `QBS_EXTRA_TICKERS` — that list is for
prices the strategy itself needs, and anything in it is a name the ranker is
entitled to put in the book.

The others are views of the database and rebuild themselves, which is what
stops them drifting from what they claim to show.

None of them can fail a phase. They are written after the orders are already
sent and recorded, so a full disk or a read-only mount logs a warning and the
session still succeeds.

### `var/strategy_book.csv`

Every preflight and trade writes a snapshot of the strategy's own book beside
the account's:

```
asof,symbol,strategy_shares,account_shares,yours,price,market_value,target_shares,target_weight
2026-09-11,MRVL,25,125,100,236.5600,5914.00,25,0.0591
```

It is a report, not a record. Every figure is recomputed from the broker and
the baseline on the next run, so editing the file changes nothing and losing it
costs nothing — which is precisely what makes it safe to keep. A CSV the
strategy read back as its position of record would drift the first time a fill
was missed, and no amount of price history could repair it. That is why the
strategy's position is derived rather than tallied.

## Optional — mirror the run log to a Google Sheet

The database stays the record; this pushes a copy you can read from a phone,
share, and chart by hand. Nothing is ever read back from the sheet.

It runs as **its own timer**, ten minutes after reconcile. That is the whole
design: the Sheets API is the least reliable dependency here — someone else's
service, over the internet, behind a quota — and putting it on the same code
path as recording what was traded would let a network blip fail the record.

**1. A service account.** Nothing else works unattended: the OAuth consent flow
needs a browser, and a refresh token eventually needs a human.

- In the Google Cloud console, create a project and enable the **Google Sheets
  API**.
- Create a **service account**, then a **JSON key** for it.
- Put the key at `var/google-sa.json` on the VM. `var/` is gitignored and
  bind-mounted, so the key never enters the image or the repository.
  `chmod 600 var/google-sa.json`.

**2. Share the sheet with it.** Create a spreadsheet, then share it with the
service account's email (`something@project.iam.gserviceaccount.com`) as an
**Editor**. The account owns nothing and can reach nothing else in your Drive —
only what you share with it.

**3. Point the trader at it.** The id is the long string in the sheet's URL,
`docs.google.com/spreadsheets/d/<THIS BIT>/edit`:

```bash
# deploy/docker/.env
QBS_SHEETS_ID=1AbC...xyz
```

**4. Try it, then schedule it.**

```bash
C="docker compose -f deploy/docker/docker-compose.yml"
$C build qbs                       # gspread is a new dependency
$C run --rm --no-deps qbs sheets

sudo cp deploy/docker/systemd/qbs-sheets.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qbs-sheets.timer
```

Five tabs, each cleared and rewritten from the database on every push:
`trades`, `selection`, `closes`, `nav`, `runs`. Rewriting rather than appending
means the push needs no memory of what it last sent — one more piece of state
that could drift — and running it twice, or after missing a week, produces the
same correct sheet.

Leave `QBS_SHEETS_ID` blank and the phase logs that there is nothing to do and
exits 0, so the timer is harmless to install before you have a sheet.

## Step 10 — The instance schedule

EventBridge Scheduler rules against the EC2 API. The simplest shape is one
window covering the whole trading day:

```
qbs-start   cron(45 8 ? * MON-FRI *)    ec2:StartInstances
qbs-stop    cron(20 16 ? * MON-FRI *)   ec2:StopInstances
```

Prefer `America/New_York` for the rules so they track DST alongside the timers.
UTC rules also work, but then you must check both DST states by hand: a window
that comfortably contains 15:30 and 16:15 New York in March may not in
November, when New York moves an hour further from UTC.

Two narrower windows — one for the morning health check, one for the session —
cost less and are equally valid, as long as each window contains the timers that
fire inside it **with margin**. Thirty minutes between boot and the trade phase
is the practical floor: the instance boots, Docker starts, the Gateway's JVM
comes up, and IBC logs in, and only then is the API listening. That is why the
preflight timer fires twice (08:50 and 15:05 ET) — each window gets a run that
walks the entire path and sends nothing, so a Gateway that came back without
logging in shows up in the journal while there is still time to act.

### What must survive a stop

Everything on the EBS volume does: `/opt/qbs`, `var/` (run log and state),
`data/` (the price cache), the installed systemd units and their enabled state,
and the built image. The one thing that does not is the Gateway's logged-in
session, which has to be rebuilt on every boot. `restart: unless-stopped`
handles that **only if the Docker service itself starts at boot**:

```bash
sudo systemctl enable docker
```

Without that line the second window comes up with no Gateway at all, and the
trade phase fails on connect. Prove it once, rather than discovering it at
15:30:

```bash
sudo reboot
# wait a minute, ssh back in
cd /opt/qbs
docker compose -f deploy/docker/docker-compose.yml ps          # Gateway Up?
docker compose -f deploy/docker/docker-compose.yml logs --tail 30 ib-gateway
docker compose -f deploy/docker/docker-compose.yml run --rm --no-deps \
  --entrypoint python qbs test_ib.py
```

The log must reach `Login has completed` and `Configuration tasks completed`,
and the test must print a `DU…` account. A reboot is the same cold start the
schedule performs, so this is the real rehearsal.

Neither rule knows about market holidays. That is harmless: the trade phase sees
no bar for today and exits 0 without trading. You pay for idle hours, not for a
bad trade.

---

## Operating it

```bash
cd /opt/qbs
C="docker compose -f deploy/docker/docker-compose.yml"

$C ps                                    # is the Gateway up
$C logs --tail 50 ib-gateway
$C run --rm --no-deps qbs report --days 20         # trades, picks, closes
$C run --rm --no-deps qbs signal --offline         # what would it hold today

systemctl list-timers 'qbs-*'
journalctl -u qbs-trade -n 100 --no-pager
sudo systemctl start qbs-preflight.service   # force a phase now
```

### Kill switch

```bash
touch /opt/qbs/var/HALT      # preflight and trade refuse; reconcile still runs
rm /opt/qbs/var/HALT         # resume
```

A missed session costs nothing — the strategy recomputes from full price history
every run, so there is never anything to catch up.

### Updating the code

```bash
cd /opt/qbs && git pull
docker compose -f deploy/docker/docker-compose.yml build qbs
```

The Gateway container is untouched by this.

---

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionRefused` from the container | wrong host | it is `ib-gateway`, not `127.0.0.1` |
| `ConnectionRefused` from the host | Gateway not up or still logging in | `$C logs ib-gateway`; first start takes minutes |
| Connects, then drops | two clients sharing an id | phases use 17, the test uses 99 — keep them apart |
| `ConnectionRefused` to `172.x.x.x:4002` from the container | the Gateway serves a different port inside — this image uses 4004 | `docker exec ib-gateway cat /proc/net/tcp \| grep -i ' 0A '`, then set `IBG_PAPER_PORT` |
| `not a known paper port` | your Gateway serves paper on a port not in the default list | add it: `QBS_PAPER_PORTS=4002,4004,…` |
| `do not look like paper accounts` | the Gateway is serving a live account | **stop** — check `TRADING_MODE=paper` and the credentials before anything else |
| Account is not `DU…` | `TRADING_MODE` not `paper` | fix `.env`, recreate the Gateway container |
| `var/` files owned by root | `QBS_UID`/`QBS_GID` unset | set them, then `sudo chown -R $(id -u):$(id -g) var/` |
| Timers fire at the wrong hour | no `tzdata` on the host | `sudo apt-get install tzdata`, `daemon-reload` |
| Trade phase: "past the 15:45 MOC cutoff" | the run took too long | usually the yfinance download; nothing to do today |
| Trade phase exits 3, "GUARD TRIPPED" | an order list breached a limit; **nothing was sent** | read the reason — a turnover breach is nearly always a bad universe download |
| OOM / container killed | no swap | step 2 |

Exit codes: `0` fine (including "market closed today"), `1` error, `2` config or
connection, `3` a guard refused the list.

---

## What this deliberately does not do

- **No healthcheck on the Gateway container.** `depends_on` here is start-order
  only. A `condition: service_healthy` gate would be better, but it needs a
  probe whose tooling I could not verify inside that image — add one once you
  have looked at what the image ships.
- **No alerting.** Failures surface as failed systemd units. Wire `OnFailure=`
  to something that emails you if you want to hear before the next morning.
- **The trader container has no IB credentials.** It talks to the Gateway over
  the network and never needs the login, so `.env` is not passed to it.
