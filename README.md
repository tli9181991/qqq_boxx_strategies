# QQQ / BOXX strategy lab

Seven trading strategies with BOXX as the cash leg, one backtest engine, and a notebook
that shows you where every signal fired.

- **Larry Connors RSI(2)** — short-term mean reversion, long-only, filtered by SMA(200)
- **GEM (Global Equities Momentum)** — Antonacci dual momentum across QQQ / VEU / BOXX
- **Volatility-targeting overlay** — scales QQQ exposure so forecast vol sits near target
- **Top-6 Nasdaq-100 momentum** — cross-sectional 12-1 momentum with a hysteresis band
- **Top-6 + VIX circuit breaker** — the same book, switched off entirely when VIX spikes
- **Top-6 vol-targeted** — the same book, scaled by *its own* realised volatility
- **Top-6 Finviz screen** — the Finviz filter-and-rank notebook, rolled forward so it
  can be held against the momentum book on identical assumptions

Plus one strategy that does not fit the daily model and runs on its own:

- **Weekly breakout, 6 slots** — a weekend Finviz watchlist traded by the M6
  resistance-breakout rules on **hourly** bars, with per-trade stops

Backtest window: **2025-01-20 (inauguration) to today**, one-day execution lag,
commission and slippage charged separately on turnover.

---

## Quick start

```bash
pip install -r requirements.txt

python run_backtest.py                    # download, backtest, print table, save charts
python run_backtest.py --sweep-band       # + the hysteresis-band sensitivity table
jupyter lab notebooks/backtest_visualization.ipynb
```

No network? Everything runs on a generated market instead:

```bash
python run_backtest.py --synthetic
```

The first live run downloads from Yahoo and caches to `data/`; later runs are instant
and work offline (`--offline`). `--no-momentum` skips the ~100-ticker Nasdaq-100
download while you iterate on the other three.

---

## ⚠️ Read this before trusting the momentum numbers

The Nasdaq-100 ranking universe defaults to **today's** index membership applied to the
whole history. For a momentum strategy that is survivorship bias of the worst kind:
today's Nasdaq-100 is, close to by definition, a list of stocks that went up, and a
strategy that may only ever buy from that list has been handed the answer in advance.

Over a twenty-month window the damage is smaller than over twenty years, but the index
reconstitutes each December plus ad-hoc M&A and delisting changes, so it is not zero.

The fix is point-in-time membership:

```python
from qbs.universe import load_pit_universe
from qbs.pipeline import run

pit = load_pit_universe("my_ndx_membership.csv")   # date,ticker rows
lab = run(cfg, pit_membership=pit)
```

`membership_mask()` turns that into a date × ticker eligibility grid the ranker
respects, so a name added in March 2025 is simply not rankable before then. Everything
else in the package is unchanged. If you are going to trade real money on this, buy the
membership history.

---

## What each strategy does

### 1. Connors RSI(2)

A pullback system, not a dip-buying system. The distinction is the trend filter.

| | Rule |
|---|---|
| Filter | no new long unless `close > SMA(200)` |
| Entry | `RSI(2) < 5` at the close |
| Exit | `close > SMA(5)`, or `RSI(2) > 70`, or 10 trading days elapse |
| When flat | holds BOXX, earning the cash yield |

RSI is **Wilder's**, seeded from the simple mean of the first `period` changes — the
definition Connors' thresholds assume. A plain `ewm()` drifts for the first few dozen
bars.

The time stop is an addition to the canonical rules. Without it a trade that neither
recovers nor triggers an exit can sit open indefinitely, which quietly turns a
mean-reversion system into a buy-and-hold one.

**Expect this strategy to be flat most of the time.** `RSI(2) < 5` above the 200-day is
rare. Check the *Avg risk exposure* column before reading anything into its Sharpe —
most of its return is the BOXX yield.

### 2. GEM

Dual momentum, evaluated on month-end closes only:

1. **Relative momentum** — of QQQ and VEU, take whichever has the higher 12-month
   trailing total return.
2. **Absolute momentum** — hold that winner only if it beat BOXX over the same 12
   months. Otherwise the whole book sits in BOXX.

The absolute-momentum hurdle is BOXX's *realised* return, not a fixed 0%. In a 4–5%
T-bill world "did equities go up?" and "did equities beat the cash I could have held?"
are very different questions, and only the second one is decision-relevant.

The holding decided at month-end M applies through month M+1; the engine's execution
lag then shifts it one more day, which matches rebalancing at the next open.

### 3. Vol targeting

```
weight = target_vol / forecast_vol,  clipped to [0, max_weight]
```

Forecast vol is an EWMA (halflife 20 days) of daily returns, annualised. Two details
matter more than the formula:

- **No-trade band.** The raw ratio moves every day. Trading that wiggle burns the edge
  in costs for no risk benefit, so the position only moves when the target has drifted
  5% away from what is held. The notebook plots banded and unbanded together — the gap
  between them is turnover you don't pay for.
- **Vol floor.** In a calm tape the ratio explodes. The floor caps how small the
  denominator gets; `max_weight` catches whatever is left.

Default `max_weight = 1.0` means **no leverage** — the strategy can only de-risk into
BOXX. Raise it to gear up in calm markets.

Vol targeting also works as an *overlay* on the other two (`vol_target_overlay(px,
base_weights=...)`), which is usually the more interesting way to use it. Section 10 of
the notebook runs that comparison.

### 4. Top-6 Nasdaq-100 momentum

Cross-sectional momentum: rank the whole index, hold the strongest names.

| | Rule |
|---|---|
| Score | **12-1 momentum** — return from 12 months ago to 1 month ago |
| Hold | the top `n_hold` (6), equal weight per *slot* |
| Exit | only once a name falls past `exit_rank` (10) — the hysteresis band |
| Filter | a name must also beat BOXX's own 12-1 return, else that slot goes to cash |
| Rebalance | daily by default, matching the intended live design |

**Why 12-1 and not 12-0.** The most recent month is skipped because short-horizon
returns *reverse* rather than persist. Including them mixes two effects with opposite
signs and blunts both.

**Why the band is the parameter that matters.** Without it, a name that slips from rank
6 to rank 7 is sold and often bought back days later — pure churn, paid for in spread
and short-term capital gains. On the bundled synthetic universe, widening the band from
0 to 4 cuts annual turnover from **43× to 8×**, and to **3×** at a band of 14. Return is
far noisier than that, which is the point: `sweep_band()` and section 9c of the notebook
exist so you look for a *plateau* rather than picking the best cell.

**Slot weighting, not survivor weighting.** With only 4 of 6 slots qualifying, the book
is 4/6 invested and 2/6 in cash. Spreading 100% across the survivors would concentrate
the portfolio exactly when the fewest names were qualifying — i.e. in a deteriorating
market, which is precisely backwards.

### 5. VIX circuit breaker

A binary risk switch laid over the momentum book. Three states, evaluated each close:

| State | Transition | Holds |
|---|---|---|
| `INVESTED` | VIX close > `exit_level` → `CASH` | the base strategy's weights |
| `CASH` | `park_after_days` elapsed → `PARKED`<br>VIX < `entry_level` **and** `min_cash_days` met → `INVESTED` | nothing — true 0% cash |
| `PARKED` | VIX < `entry_level` → `INVESTED` | BOXX |

**Why cash before BOXX.** Buying the safe asset for a two-day scare costs two spreads
to earn two days of T-bill yield — a losing trade. The breaker sits in true cash first
and only converts once the stress has actually persisted.

**Why a minimum dwell.** `min_cash_days` stops it re-entering on the bar right after a
one-day spike, which is when the tape is least readable.

**Why two thresholds.** `exit_level` above `entry_level` is a hysteresis band on
volatility — the same device as the momentum rank band, for the same reason.

**This is not volatility targeting.** Vol targeting scales exposure continuously with
forecast risk; this is on/off. They compose — run `vol_target_overlay` on the result
if you want both.

### 6. Book vol targeting

The drawdown control that works, and the one to reach for before the VIX breaker.

```
scalar = target_vol / the book's OWN realised vol,  clipped to [0, max_weight]
```

Every holding is multiplied by that single scalar, so relative position sizes never
change: this decides *how much* of the strategy you hold, never *which* names. The
freed weight parks in BOXX.

**Why the book's vol and not VIX or a QQQ trend filter.** The Top-6 book has two
different kinds of drawdown, and index signals only see one of them:

| Episode | Book DD | VIX then | QQQ then |
|---|---|---|---|
| 2025 market selloff | −31% | 28–45 — visible | −22.8% |
| 2026 concentration blow-up | −30% | **16–20** — invisible | −8.8% |

In the second episode the book lost 7–8% on days QQQ lost about 1%, with VIX *below*
its own median. Neither a VIX threshold nor `QQQ > SMA(200)` reduces that drawdown at
all — measured on this sample the SMA(200) gate leaves max drawdown unchanged at
−34.8% while costing half the return. The market was not what went wrong.

The book's own realised vol rises in **both** cases, which is the whole argument for
measuring the thing you actually hold. On the cached window, at a 25% target:

| | CAGR | Vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| Top-6 unscaled | 46.1% | 49.0% | 0.94 | −34.8% | 1.32 |
| **Top-6 vol-targeted 25%** | **26.4%** | 25.9% | 0.88 | **−18.8%** | **1.40** |
| Buy & hold QQQ | 22.2% | 22.6% | 0.82 | −22.8% | 0.98 |

Per episode the drawdown falls from −31.4% to −16.6% (the market selloff) and from
−29.5% to −11.6% (the concentration blow-up).

**It does not add return.** Sharpe is roughly unchanged — what improves is Calmar,
because de-levering cuts drawdown faster than it cuts return. It lands ahead of QQQ
only because the underlying book has the higher Sharpe to begin with, so shrinking it
to QQQ-like volatility keeps some of the edge. It also cannot help with an overnight
gap in one name: it responds to sustained volatility, not to jumps.

`sweep_target_vol()` and notebook §9f sweep the target; §6f plots the book's vol
against VIX so you can see the episodes an index signal misses. The dial is expected
to be *boring* — a kink would mean the overlay is doing more than rescaling.

### 7. Top-6 Finviz screen

`finviz_filter_with_daily_summary.ipynb` answers "what would I buy today". This rolls
it forward to every historical date so it can be held against the momentum book.

| | Rule |
|---|---|
| Filter | price > $10, close > SMA(200), quarter return > 0, **within 10% of the 52-week high** |
| Score | 1-year total return, bucketed into a 1–99 RS Rank |
| Tie-break | smallest distance below the 52-week high |
| Hold | the top 6, equal weight per slot |
| Exit | the day a name stops passing or drops out of the top 6 — **no band** |

**What was reproduced and what could not be.** Every Finviz criterion in the notebook
is derived from price or volume, which is what makes it rollable at all. Two are not
applied: market cap over $300m needs fundamentals (non-binding on the Nasdaq-100), and
average volume over 200k needs share volume — pass `volumes=` to enable it. Both
omissions are *permissive*, so they flatter this strategy rather than the other way
round. The notebook's 20%-quarterly-return gate is off by default because the notebook
itself applies it only to the sector-breakdown table, not to the list it ranks
(`min_quarter_return=0.20` switches it on).

**It is ranked against the Nasdaq-100, not against Finviz's own output.** That is
deliberate. The notebook screens the whole US market and gets ~530 names; running the
comparison that way would change the universe and the selection rule at the same time,
and there would be no way to attribute the difference to either. Worse, there is no
point-in-time version of that 530-name list — it is *today's* screener result, i.e.
names selected for having already gone up, and a backtest that may only buy from it
has been handed the answer. Ranking both strategies over the same 99 names means the
only thing the comparison can be measuring is the rule.

#### How it compares

Same universe, same six slots, same engine, same 1bp + 5bp costs, 2025-01-20 → 2026-09-08:

| | CAGR | Vol | Sharpe | Max DD | Turnover | Cost drag |
|---|---|---|---|---|---|---|
| **Top-6 NDX momentum** (12-1, band 10) | **46.1%** | 49.0% | **0.94** | **−34.8%** | **5.7×** | 0.3% |
| Top-6 Finviz screen (notebook rules) | 25.8% | 40.1% | 0.67 | −36.0% | 86.7× | 5.2% |
| Top-6 Finviz screen, monthly rebalance | 43.3% | 45.9% | 0.93 | −39.9% | 12.7× | 0.8% |
| Buy & hold QQQ | 22.2% | 22.6% | 0.82 | −22.8% | 0.6× | 0.0% |

**Roughly a third of the gap is churn, the rest is selection.** Gross of costs the
screen makes 32.5% against the momentum book's 46.6%, so about 6 of the 20-point net
gap is trading friction and the remaining 14 is the rule itself. The friction is
structural: a screen has no memory, so it re-sorts from scratch every day and a name
oscillating around rank 6 is round-tripped repeatedly. The average holding period is
**6 trading days against the momentum book's 98**, and it touches 62 distinct names
over the window where the momentum book touches 24. Rebalancing monthly removes almost
all of that and closes most of the gap.

**The binding constraint is "within 10% of the 52-week high".** It is the criterion
that decides what this strategy is, and on this universe it costs money:

| Max distance below the high | 5% | **10%** | 15% | 20% | 30% | none |
|---|---|---|---|---|---|---|
| CAGR (daily rebalance) | 20.8% | **25.8%** | 49.6% | 54.4% | 54.7% | 65.4% |
| Names passing (avg) | 21 | **32** | 39 | 42 | 44 | 45 |

Monotone, which is the shape you would rather not see in a parameter you are relying
on — there is no plateau to sit on, just a filter that helps less the tighter you set
it. (The monthly-rebalance column is *not* monotone, which is a fair reminder of how
much noise a 20-month sample carries.)

The reason is visible in the current book. On the last cached bar the two strategies
had **zero names in common**:

```
Top-6 NDX momentum:  LRCX, MU, AMAT, INTC, AMD, MRVL     (six semiconductors)
Top-6 Finviz screen: WBD, CRWD, FTNT, ROST, CSX, BIIB
```

The semiconductor complex has the six strongest 1-year returns in the index — MU at
+663%, INTC +327%, MRVL +257% — and every one of them sits 13–35% below its 52-week
high after the run. The proximity filter excludes the entire group by construction. Over
the whole window the two books share a median of 3 names out of 6, and on 44% of days
two or fewer.

**Its cash leg never engages.** A screen is an absolute test, so it *can* answer
"nothing qualifies" and sit in BOXX — that is the main structural argument for
preferring one to a ranking. On the Nasdaq-100 it never gets the chance: the worst day
in the sample still had 10 names passing, against a median of 34, so the book was 100%
invested on every one of the 410 trading days, April 2025 included. Whatever this
strategy is, it is not a de-risking mechanism on this universe. On the notebook's own
small-cap-inclusive universe it may well be; that is not testable here.

**What this does not settle.** The notebook's real screen runs on the whole US market,
and its edge — if it has one — may live in the small and mid caps the Nasdaq-100 does
not contain. Nothing above rules that out. What it does show is that the *selection
rule*, applied to the same names as the momentum book, picks differently and, on this
window, worse — and that the 52-week-high proximity filter is why.

---

### 8. Weekly breakout, 6 slots

The scenario: **screen with Finviz at the weekend, trade the list next week with the
M6 breakout rules, hold at most 6 names, and when one stops out let the slot sit in
cash until the following weekend.**

This is the only strategy here that does not go through `engine.run_backtest`, for two
reasons that are structural rather than stylistic:

- **It is hourly.** The entry is an hourly close crossing a resistance level, confirmed
  *exactly two hourly bars later*. Collapse that to daily closes and the confirmation
  window — the thing that separates a breakout from a spike — disappears entirely.
- **It is event-driven.** Each position has its own entry price and its own stop at
  `entry − R`. A weight times a close-to-close return cannot express "filled at 102.40
  on the 11:30 bar, stopped at 98.10 four sessions later".

So `breakout.py` does its own accounting from actual fills and hands back a
`BacktestResult` that `metrics.summarise` reads like any other line. The costs, the
slot weighting and the safe asset are the lab's; the fill model is the one thing that
differs, and it differs because it has to.

| | Rule |
|---|---|
| Universe | the weekend Finviz screen's top `watchlist_size` (20) by RS rank |
| Entry | regime up (EMA10 > EMA20 daily), hourly close crosses a resistance level, still above it and still in regime `confirm_hours` (2) later. Filled at that bar's close |
| Risk unit *R* | `min(\|entry × Var95\|, avg_level_gap/2 + ADR/2)` |
| Stop | close ≤ `entry − R` |
| Take profit | once above `entry + R`, exit when close < EMA10 − ADR |
| Time / trend | between ±R, exit after 3 weeks or on close < EMA10 − ADR |
| Slots | 6 concurrent, `1/6` each, idle slots in BOXX |
| Refill | **none mid-week** — a freed slot waits for the next weekend |

#### ⚠️ The notebook's breakout backtest has look-ahead in it

Four defects, found while porting. Three of them let the future leak into a trade, and
the first is severe enough that the notebook's breakout results should not be treated
as achievable.

1. **Resistance levels are drawn from the whole history.** `estimate_sr_levels` runs
   `find_peaks` over the entire daily frame *once*, and `generate_trades` then replays
   that same history against those levels. A resistance level is by construction a
   price the stock turned at — so trades are placed at levels defined by turns that
   had not happened yet. Measured on a 2.5-year sample:

   | A trade placed... | levels drawn from pivots still in the future |
   |---|---|
   | a quarter of the way in | **76%** |
   | halfway in | 49% |
   | three quarters in | 20% |

   The notebook then says *"we can manually adjust the levels for increasing the
   performance"*, which is curve-fitting on top of the leak.

2. **Var95 is taken over the whole return history**, then used to size `R` on every
   trade — including trades that predate the returns it was computed from.

3. **Daily indicators are forward-filled without a lag.** The hourly frame is resampled
   to daily and reindexed with `ffill`. Daily rows are stamped at midnight, so *every*
   hourly bar of day D receives the EMA, ATR and ADR computed from day D's **close** —
   the 10:00 bar included. Both the regime gate and the EMA exit see the day's outcome
   all day.

4. **A position could exit before it entered.** The cross is detected at bar `i`, the
   confirmation is read from bar `i+2`, and the trade is written into `active_by_level`
   while the loop is still at `i`. The exit block then runs at bars `i+1` and `i+2` and
   can close the position on a bar preceding its own `entry_time`.

All four are fixed. The three look-ahead fixes are **switches**, not silent
corrections, so you can price each one rather than take my word for it:

```python
BreakoutParams(causal_levels=False, causal_risk=False, lag_daily_indicators=False)
```

reproduces the notebook; the defaults are causal. With `causal_levels=True` the levels
are redrawn at each weekend from data up to that weekend only — which is also just
what a trader does, so the fix costs nothing in realism.

Two parameters in the notebook are inert and are not reproduced: `rr_takeprofit` (a 2R
target that `generate_trades` computes and never reads) and `require_retest`. Where its
prose and its code disagree — the prose says "10-day **sma**", the code uses the fast
**EMA** minus one ADR — the code is implemented, because the code is what produced its
numbers.

#### You need hourly data, and it expires

`yfinance` serves **at most ~730 days of hourly history**, and it is not back-fillable:
whatever you have not cached before it ages out is gone. `load_hourly()` caches
additively, one CSV per name, for that reason. This is also why there are no headline
numbers in this section — the repo's cached data is daily closes only, so the strategy
has been **validated on synthetic hourly bars, not on real ones**:

```python
from qbs.breakout import load_hourly, finviz_watchlists, weekly_breakout_book
from qbs.metrics import summarise

wl = finviz_watchlists(universe_prices, prices["BOXX"], n_watch=20)
hourly = load_hourly(sorted({t for names in wl.values() for t in names}),
                     start="2024-09-15")
book = weekly_breakout_book(hourly, wl, prices["BOXX"])
print(summarise(book.result), book.hit_rate)
print(book.trades.head())            # every fill, with its level, R and exit reason
```

`book.trades` is the thing to read first. A 6-slot book takes few enough trades that
you can audit them one by one, and a breakout strategy's behaviour lives in the exit
mix — a book that is mostly `stop_R` is telling you the levels are not holding.

**What the no-refill rule costs.** A slot freed on Tuesday sits in cash until Friday
however many watchlist names break out on Wednesday, so average exposure runs well
below 100% and the book is structurally part-invested. That is the scenario as
specified, and `WeeklyBookParams(refill_within_week=True)` measures what the constraint
is worth.

---

---

> ⚠️ **Read the default 17/16 as a warning, not a recommendation.** VIX's long-run
> median sits near 17–18, so a trigger at 17 fires on ordinary conditions rather than
> on stress: the breaker is engaged roughly half the time, which makes it a
> mostly-out-of-market strategy rather than a crash filter. On the bundled synthetic
> run it cut CAGR from 16.0% to 6.1% and more than doubled turnover, in exchange for a
> shallower drawdown. `sweep_vix()` and notebook §9e exist to make you look at this
> before trusting any single pair of levels. Note also that VIX *level* is a weak
> timing signal in the literature — high implied vol tends to precede high future
> returns (the volatility risk premium). The VIX term structure (VIX vs VIX3M) is the
> more common professional choice; `VOL_INDEX_3M` is defined for that experiment.

---

## How it fits together

**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** walks through every module --
what it does, why it is shaped that way, and the code that carries the decision.
Start there if you are reading this codebase for the first time.

---

## Layout

```
qbs/
  config.py       window, all strategy parameters, chart palette
  data.py         yfinance download + CSV cache + synthetic market and VIX generators
  universe.py     Nasdaq-100 membership, point-in-time hook, wide price loader
  indicators.py   Wilder RSI, SMA, EWMA vol, trailing return, drawdown
  strategies.py   the six ranking/overlay strategies -> weights + diagnostics + events
  screens.py      filter-based screens: the trend template and the Finviz screen,
                  both rolled forward from a notebook so they can be backtested
  breakout.py     hourly resistance-breakout trading + the weekly six-slot book;
                  its own fill-level accounting, because weights cannot express a stop
  engine.py       one backtest function: lag, commission, slippage, equity curve
  metrics.py      CAGR, Sharpe/Sortino vs BOXX, drawdown, turnover, trade log
  plotting.py     the chart system
  pipeline.py     load -> signals -> backtest in one call; sweep_band(),
                  sweep_vix(), sweep_target_vol()
run_backtest.py   CLI
notebooks/backtest_visualization.ipynb
tests/test_qbs.py 89 tests: indicators, engine, momentum, circuit-breaker,
                  vol-target and screen invariants (each strategy gets a
                  shuffled-future look-ahead test)
```

### The one convention that matters

`signals.weights.loc[t]` is **what the strategy decided while looking at the close of
day t**. The execution lag lives in `engine.run_backtest` and nowhere else, so it is
applied identically to every strategy and cannot be accidentally skipped in one of
them. Section 11 of the notebook re-runs everything with `lag=0` to show what
look-ahead would have been worth.

---

## Using it

```python
from qbs.config import (BookVolTargetParams, Config, MomentumParams,
                        RSI2Params, VolTargetParams)
from qbs.pipeline import run, sweep_band

cfg = Config()
cfg.backtest_start = "2024-09-01"
cfg.slippage_bps = 10.0                                      # pessimistic fills
cfg.vol = VolTargetParams(target_vol=0.12, max_weight=1.5)   # allow gearing
cfg.rsi2 = RSI2Params(entry_threshold=10)                    # trade more often
cfg.momentum = MomentumParams(n_hold=8, exit_rank=20,        # wider band, less churn
                              rebalance="ME")                # monthly instead of daily
cfg.book_vol = BookVolTargetParams(target_vol=0.15)          # a calmer momentum book

lab = run(cfg)
lab.summary_pretty
sweep_band(lab)                                              # is there a plateau?
```

```bash
python run_backtest.py --start 2024-09-01 --n-hold 8 --exit-rank 20 --sweep-band --csv
python run_backtest.py --slippage-bps 20            # does it survive worse fills?
python run_backtest.py --sweep-vix --vix-exit 25    # where should the VIX trigger sit?
python tests/test_qbs.py
```

Adding a seventh strategy means writing a function that returns a `StrategySignals`
and adding it to `pipeline.build_signals`. Everything downstream — engine, metrics,
charts — works on it unchanged.

---

## Running it live

The Top-6 vol-targeted book can be deployed to IB paper on a small VM. See
**[`deploy/README.md`](deploy/README.md)** for the runbook.

```
qbs/live/
  config.py    deployment settings: account, sizing, safety limits
  signals.py   today's target weights -- calls the SAME functions as the backtest
  orders.py    weights -> shares -> deltas -> guards (pure, no IB, no clock)
  broker.py    the ib_async layer: positions, MOC orders, fills
  state.py     small rolling record of run outcomes
  store.py     the run log: trades, selections, closes and NAV as SQL tables
  runner.py    phases: preflight / trade / reconcile, plus signal and report
deploy/        systemd units + timers, installer, runbook
```

```bash
python -m qbs.live.runner signal --offline   # what would it hold today?
python -m qbs.live.runner report             # the run log: trades, picks, closes
sudo ./deploy/install.sh                     # provision the VM
```

Everything the live book does is logged to SQLite at `var/qbs.db` — a
`trade_events` log plus daily `selection_events`, `position_closes`,
`portfolio_nav` and `signal_runs` snapshots. `store.to_frame(db, table)` hands
any of them to pandas.

Three systemd timers anchored to `America/New_York`, so US daylight saving
moves them for you: preflight at 08:50 (proves the whole path works, sends
nothing), rank-and-submit-MOC at 15:30, reconcile at 16:15.

**The signal is recomputed from full price history on every run.** Nothing in
the state file feeds it -- delete the file and the next run still produces
exactly the right orders. Both the hysteresis band and the vol scalar are
path-dependent, so carrying yesterday's state forward would let one missed run
silently fork the live book from the strategy. Recomputing costs seconds and
makes a missed session self-healing; never replay one by hand.

---

## Chart conventions

- **One y-axis per panel, never two.** Price and RSI, or price and vol and exposure,
  get stacked panels sharing an x-axis. A dual-axis chart lets whoever drew it pick the
  story by picking the scaling.
- **Colour-vision-safe palette**, three categorical slots, validated all-pairs.
  Benchmarks sit outside that set in grey so they never compete with a strategy.
- **Identity is never colour alone** — legend *and* direct end-labels on every
  multi-series chart; buy/sell markers differ in shape as well as hue.

---

## Caveats

**A twenty-month sample proves nothing.** Roughly 400 trading days, and for GEM about
20 monthly decisions. A Sharpe ratio estimated over that span has a standard error near
0.8 — wide enough to contain almost any conclusion you might want to draw.

- **Survivorship bias** in the Nasdaq-100 universe, unless you supply point-in-time
  membership. See the warning near the top — it flatters the momentum strategy
  specifically, and it is the largest single caveat in this package.
- **Six names is a concentrated bet.** The backtest reports portfolio volatility, but it
  cannot show you the risk that actually ends these strategies: one holding gapping 30%
  overnight on an earnings miss. A sixth of the book in each name means a 5% portfolio
  hit from a single print, and a daily-rebalanced system cannot dodge it. Resting GTC
  stops at the broker are the mitigation, and they live outside this backtest.
  The vol-target overlay is only a partial answer — it sizes the book down as its risk
  rises, but it manages sustained volatility rather than gaps, and it does not
  diversify. As of the last cached run all six holdings were semiconductors, and
  nothing in the ranker prevents that.
- **The breakout book has never been run on real bars.** Its logic is tested — slot
  cap, no look-ahead, stop placement, the no-refill rule — but on *synthetic* hourly
  data, because this repo caches daily closes only. Treat its machinery as reviewed and
  its numbers as not yet existing.
- **The Finviz comparison is a rule comparison, not a strategy verdict.** Both books
  rank the same 99 Nasdaq-100 names, which is what makes the difference attributable to
  the selection rule. The notebook's real screen runs on the whole US market, and any
  edge it has in small and mid caps is invisible here — there is no point-in-time
  version of that screener output to test against, only today's list, which is a list
  of names that already went up.
- **Short-term capital gains.** At the turnover the band sweep reports, a taxable
  account converts most of the return into income-taxed short-term gains. Compare the
  after-tax number with simply holding QQQ before concluding anything.
- **One regime.** Any strategy that happened to be defensive during the April 2025
  drawdown looks brilliant. That is not evidence it will be defensive next time.
- **GEM is being asked to do something it wasn't designed for.** Antonacci's version
  uses broad indices over a decade-plus. Twenty months with QQQ in the equity slot is a
  more concentrated, shorter bet.
- **Vol targeting rescales risk; it does not add return.** If QQQ's Sharpe is negative,
  running at 15% vol instead of 25% loses money more slowly. That is the entire claim.
- **BOXX is not a Treasury.** It is an ETF holding box spreads: counterparty and
  liquidity risk the backtest treats as zero, and a tax treatment that is the reason it
  exists. The backtest models it as a price series and nothing more.
- **No taxes, no slippage beyond the flat cost.** RSI(2) at these turnover levels in a
  taxable account would be materially worse after tax.
- Prices are yfinance's split- and dividend-adjusted closes, subject to whatever Yahoo
  has revised lately.

This is a backtesting exercise, not investment advice, and I'm not a financial adviser.
Run the parameter sweeps in section 9 before believing any single number: what you want
is a plateau, not a spike.
