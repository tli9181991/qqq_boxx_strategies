# QQQ / BOXX strategy lab

Seven trading strategies with BOXX as the cash leg, one backtest engine, and a notebook
that shows you where every signal fired.

- **Larry Connors RSI(2)** — short-term mean reversion, long-only, filtered by SMA(200)
- **GEM (Global Equities Momentum)** — Antonacci dual momentum across QQQ / VEU / BOXX
- **Volatility-targeting overlay** — scales QQQ exposure so forecast vol sits near target
- **Top-6 Nasdaq-100 momentum** — cross-sectional 6-1 momentum with a hysteresis band
- **Top-6 + VIX circuit breaker** — the same book, switched off entirely when VIX spikes
- **Top-6 vol-targeted** — the same book, scaled by *its own* realised volatility
- **Top-20 high momentum screen** — an absolute bar (over $5, over 300k shares a day,
  up more than 28% on the quarter), then ranked by relative strength

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

Optional extras, each in its own requirements file so the backtest never depends on
them: `requirements-dashboard.txt` (Streamlit),
`requirements-agent.txt` ([the LLM analyst](#the-llm-analyst)),
`requirements-live.txt` (IB trading).

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
respects, so a name added in March 2025 is simply not rankable before then.

**The file must be a complete snapshot per date, not a list of change events.** Rows
are forward-filled wholesale, so a ticker omitted from a snapshot reads as "dropped".

**And half the bias lives in the names that LEFT.** Masking today's constituents to
their join dates stops a name being ranked before it joined — but the names that were
*removed* are the ones whose absence flatters the result, and they are not in today's
list at all. So with `pit_membership` the universe is rebuilt from
`universe.pit_tickers()`, the union of every name that was ever a member, and those
are what get downloaded. Former members with no usable price history (delisted,
renamed, missing from the feed) are **reported by name** rather than dropped quietly:
they stay unrankable, that is residual bias, and delisted names are exactly the ones
that failed.

Where to get the data — see **Sourcing point-in-time membership** below.

---

### Sourcing point-in-time membership

Nothing here ships membership history; these are the routes, roughly cheapest first.
**Verify current pricing yourself** — the figures below are indicative only.

| Source | Cost | What you get |
|---|---|---|
| **Invesco QQQ daily holdings** | free | The ETF *is* the index. Daily holdings files give exact constituents **and** weights. The catch is the archive: start saving them now, because historical files are not reliably retrievable |
| **Wikipedia revision history** | free | Pull dated revisions of the Nasdaq-100 page via the MediaWiki API and parse the components table at each. Good enough for recent years. Caveats: edits lag real index changes by days, early revisions are inconsistently formatted, and you are recording *what Wikipedia said*, not what the index was |
| **Nasdaq press releases** | free | The authoritative record — annual December reconstitutions plus ad-hoc change notices. Most accurate free option, most assembly work |
| **Norgate Data** | ~$70–90/mo | The standard retail answer: survivorship-bias-free US equities including delisted names, with index constituent history. Solves the price half and the membership half together |
| **EODHD** | ~$20–100/mo | Offers historical index constituents; check NDX coverage specifically |
| **CRSP / LSEG / FactSet / Bloomberg** | institutional | Definitive, priced accordingly. CRSP via WRDS if you have academic access |

**Membership alone is not enough.** You also need prices for names that left the index,
including ones that delisted. That is the half Norgate-style vendors exist to solve and
the half a free membership list does not touch — a perfect membership file plus a price
feed that has forgotten the delisted names still leaves you biased, which is why the
loader now names the gaps instead of hiding them.

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
| Score | **6-1 momentum** — return from 6 months ago to 1 month ago |
| Hold | the top `n_hold` (6), equal weight per *slot* |
| Exit | only once a name falls past `exit_rank` (8) — the hysteresis band |
| Filter | a name must also beat BOXX's own return over the same window, else that slot goes to cash |
| Rebalance | daily by default, matching the intended live design |

**Why skip the most recent month.** The most recent month is skipped because short-horizon
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

### 7. Top-20 high momentum screen

Started as a roll-forward of `finviz_filter_with_daily_summary.ipynb`. It now applies
the **same high-momentum definition** as the market-overview tab's leader group, so a
name shown as a "momentum leader" there and a name held here are selected on one rule.

| | Rule |
|---|---|
| Filter | close > **$5**, volume > **300k shares/day**, quarterly gain > **28%** |
| Score | 1-year total return, bucketed into a 1–99 RS Rank |
| Tie-break | smallest distance below the 52-week high |
| Hold | the top 20, equal weight per slot |
| Exit | the day a name stops passing or drops out of the top 20 — **no band** |

**It is a different kind of strategy from the momentum book**, not a variant of it. The
momentum book ranks the whole universe and takes the top six whatever the market is
doing, so it is always fully invested. This screen applies an *absolute* bar first, so
in a weak tape it can return almost nobody and sit in cash. That is a feature of the
design, and the reason the two rarely hold the same names.

**The two Finviz-notebook filters are now off by default**: `above_sma` (200-day) and
`within_52w_high_pct` (10%). Both still work if set. Turning `above_sma` back on is
near-free — on the cached universe it changes the passing count by zero, because a name
up 28% on the quarter is essentially always above its 200-day average. The proximity
filter is the one that bites: it roughly halves the qualifying set and takes a full
top-20 from 15% of sessions down to 3%.

**`n_hold` is 20, and the universe decides whether that means anything.** On the
~99-name Nasdaq-100 cache a median of **11** names clear the filter, so "top 20" is
usually "everyone who qualified" — on the last cached bar, 6 names. For the ranking to
be a real selection, run it over the broad US universe
(`qbs.finviz.fetch_us_universe`), which is what the market tab already fetches.

**What is not applied.** Market cap over $300m needs fundamentals (non-binding on the
Nasdaq-100, permissive elsewhere). And `min_volume` **needs `volumes=`** — unlike the
old average-volume filter, which was skipped silently when volume was missing, this one
*raises*, because it is a leg of the definition and dropping a leg on the floor
overstates the screen. The dashboard and the backtest both opt out of it explicitly
and say so on screen, since the Nasdaq-100 cache carries closes only.

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
| Finviz screen, **as it was** (top 6, notebook rules) | 25.8% | 40.1% | 0.67 | −36.0% | 86.7× | 5.2% |
| Finviz screen, **as it was**, monthly rebalance | 43.3% | 45.9% | 0.93 | −39.9% | 12.7× | 0.8% |
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

> **Everything in this section measures the screen as it was**: top 6, price > $10,
> above the 200-day, within 10% of the 52-week high, quarter up. It now runs the
> high-momentum definition instead, with the proximity filter **off** — which is the
> very filter this section identifies as the cause. The findings are kept because they
> are why the screen changed, not because they describe what it does now. **Re-run the
> backtest before quoting any number above as current.**

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

#### Sweeping it

`sweep_breakout()` re-runs the book across a grid and returns one row per cell, the
same shape as `sweep_band()` / `sweep_vix()` / `sweep_target_vol()` — and read the same
way, for a **plateau rather than a best cell**.

```python
from qbs.breakout import sweep_breakout, lookahead_cost

sweep_breakout(hourly, wl, prices["BOXX"], {"r_mult": [0.5, 1.0, 1.5, 2.0, 3.0]})
sweep_breakout(hourly, wl, prices["BOXX"], {"confirm_hours": [1, 2, 3, 4, 6],
                                            "n_slots": [4, 6, 8]})
```

**Read `n_trades` and `expectancy_R` before CAGR.** This is the one place this harness
departs from the lab's others, and it matters. A weight-based strategy trades every
day, so its CAGR averages hundreds of decisions. A 6-slot breakout book may take 100
trades in two years — at that count CAGR is mostly noise, while the exit mix is not.
`expectancy_R` (mean profit per trade in units of risk) is the only scale on which two
configurations with *different stop widths* are comparable at all.

The R sweep on the bundled synthetic universe shows the shape to expect:

| `r_mult` | trades | hit rate | expectancy | stopped | took profit | timed out |
|---|---|---|---|---|---|---|
| 0.5 | 159 | 21% | **+0.39R** | 75% | 15% | 10% |
| 1.0 *(notebook)* | 120 | 31% | +0.35R | 54% | 22% | 24% |
| 1.5 | 110 | 38% | +0.26R | 37% | 17% | 46% |
| 2.0 | 106 | 44% | +0.25R | 21% | 14% | 65% |
| 3.0 | 113 | 44% | +0.13R | 2% | 7% | 91% |

Widening the stop walks the book from stop-dominated to time-dominated and lifts the
hit rate, while expectancy per unit of risk falls — you are paying for those extra
winners with a wider risk unit, and past ~2.0 the stop has stopped being a stop at all.
**These are synthetic numbers and prove nothing about markets**; the table is here to
show what the harness reports, not what to set.

`min_off_high_pct` turns the Finviz screen's 52-week-high filter from a *ceiling* into
a **band**. It is switched off by default (`0.0` reproduces the screener exactly) and
exists to test one hypothesis: the screen selects names within 10% of their high, which
are names that have *already* broken out, while the entry needs names approaching
resistance. Selecting 5–20% below the high instead should give the breakout something
to break. **Untested** — it is a knob for the experiment, not a recommendation.

`r_mult` and `use_var_cap` are new parameters, added because the default exit mix came
back 54% `stop_R`. R is `min(|entry × Var95|, gap/2 + ADR/2)`, and the Var95 cap
usually binds — Var95 on a volatile name is a single bad session, which a breakout
routinely gives back before it works.

**Cost of each look-ahead path.** `lookahead_cost()` prices the notebook's three leaks
one at a time, so you can see what its results were worth that a live trader could not
have had:

```python
lookahead_cost(hourly, wl, prices["BOXX"])   # 5 rows: causal, notebook, and each leak alone
```

Don't read a small gap on synthetic bars as "the leak was harmless" — generated prices
have no real pivot structure for `find_peaks` to exploit, so knowing future peaks buys
little there. On real bars, where a level genuinely marks where a stock turned, expect
more. That table exists to measure it on *your* data.

**Sweeps are not cheap, and the harness knows it.** Generating candidates is the
expensive half (levels are re-derived per name per week) and depends only on
`BreakoutParams`. Rows are grouped by their signal parameters, candidates are generated
once per group, and every `WeeklyBookParams` variation reuses them — so sweeping
`n_slots` or `watchlist_size` costs one generation for the whole column. Measured on
the synthetic fixture that is **~4× faster** than the naive loop. A test asserts a
cached row is identical to running the book from scratch.

#### Does the selection actually break out?

`notebooks/breakout_success_rate.ipynb` answers a different question from the backtest.
The book's return mixes the **selection**, the **sizing** and the **slot cap**; the
funnel isolates the first:

```
selected -> had resistance overhead -> crossed it in the waiting window
         -> the cross held the confirmation -> reached +1R
```

The **waiting window** (`wait_days`, default 7) is how long after the weekend a first
breakout still counts — exactly the week the watchlist covers. It runs on the cached
daily closes, so unlike the backtest above it produces **real numbers today**: 2,815
Finviz selections across 142 weeks of Nasdaq-100 data.

| stage | n | of picks | of previous |
|---|---|---|---|
| selected | 2,815 | 100% | — |
| had resistance overhead | 1,347 | 47.9% | 47.9% |
| crossed it in the window | 653 | 23.2% | 48.5% |
| cross confirmed (an entry) | 370 | 13.1% | 56.7% |
| reached +1R | 186 | 6.6% | 50.3% |
| closed profitable | 142 | 5.0% | 76.3% |

**The biggest loss is at the first stage, and it is structural.** The Finviz screen
requires a name to be *within 10% of its 52-week high* — and a name that close to its
high has usually already cleared every level its chart shows. **52% of picks had
nothing overhead to break out through.** The selection rule filters *for* names that
have already broken out; the entry rule needs names that have not. These two strategies
are working against each other by construction.

**Read the tail before the mean.** Of the 370 confirmed entries: 38% closed profitable,
mean **+0.41R**, median **−0.66R**. A positive mean with a negative median means the
edge is entirely in the right tail — and here it is extreme:

> **the top 10 trades are 90% of all R earned** across 369 closed trades
> (the top 3 alone are 39%)

Every one of those is a 2025–26 semiconductor or mega-cap tech name. Strip them and
there is no strategy left. That is not a reason to dismiss it — positive skew is what a
breakout book is *supposed* to look like — but it means the mean is an estimate of a
fat tail from a handful of trades in one regime.

**The screen's own ranking predicts the tail, not the hit rate:**

| RS rank | picks | has resistance | entries | win rate | mean R | median R |
|---|---|---|---|---|---|---|
| 1–5 | 710 | 37.3% | 84 | 39.3% | **+1.37** | −0.89 |
| 6–10 | 710 | 45.9% | 94 | 38.3% | +0.21 | −0.62 |
| 11–15 | 705 | 50.2% | 81 | 37.0% | +0.14 | −0.64 |
| 16–20 | 690 | 58.3% | 111 | 38.7% | +0.03 | −0.66 |

Two columns saying opposite things. `has resistance` **rises** with rank — the
strongest names are the least likely to have anything overhead, so the best picks are
the least tradeable by this entry. `mean R` **falls** with rank. But `win rate` is flat
at ~38% across every bucket, so the ranking is not predicting *whether* a breakout
works, only how far the winners run.

**`+1.37R` is not something to expect** (notebook §8b exists to stop that reading). R is
profit in units of risk on a trade that happens for only 11.8% of top-5 picks — median
1R is 2.85% of price, so it is ~3.9% of a *position*, not of the book. Dropping the 5
best of 84 trades takes it to +0.42R; the bootstrap CI is [+0.34R, +2.53R]; the median
trade is −0.89R. By year it is +0.86R (2024), +0.26R (2025), **+6.05R on 11 trades
(2026)**, and three tickers are 69% of the bucket's total R.

The *direction* does survive — permutation p = 0.002, positive in both halves and all
three years — and running the book on a 5-name watchlist beats the 20-name version
(§8c). But survivorship bias bites hardest exactly here: today's Nasdaq-100 over-
represents names that went on to become large, and ranking by RS inside it selects for
precisely those. Treat concentration as a hypothesis for point-in-time data, not a
finding.

⚠️ Close-only bars: High/Low are synthesised as the close-to-close envelope, which
understates the true range, so R is smaller and stops are tighter than they would be
live. The success rates above are **conservative**. Re-run on `load_hourly()` bars.

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
  screens.py      filter-based screens: the trend template and the high-momentum
                  screen, both rolled forward from a notebook so they can be
                  backtested
  breakout.py     hourly resistance-breakout trading + the weekly six-slot book;
                  its own fill-level accounting, because weights cannot express a stop
  engine.py       one backtest function: lag, commission, slippage, equity curve
  metrics.py      CAGR, Sharpe/Sortino vs BOXX, drawdown, turnover, trade log
  plotting.py     the chart system
  pipeline.py     load -> signals -> backtest in one call; sweep_band(),
                  sweep_vix(), sweep_target_vol()
  breadth.py      market breadth: 4% movers, % above the MAs, index stretch in
                  ATR units, momentum leaders and their sector concentration
  agent/          an LLM analyst that READS the results above
    env.py          .env loading: the shell wins, and no value is ever printed
    evidence.py     the lab's own numbers as text, each with its caveat attached
    fundamentals.py yfinance company data, cached      (no LangChain import)
    news.py         web search + Yahoo headlines       (no LangChain import)
    tools.py        the three above, as LangChain tools
    analyst.py      a Gemini agent that may call them
run_backtest.py   CLI
dashboard/app.py  Streamlit: daily picks + market overview + analyst
notebooks/backtest_visualization.ipynb
notebooks/breakout_success_rate.ipynb  the selection -> breakout funnel
tests/test_qbs.py 166 tests: indicators, engine, momentum, circuit-breaker,
                  vol-target and screen invariants (each strategy gets a
                  shuffled-future look-ahead test)
tests/test_agent.py 61 tests: the analyst's data layers, .env loading and
                  precedence, the kill switch, its tools, and one real agent
                  run driven by a scripted model (no key, no network)
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

## Dashboard

```bash
pip install -r requirements.txt -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Two tabs:

- **Daily picks** — what each of the two selection strategies held on any chosen day,
  the entries and exits that changed it, how many names they hold in common, and a
  downloadable history table. A chart panel on the right plots any name with its
  10/20/50/200-day EMAs and the nearest support and resistance levels.

  The levels are re-derived **from history up to the selected date only** — the same
  causal rule `candidate_trades` uses, so the chart never draws a line the strategy
  could not have seen on that date. When a name has cleared everything overhead the
  panel says so outright — it is in price discovery, with nothing above it to break
  through.

  **Below the plot, the same name in numbers:** trailing returns over 1w/1m/3m/6m/12m
  plus the book's own momentum score, each with its **percentile rank in the universe on
  that date** and the universe median beside it — a twelve-month return means nothing
  alone, since the question a momentum strategy asks is relative. Then where price sits
  against each EMA, the SMA 200, its 52-week high and its own ATR.

  Last comes the row that ties the panel to the three above it: **every gate each
  strategy actually applies, with the reading that decides it and a ✅/❌** — and *only*
  the gates that are switched on. A row for a filter nobody is running describes a
  strategy that does not exist, and the table gives the reader no way to tell.

  Every threshold in that table is read off `FinvizScreenParams`, `MomentumParams` and
  `BreadthParams` rather than written out again, so retuning a strategy retunes its gate
  rows, labels and verdicts with it; turn the band floor on and its row appears, turn
  "quarter up" on and its row appears, change the ranker's lookback and every row
  naming it follows. **That last one is not hypothetical** — the book moved from 12-1 to 6-1,
  and a panel with the horizon written into it would now be labelling a number the
  strategy does not use with the name of the rule it claims to explain.

  One leg is deliberately missing: the volume test (> 300k shares/day), which the
  screen and the leader group both apply and which needs volume data this panel does
  not carry — so clearing every row shown is **necessary but not sufficient**. The
  screen and the leader group show the same rules because they *are* the same
  definition, held in two places (`FinvizScreenParams`, `BreadthParams`) so either can
  be retuned alone. The momentum hurdle is BOXX over the same window where a safe asset
  is supplied, and zero where it is not — the row says which.

  Price is drawn as **candlesticks**, which need real Open/High/Low. The universe
  cache holds closes only, so the panel fetches daily OHLC for the *selected name*
  alone (`data.load_daily_ohlc`, cached per ticker under `data/ohlc/`). Where that
  cannot be had it draws a close line and says why. **Candles are never synthesised
  from closes** — a body spanning previous-close to close with no wick asserts a
  session high and low that never happened, and a chart that lies about range is worse
  than one that admits it only has closes.
- **Market overview** — the breadth monitor: 4% up/down counts, percent holding the
  20- and 50-day averages, index distance from its 50-day EMA in ATR units, and the
  momentum-leader group with its count trend.

**Data source** is a sidebar control:

- **Online (default)** — downloads and caches. It only reaches the network when the
  cache is actually behind, so a fresh cache costs nothing. If the download fails it
  falls back to the cache and says so rather than pretending to be live.
- **Offline** — cache only, never touches the network. Build the cache first with
  `python run_backtest.py`.

**Refresh now** forces a re-download even when the cache looks current.

Every tab carries a freshness banner: how many sessions behind the data is, and what
to do about it. The remedy is context-aware — it will not tell you to switch to Online
while an online download is the thing failing. Staleness is counted in weekdays with no
exchange-holiday calendar, so around a holiday it nags a day early, which is the safe
direction.

### The market tab measures the US market

A toggle on that tab picks the universe. **On** (the default) pulls the broad US
universe from the Finviz screener — the same definition the commercial dashboards
quote:

| Finviz filter | Value |
|---|---|
| Industry | `Stocks only (ex-Funds)` — drops ETFs and closed-end funds |
| Price | `Over $5` |
| Average Volume | `Over 300K` |

That is ~2,400 names rather than 99, which is what makes the readings *breadth* rather
than an index summary. Two panels light up as a consequence:

- **Sector concentration** — the screener returns `Sector` in the same response, so the
  share / pool-weight / penetration / excess-pp table now computes. `Excess pp` is the
  column to read: a sector holding 20% of the leaders is unremarkable if it *is* 20% of
  the universe.
- **The volume leg of the leader screen** — volume arrives in the same yfinance
  response as the closes, so the 300k-share test applies instead of being skipped.

#### What counts as a high-momentum stock

The momentum-leader group ("動力股") on that tab is three tests, all strictly
greater-than, applied to every name in the universe on every date:

| Leg | Threshold | Parameter |
|---|---|---|
| Any US stock or ADR, price | **> $5** | `leader_min_price` |
| Share volume | **> 300,000/day** | `leader_min_volume` |
| Quarterly gain, over 63 sessions | **> 28%** | `leader_min_quarter_return` |

**The volume leg counts shares, and price does not enter it.** That is worth stating
because the leg previously held a flat $5m/day dollar floor, and neither test is a
stricter version of the other:

| | 400k shares at $6 | 50k shares at $200 |
|---|---|---|
| Dollar volume | $2.4m/day | $10m/day |
| Old `> $5m/day` | ❌ | ✅ |
| Now `> 300k shares` | ✅ | ❌ |

A share count scales the dollar bar with price — 300k shares is $1.5m/day at $5 and
$30m/day at $100 — where the old test held the money constant and let the share count
float.

**At 300k it is close to non-binding on either universe this package uses.** The Finviz
screen already filters to names *averaging* over 300k shares, and every Nasdaq-100
constituent trades far above it, so on most days this leg removes nobody: it catches an
unusually quiet session rather than an illiquid name. Treat it as a sanity check, and
do not read the leader count as liquidity-screened beyond what the universe filter
already did. (Same-day volume, not a rolling average — a quiet holiday session can
drop a name out for a day. Say so if you would rather it averaged.)

Strictly greater-than on all three, which matters most on price: a stock sitting at
exactly $5.00 is a common thing, and `>=` would admit names the universe screen itself
excludes, so the two filters would disagree about the same name.

The 28% was 20% until it was raised by hand. It is a **preference about how selective
"high momentum" should be, not a measured optimum** — raising it shrinks the leader
count and the sector table built on it, and nothing here claims the smaller group
performs better. It lives in a diff for that reason.

**One interaction to know about.** The universe filter (`Average Volume > 300K`) now
measures the same quantity as the leader rule's volume leg, at the same threshold — the
only difference being that Finviz averages it and the leader rule reads the single
session. That is why the leg barely bites in the US-universe mode: anything that got
into the universe already clears it on a normal day.

**What it costs.** The screener paginates at 20 rows a page, so ~2,400 names is ~120
requests — minutes, not seconds, cached for a day. Prices for 2,400 names is a real
download and the cache runs to tens of megabytes. Finviz is a scrape, not an API: it
rate-limits and the layout is not a contract, so every entry point returns None rather
than half a universe — **a breadth reading over a truncated sample is wrong in a way
that looks entirely plausible.**

Turn the toggle **off** and it falls back to the cached Nasdaq-100 and says plainly
that it is measuring an index, not the market. If the Finviz fetch fails it does the
same thing with a red banner rather than quietly substituting the smaller universe —
and the banner carries **the actual reason**, not a generic "unavailable".

**If the US universe will not load:**

```bash
python -m qbs.finviz     # in the SAME environment that runs streamlit
```

Streamlit swallows tracebacks, so that check runs the same steps outside it and names
the one that breaks — interpreter, package, filter encoding, screener, yfinance:

```
  python         /usr/bin/python3
  finvizfinance  OK (v1.5.0)
  filters        OK — ind_stocksonly,sh_price_o5,sh_avgvol_o300
  screener       OK — 2431 tickers, 11 sectors
  yfinance       OK
```

The commonest cause is the dullest: `finvizfinance` installed in a notebook or on
Colab is **not** installed for the interpreter running Streamlit. `pip install -r
requirements-dashboard.txt` with that interpreter fixes it.

The SPY column and S&P 500 level are still blank — this package caches QQQ, not SPY.

---

## The LLM analyst

A Gemini agent, via LangChain, that reads everything above and writes about it. It
lives in `qbs/agent/`, is entirely optional, and **produces no numbers of its own**.

```bash
pip install -r requirements.txt -r requirements-agent.txt
cp .env.example .env && chmod 600 .env       # paste your key into it
python -m qbs.agent --check

python -m qbs.agent "Why is the momentum book holding names the screen rejects?"
python -m qbs.agent --report name --ticker MU      # no LLM, no key, no network
```

Or use the dashboard's **🤖 Analyst** tab, which hands the agent the frames the app has
already loaded instead of re-reading the cache.

### Where the key comes from

`qbs.agent` reads a `.env` at the repository root on import, so the CLI, the dashboard
and a notebook all pick it up with nothing further to do. `.env` is gitignored;
`.env.example` is the tracked template.

```
GOOGLE_API_KEY=...          # GEMINI_API_KEY works too — Google's docs use both
QBS_GEMINI_MODEL=gemini-2.5-flash    # optional
TAVILY_API_KEY=...                   # optional, better search than the default
QBS_DISABLE_ANALYST=1                # optional kill switch, see below
```

**A real environment variable always wins**, so `GOOGLE_API_KEY=... python -m qbs.agent
...` overrides the file for one run. `--check` says which source supplied each key,
by name:

```
$ python -m qbs.agent --check
.env:            loaded .env; set QBS_GEMINI_MODEL; kept the shell's GOOGLE_API_KEY
keys set:        GOOGLE_API_KEY, QBS_GEMINI_MODEL
google key:      found
model:           gemini-2.5-flash
search backends: duckduckgo
analyst:         ready
```

Nothing ever prints a value — `--check`, the dashboard caption and the load summary
carry key *names* and the file path only. The loader warns if `.env` is readable by
anyone but you, and points out an unrecognised key name, since a typo'd
`GOOGEL_API_KEY` otherwise presents as "the key is not set" while sitting in the file.

The syntax is deliberately small: `KEY=value`, one per line, `export` prefix ignored,
surrounding quotes stripped, `#` starting a comment only at the beginning of a line
(so a key containing `#` survives). No `$VAR` interpolation, no multi-line values —
export those from your shell instead. A malformed line is reported rather than
skipped, and the other lines still load.

Streamlit will not re-import a package it has already loaded, so **restart the
dashboard** after creating `.env`; a rerun alone will not pick the key up.

### Switching it off

```bash
QBS_DISABLE_ANALYST=1          # environment, or a line in .env
```

No Gemini call is made from anywhere in the package: `check_requirements` reports it,
`analyse` returns an `Answer` with `disabled=True` before loading anything, and
`build_model` refuses — so nothing reaches the API even from a caller that skipped the
check, and nothing is billed. **Everything that does not need the model keeps working**
— the picks, momentum profiles, breadth, fundamentals, search, and every
`python -m qbs.agent --report ...`. That is the point of a switch rather than an
uninstall.

Only an explicit off-word re-enables it — `0`, `false`, `no`, `off`, `none`,
`disabled`. **Any other non-empty value switches it off.** That is deliberately not
`qbs.live.config._env_bool`, which reads anything outside `("1", "true", "yes", "on")`
as false: for a flag whose job is to stop spending money, an unrecognised value has to
fail *safe*, so a half-remembered `QBS_DISABLE_ANALYST=disable` stops the spending
rather than quietly leaving it running.

`python -m qbs.agent --check` prints the switch on its own line and says **"DISABLED on
purpose"** rather than "not ready", because a deliberate shutdown and a missing key
have different remedies. The dashboard does the same: a paused notice with `unset`,
not the "create a `.env`" instructions, which would send you to fix something that is
not broken.

Streamlit reads the environment once at start-up, so **restart the app** after changing
this — a rerun alone will not pick it up.

### What it can look at

| Tool | Answers |
|---|---|
| `current_picks` | what each strategy holds on the latest bar, and how much they overlap |
| `name_momentum` | one name's returns, universe rank, location vs every MA, and each strategy gate |
| `market_breadth` | participation over the last N sessions, not the index |
| `strategy_performance` | the backtest comparison table |
| `breakout_funnel` | selection → breakout conversion, stage by stage |
| `breakout_trades` | trade statistics in R, with the concentration check |
| `fundamentals` | valuation, margins, growth, balance sheet, analyst view (yfinance) |
| `search_news` / `ticker_headlines` | the web, and Yahoo's feed for one symbol |

### The problem this is built around

Ask a language model about a backtest and it will produce a fluent, confident,
plausible paragraph **whether or not it has the numbers**. That paragraph is
indistinguishable from a correct one until you check it. So:

- **Every figure must come from a tool call.** The system prompt says an answer
  containing an unsourced number is worse than no answer, and tells the model to say
  "I don't have that" and name the missing tool instead.
- **Caveats are welded to the numbers.** `breakout_trades` cannot return
  "expectancy +1.37R" without also returning "over 11 closed trades, top 3 are 39% of
  all R, below the 30-trade floor, do not size off it". The model has nowhere to put a
  clean number.
- **Every call is shown.** The CLI prints the tool list; the dashboard renders each
  call and its full output in an expander under the answer. A figure that appears in
  the prose and in none of the traces is a fabrication, and you can see that in a
  glance rather than by re-deriving it.
- **Nothing writes.** No tool changes a parameter, places an order, or produces an
  input another run reads back. The agent is a reader of results.

### Fundamentals are not a signal

`yfinance`'s `Ticker.info` is a snapshot of **today**: today's trailing P/E, today's
analyst target, today's short interest. There is no history in it and no way to ask
what a ratio was in March. Feeding any of it into a backtest would date today's balance
sheet back over the whole sample and report a return nobody could have earned.

So the payload carries `backtest_safe=False` and an `as_of`, the rendered text leads
with "a snapshot of today only", and the system prompt forbids using a fundamental to
explain any dated signal. A missing field stays missing — never filled with a zero that
would read as a measurement.

### Search results are data, never instructions

Web results reach a model that can call tools, so they arrive fenced in an
`<untrusted_search_results>` block whose own text says what the fence means. A result
containing "ignore your previous instructions" is quoted verbatim and treated as a
string. Run with `--no-web` (or untick the box in the sidebar) and the search tools are
**absent** rather than blocked, so the model cannot report having tried.

### The escape hatch

`python -m qbs.agent --report picks|name|breadth|universe|fundamentals|news` prints
exactly what the agent would read, with no model, no key and no LangChain installed.
When an answer looks wrong, diff it against the report rather than re-prompting.

### What it is not

A research note from a capable but unaccountable junior. The numbers in it are
checkable against the tool traces; check them. Nothing here constitutes advice, and the
model is instructed not to issue buy/sell calls or position sizes.

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
