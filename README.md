# QQQ / BOXX strategy lab

Five trading strategies with BOXX as the cash leg, one backtest engine, and a notebook
that shows you where every signal fired.

- **Larry Connors RSI(2)** — short-term mean reversion, long-only, filtered by SMA(200)
- **GEM (Global Equities Momentum)** — Antonacci dual momentum across QQQ / VEU / BOXX
- **Volatility-targeting overlay** — scales QQQ exposure so forecast vol sits near target
- **Top-6 Nasdaq-100 momentum** — cross-sectional 12-1 momentum with a hysteresis band
- **Top-6 + VIX circuit breaker** — the same book, switched off entirely when VIX spikes

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

## Layout

```
qbs/
  config.py       window, all strategy parameters, chart palette
  data.py         yfinance download + CSV cache + synthetic market and VIX generators
  universe.py     Nasdaq-100 membership, point-in-time hook, wide price loader
  indicators.py   Wilder RSI, SMA, EWMA vol, trailing return, drawdown
  strategies.py   the five strategies -> target weights + diagnostics + events
  engine.py       one backtest function: lag, commission, slippage, equity curve
  metrics.py      CAGR, Sharpe/Sortino vs BOXX, drawdown, turnover, trade log
  plotting.py     the chart system
  pipeline.py     load -> signals -> backtest in one call; sweep_band(), sweep_vix()
run_backtest.py   CLI
notebooks/backtest_visualization.ipynb
tests/test_qbs.py 43 tests: indicators, engine, momentum and circuit-breaker invariants
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
from qbs.config import Config, MomentumParams, RSI2Params, VolTargetParams
from qbs.pipeline import run, sweep_band

cfg = Config()
cfg.backtest_start = "2024-09-01"
cfg.slippage_bps = 10.0                                      # pessimistic fills
cfg.vol = VolTargetParams(target_vol=0.12, max_weight=1.5)   # allow gearing
cfg.rsi2 = RSI2Params(entry_threshold=10)                    # trade more often
cfg.momentum = MomentumParams(n_hold=8, exit_rank=20,        # wider band, less churn
                              rebalance="ME")                # monthly instead of daily

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

Adding a sixth strategy means writing a function that returns a `StrategySignals`
and adding it to `pipeline.build_signals`. Everything downstream — engine, metrics,
charts — works on it unchanged.

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
