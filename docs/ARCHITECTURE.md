# Module guide

What every file in this program does, why it is shaped that way, and the code
that carries the decision.

The program has two halves that share one strategy definition:

- **`qbs/`** — the backtest. Loads prices, builds signals, simulates, measures,
  charts.
- **`qbs/live/`** — the deployment. Runs the *same* signal functions against
  today's prices and sends the resulting orders to Interactive Brokers.

They are not two implementations of the strategy. The live layer imports
`cross_sectional_momentum` and `book_vol_target` from `qbs.strategies` and calls
them exactly as the backtest does. That is the single most important structural
fact about this codebase, and most of the design below exists to protect it.

```
                     ┌──────────────────────────────────────┐
   yfinance ────────▶│  data.py        universe.py          │
                     │  QQQ/VEU/BOXX   ~100 NDX constituents │
                     └───────────────┬──────────────────────┘
                                     │  wide DataFrame: date × ticker
                                     ▼
                     ┌──────────────────────────────────────┐
                     │  indicators.py → strategies.py       │
                     │  six builders → StrategySignals      │
                     │  (target weights DECIDED at close t) │
                     └───────────────┬──────────────────────┘
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
   ┌────────────────────────┐              ┌──────────────────────────┐
   │ BACKTEST               │              │ LIVE                     │
   │ engine.py  (lag, costs)│              │ live/signals.py  last row│
   │ metrics.py (stats)     │              │ live/orders.py   → shares│
   │ plotting.py            │              │ live/broker.py   → IB    │
   │ pipeline.py  run()     │              │ live/store.py    → SQLite│
   └────────────────────────┘              └──────────────────────────┘
```

---

## The one convention everything rests on

`signals.weights.loc[t]` means **what the strategy decided while looking at the
close of day `t`**. It does *not* mean "the position held on day t".

No strategy applies the execution lag. That happens in exactly one place —
`engine.run_backtest` — so every strategy is compared under identical execution
assumptions, and a strategy cannot accidentally trade on information it did not
have. A strategy that looked good only because it had a different lag would be
a bug, not a finding.

This convention is also why the live layer is safe to build the way it is: if
weights always mean "decided at this close", then the last row of a
freshly-computed backtest *is* today's target book, with no special-casing.

---

# Part 1 — the backtest package (`qbs/`)

## `config.py` — 252 lines

Every number a user is likely to tune, in one file, so the strategy modules stay
free of magic constants. One frozen dataclass per strategy:

| Class | Governs |
|---|---|
| `RSI2Params` | Connors RSI(2) thresholds, trend filter, time stop |
| `GEMParams` | dual-momentum lookback, the risk sleeves, rebalance frequency |
| `VolTargetParams` | single-asset vol targeting |
| `MomentumParams` | 12-1 lookback, `n_hold`, the hysteresis band, weighting |
| `VixBreakerParams` | the circuit breaker's state machine thresholds |
| `BookVolTargetParams` | book-level vol targeting (the drawdown control) |
| `Config` | holds one of each, plus window and cost assumptions |

```python
BACKTEST_START = "2025-01-20"   # Trump inauguration
DOWNLOAD_START = "2023-06-01"   # covers WARMUP_DAYS + GEM's 12m lookback
EXECUTION_LAG  = 1              # decided at close t -> held from t+1
COST_BPS       = 1.0            # commission per 100% turnover
TRADING_DAYS   = 252
```

`MomentumParams` validates in `__post_init__`, because an inverted band is not a
strategy variant, it is a typo:

```python
def __post_init__(self):
    if self.exit_rank < self.n_hold:
        raise ValueError("exit_rank must be >= n_hold (the band cannot be negative)")
```

Also here: `PALETTE` (a colour-vision-safe categorical set, validated all-pairs)
and `STRATEGY_LABELS`. Benchmarks deliberately sit outside the categorical
palette in grey, so they never compete with a strategy for identity.

## `data.py` — 256 lines

Price loading, deliberately boring. Returns one wide DataFrame of
split/dividend-adjusted closes, indexed by date, columns = tickers, no gaps.
Everything downstream assumes that shape.

```python
def load_prices(tickers=TICKERS, start=DOWNLOAD_START, end=None,
                use_cache=True, refresh=False, offline=False) -> pd.DataFrame
def load_vix(...)          # ^VIX, cached as data/_VIX.csv
def synthetic_prices(...)  # a reproducible fake market
def synthetic_vix(...)
```

Three modes, and the reason for each:

- **Live** — downloads via `yfinance`, writes `data/<TICKER>.csv`.
- **`offline=True`** — reads only the cache, never touches the network. Lets the
  whole pipeline run on a plane, and lets the live deployment debug a signal
  when the feed is down.
- **`synthetic_prices()`** — a generated market so the pipeline can be unit
  tested with no data feed at all. Every test in `tests/test_qbs.py` that needs
  prices uses this.

`yfinance` is imported *lazily*, inside the download function, so offline use
does not require it installed.

## `universe.py` — 273 lines

The Nasdaq-100 ranking universe, and the file with the loudest warning in the
repo:

> Ranking against *today's* index membership is survivorship bias, and for a
> momentum strategy it is the worst kind: today's Nasdaq-100 is, almost by
> definition, a list of stocks that went up.

Three sources, best first:

```python
load_pit_universe(path)   # point-in-time date,ticker rows -- the only unbiased option
fetch_ndx()               # scrape current constituents; raises rather than silently falling back
NDX_FALLBACK              # hardcoded list, offline convenience only
```

`membership_mask()` turns any of them into a date × ticker boolean grid the
ranker respects, so switching to point-in-time data later changes one line and
nothing else:

```python
def membership_mask(index, tickers, pit=None) -> pd.DataFrame:
    """With pit=None every ticker is eligible on every date -- the biased case."""
```

`load_universe_prices()` downloads ~100 symbols in batches (yfinance gets
unreliable with 100 in one request) and caches them as a single CSV. Tickers
that come back empty are **reported and dropped**, not silently left as NaN
columns that would quietly shrink the candidate pool.

`synthetic_universe()` generates 100 fake names with a *genuine* cross-sectional
momentum effect — each gets a slowly mean-reverting drift, so past winners
really do tend to keep winning. Without that, a momentum backtest on random
walks tests nothing: it would rank pure noise, and a strategy that "worked" on
it would be a bug.

## `indicators.py` — 95 lines

Pure functions on a Series. No state, no I/O. Every one returns a Series aligned
to its input with NaN in the warm-up region — and nothing forward-fills across
that warm-up, because a NaN means "not defined yet" and the strategy layer must
treat it as *no signal* rather than as zero.

```python
wilder_rsi(close, period=2)      # Wilder's RSI, correctly seeded
sma(close, window)
realized_vol(returns, halflife=20, min_periods=20, annualise=True)
total_return(close, periods)
drawdown(equity)
```

`wilder_rsi` is hand-rolled rather than `ewm()` because Connors' thresholds
assume Wilder's specific smoothing — seeded from the simple mean of the first
`period` changes, then an exponential filter with α = 1/period:

```python
ag[period] = np.nanmean(g[1:period + 1])       # the seed
al[period] = np.nanmean(l[1:period + 1])
a = 1.0 / period
for i in range(period + 1, len(close)):
    ag[i] = ag[i - 1] * (1 - a) + g[i] * a
    al[i] = al[i - 1] * (1 - a) + l[i] * a
```

A plain `ewm()` drifts for the first few dozen bars. With period=2 the drift
washes out fast, but the seed is cheap and correct.

## `strategies.py` — 741 lines

The six strategies. Each builder takes prices and parameters and returns a
`StrategySignals`:

```python
@dataclass
class StrategySignals:
    name: str
    weights: pd.DataFrame              # DECIDED at each close
    diagnostics: pd.DataFrame          # what the chart layer draws
    events: pd.DataFrame               # discrete buy/sell rows for markers
    params: Dict
    holding: Optional[pd.Series]       # which asset(s) held each day
    holdings_log: Optional[Dict]       # momentum: date -> [tickers]
    momentum: Optional[pd.DataFrame]   # momentum: the ranking scores
    held_ranks: Optional[Dict]         # momentum: date -> {ticker: rank}
```

### 1. `connors_rsi2` — mean reversion

Buy `RSI(2) < 5` while above SMA(200); exit on SMA(5), `RSI(2) > 70`, or a
10-day time stop. Parked in BOXX when flat.

Written as a **state machine, not vectorised masks**, and deliberately so: entry
and exit are path-dependent (you cannot exit a position you never opened), and
the day count in the time stop has no vectorised form.

The time stop is an addition to the canonical rules. Without it a trade that
neither recovers nor triggers an exit sits open indefinitely, which quietly
turns a mean-reversion system into a buy-and-hold one.

There is also a subtle trap handled explicitly:

```python
else:
    # trend_sma of 0 or 1 disables the filter. SMA(1) is the close itself,
    # so `close > SMA(1)` would be False on every bar and the strategy would
    # silently never trade -- exactly the kind of no-op a parameter sweep
    # would report as "the filter is essential".
```

### 2. `gem` — Antonacci dual momentum

Month-end only. Relative momentum picks QQQ or VEU; absolute momentum then gates
it against BOXX's **realised** return, not a fixed 0%:

```python
leader = row[risk_assets].idxmax()
safe_ret = row[p.safe_asset]
pick = leader if row[leader] > safe_ret else p.safe_asset
```

In a 4–5% T-bill world, "did equities go up?" and "did equities beat the cash I
could have held?" are different questions, and only the second is
decision-relevant.

### 3. `vol_target_overlay` — single-asset vol targeting

`weight = target_vol / forecast_vol`, clipped. Two details matter more than the
formula: a **no-trade band** (the raw ratio moves every day; trading that wiggle
burns the edge in costs for no risk benefit) and a **vol floor** (in a dead-calm
tape the ratio explodes).

Passing `base_weights` turns it into an overlay on another strategy.

### 4. `cross_sectional_momentum` — the stock selection

Rank the universe by 12-1 momentum, hold the top `n_hold`, exit only once a name
falls past `exit_rank`. Three decisions worth understanding:

**The band.** Without it a name slipping from rank 6 to 7 is sold and bought back
days later — pure churn, paid in spread and short-term gains.

```python
keep = [t for t in held if rank.get(t, np.inf) <= p.exit_rank]
for t in order.index:
    if len(keep) >= p.n_hold:
        break
    if t not in keep:
        keep.append(t)
held = keep
held_ranks[dt] = {t: float(rank.get(t, np.nan)) for t in held}
```

**Slot weighting, not survivor weighting.** With only 4 of 6 slots qualifying the
book is 4/6 invested and 2/6 cash:

```python
invested = len(held) / p.n_hold
w = pd.Series(1.0 / p.n_hold, index=held)
```

Spreading 100% across the survivors would concentrate the portfolio exactly when
the fewest names qualify — i.e. in a deteriorating market, which is precisely
backwards.

**12-1, not 12-0.** The most recent month is skipped because short-horizon returns
*reverse* rather than persist; including them mixes two effects with opposite
signs and blunts both.

`held_ranks` exists for the live run log: it records the rank each held name
survived at, so the log never has to re-derive it. A re-derivation would mean
reimplementing the eligibility and absolute-momentum filters, and a second copy
of that logic is what drifts.

### 5. `vix_circuit_breaker` — the one that failed

A three-state machine over any base strategy:

| State | Transition | Holds |
|---|---|---|
| `INVESTED` | VIX > `exit_level` → `CASH` | the base weights |
| `CASH` | `park_after_days` → `PARKED`; VIX < `entry_level` and dwell met → `INVESTED` | nothing, true 0% |
| `PARKED` | VIX < `entry_level` → `INVESTED` | the safe asset |

Cash before BOXX because buying the safe asset for a two-day scare costs two
spreads to earn two days of T-bill yield. Two thresholds because a single one
round-trips the whole book every time VIX oscillates across it.

**It does not work on this book, and the code says so.** At the default 17/16 it
turned +85% into −2.7%. Kept because the sweep that demonstrates the failure is
worth more than its absence — see §9e of the notebook.

### 6. `book_vol_target` — the drawdown control that does work

Scales an entire portfolio by *its own* realised volatility:

```python
rets = prices.reindex(columns=w.columns).ffill().pct_change().fillna(0.0)

# Realised book return, on the same convention the engine uses.
book_ret = (w.shift(lag) * rets).sum(axis=1)
vol = (book_ret.ewm(halflife=p.halflife, min_periods=p.min_periods).std()
       * np.sqrt(TRADING_DAYS))
raw = (p.target_vol / vol.clip(lower=p.vol_floor)).clip(0.0, p.max_weight)

# No-trade band: hold the last scalar until the target has drifted far
# enough to be worth the turnover.
out = np.full(len(raw), np.nan)
current = np.nan
for i, r in enumerate(raw.to_numpy()):
    if np.isnan(r):
        continue
    if np.isnan(current) or abs(r - current) >= p.rebalance_band:
        current = r
    out[i] = current
scalar = pd.Series(out, index=raw.index, name="scalar").ffill().fillna(0.0)

weights = w.copy()
weights[risk_cols] = w[risk_cols].mul(scalar, axis=0)
weights[safe] = 1.0 - weights[risk_cols].sum(axis=1)
```

**Why the book's own vol and not VIX.** The Top-6 book has two kinds of drawdown:

| Episode | Book DD | VIX then | QQQ then |
|---|---|---|---|
| 2025 market selloff | −31% | 28–45, visible | −22.8% |
| 2026 concentration blow-up | −30% | **16–20, invisible** | −8.8% |

In the second, the book lost 7–8% on days QQQ lost about 1%, with VIX below its
own median. Neither a VIX threshold nor a `QQQ > SMA(200)` gate reduces that
drawdown at all — measured, the SMA(200) gate leaves max drawdown unchanged at
−34.8% while costing half the return. The market was not what went wrong.

The book's realised vol rises in **both** cases. That is the entire argument.

Note the causality: the vol estimate at `t` is built from returns the book had
actually *earned* by the close of `t` (weights shifted by `lag` before being
multiplied by returns, exactly as the engine does it), and the resulting scalar
modifies the weight *decided* at `t`, which the engine lags again. A shuffled-
future test pins this.

## `engine.py` — 132 lines

One function does the accounting for every strategy.

```python
def run_backtest(prices, signals, start=None, end=None, lag=EXECUTION_LAG,
                 cost_bps=COST_BPS, slippage_bps=0.0, initial=1.0) -> BacktestResult
```

The whole simulation is about fifteen lines:

```python
px = prices.ffill()
w_decided = signals.weights.reindex(px.index).ffill().fillna(0.0)

# The lag lives here and nowhere else.
w_held = w_decided.shift(lag).fillna(0.0)
rets = px.pct_change().fillna(0.0)

# ... window trim happens AFTER the shift, so the first backtest day
# inherits the position decided on the last day before the window ...

gross = (w_held * rets[w_held.columns]).sum(axis=1)

turnover = w_held.diff().abs().sum(axis=1)
turnover.iloc[0] = w_held.iloc[0].abs().sum()   # cost of putting the book on
commission = turnover * (cost_bps / 1e4)
slippage = turnover * (slippage_bps / 1e4)
costs = commission + slippage

net = gross - costs
equity = initial * (1.0 + net).cumprod()
```

Two details:

- **A full switch is two legs.** QQQ → BOXX gives |Δ| = 1 + 1 = 2, which is
  right: you pay to sell one and to buy the other.
- **Slippage is tracked separately from commission** because it models a
  different thing — the gap between the price the signal was computed on (~15:30,
  when the decision job runs) and the price the order fills at (the 16:00
  auction). It is execution uncertainty, roughly unbiased, and keeping it
  separate lets you see how much of the result depends on assuming it away.
  This is also the assumption the live deployment must honour, which is why it
  submits MOC orders.

## `metrics.py` — 229 lines

Statistics, computed against BOXX's **realised** return rather than zero. Over a
window where cash paid 4–5%, a Sharpe measured against zero flatters everything
and flatters the defensive strategies most, because much of what they earn is
simply the cash rate.

```python
summarise(result, rf=None) -> Dict[str, float]
summary_table(results, rf=None, labels=None) -> pd.DataFrame
format_summary(df)              # percentages as percentages
monthly_returns / monthly_return_matrix / rolling_sharpe
holdings_runs(result)           # daily membership -> one row per continuous holding
trade_log(result)               # round trips reconstructed from events
```

`Avg risk exposure` is in the table for a specific reason: RSI(2) averages 8.5%
exposure over the cached window -- flat 91.5% of the time -- so most of its
return is BOXX yield. The column stops you reading
anything into its Sharpe without noticing that.

## `plotting.py` — 959 lines

The chart system. Three conventions it enforces:

- **One y-axis per panel, never two.** Price and RSI, or price and vol and
  exposure, get stacked panels sharing an x-axis. A dual-axis chart lets whoever
  drew it pick the story by picking the scaling.
- **Colour-vision-safe palette**, three categorical slots validated all-pairs;
  benchmarks sit outside it in grey.
- **Identity is never colour alone** — legend *and* direct end-labels on every
  multi-series chart; buy/sell markers differ in shape as well as hue.

Per-strategy panels (`plot_rsi2_signals`, `plot_gem_signals`,
`plot_voltarget_signals`, `plot_momentum_holdings`, `plot_vix_regimes`,
`plot_book_voltarget`) plus cross-cutting ones (`plot_equity_curves`,
`plot_drawdowns`, `plot_rolling_sharpe`, `plot_monthly_heatmap`,
`plot_return_scatter`) and sweep charts (`plot_band_sensitivity`,
`plot_vix_sweep`, `plot_target_vol_sweep`).

`plot_book_voltarget` deserves a note: its third panel plots VIX *underneath*
the book's own volatility on a shared x-axis, so the episodes where the book's
risk climbed while the index stayed calm are visible directly. That chart is the
argument for the whole overlay.

## `pipeline.py` — 285 lines

The whole thing in one call, so the notebook and the CLI share one code path.

```python
@dataclass
class Lab:
    prices, signals, results, rf, config, universe, combined, vix
    summary / summary_pretty      # properties

def run(cfg=None, ..., with_momentum=True, with_vix=True, with_book_vt=True,
        pit_membership=None) -> Lab
def sweep_band(lab, n_holds, exit_ranks) -> pd.DataFrame
def sweep_vix(lab, exit_levels, band) -> pd.DataFrame
def sweep_target_vol(lab, targets, base_key="momentum") -> pd.DataFrame
```

`run()` loads data, builds all seven books (five strategies + two benchmarks),
and backtests them **on identical assumptions** — same window, same lag, same
costs. The momentum books are backtested against `combined` (universe + safe
asset) while the rest use the three-ticker frame; that routing is the only
special-casing in the function.

The three sweeps exist because the honest way to read this is to look for a
*plateau*, not a best cell. `sweep_target_vol`'s docstring says so explicitly:
the dial is expected to be boring, and a kink would mean the overlay is doing
something other than rescaling.

---

# Part 2 — the CLI (`run_backtest.py`, 270 lines)

```bash
python run_backtest.py                         # download, backtest, table, charts
python run_backtest.py --offline --no-charts   # from cache, no matplotlib
python run_backtest.py --synthetic             # no network at all
python run_backtest.py --sweep-band --sweep-vix --sweep-target-vol
python run_backtest.py --target-vol-book 0.15  # a conservative book
```

Flags map onto `Config` fields, then it calls `pipeline.run()` and prints. It
also emits warnings the table alone would not convey — for example, when the VIX
breaker is invested less than 60% of the window:

```
** the trigger sits inside the VIX distribution -- this is a
   mostly-out-of-market strategy, not a crash filter. Run --sweep-vix. **
```

---

# Part 3 — the live package (`qbs/live/`)

Ordered by how far each module sits from the broker.

## `live/config.py` — 169 lines

Deployment settings, deliberately separate from `qbs/config.py`. That file holds
*strategy* parameters, which must stay identical between backtest and live book
or the two stop being comparable. This one holds *deployment* parameters —
account, connection, sizing, safety limits — which have no meaning in a
backtest.

```python
@dataclass
class LiveConfig:
    ib_host, ib_port, ib_client_id, ib_account, connect_timeout, order_timeout
    notional, hold_safe_asset, safe_asset
    min_order_shares, min_order_notional
    max_order_notional, max_gross_turnover, max_positions
    moc_cutoff_hhmm, max_price_staleness_days, min_universe_coverage
    allow_live_account
    state_dir, kill_switch, dry_run, market_tz, extra_tickers
```

Loading is JSON-then-environment, with the environment always winning, so a
systemd drop-in can override one value without rewriting a tracked file. Unknown
JSON keys raise rather than being ignored — a typo in a config that silently
does nothing is worse than a crash.

```python
@property
def is_paper_port(self) -> bool:
    """IB's paper ports. 4001/7496 are the live ones and are refused by default."""
    return self.ib_port in (4002, 7497)
```

**Sizing is a fixed dollar figure, not the account's NLV.** That makes position
sizes reproducible: a difference between two days' order lists is always a
*signal* change, never an account-value change. The cost is that it does not
compound.

## `live/signals.py` — 297 lines

Today's target weights, computed by the same code that produced the backtest.

```python
@dataclass
class TargetBook:
    asof: pd.Timestamp
    weights: Dict[str, float]      # ticker -> target weight, sums to ~1.0
    prices: Dict[str, float]
    scalar: float                  # the vol-target scalar applied
    book_vol: float
    raw_holdings: List[str]
    n_rankable: int
    universe_size: int
    diagnostics: Dict[str, float]
    selection: List[Dict]          # entry/exit/hold, with rank and score
```

The core is short, because it delegates everything:

```python
mom = cross_sectional_momentum(uni, prices[safe], cfg.momentum)

combined = uni.copy()
combined[safe] = prices[safe]
vt = book_vol_target(mom, combined, cfg.book_vol, lag=cfg.execution_lag)

asof = vt.weights.index[-1]
row = vt.weights.loc[asof]
```

**Why the whole history is recomputed every day.** It would be cheaper to carry
yesterday's state forward. It would also be wrong within a fortnight: both the
hysteresis band and the vol scalar are *path-dependent*. The band needs to know
which names are currently held; the scalar needs the book's own realised return
series. Persisting that path means any missed run, any crash mid-write, any
manual intervention silently forks the live book from what the strategy says —
and you would not find out until you compared them months later.

Recomputing costs a few seconds on a t3.small and makes the live weights a pure
function of the price history. Miss a day and the next run is still exactly
right.

**The data-quality gate.** Every check is a "this is more likely a data bug than
a signal" test, and each raises rather than degrading:

```python
def check_data_quality(px, requested, safe_asset, max_staleness_days,
                       min_coverage, now=None) -> Dict[str, float]
```

- prices staler than a long weekend
- the safe asset missing or empty
- fewer than 85% of requested tickers present
- a *last bar* with prices for under 85% of the universe (a partial download)

A half-downloaded universe does not produce a slightly wrong ranking. It
produces a confident ranking of the wrong candidate set, which is far more
dangerous than no ranking at all.

**One bug worth remembering** is fossilised in `load_live_prices`: the universe
and the safe asset load by *different* paths, mirroring `pipeline.run()`. The
universe cache holds NDX members only, so asking it for BOXX returns a frame
silently missing the cash leg — and every weight downstream would be wrong.

## `live/orders.py` — 250 lines

Target weights → orders. **Pure**: no IB, no network, no clock. That is what
makes the risky part of an unattended trading loop testable.

```
target weights -> target shares -> deltas vs actual -> filter dust -> guards
```

Guards run *last*, on the final list, because that is the only point where "how
much is this session about to trade" is known.

```python
target_shares(weights, prices, notional) -> Dict[str, int]
diff_positions(target, actual, prices, min_shares, min_notional) -> List[Order]
apply_guards(orders, notional, target, ..., safe_asset="BOXX") -> List[Order]
build_orders(...) -> tuple[List[Order], Dict[str, int]]
```

Shares round **down**, so accumulated rounding across six positions plus the cash
leg cannot push the book above `notional`. The dropped fraction lands in cash,
which is the safe direction.

`diff_positions` emits **sells before buys**, so cash from an exit is available
for the entry that replaces it, and skips deltas too small to be worth a
spread — but never skips an exit as dust.

The guards are **all-or-nothing**:

```python
too_big = [o for o in orders
           if o.symbol != safe_asset and o.notional > max_order_notional]
if too_big:
    raise GuardTripped(...)

safe_leg = [o for o in orders
            if o.symbol == safe_asset and o.notional > notional * 1.01]
if safe_leg:
    raise GuardTripped(f"{safe_asset} order ... exceeds the book size")

gross = sum(o.notional for o in orders)
if gross > max_gross_turnover * notional:
    raise GuardTripped("... This is usually a data bug, not a signal ...")
```

Sending the orders that pass while dropping the one that tripped leaves the book
in a state no part of the system intended — half-rebalanced, with the strategy
believing it is fully positioned.

Note the **safe-asset exemption**, which is a real bug caught by running the
order list rather than by review: with the vol scalar at 0.36, BOXX is 64% of a
$100k book *by construction*, and a $40k per-name cap tripped on a completely
normal session. The per-name cap is about single-stock risk; the cash leg is
capped at the book size instead.

## `live/broker.py` — 289 lines

The `ib_async` layer — everything that talks to IB Gateway, and nothing else.
Kept thin and free of strategy logic so `signals.py` and `orders.py` stay
testable without a Gateway.

```python
class IBBroker:                     # context manager
    connect() / disconnect()
    positions()        -> Dict[str, int]      # STK only
    net_liquidation()  -> float
    cash_balance()     -> float
    portfolio_marks()  -> List[Dict]          # end-of-day marks
    qualify(symbols)   -> Dict[str, Contract]
    submit_moc(orders) -> List[Fill]
    cancel_all_open()  -> int
    todays_fills()     -> List[Fill]
```

**MOC is the order type that reproduces the backtest.** The backtest decides on a
price near the close and assumes the fill happens in the auction — that gap is
exactly what `slippage_bps = 5` models. IB requires MOC well before the auction
(cutoff around 15:45–15:50 ET), which is why the job submits at 15:40.

Three refusals, all deliberate:

```python
if not self.cfg.is_paper_port and not self.cfg.allow_live_account:
    raise BrokerError(f"port {self.cfg.ib_port} is a LIVE trading port. Refusing.")
```

```python
if getattr(c, "secType", None) != "STK":
    log.info("ignoring non-stock position: %s %s", c.secType, c.symbol)
    continue     # an option or future belongs to something other than this system
```

```python
if status in ("Inactive", "ApiCancelled", "Cancelled"):
    raise BrokerError(f"IB rejected {o.action} {o.quantity} {o.symbol}: {status}")
```

`portfolio_marks()` takes prices from the broker rather than re-fetching from
yfinance, because at 16:15 ET the official close may not have reached a free feed
yet — and a mark that disagrees with the account it describes is worse than no
mark.

## `live/state.py` — 96 lines

A small rolling record of run outcomes, so one phase can see what the previous
one did. Deliberately *not* the analytical record — that is `store.py`.

```python
load_state(path) / save_state(path, state)
record_run(path, phase, status, detail, keep=400)
kill_switch_engaged(path) -> bool
```

Writes are atomic — temp file in the same directory, fsync, rename:

```python
def _atomic_write(path: str, text: str) -> None:
    """A half-written state file after an instance stop is worse than no state
    file, because it looks valid until it is parsed."""
```

A corrupt file is recovered from rather than fatal: `load_state` logs and
returns a fresh structure. Nothing here feeds the signal, so that is safe.

## `live/store.py` — 393 lines

The run log: a SQLite database of what the strategy decided, traded and held.

**Five tables in two shapes.**

| Table | Shape | One row per |
|---|---|---|
| `trade_events` | **append-only** | thing that happened — submissions, fills, cancellations, guard trips, skips |
| `selection_events` | upsert | (date, symbol, event): `entry` / `exit` / `hold` with rank and 12-1 momentum |
| `position_closes` | upsert | (date, symbol): shares, close price, market value, unrealised P&L, target vs actual weight |
| `portfolio_nav` | upsert | date: book value, NetLiquidation, cash, risk weight, the vol scalar |
| `signal_runs` | upsert | (date, phase): what the signal said, kept for preflight *and* trade |

`trade_events` never deduplicates because a phase that ran twice really did
submit twice, and the log should say so. The daily snapshots upsert on their
natural key because re-running reconcile for a session must correct the row
rather than double it.

```python
@contextmanager
def connect(path: str):
    """WAL plus a busy timeout: the phases never overlap by design, but a manual
    sqlite3 session left open while a timer fires should block briefly rather
    than fail the run."""
```

Schema versioning refuses to run backwards:

```python
elif version > SCHEMA_VERSION:
    raise RuntimeError(
        f"run log is schema v{version} but this code understands v{SCHEMA_VERSION}. "
        "Upgrade the code rather than downgrading the database.")
```

**Why `hold` rows and not just entries and exits.** The boring rows are the ones
you want later: when a position turns out badly, the question is how close it was
to being dropped, and that is only answerable if the rank was recorded on every
day it survived. Those ranks come from `held_ranks` on the strategy itself, not
from re-ranking afterwards.

Reads: `recent_trades`, `trades_on`, `selection_history`, `nav_history`,
`closes_on`, `holding_periods`, `summary`, and `to_frame(db, table)` which hands
any table to pandas. `to_frame` allowlists the table name rather than
interpolating it.

As with `state.py`, **nothing here feeds the signal**. Delete the database and
the next run trades exactly the same book.

## `live/runner.py` — 533 lines

The entry point. Three scheduled phases plus two read-only ones.

```bash
python -m qbs.live.runner {preflight|trade|reconcile|signal|report}
```

| Phase | When | Does |
|---|---|---|
| `preflight` | 08:50 ET | data + signal + broker + order list. **Sends nothing.** |
| `trade` | 15:30 ET | rank, build, submit MOC before the 15:45 cutoff |
| `reconcile` | 16:15 ET | fills, end-of-day marks, cancel stragglers |
| `signal` | any time | print the target book; no broker, `--offline` available |
| `report` | any time | print the run log; no broker, no network |

Each phase is a separate process. Nothing is carried in memory between them and
nothing in the state file feeds the signal, so a phase that fails is a phase you
can simply re-run.

**Exit codes** — `0` fine (including "market closed"), `1` error, `2` config or
connection, `3` a guard refused the list. A non-zero exit fails the systemd unit,
which is what surfaces the problem.

Two time guards:

```python
def past_moc_cutoff(tz: str, cutoff_hhmm: str) -> bool:
    """IB stops accepting MOC a few minutes before 16:00. Submitting after that
    does not half-work -- the orders are rejected, and the book silently stays
    where it was while the log claims a successful run."""
```

checked *immediately before sending*, not at the top of the phase, because the
download and ranking take real time and it is the submission that must beat the
cutoff. And:

```python
if not _bar_is_today(book, live.market_tz) and not force:
    log.warning("last bar is %s, not today -- market closed or data late.")
    return EXIT_OK
```

"Is the market open" is inferred from whether the feed printed a bar today —
simple and robust, though it cannot distinguish a holiday from a data outage.

Note the ordering inside `phase_trade`: the selection is logged **before**
anything is sent, because what the strategy decided stays true whether or not the
orders make it out. And a refused session is itself a logged event, so the log
never shows a day that silently did nothing:

```python
except GuardTripped as exc:
    store.log_note(live.db_path, session, "trade", "guard_tripped", str(exc))
    store.log_signal_run(live.db_path, session, "trade", book, "guard", str(exc))
    return EXIT_GUARD
```

---

# Part 4 — deployment (`deploy/`)

```
deploy/
  install.sh              provisions a fresh box; starts nothing
  README.md               the runbook
  live.example.json       config template -> /etc/qbs/live.json
  live.env.example        environment template -> /etc/qbs/live.env
  systemd/
    qbs-preflight.{service,timer}
    qbs-trade.{service,timer}
    qbs-reconcile.{service,timer}
```

The timers are anchored to `America/New_York`, so US daylight saving moves them
automatically. Two settings carry real weight:

```ini
# qbs-trade.timer
Persistent=false
```

With `Persistent=true`, an instance that booted late — or a maintenance reboot at
16:30 — would fire immediately and submit MOC orders against a signal for an
auction that has already happened. A missed session must stay missed.

```ini
# qbs-trade.service
Restart=no
```

A retry would recompute and resubmit against a book the first attempt may already
have moved.

`install.sh` adds a **2 GB swapfile**: a t3.small has 2 GB, IB Gateway's JVM takes
most of 1 GB, and the trade job holds a ~100-ticker × 1500-row frame. Without
swap the OOM killer eventually picks the Python process mid-session, silently.
It also checks Python ≥ 3.10 (an `ib_async` requirement) and installs `tzdata`,
without which systemd falls back to UTC and every job fires at the wrong hour.

---

# Part 5 — tests

`tests/test_qbs.py` (636 lines, 52 tests) and `tests/test_live.py` (773 lines,
62 tests). Everything runs without IB Gateway, without network, and without a
clock.

Two categories, per the header of the first file: indicator correctness against
values verifiable by hand, and **structural invariants a subtle refactor could
break silently** — look-ahead, weight leakage, cost accounting.

The three that matter most:

```python
def test_live_signal_matches_the_backtest_final_weights():
    """If this ever fails, the live book has silently forked from the strategy
    the backtest reports on -- the single worst failure mode this deployment has."""
```

```python
def test_book_vol_target_has_no_look_ahead():
    """The overlay reads the book's realised returns, so a lag mistake here is
    invisible in the equity curve but would still be look-ahead. Perturbing the
    tail of the price history and checking that earlier weights are
    bit-identical catches it."""
```

```python
def test_selection_ranks_are_the_strategy_own_ranks():
    """Holds carry the rank the strategy ranked them at, not a re-derived one."""
```

The broker is covered only where it has logic of its own — the live-port refusal,
non-stock filtering, the dry-run path — against a stub `IB`. The parts that are a
straight translation into library calls are not mocked, because such a test
asserts nothing beyond "the mock was called".

---

# Appendix A — dependency map

Nothing imports upward; there are no cycles.

```
config.py ──────────────┬──────────────────┬─────────────┐
   ▲                    │                  │             │
   │              indicators.py        data.py     universe.py
   │                    │                  │             │
   └──── strategies.py ─┘                  │             │
              │                            │             │
         engine.py                         │             │
              │                            │             │
         metrics.py                        │             │
              │                            │             │
        plotting.py                        │             │
              │                            │             │
         pipeline.py ◀─────────────────────┴─────────────┘
              │
       run_backtest.py

live/orders.py     (pure -- imports nothing from qbs)
live/state.py      (pure -- stdlib only)
live/store.py      (stdlib; pandas optional, only in to_frame)
live/config.py     ──▶ qbs.config
live/signals.py    ──▶ qbs.config, qbs.strategies, qbs.data, qbs.universe
live/broker.py     ──▶ live.config, live.orders, ib_async
live/runner.py     ──▶ everything above
```

The three purest modules — `orders.py`, `state.py`, `store.py` — are the ones
carrying the most safety-critical logic, and that is not a coincidence.

# Appendix B — extending it

**A new strategy.** Write a function returning a `StrategySignals` and add it to
`pipeline.build_signals`. Everything downstream — engine, metrics, charts, the
summary table — works on it unchanged. Add a palette entry and a label in
`config.py` so the charts can name it.

**A new safety guard.** Add it to `orders.apply_guards`. It is pure, so the test
is three lines.

**A new logged field.** Add the column to the relevant `CREATE TABLE` in
`store.SCHEMA` and bump `SCHEMA_VERSION`. Every statement is
`CREATE ... IF NOT EXISTS`, so an older database picks up new tables on the next
open; new *columns* on an existing table need an explicit `ALTER`.

**Point-in-time membership** — the single highest-value improvement available:

```python
from qbs.universe import load_pit_universe
from qbs.pipeline import run

pit = load_pit_universe("my_ndx_membership.csv")   # date,ticker rows
lab = run(cfg, pit_membership=pit)
```

That removes the survivorship bias that currently flatters the momentum strategy
specifically, and it is the largest single caveat in the package.
