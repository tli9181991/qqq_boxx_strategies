# Screen fidelity: `M6_finalnotebook.ipynb` → `qbs/screens.py`

A line-by-line map so the implementation can be checked against the notebook
rather than trusted. Notebook references are to `evaluate_ticker` (cell 9)
unless noted.

## The pass criteria

The notebook's verdict is this list, with `base_ok` commented out by its author:

```python
passed = all([
    stage2_ok,
    # base_ok,
    rs_ok,
    sector_outperforms,
    above_200_ok,
    within_high_ok
])
```

| # | Notebook | `screens.py` | Status |
|---|---|---|---|
| 1 | `stage2_ok` | `stage2` | **exact** |
| 2 | `rs_ok` | `rs_ok` | **exact** |
| 3 | `sector_outperforms` | `sector_beats` | **optional, off by default** |
| 4 | `above_200_ok` | `above_200` | **exact** |
| 5 | `within_high_ok` | `within_high` | **exact** |
| — | `base_ok` | not implemented | disabled in the notebook too |

### 1. Stage 2 (Weinstein proxy)

```python
# notebook
stage2_ok = (
    last_close > sma150.iloc[-1] and
    last_close > sma200.iloc[-1] and
    slope(sma150, 20) > 0 and
    slope(sma200, 20) > 0
)
```
```python
# screens.py
stage2 = (
    (px > sma150) & (px > sma200)
    & (rolling_log_slope(sma150, p.stage2_slope_window) > 0)
    & (rolling_log_slope(sma200, p.stage2_slope_window) > 0)
)
```

Same four conditions. `stage2_slope_window` defaults to 20, the notebook's value.

### 2. Relative strength — both halves

```python
# notebook
rs6m_stock  = pct_change_over(close, 126)      # last/first - 1 over the window
rs6m_mkt    = pct_change_over(market, 126)
rs6m_vs_mkt = rs6m_stock - rs6m_mkt
ratio         = (close / market).dropna()
rs_slope_100d = slope(ratio, 100)
rs_ok = (rs6m_vs_mkt > 0) and (rs_slope_100d > 0)
```
```python
# screens.py
rs6m_stock = px / px.shift(p.rs_lookback) - 1.0
rs6m_mkt   = mkt / mkt.shift(p.rs_lookback) - 1.0
beats_market = rs6m_stock.sub(rs6m_mkt, axis=0) > 0
rs_line   = px.div(mkt, axis=0)
rs_rising = rolling_log_slope(rs_line, p.rs_slope_window) > 0
rs_ok = beats_market & rs_rising
```

`pct_change_over(s, w)` is `s.iloc[-1] / s.iloc[-w-1] - 1`, which is `shift(w)`
on a rolling basis. **This is the only criterion the `market_benchmark` enters**,
so it is the one the `^GSPC` → QQQ switch changes.

### 3. Sector strength — the one gap

```python
# notebook
sec6m = pct_change_over(etf_close, 126)
mkt6m = pct_change_over(market, 126)
sector_outperforms = (sec6m >= mkt6m)
```
```python
# screens.py -- only when sector_map and sector_prices are supplied
sec6m = sec / sec.shift(p.rs_lookback) - 1.0
sector_beats = sec6m.ge(rs6m_mkt, axis=0)
passes &= pd.concat(cols, axis=1, keys=px.columns)
```

Needs a per-ticker sector map and sector-ETF history. Neither is in the price
cache and Yahoo is unreachable through this environment's proxy, so it is
**off by default** and `params["sector_filter_applied"]` records that.

**Direction of the bias: this makes the screen more permissive**, so the results
here flatter it rather than the reverse. A test pins that enabling the filter can
only ever remove names, never admit one.

### 4 & 5. Above the 200-day, and near the 52-week high

```python
# notebook
above_200_ok    = last_close > sma200_now
high_52w        = close.rolling(252, min_periods=252).max().iloc[-1]
within_high_pct = 1.0 - (last_close / high_52w)
within_high_ok  = within_high_pct <= 0.25
```
```python
# screens.py
above_200   = px > sma200
high_52w    = px.rolling(p.high_window, min_periods=p.high_window).max()
within_high = (1.0 - px / high_52w) <= p.within_52w_high_pct
```

Identical, including the `min_periods=252` requirement that keeps a young
listing from qualifying on a short high.

## The ranking

```python
# notebook, cell 11 + cell 18
df_pass["dailyret_rank"] = df_pass["daily_annret"].rank(ascending=True)
df_pass = df_pass.sort_values(["dailyret_rank", "rs_rank", "within_high_rank"])
sel_tickers = df_pass['ticker'].iloc[-10:].tolist()      # tail = strongest
```

`daily_annret` in the notebook is annualised total return over the whole fetched
history:

```python
cum_ret      = (1 + ret).prod() - 1
daily_annret = ((1 + cum_ret) ** (252 / len(ret))) - 1
```

```python
# screens.py -- the rolling form
total  = px / px.shift(p.annret_window) - 1.0
years  = p.annret_window / TRADING_DAYS
annret = (1.0 + total).pow(1.0 / years) - 1.0
order  = cand.sort_values(ascending=False)      # strongest first
```

**One deliberate change.** The notebook's window is "everything fetched", which
is 3 years ending today. Rolled forward naively that window would *grow* with
the backtest, so the measure would not mean the same thing on the first date as
the last. `annret_window` fixes it at 756 bars (~3 years), matching
`PARAMS["period"] = "3y"`. Sort order is reversed to descending so `[:n_hold]`
takes the strongest, which is what the notebook's `.iloc[-10:]` does.

## The slope function

```python
# notebook -- per ticker, inside a loop
def slope(series, window):
    y = np.log(s.tail(window).values)
    x = np.arange(len(y))
    b1, b0 = np.polyfit(x, y, 1)
    return float(b1)
```
```python
# screens.py -- closed form, so it can run on every date
x = np.arange(window, dtype=float); x -= x.mean()
denom = float((x ** 2).sum())
logs = np.log(frame.where(frame > 0))
return logs.rolling(window).apply(lambda v: float(v.dot(x) / denom), raw=True)
```

For evenly spaced `x`, the OLS slope is the dot product with a centred ramp over
that ramp's sum of squares. `test_rolling_log_slope_matches_polyfit` asserts the
two agree to **1e-12** at several dates and both window sizes.

## Filters dropped, and why

| Notebook | Why it is not here |
|---|---|
| `avg_vol50 >= 200_000` | needs volume; the price cache is closes only. Non-binding on a Nasdaq-100 universe — every constituent trades far above 200k shares/day. |
| `is_otc_exchange(...)` | needs exchange metadata. Non-binding — every NDX constituent is NASDAQ-listed. |
| `base_ok` (depth, SMA50 drift, vol contraction) | **the notebook has it commented out of `passed`**, so including it would deviate from the notebook, not match it. |
| `min_price >= 10.0` | **kept** — `priced = px >= p.min_price`. |

The two dropped hard filters are non-binding on this universe *by inspection of
what the universe is*, not by assumption. On a broader universe (the notebook's
S&P 500, or a small-cap list) they would bind and would need volume data.

## What changed to make it a backtest

The notebook evaluates once, on the last bar: `close.iloc[-1]`, `.tail(50)`,
`sma150.iloc[-1]`. That produces a watchlist, not a track record.

`trend_template_screen` computes the same quantities as **rolling series** and
re-evaluates on each rebalance date, so every criterion at date *t* uses only
data up to and including *t*. The resulting weights are "decided at the close of
*t*", the convention every strategy in this repo follows, and
`engine.run_backtest` applies the one-day execution lag exactly as it does for
the momentum book.

`test_screen_has_no_look_ahead` rewrites the second half of the price history and
asserts every weight before the cut is bit-identical.

## Confirming it yourself

```bash
python compare_selection.py --offline --no-fetch-universe   # the numbers
python -m pytest tests/test_qbs.py -q -k "screen or slope"   # the 9 screen tests
python compare_selection.py --offline --benchmark VEU       # benchmark sensitivity
```
