"""Resistance-breakout trading on hourly bars, from M6_finalnotebook.ipynb.

This is the third notebook rolled into the package, and the first one that
does not fit the daily weight-vector model the rest of the lab uses. Two
things make it different, and both are structural rather than cosmetic:

* **It is hourly.** The entry is an hourly close crossing a resistance level,
  confirmed exactly `confirm_hours` bars later. There is no daily equivalent:
  collapse it to closes and the confirmation window disappears entirely.
* **It is event-driven.** A position has its own entry price, its own stop at
  entry - R, and its own exit conditions. `engine.run_backtest` multiplies a
  weight by a close-to-close return, which cannot express "filled at 102.40
  on the 11:30 bar, stopped at 98.10 four sessions later".

So this module does its own accounting, from actual fills, and hands back a
`BacktestResult` that `metrics.summarise` reads like any other. The costs and
the safe asset are the lab's, so the summary line is comparable; the fill
model is the one thing that differs, and it differs because it has to.

What was changed when porting, and why
--------------------------------------
The notebook's `generate_trades` has three look-ahead paths and one ordering
bug. All four are fixed here; the three look-ahead fixes are switchable via
`BreakoutParams` so the cost of each can be measured rather than asserted.

1. **Levels from the whole history.** `estimate_sr_levels` runs `find_peaks`
   over the entire daily frame once, and `generate_trades` then replays the
   same history against those levels. A resistance level is by construction a
   price the stock turned at -- so trades are placed at levels defined by
   turns that had not happened yet. Measured on a 2.5-year sample, a trade a
   quarter of the way in draws roughly three quarters of its levels from its
   own future. `causal_levels=True` recomputes levels at each weekly
   selection date from data up to that date only.

2. **Var95 from the whole history.** Same shape, smaller effect: R is sized
   from a percentile of returns the trade has not seen yet.
   `causal_risk=True` uses an expanding window.

3. **Daily indicators forward-filled without a lag.** The notebook resamples
   hourly to daily and reindexes onto the hourly index with `ffill`. Daily
   rows are stamped at midnight, so every hourly bar of day D receives the
   EMA, ATR and ADR computed from day D's close -- including the 09:30 bar.
   Both the regime gate and the EMA exit therefore see the day's outcome all
   day. `lag_daily_indicators=True` shifts one day.

4. **A position could exit before it entered.** The notebook detects a cross
   at bar `i`, looks forward to bar `i+2` to confirm, and immediately writes
   the trade into `active_by_level` -- while the loop is still at `i`. The
   exit block then runs at bars `i+1` and `i+2` and can close the position on
   a bar that precedes its own `entry_time`. Here a confirmed cross becomes a
   *pending* entry and is only exitable from its entry bar onward.

Two parameters in the notebook are inert and are not reproduced:
`rr_takeprofit` (a 2R target that is computed and never read) and
`require_retest`. Where the notebook's prose and its code disagree -- the
prose says "10-day sma", the code uses the fast EMA minus one ADR -- the code
is what is implemented, because the code is what produced its numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BreakoutParams, SAFE_ASSET, TRADING_DAYS, WeeklyBookParams
from .engine import BacktestResult
from .indicators import drawdown

OHLC = ("Open", "High", "Low", "Close")


# ==========================================================================
# Bars and indicators
# ==========================================================================

def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly bars to daily OHLCV, the notebook's resample."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in hourly.columns:
        agg["Volume"] = "sum"
    return hourly.resample("1D").agg(agg).dropna(subset=["Close"])


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range -- the notebook's definition, a simple mean of TR."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    return tr.rolling(n, min_periods=1).mean()


def daily_indicators(daily: pd.DataFrame, p: BreakoutParams) -> pd.DataFrame:
    """EMA fast/slow, ATR and ADR on daily bars.

    With `lag_daily_indicators` the frame is shifted one row, so a value
    stamped on day D was computable from day D-1's close. That is what makes
    it legal to consult at 09:30 on day D.
    """
    out = pd.DataFrame(index=daily.index)
    out["EMA_F"] = daily["Close"].ewm(span=p.ema_fast_days, adjust=False,
                                      min_periods=p.ema_fast_days).mean()
    out["EMA_S"] = daily["Close"].ewm(span=p.ema_slow_days, adjust=False,
                                      min_periods=p.ema_slow_days).mean()
    out["ATR"] = atr(daily, n=p.atr_window)
    out["ADR"] = (daily["High"] - daily["Low"]).rolling(p.adr_window).mean()
    return out.shift(1) if p.lag_daily_indicators else out


def align_indicators(hourly_index: pd.DatetimeIndex,
                     ind: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill daily indicator rows onto an hourly index."""
    return ind.reindex(ind.index.union(hourly_index)).ffill().reindex(hourly_index)


# ==========================================================================
# Support / resistance levels
# ==========================================================================

def _merge_once(levels: List[Tuple], price_tol: float) -> List[Tuple]:
    """The notebook's first merge: adjacent same-kind levels within a tolerance."""
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: (x[1], x[0]))
    merged, cur_times, cur_price, cur_kind, touches = [], [levels[0][0]], levels[0][1], levels[0][2], 1
    for t, price, kind in levels[1:]:
        if abs(price - cur_price) <= price_tol and kind == cur_kind:
            cur_price = (cur_price * touches + price) / (touches + 1)
            touches += 1
            cur_times.append(t)
        else:
            merged.append((cur_times, cur_price, cur_kind, touches))
            cur_times, cur_price, cur_kind, touches = [t], price, kind, 1
    merged.append((cur_times, cur_price, cur_kind, touches))
    return merged


def _merge_repeatedly(levels: List[Tuple], price_tol: float) -> List[Tuple]:
    """The notebook's second pass, run to a fixed point.

    Its version is an O(n^2) loop with a `pop` inside, re-scanning until a
    sweep merges nothing. Same result, expressed as a single ordered sweep
    repeated until stable -- the quadratic rescan was doing no extra work.
    """
    out = sorted(levels, key=lambda x: x[1])
    while True:
        merged, i, did = [], 0, False
        while i < len(out):
            times, price, _kind, touches = out[i]
            j = i + 1
            while j < len(out) and abs(out[j][1] - price) < price_tol:
                t2, p2, _k2, n2 = out[j]
                total = len(times) + len(t2)
                price = (price * len(times) + p2 * len(t2)) / total
                times = times + t2
                touches += n2
                j += 1
                did = True
            merged.append((times, price, "merged", touches))
            i = j
        out = merged
        if not did:
            return out


def sr_levels(daily: pd.DataFrame, p: BreakoutParams) -> List[float]:
    """Resistance/support levels from daily swing points, merged by ATR.

    `find_peaks` on highs gives resistance, on inverted lows gives support.
    Both merge passes of the notebook are applied, then the `max_levels`
    most-touched survive. Returns bare prices, which is all the trade
    generator consumes.
    """
    from scipy.signal import find_peaks

    if len(daily) < max(p.atr_window, 5):
        return []

    close, high, low = daily["Close"], daily["High"], daily["Low"]
    prom = max(1e-6, p.prominence_frac * float(close.iloc[-1]))

    peaks, _ = find_peaks(high.to_numpy(), distance=p.swing_lookback, prominence=prom)
    troughs, _ = find_peaks((-low).to_numpy(), distance=p.swing_lookback, prominence=prom)

    raw: List[Tuple] = [(daily.index[i], float(high.iloc[i]), "resistance") for i in peaks]
    raw += [(daily.index[i], float(low.iloc[i]), "support") for i in troughs]
    if not raw:
        return []

    avg_atr = float(atr(daily, p.atr_window).mean())
    merged = _merge_once(raw, price_tol=max(1e-6, p.atr_merge_mult * avg_atr))
    merged = _merge_repeatedly(merged, price_tol=max(1e-6, p.second_merge_mult * avg_atr))

    merged.sort(key=lambda x: x[3], reverse=True)
    return sorted(float(x[1]) for x in merged[:p.max_levels])


def var_risk(daily: pd.DataFrame, p: BreakoutParams) -> float:
    """|Var95| of daily returns -- the notebook's cap on the risk unit R."""
    rets = daily["Close"].pct_change().dropna()
    if rets.empty:
        return float("inf")
    return abs(float(np.percentile(rets, (1.0 - p.var_confidence) * 100.0)))


# ==========================================================================
# Trades
# ==========================================================================

@dataclass
class Trade:
    ticker: str
    entry_time: pd.Timestamp
    entry_price: float
    level: float
    R: float
    ADR: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def gross_return(self) -> float:
        if self.exit_price is None or self.entry_price <= 0:
            return 0.0
        return self.exit_price / self.entry_price - 1.0


def breakout_signals(
    hourly: pd.DataFrame,
    levels: Sequence[float],
    var95: float,
    p: BreakoutParams,
    ticker: str = "",
    ind_hourly: Optional[pd.DataFrame] = None,
    entry_end: Optional[pd.Timestamp] = None,
) -> List[Trade]:
    """The notebook's entry and exit rules over one name's hourly bars.

    Entry
        the regime is up (fast EMA above slow EMA), an hourly close crosses
        above a level, and `confirm_hours` bars later the regime is still up
        and the close is still above the level. Filled at that bar's close.

    Exit, evaluated on every bar from the entry bar onward
        close <= entry - R                                      -> "stop_R"
        close >= entry + R  and close < EMA_F - ADR             -> "take_profit"
        otherwise, held past `hold_weeks` or close < EMA_F - ADR
                                                                -> "time_or_ema"

    One position per level at a time, which is the notebook's rule: a level
    that has been broken and traded is not re-entered until that trade closes.

    `entry_end` splits the two windows the weekly book needs: new entries are
    only looked for up to that bar, but open positions keep being managed over
    every bar supplied. Without the split a trade opened on Thursday would be
    abandoned on Friday, and its `hold_weeks` exit could never fire.
    """
    if hourly.empty or not len(levels):
        return []

    df = hourly.sort_index()
    if ind_hourly is None:
        ind_hourly = align_indicators(df.index, daily_indicators(hourly_to_daily(df), p))
    ind_hourly = ind_hourly.reindex(df.index)

    close = df["Close"].to_numpy(dtype=float)
    ema_f = ind_hourly["EMA_F"].to_numpy(dtype=float)
    ema_s = ind_hourly["EMA_S"].to_numpy(dtype=float)
    adr = ind_hourly["ADR"].to_numpy(dtype=float)
    index = df.index

    lvls = sorted(float(x) for x in levels)
    gap = float(np.mean(np.diff(lvls))) if len(lvls) > 1 else float("inf")
    hold_delta = pd.Timedelta(weeks=p.hold_weeks)

    active: Dict[float, Optional[Trade]] = {L: None for L in lvls}
    pending: Dict[int, List[Tuple[float, float]]] = {}   # bar -> [(level, ADR at cross)]
    trades: List[Trade] = []

    for i in range(1, len(df)):
        now = index[i]
        c_now, c_prev = close[i], close[i - 1]
        f, s, a = ema_f[i], ema_s[i], adr[i]

        # ---- exits, on positions that are actually open --------------------
        for L, pos in active.items():
            if pos is None or pos.entry_time > now:
                continue
            stop_px = pos.entry_price - pos.R
            one_r = pos.entry_price + pos.R
            trail = f - pos.ADR if not np.isnan(f) else -np.inf

            if c_now <= stop_px:
                reason = "stop_R"
            elif c_now >= one_r:
                reason = "take_profit" if c_now < trail else None
            elif now > pos.entry_time + hold_delta or c_now < trail:
                reason = "time_or_ema"
            else:
                reason = None

            if reason is not None:
                pos.exit_time, pos.exit_price, pos.reason = now, float(c_now), reason
                trades.append(pos)
                active[L] = None

        # ---- confirmations scheduled for this bar --------------------------
        for L, adr_at_cross in pending.pop(i, []):
            if active[L] is not None:
                continue
            if np.isnan(f) or np.isnan(s) or f <= s or c_now <= L:
                continue            # regime broke, or price fell back below
            structural = gap / 2.0 + (adr_at_cross * 0.5 if not np.isnan(adr_at_cross) else 0.0)
            r_val = min(abs(c_now * var95), structural) if np.isfinite(var95) else structural
            if not np.isfinite(r_val) or r_val <= 0:
                continue
            active[L] = Trade(ticker=ticker, entry_time=now, entry_price=float(c_now),
                              level=L, R=float(r_val),
                              ADR=float(adr_at_cross) if not np.isnan(adr_at_cross) else 0.0)

        # ---- new crosses ---------------------------------------------------
        if entry_end is not None and now > entry_end:
            continue
        if np.isnan(f) or np.isnan(s) or f <= s:
            continue
        k = i + p.confirm_hours
        if k >= len(df):
            continue
        for L in lvls:
            if active[L] is None and c_prev <= L < c_now:
                pending.setdefault(k, []).append((L, a))

    trades.extend(t for t in active.values() if t is not None)
    return trades


# ==========================================================================
# The weekly six-slot book
# ==========================================================================

@dataclass
class BreakoutBook:
    """What the scenario did, at a level you can audit trade by trade."""
    result: BacktestResult
    trades: pd.DataFrame
    watchlists: Dict[pd.Timestamp, List[str]] = field(default_factory=dict)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def hit_rate(self) -> float:
        closed = self.trades[self.trades["exit_time"].notna()]
        return float((closed["gross_return"] > 0).mean()) if len(closed) else float("nan")


def weekly_breakout_book(
    hourly: Dict[str, pd.DataFrame],
    watchlists: Dict[pd.Timestamp, List[str]],
    safe_prices: pd.Series,
    p: Optional[BreakoutParams] = None,
    book: Optional[WeeklyBookParams] = None,
    cost_bps: float = 1.0,
    slippage_bps: float = 5.0,
    name: str = "breakout",
) -> BreakoutBook:
    """Trade a weekend watchlist by breakout, `n_slots` positions at a time.

    Parameters
    ----------
    hourly      : ticker -> hourly OHLC(V) frame. Only names that appear on
                  some watchlist are ever consulted.
    watchlists  : selection date -> ranked tickers eligible for the following
                  week. Build it with `finviz_watchlists`.
    safe_prices : daily closes of the safe asset. Idle slots earn this.

    The rule that shapes the result
    -------------------------------
    With `refill_within_week=False` a slot freed mid-week stays in cash until
    the next selection date. So the book cannot chase: a stop-out on Tuesday
    is a slot parked in BOXX until Friday, however many watchlist names break
    out on Wednesday. That is the scenario as specified, and it is the single
    biggest driver of average exposure.
    """
    p = p or BreakoutParams()
    book = book or WeeklyBookParams()

    sel_dates = sorted(watchlists)
    if not sel_dates:
        raise ValueError("no watchlists -- nothing can ever be traded")

    # ---- per-name trade candidates, generated once per name ---------------
    # Levels and risk are recomputed at each selection date the name appears
    # on, so a trade opened in week W is only ever placed at a level that
    # week W's chart could show.
    all_trades: List[Trade] = []
    for ticker in sorted({t for names in watchlists.values() for t in names}):
        bars = hourly.get(ticker)
        if bars is None or bars.empty:
            continue
        bars = bars.sort_index()
        daily = hourly_to_daily(bars)
        ind = align_indicators(bars.index, daily_indicators(daily, p))

        if not p.causal_levels and not p.causal_risk:
            windows = [(sel_dates[0], None, daily)]
        else:
            windows = []
            for k, sd in enumerate(sel_dates):
                if ticker not in watchlists[sd]:
                    continue
                end = sel_dates[k + 1] if k + 1 < len(sel_dates) else None
                hist = daily.loc[:sd] if p.causal_levels or p.causal_risk else daily
                windows.append((sd, end, hist))

        for sd, end, hist in windows:
            if hist.empty:
                continue
            levels = sr_levels(hist if p.causal_levels else daily, p)
            if not levels:
                continue
            v95 = var_risk(hist if p.causal_risk else daily, p)
            seg = bars.loc[sd:]
            if len(seg) < 3:
                continue
            # Entries only inside this week; exits managed over everything
            # after it, because a trade may run for `hold_weeks`. Indicators
            # come from the full series so the EMA is already warm.
            all_trades += breakout_signals(seg, levels, v95, p, ticker=ticker,
                                           ind_hourly=ind.reindex(seg.index),
                                           entry_end=end)

    # ---- allocate trades to slots, in time order --------------------------
    all_trades.sort(key=lambda t: (t.entry_time, t.ticker))
    rank_of = {sd: {t: i for i, t in enumerate(names)}
               for sd, names in watchlists.items()}

    def week_of(ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        prior = [d for d in sel_dates if d <= ts]
        return prior[-1] if prior else None

    open_slots: List[Trade] = []
    taken: List[Trade] = []
    freed_this_week = 0
    current_week: Optional[pd.Timestamp] = None

    for tr in all_trades:
        wk = week_of(tr.entry_time)
        if wk is None or tr.ticker not in rank_of.get(wk, {}):
            continue                                   # not on that week's list
        if wk != current_week:
            current_week, freed_this_week = wk, 0
        # Retire anything that closed before this entry.
        still_open = [x for x in open_slots if x.is_open or x.exit_time > tr.entry_time]
        freed_this_week += len(open_slots) - len(still_open)
        open_slots = still_open
        if any(x.ticker == tr.ticker for x in open_slots):
            continue                                   # already holding the name
        used = len(open_slots) + (0 if book.refill_within_week else freed_this_week)
        if used >= book.n_slots:
            continue
        open_slots.append(tr)
        taken.append(tr)

    return _account(taken, watchlists, safe_prices, hourly, p, book,
                    cost_bps, slippage_bps, name)


def _account(
    trades: List[Trade],
    watchlists: Dict[pd.Timestamp, List[str]],
    safe_prices: pd.Series,
    hourly: Dict[str, pd.DataFrame],
    p: BreakoutParams,
    book: WeeklyBookParams,
    cost_bps: float,
    slippage_bps: float,
    name: str,
) -> BreakoutBook:
    """Daily equity from the fills, with idle slots earning the safe asset.

    Each slot is `1 / n_slots` of the book. A slot holding a trade earns that
    trade's daily path; an empty slot earns the safe asset. Costs are charged
    on the two fills of each trade, at the slot's weight, so turnover is
    comparable with the weight-based strategies rather than being free.
    """
    safe = safe_prices.sort_index()
    safe.index = pd.to_datetime(safe.index).tz_localize(None).normalize()
    # The strategy does not exist before its first watchlist, so it is not
    # priced there either -- otherwise months of pure safe-asset return get
    # annualised into the result as if the book had been running.
    first = min(watchlists).tz_localize(None).normalize()
    safe = safe.loc[safe.index >= first]
    if safe.empty:
        raise ValueError("safe_prices does not cover the watchlist window")
    days = safe.index
    slot_w = 1.0 / book.n_slots

    tickers = sorted({t.ticker for t in trades})
    daily_close = {}
    for t in tickers:
        d = hourly_to_daily(hourly[t].sort_index())["Close"]
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        daily_close[t] = d.reindex(days).ffill()

    safe_ret = safe.pct_change().fillna(0.0)
    gross = pd.Series(0.0, index=days)
    costs = pd.Series(0.0, index=days)
    turnover = pd.Series(0.0, index=days)
    weights = pd.DataFrame(0.0, index=days, columns=tickers + [book.safe_asset])
    in_use = pd.Series(0, index=days, dtype=int)

    fee = (cost_bps + slippage_bps) / 1e4
    rows = []
    for tr in trades:
        entry_day = pd.Timestamp(tr.entry_time).tz_localize(None).normalize()
        exit_day = (pd.Timestamp(tr.exit_time).tz_localize(None).normalize()
                    if tr.exit_time is not None else days[-1])
        span = days[(days >= entry_day) & (days <= exit_day)]
        if len(span) == 0:
            continue
        px = daily_close[tr.ticker]
        # Day 1 is entry-fill to that day's close; the last day is the close
        # before the exit to the exit fill; the middle is close to close.
        path = px.reindex(span).ffill()
        legs = path.pct_change()
        legs.iloc[0] = path.iloc[0] / tr.entry_price - 1.0
        if tr.exit_price is not None and len(span) > 0:
            prev = path.iloc[-2] if len(span) > 1 else tr.entry_price
            legs.iloc[-1] = tr.exit_price / prev - 1.0
        legs = legs.fillna(0.0)

        gross.loc[span] += slot_w * legs
        weights.loc[span, tr.ticker] += slot_w
        in_use.loc[span] += 1
        costs.loc[span[0]] += slot_w * fee
        turnover.loc[span[0]] += slot_w
        if tr.exit_time is not None:
            costs.loc[span[-1]] += slot_w * fee
            turnover.loc[span[-1]] += slot_w

        rows.append(dict(ticker=tr.ticker, entry_time=tr.entry_time,
                         entry_price=tr.entry_price, exit_time=tr.exit_time,
                         exit_price=tr.exit_price, reason=tr.reason,
                         level=tr.level, R=tr.R,
                         gross_return=tr.gross_return))

    idle = (book.n_slots - in_use).clip(lower=0) * slot_w
    weights[book.safe_asset] = idle
    gross += idle * safe_ret

    net = gross - costs
    equity = (1.0 + net).cumprod()

    res = BacktestResult(
        name=name, equity=equity, returns=net, gross_returns=gross,
        weights=weights, turnover=turnover, costs=costs,
        drawdown=drawdown(equity), signals=None,
        commission=turnover * (cost_bps / 1e4),
        slippage=turnover * (slippage_bps / 1e4),
    )
    diag = pd.DataFrame({"slots_in_use": in_use,
                         "cash_slots": book.n_slots - in_use,
                         "exposure": in_use * slot_w})
    tdf = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker", "entry_time", "entry_price", "exit_time", "exit_price",
                 "reason", "level", "R", "gross_return"])
    return BreakoutBook(result=res, trades=tdf, watchlists=watchlists, diagnostics=diag)


# ==========================================================================
# Feeding it from the Finviz screen
# ==========================================================================

def finviz_watchlists(
    universe_prices: pd.DataFrame,
    safe_prices: pd.Series,
    n_watch: int = 20,
    params=None,
    volumes: Optional[pd.DataFrame] = None,
    selection_day: str = "W-FRI",
) -> Dict[pd.Timestamp, List[str]]:
    """Weekend Finviz selection: date -> the week's ranked watchlist.

    Runs `screens.finviz_momentum_screen` with `n_hold=0` so it returns every
    name that passed, in RS order, then keeps the top `n_watch`. The screen's
    own rebalance is set to `selection_day`, so the list is rebuilt on the last
    session of each week and stands for the week that follows.
    """
    from .config import FinvizScreenParams
    from .screens import finviz_momentum_screen

    base = params or FinvizScreenParams()
    p = FinvizScreenParams(**{**base.__dict__, "n_hold": 0, "exit_rank": 0,
                             "rebalance": selection_day})
    sig = finviz_momentum_screen(universe_prices, safe_prices, p, volumes=volumes)

    marks = universe_prices.index.to_series().resample(selection_day).last().dropna()
    out: Dict[pd.Timestamp, List[str]] = {}
    for d in marks:
        if d in sig.holdings_log:
            names = sig.holdings_log[d][:n_watch]
            if names:
                out[pd.Timestamp(d)] = list(names)
    return out


# ==========================================================================
# Hourly bars from yfinance
# ==========================================================================

HOURLY_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hourly")


def load_hourly(
    tickers: Sequence[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: str = HOURLY_CACHE,
    refresh: bool = False,
    offline: bool = False,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Hourly OHLCV per ticker, cached one CSV per name.

    Separate from `data.load_prices` because that loader is deliberately
    closes-only and one wide frame; this needs High and Low per name, which is
    a different shape and a much bigger cache.

    **yfinance serves at most ~730 days of hourly history**, and it is not
    back-fillable: whatever is not cached before it ages out is gone. So the
    cache is additive -- an existing file is extended rather than replaced, and
    a name whose download fails is reported and skipped rather than silently
    becoming an empty frame the trade generator would read as "no bars".
    """
    os.makedirs(cache_dir, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for t in sorted(set(tickers)):
        path = os.path.join(cache_dir, f"{t}.csv")
        cached = None
        if os.path.exists(path) and not refresh:
            cached = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
        if offline:
            if cached is None or cached.empty:
                failed.append(t)
            else:
                out[t] = cached.sort_index()
            continue

        try:
            import yfinance as yf
            raw = yf.download(t, start=start, end=end, interval="1h",
                              auto_adjust=True, progress=False, actions=False)
            if raw is None or raw.empty:
                raise RuntimeError("no rows returned")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw = raw[[c for c in ("Open", "High", "Low", "Close", "Volume")
                       if c in raw.columns]].dropna()
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            raw.index.name = "Datetime"
            fresh = raw if cached is None else pd.concat([cached, raw])
            fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
            fresh.to_csv(path)
            out[t] = fresh
        except Exception as exc:  # noqa: BLE001
            if cached is not None and not cached.empty:
                out[t] = cached.sort_index()
                if verbose:
                    print(f"[hourly] {t}: download failed ({exc}); using cache")
            else:
                failed.append(t)

    if verbose:
        if out:
            n = sum(len(v) for v in out.values())
            print(f"[hourly] {len(out)} tickers, {n:,} bars")
        if failed:
            print(f"[hourly] no data for {len(failed)}: {sorted(failed)}")
    return out


# ==========================================================================
# Synthetic hourly bars, for testing without a data feed
# ==========================================================================

def synthetic_hourly(
    tickers: Sequence[str],
    start: str = "2025-01-02",
    end: str = "2026-09-01",
    bars_per_day: int = 7,
    seed: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Hourly OHLCV with genuine trends, so breakouts are a real event.

    Random walks would make this test nothing: a breakout strategy on pure
    noise has no signal to find, and a version of it that "worked" would be a
    bug. Each name gets a slow-moving drift, the same device
    `universe.synthetic_universe` uses for the cross-sectional tests.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, end=end)
    stamps = pd.DatetimeIndex([
        d + pd.Timedelta(hours=10 + k) for d in sessions for k in range(bars_per_day)
    ])
    n = len(stamps)

    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        drift = np.zeros(n)
        drift[0] = rng.normal(0.10, 0.25)
        for i in range(1, n):
            drift[i] = 0.999 * drift[i - 1] + rng.normal(0, 0.004)
        step = drift / (252 * bars_per_day) + rng.normal(0, 0.004, n)
        close = 100 * np.exp(np.cumsum(step))
        spread = np.abs(rng.normal(0, 0.003, n)) * close
        open_ = np.concatenate([[close[0]], close[:-1]])
        out[t] = pd.DataFrame(
            {"Open": open_,
             "High": np.maximum(open_, close) + spread,
             "Low": np.minimum(open_, close) - spread,
             "Close": close,
             "Volume": rng.integers(1e5, 5e6, n)},
            index=stamps,
        )
    return out
