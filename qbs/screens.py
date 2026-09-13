"""Filter-based stock screens, as opposed to cross-sectional ranking.

`strategies.cross_sectional_momentum` asks "which are the strongest N names in
the universe" -- a *relative* question, always answerable, always full.
A screen asks "does this name pass, yes or no" -- an *absolute* question, which
on a bad day nothing passes. The two families behave differently enough at the
edges that it is worth keeping them apart.

`trend_template_screen` implements the Weinstein/Minervini-style trend template
from M6_finalnotebook.ipynb, rolled forward so it can be backtested rather than
evaluated once on the last bar.

`finviz_momentum_screen` does the same for finviz_filter_with_daily_summary.ipynb
-- a Finviz filter pass, then a relative-strength ranking of the survivors. Both
notebooks answer "what would I buy today"; rolling them forward is what turns
that answer into something you can hold against the Top-6 momentum book.

What was in the notebook and what is here
-----------------------------------------
The notebook screens the S&P 500 against ^GSPC on a single date. This version
screens whatever universe it is handed against whatever benchmark it is handed
-- QQQ, for a like-for-like comparison against the Nasdaq-100 momentum book --
and re-evaluates on every rebalance date.

Of the notebook's five pass criteria, four are reproduced exactly:

    stage2_ok        close > SMA150 and > SMA200, both slopes rising
    rs_ok            6m return beats the benchmark AND the relative-strength
                     line (close / benchmark) is sloping up over 100 days
    above_200_ok     close > SMA200
    within_high_ok   within 25% of the 52-week high

The fifth, `sector_outperforms` (the name's sector ETF must have beaten the
market over 6 months), needs sector-ETF price history and a per-ticker sector
map. Pass `sector_map` and `sector_prices` to enable it; without them the
criterion is skipped and `sector_filter_applied` on the result is False.
**Skipping it makes the screen strictly more permissive**, so a comparison run
without it flatters this strategy rather than the other way round.

The notebook's `base_ok` (base depth, SMA50 drift, volatility contraction) is
*not* included, because the notebook itself has it commented out of the `passed`
list. Its liquidity and OTC pre-filters are also dropped: they need volume and
exchange metadata, and on a Nasdaq-100 universe both are non-binding -- every
constituent is a NASDAQ-listed name trading far above 200k shares a day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FinvizScreenParams, SAFE_ASSET, TRADING_DAYS
from .strategies import StrategySignals, _empty_events


@dataclass
class TrendScreenParams:
    """Parameters of the notebook's screen, with its own defaults preserved."""
    rs_lookback: int = 126            # ~6 months, the notebook's rs_lookback
    rs_slope_window: int = 100        # notebook's rs_slope_window
    stage2_slope_window: int = 20     # notebook's stage2_slope_window
    within_52w_high_pct: float = 0.25
    high_window: int = 252
    min_price: float = 10.0
    annret_window: int = 756          # ~3y, the notebook's fetched "period"
    n_hold: int = 6                   # 0 -> hold every name that passes
    rebalance: str = "daily"          # "daily" | "ME" | "W-FRI"
    safe_asset: str = SAFE_ASSET
    equal_weight_slots: bool = True   # size per slot, matching the momentum book

    def __post_init__(self):
        if self.n_hold < 0:
            raise ValueError("n_hold must be >= 0 (0 means hold every passing name)")


def rolling_log_slope(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling OLS slope of log(x) against 0..window-1, per column.

    The notebook does this with `np.polyfit(x, np.log(y), 1)` inside a loop over
    tickers. For an evenly spaced x the OLS slope has a closed form -- the dot
    product with a centred ramp, divided by that ramp's sum of squares -- which
    is the same number and fast enough to run on every date rather than once.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denom = float((x ** 2).sum())
    logs = np.log(frame.where(frame > 0))
    return logs.rolling(window).apply(lambda v: float(v.dot(x) / denom), raw=True)


def trend_template_screen(
    universe_prices: pd.DataFrame,
    benchmark: pd.Series,
    safe_prices: pd.Series,
    params: Optional[TrendScreenParams] = None,
    sector_map: Optional[Dict[str, str]] = None,
    sector_prices: Optional[pd.DataFrame] = None,
    name: str = "screen",
) -> StrategySignals:
    """Roll the notebook's trend template forward and hold the names that pass.

    Parameters
    ----------
    universe_prices : date x ticker closes for the ranking universe.
    benchmark       : the market series the relative-strength tests use. The
                      notebook uses ^GSPC; pass QQQ to compare like for like
                      against a Nasdaq-100 book.
    safe_prices     : the cash leg. Unfilled slots, and every date on which
                      nothing passes, sit here.
    sector_map      : optional ticker -> sector-ETF symbol.
    sector_prices   : optional date x ETF closes. Both are needed to enable the
                      notebook's sector criterion; without them it is skipped.

    Ranking
    -------
    The notebook sorts survivors by annualised total return over its fetched
    history and takes the tail -- the strongest. That is reproduced here over a
    trailing `annret_window`, so the measure is the same shape on every date
    rather than growing with the sample.
    """
    p = params or TrendScreenParams()
    px = universe_prices.sort_index()
    mkt = benchmark.reindex(px.index).ffill()
    safe = safe_prices.reindex(px.index).ffill()

    # ---- the four reproducible criteria, all vectorised -------------------
    sma150 = px.rolling(150, min_periods=150).mean()
    sma200 = px.rolling(200, min_periods=200).mean()

    stage2 = (
        (px > sma150) & (px > sma200)
        & (rolling_log_slope(sma150, p.stage2_slope_window) > 0)
        & (rolling_log_slope(sma200, p.stage2_slope_window) > 0)
    )
    above_200 = px > sma200

    high_52w = px.rolling(p.high_window, min_periods=p.high_window).max()
    within_high = (1.0 - px / high_52w) <= p.within_52w_high_pct

    # Relative strength: beat the benchmark over 6 months AND have a rising
    # relative-strength line. The notebook requires both; the second is what
    # stops a name qualifying on one old jump it has been giving back since.
    rs6m_stock = px / px.shift(p.rs_lookback) - 1.0
    rs6m_mkt = mkt / mkt.shift(p.rs_lookback) - 1.0
    beats_market = rs6m_stock.sub(rs6m_mkt, axis=0) > 0
    rs_line = px.div(mkt, axis=0)
    rs_rising = rolling_log_slope(rs_line, p.rs_slope_window) > 0
    rs_ok = beats_market & rs_rising

    priced = px >= p.min_price

    passes = stage2 & above_200 & within_high & rs_ok & priced

    # ---- the fifth criterion, only if the caller supplied the data --------
    sector_filter_applied = bool(sector_map) and sector_prices is not None
    if sector_filter_applied:
        sec = sector_prices.reindex(px.index).ffill()
        sec6m = sec / sec.shift(p.rs_lookback) - 1.0
        sector_beats = sec6m.ge(rs6m_mkt, axis=0)
        cols = []
        for t in px.columns:
            etf = sector_map.get(t)
            cols.append(sector_beats[etf] if etf in sector_beats.columns
                        else pd.Series(False, index=px.index))
        passes &= pd.concat(cols, axis=1, keys=px.columns)

    # ---- the notebook's ranking measure -----------------------------------
    # Annualised total return over a trailing window. `annret_window` bars of
    # history compounded, then annualised -- the rolling form of the notebook's
    # (1 + cum_ret) ** (252 / len(ret)) - 1.
    total = px / px.shift(p.annret_window) - 1.0
    years = p.annret_window / TRADING_DAYS
    annret = (1.0 + total).pow(1.0 / years) - 1.0

    # ---- rebalance calendar -----------------------------------------------
    if p.rebalance == "daily":
        rebal_set = set(px.index)
    else:
        marks = px.index.to_series().resample(p.rebalance).last().dropna()
        rebal_set = {d for d in marks if d in px.index}

    assets = list(px.columns) + [p.safe_asset]
    weights = pd.DataFrame(0.0, index=px.index, columns=assets)

    held: List[str] = []
    events: List[Dict] = []
    holdings_log: Dict[pd.Timestamp, List[str]] = {}
    held_ranks: Dict[pd.Timestamp, Dict[str, float]] = {}
    n_passing: Dict[pd.Timestamp, int] = {}

    for dt in px.index:
        if dt in rebal_set:
            ok = passes.loc[dt]
            cand = annret.loc[dt][ok.fillna(False)].dropna()
            order = cand.sort_values(ascending=False)
            rank = pd.Series(np.arange(1, len(order) + 1), index=order.index)
            n_passing[dt] = int(len(order))

            # A screen has no hysteresis band: a name that stops passing is out
            # that day. That is the notebook's rule, and it is the single
            # biggest behavioural difference from the momentum book.
            target = list(order.index) if p.n_hold == 0 else list(order.index[:p.n_hold])

            for t in held:
                if t not in target:
                    events.append(dict(
                        date=dt, action="sell", asset=t,
                        price=float(px.at[dt, t]) if t in px.columns else np.nan,
                        reason=("fails the screen" if not bool(ok.get(t, False))
                                else f"rank {rank.get(t, float('nan')):.0f} outside top {p.n_hold}"),
                        rank=float(rank.get(t, np.nan)),
                        score=float(annret.loc[dt].get(t, np.nan)),
                    ))
            for t in target:
                if t not in held:
                    events.append(dict(
                        date=dt, action="buy", asset=t, price=float(px.at[dt, t]),
                        reason=f"rank {rank[t]:.0f}, ann. return {order[t]:+.1%}",
                        rank=float(rank[t]), score=float(order[t]),
                    ))
            held = target
            held_ranks[dt] = {t: float(rank.get(t, np.nan)) for t in held}
        else:
            n_passing[dt] = n_passing.get(dt, len(held))

        holdings_log[dt] = list(held)

        if held:
            # Slot weighting, matching the momentum book so the comparison is
            # about selection rather than about sizing. With n_hold=0 there are
            # no fixed slots, so the book is fully invested across whatever
            # passed.
            slots = p.n_hold if (p.n_hold and p.equal_weight_slots) else len(held)
            weights.loc[dt, held] = 1.0 / slots
        risk_total = float(weights.loc[dt, px.columns].sum())
        weights.at[dt, p.safe_asset] = max(0.0, 1.0 - risk_total)

    diagnostics = pd.DataFrame({
        "n_passing": pd.Series(n_passing),
        "n_held": pd.Series({d: len(v) for d, v in holdings_log.items()}),
        "n_stage2": stage2.sum(axis=1),
        "n_rs_ok": rs_ok.sum(axis=1),
        "n_within_high": within_high.sum(axis=1),
        "benchmark_6m": rs6m_mkt,
    })

    ev = pd.DataFrame(events) if events else _empty_events()
    sig = StrategySignals(name, weights, diagnostics, ev, params=p.__dict__.copy())
    sig.holding = pd.Series({d: ",".join(v) for d, v in holdings_log.items()})
    sig.holdings_log = holdings_log
    sig.held_ranks = held_ranks
    sig.momentum = annret
    sig.params["sector_filter_applied"] = sector_filter_applied
    return sig


# ==========================================================================
# The Finviz screener, rolled forward
# ==========================================================================
# finviz_filter_with_daily_summary.ipynb runs three stages:
#
#   1. a Finviz filter pass  -- market cap, "Quarter Up", price > $10, above
#      SMA200, within 10% of the 52-week high, average volume > 200k
#   2. a sector-concentration table over the names that also clear $5 close,
#      $5m turnover and +20% on the quarter
#   3. an RS ranking of the STAGE-1 survivors (note: not the stage-2 subset --
#      the notebook ranks `candidate_tickers`), sorted by RS Rank descending
#      and then by distance below the 52-week high ascending
#
# Stage 3 is the selection rule, so that is what is reproduced here. Stage 2
# is available through `min_quarter_return` / `min_turnover` but is off by
# default, because in the notebook it feeds the breakdown table and nothing
# else. Stage 1 is a set of absolute tests, which is what makes this a screen
# rather than a ranking: on a bad day nothing passes and the book is in cash.


def finviz_momentum_screen(
    universe_prices: pd.DataFrame,
    safe_prices: pd.Series,
    params: Optional[FinvizScreenParams] = None,
    volumes: Optional[pd.DataFrame] = None,
    name: str = "finviz",
) -> StrategySignals:
    """Roll the Finviz screen forward and hold its top `n_hold` names.

    Parameters
    ----------
    universe_prices : date x ticker adjusted closes for the ranking universe.
    safe_prices     : the cash leg. Unfilled slots, and every date on which
                      too few names pass, sit here.
    volumes         : optional date x ticker share volume. Supplying it enables
                      the "Average Volume over 200K" criterion (and
                      `min_turnover`, if set); without it both are skipped and
                      `volume_filter_applied` on the result is False.

    The ranking
    -----------
    The notebook's `rank_momentum_stocks`: bucket the survivors' 1-year returns
    into `rs_buckets` percentile buckets to get an RS Rank of 1-99, sort by that
    descending, and break ties on the smallest distance below the 52-week high.

    Bucketing a rank is a monotone transform, so on a universe no larger than
    `rs_buckets` every name lands in its own bucket and the rule collapses to
    "sort by 1-year return". That is exactly what happens on a Nasdaq-100
    universe -- the tie-break never fires there. It is reproduced faithfully
    anyway, because on the notebook's own ~530-name screen it does fire.

    The 52-week high
    ----------------
    The notebook takes `df['High'].max()` -- an intraday high. This uses a
    rolling max of closes, because a wide universe cached as closes is what the
    rest of the package carries. A close-based high is never higher than the
    intraday one, so "within 10% of the high" admits slightly MORE names here.
    """
    p = params or FinvizScreenParams()
    px = universe_prices.sort_index()
    safe = safe_prices.reindex(px.index).ffill()

    # ---- stage 1: the Finviz filters --------------------------------------
    sma = px.rolling(p.above_sma, min_periods=p.above_sma).mean()
    above_sma = px > sma

    high_52w = px.rolling(p.high_window, min_periods=p.high_window).max()
    pct_off_high = 1.0 - px / high_52w
    within_high = pct_off_high <= p.within_52w_high_pct

    priced = px >= p.min_price

    quarter_ret = px / px.shift(p.quarter_lookback) - 1.0
    quarter_up = quarter_ret > 0 if p.require_quarter_up else px.notna()

    # A name needs enough history before any of this means anything.
    history = px.notna().cumsum()
    has_history = history >= p.min_history

    passes = above_sma & within_high & priced & quarter_up & has_history

    # ---- the volume criteria, only if the caller supplied the data --------
    volume_filter_applied = volumes is not None
    if volume_filter_applied:
        vol = volumes.reindex(index=px.index, columns=px.columns).ffill()
        avg_vol = vol.rolling(p.avg_volume_window,
                              min_periods=p.avg_volume_window).mean()
        passes &= avg_vol > p.min_avg_volume
        if p.min_turnover is not None:
            passes &= (px * vol) >= p.min_turnover
    elif p.min_turnover is not None:
        raise ValueError("min_turnover needs `volumes`; pass it or leave the "
                         "parameter at None")

    # ---- the notebook's stage-2 gate, if it was switched on ---------------
    if p.min_quarter_return is not None:
        passes &= quarter_ret >= p.min_quarter_return

    # ---- stage 3: the RS measure ------------------------------------------
    # Perf_1Y, on a fixed lookback. See FinvizScreenParams on why it is fixed.
    perf = px / px.shift(p.rs_lookback) - 1.0
    passes &= perf.notna()

    # ---- rebalance calendar -----------------------------------------------
    if p.rebalance == "daily":
        rebal_set = set(px.index)
    else:
        marks = px.index.to_series().resample(p.rebalance).last().dropna()
        rebal_set = {d for d in marks if d in px.index}

    assets = list(px.columns) + [p.safe_asset]
    weights = pd.DataFrame(0.0, index=px.index, columns=assets)

    held: List[str] = []
    events: List[Dict] = []
    holdings_log: Dict[pd.Timestamp, List[str]] = {}
    held_ranks: Dict[pd.Timestamp, Dict[str, float]] = {}
    n_passing: Dict[pd.Timestamp, float] = {}
    last_n_passing = np.nan

    for dt in px.index:
        if dt in rebal_set:
            ok = passes.loc[dt].fillna(False)
            cand = perf.loc[dt][ok].dropna()
            n_passing[dt] = last_n_passing = float(len(cand))

            order, rs_rank = _rs_rank_order(cand, pct_off_high.loc[dt], p.rs_buckets)
            rank = pd.Series(np.arange(1, len(order) + 1), index=order.index)

            if p.exit_rank:
                # A name that stops passing is simply absent from `rank`, so
                # `inf` drops it -- the band only ever protects a name that is
                # still passing but has slipped down the ordering.
                keep = [t for t in held if rank.get(t, np.inf) <= p.exit_rank]
                for t in order.index:
                    if p.n_hold and len(keep) >= p.n_hold:
                        break
                    if t not in keep:
                        keep.append(t)
                target = keep
            else:
                # The notebook's rule: re-screen from scratch, no memory.
                target = list(order.index) if p.n_hold == 0 else list(order.index[:p.n_hold])

            for t in held:
                if t not in target:
                    r = rank.get(t, float("nan"))
                    if not bool(ok.get(t, False)):
                        why = "fails the screen"
                    elif p.exit_rank:
                        why = f"RS rank {r:.0f} > {p.exit_rank}"
                    else:
                        why = f"RS rank {r:.0f} outside top {p.n_hold}"
                    events.append(dict(
                        date=dt, action="sell", asset=t,
                        price=float(px.at[dt, t]) if t in px.columns else np.nan,
                        reason=why,
                        rank=float(rank.get(t, np.nan)),
                        score=float(perf.loc[dt].get(t, np.nan)),
                    ))
            for t in target:
                if t not in held:
                    events.append(dict(
                        date=dt, action="buy", asset=t, price=float(px.at[dt, t]),
                        reason=(f"rank {rank[t]:.0f}, RS {rs_rank[t]:.0f}, "
                                f"1y {cand[t]:+.1%}, {pct_off_high.at[dt, t]:.1%} off high"),
                        rank=float(rank[t]), score=float(cand[t]),
                    ))
            held = target
            held_ranks[dt] = {t: float(rank.get(t, np.nan)) for t in held}
        else:
            # Carried forward, not recomputed: on a non-rebalance day the screen
            # was not evaluated, so the last count is the only honest answer.
            # Recording `len(held)` here would put the book's size in a column
            # named for the candidate pool's.
            n_passing[dt] = last_n_passing

        holdings_log[dt] = list(held)

        if held:
            # Per-slot weighting, matching the momentum book, so the comparison
            # is about selection rather than about sizing.
            slots = p.n_hold if (p.n_hold and p.equal_weight_slots) else len(held)
            weights.loc[dt, held] = 1.0 / slots
        risk_total = float(weights.loc[dt, px.columns].sum())
        weights.at[dt, p.safe_asset] = max(0.0, 1.0 - risk_total)

    diagnostics = pd.DataFrame({
        "n_passing": pd.Series(n_passing),
        "n_held": pd.Series({d: len(v) for d, v in holdings_log.items()}),
        "cash_slots": pd.Series({d: max(0, p.n_hold - len(v))
                                 for d, v in holdings_log.items()}),
        "n_above_sma": above_sma.sum(axis=1),
        "n_within_high": within_high.sum(axis=1),
        "n_quarter_up": quarter_up.sum(axis=1),
    })

    ev = pd.DataFrame(events) if events else _empty_events()
    sig = StrategySignals(name, weights, diagnostics, ev, params=p.__dict__.copy())
    sig.holding = pd.Series({d: ",".join(v) for d, v in holdings_log.items()})
    sig.holdings_log = holdings_log
    sig.held_ranks = held_ranks
    sig.momentum = perf
    sig.params["volume_filter_applied"] = volume_filter_applied
    sig.params["market_cap_filter_applied"] = False
    return sig


def _rs_rank_order(candidates: pd.Series, pct_off_high: pd.Series, buckets: int):
    """The notebook's RS Rank ordering: bucket, sort descending, tie-break on high.

    Returns the candidates in selection order plus their 1-99 RS Rank. An empty
    candidate set returns empty series rather than raising -- on a bad day
    nothing passes, and that has to mean cash, not an error.
    """
    if candidates.empty:
        return candidates, candidates

    n = len(candidates)
    q = min(buckets, n)
    if q < 2:
        rs = pd.Series(1.0, index=candidates.index)
    else:
        # rank(method="first") makes the input strictly increasing, so qcut
        # splits it into q equal-sized buckets with no duplicate-edge collapse.
        rs = pd.Series(
            pd.qcut(candidates.rank(method="first"), q=q, labels=False,
                    duplicates="drop"),
            index=candidates.index,
        ).astype(float) + 1.0

    frame = pd.DataFrame({
        "rs": rs,
        "off_high": pct_off_high.reindex(candidates.index).fillna(np.inf),
    })
    frame = frame.sort_values(["rs", "off_high"], ascending=[False, True])
    return candidates.reindex(frame.index), rs.reindex(frame.index)
