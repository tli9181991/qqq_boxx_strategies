"""Filter-based stock screens, as opposed to cross-sectional ranking.

`strategies.cross_sectional_momentum` asks "which are the strongest N names in
the universe" -- a *relative* question, always answerable, always full.
A screen asks "does this name pass, yes or no" -- an *absolute* question, which
on a bad day nothing passes. The two families behave differently enough at the
edges that it is worth keeping them apart.

`trend_template_screen` implements the Weinstein/Minervini-style trend template
from M6_finalnotebook.ipynb, rolled forward so it can be backtested rather than
evaluated once on the last bar.

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

from .config import SAFE_ASSET, TRADING_DAYS
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
