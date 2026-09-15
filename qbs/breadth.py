"""Market breadth and momentum-leader statistics.

The numbers behind the dashboard's market tab: how many names moved 4%, how
many hold their moving averages, how far the index has stretched from its own
50-day EMA, and what the leadership group looks like by sector.

Read this before trusting a reading
-----------------------------------
Breadth is a statement about a UNIVERSE, and it is only as good as that
universe. The reference dashboard this mirrors samples ~2,400 US common stocks
and ADRs from the consolidated tape. This package ships ~99 Nasdaq-100 daily
closes. The same function over those two inputs produces two different
measurements, and only one of them is "market breadth":

* A 99-name mega-cap index is not the market. Its `up4` counts are smaller by
  construction, its percent-above-MA lines move together far more, and a
  reading of "28% above the 20-day" means something quite different when the
  sample is one sector-concentrated index.
* Every function here therefore reports `n_stocks` alongside the reading, and
  `BreadthResult.universe_note` carries the caveat to the UI. Show both.

Two inputs are optional because this package does not carry them, and the
metrics that need them are skipped rather than approximated:

* **volume** -- the leadership screen's turnover test ($5m/day) cannot run
  without it. Pass `volumes=` to enable it. Without it the leader set is
  defined on price and quarterly return alone, which is strictly MORE
  permissive, so the leader count is an over-estimate.
* **sector map** -- the sector table needs `ticker -> sector`. Without one
  `sector_breakdown` returns empty rather than inventing a classification.

`spx` is likewise whatever index series the caller hands over; this package
caches QQQ, not SPY or ^GSPC, and the column is simply absent if nothing is
passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BreadthParams:
    """Thresholds for the breadth table, matching the reference dashboard."""
    move_pct: float = 0.04            # the "4% up / 4% down" test
    ma_fast: int = 20                 # "% holding the 20-day line"
    ma_slow: int = 50                 # "% holding the 50-day line"
    ema_span: int = 50                # index distance is measured from EMA(50)
    atr_window: int = 14              # ... in units of ATR(14)

    # Momentum leaders ("動力股"): the finviz notebook's own definition.
    leader_min_price: float = 5.0
    leader_min_turnover: float = 5_000_000.0   # needs `volumes`
    leader_quarter_days: int = 63
    leader_min_quarter_return: float = 0.20

    # Colour thresholds, kept here so the UI never invents its own.
    pulse_strong: int = 300           # |count| at or above this reads as strong
    ma_extreme_low_fast: float = 10.0
    ma_extreme_low_slow: float = 20.0
    ma_extreme_high_fast: float = 90.0
    ma_extreme_high_slow: float = 80.0
    atr_stretched: float = 5.0        # |ATR distance| beyond this is extended


@dataclass
class BreadthResult:
    """The daily table plus what the caller needs to caveat it honestly."""
    table: pd.DataFrame
    n_stocks: int
    universe_note: str
    has_volume: bool = False
    has_index: bool = False
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None

    @property
    def latest(self) -> pd.Series:
        return self.table.iloc[-1]

    @property
    def previous(self) -> Optional[pd.Series]:
        return self.table.iloc[-2] if len(self.table) > 1 else None


# --------------------------------------------------------------------------
# Index stretch
# --------------------------------------------------------------------------

def atr_distance(
    close: pd.Series,
    high: Optional[pd.Series] = None,
    low: Optional[pd.Series] = None,
    ema_span: int = 50,
    atr_window: int = 14,
) -> pd.Series:
    """(close - EMA) / ATR -- how far an index has stretched, in ATR units.

    This is `calculate_index_atr_distance` from the finviz notebook, kept to
    the same definition so the dashboard and that notebook agree.

    With no high/low the true range degenerates to the close-to-close move,
    which understates the real range and therefore OVERSTATES the distance.
    A reading built that way is an upper bound on how stretched things are.
    """
    close = close.astype(float)
    if high is None or low is None:
        high = pd.concat([close, close.shift()], axis=1).max(axis=1)
        low = pd.concat([close, close.shift()], axis=1).min(axis=1)

    ema = close.ewm(span=ema_span, adjust=False).mean()
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean()
    return (close - ema) / atr.replace(0.0, np.nan)


# --------------------------------------------------------------------------
# Momentum leaders
# --------------------------------------------------------------------------

def leader_mask(
    closes: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    p: Optional[BreadthParams] = None,
) -> pd.DataFrame:
    """date x ticker boolean: was this name a momentum leader that day?

    The finviz notebook's `動力股` rule -- close at or above `leader_min_price`,
    dollar turnover at or above `leader_min_turnover`, and a quarterly return
    at or above `leader_min_quarter_return`. The turnover leg is skipped when
    no volume is supplied, which makes the set strictly larger.
    """
    p = p or BreadthParams()
    px = closes.sort_index()
    qtr = px / px.shift(p.leader_quarter_days) - 1.0

    ok = (px >= p.leader_min_price) & (qtr >= p.leader_min_quarter_return)
    if volumes is not None:
        vol = volumes.reindex(index=px.index, columns=px.columns)
        ok &= (px * vol) >= p.leader_min_turnover
    return ok.fillna(False)


def sector_breakdown(
    closes: pd.DataFrame,
    sector_map: Dict[str, str],
    asof: Optional[pd.Timestamp] = None,
    volumes: Optional[pd.DataFrame] = None,
    p: Optional[BreadthParams] = None,
) -> pd.DataFrame:
    """Which sectors the leadership is concentrated in, on one date.

    Columns, in the reference dashboard's terms:

        n            leaders in this sector
        share_pct    that sector's share of ALL leaders
        pool_pct     the sector's share of the analysed universe -- its own
                     "body weight", the grey tick the bar has to beat
        penetration  leaders in the sector / analysed names in the sector
        excess_pp    share_pct - pool_pct, in percentage points

    `excess_pp` is the column that carries information. A sector holding 20%
    of the leaders is unremarkable if it is 20% of the universe; the same 20%
    from a sector that is 5% of the universe is the finding.
    """
    p = p or BreadthParams()
    if not sector_map:
        return pd.DataFrame(columns=["sector", "n", "share_pct", "pool_pct",
                                     "penetration", "excess_pp"])

    px = closes.sort_index()
    asof = pd.Timestamp(asof) if asof is not None else px.index[-1]
    leaders = leader_mask(px, volumes, p).loc[asof]

    # The analysed pool is every name with enough history to be judged at all.
    analysed = px.loc[asof].notna() & px.shift(p.leader_quarter_days).loc[asof].notna()
    sectors = pd.Series({t: sector_map.get(t, "Unclassified") for t in px.columns})

    pool = sectors[analysed].value_counts()
    lead = sectors[leaders & analysed].value_counts()
    total_lead, total_pool = int(lead.sum()), int(pool.sum())
    if total_lead == 0 or total_pool == 0:
        return pd.DataFrame(columns=["sector", "n", "share_pct", "pool_pct",
                                     "penetration", "excess_pp"])

    rows = []
    for sec, n in lead.items():
        in_pool = int(pool.get(sec, 0))
        share = 100.0 * n / total_lead
        pool_pct = 100.0 * in_pool / total_pool
        rows.append({
            "sector": sec, "n": int(n),
            "share_pct": share, "pool_pct": pool_pct,
            "penetration": (100.0 * n / in_pool) if in_pool else np.nan,
            "excess_pp": share - pool_pct,
        })
    return (pd.DataFrame(rows).sort_values("n", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# The daily table
# --------------------------------------------------------------------------

def _pct_above(px: pd.DataFrame, ma: pd.DataFrame,
               priced: pd.DataFrame) -> pd.Series:
    """Percent of names trading above a moving average, over names that HAVE one.

    Two traps here, both of which produce a confidently wrong number rather
    than a missing one:

    * During warm-up the average is NaN, and `px > NaN` is False -- not NaN.
      Counting those Falses reports "0% above the 50-day" for the first 49
      sessions, which on a chart reads as a total collapse of breadth at
      exactly the left edge of every series.
    * A name too young to have the average still counts in `priced`, so
      leaving it in the denominator drags the percentage down for as long as
      it is young.

    Both are fixed by making the denominator "names with an average today".
    """
    valid = ma.notna() & priced
    n = valid.sum(axis=1)
    above = (px > ma) & valid
    return 100.0 * above.sum(axis=1) / n.replace(0, np.nan)


def daily_breadth(
    closes: pd.DataFrame,
    qqq: Optional[pd.Series] = None,
    spy: Optional[pd.Series] = None,
    spx: Optional[pd.Series] = None,
    volumes: Optional[pd.DataFrame] = None,
    p: Optional[BreadthParams] = None,
    universe_note: str = "",
) -> BreadthResult:
    """One row per session: the whole breadth monitor.

    Columns
    -------
    up4 / dn4            names up / down `move_pct` on the day
    pct_above_fast/slow  percent holding the 20- and 50-day averages
    spy_atr / qqq_atr    index distance from EMA(50), in ATR(14) units
    mli_pct              the leadership group's mean return that day
    mli_up_pct           share of leaders that rose
    mli_n                how many leaders there were
    n_stocks             names with a price that day -- the sample behind the row
    spx                  whatever index level the caller passed, or absent
    """
    p = p or BreadthParams()
    px = closes.sort_index()
    if px.empty:
        raise ValueError("no prices -- nothing to measure")

    rets = px.pct_change()
    priced = px.notna()

    out = pd.DataFrame(index=px.index)
    out["up4"] = (rets >= p.move_pct).sum(axis=1)
    out["dn4"] = (rets <= -p.move_pct).sum(axis=1)

    fast = px.rolling(p.ma_fast, min_periods=p.ma_fast).mean()
    slow = px.rolling(p.ma_slow, min_periods=p.ma_slow).mean()
    out["pct_above_fast"] = _pct_above(px, fast, priced)
    out["pct_above_slow"] = _pct_above(px, slow, priced)

    has_index = False
    for name, series in (("spy_atr", spy), ("qqq_atr", qqq)):
        if series is not None:
            out[name] = atr_distance(series.reindex(px.index).ffill(),
                                     ema_span=p.ema_span,
                                     atr_window=p.atr_window)
            has_index = True
        else:
            out[name] = np.nan

    leaders = leader_mask(px, volumes, p)
    lead_ret = rets.where(leaders)
    out["mli_n"] = leaders.sum(axis=1)
    out["mli_pct"] = 100.0 * lead_ret.mean(axis=1)
    out["mli_up_pct"] = 100.0 * (lead_ret > 0).sum(axis=1) / out["mli_n"].replace(0, np.nan)

    out["n_stocks"] = priced.sum(axis=1)
    if spx is not None:
        out["spx"] = spx.reindex(px.index).ffill()

    # Rows before the slow average exists are not readings, they are warm-up.
    out = out.loc[out["pct_above_slow"].notna()]

    return BreadthResult(
        table=out, n_stocks=int(priced.iloc[-1].sum()),
        universe_note=universe_note or f"{px.shape[1]} tickers",
        has_volume=volumes is not None, has_index=has_index,
        start=out.index.min() if len(out) else None,
        end=out.index.max() if len(out) else None,
    )


# --------------------------------------------------------------------------
# Colour rules, defined once so the table and the chart never disagree
# --------------------------------------------------------------------------

def pulse_class(count: int, direction: str, p: Optional[BreadthParams] = None) -> str:
    """`up_strong` / `up` / `down` / `down_strong` for a 4% move count."""
    p = p or BreadthParams()
    strong = abs(int(count)) >= p.pulse_strong
    if direction == "up":
        return "up_strong" if strong else "up"
    return "down_strong" if strong else "down"


def ma_class(value: float, which: str, p: Optional[BreadthParams] = None) -> str:
    """Where a percent-above-MA reading sits: extreme low .. extreme high."""
    p = p or BreadthParams()
    if pd.isna(value):
        return "none"
    low = p.ma_extreme_low_fast if which == "fast" else p.ma_extreme_low_slow
    high = p.ma_extreme_high_fast if which == "fast" else p.ma_extreme_high_slow
    if value <= low:
        return "extreme_low"
    if value >= high:
        return "extreme_high"
    if value >= 70.0:
        return "high"
    if value >= 30.0:
        return "mid"
    return "low"


def atr_class(value: float, p: Optional[BreadthParams] = None) -> str:
    """`stretched` above +5 ATR, `oversold` below -5, else `normal`."""
    p = p or BreadthParams()
    if pd.isna(value):
        return "none"
    if value >= p.atr_stretched:
        return "stretched"
    if value <= -p.atr_stretched:
        return "oversold"
    return "normal"
