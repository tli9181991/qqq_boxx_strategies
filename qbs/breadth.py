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

* **volume** -- the leadership screen's volume test (300k shares/day) cannot run
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import FinvizScreenParams, MomentumParams


@dataclass
class BreadthParams:
    """Thresholds for the breadth table, matching the reference dashboard."""
    move_pct: float = 0.04            # the "4% up / 4% down" test
    ma_fast: int = 20                 # "% holding the 20-day line"
    ma_slow: int = 50                 # "% holding the 50-day line"
    ema_span: int = 50                # index distance is measured from EMA(50)
    atr_window: int = 14              # ... in units of ATR(14)

    # Momentum leaders ("動力股"). Three legs, all strictly greater-than:
    # a US stock or ADR over $5, trading more than 300k shares a day, up more
    # than 28% on the quarter. The 28% was 20% until it was raised by hand --
    # it is a preference about how selective "high momentum" should be, not a
    # measured optimum, so it lives in a diff rather than being tuned.
    leader_min_price: float = 5.0
    # SHARES traded, not dollars: the leg is a share count and price does not
    # enter it. 300k shares is $1.5m/day at $5 and $30m/day at $100, so this
    # scales with price rather than holding a flat dollar floor -- which is
    # the opposite of the $5m dollar-volume test it replaced.
    leader_min_volume: float = 300_000.0            # needs `volumes`
    leader_quarter_days: int = 63              # ~one quarter of trading
    leader_min_quarter_return: float = 0.28

    # Colour thresholds, kept here so the UI never invents its own.
    pulse_strong: int = 300           # |count| at or above this reads as strong

    # Daily-monitor colour bands, as upper edges: (dark, light, light) with
    # everything above the last edge in the fourth band.
    #
    # Green means BULLISH on both columns, not "this is the up column". A day
    # with 30 names down 4% is a good day, so it shades green exactly as a day
    # with 400 names up 4% does. Colouring dn4 red at every level would make
    # a calm tape look like a falling one.
    up4_bands: Tuple[float, float, float] = (50.0, 100.0, 300.0)
    dn4_bands: Tuple[float, float, float] = (50.0, 100.0, 200.0)

    # The "% above the 20-day" column is a two-state read of the CURRENT tape,
    # so only the most recent sessions are shaded. Colouring the whole history
    # turns a regime indicator into wallpaper -- the eye stops seeing it.
    ma_fast_recent: int = 10          # sessions shaded, newest first
    ma_fast_green: float = 20.0       # above this is light green, at or below red
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
    # Sessions left out because most names had no close that day:
    # {date: (names that had one, names a normal session has)}.
    gaps: Dict[pd.Timestamp, Tuple[int, int]] = field(default_factory=dict)

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

    The finviz notebook's `動力股` rule -- close ABOVE `leader_min_price`,
    share volume ABOVE `leader_min_volume`, and a quarterly return ABOVE
    `leader_min_quarter_return`. The volume leg is skipped when no volume is
    supplied, which makes the set strictly larger.

    The volume leg counts SHARES. Price does not enter it, so it is worth
    knowing which names this admits that a dollar test would not, and the
    reverse: 400k shares of a $6 stock is $2.4m a day and passes here while
    failing a $5m floor, and 50k shares of a $200 stock is $10m a day and
    fails here while clearing that floor. Neither test is a stricter version
    of the other.

    At 300k it is also close to non-binding on either universe this package
    uses. The Finviz screen already filters to names averaging over 300k
    shares, and every Nasdaq-100 constituent trades far above it, so on most
    days this leg removes nobody -- it catches an unusually quiet session
    rather than an illiquid name.

    Strictly greater-than on all three, matching both the stated definition
    and Finviz's own "Over $5". It is not pedantry on the price leg: a stock
    sitting at exactly $5.00 is a common thing, and `>=` would admit names the
    universe screen itself excludes -- so the two filters would disagree about
    the same name.
    """
    p = p or BreadthParams()
    px = closes.sort_index()
    qtr = px / px.shift(p.leader_quarter_days) - 1.0

    ok = (px > p.leader_min_price) & (qtr > p.leader_min_quarter_return)
    if volumes is not None:
        vol = volumes.reindex(index=px.index, columns=px.columns)
        ok &= vol > p.leader_min_volume
    return ok.fillna(False)


def leader_eligibility(
    closes: pd.DataFrame,
    volumes: Optional[pd.DataFrame] = None,
    p: Optional[BreadthParams] = None,
    existing: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """`leader_mask`, shaped for the ranker's `eligible` argument.

    Kept here rather than at the call sites so the dashboard's definition of a
    momentum leader and the ranker's are the same object, not two readings of
    the same sentence. `existing` is any mask already in force (point-in-time
    index membership, say) and is ANDed in.
    """
    mask = leader_mask(closes, volumes=volumes, p=p)
    if existing is not None:
        mask &= existing.reindex(index=mask.index, columns=mask.columns).fillna(False)
    return mask


def relative_volume(
    volumes: pd.DataFrame,
    fast: int = 5,
    slow: int = 50,
) -> pd.DataFrame:
    """Recent share volume over its own longer average, per name.

    The ratio, not a level. An absolute floor says nothing inside the
    Nasdaq-100 -- every constituent trades millions of shares, so a 300k test
    removes nobody but a half-day -- whereas a name's volume against its own
    norm distinguishes a move the market is participating in from one it is
    ignoring. Above 1.0 is busier than usual; below, quieter.
    """
    v = volumes.sort_index().astype(float)
    return (v.rolling(fast, min_periods=max(fast // 2, 1)).mean()
            / v.rolling(slow, min_periods=max(slow // 2, 1)).mean())


def volume_eligibility(
    volumes: pd.DataFrame,
    min_ratio: float = 0.0,
    max_ratio: float = 0.0,
    min_shares: float = 0.0,
    fast: int = 5,
    slow: int = 50,
    existing: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """A date x ticker mask for the ranker, from volume alone.

    Any leg left at 0.0 is off, so the default is a mask that admits
    everything -- a filter nobody configured must not quietly remove names.

    `min_ratio` demands participation (volume running above its own average);
    `max_ratio` is the opposite test, excluding a name whose move is happening
    on unusually thin trade; `min_shares` is the absolute floor, kept for
    completeness and near-useless on a large-cap index.
    """
    rel = relative_volume(volumes, fast=fast, slow=slow)
    ok = pd.DataFrame(True, index=rel.index, columns=rel.columns)
    if min_ratio:
        ok &= rel > min_ratio
    if max_ratio:
        ok &= rel < max_ratio
    if min_shares:
        ok &= volumes.reindex_like(rel) > min_shares
    ok = ok.fillna(False)
    if existing is not None:
        ok &= existing.reindex(index=ok.index, columns=ok.columns).fillna(False)
    return ok


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


def sector_leaders(
    closes: pd.DataFrame,
    sector_map: Dict[str, str],
    asof: Optional[pd.Timestamp] = None,
    volumes: Optional[pd.DataFrame] = None,
    per_sector: int = 5,
    p: Optional[BreadthParams] = None,
    momentum: Optional[MomentumParams] = None,
) -> pd.DataFrame:
    """The strongest momentum leaders inside each sector, on one date.

    `sector_breakdown` says WHERE the leadership is. This says WHO it is, at
    the same moment and under the same rule -- a concentration reading nobody
    can name the members of is a number to nod at rather than act on.

    Columns: `sector`, `symbol`, `score` (the same 6-1 momentum the ranker
    uses), `rank_in_sector`, and `n_sector` -- every leader that sector has,
    not just the ones listed, so a truncated list is visibly truncated.

    Ordered by sector size and then by score, which is `sector_breakdown`'s
    own order, so the two tables read down the page in the same sequence.

    `per_sector` caps each sector's list; 0 means no cap. The cap exists
    because the US universe produces leaders in the hundreds and a page
    listing all of them is a page nobody reads. A name with too little
    history to score is dropped rather than sorted last -- an unscored name
    is not a weak one.
    """
    p = p or BreadthParams()
    cols = ["sector", "symbol", "score", "rank_in_sector", "n_sector"]
    if not sector_map or closes.empty:
        return pd.DataFrame(columns=cols)

    px = closes.sort_index()
    asof = pd.Timestamp(asof) if asof is not None else px.index[-1]
    if asof not in px.index:
        prior = px.index[px.index <= asof]
        if len(prior) == 0:
            return pd.DataFrame(columns=cols)
        asof = prior[-1]
    px = px.loc[:asof]

    leaders = leader_mask(px, volumes, p).loc[asof]
    named = [t for t in px.columns if bool(leaders.get(t, False))]
    if not named:
        return pd.DataFrame(columns=cols)

    # The ranker's own measure, from the ranker's own window helper: the
    # table's order has to be the order the strategy would put them in, and
    # a second expression for "6-1 momentum" is how that stops being true.
    look, skip = _momentum_window(momentum)
    if len(px) <= look:
        return pd.DataFrame(columns=cols)
    score = px.iloc[-1 - skip] / px.iloc[-1 - look] - 1.0

    rows = []
    for t in named:
        sc = score.get(t, np.nan)
        if sc != sc:                      # unscored, not weak
            continue
        rows.append({"sector": sector_map.get(t, "Unclassified"),
                     "symbol": t, "score": float(sc)})
    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    out["n_sector"] = out.groupby("sector")["symbol"].transform("size")
    out = out.sort_values(["n_sector", "sector", "score"],
                          ascending=[False, True, False])
    out["rank_in_sector"] = out.groupby("sector").cumcount() + 1
    if per_sector:
        out = out[out["rank_in_sector"] <= int(per_sector)]
    return out[cols].reset_index(drop=True)


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
    qqq_ohlc: Optional[pd.DataFrame] = None,
    spy_ohlc: Optional[pd.DataFrame] = None,
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

    A session most names have no close for (`qbs.data.thin_rows`) is left
    OUT and listed in `gaps`. Kept in, it poisons more than its own row: with
    no close that day, that day's return and the next day's are undefined for
    every missing name, and every 20- and 50-day average through it is
    undefined for the next 20 and 50 sessions -- so weeks of rows get read
    over the few hundred names that happened to be there. Left out, the
    averages run over the sessions that exist, and the ONE row after the gap,
    whose return would span two sessions, shows its move counts as missing
    rather than as a day's.

    `qqq_ohlc` / `spy_ohlc` (Open/High/Low/Close) give that index's ATR a
    real true range, and stand in for `qqq` / `spy` when those are None.
    Without it the range is close-to-close, which understates it and so
    overstates the distance -- see `atr_distance`.
    """
    from .data import thin_rows

    p = p or BreadthParams()
    px = closes.sort_index()
    if px.empty:
        raise ValueError("no prices -- nothing to measure")
    gaps = thin_rows(px)
    after_gap = []
    if gaps:
        px = px.drop(index=list(gaps))
        after_gap = sorted({px.index[i] for i in px.index.searchsorted(list(gaps))
                            if i < len(px)})

    rets = px.pct_change()
    priced = px.notna()

    out = pd.DataFrame(index=px.index)
    # Float, not int: the row after a dropped session carries NaN here.
    out["up4"] = (rets >= p.move_pct).sum(axis=1).astype(float)
    out["dn4"] = (rets <= -p.move_pct).sum(axis=1).astype(float)

    fast = px.rolling(p.ma_fast, min_periods=p.ma_fast).mean()
    slow = px.rolling(p.ma_slow, min_periods=p.ma_slow).mean()
    out["pct_above_fast"] = _pct_above(px, fast, priced)
    out["pct_above_slow"] = _pct_above(px, slow, priced)

    has_index = False
    for name, series, ohlc in (("spy_atr", spy, spy_ohlc),
                               ("qqq_atr", qqq, qqq_ohlc)):
        if ohlc is not None and not {"High", "Low", "Close"} <= set(ohlc.columns):
            ohlc = None
        if series is not None or ohlc is not None:
            if ohlc is not None:
                # On the OHLC frame's own calendar, then aligned: the EMA and
                # the ATR are properties of the index's sessions, not of
                # whichever days the stock universe happens to have.
                bars = ohlc.sort_index()
                d = atr_distance(bars["Close"], bars["High"], bars["Low"],
                                 ema_span=p.ema_span, atr_window=p.atr_window)
                out[name] = d.reindex(px.index)
                has_index = True
                continue
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
    # A return across a dropped session is a two-day move; counting it as a
    # day's would put a double-sized session in the 4% columns.
    if after_gap:
        out.loc[after_gap, ["up4", "dn4", "mli_pct", "mli_up_pct"]] = np.nan
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
        gaps=gaps,
    )


# --------------------------------------------------------------------------
# Colour rules, defined once so the table and the chart never disagree
# --------------------------------------------------------------------------

def pulse_class(count: int, direction: str, p: Optional[BreadthParams] = None) -> str:
    """`up_strong` / `up` / `down` / `down_strong` for a 4% move count."""
    p = p or BreadthParams()
    strong = pd.notna(count) and abs(int(count)) >= p.pulse_strong
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


PULSE_CELLS = ("dark_red", "light_red", "light_green", "dark_green")


def pulse_cell(value: float, which: str, p: Optional[BreadthParams] = None) -> str:
    """Which colour band a 4%-mover count falls in: dark_red .. dark_green.

    `which` is "up" or "down", and the two run in OPPOSITE directions -- a big
    up count is bullish, a big down count is not -- so the bands are read
    ascending for "up" and descending for "down". One function rather than
    two because getting the direction backwards on one of them is the easy
    mistake, and here it is a single reversal that a test can pin.
    """
    p = p or BreadthParams()
    if pd.isna(value):
        return "none"
    bands = p.up4_bands if which == "up" else p.dn4_bands
    idx = sum(value > edge for edge in bands)      # 0..3
    order = PULSE_CELLS if which == "up" else tuple(reversed(PULSE_CELLS))
    return order[idx]


def ma_fast_cell(value: float, rank: int,
                 p: Optional[BreadthParams] = None) -> str:
    """Two-state shading for the %-above-20-day column, recent rows only.

    `rank` is how far back the row is, 0 for the newest session. Rows beyond
    `ma_fast_recent` come back "none" and stay uncoloured, which is the point:
    this column answers "what is the tape doing NOW", and a shaded year of it
    is wallpaper.
    """
    p = p or BreadthParams()
    if pd.isna(value) or rank >= p.ma_fast_recent:
        return "none"
    return "light_green" if value > p.ma_fast_green else "light_red"


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


# --------------------------------------------------------------------------
# One name's momentum, numerically
# --------------------------------------------------------------------------

MOMENTUM_HORIZONS = (("1 week", 5), ("1 month", 21), ("3 months", 63),
                     ("6 months", 126), ("12 months", 252))


def momentum_label(p: Optional[MomentumParams] = None) -> str:
    """The book's own lookback, as the "6-1" style shorthand.

    Callers append the word "momentum" where it reads well, so the bare form
    also works inside a sentence.

    Derived, never written out. The ranker's lookback is a config value that
    has already moved once (12-1 to 6-1); a literal here would leave this
    table calling a 6-1 number "12-1 momentum", which is the exact failure
    this package keeps finding in its own logs.
    """
    p = p or MomentumParams()
    return f"{p.lookback_months:g}-{p.skip_months:g}"


def _momentum_window(p: Optional[MomentumParams] = None) -> tuple:
    """`(lookback, skip)` in trading days, matching `cross_sectional_momentum`."""
    p = p or MomentumParams()
    return int(round(p.lookback_months * 21)), int(round(p.skip_months * 21))


def momentum_profile(
    closes: pd.DataFrame,
    ticker: str,
    asof: Optional[pd.Timestamp] = None,
    p: Optional[BreadthParams] = None,
    safe: Optional[pd.Series] = None,
    ema_spans: Sequence[int] = (10, 20, 50, 200),
    screen: Optional[FinvizScreenParams] = None,
    momentum: Optional[MomentumParams] = None,
) -> Dict[str, pd.DataFrame]:
    """The numbers behind a price chart: returns, rank, location, and gates.

    Returns three frames under the keys `returns`, `trend` and `gates`.

    **Rank is a percentile within the universe you pass**, on the date you
    ask about, computed from the same trailing returns. A "+120% over twelve
    months" tells you nothing on its own -- the whole question a momentum
    strategy asks is *relative*, so the number only means something next to
    what every other candidate did over the same window.

    `gates` is the part worth reading. Each row is a rule one of the
    strategies in this package actually applies, with the value that decides
    it, so a chart can answer "why is the Finviz screen not holding this?"
    instead of leaving the reader to infer it. Every threshold is read off
    `screen` and `p`, never written out again, so retuning the strategy
    retunes the table with it.

    Two honest gaps. The momentum hurdle is measured against `safe` (BOXX) when
    supplied and against zero when not -- a weaker test, and the row says
    which was used. And the two legs that need share volume (the screen's
    average-volume filter, the leader rule's share volume) are omitted rather
    than guessed, so passing every row here is necessary but not sufficient.

    `Fmt` tells a caller how to render `Value`: `pct` a signed percentage,
    `price` a dollar level, `off` a distance BELOW the 52-week high, which is
    a positive number meaning the opposite of a gain and must not be printed
    with a `+`.
    """
    p = p or BreadthParams()
    screen = screen or FinvizScreenParams()
    px = closes.sort_index()
    asof = pd.Timestamp(asof) if asof is not None else px.index[-1]
    px = px.loc[:asof]
    if ticker not in px.columns or px[ticker].dropna().empty:
        return {"returns": pd.DataFrame(), "trend": pd.DataFrame(),
                "gates": pd.DataFrame()}

    s = px[ticker]
    last = float(s.iloc[-1])

    # ---- returns and the cross-sectional rank of each ---------------------
    rows = []
    for label, n in MOMENTUM_HORIZONS:
        if len(px) <= n:
            continue
        universe_ret = px.iloc[-1] / px.iloc[-1 - n] - 1.0
        r = universe_ret.get(ticker, np.nan)
        rows.append({"Horizon": label, "Return": r,
                     "Rank": _percentile(universe_ret, ticker),
                     "Universe median": universe_ret.median()})

    # The momentum book's own score: the return from `lookback` ago to `skip`
    # ago, on the window the ranker is configured with rather than a literal.
    look, skip = _momentum_window(momentum)
    mom_label = momentum_label(momentum)
    mom_score = np.nan
    if len(px) > look:
        u = px.iloc[-1 - skip] / px.iloc[-1 - look] - 1.0
        mom_score = u.get(ticker, np.nan)
        rows.append({"Horizon": f"{mom_label} momentum", "Return": mom_score,
                     "Rank": _percentile(u, ticker),
                     "Universe median": u.median()})
    returns = pd.DataFrame(rows)

    # ---- where the price sits ---------------------------------------------
    trend_rows = []
    for span in ema_spans:
        ema = s.ewm(span=span, adjust=False, min_periods=span).mean()
        v = float(ema.iloc[-1]) if ema.notna().any() else np.nan
        trend_rows.append({"Measure": f"vs EMA {span}",
                           "Value": (last / v - 1.0) if v and not np.isnan(v) else np.nan,
                           "Unit": "%"})

    high_252 = float(s.tail(252).max()) if len(s) >= 2 else np.nan
    trend_rows.append({"Measure": "Off 52-week high",
                       "Value": (high_252 - last) / high_252 if high_252 else np.nan,
                       "Unit": "%"})
    sma200 = s.rolling(200, min_periods=200).mean()
    sma_v = float(sma200.iloc[-1]) if sma200.notna().any() else np.nan
    trend_rows.append({"Measure": "vs SMA 200",
                       "Value": (last / sma_v - 1.0) if sma_v and not np.isnan(sma_v) else np.nan,
                       "Unit": "%"})
    atr_pct = _atr_pct(s, p.atr_window)
    trend_rows.append({"Measure": f"ATR({p.atr_window})", "Value": atr_pct, "Unit": "%"})
    trend_rows.append({"Measure": "Distance from 50-day EMA",
                       "Value": float(atr_distance(s, ema_span=p.ema_span,
                                                   atr_window=p.atr_window).iloc[-1]),
                       "Unit": "ATR"})
    trend_rows.append({"Measure": "Annualised volatility",
                       "Value": float(s.pct_change().tail(252).std() * np.sqrt(252)),
                       "Unit": "%"})
    trend = pd.DataFrame(trend_rows)

    # ---- the gates each strategy actually applies --------------------------
    # Every threshold here is read off the screen parameters rather than
    # written out again, so the panel cannot drift from the screen it claims
    # to be reporting: retune `FinvizScreenParams` and these rows retune with
    # it. Only the quarter window is the screen's own (`quarter_lookback`),
    # which is why the leader row below recomputes its gain separately.
    def _gain(n: int) -> float:
        return last / float(s.iloc[-1 - n]) - 1.0 if len(s) > n else np.nan

    screen_qtr = _gain(screen.quarter_lookback)
    leader_qtr = _gain(p.leader_quarter_days)
    high_w = float(s.tail(screen.high_window).max()) if len(s) >= 2 else np.nan
    off_high = (high_w - last) / high_w if high_w else np.nan
    sma_last = np.nan
    if screen.above_sma:
        sma_n = s.rolling(screen.above_sma, min_periods=screen.above_sma).mean()
        sma_last = float(sma_n.iloc[-1]) if sma_n.notna().any() else np.nan

    # The hurdle is measured over the SAME window as the score. Comparing a
    # 6-1 stock return against a 12-1 cash return would be a different test
    # from the one `absolute_filter` applies.
    hurdle, hurdle_label = 0.0, "zero (no safe asset supplied)"
    if safe is not None:
        sf = safe.reindex(px.index).ffill()
        if len(sf) > look and pd.notna(sf.iloc[-1 - skip]) and pd.notna(sf.iloc[-1 - look]):
            hurdle = float(sf.iloc[-1 - skip] / sf.iloc[-1 - look] - 1.0)
            hurdle_label = f"BOXX over the same window ({hurdle:+.1%})"

    rows = [
        {"Strategy": "Momentum book",
         "Rule": f"{mom_label} momentum beats {hurdle_label}",
         "Value": mom_score, "Fmt": "pct",
         "Pass": bool(mom_score > hurdle) if pd.notna(mom_score) else None},
        # `>`, matching `priced` in `finviz_momentum_screen`.
        {"Strategy": "Momentum screen",
         "Rule": f"close over ${screen.min_price:,.0f}",
         "Value": last, "Fmt": "price", "Pass": bool(last > screen.min_price)},
    ]
    # Rows only for the legs the screen actually applies. A row for a filter
    # that is switched off would report a strategy nobody is running, and the
    # reader has no way to tell the difference from the table.
    if screen.above_sma:
        rows.append({
            "Strategy": "Momentum screen", "Rule": f"above SMA {screen.above_sma}",
            "Value": (last / sma_last - 1.0) if sma_last and pd.notna(sma_last) else np.nan,
            "Fmt": "pct", "Pass": bool(last > sma_last) if pd.notna(sma_last) else None})
    if screen.within_52w_high_pct is not None:
        rows.append({
            "Strategy": "Momentum screen",
            "Rule": f"within {screen.within_52w_high_pct:.0%} of "
                    f"{screen.high_window}-day high",
            "Value": off_high, "Fmt": "off",
            "Pass": bool(off_high <= screen.within_52w_high_pct)
            if pd.notna(off_high) else None})
    if screen.min_off_high_pct:
        # The band floor, only when one is configured -- the notebook's rule is
        # a ceiling alone, and a row asserting a floor it does not apply would
        # be reporting a strategy nobody is running.
        rows.append({"Strategy": "Momentum screen",
                     "Rule": f"at least {screen.min_off_high_pct:.0%} off the high",
                     "Value": off_high, "Fmt": "off",
                     "Pass": bool(off_high >= screen.min_off_high_pct)
                     if pd.notna(off_high) else None})
    if screen.require_quarter_up:
        rows.append({"Strategy": "Momentum screen",
                     "Rule": f"quarter up ({screen.quarter_lookback}d)",
                     "Value": screen_qtr, "Fmt": "pct",
                     "Pass": bool(screen_qtr > 0) if pd.notna(screen_qtr) else None})
    if screen.min_quarter_return is not None:
        rows.append({"Strategy": "Momentum screen",
                     "Rule": f"quarterly gain over {screen.min_quarter_return:.0%} "
                             f"({screen.quarter_lookback}d)",
                     "Value": screen_qtr, "Fmt": "pct",
                     "Pass": bool(screen_qtr >= screen.min_quarter_return)
                     if pd.notna(screen_qtr) else None})
    # The leader rule has three legs (`leader_mask`): price, share volume and
    # quarterly gain. Two are shown. The volume leg needs share volume, which
    # this panel does not carry, so it is left out rather than guessed -- a
    # name passing both rows below may still miss on volume.
    rows.append({"Strategy": "Momentum leader",
                 "Rule": f"close over ${p.leader_min_price:,.0f}",
                 "Value": last, "Fmt": "price",
                 "Pass": bool(last > p.leader_min_price)})
    rows.append({"Strategy": "Momentum leader",
                 "Rule": f"quarterly gain over {p.leader_min_quarter_return:.0%} "
                         f"({p.leader_quarter_days}d)",
                 "Value": leader_qtr, "Fmt": "pct",
                 "Pass": bool(leader_qtr > p.leader_min_quarter_return)
                 if pd.notna(leader_qtr) else None})
    gates = pd.DataFrame(rows)
    return {"returns": returns, "trend": trend, "gates": gates}


def _percentile(universe: pd.Series, ticker: str) -> float:
    """Where `ticker` sits in the cross-section, 1-99 like the notebook's RS Rank."""
    clean = universe.dropna()
    if ticker not in clean.index or len(clean) < 2:
        return np.nan
    return float(round(clean.rank(pct=True)[ticker] * 98 + 1))


def _atr_pct(close: pd.Series, window: int) -> float:
    """ATR as a fraction of price, from the close-to-close range.

    Close-only input, so this is the conservative (narrower) reading of range
    -- see `atr_distance` for the same caveat.
    """
    prev = close.shift()
    tr = pd.concat([(close - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    if atr.notna().sum() == 0 or close.iloc[-1] == 0:
        return float("nan")
    return float(atr.iloc[-1] / close.iloc[-1])


# --------------------------------------------------------------------------
# Bear-market checklist
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ChecklistRules:
    """Thresholds behind the checklist's automatic answers.

    The checklist's questions are qualitative ("stayed below 20% for over 10
    days", "continuing to decline without stabilizing"); these are the numbers
    that turn each into a yes/no, kept in one place so they can be tuned.
    """
    oversold_ma: int = 40               # Q1: % above the 40-day average ...
    oversold_level: float = 20.0        # ... below 20% ...
    oversold_days: int = 10             # ... for MORE than 10 sessions
    high_window: int = 252              # a "new high" / "new low" is a 52-week one
    final_high_ma: int = 50             # Q2: leaders above their 50-day average ...
    final_high_level: float = 30.0      # ... below 30% on the index's new-high day
    rising_days: int = 5                # Q3: "the index is rising" = up over 5 sessions
    mli_lookback: int = 5               # Q4: leaders fewer than 5 sessions ago ...
    mli_low_window: int = 10            # ... and at their lowest in 10 sessions


def bear_checklist(
    closes: pd.DataFrame,
    index_close: pd.Series,
    volumes: Optional[pd.DataFrame] = None,
    p: Optional[BreadthParams] = None,
    rules: Optional[ChecklistRules] = None,
    index_name: str = "QQQ",
) -> List[Dict]:
    """The four checklist questions this data can answer, on the last session.

    Each row is `{key, answer, reading}` -- `answer` True ("yes", the bearish
    reading), False, or None when there is not enough history to say, and
    `reading` the numbers behind it, so a "yes" can be checked rather than
    trusted.

    What is measured is this universe, not the NYSE: the % above the 40-day
    average and the new-high / new-low counts are over the names in
    `closes`, and "the index" is `index_close`. Sessions most names are
    missing are left out first, as in `daily_breadth`.
    """
    from .data import thin_rows

    p = p or BreadthParams()
    r = rules or ChecklistRules()
    px = closes.sort_index()
    gaps = thin_rows(px)
    if gaps:
        px = px.drop(index=list(gaps))
    idx = index_close.sort_index().reindex(px.index).ffill()
    out: List[Dict] = []

    # Q1 -- oversold that has turned into a trend.
    ma40 = px.rolling(r.oversold_ma, min_periods=r.oversold_ma).mean()
    pct40 = _pct_above(px, ma40, px.notna()).dropna()
    if pct40.empty:
        out.append(dict(key="oversold", answer=None,
                        reading=f"not enough history for a {r.oversold_ma}-day average"))
    else:
        below = (pct40 < r.oversold_level).to_numpy()[::-1]
        run = int(np.argmin(below)) if not below.all() else len(below)
        out.append(dict(
            key="oversold", answer=run > r.oversold_days,
            reading=(f"{pct40.iloc[-1]:.1f}% above the {r.oversold_ma}-day · "
                     + (f"below {r.oversold_level:.0f}% for {run} session"
                        f"{'s' if run != 1 else ''}" if run else
                        f"not below {r.oversold_level:.0f}%"))))

    # Q2 -- "final new high": the index makes one while leaders do not follow.
    hi = idx.rolling(r.high_window, min_periods=r.high_window).max()
    if pd.isna(hi.iloc[-1]):
        out.append(dict(key="final_high", answer=None,
                        reading=f"under {r.high_window} sessions of {index_name}"))
    else:
        new_high = bool(idx.iloc[-1] >= hi.iloc[-1])
        leaders = leader_mask(px, volumes, p).iloc[-1]
        ma50 = px.rolling(r.final_high_ma, min_periods=r.final_high_ma).mean().iloc[-1]
        valid = leaders & ma50.notna()
        n = int(valid.sum())
        share = (100.0 * (px.iloc[-1][valid] > ma50[valid]).sum() / n) if n else np.nan
        lead_txt = (f"{share:.0f}% of {n} leaders above their {r.final_high_ma}-day"
                    if n else "no leaders to measure")
        if new_high:
            out.append(dict(key="final_high",
                            answer=bool(n and share < r.final_high_level),
                            reading=f"{index_name} at a {r.high_window}-day high · {lead_txt}"))
        else:
            off = idx.iloc[-1] / hi.iloc[-1] - 1.0
            out.append(dict(key="final_high", answer=False,
                            reading=f"{index_name} not at a new high "
                                    f"({off:.1%} off its {r.high_window}-day high) · {lead_txt}"))

    # Q3 -- the index rising over a tape making more new lows than new highs.
    roll_hi = px.rolling(r.high_window, min_periods=r.high_window).max()
    roll_lo = px.rolling(r.high_window, min_periods=r.high_window).min()
    last = px.iloc[-1]
    has = roll_hi.iloc[-1].notna() & last.notna()
    n_hi = int((last[has] >= roll_hi.iloc[-1][has]).sum())
    n_lo = int((last[has] <= roll_lo.iloc[-1][has]).sum())
    if len(idx) <= r.rising_days or not has.any():
        out.append(dict(key="divergence", answer=None,
                        reading="not enough history for 52-week highs and lows"))
    else:
        chg = idx.iloc[-1] / idx.iloc[-1 - r.rising_days] - 1.0
        out.append(dict(
            key="divergence", answer=bool(chg > 0 and n_lo > n_hi),
            reading=(f"{index_name} {chg:+.1%} over {r.rising_days} sessions · "
                     f"{n_hi} new highs vs {n_lo} new lows")))

    # Q4 -- leaders still shrinking, no floor yet.
    mli = leader_mask(px, volumes, p).sum(axis=1)
    need = max(r.mli_lookback, r.mli_low_window) + 1
    if len(mli) < need or mli.iloc[-need:].eq(0).all():
        out.append(dict(key="mli", answer=None, reading="not enough history"))
    else:
        now, then = int(mli.iloc[-1]), int(mli.iloc[-1 - r.mli_lookback])
        floor = int(mli.iloc[-1 - r.mli_low_window:-1].min())
        out.append(dict(
            key="mli", answer=bool(now < then and now <= floor),
            reading=(f"{now} leaders now · {r.mli_lookback} sessions ago: {then} · "
                     f"lowest of the prior {r.mli_low_window}: {floor}")))
    return out
