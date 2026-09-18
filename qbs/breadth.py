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
from typing import Dict, List, Optional, Sequence

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
        # `>=`, matching `priced` in `finviz_momentum_screen` -- a name sitting
        # exactly on the threshold passes the screen, so it passes here too.
        {"Strategy": "Finviz screen",
         "Rule": f"close at or above ${screen.min_price:,.0f}",
         "Value": last, "Fmt": "price", "Pass": bool(last >= screen.min_price)},
        {"Strategy": "Finviz screen", "Rule": f"above SMA {screen.above_sma}",
         "Value": (last / sma_last - 1.0) if sma_last and pd.notna(sma_last) else np.nan,
         "Fmt": "pct", "Pass": bool(last > sma_last) if pd.notna(sma_last) else None},
        {"Strategy": "Finviz screen",
         "Rule": f"within {screen.within_52w_high_pct:.0%} of "
                 f"{screen.high_window}-day high",
         "Value": off_high, "Fmt": "off",
         "Pass": bool(off_high <= screen.within_52w_high_pct)
         if pd.notna(off_high) else None},
    ]
    if screen.min_off_high_pct:
        # The band floor, only when one is configured -- the notebook's rule is
        # a ceiling alone, and a row asserting a floor it does not apply would
        # be reporting a strategy nobody is running.
        rows.append({"Strategy": "Finviz screen",
                     "Rule": f"at least {screen.min_off_high_pct:.0%} off the high",
                     "Value": off_high, "Fmt": "off",
                     "Pass": bool(off_high >= screen.min_off_high_pct)
                     if pd.notna(off_high) else None})
    if screen.require_quarter_up:
        rows.append({"Strategy": "Finviz screen",
                     "Rule": f"quarter up ({screen.quarter_lookback}d)",
                     "Value": screen_qtr, "Fmt": "pct",
                     "Pass": bool(screen_qtr > 0) if pd.notna(screen_qtr) else None})
    if screen.min_quarter_return is not None:
        rows.append({"Strategy": "Finviz screen",
                     "Rule": f"quarterly gain over {screen.min_quarter_return:.0%}",
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
