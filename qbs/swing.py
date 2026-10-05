"""Swing-trade stock selection: six setups, each a screen on daily closes.

| Setup                  | Looks for                                         | Hold    |
|------------------------|---------------------------------------------------|---------|
| Trend pullback         | a strong stock pulled back to support             | 3-15 d  |
| Breakout retest        | a broken resistance retested from above, holding  | 3-20 d  |
| Range bounce           | a turn up off the bottom of a sideways box        | 2-10 d  |
| Mean reversion         | a short-term oversold dip in a long-term uptrend  | 1-7 d   |
| Volatility contraction | ranges narrowing near the highs, before expansion | 3-15 d  |
| Relative strength      | names holding up while the market pulls back      | 5-20 d  |

Every screen reads closes only, so it runs on the cached universe with no
network. Ranges come from the close-to-close envelope (`closes_to_bars`),
which understates a real session's high-low range, and there is no volume
leg anywhere -- a breakout or a dry-up cannot be confirmed on volume here.

Levels are the dashboard's: `qbs.breakout.sr_levels`, then snapped to round
numbers and merged within one daily range (`qbs.breakout_monitor.
clean_levels`), so the support and resistance a screen reads are the lines
the charts draw.

Each screen returns None (the name does not qualify) or a dict with the
same core keys -- `close`, `stop`, `target`, `rr` (reward over risk from the
close) -- plus the readings that made it qualify. Stops and targets are
reference prices for sizing a trade, not orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .breakout import atr, closes_to_bars, sr_levels
from .breakout_monitor import average_daily_range, clean_levels, is_crypto
from .config import BreakoutParams
from .indicators import wilder_rsi


@dataclass(frozen=True)
class SwingParams:
    # trend pullback
    tp_min_pullback: float = 0.03      # off the 20-day closing high by at least
    tp_max_pullback: float = 0.15      # ... and at most this
    tp_support_atr: float = 1.0        # close within this many ATRs of support
    tp_rsi: tuple = (35.0, 55.0)       # RSI(14) band: cooled off, not broken
    # breakout retest
    br_min_age: int = 3                # breakout this many sessions ago, at least
    br_max_age: int = 20               # ... and at most
    br_run_atr: float = 1.0            # it ran at least this far past the level
    br_hold_atr: float = 0.5           # no close more than this below the level
    br_near_atr: float = 1.0           # now back within this far above it
    # range bounce
    rb_window: int = 40                # the box
    rb_width: tuple = (0.06, 0.30)     # box height as a fraction of its low
    rb_flat: float = 0.03              # SMA20 moved less than this over 10 bars
    rb_bottom: float = 0.30            # close in the lowest 30% of the box
    rb_touches: int = 2                # closes within 1 ATR of each edge
    # mean reversion (Connors RSI(2))
    mr_rsi2: float = 10.0              # RSI(2) below this; Connors' strict is 5
    # volatility contraction
    vc_atr_ratio: float = 0.75         # ATR(10) / ATR(50) below this
    vc_range_pct: float = 0.20         # 10-day range in its lowest 20% of 120 days
    vc_off_high: float = 0.15          # within this of the 52-week closing high
    # relative strength
    rs_lookback: int = 10              # return window for the comparison
    rs_min_gap: float = 0.05           # beat the market by 5 points or more
    rs_near_high: float = 0.05         # within 5% of its own 20-day high
    rs_mkt_pullback: float = 0.03      # market off its 20-day high by this ...
    rs_mkt_drop: float = -0.02         # ... or down this much over the window


@dataclass(frozen=True)
class Setup:
    key: str
    name: str
    looks_for: str
    hold: str


SETUPS = (
    Setup("trend_pullback", "Trend pullback",
          "A strong stock pulled back to support, ready to resume", "3–15 d"),
    Setup("breakout_retest", "Breakout retest",
          "After a breakout, a retest of the broken level that holds", "3–20 d"),
    Setup("range_bounce", "Range bounce",
          "A bounce off the bottom of a box toward its top", "2–10 d"),
    Setup("mean_reversion", "Mean reversion",
          "A short-term oversold dip, bought for the snap-back", "1–7 d"),
    Setup("volatility_contraction", "Volatility contraction",
          "Ranges narrowing, waiting for the expansion", "3–15 d"),
    Setup("relative_strength", "Relative strength swing",
          "Names that stay strong while the market pulls back", "5–20 d"),
)

MIN_BARS = 260


class Ctx:
    """Everything the screens read for one name, computed once."""

    def __init__(self, close: pd.Series, ticker: str = ""):
        c = close.dropna().astype(float)
        self.ticker = ticker
        self.c = c
        self.bars = closes_to_bars(c.to_frame("x"))["x"]
        self.last = float(c.iloc[-1])
        self.prev = float(c.iloc[-2])
        self.sma20 = c.rolling(20).mean()
        self.sma50 = c.rolling(50).mean()
        self.sma200 = c.rolling(200).mean()
        self.ema20 = c.ewm(span=20, adjust=False).mean()
        self.atr14 = atr(self.bars, 14)
        self.a = float(self.atr14.iloc[-1])
        self.rsi14 = wilder_rsi(c, 14)
        self.rsi2 = wilder_rsi(c, 2)
        self.high20 = float(c.iloc[-20:].max())
        self._levels = None

    @property
    def levels(self) -> List[float]:
        if self._levels is None:
            self._levels = self.levels_at(len(self.c))
        return self._levels

    def levels_at(self, n: int) -> List[float]:
        """Cleaned levels from the first `n` closes only (no look-ahead)."""
        c = self.c.iloc[:n]
        bars = self.bars.iloc[:n]
        raw = sr_levels(bars, BreakoutParams())
        return clean_levels(raw, float(c.iloc[-1]), not is_crypto(self.ticker),
                            average_daily_range(bars))

    def support_below(self) -> Optional[float]:
        below = [x for x in self.levels if x < self.last]
        return max(below) if below else None

    def resistance_above(self, price: Optional[float] = None) -> Optional[float]:
        p = self.last if price is None else price
        above = [x for x in self.levels if x > p]
        return min(above) if above else None


def _rr(close: float, stop: Optional[float], target: Optional[float]):
    if stop is None or target is None or stop >= close or target <= close:
        return None
    return (target - close) / (close - stop)


def _out(ctx: Ctx, stop, target, **readings) -> Dict:
    return {"close": ctx.last, "stop": stop, "target": target,
            "rr": _rr(ctx.last, stop, target), "atr": ctx.a, **readings}


def trend_pullback(ctx: Ctx, mkt: Optional[pd.Series], p: SwingParams) -> Optional[Dict]:
    """Uptrend (close > SMA200, SMA50 > SMA200 and rising), 3-15% off the
    20-day closing high, within one ATR of support (EMA20, SMA50 or the
    nearest level below; the close no more than half an ATR under it),
    RSI(14) cooled to 35-55. Stop an ATR under the lower of support and the
    close; target the 20-day high."""
    s50, s200 = ctx.sma50.iloc[-1], ctx.sma200.iloc[-1]
    if not (ctx.last > s200 and s50 > s200 and s50 > ctx.sma50.iloc[-11]):
        return None
    off = ctx.last / ctx.high20 - 1
    if not (-p.tp_max_pullback <= off <= -p.tp_min_pullback):
        return None
    rsi = float(ctx.rsi14.iloc[-1])
    if not (p.tp_rsi[0] <= rsi <= p.tp_rsi[1]):
        return None
    supports = {"EMA20": float(ctx.ema20.iloc[-1]), "SMA50": float(s50)}
    lvl = ctx.support_below()
    if lvl is not None:
        supports["level"] = lvl
    name, sup = min(supports.items(), key=lambda kv: abs(ctx.last - kv[1]))
    if not (sup - 0.5 * ctx.a <= ctx.last <= sup + p.tp_support_atr * ctx.a):
        return None
    return _out(ctx, min(sup, ctx.last) - ctx.a, ctx.high20,
                pullback=off, rsi14=rsi, support=sup, support_kind=name,
                turning_up=ctx.last > ctx.prev)


def breakout_retest(ctx: Ctx, mkt: Optional[pd.Series], p: SwingParams) -> Optional[Dict]:
    """A resistance level known BEFORE the breakout (levels from history
    ending `br_max_age + 1` sessions ago), closed above 3-20 sessions ago,
    that then ran at least an ATR past it, never closed more than half an
    ATR back under it, and is now within an ATR above it. The highest such
    level wins. Stop an ATR under the lower of the level and the close;
    target the post-breakout high."""
    c, a = ctx.c, ctx.a
    n = len(c)
    cut = n - (p.br_max_age + 1)
    pre = float(c.iloc[cut - 1])
    best = None
    for L in sorted((x for x in ctx.levels_at(cut) if x > pre), reverse=True):
        recent = c.iloc[cut - 1:]
        above = (recent > L).to_numpy()
        crosses = [i for i in range(1, len(above)) if above[i] and not above[i - 1]]
        if not crosses:
            continue
        b = cut - 1 + crosses[0]               # breakout bar
        age = n - 1 - b
        if not (p.br_min_age <= age <= p.br_max_age):
            continue
        since = c.iloc[b:]
        if since.max() < L + p.br_run_atr * a:
            continue
        if since.min() < L - p.br_hold_atr * a:
            continue
        if not (L - p.br_hold_atr * a <= ctx.last <= L + p.br_near_atr * a):
            continue
        best = (L, age, float(since.max()))
        break
    if best is None:
        return None
    L, age, peak = best
    return _out(ctx, min(L, ctx.last) - a, peak, level=L, age=age,
                vs_level_atr=(ctx.last - L) / a)


def range_bounce(ctx: Ctx, mkt: Optional[pd.Series], p: SwingParams) -> Optional[Dict]:
    """A 40-session box 6-30% tall with a flat SMA20 (under 3% over 10
    bars), both edges touched at least twice (closes within an ATR), the
    close in the bottom 30% of it and up on the day. Stop half an ATR under
    the box; target its top."""
    box = ctx.c.iloc[-p.rb_window:]
    hi, lo = float(box.max()), float(box.min())
    width = hi / lo - 1
    if not (p.rb_width[0] <= width <= p.rb_width[1]):
        return None
    if abs(ctx.sma20.iloc[-1] / ctx.sma20.iloc[-11] - 1) >= p.rb_flat:
        return None
    if ((box >= hi - ctx.a).sum() < p.rb_touches
            or (box <= lo + ctx.a).sum() < p.rb_touches):
        return None
    pos = (ctx.last - lo) / (hi - lo)
    if pos > p.rb_bottom or ctx.last <= ctx.prev:
        return None
    return _out(ctx, lo - 0.5 * ctx.a, hi, box_low=lo, box_high=hi,
                box_width=width, position=pos)


def mean_reversion(ctx: Ctx, mkt: Optional[pd.Series], p: SwingParams) -> Optional[Dict]:
    """Connors RSI(2): above the 200-day SMA, RSI(2) under 10 (5 is his
    strict version). His exit is the first close back above the 5-day SMA,
    which is the target here; the stop is two ATRs under the close, since
    the original has a time stop rather than a price stop."""
    if ctx.last <= ctx.sma200.iloc[-1]:
        return None
    r2 = float(ctx.rsi2.iloc[-1])
    if r2 >= p.mr_rsi2:
        return None
    sma5 = float(ctx.c.iloc[-5:].mean())
    return _out(ctx, ctx.last - 2 * ctx.a, sma5, rsi2=r2,
                below_sma5=ctx.last / sma5 - 1)


def volatility_contraction(ctx: Ctx, mkt: Optional[pd.Series], p: SwingParams) -> Optional[Dict]:
    """Uptrend (close > SMA50 > SMA200), within 15% of the 52-week closing
    high, ATR(10) under 0.75x ATR(50), and the 10-day closing range in the
    lowest 20% of its own last 120 days. The trade is the break of the
    10-day high (the pivot): stop a quarter ATR under the 10-day low,
    target two risk-units above the pivot."""
    c = ctx.c
    s50, s200 = ctx.sma50.iloc[-1], ctx.sma200.iloc[-1]
    if not (ctx.last > s50 > s200):
        return None
    hi52 = float(c.iloc[-252:].max())
    if ctx.last < hi52 * (1 - p.vc_off_high):
        return None
    a10 = float(atr(ctx.bars, 10).iloc[-1])
    a50 = float(atr(ctx.bars, 50).iloc[-1])
    ratio = a10 / a50 if a50 > 0 else np.nan
    if not ratio < p.vc_atr_ratio:
        return None
    rng = (c.rolling(10).max() - c.rolling(10).min()) / c
    hist = rng.iloc[-120:].dropna()
    pct = float((hist < hist.iloc[-1]).mean())
    if pct > p.vc_range_pct:
        return None
    pivot = float(c.iloc[-10:].max())
    stop = float(c.iloc[-10:].min()) - 0.25 * ctx.a
    out = _out(ctx, stop, pivot + 2 * (pivot - stop), atr_ratio=ratio,
               range_pct=pct, pivot=pivot, off_high=ctx.last / hi52 - 1)
    # The trade triggers above the pivot, so R:R is measured from there.
    out["rr"] = _rr(pivot, stop, out["target"]) if pivot > stop else None
    return out


def market_pullback(mkt: pd.Series, p: SwingParams) -> Dict:
    """Is the market (QQQ) pulling back? Off its 20-day high by 3% or more,
    or down 2% or more over the comparison window."""
    m = mkt.dropna().astype(float)
    off = float(m.iloc[-1] / m.iloc[-20:].max() - 1)
    ret = float(m.iloc[-1] / m.iloc[-1 - p.rs_lookback] - 1)
    return {"off_high": off, "ret": ret,
            "pullback": off <= -p.rs_mkt_pullback or ret <= p.rs_mkt_drop}


def relative_strength(ctx: Ctx, mkt: Optional[pd.Series], p: SwingParams) -> Optional[Dict]:
    """Beat the market by 5 points or more over 10 sessions, within 5% of its
    own 20-day high, above its SMA50. Reported whether or not the market is
    pulling back -- the tab says which regime it is, because "strong in a
    weak tape" only means something in a weak tape. Stop two ATRs under the
    close; target the nearest resistance above, else none."""
    if mkt is None:
        return None
    m = mkt.reindex(ctx.c.index).ffill()
    k = p.rs_lookback
    ret = ctx.last / float(ctx.c.iloc[-1 - k]) - 1
    mret = float(m.iloc[-1] / m.iloc[-1 - k] - 1)
    gap = ret - mret
    if gap < p.rs_min_gap:
        return None
    if ctx.last < ctx.high20 * (1 - p.rs_near_high):
        return None
    if ctx.last <= ctx.sma50.iloc[-1]:
        return None
    return _out(ctx, ctx.last - 2 * ctx.a, ctx.resistance_above(),
                ret=ret, mkt_ret=mret, rs_gap=gap,
                off_high=ctx.last / ctx.high20 - 1)


SCREENS: Dict[str, Callable] = {
    "trend_pullback": trend_pullback,
    "breakout_retest": breakout_retest,
    "range_bounce": range_bounce,
    "mean_reversion": mean_reversion,
    "volatility_contraction": volatility_contraction,
    "relative_strength": relative_strength,
}


# How each setup's table is ranked: (column, best is largest). Reward/risk
# where the stop and target are the setup's own; for mean reversion the
# deepest RSI(2), for contraction the tightest ATR ratio (its target is a
# fixed 2R, so R:R would rank nothing), for relative strength the gap.
SORT = {"trend_pullback": ("rr", True), "breakout_retest": ("rr", True),
        "range_bounce": ("rr", True), "mean_reversion": ("rsi2", False),
        "volatility_contraction": ("atr_ratio", False),
        "relative_strength": ("rs_gap", True)}


def scan(closes: pd.DataFrame, market: Optional[pd.Series],
         p: SwingParams = SwingParams()) -> Dict[str, pd.DataFrame]:
    """Every screen over every column of `closes`, as of its last row.

    Returns one frame per setup key, a row per qualifying name (index =
    ticker), sorted best first (see SORT). Names with under
    `MIN_BARS` closes are skipped -- every screen needs the 200-day SMA.
    """
    rows: Dict[str, List[Dict]] = {k: [] for k in SCREENS}
    for t in closes.columns:
        c = closes[t].dropna()
        if len(c) < MIN_BARS:
            continue
        ctx = Ctx(c, t)
        for key, fn in SCREENS.items():
            hit = fn(ctx, market, p)
            if hit is not None:
                rows[key].append({"ticker": t, **hit})
    out = {}
    for key, rs in rows.items():
        df = pd.DataFrame(rs)
        if not df.empty:
            df = df.set_index("ticker")
            col, desc = SORT[key]
            df = df.sort_values(col, ascending=not desc, na_position="last")
        out[key] = df
    return out
