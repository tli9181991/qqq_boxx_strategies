"""Volume and hammer-candle readings for the picks tab's analysis table.

Descriptive only. Nothing here feeds a ranking, a screen or an order: it is a
second look at names the books already chose, for a person reading the tab.

The hammer rules
----------------
A hammer is a session that sold off hard and was bought back up to near its
open: a long lower shadow, a small body near the top of the range, and little
or no upper shadow. The textbook description is qualitative, so these are the
thresholds used to make it a yes/no, each a fraction of the session's range
(high - low) so a $30 name and a $900 name are judged alike:

1. **Small body** -- |close - open| <= `max_body` of the range (35%).
2. **Long lower shadow** -- min(open, close) - low >= `min_lower_to_body` x the
   body (2x, the usual definition) AND >= `min_lower` of the range (55%). The
   second leg matters for a near-doji: with a body of almost zero, "twice the
   body" is satisfied by any shadow at all.
3. **Little upper shadow** -- high - max(open, close) <= `max_upper` of the
   range (15%). This is what separates a hammer from a spinning top.
4. **A range worth reading** -- the range >= `min_range_atr` x the prior
   14-session ATR (0.5x). A quiet session whose whole range is a few cents
   draws a perfect hammer that says nothing about buyers or sellers.

Rules 1-4 are the *shape*. The pattern also needs **context**: a hammer is a
reversal signal only after a decline, so the close before the candle must be
below the close `trend_days` (5) sessions earlier. The same shape after a rise
is a *hanging man*, which reads the opposite way, so the two are reported
apart rather than merged into one "hammer" flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HammerRules:
    max_body: float = 0.35          # body as a share of the range, at most
    min_lower_to_body: float = 2.0  # lower shadow as a multiple of the body
    min_lower: float = 0.55         # lower shadow as a share of the range
    max_upper: float = 0.15         # upper shadow as a share of the range
    min_range_atr: float = 0.5      # range as a multiple of the prior ATR
    atr_window: int = 14
    trend_days: int = 5             # decline measured into the candle


def hammer_frame(ohlc: pd.DataFrame,
                 rules: Optional[HammerRules] = None,
                 span: int = 1) -> pd.DataFrame:
    """Per-session candle geometry and the hammer verdict.

    `span` merges that many sessions into one candle ending on each date --
    the way a chart reader squints at two or three bars together. The merged
    candle opens at its first session's open, closes at its last session's
    close, and spans their highest high and lowest low, so a sell-off on one
    day bought back on the next reads as the long lower shadow it is. With a
    merged candle the prior decline is measured into its FIRST session, the
    ATR is the one known before that session, and the range floor grows with
    the square root of `span`, since a k-day range is naturally wider than a
    day's. `span=1` is the single-session candle, unchanged.

    Columns: `body`, `lower`, `upper` (shares of the range), `range_atr`
    (range over the prior ATR), `prior_ret` (the `trend_days` return into the
    candle), `shape` (rules 1-4), `hammer` (shape after a decline) and
    `hanging_man` (shape after a rise).

    A zero-range session (high == low) has no geometry: its shares are NaN and
    it is never a hammer. The ATR and the prior return use only sessions
    BEFORE the candle, so a bar is judged against what preceded it.
    """
    r = rules or HammerRules()
    span = max(1, int(span))
    do, dh, dl, c = (ohlc[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    # Daily true range and ATR first: they describe the sessions BEFORE the
    # candle, whatever its length.
    prev = c.shift(1)
    tr = pd.concat([dh - dl, (dh - prev).abs(), (dl - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(r.atr_window, min_periods=r.atr_window).mean().shift(span)
    o = do.shift(span - 1)
    h = dh.rolling(span, min_periods=span).max()
    l = dl.rolling(span, min_periods=span).min()
    rng = (h - l).where(h > l)
    top, bot = np.maximum(o, c), np.minimum(o, c)
    body = (c - o).abs() / rng
    lower = (bot - l) / rng
    upper = (h - top) / rng

    range_atr = (h - l) / atr
    floor = r.min_range_atr * np.sqrt(span)

    before = c.shift(span)               # the close before the candle's first session
    prior_ret = before / c.shift(span + r.trend_days) - 1.0

    shape = ((body <= r.max_body)
             & (lower >= r.min_lower)
             & (lower >= r.min_lower_to_body * body)
             & (upper <= r.max_upper)
             & (range_atr >= floor))
    shape = shape.fillna(False).astype(bool)
    return pd.DataFrame({
        "body": body, "lower": lower, "upper": upper,
        "range_atr": range_atr, "prior_ret": prior_ret,
        "shape": shape,
        "hammer": shape & (prior_ret < 0).fillna(False),
        "hanging_man": shape & (prior_ret > 0).fillna(False),
    }, index=ohlc.index)


def volume_stats(volume: pd.Series, close: Optional[pd.Series] = None,
                 window: int = 20) -> Dict[str, float]:
    """Last session's volume against the average of the `window` before it.

    The average EXCLUDES the last session, so a volume spike is measured
    against a baseline it did not help set -- including it would pull the
    average toward the spike and understate the ratio, most of all on the
    days the ratio matters.

    `avg_value` is the average dollar volume (close x shares) over the same
    window when closes are given, which is what makes two names' liquidity
    comparable.
    """
    v = volume.dropna().astype(float)
    nan = float("nan")
    if v.empty:
        return dict(last=nan, avg=nan, ratio=nan, avg_value=nan, n=0)
    last = float(v.iloc[-1])
    prior = v.iloc[:-1].tail(window)
    avg = float(prior.mean()) if len(prior) else nan
    ratio = last / avg if avg == avg and avg > 0 else nan
    avg_value = nan
    if close is not None and len(prior):
        dv = (close.astype(float).reindex(prior.index) * prior).dropna()
        avg_value = float(dv.mean()) if len(dv) else nan
    return dict(last=last, avg=avg, ratio=ratio, avg_value=avg_value,
                n=int(len(prior)))
