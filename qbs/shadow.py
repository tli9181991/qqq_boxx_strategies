"""Shadow books: what a candidate ranking rule would have held, logged daily.

Why this exists
---------------
Six years of cached history contain twelve independent half-year blocks. Every
selection variant measured against them -- trend gates, stop losses, shorter
lookbacks, liquidity screens, composite scores -- either failed outright or won
on one block and lost the advantage the moment that block was removed. At that
ratio of variants to blocks a backtest cannot settle anything; only data the
rule has never seen can.

So a candidate is not switched on to find out whether it works. It is scored
every day beside the live book, in a log, holding nothing and sending no
orders. After a year there are two return streams to compare, and the
comparison is out of sample.

Nothing here can affect the live book. The shadow rankings are computed from
the same prices after the real targets are decided, and the caller writes them
to a CSV. A failure in this module must never reach an order.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MomentumParams
from .strategies import cross_sectional_momentum

log = logging.getLogger(__name__)

# The candidate under observation, as of 2026-09. See `turn_score`.
DEFAULT_WEIGHTS: tuple = (0.5, 1.25, 2.0)


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: each row standardised across names.

    Standardised per date, not per name, because the score is only ever used to
    order the universe on one day against itself.
    """
    mean = frame.mean(axis=1)
    sd = frame.std(axis=1)
    return frame.sub(mean, axis=0).div(sd.replace(0.0, np.nan), axis=0)


def turn_score(prices: pd.DataFrame, weight: float,
               params: Optional[MomentumParams] = None) -> pd.DataFrame:
    """z(6-1) + `weight` * z(turn), the candidate tilt.

    `turn` is the recent month's return rate minus the prior quarter's, both as
    per-day rates so the 21-day and 42-day windows are comparable -- a raw
    difference of two unequal windows measures window length as much as it
    measures acceleration.

    The idea it encodes: a name whose last month is strong while its prior
    quarter was flat or falling has only just turned up, and has more of its
    move ahead of it than a name that has been running for two quarters. The
    six-year sample does not support it (the tilt's whole advantage sits in one
    half-year block), which is why it is logged rather than traded.
    """
    p = params or MomentumParams()
    look = int(round(p.lookback_months * 21))
    skip = int(round(p.skip_months * 21))
    quarter = 63

    mom = prices.shift(skip) / prices.shift(look) - 1.0
    recent = (prices / prices.shift(skip) - 1.0) / skip
    prior = (prices.shift(skip) / prices.shift(quarter) - 1.0) / (quarter - skip)
    return _zscore(mom) + weight * _zscore(recent - prior)


def shadow_books(
    universe: pd.DataFrame,
    safe_prices: pd.Series,
    params: Optional[MomentumParams] = None,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    asof: Optional[pd.Timestamp] = None,
) -> List[Dict]:
    """What each candidate weight would hold on `asof`, and at what rank.

    Scored by `cross_sectional_momentum` itself, with the score swapped -- the
    band, the slot weighting and the absolute filter are the live ones, not a
    second implementation of them. A second implementation is what drifts, and
    a shadow log that drifts from the book it is being compared against
    measures the drift rather than the rule.

    Returns one row per name held, tagged with the weight that held it. An
    empty list if nothing could be scored; this is a log, not a signal, so it
    fails quiet and the caller carries on.
    """
    p = params or MomentumParams()
    rows: List[Dict] = []
    for w in weights:
        try:
            sig = cross_sectional_momentum(
                universe, safe_prices, p,
                score=turn_score(universe, w, p), record_ranks=0)
            dt = asof or sig.weights.index[-1]
            held = list((sig.holdings_log or {}).get(dt, []))
            ranks = (sig.held_ranks or {}).get(dt, {})
            for slot, t in enumerate(held, 1):
                rows.append(dict(weight=w, slot=slot, symbol=t,
                                 rank=ranks.get(t, float("nan"))))
        except Exception as exc:          # noqa: BLE001 -- a log must not raise
            log.warning("shadow book w=%s could not be scored (%s: %s)",
                        w, type(exc).__name__, exc)
    return rows
