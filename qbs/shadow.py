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

from .config import MomentumParams, ResidualMomentumParams
from .strategies import cross_sectional_momentum, residual_momentum_score

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


def parse_watchlist(raw: Optional[str]) -> List[str]:
    """Tickers out of a `QBS_WATCHLIST` string: comma- or space-separated.

    One parser, because the live runner and the dashboard read the same
    variable and a watchlist that means two things in two places is worse
    than one that means nothing.

    Upper-cased, and de-duplicated in order: a name listed twice is one
    watched name, not two identical rows reporting the same rank.
    """
    if not raw:
        return []
    return list(dict.fromkeys(
        t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()))


def _place(universe: pd.DataFrame, safe_prices: pd.Series, watch: pd.DataFrame,
           p: MomentumParams, score,
           asof: Optional[pd.Timestamp]):
    """Rank the constituents plus the watched outsiders once, then split them.

    Returns `(base, placed, outsiders)`: `base` is the constituents' own
    ranking as `[(symbol, score), ...]` best first, with the outsiders taken
    back out; `placed` is every ranked name's score, outsiders included. None
    when nothing could be scored.

    `score` swaps what the ranking sorts on -- a frame, or a function of the
    joined frame; None is the book's 6-1 -- through
    the same `cross_sectional_momentum` loop the book uses -- so the absolute
    filter and the history rule are the live ones for every score.

    Every rank a caller derives is measured against `base`, so adding a name
    to the watchlist cannot change what any other row reports.
    """
    if watch.empty:
        return None
    outsiders = [t for t in watch.columns if t not in universe.columns]
    try:
        frame = universe.join(watch[outsiders], how="left") if outsiders else universe
        if callable(score):
            score = score(frame)
        sig = cross_sectional_momentum(frame, safe_prices, p, score=score,
                                       record_ranks=frame.shape[1])
        dt = asof or sig.weights.index[-1]
        ranked = (sig.rank_log or {}).get(dt, [])
    except Exception as exc:              # noqa: BLE001 -- a log must not raise
        log.warning("watchlist could not be scored (%s: %s)", type(exc).__name__, exc)
        return None
    outsider_set = set(outsiders)
    base = [(t, sc) for t, _, sc in ranked if t not in outsider_set]
    placed = {t: sc for t, _, sc in ranked}
    return base, placed, outsider_set


def _rank_in(base: List[tuple], symbol: str, score: float) -> float:
    """Where `score` places in `base`; NaN for a name that was not ranked.

    Absent from the ranking means filtered out, not placed last: too little
    history, or it lost to the safe asset over the same window. Reporting that
    as "rank 99" would read as a weak name rather than an excluded one.

    Its own score is in `base` when it is a constituent and absent when it is
    not, so counting what strictly outranks it gives the standing rank in the
    first case and the interpolated one in the second, from the same
    expression.
    """
    if score != score:
        return float("nan")
    return float(1 + sum(1 for other, sc in base
                         if other != symbol and sc > score))


def watchlist_rows(
    universe: pd.DataFrame,
    safe_prices: pd.Series,
    watch: pd.DataFrame,
    params: Optional[MomentumParams] = None,
    asof: Optional[pd.Timestamp] = None,
) -> List[Dict]:
    """Where each watched name places in the constituents' 6-1 ranking.

    For reading a stock against the book without letting the book buy it. TSM
    is the case this was built for: a semiconductor the whole book is
    correlated with, and not a Nasdaq-100 constituent. Answering "is TSM
    stronger than what we hold?" by adding TSM to `extra_tickers` would answer
    a different question -- anything in the price frame used to be a name the
    ranker could put in the book.

    A watchlist is a list, and a list will mix the two kinds:

    * A **constituent** (GOOGL) is already ranked. Its rank is reported as it
      stands in the ranking the book acts on -- no interpolation, because none
      is needed and inventing one would disagree with `ranking_log.csv`.
    * An **outsider** (TSM) is interpolated into that same ranking: the place
      it would take among the constituents, and nothing else moves.

    Each name is placed against the constituents alone, never against the other
    watched names. Two outsiders on the list must not shift each other's
    reported rank -- each row answers one question, and one outsider's momentum
    has nothing to do with another's.

    A rank of 7 does not mean the book should hold it. For an outsider it means
    the book would hold it if it were a constituent, which is a different
    claim, and the reason this is a log rather than a signal.
    """
    p = params or MomentumParams()
    placed_rows = _place(universe, safe_prices, watch, p, score=None, asof=asof)
    if placed_rows is None:
        return []
    base, placed, outsider_set = placed_rows
    base_scores = [sc for _, sc in base]

    # A cutoff is the score of the name in that slot, so it exists only when
    # the slot is filled: pass `book` and the book holds you, fail `band` and
    # the book sells you.
    book_cut = base_scores[p.n_hold - 1] if len(base_scores) >= p.n_hold else float("nan")
    band_cut = base_scores[p.exit_rank - 1] if len(base_scores) >= p.exit_rank else float("nan")

    rows: List[Dict] = []
    for t in watch.columns:
        score = placed.get(t, float("nan"))
        rank = _rank_in(base, t, score)
        rows.append(dict(
            symbol=t, rank=rank, score=score,
            book_cutoff=book_cut, band_cutoff=band_cut,
            constituent=t not in outsider_set,
            beats_book=bool(rank == rank and rank <= p.n_hold),
        ))
    return rows


def watchlist_residual_ranks(
    universe: pd.DataFrame,
    safe_prices: pd.Series,
    market: pd.Series,
    watch: pd.DataFrame,
    params: Optional[ResidualMomentumParams] = None,
    asof: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """Each watched name's place in the RESIDUAL momentum book's ranking.

    The same placement `watchlist_rows` makes, on the score the residual book
    sorts on (`residual_momentum_score`) and with its own 12-1 window for the
    absolute filter -- i.e. the ranking `residual_momentum` acts on. A
    constituent reports its standing rank there; an outsider is interpolated
    into it without joining it.

    The residual score of one name depends only on that name and the market,
    so joining an outsider cannot move a constituent's score, and the rank is
    counted against the constituents alone as before.

    Returns `{symbol: rank}`, NaN for a name the ranker filtered out, and an
    empty dict if nothing could be scored -- a log must not raise.
    """
    return {t: r for t, (r, _) in watchlist_residual_rows(
        universe, safe_prices, market, watch, params, asof).items()}


def watchlist_residual_rows(
    universe: pd.DataFrame,
    safe_prices: pd.Series,
    market: pd.Series,
    watch: pd.DataFrame,
    params: Optional[ResidualMomentumParams] = None,
    asof: Optional[pd.Timestamp] = None,
) -> Dict[str, tuple]:
    """`watchlist_residual_ranks` with the score kept: `{symbol: (rank, score)}`.

    The score is the residual book's own sort key (a t-statistic on the
    name's market-neutral drift), NaN where the ranker filtered it out.
    """
    rp = params or ResidualMomentumParams()
    mp = MomentumParams(
        lookback_months=rp.lookback_months, skip_months=rp.skip_months,
        n_hold=rp.n_hold, exit_rank=rp.exit_rank, rebalance=rp.rebalance,
        absolute_filter=rp.absolute_filter, safe_asset=rp.safe_asset,
        min_history=rp.min_history)
    placed_rows = _place(
        universe, safe_prices, watch, mp,
        score=lambda frame: residual_momentum_score(frame, market, rp),
        asof=asof)
    if placed_rows is None:
        return {}
    base, placed, _ = placed_rows
    out = {}
    for t in watch.columns:
        sc = placed.get(t, float("nan"))
        out[t] = (_rank_in(base, t, sc), sc)
    return out
