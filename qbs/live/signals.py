"""Today's target weights, computed by the same code that produced the backtest.

The one rule this module exists to enforce: **the live book and the backtest
run the same functions.** Not a reimplementation, not a "live version" of the
ranking that drifts out of sync after two bugfixes -- literally
`cross_sectional_momentum` and `book_vol_target` from `qbs.strategies`, fed
fresh prices.

Why the whole history is recomputed every day
---------------------------------------------
It would be cheaper to carry yesterday's state forward and update it. It would
also be wrong within a fortnight. Both the hysteresis band and the vol scalar
are *path-dependent*: the band needs to know which names are currently held,
and the scalar needs the book's own realised return series. Persisting that
path in a state file means any missed run, any crash mid-write, any manual
intervention silently forks the live book away from what the strategy actually
says -- and you would not find out until you compared the two months later.

Recomputing from scratch costs a few seconds on a t3.small and makes the live
weights a pure function of the price history. Miss a day and the next run is
still exactly right. That property is worth far more than the seconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config, SAFE_ASSET
from ..strategies import book_vol_target, cross_sectional_momentum
from ..data import load_prices
from ..universe import load_universe, load_universe_prices

log = logging.getLogger(__name__)


@dataclass
class TargetBook:
    """What the strategy wants to hold, as of `asof`."""
    asof: pd.Timestamp
    weights: Dict[str, float]          # ticker -> target weight, sums to ~1.0
    prices: Dict[str, float]           # ticker -> the price the weight was computed on
    scalar: float                      # the vol-target scalar actually applied
    book_vol: float                    # the book's own realised vol estimate
    raw_holdings: List[str] = field(default_factory=list)   # the 6 names before scaling
    # Every name the strategy is entitled to trade: the rankable universe plus
    # the safe asset. `weights` carries only today's non-zero targets, so it
    # cannot tell a name the strategy just exited from a holding that was never
    # the strategy's business. The order builder needs that distinction to sell
    # the first and leave the second alone.
    universe: List[str] = field(default_factory=list)
    n_rankable: int = 0
    universe_size: int = 0
    diagnostics: Dict[str, float] = field(default_factory=dict)
    selection: List[Dict] = field(default_factory=list)     # entry/exit/hold, with rank

    @property
    def risk_weight(self) -> float:
        return float(sum(w for t, w in self.weights.items() if t != SAFE_ASSET))

    def describe(self) -> str:
        held = ", ".join(f"{t} {self.weights[t]:.1%}"
                         for t in self.raw_holdings if self.weights.get(t, 0) > 0)
        return (f"{self.asof:%Y-%m-%d}  book {self.risk_weight:.0%} "
                f"(scalar {self.scalar:.2f}, book vol {self.book_vol:.0%})\n"
                f"  holdings: {held or '(none)'}\n"
                f"  {SAFE_ASSET}: {self.weights.get(SAFE_ASSET, 0.0):.1%}")


class SignalError(RuntimeError):
    """Raised when the inputs are not good enough to trade on."""


def load_live_prices(
    cfg: Config,
    tickers: Optional[List[str]] = None,
    fetch_universe: bool = True,
    refresh: bool = True,
    extra: Optional[List[str]] = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Download the ranking universe plus the safe asset as one wide frame.

    The two are loaded by different paths, mirroring `pipeline.run()`: the
    ~100 constituents come from the batched universe cache, and the safe asset
    from its own per-ticker cache. That split matters -- the universe cache
    holds NDX members only, so asking it for BOXX returns a frame silently
    missing the cash leg, and every weight downstream would be wrong.

    `refresh=True` by default: a live run must not rank on last week's file.
    """
    if tickers is None:
        tickers = load_universe(fetch=fetch_universe, warn=False)
    safe = cfg.momentum.safe_asset

    wanted = sorted((set(tickers) | set(extra or [])) - {safe})
    uni = load_universe_prices(wanted, start=cfg.download_start, end=None,
                               refresh=refresh, verbose=False)

    core = load_prices([safe], start=cfg.download_start, end=None,
                       refresh=refresh, offline=offline)
    if safe not in core.columns or core[safe].dropna().empty:
        raise SignalError(f"could not load any prices for the safe asset {safe}")

    px = uni.copy()
    px.index = pd.to_datetime(px.index)
    core.index = pd.to_datetime(core.index)
    # Union the calendars, then forward-fill: a name that did not print on a
    # day the rest of the market did must not truncate the whole frame.
    px = px.reindex(px.index.union(core.index)).ffill()
    px[safe] = core[safe].reindex(px.index).ffill()
    return px.sort_index()


def check_data_quality(
    px: pd.DataFrame,
    requested: List[str],
    safe_asset: str,
    max_staleness_days: int,
    min_coverage: float,
    now: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """Refuse to trade on inputs that look broken. Returns a diagnostics dict.

    Every check here is a "this is more likely a data bug than a signal" test.
    A half-downloaded universe does not produce a *slightly* wrong ranking, it
    produces a confident ranking of the wrong candidate set -- which is far
    more dangerous than no ranking at all.
    """
    if px.empty:
        raise SignalError("price frame is empty")

    now = pd.Timestamp(now or pd.Timestamp.utcnow().normalize())
    last = pd.Timestamp(px.index.max())
    staleness = int(np.busday_count(last.date(), now.date()))
    if staleness > max_staleness_days:
        raise SignalError(
            f"price data ends {last:%Y-%m-%d}, {staleness} business days before "
            f"{now:%Y-%m-%d} (limit {max_staleness_days}). Refusing to trade on stale prices.")

    if safe_asset not in px.columns:
        raise SignalError(f"safe asset {safe_asset} missing from the price frame")
    if px[safe_asset].dropna().empty:
        raise SignalError(f"safe asset {safe_asset} has no prices")

    have = [t for t in requested if t in px.columns and px[t].notna().sum() > 0]
    coverage = len(have) / max(1, len(requested))
    if coverage < min_coverage:
        missing = sorted(set(requested) - set(have))
        raise SignalError(
            f"only {len(have)}/{len(requested)} tickers ({coverage:.0%}) have prices, "
            f"below the {min_coverage:.0%} floor. Missing: {missing[:12]}"
            f"{' ...' if len(missing) > 12 else ''}")

    # A row where most of the universe is NaN means a partial last bar.
    last_row_cover = float(px.loc[last, have].notna().mean())
    if last_row_cover < min_coverage:
        raise SignalError(
            f"the last bar ({last:%Y-%m-%d}) has prices for only {last_row_cover:.0%} "
            f"of the universe -- looks like a partial download")

    return {
        "last_bar": last,
        "staleness_days": staleness,
        "coverage": coverage,
        "last_row_coverage": last_row_cover,
        "n_tickers": len(have),
        "n_rows": len(px),
    }



def _selection_rows(mom, asof: pd.Timestamp, held: List[str]) -> List[Dict]:
    """Today's stock-selection decisions, as structured rows.

    Three event types, and the boring one matters most. `entry` and `exit` are
    what changed; `hold` is every name that survived, with the rank it survived
    at. Without the holds you cannot answer "how close was that name to being
    dropped" after the fact, which is the question you actually have when a
    position turns out badly.

    Rank and score come from the strategy itself -- entries and exits from its
    event rows, holds from the rank map it records on each rebalance. Nothing
    is recomputed here: doing so would mean reimplementing the eligibility and
    absolute-momentum filters, and a copy of that logic drifting out of sync is
    exactly the failure this whole live layer exists to prevent.
    """
    rows: List[Dict] = []
    hurdle = None
    if "safe_momentum" in getattr(mom, "diagnostics", pd.DataFrame()).columns:
        v = mom.diagnostics.loc[asof, "safe_momentum"]
        hurdle = None if pd.isna(v) else float(v)

    ev = mom.events
    today = (ev[ev["date"] == asof] if ev is not None and not ev.empty
             else pd.DataFrame())

    def _num(v):
        return None if v is None or pd.isna(v) else float(v)

    changed = set()
    for _, r in today.iterrows():
        sym = str(r["asset"])
        changed.add(sym)
        rows.append(dict(
            symbol=sym,
            event="entry" if r["action"] == "buy" else "exit",
            rank=None if _num(r.get("rank")) is None else int(r["rank"]),
            score=_num(r.get("score")),
            hurdle=hurdle,
            reason=str(r.get("reason", "")),
        ))

    # Everything still held that did not change today, at the rank the
    # strategy actually ranked it -- not a re-derived one.
    scores = mom.momentum.loc[asof] if mom.momentum is not None else None
    ranks = (mom.held_ranks or {}).get(asof, {})
    for sym in held:
        if sym in changed:
            continue
        r = _num(ranks.get(sym))
        rows.append(dict(
            symbol=sym, event="hold",
            rank=None if r is None else int(r),
            score=_num(scores.get(sym)) if scores is not None else None,
            hurdle=hurdle,
            reason="still within the exit band",
        ))
    return rows


def compute_targets(
    cfg: Config,
    prices: pd.DataFrame,
    requested: Optional[List[str]] = None,
    max_staleness_days: int = 5,
    min_coverage: float = 0.85,
    now: Optional[pd.Timestamp] = None,
) -> TargetBook:
    """Run the real strategy over the real history and return today's last row.

    This is the whole live signal path. It calls exactly the two functions the
    backtest calls, in the same order, with the same config object.
    """
    safe = cfg.momentum.safe_asset
    requested = requested or [c for c in prices.columns if c != safe]

    diag = check_data_quality(prices, requested, safe, max_staleness_days,
                              min_coverage, now=now)

    uni = prices.drop(columns=[safe])
    # Same pruning the pipeline does: a name without enough history is not
    # rankable, and leaving it in as a NaN column shrinks the candidate pool.
    uni = uni.loc[:, uni.notna().sum() >= cfg.momentum.min_history]
    if uni.shape[1] < cfg.momentum.n_hold:
        raise SignalError(
            f"only {uni.shape[1]} names have the {cfg.momentum.min_history} days of "
            f"history the ranker needs; cannot fill {cfg.momentum.n_hold} slots")

    mom = cross_sectional_momentum(uni, prices[safe], cfg.momentum)

    combined = uni.copy()
    combined[safe] = prices[safe]
    vt = book_vol_target(mom, combined, cfg.book_vol, lag=cfg.execution_lag)

    asof = vt.weights.index[-1]
    row = vt.weights.loc[asof]
    weights = {t: float(w) for t, w in row.items() if abs(float(w)) > 1e-9}

    vdiag = vt.diagnostics.loc[asof]
    held = list((mom.holdings_log or {}).get(asof, []))
    selection = _selection_rows(mom, asof, held)

    universe = sorted(set(uni.columns) | {safe})

    last_px = prices.loc[asof]
    # Priced across the whole universe, not just today's targets: an exit needs
    # a price to be logged with a real notional, and the name being exited is
    # by definition not in `weights`.
    px_map = {t: float(last_px[t]) for t in universe if t in last_px.index
              and not pd.isna(last_px[t])}
    missing_px = sorted(set(weights) - set(px_map))
    if missing_px:
        raise SignalError(f"no price on {asof:%Y-%m-%d} for target holdings: {missing_px}")

    book = TargetBook(
        asof=asof,
        weights=weights,
        prices=px_map,
        scalar=float(vdiag["scalar"]),
        book_vol=float(vdiag["book_vol"]) if not pd.isna(vdiag["book_vol"]) else float("nan"),
        raw_holdings=held,
        universe=universe,
        n_rankable=int(mom.diagnostics.loc[asof, "n_rankable"]),
        universe_size=uni.shape[1],
        diagnostics={k: v for k, v in diag.items()},
        selection=selection,
    )

    total = sum(book.weights.values())
    if not 0.99 <= total <= 1.01:
        raise SignalError(f"target weights sum to {total:.4f}, expected 1.0")

    log.info("signal %s", book.describe().replace("\n", " | "))
    return book
