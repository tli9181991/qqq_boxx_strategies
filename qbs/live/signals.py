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
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config, SAFE_ASSET
from ..strategies import (book_vol_target, cross_sectional_momentum, drawdown_stop,
                          residual_momentum)
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
    halted: bool = False               # the drawdown breaker is holding the book flat
    book_drawdown: float = 0.0         # the book's own drawdown, acted on or not
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
    ranking: List[Dict] = field(default_factory=list)       # the whole day's ranking
    # What candidate ranking rules would be holding today. Logged, never
    # traded: no order builder reads this, and nothing downstream of it can.
    shadow: List[Dict] = field(default_factory=list)
    # Where a watched non-constituent would have ranked. Same guarantee.
    watchlist: List[Dict] = field(default_factory=list)
    # Per-strategy targets are retained even though IB sees their aggregate.
    # This makes an overlap explicit: $6k MRVL in each sleeve becomes a $12k
    # broker target while the residual sleeve is funded by reducing BOXX.
    strategy_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    strategy_holdings: Dict[str, List[str]] = field(default_factory=dict)
    strategy_notionals: Dict[str, float] = field(default_factory=dict)
    strategy_daily_returns: Dict[str, float] = field(default_factory=dict)

    @property
    def risk_weight(self) -> float:
        return float(sum(w for t, w in self.weights.items() if t != SAFE_ASSET))

    def describe(self) -> str:
        held = ", ".join(f"{t} {self.weights[t]:.1%}"
                         for t in self.raw_holdings if self.weights.get(t, 0) > 0)
        extra = ""
        if self.strategy_holdings.get("resmom"):
            extra = ("\n  residual: "
                     + ", ".join(self.strategy_holdings["resmom"]))
        return (f"{self.asof:%Y-%m-%d}  book {self.risk_weight:.0%} "
                f"(scalar {self.scalar:.2f}, book vol {self.book_vol:.0%})\n"
                f"  holdings: {held or '(none)'}\n"
                f"  {SAFE_ASSET}: {self.weights.get(SAFE_ASSET, 0.0):.1%}"
                f"{extra}")


class SignalError(RuntimeError):
    """Raised when the inputs are not good enough to trade on."""


def load_live_prices(
    cfg: Config,
    tickers: Optional[List[str]] = None,
    fetch_universe: bool = True,
    refresh: bool = True,
    extra: Optional[List[str]] = None,
    watch: Optional[List[str]] = None,
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

    # The benchmark is not an index constituent, so it comes down the
    # per-ticker path beside the safe asset. Routed through the universe cache
    # it would be dropped without a word, exactly as BOXX once was.
    per_ticker = [safe]
    bench = cfg.dd_stop_benchmark
    if cfg.dd_stop.enabled and cfg.dd_stop.qqq_drawdown and bench not in per_ticker:
        per_ticker.append(bench)
    # Watched names come down this path for the same reason the benchmark does:
    # they are not index constituents, and the universe cache returns only what
    # it already holds on a cache hit, so a watch name routed through it would
    # vanish without a word on every run that did not refresh.
    watch_wanted = [t for t in (watch or []) if t not in per_ticker and t not in wanted]
    per_ticker.extend(watch_wanted)

    core = load_prices(per_ticker, start=cfg.download_start, end=None,
                       refresh=refresh, offline=offline)
    # A watch name that will not download is a log line, never a failed run.
    for t in list(watch_wanted):
        if t not in core.columns or core[t].dropna().empty:
            log.warning("no prices for watched name %s; skipping it today", t)
            per_ticker.remove(t)
    if safe not in core.columns or core[safe].dropna().empty:
        raise SignalError(f"could not load any prices for the safe asset {safe}")
    if len(per_ticker) > 1 and (bench not in core.columns
                                or core[bench].dropna().empty):
        raise SignalError(
            f"the drawdown stop needs {bench} but no prices came back for it. "
            "Set dd_stop.qqq_drawdown to 0 to drop the benchmark leg -- it is "
            "redundant against the book's own drawdown.")

    px = uni.copy()
    px.index = pd.to_datetime(px.index)
    core.index = pd.to_datetime(core.index)
    # Union the calendars, then forward-fill: a name that did not print on a
    # day the rest of the market did must not truncate the whole frame.
    px = px.reindex(px.index.union(core.index)).ffill()
    for t in per_ticker:
        px[t] = core[t].reindex(px.index).ffill()
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
    exclude: Optional[List[str]] = None,
    record_ranks: int = 25,
    shadow_weights: Sequence[float] = (),
    watch_names: Sequence[str] = (),
    base_notional: float = 1.0,
    residual_notional: float = 0.0,
) -> TargetBook:
    """Run the real strategy over the real history and return today's last row.

    This is the whole live signal path. It calls exactly the two functions the
    backtest calls, in the same order, with the same config object.
    """
    safe = cfg.momentum.safe_asset
    requested = requested or [c for c in prices.columns if c != safe]

    diag = check_data_quality(prices, requested, safe, max_staleness_days,
                              min_coverage, now=now)

    # The rankable universe is what was *requested*, not whatever happens to be
    # in the frame. The frame also carries the safe asset and, when the
    # drawdown stop is on, the benchmark -- and "everything except the safe
    # asset" quietly made QQQ a candidate the book could buy. It never came
    # close (its best 6-1 rank in six years is 22, against an exit_rank of 8,
    # because an index of a hundred names cannot out-momentum its own top six),
    # but the live path was ranking a set the backtest never saw, and any name
    # added to the frame for observation would have inherited the same right to
    # take a slot.
    uni = prices[[c for c in prices.columns if c in set(requested) and c != safe]]
    # Same pruning the pipeline does: a name without enough history is not
    # rankable, and leaving it in as a NaN column shrinks the candidate pool.
    uni = uni.loc[:, uni.notna().sum() >= cfg.momentum.min_history]

    # `tradeable` is fixed before exclusions and is what the order builder gets.
    # An excluded name the strategy still holds has to remain sellable: drop it
    # from the *tradeable* set as well and the order builder reads the position
    # as somebody else's and never closes it, which is the exact bug the
    # universe field exists to prevent.
    tradeable = sorted(set(uni.columns) | {safe})

    wanted_out = {str(t).strip().upper() for t in (exclude or [])} - {""}
    dropped = sorted(wanted_out & set(uni.columns))
    unmatched = sorted(wanted_out - set(uni.columns))
    if unmatched:
        # Usually harmless -- a name you hold that was never in the index, so
        # there was nothing to exclude. Occasionally a typo, which would
        # otherwise show up only as the book still holding the name.
        log.info("exclusions that match nothing in the ranking universe: %s",
                 ", ".join(unmatched))
    if dropped:
        # Skipped names are not replaced by nothing -- the ranker simply fills
        # the slot with the next name down, which is the point.
        log.info("excluded from ranking (held outside the strategy): %s",
                 ", ".join(dropped))
        uni = uni.drop(columns=dropped)

    if uni.shape[1] < cfg.momentum.n_hold:
        raise SignalError(
            f"only {uni.shape[1]} names have the {cfg.momentum.min_history} days of "
            f"history the ranker needs{' after exclusions' if dropped else ''}; "
            f"cannot fill {cfg.momentum.n_hold} slots")

    # The dashboard's momentum-leader screen, when configured. No volume is
    # available on this path, which skips that leg and makes the set slightly
    # larger -- at 300k shares it is close to non-binding on the Nasdaq-100
    # anyway, where every constituent trades far above it.
    leaders = None
    if cfg.use_leader_filter:
        from ..breadth import leader_eligibility
        leaders = leader_eligibility(uni, volumes=None)
        n_lead = int(leaders.iloc[-1].sum())
        log.info("momentum-leader filter: %d of %d names pass", n_lead, uni.shape[1])
        if n_lead < cfg.momentum.n_hold:
            log.warning("only %d leaders for %d slots; the rest goes to cash",
                        n_lead, cfg.momentum.n_hold)

    mom = cross_sectional_momentum(uni, prices[safe], cfg.momentum,
                                   eligible=leaders, record_ranks=record_ranks)

    combined = uni.copy()
    combined[safe] = prices[safe]
    vt = book_vol_target(mom, combined, cfg.book_vol, lag=cfg.execution_lag)

    # The drawdown breaker, off unless configured on. It reads the book's own
    # equity over the whole history, which this path already recomputes every
    # run -- so it needs no stored state and a missed session cannot desync it.
    wants_bench = cfg.dd_stop.enabled and bool(cfg.dd_stop.qqq_drawdown)
    bench = (prices[cfg.dd_stop_benchmark]
             if wants_bench and cfg.dd_stop_benchmark in prices.columns else None)
    if wants_bench and bench is None:
        raise SignalError(
            f"dd_stop.qqq_drawdown is set but {cfg.dd_stop_benchmark!r} is not in the "
            "price frame. Add it to extra_tickers, or set qqq_drawdown to 0 -- it was "
            "measured redundant against the book's own drawdown.")
    vt = drawdown_stop(vt, combined, cfg.dd_stop, lag=cfg.execution_lag,
                       benchmark=bench)

    asof = vt.weights.index[-1]
    row = vt.weights.loc[asof]
    momentum_weights = {t: float(w) for t, w in row.items()
                        if abs(float(w)) > 1e-9}
    residual_weights: Dict[str, float] = {}
    residual_held: List[str] = []
    residual_sig = None
    if residual_notional > 0:
        if cfg.resmom.n_hold != 6:
            raise SignalError("the live residual sleeve must have exactly six slots")
        market = prices.get(cfg.resmom.market_asset)
        if market is None:
            raise SignalError(
                f"the residual sleeve needs market series {cfg.resmom.market_asset}")
        residual_sig = residual_momentum(uni, prices[safe], market, cfg.resmom)
        rrow = residual_sig.weights.loc[asof]
        residual_weights = {t: float(w) for t, w in rrow.items()
                            if abs(float(w)) > 1e-9}
        residual_held = list((residual_sig.holdings_log or {}).get(asof, []))

    if base_notional <= 0:
        raise SignalError("base_notional must be positive")
    if residual_notional < 0:
        raise SignalError("residual_notional must not be negative")
    if residual_notional > base_notional:
        raise SignalError("residual_notional cannot exceed the aggregate book")
    target_dollars: Dict[str, float] = {}
    for ticker, weight in momentum_weights.items():
        target_dollars[ticker] = target_dollars.get(ticker, 0.0) + weight * base_notional
    available_safe = target_dollars.get(safe, 0.0)
    if residual_notional > available_safe + 1e-6:
        raise SignalError(
            f"residual sleeve needs ${residual_notional:,.0f} from {safe}, but the "
            f"main strategy currently allocates only ${available_safe:,.0f}. "
            "Refusing to add leverage; reduce QBS_RESMOM_NOTIONAL or wait for a "
            "larger safe-asset allocation.")
    target_dollars[safe] = available_safe - residual_notional
    for ticker, weight in residual_weights.items():
        target_dollars[ticker] = (target_dollars.get(ticker, 0.0)
                                  + weight * residual_notional)
    weights = {t: dollars / base_notional for t, dollars in target_dollars.items()
               if abs(dollars) > 1e-9}

    def _last_net_return(signal) -> float:
        held_weights = signal.weights.reindex(combined.index).ffill().shift(
            cfg.execution_lag).fillna(0.0)
        asset_returns = combined.pct_change().fillna(0.0)
        gross = (held_weights * asset_returns[held_weights.columns]).sum(axis=1)
        turnover = held_weights.diff().abs().sum(axis=1)
        cost_rate = (cfg.cost_bps + cfg.slippage_bps) / 1e4
        return float(gross.loc[asof] - turnover.loc[asof] * cost_rate)

    daily_returns = {"momentum": _last_net_return(vt)}
    if residual_sig is not None:
        daily_returns["resmom"] = _last_net_return(residual_sig)

    vdiag = vt.diagnostics.loc[asof]
    halted = bool(vt.diagnostics.get("blocked", pd.Series(dtype=float)).get(asof, 0.0))
    book_dd = float(vt.diagnostics.get("book_drawdown", pd.Series(dtype=float))
                    .get(asof, 0.0))
    held = list((mom.holdings_log or {}).get(asof, []))
    selection = _selection_rows(mom, asof, held)
    ranking = [
        dict(symbol=t, rank=r, score=sc, held=t in held)
        for t, r, sc in (mom.rank_log or {}).get(asof, [])
    ]

    # Scored after the real book is decided, from the same pruned universe, and
    # guarded: a candidate that cannot be scored costs a log line, never a run.
    shadow: List[Dict] = []
    if shadow_weights:
        from ..shadow import shadow_books
        try:
            shadow = shadow_books(uni, prices[safe], cfg.momentum,
                                  weights=list(shadow_weights), asof=asof)
        except Exception as exc:          # noqa: BLE001
            log.warning("shadow books could not be scored (%s: %s); the live book "
                        "is unaffected", type(exc).__name__, exc)

    watchlist: List[Dict] = []
    # Constituents stay on the list: a watchlist is a list, and dropping the
    # names that happen to be in the index would look like the feature failing
    # on half of them. They are reported at their standing rank rather than
    # interpolated into one.
    wanted_watch = [t for t in watch_names if t in prices.columns]
    if wanted_watch:
        from ..shadow import watchlist_rows
        watchlist = watchlist_rows(uni, prices[safe], prices[wanted_watch],
                                   cfg.momentum, asof=asof)
        for r in watchlist:
            # A cutoff is blank when that slot is unfilled, and "needs nan%"
            # would read as a broken number rather than an empty seat.
            def _cut(v, label):
                return f"{label} {100 * v:+.1f}%" if v == v else f"{label} unfilled"
            place = (f"rank {r['rank']:.0f}" if r["rank"] == r["rank"]
                     else "unranked (lost to the safe asset, or too little history)")
            log.info("watch %s: %s%s, 6-1 momentum %+.1f%% (%s, %s)",
                     r["symbol"], place,
                     "" if r["constituent"] else " if it were a constituent",
                     100 * r["score"],
                     _cut(r["book_cutoff"], "book needs"),
                     _cut(r["band_cutoff"], "band"))

    universe = tradeable

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
        halted=halted,
        book_drawdown=book_dd,
        book_vol=float(vdiag["book_vol"]) if not pd.isna(vdiag["book_vol"]) else float("nan"),
        raw_holdings=held,
        universe=universe,
        n_rankable=int(mom.diagnostics.loc[asof, "n_rankable"]),
        universe_size=uni.shape[1],
        diagnostics={k: v for k, v in diag.items()},
        selection=selection,
        ranking=ranking,
        shadow=shadow,
        watchlist=watchlist,
        strategy_weights={"momentum": momentum_weights,
                          **({"resmom": residual_weights} if residual_weights else {})},
        strategy_holdings={"momentum": held,
                           **({"resmom": residual_held} if residual_weights else {})},
        strategy_notionals={"momentum": base_notional,
                            **({"resmom": residual_notional} if residual_weights else {})},
        strategy_daily_returns=daily_returns,
    )

    total = sum(book.weights.values())
    if not 0.99 <= total <= 1.01:
        raise SignalError(f"target weights sum to {total:.4f}, expected 1.0")

    log.info("signal %s", book.describe().replace("\n", " | "))
    return book
