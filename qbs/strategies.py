"""The strategies.

Each builder takes prices and parameters and returns a `StrategySignals`:

    weights      target weights DECIDED AT THE CLOSE of each date
    diagnostics  the indicator series the chart layer draws
    events       discrete buy/sell rows, for markers on the price chart

Nothing here applies the execution lag. The engine does that in one place
(`engine.run_backtest`), so a weight row here always means "what I decided
looking at today's close", and it is impossible to accidentally trade on
information you did not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    BookVolTargetParams, EXECUTION_LAG, GEMParams, MomentumParams, RSI2Params,
    SAFE_ASSET, TRADING_DAYS, VixBreakerParams, VolTargetParams,
)
from .indicators import realized_vol, sma, total_return, wilder_rsi


@dataclass
class StrategySignals:
    name: str
    weights: pd.DataFrame
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: Dict = field(default_factory=dict)
    holding: Optional[pd.Series] = None       # which asset(s) held on each day
    holdings_log: Optional[Dict] = None       # momentum: date -> list of tickers
    momentum: Optional[pd.DataFrame] = None   # momentum: the ranking scores
    held_ranks: Optional[Dict] = None         # momentum: date -> {ticker: rank}

    def exposure(self, asset: str) -> pd.Series:
        if asset not in self.weights.columns:
            return pd.Series(0.0, index=self.weights.index)
        return self.weights[asset]


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "action", "asset", "price", "reason",
                                 "rank", "score"])


# ==========================================================================
# 1. Larry Connors RSI(2)
# ==========================================================================

def connors_rsi2(
    prices: pd.DataFrame,
    risk_asset: str = "QQQ",
    safe_asset: str = SAFE_ASSET,
    params: Optional[RSI2Params] = None,
) -> StrategySignals:
    """Connors' RSI(2) mean reversion, long-only, with cash parked in BOXX.

    The rules, in the order they are evaluated each day:

    1. Trend filter -- no long is opened unless close > SMA(200). Connors is
       explicit that RSI(2) is a *pullback* system, not a falling-knife system;
       without this filter it buys every leg of a bear market.
    2. Entry -- RSI(2) closes below `entry_threshold` (default 5).
    3. Exit, whichever comes first:
         a. close crosses back above SMA(5)  (Connors' default exit)
         b. RSI(2) closes above `exit_rsi`
         c. `max_hold_days` elapse -- a time stop, so a trade that neither
            recovers nor triggers cannot sit open indefinitely.

    A state machine rather than vectorised boolean masks: entry and exit are
    path-dependent (you cannot exit a position you never opened), and the day
    count in rule 3c has no vectorised form.
    """
    p = params or RSI2Params()
    close = prices[risk_asset].astype(float)

    rsi = wilder_rsi(close, p.rsi_period)
    fast = sma(close, p.exit_sma)

    if p.trend_sma and p.trend_sma > 1:
        trend = sma(close, p.trend_sma)
        above_trend = close > trend
    else:
        # trend_sma of 0 or 1 disables the filter. SMA(1) is the close itself,
        # so `close > SMA(1)` would be False on every bar and the strategy would
        # silently never trade -- exactly the kind of no-op a parameter sweep
        # would report as "the filter is essential".
        trend = pd.Series(np.nan, index=close.index, name="SMA_off")
        above_trend = pd.Series(True, index=close.index)

    idx = close.index
    in_pos = np.zeros(len(idx), dtype=bool)
    held = 0
    position = False
    events: List[Dict] = []

    filter_on = bool(p.trend_sma and p.trend_sma > 1)
    for i in range(len(idx)):
        r, c, f, t = rsi.iat[i], close.iat[i], fast.iat[i], trend.iat[i]
        indicators_ready = not np.isnan(r) and (not filter_on or not np.isnan(t))

        if position:
            held += 1
            exit_reason = None
            if p.use_sma_exit and not np.isnan(f) and c > f:
                exit_reason = f"close > SMA({p.exit_sma})"
            elif not np.isnan(r) and r > p.exit_rsi:
                exit_reason = f"RSI({p.rsi_period}) > {p.exit_rsi:g}"
            elif held >= p.max_hold_days:
                exit_reason = f"time stop ({p.max_hold_days}d)"

            if exit_reason:
                position = False
                held = 0
                events.append(dict(date=idx[i], action="sell", asset=risk_asset,
                                   price=float(c), reason=exit_reason))
        elif indicators_ready and above_trend.iat[i] and r < p.entry_threshold:
            position = True
            held = 0
            events.append(dict(date=idx[i], action="buy", asset=risk_asset,
                               price=float(c),
                               reason=f"RSI({p.rsi_period})={r:.1f} < {p.entry_threshold:g}, above SMA({p.trend_sma})"))

        in_pos[i] = position

    risk_w = pd.Series(in_pos.astype(float), index=idx, name=risk_asset)
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights[risk_asset] = risk_w
    if p.park_in_safe and safe_asset in weights.columns:
        weights[safe_asset] = 1.0 - risk_w

    diagnostics = pd.DataFrame({
        "close": close,
        "rsi": rsi,
        f"sma{p.trend_sma}": trend,
        f"sma{p.exit_sma}": fast,
        "above_trend": above_trend.astype(float),
    })

    ev = pd.DataFrame(events) if events else _empty_events()
    return StrategySignals("rsi2", weights, diagnostics, ev, params=p.__dict__.copy())


# ==========================================================================
# 2. GEM -- Global Equities Momentum (Antonacci)
# ==========================================================================

def gem(
    prices: pd.DataFrame,
    params: Optional[GEMParams] = None,
) -> StrategySignals:
    """Dual momentum: relative momentum picks the sleeve, absolute momentum gates it.

    Evaluated on month-end closes only:

    * Relative -- compare the trailing `lookback_months` total return of each
      risk asset (QQQ vs VEU) and take the leader.
    * Absolute -- compare that leader against the safe asset's own trailing
      return. Only if the leader wins does the portfolio hold equity;
      otherwise the whole book sits in BOXX.

    Using BOXX's realised return as the absolute-momentum hurdle (rather than a
    fixed 0%) is the honest comparison: it asks "did equities beat the cash I
    could actually have held?", and in a 4-5% T-bill world those are very
    different questions.

    The holding decided at month-end M applies for all of month M+1. The
    engine's execution lag then shifts it one more day, so the first day of the
    new month still trades on the old weight -- which is what happens in
    practice when you rebalance at the next open.
    """
    p = params or GEMParams()
    risk_assets = [a for a in p.risk_assets if a in prices.columns]
    if not risk_assets:
        raise ValueError(f"none of {p.risk_assets} present in prices")
    if p.safe_asset not in prices.columns:
        raise ValueError(f"safe asset {p.safe_asset} missing from prices")

    universe = risk_assets + [p.safe_asset]
    monthly = prices[universe].resample(p.rebalance).last()
    mom = monthly.apply(lambda s: total_return(s, p.lookback_months))

    holdings: Dict[pd.Timestamp, str] = {}
    scores: List[Dict] = []
    for dt, row in mom.iterrows():
        if row[universe].isna().any():
            continue
        leader = row[risk_assets].idxmax()
        safe_ret = row[p.safe_asset]
        pick = leader if row[leader] > safe_ret else p.safe_asset
        holdings[dt] = pick
        scores.append(dict(date=dt, leader=leader, pick=pick,
                           **{f"mom_{a}": float(row[a]) for a in universe}))

    if not holdings:
        raise ValueError(
            "GEM produced no decisions -- not enough history for the "
            f"{p.lookback_months}-month lookback. Extend DOWNLOAD_START."
        )

    hold_s = pd.Series(holdings).sort_index()
    # Month-end decision -> effective from the next trading day onward.
    daily_pick = hold_s.reindex(prices.index.union(hold_s.index)).ffill().reindex(prices.index)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for asset in universe:
        weights[asset] = (daily_pick == asset).astype(float)

    diagnostics = pd.DataFrame(scores).set_index("date") if scores else pd.DataFrame()

    # Events: one row per change of holding.
    changed = hold_s.ne(hold_s.shift())
    ev_rows = []
    for dt, pick in hold_s[changed].items():
        prev = hold_s.shift().get(dt)
        px_at = prices.loc[:dt]
        price = float(px_at[pick].iloc[-1]) if len(px_at) else np.nan
        ev_rows.append(dict(
            date=dt, action="rotate", asset=pick, price=price,
            reason=f"{'start' if pd.isna(prev) else prev} -> {pick}",
        ))
    ev = pd.DataFrame(ev_rows) if ev_rows else _empty_events()

    sig = StrategySignals("gem", weights, diagnostics, ev, params=p.__dict__.copy())
    sig.holding = daily_pick  # convenience for the regime-shading chart
    return sig


# ==========================================================================
# 3. Volatility-targeting overlay
# ==========================================================================

def vol_target_overlay(
    prices: pd.DataFrame,
    base_weights: Optional[pd.DataFrame] = None,
    risk_asset: str = "QQQ",
    safe_asset: str = SAFE_ASSET,
    params: Optional[VolTargetParams] = None,
    name: str = "voltarget",
) -> StrategySignals:
    """Scale risk exposure so forecast volatility sits near `target_vol`.

    weight = target_vol / forecast_vol, clipped to [min_weight, max_weight].

    Two details that matter more than the formula:

    * **Rebalance band.** The raw ratio wiggles every single day. Trading that
      wiggle burns the edge in costs for no risk benefit, so the position only
      moves when the target has drifted `rebalance_band` away from what is held.
    * **Vol floor.** In a dead-calm tape the ratio explodes. `vol_floor` caps
      how small the denominator may get; `max_weight` catches whatever is left.

    Passing `base_weights` turns this into an *overlay* on another strategy:
    that strategy's risk-asset weight is multiplied by the vol scalar, so the
    overlay can only ever reduce (or, with max_weight > 1, gear) an exposure
    the base strategy already wanted. With `base_weights=None` it runs
    standalone on `risk_asset`, which is the third strategy in its own right.
    """
    p = params or VolTargetParams()
    rets = prices[risk_asset].pct_change()
    vol = realized_vol(rets, halflife=p.halflife, min_periods=p.min_periods)
    vol_used = vol.clip(lower=p.vol_floor)

    raw = (p.target_vol / vol_used).clip(lower=p.min_weight, upper=p.max_weight)

    # Apply the no-trade band: walk forward holding the last weight until the
    # target has moved far enough to be worth the turnover.
    target = np.full(len(raw), np.nan)
    current = np.nan
    raw_v = raw.to_numpy()
    for i in range(len(raw_v)):
        r = raw_v[i]
        if np.isnan(r):
            continue
        if np.isnan(current) or abs(r - current) >= p.rebalance_band:
            current = r
        target[i] = current
    scalar = pd.Series(target, index=raw.index, name="vol_scalar").fillna(0.0)

    if base_weights is None:
        risk_w = scalar
    else:
        base = base_weights.reindex(prices.index).ffill().fillna(0.0)
        risk_w = (base[risk_asset] * scalar).clip(lower=0.0, upper=p.max_weight)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[risk_asset] = risk_w
    if safe_asset in weights.columns:
        weights[safe_asset] = 1.0 - risk_w

    diagnostics = pd.DataFrame({
        "close": prices[risk_asset],
        "realized_vol": vol,
        "raw_weight": raw,
        "target_weight": scalar,
        "risk_weight": risk_w,
    })

    # Events: a marker every time the traded weight actually moves.
    moved = risk_w.diff().abs() > 1e-9
    ev_rows = []
    for dt in risk_w.index[moved.fillna(False)]:
        prev = float(risk_w.shift().loc[dt])
        now = float(risk_w.loc[dt])
        ev_rows.append(dict(
            date=dt,
            action="buy" if now > prev else "sell",
            asset=risk_asset,
            price=float(prices[risk_asset].loc[dt]),
            reason=f"exposure {prev:.0%} -> {now:.0%} (vol {vol.loc[dt]:.1%})",
        ))
    ev = pd.DataFrame(ev_rows) if ev_rows else _empty_events()

    return StrategySignals(name, weights, diagnostics, ev, params=p.__dict__.copy())


# ==========================================================================
# 4. Top-N cross-sectional momentum (Nasdaq-100) with a hysteresis band
# ==========================================================================

def cross_sectional_momentum(
    universe_prices: pd.DataFrame,
    safe_prices: pd.Series,
    params: Optional[MomentumParams] = None,
    eligible: Optional[pd.DataFrame] = None,
    name: str = "momentum",
) -> StrategySignals:
    """Rank the universe by 12-1 momentum, hold the top N, exit on a band.

    Parameters
    ----------
    universe_prices : date x ticker adjusted closes for the ranking universe.
    safe_prices     : the safe asset (BOXX) -- the cash leg, and the hurdle for
                      the absolute-momentum filter.
    eligible        : optional date x ticker boolean mask of index membership.
                      Pass `universe.membership_mask(...)` with point-in-time
                      data to remove survivorship bias; None means every column
                      is rankable on every date, which is the biased case.

    The momentum measure
    --------------------
    Return from `lookback_months` ago to `skip_months` ago -- the standard
    "12-1" construction. The most recent month is skipped because short-horizon
    returns *reverse* rather than persist; including them mixes two effects
    with opposite signs and blunts both.

    The band
    --------
    On each rebalance date, names already held survive while their rank is
    within `exit_rank`; freed slots go to the highest-ranked names not already
    held. This is what stops a name oscillating around rank 6 from being
    round-tripped every other week.

    Anything that cannot be filled -- too few eligible names, or (with
    `absolute_filter`) too few names beating the safe asset -- goes to cash in
    the safe asset rather than being force-allocated into a falling stock.
    """
    p = params or MomentumParams()

    px = universe_prices.sort_index()
    safe = safe_prices.reindex(px.index).ffill()
    safe.name = p.safe_asset

    look = int(round(p.lookback_months * 21))
    skip = int(round(p.skip_months * 21))
    if look <= skip:
        raise ValueError("lookback_months must exceed skip_months")

    # Momentum: price `skip` days ago over price `look` days ago, minus 1.
    # Both legs are strictly in the past at every row -- no look-ahead.
    lagged = px.shift(skip)
    mom = lagged / px.shift(look) - 1.0
    safe_mom = safe.shift(skip) / safe.shift(look) - 1.0

    # A name needs enough history before it can be ranked at all.
    history = px.notna().cumsum()
    rankable = mom.notna() & (history >= p.min_history)
    if eligible is not None:
        rankable &= eligible.reindex(index=px.index, columns=px.columns).fillna(False)

    if p.weighting == "inv_vol":
        vol = px.pct_change().rolling(p.vol_lookback, min_periods=p.vol_lookback // 2).std()
    else:
        vol = None

    # Rebalance calendar.
    if p.rebalance == "daily":
        rebal_dates = list(px.index)
    else:
        marks = px.index.to_series().resample(p.rebalance).last().dropna()
        rebal_dates = [d for d in marks if d in px.index]

    rebal_set = set(rebal_dates)
    assets = list(px.columns) + [p.safe_asset]
    weights = pd.DataFrame(0.0, index=px.index, columns=assets)

    held: List[str] = []
    events: List[Dict] = []
    holdings_log: Dict[pd.Timestamp, List[str]] = {}
    held_ranks: Dict[pd.Timestamp, Dict[str, float]] = {}
    n_cash_slots: Dict[pd.Timestamp, int] = {}

    for dt in px.index:
        if dt in rebal_set:
            row = mom.loc[dt]
            ok = rankable.loc[dt]
            cand = row[ok].dropna()

            if p.absolute_filter:
                hurdle = safe_mom.loc[dt]
                if not np.isnan(hurdle):
                    cand = cand[cand > hurdle]

            # Rank 1 = strongest. Names failing the absolute filter are simply
            # absent, so they rank as infinitely bad and will be dropped.
            order = cand.sort_values(ascending=False)
            rank = pd.Series(np.arange(1, len(order) + 1), index=order.index)

            keep = [t for t in held if rank.get(t, np.inf) <= p.exit_rank]
            for t in held:
                if t not in keep:
                    events.append(dict(
                        date=dt, action="sell", asset=t,
                        price=float(px.at[dt, t]) if t in px.columns else np.nan,
                        reason=(f"rank {rank.get(t, float('nan')):.0f} > {p.exit_rank}"
                                if t in rank.index else "no longer eligible"),
                        # Structured alongside the prose: the live run log stores
                        # these as columns, and parsing them back out of `reason`
                        # would break the first time the wording changed.
                        rank=float(rank.get(t, np.nan)),
                        score=float(row.get(t, np.nan)),
                    ))

            for t in order.index:
                if len(keep) >= p.n_hold:
                    break
                if t not in keep:
                    keep.append(t)
                    events.append(dict(
                        date=dt, action="buy", asset=t, price=float(px.at[dt, t]),
                        reason=f"rank {rank[t]:.0f}, 12-1 mom {order[t]:+.1%}",
                        rank=float(rank[t]),
                        score=float(order[t]),
                    ))
            held = keep
            # The rank each held name survived at, recorded where it is known
            # exactly. Recomputing this outside the loop would mean redoing the
            # eligibility and absolute-momentum filters, and a copy of that
            # logic is precisely what drifts.
            held_ranks[dt] = {t: float(rank.get(t, np.nan)) for t in held}

        holdings_log[dt] = list(held)
        n_cash_slots[dt] = p.n_hold - len(held)

        if held:
            # Weighting is per *slot*, not per held name: with only 4 of 6
            # slots filled the book is 4/6 invested and 2/6 in cash. Spreading
            # 100% across the survivors would quietly concentrate the portfolio
            # exactly when the fewest names were qualifying -- i.e. in a
            # deteriorating market, which is precisely backwards.
            invested = len(held) / p.n_hold
            if p.weighting == "inv_vol" and vol is not None:
                v = vol.loc[dt, held].replace(0.0, np.nan)
                inv = 1.0 / v
                w = (inv / inv.sum() * invested if inv.notna().any() and inv.sum() > 0
                     else pd.Series(1.0 / p.n_hold, index=held))
            else:
                w = pd.Series(1.0 / p.n_hold, index=held)
            weights.loc[dt, held] = w.reindex(held).fillna(0.0).to_numpy()

        risk_total = float(weights.loc[dt, px.columns].sum())
        weights.at[dt, p.safe_asset] = max(0.0, 1.0 - risk_total)

    diagnostics = pd.DataFrame({
        "n_held": pd.Series({d: len(v) for d, v in holdings_log.items()}),
        "cash_slots": pd.Series(n_cash_slots),
        "n_rankable": rankable.sum(axis=1),
        "safe_momentum": safe_mom,
    })

    ev = pd.DataFrame(events) if events else _empty_events()
    sig = StrategySignals(name, weights, diagnostics, ev, params=p.__dict__.copy())
    sig.holding = pd.Series({d: ",".join(v) for d, v in holdings_log.items()})
    sig.holdings_log = holdings_log
    sig.held_ranks = held_ranks
    sig.momentum = mom
    return sig


# ==========================================================================
# 5. VIX circuit breaker -- a risk-off wrapper for any base strategy
# ==========================================================================

def vix_circuit_breaker(
    base: StrategySignals,
    vix: pd.Series,
    params: Optional[VixBreakerParams] = None,
    name: Optional[str] = None,
) -> StrategySignals:
    """Lay a binary risk-off switch over `base`, driven by the VIX close.

    Three states, evaluated on each close:

    ============  =========================================  ==============
    state         transition                                 holds
    ============  =========================================  ==============
    INVESTED      VIX > exit_level                           the base weights
    CASH          park_after_days elapsed  -> PARKED         nothing (0%)
                  VIX < entry_level and min_cash_days met
                  -> INVESTED
    PARKED        VIX < entry_level -> INVESTED              the safe asset
    ============  =========================================  ==============

    Why two thresholds. `exit_level` above `entry_level` is a hysteresis band
    on volatility, the same device as the momentum rank band. With a single
    threshold, a VIX oscillating around it would round-trip the entire book
    every other day.

    Why CASH before PARKED. Buying the safe asset for a two-day scare costs
    two spreads for two days of T-bill yield, which is a losing trade. The
    breaker therefore sits in true cash first and only converts to the safe
    asset once the stress has actually persisted.

    Why a minimum dwell. `min_cash_days` stops the breaker re-entering on the
    very next bar after a one-day spike, which is precisely when the tape is
    least readable.

    Note what this is NOT: it is not volatility targeting. Vol targeting
    scales exposure continuously with forecast risk; this is on/off. The two
    compose -- run `vol_target_overlay` on the result if you want both.
    """
    p = params or VixBreakerParams()
    if p.entry_level > p.exit_level:
        raise ValueError("entry_level must be <= exit_level (the band cannot be inverted)")

    w_base = base.weights
    idx = w_base.index
    v = vix.reindex(idx).ffill()

    assets = list(w_base.columns)
    if p.safe_asset not in assets:
        assets = assets + [p.safe_asset]
    weights = pd.DataFrame(0.0, index=idx, columns=assets)

    risk_cols = [c for c in w_base.columns if c != p.safe_asset]

    state = "INVESTED"
    days_in_cash = 0
    states: List[str] = []
    events: List[Dict] = []

    for i, dt in enumerate(idx):
        vx = v.iat[i]
        known = not np.isnan(vx)

        if state == "INVESTED":
            if known and vx > p.exit_level:
                state, days_in_cash = "CASH", 0
                events.append(dict(date=dt, action="sell", asset="BOOK",
                                   price=float(vx),
                                   reason=f"VIX {vx:.1f} > {p.exit_level:g} -- to cash"))
        elif state == "CASH":
            days_in_cash += 1
            recovered = known and vx < p.entry_level
            if recovered and days_in_cash >= p.min_cash_days:
                state = "INVESTED"
                events.append(dict(date=dt, action="buy", asset="BOOK", price=float(vx),
                                   reason=f"VIX {vx:.1f} < {p.entry_level:g} after "
                                          f"{days_in_cash}d -- back in"))
            elif days_in_cash >= p.park_after_days:
                state = "PARKED"
                events.append(dict(date=dt, action="park", asset=p.safe_asset,
                                   price=float(vx),
                                   reason=f"still elevated after {days_in_cash}d -- "
                                          f"cash into {p.safe_asset}"))
        elif state == "PARKED":
            days_in_cash += 1
            if known and vx < p.entry_level:
                state = "INVESTED"
                events.append(dict(date=dt, action="buy", asset="BOOK", price=float(vx),
                                   reason=f"VIX {vx:.1f} < {p.entry_level:g} -- back in"))

        if state == "INVESTED":
            days_in_cash = 0
            weights.loc[dt, w_base.columns] = w_base.loc[dt].to_numpy()
        elif state == "PARKED":
            weights.at[dt, p.safe_asset] = 1.0
        else:
            # CASH: hold nothing. Unallocated weight earns exactly 0% in the
            # engine, which is what physical cash does.
            if not p.sell_safe_too and p.safe_asset in w_base.columns:
                weights.at[dt, p.safe_asset] = float(w_base.at[dt, p.safe_asset])

        states.append(state)

    regime = pd.Series(states, index=idx, name="regime")
    diagnostics = pd.DataFrame({
        "vix": v,
        "regime": regime,
        "invested": (regime == "INVESTED").astype(float),
        "base_risk_weight": w_base[risk_cols].sum(axis=1) if risk_cols else 0.0,
        "risk_weight": weights[[c for c in weights.columns if c != p.safe_asset]].sum(axis=1),
    })

    ev = pd.DataFrame(events) if events else _empty_events()
    sig = StrategySignals(name or f"{base.name}_vix", weights, diagnostics, ev,
                          params=p.__dict__.copy())
    sig.holding = regime
    sig.holdings_log = base.holdings_log
    sig.held_ranks = base.held_ranks
    sig.momentum = base.momentum
    return sig


# ==========================================================================
# 6. Book-level volatility targeting -- scale a whole portfolio by its own vol
# ==========================================================================

def book_vol_target(
    base: StrategySignals,
    prices: pd.DataFrame,
    params: Optional[BookVolTargetParams] = None,
    lag: int = EXECUTION_LAG,
    name: Optional[str] = None,
) -> StrategySignals:
    """Scale an entire book so its *own* realised volatility sits near target.

        scalar = target_vol / realised_book_vol,  clipped to [0, max_weight]

    Every risk weight is multiplied by that one scalar; the freed weight parks
    in the safe asset. Relative position sizes within the book are untouched --
    this changes how much of the strategy you hold, never which names.

    Why this and not a VIX breaker or an index trend filter
    -------------------------------------------------------
    Those are *index* signals, and a concentrated book has two distinct kinds
    of drawdown. One is a market selloff, which they can see. The other is the
    book's own holdings decoupling from a calm index -- on the Top-6 momentum
    strategy the worst such episode ran while VIX sat at 16-20, below its own
    median, and QQQ was barely moving. No threshold on VIX or on QQQ's SMA(200)
    reduces that drawdown at all, because the market was not what went wrong.

    The book's realised volatility rises in *both* cases, which is the whole
    argument for measuring the thing you actually hold.

    What it does not do
    -------------------
    It does not add return -- it rescales risk, so Sharpe is roughly unchanged
    and what improves is Calmar. It also cannot protect against an overnight
    gap in a single name: it responds to sustained volatility, not to jumps.

    Causality
    ---------
    The vol estimate at date `t` is built from returns the book had actually
    *earned* by the close of `t`: weights are shifted by `lag` before being
    multiplied by returns, exactly as `engine.run_backtest` does it. The
    resulting scalar then modifies the weight *decided* at `t`, which the
    engine lags again before trading it. Nothing here can see the future --
    `tests/test_qbs.py` pins this with a shuffled-future test.
    """
    p = params or BookVolTargetParams()
    safe = p.safe_asset

    w = base.weights.reindex(prices.index).ffill().fillna(0.0)
    if safe not in w.columns:
        w[safe] = 0.0
    risk_cols = [c for c in w.columns if c != safe]

    rets = prices.reindex(columns=w.columns).ffill().pct_change().fillna(0.0)

    # Realised book return, on the same convention the engine uses.
    book_ret = (w.shift(lag) * rets).sum(axis=1)
    vol = (book_ret.ewm(halflife=p.halflife, min_periods=p.min_periods).std()
           * np.sqrt(TRADING_DAYS))
    raw = (p.target_vol / vol.clip(lower=p.vol_floor)).clip(0.0, p.max_weight)

    # No-trade band: hold the last scalar until the target has drifted far
    # enough to be worth the turnover. Same device as VolTargetParams.
    out = np.full(len(raw), np.nan)
    current = np.nan
    for i, r in enumerate(raw.to_numpy()):
        if np.isnan(r):
            continue
        if np.isnan(current) or abs(r - current) >= p.rebalance_band:
            current = r
        out[i] = current
    # Leading NaN is the warm-up: no vol estimate yet means no position.
    scalar = pd.Series(out, index=raw.index, name="scalar").ffill().fillna(0.0)

    weights = w.copy()
    weights[risk_cols] = w[risk_cols].mul(scalar, axis=0)
    weights[safe] = 1.0 - weights[risk_cols].sum(axis=1)

    risk_weight = weights[risk_cols].sum(axis=1)
    diagnostics = pd.DataFrame({
        "book_vol": vol,
        "raw_scalar": raw,
        "scalar": scalar,
        "base_risk_weight": w[risk_cols].sum(axis=1),
        "risk_weight": risk_weight,
    })

    # Events: one row each time the scalar actually moves the book.
    moved = scalar.diff().abs() > 1e-9
    ev_rows = []
    for dt in scalar.index[moved.fillna(False)]:
        prev, now = float(scalar.shift().loc[dt]), float(scalar.loc[dt])
        ev_rows.append(dict(
            date=dt,
            action="buy" if now > prev else "sell",
            asset="BOOK",
            price=float(vol.loc[dt]) if not np.isnan(vol.loc[dt]) else np.nan,
            reason=f"book vol {vol.loc[dt]:.1%} -> scale {prev:.0%} to {now:.0%}",
        ))
    ev = pd.DataFrame(ev_rows) if ev_rows else _empty_events()

    sig = StrategySignals(name or f"{base.name}_vt", weights, diagnostics, ev,
                          params=p.__dict__.copy())
    sig.holding = base.holding
    sig.holdings_log = base.holdings_log
    sig.held_ranks = base.held_ranks
    sig.momentum = base.momentum
    return sig


# ==========================================================================
# Benchmarks
# ==========================================================================

def buy_and_hold(prices: pd.DataFrame, asset: str) -> StrategySignals:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[asset] = 1.0
    return StrategySignals(f"bh_{asset.lower()}", weights,
                           pd.DataFrame({"close": prices[asset]}), _empty_events())
