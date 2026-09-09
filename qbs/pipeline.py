"""The whole thing in one call, so the notebook and the CLI share one code path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .config import (
    BACKTEST_END, BACKTEST_START, Config, INTL_ASSET, RISK_ASSET,
    SAFE_ASSET, STRATEGY_LABELS, VOL_INDEX,
)
from .data import load_prices, load_vix, synthetic_prices, synthetic_vix
from .engine import BacktestResult, run_backtest
from .metrics import format_summary, summary_table
from .strategies import (
    StrategySignals, book_vol_target, buy_and_hold, connors_rsi2,
    cross_sectional_momentum, gem, vix_circuit_breaker, vol_target_overlay,
)
from .universe import (
    load_universe, load_universe_prices, membership_mask, synthetic_universe,
)


@dataclass
class Lab:
    prices: pd.DataFrame                       # QQQ / VEU / BOXX
    signals: Dict[str, StrategySignals]
    results: Dict[str, BacktestResult]
    rf: pd.Series
    config: Config
    universe: Optional[pd.DataFrame] = None    # NDX constituent closes
    combined: Optional[pd.DataFrame] = None    # universe + safe asset, for the engine
    vix: Optional[pd.Series] = None            # VIX closes driving the circuit breaker

    @property
    def summary(self) -> pd.DataFrame:
        return summary_table(self.results, rf=self.rf, labels=STRATEGY_LABELS)

    @property
    def summary_pretty(self) -> pd.DataFrame:
        return format_summary(self.summary)


def build_signals(prices: pd.DataFrame, cfg: Optional[Config] = None) -> Dict[str, StrategySignals]:
    """The three original strategies plus the two benchmarks."""
    cfg = cfg or Config()
    return {
        "rsi2": connors_rsi2(prices, RISK_ASSET, SAFE_ASSET, cfg.rsi2),
        "gem": gem(prices, cfg.gem),
        "voltarget": vol_target_overlay(prices, None, RISK_ASSET, SAFE_ASSET, cfg.vol),
        "bh_qqq": buy_and_hold(prices, RISK_ASSET),
        "bh_boxx": buy_and_hold(prices, SAFE_ASSET),
    }


def run(
    cfg: Optional[Config] = None,
    prices: Optional[pd.DataFrame] = None,
    universe_prices: Optional[pd.DataFrame] = None,
    offline: bool = False,
    refresh: bool = False,
    use_synthetic: bool = False,
    with_momentum: bool = True,
    with_vix: bool = True,
    with_book_vt: bool = True,
    vix: Optional[pd.Series] = None,
    fetch_universe: bool = True,
    pit_membership: Optional[pd.DataFrame] = None,
) -> Lab:
    """Load data, build every signal, backtest them all on the same window.

    `with_momentum=False` skips the Nasdaq-100 download, which is much the
    slowest part -- useful while iterating on the other three strategies.
    `with_book_vt=False` drops the vol-targeted variant of the momentum book.
    `pit_membership` takes a point-in-time membership frame (see
    `universe.load_pit_universe`) to remove survivorship bias from the ranking.
    """
    cfg = cfg or Config()

    if prices is None:
        if use_synthetic:
            prices = synthetic_prices(start=cfg.download_start, tickers=cfg.tickers)
        else:
            prices = load_prices(cfg.tickers, start=cfg.download_start,
                                 end=cfg.backtest_end, offline=offline, refresh=refresh)

    signals = build_signals(prices, cfg)

    # ---- the wide-universe strategy -------------------------------------
    combined = None
    if with_momentum:
        if universe_prices is None:
            if use_synthetic:
                universe_prices = synthetic_universe(start=cfg.download_start)
            else:
                tickers = load_universe(fetch=fetch_universe)
                universe_prices = load_universe_prices(
                    tickers, start=cfg.download_start, end=cfg.backtest_end,
                    refresh=refresh,
                )

        # Align the universe onto the core calendar and drop names whose
        # history is too short to rank -- they would otherwise sit as NaN
        # columns that quietly shrink the candidate pool.
        uni = universe_prices.reindex(prices.index).ffill()
        uni = uni.loc[:, uni.notna().sum() >= cfg.momentum.min_history]

        eligible = (membership_mask(uni.index, list(uni.columns), pit_membership)
                    if pit_membership is not None else None)

        mom_sig = cross_sectional_momentum(
            uni, prices[SAFE_ASSET], cfg.momentum, eligible=eligible,
        )
        signals["momentum"] = mom_sig

        combined = uni.copy()
        combined[SAFE_ASSET] = prices[SAFE_ASSET]

        # ---- the VIX circuit breaker on top of it -----------------------
        if with_vix:
            if vix is None:
                if use_synthetic:
                    vix = synthetic_vix(prices)
                else:
                    try:
                        vix = load_vix(start=cfg.download_start, end=cfg.backtest_end,
                                       offline=offline, refresh=refresh)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[vix] could not load {VOL_INDEX} ({exc}); "
                              "skipping the circuit-breaker strategy.")
                        vix = None
            if vix is not None:
                signals["momentum_vix"] = vix_circuit_breaker(
                    mom_sig, vix, cfg.vix, name="momentum_vix")

        # ---- the same book, scaled by its OWN realised volatility --------
        # The momentum book runs near 49% vol at beta ~1.7 to QQQ, so most of
        # its drawdown is leverage rather than selection. Unlike the VIX
        # breaker this reacts to the book's own risk, which is why it also
        # cuts the drawdowns that happen while the index stays calm.
        if with_book_vt:
            signals["momentum_vt"] = book_vol_target(
                mom_sig, combined, cfg.book_vol, lag=cfg.execution_lag,
                name="momentum_vt")

    # ---- backtest everything on identical assumptions -------------------
    results: Dict[str, BacktestResult] = {}
    for key, sig in signals.items():
        book = (combined if key in ("momentum", "momentum_vix", "momentum_vt")
                else prices)
        results[key] = run_backtest(
            book, sig, start=cfg.backtest_start, end=cfg.backtest_end,
            lag=cfg.execution_lag, cost_bps=cfg.cost_bps,
            slippage_bps=cfg.slippage_bps,
        )

    rf = (prices[SAFE_ASSET].pct_change()
          .reindex(results["bh_qqq"].returns.index).fillna(0.0))

    return Lab(prices=prices, signals=signals, results=results, rf=rf, config=cfg,
               universe=universe_prices, combined=combined, vix=vix)


def sweep_band(
    lab: Lab,
    n_holds: List[int] = (4, 6, 8, 10),
    exit_ranks: List[int] = (6, 8, 10, 15, 20, 25),
) -> pd.DataFrame:
    """Re-run the momentum strategy across (n_hold, exit_rank) pairs.

    The band is the parameter most likely to be over-fitted, so this is the
    sweep to look at first. What you want is a plateau where a wider band cuts
    turnover without giving back much return -- not a single lucky cell.
    """
    from .metrics import summarise
    from .config import MomentumParams

    if lab.combined is None or "momentum" not in lab.signals:
        raise ValueError("run(with_momentum=True) first")

    cfg = lab.config
    uni = lab.combined.drop(columns=[SAFE_ASSET])
    rows = []
    for n in n_holds:
        for x in exit_ranks:
            if x < n:
                continue
            p = MomentumParams(**{**cfg.momentum.__dict__, "n_hold": n, "exit_rank": x})
            sig = cross_sectional_momentum(uni, lab.prices[SAFE_ASSET], p)
            res = run_backtest(lab.combined, sig, start=cfg.backtest_start,
                               end=cfg.backtest_end, lag=cfg.execution_lag,
                               cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps)
            s = summarise(res, rf=lab.rf)
            rows.append({
                "n_hold": n, "exit_rank": x, "band": x - n,
                "CAGR": s["CAGR"], "Ann. vol": s["Ann. vol"],
                "Sharpe": s["Sharpe (vs BOXX)"], "MaxDD": s["Max drawdown"],
                "Ann. turnover": s["Ann. turnover"],
                "Cost drag": s["Cost drag (ann.)"],
            })
    return pd.DataFrame(rows)


def sweep_vix(
    lab: Lab,
    exit_levels: List[float] = (16, 17, 18, 20, 22, 25, 30, 35),
    band: float = 1.0,
) -> pd.DataFrame:
    """Re-run the circuit breaker across VIX trigger levels.

    The column to read first is **Time invested**. A trigger that leaves the
    strategy in cash most of the sample is not protecting a portfolio, it is
    replacing one -- and any drawdown improvement it shows is bought by
    simply not participating. Compare each row's CAGR against the unprotected
    momentum strategy in the same window before concluding the breaker helped.
    """
    from .metrics import summarise
    from .config import VixBreakerParams

    if lab.vix is None or "momentum" not in lab.signals:
        raise ValueError("run(with_momentum=True, with_vix=True) first")

    cfg = lab.config
    rows = []
    for lvl in exit_levels:
        p = VixBreakerParams(**{**cfg.vix.__dict__,
                                "exit_level": float(lvl),
                                "entry_level": float(lvl) - band})
        sig = vix_circuit_breaker(lab.signals["momentum"], lab.vix, p)
        res = run_backtest(lab.combined, sig, start=cfg.backtest_start,
                           end=cfg.backtest_end, lag=cfg.execution_lag,
                           cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps)
        s = summarise(res, rf=lab.rf)
        d = sig.diagnostics.loc[res.start:res.end]
        rows.append({
            "exit": lvl, "entry": lvl - band,
            "Time invested": float((d["regime"] == "INVESTED").mean()),
            "Trips": int((sig.events["action"] == "sell").sum()) if not sig.events.empty else 0,
            "CAGR": s["CAGR"], "Ann. vol": s["Ann. vol"],
            "Sharpe": s["Sharpe (vs BOXX)"], "MaxDD": s["Max drawdown"],
            "Ann. turnover": s["Ann. turnover"],
        })
    return pd.DataFrame(rows)


def sweep_target_vol(
    lab: Lab,
    targets: List[float] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    base_key: str = "momentum",
) -> pd.DataFrame:
    """Re-run the book vol-target overlay across annualised vol targets.

    Unlike `sweep_vix`, this dial is expected to be smooth: a lower target is
    simply less of the same book, so CAGR and drawdown should both fall
    monotonically and Calmar should stay roughly flat. Read it that way --
    a *kink* would mean something is wrong, and the column that matters when
    choosing is `Max drawdown`, because that is the one you have to sit
    through. `Avg exposure` tells you how much of the book you are actually
    holding to get there.
    """
    from .metrics import summarise
    from .config import BookVolTargetParams

    if lab.combined is None or base_key not in lab.signals:
        raise ValueError("run(with_momentum=True) first")

    cfg = lab.config
    rows = []
    for tv in targets:
        p = BookVolTargetParams(**{**cfg.book_vol.__dict__, "target_vol": float(tv)})
        sig = book_vol_target(lab.signals[base_key], lab.combined, p,
                              lag=cfg.execution_lag)
        res = run_backtest(lab.combined, sig, start=cfg.backtest_start,
                           end=cfg.backtest_end, lag=cfg.execution_lag,
                           cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps)
        s = summarise(res, rf=lab.rf)
        rows.append({
            "Target vol": tv, "CAGR": s["CAGR"], "Ann. vol": s["Ann. vol"],
            "Sharpe": s["Sharpe (vs BOXX)"], "Max drawdown": s["Max drawdown"],
            "Calmar": s["Calmar"], "Ann. turnover": s["Ann. turnover"],
            "Avg exposure": s["Avg risk exposure"],
        })
    return pd.DataFrame(rows)
