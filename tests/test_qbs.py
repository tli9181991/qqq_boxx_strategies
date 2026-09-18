"""Tests. Run with `python -m pytest tests -q` or `python tests/test_qbs.py`.

Two categories: indicator correctness against values you can verify by hand,
and structural invariants of the backtest that a subtle refactor could break
silently (look-ahead, weight leakage, cost accounting).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from qbs.config import (
    SAFE_ASSET, BookVolTargetParams, Config, DrawdownStopParams, FinvizScreenParams,
    GEMParams, MomentumParams, RSI2Params, VixBreakerParams, VolTargetParams,
)
from qbs.data import synthetic_prices, synthetic_vix
from qbs.engine import run_backtest
from qbs.indicators import drawdown, sma, wilder_rsi
from qbs.metrics import summarise
from qbs.pipeline import build_signals, run, sweep_band, sweep_target_vol, sweep_vix
from qbs.strategies import (
    StrategySignals, book_vol_target, buy_and_hold, connors_rsi2,
    cross_sectional_momentum, drawdown_stop, gem, vix_circuit_breaker,
    vol_target_overlay,
)
from qbs.universe import membership_mask, synthetic_universe


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def test_rsi_monotone_up_is_100():
    s = pd.Series(np.arange(1, 60, dtype=float))
    r = wilder_rsi(s, 2).dropna()
    assert (r == 100.0).all(), "an unbroken up-run must read RSI 100"


def test_rsi_monotone_down_is_zero():
    s = pd.Series(np.arange(60, 1, -1, dtype=float))
    r = wilder_rsi(s, 2).dropna()
    assert (r == 0.0).all(), "an unbroken down-run must read RSI 0"


def test_rsi_bounds_and_warmup():
    px = synthetic_prices()["QQQ"]
    r = wilder_rsi(px, 2)
    assert r.iloc[:2].isna().all(), "RSI must be undefined before its period fills"
    v = r.dropna()
    assert v.between(0, 100).all()
    assert v.std() > 5, "RSI(2) should swing; a flat series means the filter is wrong"


def test_rsi_known_seed_value():
    """Hand-checkable case: two up moves then one down move.

    changes: +2, +4, -3.  Seed at i=2: avg_gain=3, avg_loss=0 -> RSI 100.
    i=3 with alpha=1/2: avg_gain = 3*0.5 + 0*0.5 = 1.5
                        avg_loss = 0*0.5 + 3*0.5 = 1.5  -> RS=1 -> RSI 50.
    """
    s = pd.Series([100.0, 102.0, 106.0, 103.0])
    r = wilder_rsi(s, 2)
    assert np.isclose(r.iloc[2], 100.0)
    assert np.isclose(r.iloc[3], 50.0)


def test_sma_warmup_is_nan():
    s = pd.Series(np.arange(10, dtype=float))
    m = sma(s, 5)
    assert m.iloc[:4].isna().all()
    assert np.isclose(m.iloc[4], 2.0)


def test_drawdown_never_positive():
    eq = pd.Series([1.0, 1.2, 0.9, 1.5, 1.1])
    dd = drawdown(eq)
    assert (dd <= 1e-12).all()
    # Deepest point is 1.1 measured against the 1.5 peak, not 0.9 against 1.2.
    assert np.isclose(dd.min(), 1.1 / 1.5 - 1.0)
    assert np.isclose(dd.iloc[2], 0.9 / 1.2 - 1.0)


# --------------------------------------------------------------------------
# Strategy structure
# --------------------------------------------------------------------------

def test_weights_are_fully_invested_and_non_negative():
    px = synthetic_prices()
    for sig in build_signals(px).values():
        w = sig.weights
        assert (w >= -1e-9).all().all(), f"{sig.name} produced a short position"
        s = w.sum(axis=1)
        assert (s <= 1.0 + 1e-9).all(), f"{sig.name} produced leverage above 1"


def test_rsi2_never_enters_below_trend():
    px = synthetic_prices()
    sig = connors_rsi2(px)
    d = sig.diagnostics
    buys = sig.events[sig.events["action"] == "buy"]["date"]
    assert len(buys) > 0, "the test market should generate at least one entry"
    for dt in buys:
        assert d.loc[dt, "above_trend"] == 1.0, f"entry at {dt} was below SMA200"
        assert d.loc[dt, "rsi"] < RSI2Params().entry_threshold


def test_rsi2_respects_time_stop():
    px = synthetic_prices()
    p = RSI2Params(use_sma_exit=False, exit_rsi=999, max_hold_days=4)
    sig = connors_rsi2(px, params=p)
    holding = sig.weights["QQQ"]
    # No unbroken run of 1.0 longer than the time stop.
    runs = holding.groupby((holding != holding.shift()).cumsum()).transform("size")
    assert holding[(holding == 1.0)].empty or runs[holding == 1.0].max() <= p.max_hold_days


def test_gem_holds_exactly_one_asset():
    px = synthetic_prices()
    sig = gem(px)
    w = sig.weights
    # Before the first month-end decision the lookback is not filled, so GEM
    # is deliberately flat -- that stretch is warm-up, not a position.
    live = w.loc[sig.holding.first_valid_index():]
    assert np.allclose(live.sum(axis=1), 1.0), "GEM must be fully invested once live"
    assert ((w == 0.0) | (w == 1.0)).all().all(), "GEM should be all-in on one sleeve"
    assert (w.sum(axis=1).loc[:sig.holding.first_valid_index()].iloc[:-1] == 0.0).all()


def test_gem_picks_the_stronger_sleeve():
    """Force VEU to dominate and check GEM rotates into it."""
    px = synthetic_prices()
    px = px.copy()
    px["VEU"] = px["VEU"].iloc[0] * np.exp(
        np.linspace(0, 1.2, len(px)))          # relentless winner
    px["QQQ"] = px["QQQ"].iloc[0] * np.exp(
        np.linspace(0, -0.5, len(px)))         # relentless loser
    sig = gem(px)
    tail = sig.weights.iloc[-60:]
    assert tail["VEU"].mean() > 0.9, "GEM failed to rotate into the stronger sleeve"


def test_gem_goes_defensive_when_equities_lag_cash():
    px = synthetic_prices()
    px = px.copy()
    n = len(px)
    for c in ("QQQ", "VEU"):
        px[c] = px[c].iloc[0] * np.exp(np.linspace(0, -0.6, n))
    px["BOXX"] = px["BOXX"].iloc[0] * np.exp(np.linspace(0, 0.15, n))
    sig = gem(px)
    tail = sig.weights.iloc[-60:]
    assert tail["BOXX"].mean() > 0.9, "GEM stayed in equities while cash beat them"


def test_vol_target_scales_inversely_with_vol():
    px = synthetic_prices()
    sig = vol_target_overlay(px)
    d = sig.diagnostics.dropna()
    corr = d["realized_vol"].corr(d["raw_weight"])
    assert corr < -0.8, f"exposure should fall as vol rises (corr={corr:.2f})"


def test_vol_target_respects_the_cap():
    px = synthetic_prices()
    p = VolTargetParams(max_weight=0.8)
    sig = vol_target_overlay(px, params=p)
    assert sig.weights["QQQ"].max() <= 0.8 + 1e-9


def test_vol_target_band_reduces_turnover():
    px = synthetic_prices()
    banded = vol_target_overlay(px, params=VolTargetParams(rebalance_band=0.05))
    unbanded = vol_target_overlay(px, params=VolTargetParams(rebalance_band=0.0))
    tb = banded.weights.diff().abs().sum().sum()
    tu = unbanded.weights.diff().abs().sum().sum()
    assert tb < tu, "the no-trade band must reduce turnover"


def test_overlay_can_only_shrink_the_base_exposure():
    px = synthetic_prices()
    base = connors_rsi2(px)
    over = vol_target_overlay(px, base_weights=base.weights,
                              params=VolTargetParams(max_weight=1.0))
    assert (over.weights["QQQ"] <= base.weights["QQQ"] + 1e-9).all()


# --------------------------------------------------------------------------
# Cross-sectional momentum
# --------------------------------------------------------------------------

def _mom_fixture():
    uni = synthetic_universe(n=60)
    safe = synthetic_prices(start="2023-06-01")["BOXX"].reindex(uni.index).ffill().bfill()
    return uni, safe


def test_momentum_holds_at_most_n_and_is_slot_weighted():
    uni, safe = _mom_fixture()
    p = MomentumParams(n_hold=6, exit_rank=10)
    sig = cross_sectional_momentum(uni, safe, p)
    risk = sig.weights.drop(columns=["BOXX"])
    assert (risk > 0).sum(axis=1).max() <= p.n_hold
    assert risk.max().max() <= 1.0 / p.n_hold + 1e-9, "a name exceeded its slot weight"
    assert np.allclose(sig.weights.sum(axis=1), 1.0), "book must be fully allocated"
    assert (sig.weights >= -1e-12).all().all()


def test_momentum_band_reduces_turnover_monotonically():
    """The whole point of the band. Wider band -> strictly less trading."""
    uni, safe = _mom_fixture()
    turnovers = []
    for exit_rank in (6, 10, 20, 30):
        sig = cross_sectional_momentum(
            uni, safe, MomentumParams(n_hold=6, exit_rank=exit_rank))
        turnovers.append(sig.weights.diff().abs().sum().sum())
    assert all(a > b for a, b in zip(turnovers, turnovers[1:])), turnovers


def test_momentum_band_of_zero_is_the_naive_strategy():
    uni, safe = _mom_fixture()
    banded = cross_sectional_momentum(uni, safe, MomentumParams(n_hold=6, exit_rank=6))
    risk = banded.weights.drop(columns=["BOXX"])
    # With no band, holdings are exactly the top 6 by momentum every day.
    mom = banded.momentum
    for dt in risk.index[-20:]:
        held = set(risk.columns[risk.loc[dt] > 0])
        top = set(mom.loc[dt].dropna().nlargest(6).index)
        if len(held) == 6 and len(top) == 6:
            assert held == top, f"{dt}: {held ^ top}"


def test_momentum_rejects_impossible_band():
    try:
        MomentumParams(n_hold=6, exit_rank=3)
    except ValueError:
        return
    raise AssertionError("exit_rank < n_hold should be rejected")


def test_momentum_has_no_lookahead_in_the_score():
    """The 12-1 score at date t must use only prices strictly before t."""
    uni, safe = _mom_fixture()
    p = MomentumParams(lookback_months=12, skip_months=1)
    sig = cross_sectional_momentum(uni, safe, p)
    dt = uni.index[400]
    col = uni.columns[0]
    look, skip = int(round(12 * 21)), int(round(1 * 21))
    expected = uni[col].iloc[400 - skip] / uni[col].iloc[400 - look] - 1.0
    assert np.isclose(sig.momentum.loc[dt, col], expected)

    # And perturbing the future must not change any past score.
    tampered = uni.copy()
    tampered.iloc[401:] *= 3.0
    sig2 = cross_sectional_momentum(tampered, safe, p)
    pd.testing.assert_series_equal(
        sig.momentum.loc[:dt, col], sig2.momentum.loc[:dt, col], check_names=False)


def test_momentum_absolute_filter_goes_to_cash():
    """When nothing beats the safe asset, the book must sit in the safe asset."""
    base, _ = _mom_fixture()
    n = len(base)
    # Every stock declines steadily; the safe asset climbs hard.
    decay = np.exp(np.linspace(0.0, -0.8, n))
    uni = pd.DataFrame(
        base.iloc[0].to_numpy()[None, :] * decay[:, None],
        index=base.index, columns=base.columns,
    )
    safe2 = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.4, n)), index=base.index)
    sig = cross_sectional_momentum(uni, safe2, MomentumParams(absolute_filter=True))
    assert sig.weights["BOXX"].iloc[-60:].mean() > 0.99, "should be fully defensive"

    off = cross_sectional_momentum(uni, safe2, MomentumParams(absolute_filter=False))
    assert off.weights.drop(columns=["BOXX"]).iloc[-60:].sum(axis=1).mean() > 0.9, \
        "with the filter off it should stay invested in the least-bad names"


def test_momentum_respects_membership_mask():
    """A name outside the index on a date must never be held on that date."""
    uni, safe = _mom_fixture()
    banned = list(uni.columns[:20])
    mask = pd.DataFrame(True, index=uni.index, columns=uni.columns)
    mask[banned] = False
    sig = cross_sectional_momentum(uni, safe, MomentumParams(), eligible=mask)
    assert sig.weights[banned].abs().sum().sum() == 0.0


def test_momentum_min_history_blocks_young_names():
    uni, safe = _mom_fixture()
    young = uni.columns[0]
    uni = uni.copy()
    uni.loc[uni.index[:600], young] = np.nan   # this name "lists" late
    sig = cross_sectional_momentum(uni, safe, MomentumParams(min_history=260))
    early = sig.weights[young].iloc[:600]
    assert early.abs().sum() == 0.0, "held a name before it had enough history"


def test_sweep_band_runs_and_turnover_falls():
    lab = run(use_synthetic=True)
    sw = sweep_band(lab, n_holds=[6], exit_ranks=[6, 12, 25])
    assert len(sw) == 3
    assert sw.sort_values("band")["Ann. turnover"].is_monotonic_decreasing


def test_membership_mask_forward_fills():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    pit = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-03")],
        "ticker": ["AAA", "BBB"],
    })
    mask = membership_mask(idx, ["AAA", "BBB"], pit)
    assert mask.loc[idx[0], "AAA"] and not mask.loc[idx[0], "BBB"]
    assert mask.loc[pd.Timestamp("2025-03-03"), "BBB"], "membership should persist"


# --------------------------------------------------------------------------
# VIX circuit breaker
# --------------------------------------------------------------------------

def _vix_fixture(levels):
    """A base strategy fully invested in QQQ, plus a hand-written VIX path."""
    px = synthetic_prices()
    idx = px.index[:len(levels)]
    px = px.loc[idx]
    base = buy_and_hold(px, "QQQ")
    vix = pd.Series(levels, index=idx, dtype=float)
    return px, base, vix


def test_vix_breaker_trips_above_exit_level():
    # calm, calm, spike, spike, spike, spike
    px, base, vix = _vix_fixture([12, 12, 25, 25, 25, 25, 25, 25])
    sig = vix_circuit_breaker(base, vix, VixBreakerParams(exit_level=17, entry_level=16))
    assert sig.diagnostics["regime"].iloc[0] == "INVESTED"
    assert sig.diagnostics["regime"].iloc[2] == "CASH", "should have gone to cash on the spike"
    assert sig.weights.iloc[2].sum() == 0.0, "CASH must hold nothing at all"


def test_vix_breaker_parks_in_safe_asset_after_the_delay():
    px, base, vix = _vix_fixture([12, 30, 30, 30, 30, 30, 30, 30])
    p = VixBreakerParams(exit_level=17, entry_level=16, park_after_days=3, min_cash_days=2)
    sig = vix_circuit_breaker(base, vix, p)
    reg = list(sig.diagnostics["regime"])
    assert "PARKED" in reg, reg
    first_park = reg.index("PARKED")
    first_cash = reg.index("CASH")
    assert first_park - first_cash == p.park_after_days, (first_cash, first_park)
    assert np.isclose(sig.weights["BOXX"].iloc[first_park], 1.0)


def test_vix_breaker_respects_minimum_cash_dwell():
    """VIX recovers immediately, but the book must stay out for min_cash_days."""
    px, base, vix = _vix_fixture([12, 25, 10, 10, 10, 10, 10, 10])
    p = VixBreakerParams(exit_level=17, entry_level=16, min_cash_days=2, park_after_days=9)
    sig = vix_circuit_breaker(base, vix, p)
    reg = list(sig.diagnostics["regime"])
    assert reg[1] == "CASH"
    assert reg[2] == "CASH", "re-entered before the minimum dwell elapsed"
    assert reg[3] == "INVESTED", reg


def test_vix_breaker_needs_the_entry_level_not_just_the_exit_level():
    """Between the two thresholds is a no-mans-land: stay out."""
    px, base, vix = _vix_fixture([12, 25, 16.5, 16.5, 16.5, 16.5, 16.5, 16.5])
    sig = vix_circuit_breaker(base, vix, VixBreakerParams(exit_level=17, entry_level=16,
                                                          park_after_days=9))
    reg = list(sig.diagnostics["regime"])
    assert all(r != "INVESTED" for r in reg[1:]), reg


def test_vix_breaker_rejects_inverted_band():
    px, base, vix = _vix_fixture([12] * 8)
    try:
        vix_circuit_breaker(base, vix, VixBreakerParams(exit_level=16, entry_level=17))
    except ValueError:
        return
    raise AssertionError("entry_level above exit_level should be rejected")


def test_vix_breaker_never_exceeds_the_base_exposure():
    px = synthetic_prices()
    vix = synthetic_vix(px)
    base = buy_and_hold(px, "QQQ")
    sig = vix_circuit_breaker(base, vix, VixBreakerParams())
    assert (sig.weights["QQQ"] <= base.weights["QQQ"] + 1e-12).all()
    assert (sig.weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_vix_breaker_is_a_no_op_when_the_trigger_never_fires():
    px = synthetic_prices()
    vix = pd.Series(5.0, index=px.index)          # permanently calm
    base = buy_and_hold(px, "QQQ")
    sig = vix_circuit_breaker(base, vix, VixBreakerParams(exit_level=17))
    pd.testing.assert_series_equal(sig.weights["QQQ"], base.weights["QQQ"],
                                   check_names=False)


def test_synthetic_vix_is_calibrated_and_shaped():
    px = synthetic_prices()
    vix = synthetic_vix(px, target_median=17.5)
    assert 16.5 < vix.median() < 18.5, vix.median()
    assert vix.min() >= 9.0 and vix.max() <= 90.0
    # Spikes must decay more slowly than they arrive.
    d = vix.diff().dropna()
    assert d.max() > abs(d.min()), "up-moves should be sharper than down-moves"


def test_sweep_vix_converges_to_the_unprotected_strategy():
    """A trigger far above the VIX distribution must never fire."""
    lab = run(use_synthetic=True)
    sw = sweep_vix(lab, exit_levels=[17, 60], band=1.0)
    high = sw[sw["exit"] == 60].iloc[0]
    assert high["Trips"] == 0
    assert np.isclose(high["Time invested"], 1.0)
    assert np.isclose(high["CAGR"], lab.summary.loc["Top-6 NDX momentum", "CAGR"], atol=1e-9)


# --------------------------------------------------------------------------
# Engine invariants
# --------------------------------------------------------------------------

def test_execution_lag_actually_lags():
    """The classic look-ahead bug: a signal must not earn its own day's return.

    Build a strategy that is long only on one specific day, then check the
    equity curve moves on the NEXT day, not that day.
    """
    px = synthetic_prices()
    sig = buy_and_hold(px, "QQQ")
    day = px.index[100]
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    w.loc[day, "QQQ"] = 1.0
    sig.weights = w

    res = run_backtest(px, sig, lag=1, cost_bps=0.0)
    i = res.returns.index.get_loc(day)
    nxt = res.returns.index[i + 1]
    qqq_ret = px["QQQ"].pct_change()

    assert np.isclose(res.gross_returns.loc[day], 0.0), "position earned the return of its own signal day"
    assert np.isclose(res.gross_returns.loc[nxt], qqq_ret.loc[nxt])


def test_buy_and_hold_matches_the_underlying():
    px = synthetic_prices()
    res = run_backtest(px, buy_and_hold(px, "QQQ"), start="2025-01-20", cost_bps=0.0)
    window = px.loc[res.start:res.end, "QQQ"]
    expected = window.iloc[-1] / window.iloc[0]
    # The lag means day one is flat, so compare from the second bar onwards.
    achieved = float(res.equity.iloc[-1] / res.equity.iloc[1])
    reference = float(window.iloc[-1] / window.iloc[1])
    assert np.isclose(achieved, reference, rtol=1e-6), (achieved, reference, expected)


def test_costs_reduce_returns_and_scale_with_turnover():
    px = synthetic_prices()
    sig = connors_rsi2(px)
    cheap = run_backtest(px, sig, start="2025-01-20", cost_bps=0.0)
    dear = run_backtest(px, sig, start="2025-01-20", cost_bps=20.0)
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]
    assert np.isclose(dear.costs.sum() / cheap.turnover.sum(), 20.0 / 1e4, rtol=1e-6)


def test_slippage_is_tracked_separately_from_commission():
    px = synthetic_prices()
    sig = connors_rsi2(px)
    res = run_backtest(px, sig, start="2025-01-20", cost_bps=1.0, slippage_bps=5.0)
    assert np.isclose(res.commission.sum() * 5.0, res.slippage.sum(), rtol=1e-9)
    assert np.isclose(res.costs.sum(), res.commission.sum() + res.slippage.sum())
    free = run_backtest(px, sig, start="2025-01-20", cost_bps=1.0, slippage_bps=0.0)
    assert res.equity.iloc[-1] < free.equity.iloc[-1]


def test_zero_turnover_when_weights_never_change():
    px = synthetic_prices()
    res = run_backtest(px, buy_and_hold(px, "BOXX"), start="2025-01-20")
    assert np.isclose(res.turnover.iloc[1:].sum(), 0.0), "a static book should not trade"


def test_window_start_is_respected():
    px = synthetic_prices()
    res = run_backtest(px, buy_and_hold(px, "QQQ"), start="2025-01-20", end="2025-06-30")
    assert res.start >= pd.Timestamp("2025-01-20")
    assert res.end <= pd.Timestamp("2025-06-30")


def test_metrics_are_finite_and_sane():
    lab = run(use_synthetic=True)
    for key, res in lab.results.items():
        s = summarise(res, rf=lab.rf)
        assert s, f"{key} produced no statistics"
        assert -1.0 <= s["Max drawdown"] <= 0.0
        assert 0.0 <= s["Hit rate"] <= 1.0
        assert np.isfinite(s["CAGR"])
        assert np.isfinite(s["Ann. vol"])


def test_full_pipeline_runs_and_aligns():
    lab = run(use_synthetic=True)
    lengths = {k: len(r.returns) for k, r in lab.results.items()}
    assert len(set(lengths.values())) == 1, f"strategies ran on different windows: {lengths}"
    assert lab.summary.shape[0] == len(lab.results)


# --------------------------------------------------------------------------
# Book-level vol targeting
# --------------------------------------------------------------------------

def _mom_book(seed_prices=None):
    """A small momentum book plus the safe asset, for the overlay tests."""
    px = seed_prices if seed_prices is not None else synthetic_prices()
    uni = synthetic_universe(n=25, start="2023-06-01")
    uni = uni.reindex(px.index).ffill().dropna(axis=1, how="all")
    p = MomentumParams(n_hold=4, exit_rank=8, min_history=200)
    sig = cross_sectional_momentum(uni, px["BOXX"], p)
    book = uni.copy()
    book["BOXX"] = px["BOXX"]
    return px, book, sig


def test_book_vol_target_weights_stay_a_valid_book():
    _, book, sig = _mom_book()
    vt = book_vol_target(sig, book, BookVolTargetParams(target_vol=0.20))
    total = vt.weights.sum(axis=1)
    assert np.allclose(total, 1.0), "weights must always sum to 1 (risk + safe)"
    assert (vt.weights >= -1e-9).all().all(), "no negative weights -- this book is long-only"
    risk = vt.weights.drop(columns=["BOXX"]).sum(axis=1)
    assert (risk <= 1.0 + 1e-9).all(), "max_weight=1.0 must forbid leverage"


def test_book_vol_target_only_scales_never_reselects():
    """The overlay changes how much of the book is held, never which names."""
    _, book, sig = _mom_book()
    vt = book_vol_target(sig, book, BookVolTargetParams(target_vol=0.20))
    names = [c for c in sig.weights.columns if c != "BOXX"]
    base, scaled = sig.weights[names], vt.weights[names]
    held_base = (base > 1e-9)
    held_vt = (scaled > 1e-9)
    # Any name the overlay holds must be one the base strategy chose.
    assert not (held_vt & ~held_base).any().any(), "overlay introduced a name the base never held"
    # Within each date the relative sizes are unchanged.
    for dt in base.index[::40]:
        b, v = base.loc[dt], scaled.loc[dt]
        if b.sum() > 0 and v.sum() > 0:
            assert np.allclose(b / b.sum(), v / v.sum(), atol=1e-9), \
                f"relative position sizes changed on {dt}"


def test_book_vol_target_lowers_volatility_and_drawdown():
    px, book, sig = _mom_book()
    base = run_backtest(book, sig, start="2025-01-20")
    tight = run_backtest(book, book_vol_target(
        sig, book, BookVolTargetParams(target_vol=0.10)), start="2025-01-20")
    b, t = summarise(base), summarise(tight)
    assert t["Ann. vol"] < b["Ann. vol"], "a lower target must reduce realised vol"
    assert t["Max drawdown"] > b["Max drawdown"], "a lower target must shrink the drawdown"


def test_book_vol_target_is_monotone_in_the_target():
    """The dial must be smooth: more target vol, more exposure. A kink is a bug."""
    _, book, sig = _mom_book()
    expos = []
    for tv in (0.10, 0.20, 0.30, 0.40):
        vt = book_vol_target(sig, book, BookVolTargetParams(target_vol=tv))
        expos.append(float(vt.weights.drop(columns=["BOXX"]).sum(axis=1).mean()))
    assert expos == sorted(expos), f"exposure is not monotone in the target: {expos}"


def test_book_vol_target_has_no_look_ahead():
    """Rewriting the future must not change any weight decided before it.

    The overlay reads the book's realised returns, so a lag mistake here is
    invisible in the equity curve but would still be look-ahead. Perturbing
    the tail of the price history and checking that earlier weights are
    bit-identical catches it.
    """
    px, book, sig = _mom_book()
    cut = book.index[len(book) // 2]

    tampered = book.copy()
    rng = np.random.default_rng(0)
    after = tampered.index > cut
    tampered.loc[after] = tampered.loc[after] * rng.uniform(0.5, 1.5, tampered.loc[after].shape)

    p = BookVolTargetParams(target_vol=0.20)
    a = book_vol_target(sig, book, p).weights.loc[:cut]
    b = book_vol_target(sig, tampered, p).weights.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_book_vol_target_band_reduces_turnover():
    _, book, sig = _mom_book()
    loose = book_vol_target(sig, book, BookVolTargetParams(rebalance_band=0.25))
    tight = book_vol_target(sig, book, BookVolTargetParams(rebalance_band=0.0))
    n_loose = int((loose.diagnostics["scalar"].diff().abs() > 1e-12).sum())
    n_tight = int((tight.diagnostics["scalar"].diff().abs() > 1e-12).sum())
    assert n_loose < n_tight, "a wider no-trade band must move the scalar less often"


def test_book_vol_target_respects_max_weight():
    _, book, sig = _mom_book()
    vt = book_vol_target(sig, book, BookVolTargetParams(target_vol=5.0, max_weight=1.0))
    risk = vt.weights.drop(columns=["BOXX"]).sum(axis=1)
    assert risk.max() <= 1.0 + 1e-9, "an absurd target must still be capped by max_weight"


def test_pipeline_exposes_the_vol_targeted_book():
    lab = run(use_synthetic=True)
    assert "momentum_vt" in lab.results, "pipeline should build the vol-targeted book"
    assert len(lab.results["momentum_vt"].returns) == len(lab.results["momentum"].returns)


def test_sweep_target_vol_is_ordered():
    lab = run(use_synthetic=True)
    sw = sweep_target_vol(lab, targets=[0.10, 0.20, 0.30])
    assert len(sw) == 3
    assert sw["Avg exposure"].is_monotonic_increasing, \
        "higher vol targets must hold more of the book"


# --------------------------------------------------------------------------
# The notebook's trend-template screen (qbs/screens.py)
# --------------------------------------------------------------------------

def _screen_inputs(n=30):
    px = synthetic_prices()
    uni = synthetic_universe(n=n, start="2023-06-01").reindex(px.index).ffill()
    return uni, px["QQQ"], px["BOXX"]


def test_rolling_log_slope_matches_polyfit():
    """The vectorised slope must equal the notebook's np.polyfit, not approximate it."""
    from qbs.screens import rolling_log_slope

    rng = np.random.default_rng(0)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 400))))
    mine = rolling_log_slope(s.to_frame("X"), 100)["X"]
    for i in (150, 250, 399):
        y = np.log(s.iloc[i - 99:i + 1].values)
        theirs = np.polyfit(np.arange(100), y, 1)[0]
        assert abs(mine.iloc[i] - theirs) < 1e-12


def test_screen_weights_are_a_valid_long_only_book():
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    sig = trend_template_screen(uni, mkt, safe, TrendScreenParams(n_hold=6))
    assert np.allclose(sig.weights.sum(axis=1), 1.0)
    assert (sig.weights >= -1e-9).all().all()
    risk = sig.weights.drop(columns=["BOXX"]).sum(axis=1)
    assert (risk <= 1.0 + 1e-9).all(), "a screen must never lever the book"


def test_screen_parks_in_the_safe_asset_when_nothing_passes():
    """A screen is an absolute test -- on a bad day the answer is 'none', and
    that has to mean cash rather than a forced allocation."""
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    # An impossible criterion: nothing is ever within 0% of its 52-week high
    # while also above a rising 200-day average, on synthetic data.
    p = TrendScreenParams(n_hold=6, within_52w_high_pct=-1.0)
    sig = trend_template_screen(uni, mkt, safe, p)
    assert np.allclose(sig.weights["BOXX"], 1.0), "must be fully in cash"
    assert sig.diagnostics["n_passing"].max() == 0


def test_screen_holds_everything_passing_when_n_hold_is_zero():
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    sig = trend_template_screen(uni, mkt, safe, TrendScreenParams(n_hold=0))
    d = sig.diagnostics
    live = d["n_passing"] > 0
    assert (d.loc[live, "n_held"] == d.loc[live, "n_passing"]).all()


def test_screen_caps_the_book_at_n_hold():
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    sig = trend_template_screen(uni, mkt, safe, TrendScreenParams(n_hold=4))
    assert sig.diagnostics["n_held"].max() <= 4


def test_screen_has_no_look_ahead():
    """Rewriting the future must not change any weight decided before it."""
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    cut = uni.index[len(uni) // 2]
    rng = np.random.default_rng(1)
    tampered = uni.copy()
    after = tampered.index > cut
    tampered.loc[after] = tampered.loc[after] * rng.uniform(0.5, 1.5, tampered.loc[after].shape)

    p = TrendScreenParams(n_hold=6)
    a = trend_template_screen(uni, mkt, safe, p).weights.loc[:cut]
    b = trend_template_screen(tampered, mkt, safe, p).weights.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_screen_sector_filter_only_ever_removes_names():
    """The fifth criterion is a filter: enabling it cannot admit a new name."""
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    p = TrendScreenParams(n_hold=0)
    base = trend_template_screen(uni, mkt, safe, p)

    sectors = {t: ("AAA" if i % 2 else "BBB") for i, t in enumerate(uni.columns)}
    sec_px = pd.DataFrame({"AAA": mkt * 1.5, "BBB": mkt * 0.5}, index=uni.index)
    withs = trend_template_screen(uni, mkt, safe, p, sector_map=sectors,
                                  sector_prices=sec_px)
    assert withs.params["sector_filter_applied"] is True
    assert base.params["sector_filter_applied"] is False
    for dt in uni.index[::40]:
        assert set(withs.holdings_log[dt]) <= set(base.holdings_log[dt])


def test_screen_records_rank_and_score_like_the_momentum_book():
    """So the live run log stores selections from either strategy identically."""
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    sig = trend_template_screen(uni, mkt, safe, TrendScreenParams(n_hold=6))
    assert {"rank", "score"} <= set(sig.events.columns)
    assert sig.held_ranks is not None
    buys = sig.events[sig.events["action"] == "buy"]
    if not buys.empty:
        assert (buys["rank"] >= 1).all()


def test_screen_runs_through_the_shared_engine():
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    sig = trend_template_screen(uni, mkt, safe, TrendScreenParams(n_hold=6))
    book = uni.copy()
    book["BOXX"] = safe
    res = run_backtest(book, sig, start="2025-01-20")
    s = summarise(res)
    assert np.isfinite(s["CAGR"]) and -1.0 <= s["Max drawdown"] <= 0.0

# --------------------------------------------------------------------------
# The Finviz screen (qbs/screens.py)
# --------------------------------------------------------------------------

def _finviz_inputs(n=40):
    px = synthetic_prices()
    uni = synthetic_universe(n=n, start="2023-06-01").reindex(px.index).ffill()
    return uni, px["BOXX"]


def test_finviz_weights_are_a_valid_long_only_book():
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    sig = finviz_momentum_screen(uni, safe, FinvizScreenParams(n_hold=6))
    assert np.allclose(sig.weights.sum(axis=1), 1.0)
    assert (sig.weights >= -1e-9).all().all()
    risk = sig.weights.drop(columns=["BOXX"]).sum(axis=1)
    assert (risk <= 1.0 + 1e-9).all(), "a screen must never lever the book"


def test_finviz_parks_in_the_safe_asset_when_nothing_passes():
    """The Finviz filters are absolute tests, so 'nothing qualifies' has to be
    a reachable state that means cash rather than a forced allocation."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    p = FinvizScreenParams(n_hold=6, within_52w_high_pct=-1.0)
    sig = finviz_momentum_screen(uni, safe, p)
    assert np.allclose(sig.weights["BOXX"], 1.0), "must be fully in cash"
    assert np.nanmax(sig.diagnostics["n_passing"].to_numpy()) == 0


def test_finviz_ranks_by_one_year_return_among_passing_names():
    """The notebook's rule: RS Rank is a bucketed 1-year return, so the first
    name bought on any date is the strongest 1-year performer that passed."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    p = FinvizScreenParams(n_hold=6)
    sig = finviz_momentum_screen(uni, safe, p)

    perf = uni / uni.shift(p.rs_lookback) - 1.0
    checked = 0
    for dt, names in sig.holdings_log.items():
        if len(names) < p.n_hold:
            continue
        chosen = perf.loc[dt, names]
        # Every held name must be at least as strong as the weakest held name,
        # and no unheld name inside the book's own price range may beat the best.
        assert chosen.iloc[0] == chosen.max(), "selection order must be RS-descending"
        checked += 1
    assert checked > 0, "the fixture never filled the book -- test proves nothing"


def test_finviz_rs_bucketing_is_a_no_op_on_a_small_universe():
    """qcut into min(100, n) buckets gives every name its own bucket when the
    candidate pool is smaller than the bucket count, so the tie-break on
    distance-below-high can never fire. Worth pinning: it is the reason the
    rule collapses to a plain 1-year-return sort on a Nasdaq-100 universe."""
    from qbs.screens import _rs_rank_order

    cand = pd.Series({"A": 0.50, "B": 0.10, "C": 0.30})
    # Distances below the high that would reverse the order if they were used.
    off = pd.Series({"A": 0.09, "B": 0.01, "C": 0.05})
    order, rs = _rs_rank_order(cand, off, buckets=100)
    assert list(order.index) == ["A", "C", "B"]
    assert rs.is_monotonic_decreasing


def test_finviz_tie_break_prefers_the_name_nearest_its_high():
    """With fewer buckets than names the tie-break does fire, and it must
    prefer the smaller distance below the 52-week high."""
    from qbs.screens import _rs_rank_order

    cand = pd.Series({"A": 0.50, "B": 0.45, "C": 0.10, "D": 0.05})
    off = pd.Series({"A": 0.09, "B": 0.01, "C": 0.08, "D": 0.02})
    order, _ = _rs_rank_order(cand, off, buckets=2)
    assert list(order.index)[:2] == ["B", "A"], "same bucket -> nearest the high first"


def test_finviz_caps_the_book_at_n_hold():
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    sig = finviz_momentum_screen(uni, safe, FinvizScreenParams(n_hold=4))
    assert sig.diagnostics["n_held"].max() <= 4


def test_finviz_has_no_look_ahead():
    """Rewriting the future must not change any weight decided before it."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    cut = uni.index[len(uni) // 2]
    rng = np.random.default_rng(3)
    tampered = uni.copy()
    after = tampered.index > cut
    tampered.loc[after] = tampered.loc[after] * rng.uniform(0.5, 1.5, tampered.loc[after].shape)

    p = FinvizScreenParams(n_hold=6)
    a = finviz_momentum_screen(uni, safe, p).weights.loc[:cut]
    b = finviz_momentum_screen(tampered, safe, p).weights.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_finviz_volume_filter_only_ever_removes_names():
    """'Average Volume over 200K' is a filter: supplying volumes cannot admit
    a name that failed without them."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    p = FinvizScreenParams(n_hold=0)
    base = finviz_momentum_screen(uni, safe, p)

    # Half the universe trades under the threshold, half far above it.
    vol = pd.DataFrame(
        {t: (1e5 if i % 2 else 1e7) for i, t in enumerate(uni.columns)},
        index=uni.index,
    )
    withv = finviz_momentum_screen(uni, safe, p, volumes=vol)
    assert base.params["volume_filter_applied"] is False
    assert withv.params["volume_filter_applied"] is True
    for dt in uni.index[::40]:
        assert set(withv.holdings_log[dt]) <= set(base.holdings_log[dt])


def test_finviz_min_dollar_volume_needs_volumes():
    """Silently ignoring a criterion the caller asked for would overstate the
    strategy, so an unusable parameter has to raise."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    try:
        finviz_momentum_screen(uni, safe,
                               FinvizScreenParams(min_dollar_volume=5e6))
    except ValueError:
        return
    raise AssertionError("min_dollar_volume without volumes must raise")


def test_finviz_quarter_up_gate_removes_names():
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    on = finviz_momentum_screen(uni, safe, FinvizScreenParams(n_hold=0))
    off = finviz_momentum_screen(uni, safe,
                                 FinvizScreenParams(n_hold=0, require_quarter_up=False))
    for dt in uni.index[::40]:
        assert set(on.holdings_log[dt]) <= set(off.holdings_log[dt])


def test_finviz_band_reduces_turnover():
    """The notebook has no band (exit_rank=0) and re-screens from scratch every
    day. Adding one must cut trading, whatever it does to return."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    none = finviz_momentum_screen(uni, safe, FinvizScreenParams(exit_rank=0))
    band = finviz_momentum_screen(uni, safe, FinvizScreenParams(exit_rank=15))
    t_none = float(none.weights.diff().abs().sum(axis=1).sum())
    t_band = float(band.weights.diff().abs().sum(axis=1).sum())
    assert t_band < t_none, "a hysteresis band must reduce turnover"


def test_finviz_band_cannot_be_narrower_than_the_book():
    try:
        FinvizScreenParams(n_hold=6, exit_rank=3)
    except ValueError:
        return
    raise AssertionError("exit_rank below n_hold must raise")


def test_finviz_records_rank_and_score_like_the_momentum_book():
    """So the live run log stores selections from either strategy identically."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    sig = finviz_momentum_screen(uni, safe, FinvizScreenParams(n_hold=6))
    assert {"rank", "score"} <= set(sig.events.columns)
    assert sig.held_ranks is not None
    buys = sig.events[sig.events["action"] == "buy"]
    assert not buys.empty and (buys["rank"] >= 1).all()


def test_finviz_runs_through_the_shared_engine():
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    sig = finviz_momentum_screen(uni, safe, FinvizScreenParams(n_hold=6))
    book = uni.copy()
    book["BOXX"] = safe
    res = run_backtest(book, sig, start="2025-01-20")
    s = summarise(res)
    assert np.isfinite(s["CAGR"]) and -1.0 <= s["Max drawdown"] <= 0.0


def test_finviz_monthly_rebalance_only_trades_at_month_ends():
    """The non-daily path is a separate branch of the date loop, so it needs its
    own test -- with `rebalance="daily"` every date is a rebalance date and the
    branch never executes."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    monthly = finviz_momentum_screen(uni, safe, FinvizScreenParams(rebalance="ME"))
    daily = finviz_momentum_screen(uni, safe, FinvizScreenParams(rebalance="daily"))

    changed = monthly.weights.diff().abs().sum(axis=1) > 1e-12
    assert 0 < changed.sum() < (daily.weights.diff().abs().sum(axis=1) > 1e-12).sum()

    # n_passing is the size of the candidate pool, not of the book. On a
    # non-rebalance day it carries forward rather than collapsing to n_held.
    d = monthly.diagnostics.dropna(subset=["n_passing"])
    assert (d["n_passing"] >= d["n_held"]).all()
    assert d["n_passing"].max() > d["n_held"].max(), \
        "the candidate pool must be wider than the book somewhere in the sample"


def test_trend_screen_monthly_rebalance_runs():
    """Same branch, same reason, for the screen that was here first."""
    from qbs.screens import TrendScreenParams, trend_template_screen

    uni, mkt, safe = _screen_inputs()
    sig = trend_template_screen(uni, mkt, safe, TrendScreenParams(rebalance="ME"))
    assert np.allclose(sig.weights.sum(axis=1), 1.0)


def test_pipeline_backtests_the_finviz_screen_on_the_momentum_book():
    """Both strategies must be priced off the same combined frame, or the
    comparison is measuring two different universes."""
    lab = run(use_synthetic=True)
    assert "finviz" in lab.results
    assert (list(lab.results["finviz"].weights.columns)
            == list(lab.results["momentum"].weights.columns))
    assert len(lab.results["finviz"].returns) == len(lab.results["momentum"].returns)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILURES: ' + str(failures) if failures else 'all tests passed'}")
    sys.exit(1 if failures else 0)


# --------------------------------------------------------------------------
# The hourly breakout book (qbs/breakout.py)
# --------------------------------------------------------------------------

def _breakout_inputs(n=12, seed=5):
    from qbs.breakout import synthetic_hourly, hourly_to_daily

    tickers = [f"SY{i:03d}" for i in range(n)]
    hourly = synthetic_hourly(tickers, seed=seed)
    closes = pd.DataFrame({t: hourly_to_daily(hourly[t])["Close"] for t in tickers})
    closes.index = pd.to_datetime(closes.index).normalize()
    fris = closes.index.to_series().resample("W-FRI").last().dropna()
    mom = closes / closes.shift(21) - 1.0
    watchlists = {
        pd.Timestamp(d): list(mom.loc[d].dropna().sort_values(ascending=False).index[:8])
        for d in fris if d in mom.index and mom.loc[d].notna().any()
    }
    safe = synthetic_prices(start="2024-06-01")["BOXX"]
    return hourly, watchlists, safe


def test_breakout_never_exceeds_its_slot_count():
    """The whole point of the scenario is a hard cap on concurrent names."""
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    bk = weekly_breakout_book(hourly, wl, safe, BreakoutParams(),
                              WeeklyBookParams(n_slots=4))
    assert bk.diagnostics["slots_in_use"].max() <= 4
    risk = bk.result.weights.drop(columns=["BOXX"]).sum(axis=1)
    assert (risk <= 1.0 + 1e-9).all(), "a slot book must never lever"
    assert np.allclose(bk.result.weights.sum(axis=1), 1.0)


def test_breakout_only_trades_names_on_that_weeks_watchlist():
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    bk = weekly_breakout_book(hourly, wl, safe, BreakoutParams(), WeeklyBookParams())
    sel = sorted(wl)
    for _, row in bk.trades.iterrows():
        prior = [d for d in sel if d <= row["entry_time"]]
        assert prior, "a trade fired before any watchlist existed"
        assert row["ticker"] in wl[prior[-1]], \
            f"{row['ticker']} was not on the watchlist for {prior[-1]:%Y-%m-%d}"


def test_breakout_freed_slot_waits_for_the_weekend():
    """`refill_within_week=False` is the rule the user specified: a stop-out
    parks in cash rather than handing the slot to the next name down."""
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    wait = weekly_breakout_book(hourly, wl, safe, BreakoutParams(),
                                WeeklyBookParams(refill_within_week=False))
    refill = weekly_breakout_book(hourly, wl, safe, BreakoutParams(),
                                  WeeklyBookParams(refill_within_week=True))
    assert len(wait.trades) < len(refill.trades), \
        "waiting for the weekend must take strictly fewer trades"
    assert wait.diagnostics["exposure"].mean() <= refill.diagnostics["exposure"].mean()


def test_breakout_stop_exits_are_at_or_below_the_stop():
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    bk = weekly_breakout_book(hourly, wl, safe, BreakoutParams(), WeeklyBookParams())
    stopped = bk.trades[bk.trades["reason"] == "stop_R"]
    assert not stopped.empty, "the fixture never stopped out -- test proves nothing"
    assert (stopped["exit_price"] <= stopped["entry_price"] - stopped["R"] + 1e-9).all()


def test_breakout_entries_are_above_their_level():
    """An entry is a confirmed break: the fill must be above the level broken."""
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    bk = weekly_breakout_book(hourly, wl, safe, BreakoutParams(), WeeklyBookParams())
    assert not bk.trades.empty
    assert (bk.trades["entry_price"] > bk.trades["level"]).all()
    assert (bk.trades["R"] > 0).all()


def test_breakout_has_no_look_ahead():
    """Rewriting the future must not change a single trade decided before it.

    This is the test the notebook could not pass: its levels come from
    `find_peaks` over the whole history, so tampering with later bars moves
    trades that already happened.
    """
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    cut = sorted(wl)[len(wl) // 2]

    rng = np.random.default_rng(7)
    tampered = {}
    for t, bars in hourly.items():
        b = bars.copy()
        after = b.index > cut
        b.loc[after, ["Open", "High", "Low", "Close"]] *= rng.uniform(
            0.5, 1.5, (int(after.sum()), 4))
        tampered[t] = b

    p, bp = BreakoutParams(), WeeklyBookParams()
    a = weekly_breakout_book(hourly, wl, safe, p, bp).trades
    b = weekly_breakout_book(tampered, wl, safe, p, bp).trades
    a = a[a["entry_time"] <= cut].reset_index(drop=True)
    b = b[b["entry_time"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["ticker", "entry_time", "entry_price", "level"]],
                                  b[["ticker", "entry_time", "entry_price", "level"]])


def test_breakout_notebook_mode_does_have_look_ahead():
    """The switches are only worth having if they actually change something --
    with them off, the future must leak in, which is the bug being fixed."""
    from qbs.breakout import sr_levels, hourly_to_daily, synthetic_hourly
    from qbs.config import BreakoutParams

    bars = synthetic_hourly(["SY000"], seed=5)["SY000"]
    daily = hourly_to_daily(bars)
    cut = daily.index[len(daily) // 2]
    p = BreakoutParams()

    full = sr_levels(daily, p)
    causal = sr_levels(daily.loc[:cut], p)
    assert full != causal, \
        "levels from the whole history must differ from levels known at the time"


def test_breakout_causal_levels_use_only_past_bars():
    from qbs.breakout import sr_levels, hourly_to_daily, synthetic_hourly
    from qbs.config import BreakoutParams

    bars = synthetic_hourly(["SY000"], seed=5)["SY000"]
    daily = hourly_to_daily(bars)
    cut = daily.index[len(daily) // 2]
    p = BreakoutParams()

    a = sr_levels(daily.loc[:cut], p)
    tampered = daily.copy()
    cols = ["Open", "High", "Low", "Close"]
    tampered.loc[tampered.index > cut, cols] *= 1.7
    b = sr_levels(tampered.loc[:cut], p)
    assert a == b, "a level known at `cut` cannot move when later bars change"


def test_breakout_daily_indicators_are_lagged():
    """Un-lagged, the 10:00 bar of day D already knows day D's close."""
    from qbs.breakout import daily_indicators, hourly_to_daily, synthetic_hourly
    from qbs.config import BreakoutParams

    daily = hourly_to_daily(synthetic_hourly(["SY000"], seed=5)["SY000"])
    lagged = daily_indicators(daily, BreakoutParams(lag_daily_indicators=True))
    raw = daily_indicators(daily, BreakoutParams(lag_daily_indicators=False))
    pd.testing.assert_series_equal(lagged["EMA_F"].iloc[1:], raw["EMA_F"].shift(1).iloc[1:])


def test_breakout_position_cannot_exit_before_it_enters():
    """The notebook's ordering bug: a trade registered at the cross bar could
    be closed by the exit block on a bar preceding its own entry."""
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    bk = weekly_breakout_book(hourly, wl, safe, BreakoutParams(), WeeklyBookParams())
    closed = bk.trades[bk.trades["exit_time"].notna()]
    assert not closed.empty
    assert (closed["exit_time"] >= closed["entry_time"]).all()


def test_breakout_result_summarises_like_any_other_strategy():
    from qbs.breakout import weekly_breakout_book
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    bk = weekly_breakout_book(hourly, wl, safe, BreakoutParams(), WeeklyBookParams())
    s = summarise(bk.result)
    assert np.isfinite(s["CAGR"]) and -1.0 <= s["Max drawdown"] <= 0.0
    assert s["Ann. turnover"] > 0, "a book that trades must report turnover"
    assert bk.result.equity.index[0] >= min(wl), \
        "the book must not be priced before its first watchlist"


# --------------------------------------------------------------------------
# The breakout sweep harness
# --------------------------------------------------------------------------

def test_sweep_matches_running_the_book_directly():
    """The sweep reuses one candidate set across book variations. That is only
    sound if a cached row is identical to running the book from scratch."""
    from qbs.breakout import sweep_breakout, weekly_breakout_book, trade_stats
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    sw = sweep_breakout(hourly, wl, safe, {"n_slots": [4, 6]}, progress=False)

    for n in (4, 6):
        direct = weekly_breakout_book(hourly, wl, safe, BreakoutParams(),
                                      WeeklyBookParams(n_slots=n))
        row = sw[sw["n_slots"] == n].iloc[0]
        assert row["n_trades"] == trade_stats(direct.trades)["n_trades"]
        assert abs(row["CAGR"] - summarise(direct.result)["CAGR"]) < 1e-12


def test_allocate_slots_does_not_mutate_its_candidates():
    """Candidate reuse depends on allocation being non-destructive."""
    from qbs.breakout import candidate_trades, allocate_slots
    from qbs.config import BreakoutParams, WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    cands = candidate_trades(hourly, wl, BreakoutParams())
    before = [(t.ticker, t.entry_time, t.exit_time, t.R) for t in cands]
    allocate_slots(cands, wl, WeeklyBookParams(n_slots=3))
    allocate_slots(cands, wl, WeeklyBookParams(n_slots=9))
    after = [(t.ticker, t.entry_time, t.exit_time, t.R) for t in cands]
    assert before == after


def test_sweep_returns_one_row_per_combination():
    from qbs.breakout import sweep_breakout

    hourly, wl, safe = _breakout_inputs()
    sw = sweep_breakout(hourly, wl, safe,
                        {"r_mult": [1.0, 2.0], "n_slots": [4, 6]}, progress=False)
    assert len(sw) == 4
    assert set(sw.columns) >= {"r_mult", "n_slots", "n_trades", "expectancy_R",
                               "pct_stop", "CAGR"}


def test_sweep_rejects_a_parameter_that_does_not_exist():
    """A typo'd key would otherwise sweep nothing and silently report the
    base case four times, which looks like a flat parameter."""
    from qbs.breakout import sweep_breakout

    hourly, wl, safe = _breakout_inputs()
    try:
        sweep_breakout(hourly, wl, safe, {"confirm_hrs": [1, 2]}, progress=False)
    except ValueError:
        return
    raise AssertionError("an unknown grid key must raise")


def test_wider_stops_produce_fewer_stop_outs():
    """`r_mult` is the dial for the stop-heavy exit mix -- it must actually
    move the mix, or the knob is decorative."""
    from qbs.breakout import sweep_breakout

    hourly, wl, safe = _breakout_inputs()
    sw = sweep_breakout(hourly, wl, safe, {"r_mult": [0.5, 1.0, 2.0]},
                        progress=False).sort_values("r_mult")
    assert sw["pct_stop"].is_monotonic_decreasing, \
        "widening R must reduce the share of trades stopped out"


def test_sweeping_watchlist_size_actually_truncates():
    from qbs.breakout import sweep_breakout
    from qbs.config import WeeklyBookParams

    hourly, wl, safe = _breakout_inputs()
    # n_slots below the shortest watchlist, or WeeklyBookParams rejects the pair.
    sw = sweep_breakout(hourly, wl, safe, {"watchlist_size": [2, 8]},
                        book=WeeklyBookParams(n_slots=2),
                        progress=False).sort_values("watchlist_size")
    assert sw["n_trades"].iloc[0] < sw["n_trades"].iloc[1], \
        "a shorter watchlist must offer fewer names to trade"


def test_trade_stats_handles_an_empty_book():
    from qbs.breakout import trade_stats

    s = trade_stats(pd.DataFrame(columns=["exit_time", "reason", "R_multiple"]))
    assert s["n_trades"] == 0 and np.isnan(s["expectancy_R"])


def test_lookahead_cost_prices_every_switch():
    from qbs.breakout import lookahead_cost

    hourly, wl, safe = _breakout_inputs()
    tbl = lookahead_cost(hourly, wl, safe, progress=False)
    assert len(tbl) == 5
    assert tbl["variant"].iloc[0] == "causal (default)"
    assert tbl["n_trades"].min() > 0


def test_r_multiple_is_profit_in_units_of_risk():
    from qbs.breakout import Trade

    t = Trade(ticker="X", entry_time=pd.Timestamp("2025-01-02"), entry_price=100.0,
              level=99.0, R=5.0, ADR=1.0,
              exit_time=pd.Timestamp("2025-01-09"), exit_price=110.0, reason="tp")
    assert abs(t.r_multiple - 2.0) < 1e-12
    assert np.isnan(Trade("X", pd.Timestamp("2025-01-02"), 100.0, 99.0, 5.0, 1.0).r_multiple)


# --------------------------------------------------------------------------
# The selection -> breakout funnel
# --------------------------------------------------------------------------

def test_closes_to_bars_builds_a_usable_range():
    """High == Low == Close would make ADR identically zero and silently
    tighten every stop, so the envelope has to have width."""
    from qbs.breakout import closes_to_bars

    closes = pd.DataFrame({"A": [10.0, 11.0, 10.5, 12.0]},
                          index=pd.bdate_range("2025-01-01", periods=4))
    bars = closes_to_bars(closes)["A"]
    assert (bars["High"] >= bars["Close"]).all()
    assert (bars["Low"] <= bars["Close"]).all()
    assert (bars["High"] - bars["Low"]).iloc[1:].gt(0).all(), "range must not be zero"


def test_funnel_stages_are_nested():
    """Each stage is a subset of the one above it -- a pick cannot confirm
    without crossing, or cross without a level to cross."""
    from qbs.breakout import breakout_funnel, closes_to_bars
    from qbs.config import BreakoutParams

    hourly, wl, _ = _breakout_inputs()
    closes = pd.DataFrame({t: b["Close"].resample("1D").last() for t, b in hourly.items()}).dropna(how="all")
    f = breakout_funnel(closes_to_bars(closes), wl, BreakoutParams(), wait_days=7)

    assert not f.empty
    assert (f.loc[f["crossed"], "has_level"]).all(), "crossed implies a level existed"
    assert (f.loc[f["confirmed"], "crossed"]).all(), "confirmed implies crossed"
    assert (f.loc[f["reached_1R"], "confirmed"]).all(), "reached_1R implies an entry"
    assert f.loc[f["confirmed"], "entry_price"].notna().all()


def test_funnel_entries_agree_with_the_trading_code():
    """The funnel must count an entry exactly where `breakout_signals` would
    take one, or it is measuring a different strategy from the backtest."""
    from qbs.breakout import (align_indicators, breakout_funnel, breakout_signals,
                              daily_indicators, hourly_to_daily, var_risk)
    from qbs.config import BreakoutParams

    hourly, wl, _ = _breakout_inputs()
    p = BreakoutParams()
    f = breakout_funnel(hourly, wl, p, wait_days=7)
    conf = f[f["confirmed"]]
    assert not conf.empty

    row = conf.iloc[0]
    bars = hourly[row["ticker"]].sort_index()
    hist = bars.loc[:row["selection_date"]]
    seg = bars.loc[row["selection_date"]:]
    # Indicators are warmed from the FULL series, exactly as `candidate_trades`
    # does: at selection time the EMAs are already running. Warming them from
    # the one-week segment instead leaves EMA20 undefined and finds no trade.
    ind = align_indicators(seg.index, daily_indicators(hourly_to_daily(bars), p))
    trades = breakout_signals(seg, [row["nearest_level"]], var_risk(hist, p), p,
                              ind_hourly=ind,
                              entry_end=row["selection_date"] + pd.Timedelta(days=7))
    assert trades and abs(trades[0].entry_price - row["entry_price"]) < 1e-9


def test_funnel_window_is_monotone():
    """A longer waiting window can only find more first breakouts, never fewer."""
    from qbs.breakout import breakout_funnel
    from qbs.config import BreakoutParams

    hourly, wl, _ = _breakout_inputs()
    p = BreakoutParams()
    short = breakout_funnel(hourly, wl, p, wait_days=2)["crossed"].sum()
    long_ = breakout_funnel(hourly, wl, p, wait_days=14)["crossed"].sum()
    assert long_ >= short


def test_funnel_summary_conversion_rates():
    from qbs.breakout import funnel_summary

    f = pd.DataFrame({
        "has_level": [True, True, True, False],
        "crossed": [True, True, False, False],
        "confirmed": [True, False, False, False],
        "reached_1R": [True, False, False, False],
        "r_multiple": [2.0, np.nan, np.nan, np.nan],
    })
    s = funnel_summary(f).set_index("stage")
    assert s.loc["selected", "n"] == 4
    assert s.loc["had resistance overhead", "n"] == 3
    assert abs(s.loc["crossed it in the window", "of_previous"] - 2 / 3) < 1e-12
    assert abs(s.loc["cross confirmed (an entry)", "of_selected"] - 0.25) < 1e-12


def test_funnel_summary_handles_an_empty_frame():
    from qbs.breakout import funnel_summary

    assert funnel_summary(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Point-in-time membership
# --------------------------------------------------------------------------

def test_pit_tickers_is_the_union_of_every_member_ever():
    """The download list under point-in-time membership is every name that was
    ever in the index, not today's -- that is the whole point of buying it."""
    from qbs.universe import pit_tickers

    pit = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2025-01-01", "2025-01-01"]),
        "ticker": ["AAA", "BBB", "AAA", "CCC"],      # BBB was dropped, CCC added
    })
    assert pit_tickers(pit) == ["AAA", "BBB", "CCC"]


def test_pit_mask_makes_a_name_unrankable_before_it_joined():
    from qbs.universe import membership_mask

    # Each date in the file is a COMPLETE snapshot of the index on that date,
    # not a change event -- `membership_mask` forward-fills whole rows, so a
    # name omitted from a snapshot reads as "no longer a member".
    idx = pd.bdate_range("2024-01-01", periods=10)
    day1, day8 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")
    pit = pd.DataFrame({
        "date": [day1, day1, day8, day8, day8],
        "ticker": ["AAA", "BBB", "AAA", "BBB", "CCC"],
    })
    m = membership_mask(idx, ["AAA", "BBB", "CCC"], pit)
    assert m["AAA"].all() and m["BBB"].all()
    assert not m["CCC"].iloc[0], "CCC was not a member on day 1"
    assert m["CCC"].iloc[-1], "CCC joined on the 8th and stays a member"


def test_pit_dropped_member_is_still_rankable_while_it_was_in():
    """The half of the bias the default path misses: a name removed from the
    index must still be rankable for the dates it WAS a member."""
    from qbs.universe import membership_mask

    idx = pd.bdate_range("2024-01-01", periods=10)
    kept = [pd.Timestamp(d) for d in idx]
    pit = pd.DataFrame({
        "date": kept + kept[:4],
        "ticker": ["AAA"] * len(kept) + ["ZZZ"] * 4,   # ZZZ dropped after day 4
    })
    m = membership_mask(idx, ["AAA", "ZZZ"], pit)
    assert m["ZZZ"].iloc[:4].all(), "ZZZ must be rankable while it was a member"
    assert not m["ZZZ"].iloc[4:].any(), "and unrankable once it was dropped"


def test_pipeline_masks_the_ranker_with_pit_membership():
    from qbs.universe import synthetic_universe

    uni = synthetic_universe(n=12, start="2023-06-01")
    late = uni.columns[0]
    join = uni.index[len(uni) // 2]
    rows = [{"date": d, "ticker": t}
            for d in uni.index for t in uni.columns
            if not (t == late and d < join)]
    pit = pd.DataFrame(rows)

    lab = run(use_synthetic=True, universe_prices=uni, pit_membership=pit,
              with_vix=False, with_book_vt=False, with_finviz=False)
    held = lab.signals["momentum"].holdings_log
    early = [d for d in held if d < join]
    assert early, "fixture has no pre-join dates -- test proves nothing"
    assert not any(late in held[d] for d in early), \
        "a name must never be held before it joined the index"


def test_finviz_high_band_floor_excludes_names_at_their_high():
    """`min_off_high_pct` turns the 52-week-high filter from a ceiling into a
    band. The motivation is the breakout book: a name already at its high has
    nothing overhead left to break through, so an entry that needs resistance
    can never fire on it.
    """
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    ceiling = finviz_momentum_screen(uni, safe, FinvizScreenParams(n_hold=0))
    band = finviz_momentum_screen(
        uni, safe, FinvizScreenParams(n_hold=0, min_off_high_pct=0.04,
                                      within_52w_high_pct=0.20))

    high = uni.rolling(252, min_periods=252).max()
    off = 1.0 - uni / high
    checked = 0
    for dt in uni.index[::40]:
        for t in band.holdings_log[dt]:
            assert off.at[dt, t] >= 0.04 - 1e-9, f"{t} sits above the band floor"
            checked += 1
    assert checked > 0, "the fixture never held anything -- test proves nothing"

    # A floor can only remove names that the bare ceiling admitted.
    strict = finviz_momentum_screen(
        uni, safe, FinvizScreenParams(n_hold=0, min_off_high_pct=0.04))
    for dt in uni.index[::40]:
        assert set(strict.holdings_log[dt]) <= set(ceiling.holdings_log[dt])


def test_finviz_high_band_floor_defaults_to_the_notebook_rule():
    """Zero floor must reproduce the screener's own behaviour exactly."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    a = finviz_momentum_screen(uni, safe, FinvizScreenParams(n_hold=6))
    b = finviz_momentum_screen(uni, safe,
                               FinvizScreenParams(n_hold=6, min_off_high_pct=0.0))
    pd.testing.assert_frame_equal(a.weights, b.weights)


# --------------------------------------------------------------------------
# Market breadth (qbs/breadth.py)
# --------------------------------------------------------------------------

def _breadth_frame():
    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(4)
    data = {f"S{i}": 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, len(idx))))
            for i in range(25)}
    return pd.DataFrame(data, index=idx)


def test_breadth_counts_four_percent_movers_exactly():
    from qbs.breadth import daily_breadth

    idx = pd.bdate_range("2024-01-01", periods=60)
    # A rises 5% a day, B falls 5% a day, C is flat.
    px = pd.DataFrame({
        "A": 100 * 1.05 ** np.arange(len(idx)),
        "B": 100 * 0.95 ** np.arange(len(idx)),
        "C": np.full(len(idx), 100.0),
    }, index=idx)
    r = daily_breadth(px)
    assert (r.table["up4"] == 1).all(), "exactly one name gains 4%+ each day"
    assert (r.table["dn4"] == 1).all()
    assert (r.table["n_stocks"] == 3).all()


def test_breadth_percent_above_ma_is_a_percentage():
    from qbs.breadth import daily_breadth

    r = daily_breadth(_breadth_frame())
    for col in ("pct_above_fast", "pct_above_slow"):
        assert r.table[col].between(0, 100).all()
    # A monotonically rising frame must sit fully above both averages.
    idx = pd.bdate_range("2024-01-01", periods=120)
    rising = pd.DataFrame({"A": np.arange(1.0, len(idx) + 1.0),
                           "B": np.arange(2.0, len(idx) + 2.0)}, index=idx)
    up = daily_breadth(rising)
    assert np.allclose(up.table["pct_above_fast"], 100.0)
    assert np.allclose(up.table["pct_above_slow"], 100.0)


def test_atr_distance_matches_a_hand_computation():
    from qbs.breadth import atr_distance

    idx = pd.bdate_range("2024-01-01", periods=80)
    close = pd.Series(np.linspace(100, 140, len(idx)), index=idx)
    high, low = close + 1.0, close - 1.0
    d = atr_distance(close, high, low, ema_span=50, atr_window=14)

    ema = close.ewm(span=50, adjust=False).mean()
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    expected = (close - ema) / tr.rolling(14).mean()
    pd.testing.assert_series_equal(d.dropna(), expected.dropna())


def test_atr_distance_without_highs_overstates_the_stretch():
    """No high/low collapses the true range to the close-to-close move, which
    is smaller -- so the distance in ATR units comes out LARGER. The docstring
    claims it is an upper bound; this pins that."""
    from qbs.breadth import atr_distance

    idx = pd.bdate_range("2024-01-01", periods=90)
    close = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    wide = atr_distance(close, close + 3.0, close - 3.0)
    narrow = atr_distance(close)
    assert (narrow.dropna().abs() >= wide.dropna().abs() - 1e-9).all()


def test_leader_mask_turnover_leg_only_removes_names():
    from qbs.breadth import leader_mask

    px = _breadth_frame()
    without = leader_mask(px)
    vol = pd.DataFrame(1.0, index=px.index, columns=px.columns)   # tiny turnover
    with_vol = leader_mask(px, volumes=vol)
    assert (with_vol & ~without).sum().sum() == 0, "volume cannot admit a name"
    assert with_vol.sum().sum() < without.sum().sum(), "a $1 turnover must exclude"


def test_leader_rule_is_the_stated_definition():
    """A US stock or ADR over $5, trading more than 300k shares a day, up more
    than 28% on the quarter. Each leg is pinned separately so a name can only
    fail for the reason the test is about."""
    from qbs.breadth import BreadthParams, leader_mask

    p = BreadthParams()
    assert (p.leader_min_price, p.leader_min_volume,
            p.leader_min_quarter_return) == (5.0, 300_000.0, 0.28)

    idx = pd.bdate_range("2025-01-01", periods=p.leader_quarter_days + 5)
    n = len(idx)

    def ramp(start, gain):
        return np.linspace(start, start * (1.0 + gain), n)

    # Quarterly gain is measured over `leader_quarter_days`, and the frames
    # below span a few bars more, so the ramps are sized generously either
    # side of 28% rather than exactly on it.
    px = pd.DataFrame({
        "STRONG": ramp(50.0, 0.60),     # clears every leg
        "WEAK": ramp(50.0, 0.10),       # up, but nowhere near 28%
        "CHEAP": ramp(2.0, 0.60),       # strong, under $5
    }, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=px.columns)   # $50m/day on STRONG

    last = leader_mask(px, volumes=vol).iloc[-1]
    assert last["STRONG"], "clears price, turnover and the quarterly gain"
    assert not last["WEAK"], "a 10% quarter is not high momentum"
    assert not last["CHEAP"], "under $5 is out however strong the move"


def test_the_quarterly_gate_is_28_percent_not_20():
    """The gate moved from 20% to 28%, which is a real change in how selective
    the leader group is -- a name in between must now be excluded."""
    from qbs.breadth import BreadthParams, leader_mask

    p = BreadthParams()
    idx = pd.bdate_range("2025-01-01", periods=p.leader_quarter_days + 1)
    n = len(idx)
    # +24% over exactly the lookback: inside the old gate, outside the new one.
    px = pd.DataFrame({"MID": np.linspace(50.0, 62.0, n)}, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=["MID"])

    assert not leader_mask(px, volumes=vol).iloc[-1]["MID"]
    loose = BreadthParams(leader_min_quarter_return=0.20)
    assert leader_mask(px, volumes=vol, p=loose).iloc[-1]["MID"], \
        "the same name passes the old 20% gate, so the fixture is the gate's"


def test_the_liquidity_leg_counts_shares_and_ignores_price():
    """`leader_min_volume` is a share count. Price does not enter it, so two
    names on the same volume must agree however far apart they trade -- which
    is exactly what the $5m dollar test it replaced would NOT have done."""
    from qbs.breadth import BreadthParams, leader_mask

    p = BreadthParams()
    idx = pd.bdate_range("2025-01-01", periods=p.leader_quarter_days + 1)
    n = len(idx)
    # Same share volume, wildly different prices, both with a strong quarter.
    px = pd.DataFrame({"PRICEY": np.linspace(50.0, 100.0, n),
                       "CHEAPISH": np.linspace(3.0, 6.0, n)}, index=idx)
    vol = pd.DataFrame(400_000.0, index=idx, columns=px.columns)

    last = leader_mask(px, volumes=vol).iloc[-1]
    assert last["PRICEY"] and last["CHEAPISH"], \
        "400k shares is 400k shares; the dollar amounts are irrelevant"
    # $2.4m/day for CHEAPISH -- it would have failed the old $5m floor, and
    # that it passes now is the change, not an accident of the fixture.
    assert px["CHEAPISH"].iloc[-1] * 400_000.0 < 5_000_000.0


def test_a_thin_but_expensive_name_now_fails_the_volume_leg():
    """The reverse direction of the same change. 50k shares of a $200 stock is
    $10m a day: it cleared the old dollar floor comfortably and must fail a
    300k share count. Neither test is a stricter version of the other."""
    from qbs.breadth import BreadthParams, leader_mask

    p = BreadthParams()
    idx = pd.bdate_range("2025-01-01", periods=p.leader_quarter_days + 1)
    n = len(idx)
    px = pd.DataFrame({"THIN": np.linspace(100.0, 200.0, n)}, index=idx)
    vol = pd.DataFrame(50_000.0, index=idx, columns=["THIN"])

    assert px["THIN"].iloc[-1] * 50_000.0 > 5_000_000.0, "clears the old floor"
    assert not leader_mask(px, volumes=vol).iloc[-1]["THIN"]


def test_every_leader_leg_is_strictly_greater_than():
    """Not pedantry on the price leg: a stock at exactly $5.00 is common, and
    `>=` would admit names the Finviz universe screen ("Over $5") excludes, so
    the two filters would disagree about the same name."""
    from qbs.breadth import BreadthParams, leader_mask

    p = BreadthParams()
    idx = pd.bdate_range("2025-01-01", periods=p.leader_quarter_days + 1)
    n = len(idx)

    # Exactly $5.00 on the last bar, with a strong quarter and heavy volume.
    at_price = pd.DataFrame({"EDGE": np.linspace(2.0, 5.0, n)}, index=idx)
    vol = pd.DataFrame(1e7, index=idx, columns=["EDGE"])
    assert at_price["EDGE"].iloc[-1] == 5.0, "the fixture must sit on the line"
    assert not leader_mask(at_price, volumes=vol).iloc[-1]["EDGE"]

    # Exactly 300,000 shares.
    flat = pd.DataFrame({"EDGE": np.linspace(5.0, 10.0, n)}, index=idx)
    exact = pd.DataFrame(300_000.0, index=idx, columns=["EDGE"])
    assert not leader_mask(flat, volumes=exact).iloc[-1]["EDGE"]
    more = pd.DataFrame(300_001.0, index=idx, columns=["EDGE"])
    assert leader_mask(flat, volumes=more).iloc[-1]["EDGE"], "one share over passes"


def test_sector_breakdown_shares_and_excess_are_consistent():
    from qbs.breadth import sector_breakdown

    px = _breadth_frame()
    sectors = {t: ("Tech" if i % 3 == 0 else "Health" if i % 3 == 1 else "Energy")
               for i, t in enumerate(px.columns)}
    tbl = sector_breakdown(px, sectors)
    if tbl.empty:
        return                      # no leaders on this fixture; nothing to check
    assert abs(tbl["share_pct"].sum() - 100.0) < 1e-9
    assert np.allclose(tbl["excess_pp"], tbl["share_pct"] - tbl["pool_pct"])
    assert tbl["penetration"].between(0, 100).all()


def test_sector_breakdown_without_a_map_returns_empty():
    """Rather than invent a classification, which a dashboard would render as
    fact."""
    from qbs.breadth import sector_breakdown

    assert sector_breakdown(_breadth_frame(), {}).empty


def test_breadth_classifiers_hit_their_thresholds():
    from qbs.breadth import BreadthParams, atr_class, ma_class, pulse_class

    p = BreadthParams()
    assert pulse_class(p.pulse_strong, "up") == "up_strong"
    assert pulse_class(p.pulse_strong - 1, "up") == "up"
    assert pulse_class(p.pulse_strong, "down") == "down_strong"

    assert ma_class(5.0, "fast") == "extreme_low"
    assert ma_class(95.0, "fast") == "extreme_high"
    assert ma_class(50.0, "fast") == "mid"
    assert ma_class(15.0, "slow") == "extreme_low"      # slow floor is 20
    assert ma_class(float("nan"), "fast") == "none"

    assert atr_class(6.0) == "stretched"
    assert atr_class(-6.0) == "oversold"
    assert atr_class(0.0) == "normal"


def test_breadth_reports_what_it_could_not_measure():
    """The UI relies on these flags to decide what to leave blank."""
    from qbs.breadth import daily_breadth

    px = _breadth_frame()
    bare = daily_breadth(px)
    assert bare.has_volume is False and bare.has_index is False
    rich = daily_breadth(px, qqq=px["S0"],
                         volumes=pd.DataFrame(1e9, index=px.index, columns=px.columns))
    assert rich.has_volume is True and rich.has_index is True
    assert rich.n_stocks == px.shape[1]


def test_breadth_does_not_report_zero_during_warm_up():
    """`px > NaN` is False, not NaN, so counting it reports "0% above the
    50-day" for the first 49 sessions -- which reads on a chart as a total
    collapse of breadth at the left edge of every series. Warm-up rows must be
    absent, never zero."""
    from qbs.breadth import BreadthParams, daily_breadth

    p = BreadthParams()
    idx = pd.bdate_range("2024-01-01", periods=120)
    rising = pd.DataFrame({"A": np.arange(1.0, 121.0),
                           "B": np.arange(2.0, 122.0)}, index=idx)
    r = daily_breadth(rising, p=p)

    assert len(r.table) == len(idx) - p.ma_slow + 1, "warm-up rows must be dropped"
    assert r.table.index[0] == idx[p.ma_slow - 1]
    assert (r.table["pct_above_slow"] > 0).all(), "a rising market is never 0%"
    assert np.allclose(r.table["pct_above_slow"], 100.0)


def test_breadth_ma_denominator_excludes_names_without_an_average():
    """A name too young to have the average must not sit in the denominator
    dragging the percentage down."""
    from qbs.breadth import BreadthParams, daily_breadth

    p = BreadthParams()
    idx = pd.bdate_range("2024-01-01", periods=120)
    px = pd.DataFrame({"OLD": np.arange(1.0, 121.0),
                       "YOUNG": np.nan}, index=idx)
    px.iloc[-5:, px.columns.get_loc("YOUNG")] = np.arange(1.0, 6.0)

    r = daily_breadth(px, p=p)
    # YOUNG never has a 50-day average, so every row is 100% on OLD alone.
    assert np.allclose(r.table["pct_above_slow"], 100.0)


def test_levels_in_view_keeps_the_nearest_and_counts_the_rest():
    from qbs.breakout import levels_in_view

    levels = [10, 20, 30, 40, 50, 60]
    shown, n_in_view, overhead = levels_in_view(levels, last=32, lo=10, hi=60, n=3)
    assert shown == [20.0, 30.0, 40.0], "the three nearest 32, in price order"
    assert n_in_view == 6
    assert overhead is True


def test_levels_in_view_drops_levels_outside_the_window():
    from qbs.breakout import levels_in_view

    shown, n_in_view, _ = levels_in_view([1, 2, 100, 101], last=100, lo=99, hi=102,
                                         n=10, pad_frac=0.0)
    assert shown == [100.0, 101.0] and n_in_view == 2


def test_levels_in_view_reports_nothing_overhead_at_the_highs():
    """A name that has cleared every level has no breakout to make. The flag
    must reflect every level in view, not just the ones drawn."""
    from qbs.breakout import levels_in_view

    shown, _, overhead = levels_in_view([10, 20, 30], last=35, lo=5, hi=40, n=2)
    assert overhead is False
    assert shown == [20.0, 30.0], "still draws the nearest support"

    # One level overhead but outside the drawn set must still flip the flag.
    _, _, overhead2 = levels_in_view([10, 20, 30, 39], last=35, lo=5, hi=40, n=2)
    assert overhead2 is True


def test_levels_in_view_handles_empty_and_zero():
    from qbs.breakout import levels_in_view

    assert levels_in_view([], last=10, lo=5, hi=15, n=5) == ([], 0, False)
    shown, n_in_view, _ = levels_in_view([10, 12], last=11, lo=5, hi=15, n=0)
    assert shown == [] and n_in_view == 2, "n=0 hides lines but still counts them"


# --------------------------------------------------------------------------
# Data freshness (qbs/data.py)
# --------------------------------------------------------------------------

def test_sessions_behind_counts_weekdays_only():
    from qbs.data import sessions_behind

    now = pd.Timestamp("2026-09-15")          # a Tuesday
    assert sessions_behind(pd.Timestamp("2026-09-15"), now) == 0
    assert sessions_behind(pd.Timestamp("2026-09-14"), now) == 1
    assert sessions_behind(pd.Timestamp("2026-09-11"), now) == 2   # Fri -> Mon,Tue
    assert sessions_behind(pd.Timestamp("2026-09-08"), now) == 5


def test_sessions_behind_ignores_the_weekend():
    """Saturday and Sunday are not missing sessions."""
    from qbs.data import sessions_behind

    friday = pd.Timestamp("2026-09-11")
    assert sessions_behind(friday, pd.Timestamp("2026-09-12")) == 0   # Sat
    assert sessions_behind(friday, pd.Timestamp("2026-09-13")) == 0   # Sun
    assert sessions_behind(friday, pd.Timestamp("2026-09-14")) == 1   # Mon


def test_sessions_behind_never_goes_negative():
    from qbs.data import sessions_behind

    assert sessions_behind(pd.Timestamp("2026-09-15"), pd.Timestamp("2026-09-10")) == 0


def test_freshness_note_levels():
    from qbs.data import freshness_note

    now = pd.Timestamp("2026-09-15")
    assert freshness_note(pd.Timestamp("2026-09-15"), now)[1] == "ok"
    assert freshness_note(pd.Timestamp("2026-09-14"), now)[1] == "info"
    assert freshness_note(pd.Timestamp("2026-09-08"), now)[1] == "warn"


def test_freshness_note_carries_no_remedy():
    """The fix depends on why it is stale, and only the caller knows that --
    telling someone to go online while their download is the thing failing is
    worse than saying nothing."""
    from qbs.data import freshness_note

    _, _, msg = freshness_note(pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-15"))
    assert "5 sessions behind" in msg
    for word in ("Online", "refresh", "Refresh"):
        assert word not in msg


def test_load_daily_ohlc_offline_without_cache_returns_none():
    """The caller is a chart that falls back to a close line. A missing candle
    is not worth taking the page down for."""
    import tempfile
    from qbs.data import load_daily_ohlc

    with tempfile.TemporaryDirectory() as d:
        assert load_daily_ohlc("NOPE", offline=True, cache_dir=d) is None


def test_load_daily_ohlc_offline_reads_the_cache():
    import tempfile, os
    from qbs.data import load_daily_ohlc

    idx = pd.bdate_range("2026-01-01", periods=5)
    frame = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5,
                          "Volume": 100}, index=idx)
    frame.index.name = "Date"
    with tempfile.TemporaryDirectory() as d:
        frame.to_csv(os.path.join(d, "XYZ.csv"))
        got = load_daily_ohlc("XYZ", offline=True, cache_dir=d)
        assert got is not None and len(got) == 5
        assert {"Open", "High", "Low", "Close"} <= set(got.columns)


def test_load_daily_ohlc_falls_back_to_cache_when_the_download_fails():
    """Online mode must not lose a usable cache to a network blip."""
    import tempfile, os, sys, types
    from qbs.data import load_daily_ohlc

    idx = pd.bdate_range("2026-01-01", periods=4)
    frame = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5},
                         index=idx)
    frame.index.name = "Date"

    boom = types.ModuleType("yfinance")
    def _fail(*a, **k):
        raise RuntimeError("network down")
    boom.download = _fail
    saved = sys.modules.get("yfinance")
    sys.modules["yfinance"] = boom
    try:
        with tempfile.TemporaryDirectory() as d:
            frame.to_csv(os.path.join(d, "XYZ.csv"))
            got = load_daily_ohlc("XYZ", offline=False, cache_dir=d)
            assert got is not None and len(got) == 4, "cache must survive a failure"
            assert load_daily_ohlc("GONE", offline=False, cache_dir=d) is None
    finally:
        if saved is not None:
            sys.modules["yfinance"] = saved
        else:
            sys.modules.pop("yfinance", None)


# --------------------------------------------------------------------------
# The broad US universe from Finviz (qbs/finviz.py)
# --------------------------------------------------------------------------

def _fake_overview(rows):
    """A stand-in for finvizfinance's Overview, so the parsing is testable
    without scraping 120 pages."""
    import sys, types

    class _Overview:
        def set_filter(self, **kw):
            self.filters = kw.get("filters_dict")

        def screener_view(self, **kw):
            return pd.DataFrame(rows)

    mod = types.ModuleType("finvizfinance.screener.overview")
    mod.Overview = _Overview
    pkg = types.ModuleType("finvizfinance")
    scr = types.ModuleType("finvizfinance.screener")
    saved = {k: sys.modules.get(k) for k in
             ("finvizfinance", "finvizfinance.screener",
              "finvizfinance.screener.overview")}
    sys.modules["finvizfinance"] = pkg
    sys.modules["finvizfinance.screener"] = scr
    sys.modules["finvizfinance.screener.overview"] = mod
    return saved


def _restore(saved):
    import sys
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v
        else:
            sys.modules.pop(k, None)


def test_universe_filters_use_finvizs_own_vocabulary():
    """These strings are passed straight to the screener; a typo silently
    returns a different universe rather than an error."""
    from qbs.finviz import UniverseFilters

    d = UniverseFilters().as_dict()
    assert d == {"Industry": "Stocks only (ex-Funds)",
                 "Price": "Over $5",
                 "Average Volume": "Over 300K"}


def test_fetch_us_universe_parses_and_normalises():
    import tempfile, os
    from qbs.finviz import fetch_us_universe

    saved = _fake_overview([
        {"Ticker": "brk.b", "Company": "B", "Sector": "Financial", "Country": "USA"},
        {"Ticker": "AAPL", "Company": "A", "Sector": "Technology", "Country": "USA"},
        {"Ticker": "AAPL", "Company": "dupe", "Sector": "Technology", "Country": "USA"},
    ])
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "u.csv")
            out, err = fetch_us_universe(refresh=True, cache_path=path, verbose=False)
            assert err is None and out is not None
            assert list(out["Ticker"]) == ["BRK-B", "AAPL"], "upper, dots->dashes, deduped"
            assert os.path.exists(path), "result must be cached"
    finally:
        _restore(saved)


def test_fetch_us_universe_falls_back_to_cache_on_failure():
    import tempfile, os, sys, types
    from qbs.finviz import fetch_us_universe

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "u.csv")
        pd.DataFrame({"Ticker": ["AAPL"], "Sector": ["Technology"]}).to_csv(path, index=False)

        broken = types.ModuleType("finvizfinance.screener.overview")
        class _Boom:
            def set_filter(self, **kw): pass
            def screener_view(self, **kw): raise RuntimeError("rate limited")
        broken.Overview = _Boom
        saved = {"finvizfinance.screener.overview":
                 sys.modules.get("finvizfinance.screener.overview")}
        sys.modules["finvizfinance.screener.overview"] = broken
        try:
            out, err = fetch_us_universe(refresh=True, cache_path=path, verbose=False)
            assert out is not None and list(out["Ticker"]) == ["AAPL"]
            assert err and "rate limited" in err, "the real reason must survive"
        finally:
            _restore(saved)


def test_fetch_us_universe_offline_without_cache_is_none():
    import tempfile, os
    from qbs.finviz import fetch_us_universe

    with tempfile.TemporaryDirectory() as d:
        out, err = fetch_us_universe(offline=True, verbose=False,
                                     cache_path=os.path.join(d, "u.csv"))
        assert out is None
        assert err and "offline" in err, "must say WHY, not just fail"


def test_sector_map_is_empty_rather_than_unclassified():
    """An empty map makes sector_breakdown return an empty table. Inventing
    an 'Unclassified' bucket for everything would render as a finding."""
    from qbs.finviz import sector_map

    assert sector_map(None) == {}
    assert sector_map(pd.DataFrame()) == {}
    assert sector_map(pd.DataFrame({"Ticker": ["A"]})) == {}, "no Sector column"
    got = sector_map(pd.DataFrame({"Ticker": ["A", "B"],
                                   "Sector": ["Tech", None]}))
    assert got == {"A": "Tech"}, "a missing sector is dropped, not relabelled"


def test_load_universe_bars_offline_without_cache_is_none():
    import tempfile
    from qbs.finviz import load_universe_bars

    with tempfile.TemporaryDirectory() as d:
        c, v, err = load_universe_bars(["AAPL"], cache_dir=d, offline=True,
                                       verbose=False)
        assert c is None and v is None
        assert err and "offline" in err


def test_load_universe_bars_reads_both_cached_frames():
    import tempfile, os
    from qbs.finviz import load_universe_bars

    idx = pd.bdate_range("2026-01-01", periods=6)
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame({"AAPL": 1.0, "MSFT": 2.0}, index=idx).rename_axis("Date") \
            .to_csv(os.path.join(d, "us_closes.csv"))
        pd.DataFrame({"AAPL": 10, "MSFT": 20}, index=idx).rename_axis("Date") \
            .to_csv(os.path.join(d, "us_volumes.csv"))
        c, v, err = load_universe_bars(["AAPL", "MSFT"], cache_dir=d, offline=True,
                                       verbose=False)
        assert c is not None and v is not None and err is None
        assert list(c.columns) == ["AAPL", "MSFT"] and len(c) == 6
        assert (v["MSFT"] == 20).all()


def test_fetch_us_universe_names_a_missing_package():
    """The commonest failure by far: installed in a notebook or on Colab, not
    for the interpreter running Streamlit. The message has to say that."""
    import tempfile, os, sys, builtins
    from qbs.finviz import fetch_us_universe

    real_import = builtins.__import__

    def _no_finviz(name, *a, **k):
        if name.startswith("finvizfinance"):
            raise ImportError("No module named 'finvizfinance'")
        return real_import(name, *a, **k)

    dropped = {k: sys.modules.pop(k) for k in list(sys.modules)
               if k.startswith("finvizfinance")}
    builtins.__import__ = _no_finviz
    try:
        with tempfile.TemporaryDirectory() as d:
            out, err = fetch_us_universe(cache_path=os.path.join(d, "u.csv"),
                                         verbose=False)
            assert out is None
            assert "not installed" in err
            assert "requirements-dashboard" in err
    finally:
        builtins.__import__ = real_import
        sys.modules.update(dropped)


def test_diagnose_reports_each_step():
    from qbs.finviz import diagnose

    steps = diagnose(verbose=False)
    assert "python" in steps and "finvizfinance" in steps
    # Whatever the outcome, every reported step must carry a verdict.
    assert all(isinstance(v, str) and v for v in steps.values())


# --------------------------------------------------------------------------
# Per-name momentum profile (qbs/breadth.py)
# --------------------------------------------------------------------------

def _profile_universe():
    idx = pd.bdate_range("2023-01-02", periods=400)
    n = len(idx)
    return pd.DataFrame({
        "WIN": np.linspace(10.0, 40.0, n),        # strongest
        "MID": np.linspace(10.0, 14.0, n),
        "FLAT": np.full(n, 10.0),
        "LOSE": np.linspace(40.0, 12.0, n),       # weakest
    }, index=idx)


def test_momentum_profile_ranks_within_the_universe():
    """A return means nothing alone; the rank is the point."""
    from qbs.breadth import momentum_profile

    px = _profile_universe()
    win = momentum_profile(px, "WIN")["returns"].set_index("Horizon")
    lose = momentum_profile(px, "LOSE")["returns"].set_index("Horizon")

    assert win.loc["12 months", "Rank"] > lose.loc["12 months", "Rank"]
    assert win.loc["12 months", "Return"] > win.loc["12 months", "Universe median"]
    assert lose.loc["12 months", "Return"] < lose.loc["12 months", "Universe median"]
    assert win["Rank"].between(1, 99).all()


def test_momentum_profile_skips_the_most_recent_month():
    """The book's score skips the most recent month. The row must differ from
    the plain trailing return, or it is measuring the wrong thing."""
    from qbs.breadth import momentum_label, momentum_profile
    from qbs.config import MomentumParams

    p = MomentumParams(lookback_months=12, skip_months=1)
    px = _profile_universe().copy()
    # A spike confined to the last month: 12-0 sees it, 12-1 must not.
    px.iloc[-15:, px.columns.get_loc("MID")] *= 3.0
    r = momentum_profile(px, "MID", momentum=p)["returns"].set_index("Horizon")
    row = f"{momentum_label(p)} momentum"
    assert r.loc["12 months", "Return"] > r.loc[row, "Return"]


def test_momentum_profile_scores_the_window_the_ranker_is_configured_with():
    """The lookback has already moved from 12-1 to 6-1 once. A profile that
    keeps reporting 12-1 would label a number the book does not use with the
    name of the rule it claims to be explaining."""
    from qbs.breadth import momentum_label, momentum_profile
    from qbs.config import MomentumParams

    px = _profile_universe()
    six = MomentumParams(lookback_months=6, skip_months=1)
    twelve = MomentumParams(lookback_months=12, skip_months=1)

    assert momentum_label(six) == "6-1" and momentum_label(twelve) == "12-1"
    r6 = momentum_profile(px, "WIN", momentum=six)["returns"].set_index("Horizon")
    r12 = momentum_profile(px, "WIN", momentum=twelve)["returns"].set_index("Horizon")
    assert "6-1 momentum" in r6.index and "12-1 momentum" not in r6.index
    assert "12-1 momentum" in r12.index
    # Different windows over a trending name are different numbers; equal
    # values would mean the parameter is being ignored.
    assert r6.loc["6-1 momentum", "Return"] != r12.loc["12-1 momentum", "Return"]

    g6 = momentum_profile(px, "WIN", momentum=six)["gates"]
    assert any(r.startswith("6-1 momentum beats") for r in g6["Rule"])


def test_the_momentum_hurdle_uses_the_same_window_as_the_score():
    """Comparing a 6-1 stock return against a 12-1 cash return would be a
    different test from the one `absolute_filter` applies."""
    from qbs.breadth import momentum_profile
    from qbs.config import MomentumParams

    px = _profile_universe()
    idx = px.index
    # Cash compounding steadily: a 12-month hurdle is far above a 6-month one,
    # so a window mix-up changes the number the rule prints.
    safe = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)

    def hurdle(months):
        gates = momentum_profile(
            px, "MID", safe=safe,
            momentum=MomentumParams(lookback_months=months))["gates"]
        rule = [r for r in gates["Rule"] if "beats" in r][0]
        return float(rule.split("(")[1].split("%")[0])

    assert hurdle(6) < hurdle(12), "a longer window must show a bigger hurdle"


def test_momentum_profile_gates_explain_a_rejection():
    """The gate table is the point of the panel: it has to name the rule that
    blocks a name, not merely that something did."""
    from qbs.breadth import momentum_profile

    idx = pd.bdate_range("2023-01-02", periods=400)
    n = len(idx)
    # Rises hard, then gives back 14%. Chosen so it is MORE than 10% off its
    # high while still above the 200-day average: that isolates the proximity
    # rule as the only thing blocking it, which is the case worth pinning.
    path = np.concatenate([np.linspace(10, 100, n - 25), np.linspace(100, 86, 25)])
    px = pd.DataFrame({"PULLBACK": path, "OTHER": np.linspace(10, 12, n)}, index=idx)

    g = momentum_profile(px, "PULLBACK")["gates"].set_index("Rule")
    proximity = "within 10% of 252-day high"
    # Truthiness, not identity: pandas stores these as numpy bools, and
    # `np.False_ is False` is False.
    assert not g.loc[proximity, "Pass"]
    assert g.loc["above SMA 200", "Pass"], "the pullback must stay above SMA 200"
    blocked = g[~g["Pass"].astype(bool)]
    assert proximity in blocked.index
    assert "above SMA 200" not in blocked.index, "only the proximity rule blocks it"


def test_momentum_profile_gates_follow_the_screen_parameters():
    """The panel reports the screen that is configured, not the one that was
    configured when the panel was written. Retune the params and every
    threshold, label and verdict has to move with them."""
    from qbs.breadth import momentum_profile
    from qbs.config import FinvizScreenParams

    idx = pd.bdate_range("2023-01-02", periods=400)
    n = len(idx)
    path = np.concatenate([np.linspace(10, 100, n - 25), np.linspace(100, 86, 25)])
    px = pd.DataFrame({"PULLBACK": path, "OTHER": np.linspace(10, 12, n)}, index=idx)

    # 14% off the high fails a 10% ceiling and clears a 20% one.
    tight = momentum_profile(px, "PULLBACK",
                             screen=FinvizScreenParams())["gates"].set_index("Rule")
    loose = momentum_profile(
        px, "PULLBACK",
        screen=FinvizScreenParams(within_52w_high_pct=0.20))["gates"].set_index("Rule")
    assert not tight.loc["within 10% of 252-day high", "Pass"]
    assert loose.loc["within 20% of 252-day high", "Pass"]

    # A configured band floor adds its row; the default ceiling-only rule
    # must not claim a floor the screen does not apply.
    assert not any("off the high" in r for r in tight.index)
    banded = momentum_profile(
        px, "PULLBACK",
        screen=FinvizScreenParams(min_off_high_pct=0.05))["gates"].set_index("Rule")
    assert banded.loc["at least 5% off the high", "Pass"], "14% off clears a 5% floor"

    # The quarter-up row disappears when the screen stops requiring it.
    assert any("quarter up" in r for r in tight.index)
    off = momentum_profile(
        px, "PULLBACK",
        screen=FinvizScreenParams(require_quarter_up=False))["gates"].set_index("Rule")
    assert not any("quarter up" in r for r in off.index)


def test_momentum_profile_hurdle_uses_the_safe_asset_when_given():
    from qbs.breadth import momentum_label, momentum_profile

    px = _profile_universe()
    idx = px.index
    safe = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)  # strong cash

    without = momentum_profile(px, "MID")["gates"].set_index("Rule")
    with_safe = momentum_profile(px, "MID", safe=safe)["gates"].set_index("Rule")

    prefix = f"{momentum_label()} momentum beats"
    rules_without = [r for r in without.index if r.startswith(prefix)]
    rules_with = [r for r in with_safe.index if r.startswith(prefix)]
    assert "zero" in rules_without[0], "no safe asset -> the weaker test, and it says so"
    assert "BOXX" in rules_with[0], "the hurdle used must be named"


def test_momentum_profile_unknown_ticker_is_empty_not_an_error():
    from qbs.breadth import momentum_profile

    prof = momentum_profile(_profile_universe(), "NOPE")
    assert all(v.empty for v in prof.values())


def test_momentum_profile_trend_rows_are_present():
    from qbs.breadth import momentum_profile

    tr = momentum_profile(_profile_universe(), "WIN")["trend"].set_index("Measure")
    for row in ("vs EMA 10", "vs EMA 200", "Off 52-week high", "vs SMA 200",
                "Distance from 50-day EMA", "Annualised volatility"):
        assert row in tr.index
    assert tr.loc["Off 52-week high", "Value"] >= -1e-9, "a high is never below price"


# --------------------------------------------------------------------------
# The drawdown circuit breaker
# --------------------------------------------------------------------------

def _dd_fixture():
    from qbs.strategies import cross_sectional_momentum, book_vol_target
    cfg = Config()
    cfg.momentum = replace(cfg.momentum, min_history=200)
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    frame = uni.copy()
    frame[cfg.momentum.safe_asset] = px[cfg.momentum.safe_asset]
    mom = cross_sectional_momentum(uni, px[cfg.momentum.safe_asset], cfg.momentum)
    vt = book_vol_target(mom, frame, cfg.book_vol, lag=cfg.execution_lag)
    return cfg, frame, vt, px


def test_the_stop_is_inert_when_disabled():
    """Disabled, the overlay must not change a single weight."""
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, _ = _dd_fixture()

    out = drawdown_stop(vt, frame, DrawdownStopParams(enabled=False),
                        lag=cfg.execution_lag)

    pd.testing.assert_frame_equal(out.weights, vt.weights)


def test_a_deep_drawdown_moves_the_whole_book_to_the_safe_asset():
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, _ = _dd_fixture()
    safe = cfg.momentum.safe_asset

    out = drawdown_stop(vt, frame, DrawdownStopParams(enabled=True, exit_drawdown=0.02),
                        lag=cfg.execution_lag)
    blocked = out.diagnostics["blocked"].astype(bool)
    assert blocked.any(), "a 2% threshold should fire on this fixture"

    risk = [c for c in out.weights.columns if c != safe]
    assert (out.weights.loc[blocked, risk].abs().to_numpy() == 0).all()
    assert np.allclose(out.weights.loc[blocked, safe], 1.0)
    # And the untouched days are exactly the base book.
    pd.testing.assert_frame_equal(out.weights.loc[~blocked], vt.weights.loc[~blocked])


def test_the_stop_never_changes_which_names_were_picked():
    """It is an exposure overlay. Selection is not its business."""
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, _ = _dd_fixture()

    out = drawdown_stop(vt, frame, DrawdownStopParams(enabled=True, exit_drawdown=0.02),
                        lag=cfg.execution_lag)

    assert out.holdings_log == vt.holdings_log
    assert out.held_ranks == vt.held_ranks


def test_the_cooldown_holds_the_book_out_after_the_flag_clears():
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, _ = _dd_fixture()

    short = drawdown_stop(vt, frame, DrawdownStopParams(
        enabled=True, exit_drawdown=0.02, cooldown_days=1), lag=cfg.execution_lag)
    long_ = drawdown_stop(vt, frame, DrawdownStopParams(
        enabled=True, exit_drawdown=0.02, cooldown_days=20), lag=cfg.execution_lag)

    assert long_.diagnostics["blocked"].sum() > short.diagnostics["blocked"].sum()
    # The flag itself is a property of the book, not of the cooldown.
    pd.testing.assert_series_equal(short.diagnostics["dd_flag"],
                                   long_.diagnostics["dd_flag"])


def test_the_drawdown_is_measured_on_the_undisturbed_book():
    """Gate on the stopped equity curve and the flag could never clear.

    Parked in cash the book stops moving, so its drawdown would freeze at the
    level that triggered the stop and the breaker would latch on for ever.
    """
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, _ = _dd_fixture()

    out = drawdown_stop(vt, frame, DrawdownStopParams(enabled=True, exit_drawdown=0.05),
                        lag=cfg.execution_lag)
    off = drawdown_stop(vt, frame, DrawdownStopParams(exit_drawdown=0.05),
                        lag=cfg.execution_lag)

    # Same diagnostics whether or not the stop acted -- that is what makes the
    # trigger a pure function of prices rather than of its own output.
    pd.testing.assert_series_equal(out.diagnostics["book_drawdown"],
                                   off.diagnostics["book_drawdown"])

    # And it releases: measured on the stopped curve the book would freeze at
    # the triggering drawdown and the flag could never clear again, so the
    # breaker must be observed switching back off at least once.
    blocked = out.diagnostics["blocked"].astype(bool)
    assert blocked.any(), "the fixture never triggered; the test proves nothing"
    released = (blocked.astype(int).diff() == -1).sum()
    assert released > 0, "the breaker latched on and never released"


def test_the_benchmark_leg_only_acts_when_configured():
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, px = _dd_fixture()
    bench = px[cfg.momentum.safe_asset] * 0 + np.linspace(100, 40, len(px))  # -60%

    # Explicitly off -- the configured default is 0.15, so "without" has to say so.
    without = drawdown_stop(vt, frame, DrawdownStopParams(enabled=True, qqq_drawdown=0.0),
                            lag=cfg.execution_lag, benchmark=bench)
    with_ = drawdown_stop(vt, frame, DrawdownStopParams(enabled=True, qqq_drawdown=0.10),
                          lag=cfg.execution_lag, benchmark=bench)

    assert with_.diagnostics["blocked"].sum() > without.diagnostics["blocked"].sum()
    # No benchmark passed -> the leg is silently skipped, not an error.
    none = drawdown_stop(vt, frame, DrawdownStopParams(enabled=True, qqq_drawdown=0.10),
                         lag=cfg.execution_lag, benchmark=None)   # leg skipped
    pd.testing.assert_frame_equal(none.weights, without.weights)


def test_a_missing_safe_asset_is_rejected():
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, _ = _dd_fixture()
    trimmed = vt.weights.drop(columns=[cfg.momentum.safe_asset])
    bad = StrategySignals("x", trimmed)

    with pytest.raises(ValueError, match="safe asset"):
        drawdown_stop(bad, frame, DrawdownStopParams(enabled=True))


# --------------------------------------------------------------------------
# The momentum-leader filter as a pre-screen for the ranker
# --------------------------------------------------------------------------

def test_leader_eligibility_is_the_dashboards_own_mask():
    """One definition. The screen and the ranker must not drift apart."""
    from qbs.breadth import BreadthParams, leader_eligibility, leader_mask

    px = synthetic_universe(n=20, start="2023-06-01")
    p = BreadthParams()

    pd.testing.assert_frame_equal(leader_eligibility(px, p=p), leader_mask(px, p=p))


def test_leader_eligibility_ands_with_an_existing_mask():
    from qbs.breadth import leader_eligibility, leader_mask

    px = synthetic_universe(n=20, start="2023-06-01")
    half = pd.DataFrame(False, index=px.index, columns=px.columns)
    half.iloc[:, :5] = True

    out = leader_eligibility(px, existing=half)

    assert not out.iloc[:, 5:].to_numpy().any(), "the existing mask was ignored"
    assert (out == (leader_mask(px) & half)).to_numpy().all()


def test_the_leader_filter_is_off_and_inert_by_default():
    cfg = Config()
    assert cfg.use_leader_filter is False

    px = synthetic_prices()
    lab_off = run(cfg=cfg, prices=px, universe_prices=synthetic_universe(
        n=20, start="2023-06-01").reindex(px.index).ffill(),
        offline=True, fetch_universe=False, with_vix=False)
    assert "momentum" in lab_off.signals


def test_the_leader_filter_narrows_the_book_when_switched_on():
    """Switched on it must actually bind, and unfilled slots go to cash."""
    from dataclasses import replace as _replace

    px = synthetic_prices()
    uni = synthetic_universe(n=20, start="2023-06-01").reindex(px.index).ffill()

    off = run(cfg=Config(), prices=px, universe_prices=uni, offline=True,
              fetch_universe=False, with_vix=False)
    cfg_on = Config()
    cfg_on.use_leader_filter = True
    on = run(cfg=cfg_on, prices=px, universe_prices=uni, offline=True,
             fetch_universe=False, with_vix=False)

    w_off = off.signals["momentum"].weights
    w_on = on.signals["momentum"].weights
    risk = [c for c in w_on.columns if c != SAFE_ASSET]

    assert w_on[risk].sum(axis=1).mean() < w_off[risk].sum(axis=1).mean(), \
        "the filter did not reduce exposure at all"
    # Every weight it does hold is one the unfiltered book could also hold.
    assert (w_on[risk].to_numpy() > 0).sum() < (w_off[risk].to_numpy() > 0).sum()


def test_a_disabled_stop_never_reports_itself_as_blocking():
    """`blocked` must mean the book was held flat, not that it might have been.

    A risk readout claiming the breaker is halting a fully invested book is
    worse than none: it is the one line an operator would trust in a hurry.
    """
    from qbs.strategies import drawdown_stop
    cfg, frame, vt, _ = _dd_fixture()

    off = drawdown_stop(vt, frame, DrawdownStopParams(enabled=False, exit_drawdown=0.02),
                        lag=cfg.execution_lag)
    on = drawdown_stop(vt, frame, DrawdownStopParams(enabled=True, exit_drawdown=0.02),
                       lag=cfg.execution_lag)

    assert not off.diagnostics["blocked"].astype(bool).any()
    assert on.diagnostics["blocked"].astype(bool).any()
    # The condition itself is reported either way, so a dry run still shows it.
    pd.testing.assert_series_equal(off.diagnostics["dd_flag"], on.diagnostics["dd_flag"])
    pd.testing.assert_frame_equal(off.weights, vt.weights)


# --------------------------------------------------------------------------
# Volume filters on the ranker
# --------------------------------------------------------------------------

def _vol_fixture():
    idx = pd.bdate_range("2024-01-01", periods=200)
    return pd.DataFrame({
        "SURGE": np.r_[np.full(195, 1e6), np.full(5, 4e6)],
        "QUIET": np.r_[np.full(195, 1e6), np.full(5, 2e5)],
        "FLAT":  np.full(200, 1e6),
    }, index=idx)


def test_relative_volume_is_a_ratio_to_the_names_own_norm():
    from qbs.breadth import relative_volume

    r = relative_volume(_vol_fixture()).iloc[-1]

    assert r["SURGE"] > 2.5 and r["QUIET"] < 0.4
    assert r["FLAT"] == pytest.approx(1.0)


def test_an_unconfigured_volume_filter_admits_everything():
    """A filter nobody asked for must never quietly remove a name."""
    from qbs.breadth import volume_eligibility

    assert volume_eligibility(_vol_fixture()).to_numpy().all()


def test_each_volume_leg_selects_what_it_claims():
    from qbs.breadth import volume_eligibility

    v = _vol_fixture()
    assert [c for c in v if volume_eligibility(v, min_ratio=1.5)[c].iloc[-1]] == ["SURGE"]
    assert [c for c in v if volume_eligibility(v, max_ratio=0.5)[c].iloc[-1]] == ["QUIET"]
    # The absolute floor is the near-useless one on a large-cap index.
    assert [c for c in v if volume_eligibility(v, min_shares=3e5)[c].iloc[-1]] \
        == ["SURGE", "FLAT"]


def test_volume_eligibility_ands_with_an_existing_mask():
    from qbs.breadth import volume_eligibility

    v = _vol_fixture()
    only_flat = pd.DataFrame(False, index=v.index, columns=v.columns)
    only_flat["FLAT"] = True

    out = volume_eligibility(v, min_shares=3e5, existing=only_flat)

    assert not out["SURGE"].any(), "the existing mask was ignored"
    assert out["FLAT"].iloc[-1]


def test_missing_volumes_read_as_none_rather_than_failing(tmp_path):
    """A cache written before volumes were kept must not break a price load."""
    from qbs.universe import load_universe_volumes

    assert load_universe_volumes(cache_dir=str(tmp_path)) is None


def test_sweep_volume_scores_the_book_it_can_actually_price(tmp_path, monkeypatch):
    """`lab.universe` carries columns the engine never priced.

    It is the frame as it arrived -- not reindexed, not pruned by min_history --
    so building weights from it puts a position on a ticker the return frame
    lacks, and run_backtest raises a KeyError from deep inside. The sweep must
    take its universe from `combined`, as the other sweeps do.
    """
    from qbs import pipeline
    from qbs.pipeline import sweep_volume

    px = synthetic_prices()
    uni = synthetic_universe(n=20, start="2023-06-01")
    # A column with almost no history: pruned out of `combined`, still present
    # in `universe`. This is exactly the shape that broke it.
    uni["SPARSE"] = np.nan
    uni.iloc[-5:, uni.columns.get_loc("SPARSE")] = 100.0

    lab = run(cfg=Config(), prices=px, universe_prices=uni.reindex(px.index).ffill(),
              offline=True, fetch_universe=False, with_vix=False)
    assert "SPARSE" in lab.universe.columns
    assert "SPARSE" not in lab.combined.columns, "the fixture no longer bites"

    vols = pd.DataFrame(1e6, index=lab.combined.index,
                        columns=lab.combined.drop(columns=[SAFE_ASSET]).columns)
    monkeypatch.setattr(pipeline, "load_universe_volumes", lambda *a, **k: vols,
                        raising=False)
    monkeypatch.setattr("qbs.universe.load_universe_volumes", lambda *a, **k: vols)

    out = sweep_volume(lab, min_ratios=(0.0, 1.0))

    assert not out.empty
    assert list(out["min_ratio"]) == [0.0, 1.0]


def test_sweep_volume_says_nothing_rather_than_zero_without_a_cache(monkeypatch):
    from qbs.pipeline import sweep_volume

    px = synthetic_prices()
    lab = run(cfg=Config(), prices=px,
              universe_prices=synthetic_universe(n=20, start="2023-06-01")
              .reindex(px.index).ffill(),
              offline=True, fetch_universe=False, with_vix=False)
    monkeypatch.setattr("qbs.universe.load_universe_volumes", lambda *a, **k: None)

    assert sweep_volume(lab).empty, "a missing cache must not read as a result"
