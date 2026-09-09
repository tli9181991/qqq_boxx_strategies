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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import (
    Config, GEMParams, MomentumParams, RSI2Params, VixBreakerParams, VolTargetParams,
)
from qbs.data import synthetic_prices, synthetic_vix
from qbs.engine import run_backtest
from qbs.indicators import drawdown, sma, wilder_rsi
from qbs.metrics import summarise
from qbs.pipeline import build_signals, run, sweep_band, sweep_vix
from qbs.strategies import (
    buy_and_hold, connors_rsi2, cross_sectional_momentum, gem,
    vix_circuit_breaker, vol_target_overlay,
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
