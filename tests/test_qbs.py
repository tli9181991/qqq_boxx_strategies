"""Tests. Run with `python -m pytest tests -q` or `python tests/test_qbs.py`.

Two categories: indicator correctness against values you can verify by hand,
and structural invariants of the backtest that a subtle refactor could break
silently (look-ahead, weight leakage, cost accounting).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from qbs.config import (
    SAFE_ASSET, BookVolTargetParams, Config, DrawdownStopParams, FinvizScreenParams,
    GEMParams, MomentumParams, RSI2Params, ResidualMomentumParams,
    VixBreakerParams, VolTargetParams,
)
from qbs.data import synthetic_prices, synthetic_vix
from qbs.engine import run_backtest
from qbs.indicators import drawdown, sma, wilder_rsi
from qbs.metrics import summarise
from qbs.pipeline import build_signals, run, sweep_band, sweep_target_vol, sweep_vix
from qbs.strategies import (
    StrategySignals, book_vol_target, buy_and_hold, connors_rsi2,
    cross_sectional_momentum, drawdown_stop, gem, residual_momentum,
    residual_momentum_score, vix_circuit_breaker, vol_target_overlay,
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


def _resmom_fixture():
    uni, safe = _mom_fixture()
    # The market factor: an equal-weight index of the universe itself, which is
    # what a market model would be regressing against here.
    mkt = uni.mean(axis=1)
    return uni, safe, mkt


def test_residual_score_strips_the_market_component():
    """A name that is purely market beta must score near zero.

    The mechanism test: build one name that is exactly 2x the market with no
    idiosyncratic component, and one that drifts up on its own. Total-return
    momentum prefers the leveraged market name in a rising market; residual
    momentum must not.
    """
    n = 700
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(7)
    mkt_r = rng.normal(0.0008, 0.01, n)               # a rising market
    mkt = pd.Series(100 * np.cumprod(1 + mkt_r), index=idx)

    # 2x the market plus a little idiosyncratic noise that goes nowhere. Its
    # TOTAL return is the largest of the three by far, because the market rose.
    beta_noise = rng.normal(0.0, 0.004, n)
    pure_beta = pd.Series(100 * np.cumprod(1 + 2.0 * mkt_r + beta_noise), index=idx)
    # Same market beta, but with idiosyncratic drift on top.
    drifter = pd.Series(100 * np.cumprod(1 + mkt_r + rng.normal(0.0009, 0.004, n)),
                        index=idx)
    uni = pd.DataFrame({"BETA": pure_beta, "DRIFT": drifter})

    p = ResidualMomentumParams(beta_window=252)
    total = uni / uni.shift(21) - 1.0          # what total-return momentum sees
    sc = residual_momentum_score(uni, mkt, p)
    last = sc.dropna().iloc[-1]
    last_total = total.loc[last.name]

    assert last_total["BETA"] > last_total["DRIFT"], (
        "fixture is wrong: the leveraged name should win on TOTAL return")
    assert last["DRIFT"] > last["BETA"], (
        f"residual momentum should prefer idiosyncratic drift: {dict(last)}")


def test_residual_score_is_unrankable_when_the_market_explains_everything():
    """A perfect market replica is a 0/0, and must not top the book.

    Without the residual-vol floor the ratio of two floating-point dust terms
    can come back arbitrarily large, which would hand the strongest rank to
    the one name carrying no idiosyncratic information at all.
    """
    n = 700
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(7)
    mkt_r = rng.normal(0.0008, 0.01, n)
    mkt = pd.Series(100 * np.cumprod(1 + mkt_r), index=idx)
    replica = pd.Series(100 * np.cumprod(1 + mkt_r), index=idx)   # residual == 0
    real = pd.Series(100 * np.cumprod(1 + mkt_r + rng.normal(0.0005, 0.004, n)),
                     index=idx)
    uni = pd.DataFrame({"REPLICA": replica, "REAL": real})

    sc = residual_momentum_score(uni, mkt, ResidualMomentumParams())
    tail = sc.dropna(how="all").iloc[-1]
    assert np.isnan(tail["REPLICA"]), f"replica should be unrankable, got {tail['REPLICA']}"
    assert np.isfinite(tail["REAL"])


def test_residual_momentum_has_no_look_ahead():
    """Beta, residuals and the score must all be blind to the future."""
    uni, safe, mkt = _resmom_fixture()
    cut = uni.index[len(uni) // 2]
    rng = np.random.default_rng(0)
    tu, tm = uni.copy(), mkt.copy()
    after = tu.index > cut
    tu.loc[after] = tu.loc[after] * rng.uniform(0.5, 1.5, tu.loc[after].shape)
    tm.loc[after] = tm.loc[after] * rng.uniform(0.5, 1.5, after.sum())

    p = ResidualMomentumParams()
    a = residual_momentum(uni, safe, mkt, p).weights.loc[:cut]
    b = residual_momentum(tu, safe, tm, p).weights.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_residual_momentum_obeys_the_slot_invariants():
    """Whatever it ranks on, the book is still six slots and fully allocated."""
    uni, safe, mkt = _resmom_fixture()
    p = ResidualMomentumParams(n_hold=6, exit_rank=10)
    sig = residual_momentum(uni, safe, mkt, p)
    risk = sig.weights.drop(columns=["BOXX"])
    assert (risk > 0).sum(axis=1).max() <= p.n_hold
    assert risk.max().max() <= 1.0 / p.n_hold + 1e-9
    assert np.allclose(sig.weights.sum(axis=1), 1.0)
    assert (sig.weights >= -1e-12).all().all()


def test_residual_momentum_standardisation_changes_the_ranking():
    """`standardise` is a real choice, not a no-op knob."""
    uni, safe, mkt = _resmom_fixture()
    raw = residual_momentum_score(uni, mkt, ResidualMomentumParams(standardise=False))
    std = residual_momentum_score(uni, mkt, ResidualMomentumParams(standardise=True))
    both = raw.dropna(how="all").index.intersection(std.dropna(how="all").index)
    assert len(both) > 0
    changed = (raw.loc[both].rank(axis=1) != std.loc[both].rank(axis=1)).to_numpy().any()
    assert changed, "standardising did not change any ranking"


def test_residual_momentum_rejects_bad_parameters():
    for kwargs in ({"n_hold": 6, "exit_rank": 3},
                   {"lookback_months": 1, "skip_months": 1},
                   {"beta_window": 5}):
        try:
            ResidualMomentumParams(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} should have been rejected")


def test_corr_cap_off_is_bit_identical_to_the_plain_ranker():
    """The cap must default to OFF and change nothing when it is.

    A new selection filter that quietly moves the shipped strategy would
    invalidate every number in the README, so this is the test that matters
    most about it.
    """
    uni, safe = _mom_fixture()
    plain = cross_sectional_momentum(uni, safe, MomentumParams())
    explicit_off = cross_sectional_momentum(uni, safe, MomentumParams(max_corr=None))
    pd.testing.assert_frame_equal(plain.weights, explicit_off.weights)


def test_corr_cap_raises_the_number_of_independent_bets():
    """Tightening the cap must lower the held book's mean pairwise correlation.

    This is the mechanism the parameter exists for, and unlike its effect on
    return it should be close to monotone. The fixture is built so that the
    top-ranked names are deliberately near-duplicates of each other.
    """
    uni, safe = _mom_fixture()
    rets = uni.pct_change()

    def mean_corr(sig):
        vals = []
        for dt, names in sig.holdings_log.items():
            if len(names) < 2:
                continue
            win = rets.loc[:dt, names].tail(60)
            if len(win) < 30:
                continue
            c = win.corr().to_numpy()
            iu = np.triu_indices_from(c, 1)
            if np.isfinite(c[iu]).any():
                vals.append(np.nanmean(c[iu]))
        return float(np.mean(vals)) if vals else np.nan

    loose = mean_corr(cross_sectional_momentum(uni, safe, MomentumParams()))
    tight = mean_corr(cross_sectional_momentum(uni, safe, MomentumParams(max_corr=0.5)))
    assert tight <= loose + 1e-9, f"cap did not decorrelate the book: {loose} -> {tight}"


def test_corr_cap_never_exceeds_the_slot_count():
    """The cap may leave slots in cash, but must never overfill the book."""
    uni, safe = _mom_fixture()
    p = MomentumParams(max_corr=0.3, n_hold=6)
    sig = cross_sectional_momentum(uni, safe, p)
    assert all(len(v) <= p.n_hold for v in sig.holdings_log.values())
    risky = sig.weights.drop(columns=["BOXX"]).sum(axis=1)
    assert (risky <= 1.0 + 1e-9).all()
    assert np.allclose(sig.weights.sum(axis=1), 1.0)


def test_corr_cap_has_no_look_ahead():
    """The correlation matrix at t must not see a single return after t."""
    uni, safe = _mom_fixture()
    cut = uni.index[len(uni) // 2]
    rng = np.random.default_rng(0)
    tampered = uni.copy()
    after = tampered.index > cut
    tampered.loc[after] = tampered.loc[after] * rng.uniform(0.5, 1.5, tampered.loc[after].shape)

    p = MomentumParams(max_corr=0.7)
    a = cross_sectional_momentum(uni, safe, p).weights.loc[:cut]
    b = cross_sectional_momentum(tampered, safe, p).weights.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_corr_cap_rejects_an_impossible_pool():
    """corr_pool below n_hold could never fill the book -- fail loudly."""
    try:
        MomentumParams(n_hold=6, corr_pool=3)
    except ValueError:
        pass
    else:
        raise AssertionError("corr_pool < n_hold should be rejected")
    try:
        MomentumParams(max_corr=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range max_corr should be rejected")


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


def _screen(**kw):
    """Screen params for a closes-only fixture.

    `min_volume` is ON in the real defaults and RAISES without volumes, on
    purpose -- it is a leg of the high-momentum definition and skipping it
    silently would overstate the screen. Tests that are not about volume opt
    out here once instead of in twenty places.
    """
    kw.setdefault("min_volume", None)
    return FinvizScreenParams(**kw)


def _notebook_screen(**kw):
    """The original Finviz-notebook filter set, which is no longer the default.

    Kept as a fixture because those legs still work and several tests exist to
    pin their behaviour; they just no longer describe what the screen does out
    of the box.
    """
    for key, value in (("min_price", 10.0), ("above_sma", 200),
                       ("within_52w_high_pct", 0.10),
                       ("require_quarter_up", True),
                       ("min_quarter_return", None)):
        kw.setdefault(key, value)
    return _screen(**kw)


def test_finviz_weights_are_a_valid_long_only_book():
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    sig = finviz_momentum_screen(uni, safe, _screen(n_hold=6))
    assert np.allclose(sig.weights.sum(axis=1), 1.0)
    assert (sig.weights >= -1e-9).all().all()
    risk = sig.weights.drop(columns=["BOXX"]).sum(axis=1)
    assert (risk <= 1.0 + 1e-9).all(), "a screen must never lever the book"


def test_finviz_parks_in_the_safe_asset_when_nothing_passes():
    """The Finviz filters are absolute tests, so 'nothing qualifies' has to be
    a reachable state that means cash rather than a forced allocation."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    p = _screen(n_hold=6, within_52w_high_pct=-1.0)
    sig = finviz_momentum_screen(uni, safe, p)
    assert np.allclose(sig.weights["BOXX"], 1.0), "must be fully in cash"
    assert np.nanmax(sig.diagnostics["n_passing"].to_numpy()) == 0


def test_finviz_ranks_by_one_year_return_among_passing_names():
    """The notebook's rule: RS Rank is a bucketed 1-year return, so the first
    name bought on any date is the strongest 1-year performer that passed."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    p = _screen(n_hold=6)
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
    sig = finviz_momentum_screen(uni, safe, _screen(n_hold=4))
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

    p = _screen(n_hold=6)
    a = finviz_momentum_screen(uni, safe, p).weights.loc[:cut]
    b = finviz_momentum_screen(tampered, safe, p).weights.loc[:cut]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_finviz_volume_filter_only_ever_removes_names():
    """'Average Volume over 200K' is a filter: supplying volumes cannot admit
    a name that failed without them."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    p = _screen(n_hold=0)
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


def test_the_screen_defaults_are_the_high_momentum_definition():
    """The screen and the market tab's leader group must apply one rule. If
    these drift apart, a name can be a "momentum leader" on one tab and fail
    the screen on the other, with nothing on screen explaining why."""
    from qbs.breadth import BreadthParams

    screen, leader = FinvizScreenParams(), BreadthParams()
    assert screen.min_price == leader.leader_min_price == 5.0
    assert screen.min_volume == leader.leader_min_volume == 300_000.0
    assert screen.min_quarter_return == leader.leader_min_quarter_return == 0.28
    assert screen.quarter_lookback == leader.leader_quarter_days == 63
    # And the notebook legs the definition does not include are off.
    assert screen.above_sma == 0
    assert screen.within_52w_high_pct is None
    assert screen.n_hold == 20


def test_the_screen_refuses_to_skip_its_volume_leg():
    """`min_avg_volume` was skipped silently when volume was missing. This one
    is a leg of the definition, so dropping it on the floor would overstate
    the screen -- the caller has to opt out deliberately."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    try:
        finviz_momentum_screen(uni, safe, FinvizScreenParams())
    except ValueError as exc:
        assert "min_volume" in str(exc) and "volumes" in str(exc)
    else:
        raise AssertionError("the default screen must raise without volumes")

    # Opting out explicitly is fine and is what the dashboard does.
    sig = finviz_momentum_screen(uni, safe, _screen())
    assert sig.weights.notna().any().any()


def test_the_breakout_watchlist_keeps_the_notebook_rules():
    """The breakout strategy's entries assume names selected NEAR their highs
    -- that is what `min_off_high_pct` turns into a band. The screen no longer
    selects that way, so following its defaults would change every result in
    that module without changing a line of it."""
    from qbs.config import notebook_screen_params

    nb = notebook_screen_params()
    assert nb.within_52w_high_pct == 0.10, "the band the breakout needs"
    assert nb.above_sma == 200 and nb.min_price == 10.0
    assert nb.min_quarter_return is None
    assert nb.min_volume is None, "closes-only fixtures must still run"

    # And the watchlist builder defaults to it rather than to the screen's.
    uni, safe = _finviz_inputs()
    from qbs.breakout import finviz_watchlists
    wl = finviz_watchlists(uni, safe, n_watch=10)
    assert wl, "the notebook rules must still produce a watchlist"


def test_finviz_min_dollar_volume_needs_volumes():
    """Silently ignoring a criterion the caller asked for would overstate the
    strategy, so an unusable parameter has to raise."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    try:
        finviz_momentum_screen(uni, safe,
                               _screen(min_dollar_volume=5e6))
    except ValueError:
        return
    raise AssertionError("min_dollar_volume without volumes must raise")


def test_finviz_quarter_up_gate_removes_names():
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    on = finviz_momentum_screen(uni, safe, _screen(n_hold=0))
    off = finviz_momentum_screen(uni, safe,
                                 _screen(n_hold=0, require_quarter_up=False))
    for dt in uni.index[::40]:
        assert set(on.holdings_log[dt]) <= set(off.holdings_log[dt])


def test_finviz_band_reduces_turnover():
    """The notebook has no band (exit_rank=0) and re-screens from scratch every
    day. Adding one must cut trading, whatever it does to return."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    # n_hold pinned: the default is 20 and `exit_rank` must not be narrower
    # than the book, so a 15-wide band needs a book smaller than 15.
    none = finviz_momentum_screen(uni, safe, _screen(n_hold=6, exit_rank=0))
    band = finviz_momentum_screen(uni, safe, _screen(n_hold=6, exit_rank=15))
    t_none = float(none.weights.diff().abs().sum(axis=1).sum())
    t_band = float(band.weights.diff().abs().sum(axis=1).sum())
    assert t_band < t_none, "a hysteresis band must reduce turnover"


def test_finviz_band_cannot_be_narrower_than_the_book():
    try:
        _screen(n_hold=6, exit_rank=3)
    except ValueError:
        return
    raise AssertionError("exit_rank below n_hold must raise")


def test_finviz_records_rank_and_score_like_the_momentum_book():
    """So the live run log stores selections from either strategy identically."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    sig = finviz_momentum_screen(uni, safe, _screen(n_hold=6))
    assert {"rank", "score"} <= set(sig.events.columns)
    assert sig.held_ranks is not None
    buys = sig.events[sig.events["action"] == "buy"]
    assert not buys.empty and (buys["rank"] >= 1).all()


def test_finviz_runs_through_the_shared_engine():
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    sig = finviz_momentum_screen(uni, safe, _screen(n_hold=6))
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
    # n_hold pinned well below the number that passes on this fixture, so the
    # candidate pool can actually be wider than the book -- which is what the
    # last assertion here is about. The default 20 exceeds the pool.
    monthly = finviz_momentum_screen(uni, safe, _screen(n_hold=3, rebalance="ME"))
    daily = finviz_momentum_screen(uni, safe, _screen(n_hold=3, rebalance="daily"))

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
    ceiling = finviz_momentum_screen(uni, safe, _screen(n_hold=0))
    band = finviz_momentum_screen(
        uni, safe, _screen(n_hold=0, min_off_high_pct=0.04,
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
        uni, safe, _screen(n_hold=0, min_off_high_pct=0.04))
    for dt in uni.index[::40]:
        assert set(strict.holdings_log[dt]) <= set(ceiling.holdings_log[dt])


def test_finviz_high_band_floor_defaults_to_the_notebook_rule():
    """Zero floor must reproduce the screener's own behaviour exactly."""
    from qbs.screens import finviz_momentum_screen

    uni, safe = _finviz_inputs()
    a = finviz_momentum_screen(uni, safe, _screen(n_hold=6))
    b = finviz_momentum_screen(uni, safe,
                               _screen(n_hold=6, min_off_high_pct=0.0))
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


def test_the_4pc_colour_bands_match_their_edges():
    """Boundary values, both columns, because an off-by-one on a band edge is
    invisible on screen -- the cell is simply the wrong shade."""
    from qbs.breadth import pulse_cell

    for value, want in ((0, "dark_red"), (50, "dark_red"), (51, "light_red"),
                        (100, "light_red"), (101, "light_green"),
                        (300, "light_green"), (301, "dark_green"),
                        (9999, "dark_green")):
        assert pulse_cell(value, "up") == want, f"up 4% at {value}"

    for value, want in ((0, "dark_green"), (50, "dark_green"),
                        (51, "light_green"), (100, "light_green"),
                        (101, "light_red"), (200, "light_red"),
                        (201, "dark_red"), (9999, "dark_red")):
        assert pulse_cell(value, "down") == want, f"down 4% at {value}"


def test_the_20day_column_shades_only_recent_sessions():
    """It answers "what is the tape doing NOW". Shading the whole history
    turns a regime indicator into wallpaper and the eye stops seeing it."""
    from qbs.breadth import BreadthParams, ma_fast_cell

    p = BreadthParams()
    assert ma_fast_cell(55.0, 0) == "light_green"
    assert ma_fast_cell(55.0, p.ma_fast_recent - 1) == "light_green"
    assert ma_fast_cell(55.0, p.ma_fast_recent) == "none", "11th row back is bare"
    assert ma_fast_cell(55.0, 200) == "none"


def test_the_20day_column_is_two_state_around_its_threshold():
    from qbs.breadth import BreadthParams, ma_fast_cell

    p = BreadthParams()
    assert p.ma_fast_green == 20.0
    assert ma_fast_cell(20.1, 0) == "light_green"
    assert ma_fast_cell(20.0, 0) == "light_red", "at the threshold is not above it"
    assert ma_fast_cell(0.0, 0) == "light_red"
    assert ma_fast_cell(float("nan"), 0) == "none"
    # Only ever the light shades -- the dark pair belongs to the 4% columns.
    shades = {ma_fast_cell(v, 0) for v in (0.0, 20.0, 20.1, 99.0)}
    assert shades == {"light_red", "light_green"}


def test_the_20day_shading_follows_its_parameters():
    from qbs.breadth import BreadthParams, ma_fast_cell

    strict = BreadthParams(ma_fast_recent=3, ma_fast_green=60.0)
    assert ma_fast_cell(55.0, 0, strict) == "light_red", "55 is below a 60 bar"
    assert ma_fast_cell(55.0, 3, strict) == "none", "only 3 rows shaded"
    assert ma_fast_cell(55.0, 3) == "light_green", "the default still shades 10"


def test_green_is_bullish_in_both_4pc_columns():
    """The two run in opposite directions: a big up count is bullish, a big
    down count is not. Getting that reversal backwards is the easy mistake and
    would make a calm tape look like a falling one."""
    from qbs.breadth import pulse_cell

    assert pulse_cell(500, "up") == "dark_green"
    assert pulse_cell(500, "down") == "dark_red"
    assert pulse_cell(10, "up") == "dark_red"
    assert pulse_cell(10, "down") == "dark_green"


def test_a_missing_4pc_count_has_no_colour():
    from qbs.breadth import pulse_cell

    assert pulse_cell(float("nan"), "up") == "none"
    assert pulse_cell(float("nan"), "down") == "none"


def test_the_4pc_bands_follow_their_parameters():
    """The UI reads the edges off BreadthParams rather than repeating them, so
    retuning has to move the colours."""
    from qbs.breadth import BreadthParams, pulse_cell

    tight = BreadthParams(up4_bands=(10.0, 20.0, 30.0))
    assert pulse_cell(35, "up", tight) == "dark_green", "35 clears a 30 top edge"
    assert pulse_cell(35, "up") == "dark_red", \
        "and is still in the bottom band under the default 50/100/300"


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


def _leader_fixture():
    """Three sectors, known relative strength inside each.

    Built long enough to be scorable over the ranker's own 6-1 window, and
    priced so every name clears the leader rule -- the point of these tests
    is the ordering and the grouping, not the filter, which `leader_mask`
    has its own tests for.
    """
    from qbs.breadth import _momentum_window

    look, _ = _momentum_window()
    idx = pd.bdate_range("2024-01-01", periods=look + 90)
    n = len(idx)
    # COMPOUNDING, not a straight line. A linear ramp's trailing-quarter
    # return shrinks as the base grows, so the slower names would fail the
    # leader rule's 28% quarterly gate and drop out of a fixture that is
    # supposed to be testing the grouping. A constant daily rate keeps every
    # name's quarterly gain constant and above the gate, while the rates
    # still order them.
    rates = {"AAA": 0.020, "BBB": 0.015, "CCC": 0.010,   # Tech
             "DDD": 0.025, "EEE": 0.006,                  # Health
             "FFF": 0.012}                                # Energy
    px = pd.DataFrame({t: 10.0 * (1.0 + r) ** np.arange(n)
                       for t, r in rates.items()}, index=idx)
    vols = pd.DataFrame(1e7, index=idx, columns=px.columns)
    sectors = {"AAA": "Tech", "BBB": "Tech", "CCC": "Tech",
               "DDD": "Health", "EEE": "Health", "FFF": "Energy"}
    return px, vols, sectors


def test_sector_leaders_group_by_sector_and_rank_inside_it():
    """The naming half of the concentration reading: a sector share nobody
    can name the members of is a number to nod at rather than act on."""
    from qbs.breadth import sector_leaders

    px, vols, sectors = _leader_fixture()
    out = sector_leaders(px, sectors, volumes=vols, per_sector=0)

    assert set(out["symbol"]) == set(px.columns), "every leader is named"
    # Sector order follows `sector_breakdown`: biggest group first.
    assert list(out["sector"].unique()) == ["Tech", "Health", "Energy"]
    # And strongest first inside each group.
    tech = out[out["sector"] == "Tech"]
    assert list(tech["symbol"]) == ["AAA", "BBB", "CCC"]
    assert list(tech["rank_in_sector"]) == [1, 2, 3]
    assert tech["score"].is_monotonic_decreasing
    assert list(out[out["sector"] == "Health"]["symbol"]) == ["DDD", "EEE"]
    # n_sector counts the whole sector, not the rows shown.
    assert set(out[out["sector"] == "Tech"]["n_sector"]) == {3}


def test_sector_leaders_cap_is_per_sector_not_overall():
    """A global cap would hand the whole list to the biggest sector and
    report the others as leaderless."""
    from qbs.breadth import sector_leaders

    px, vols, sectors = _leader_fixture()
    out = sector_leaders(px, sectors, volumes=vols, per_sector=1)

    assert len(out) == 3, "one per sector, all three sectors present"
    assert list(out["symbol"]) == ["AAA", "DDD", "FFF"], "each sector's best"
    # Truncated, and visibly so: the count is of the sector, not the rows.
    assert list(out[out["sector"] == "Tech"]["n_sector"]) == [3]


def test_sector_leaders_order_matches_the_breakdown_it_sits_under():
    """The two tables read down the page together, so a sector cannot be
    third in one and first in the other."""
    from qbs.breadth import sector_breakdown, sector_leaders

    px, vols, sectors = _leader_fixture()
    breakdown = sector_breakdown(px, sectors, volumes=vols)
    leaders = sector_leaders(px, sectors, volumes=vols, per_sector=0)

    assert (list(breakdown["sector"])
            == list(leaders["sector"].unique())), "same sequence"
    # And the member counts agree with the breakdown's own n.
    counts = leaders.groupby("sector")["n_sector"].first()
    for row in breakdown.itertuples():
        assert counts[row.sector] == row.n


def test_sector_leaders_needs_a_map_and_enough_history():
    """Both gaps fail empty rather than inventing a label or a score. An
    unscored name is not a weak one, and 'Unclassified' rendered as a sector
    is a classification this package did not make."""
    from qbs.breadth import sector_leaders

    px, vols, sectors = _leader_fixture()
    assert sector_leaders(px, {}, volumes=vols).empty, "no map, no table"
    assert sector_leaders(px.iloc[:5], sectors, volumes=vols).empty, \
        "too short to score over the 6-1 window"
    assert sector_leaders(pd.DataFrame(), sectors).empty

    # A name with no sector is labelled, not dropped -- it IS a leader, and
    # silently losing it would understate the leadership count.
    partial = {k: v for k, v in sectors.items() if k != "FFF"}
    out = sector_leaders(px, partial, volumes=vols, per_sector=0)
    assert "Unclassified" in set(out["sector"])
    assert "FFF" in set(out["symbol"])


def test_sector_leaders_accept_a_date_the_frame_does_not_hold():
    """The dashboard reads leaders on the market cache's last bar and ranks
    on the book cache's, and the two are not always the same day."""
    from qbs.breadth import sector_leaders

    px, vols, sectors = _leader_fixture()
    asof = px.index[-1] + pd.Timedelta(days=3)
    out = sector_leaders(px, sectors, volumes=vols, asof=asof, per_sector=0)
    assert not out.empty, "it falls back to the last bar at or before asof"

    before = sector_leaders(px, sectors, volumes=vols, per_sector=0)
    assert list(out["symbol"]) == list(before["symbol"])

    # Earlier than anything in the frame is empty, not the first bar.
    assert sector_leaders(px, sectors, volumes=vols,
                          asof=px.index[0] - pd.Timedelta(days=1)).empty


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

def _et(stamp):
    """A moment in exchange time, which is the clock these functions use."""
    return pd.Timestamp(stamp, tz="America/New_York")


def _wide(n_rows=60, n_cols=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-06-01", periods=n_rows)
    steps = rng.normal(0.0005, 0.01, size=(n_rows, n_cols))
    px = 100 * np.exp(np.cumsum(steps, axis=0))
    return pd.DataFrame(px, index=idx,
                        columns=[f"T{i:03d}" for i in range(n_cols)])


def test_a_torn_trailing_bar_is_dropped():
    """The bug this exists for: a download landing while the provider is
    still publishing puts a session-shaped row with twelve names in it into
    a frame of two and a half thousand."""
    from qbs.data import drop_partial_bars

    px = _wide()
    torn = px.copy()
    nxt = px.index[-1] + pd.Timedelta(days=1)
    torn.loc[nxt] = np.nan
    torn.loc[nxt, list(px.columns[:12])] = 100.0

    out, dropped, cover = drop_partial_bars(torn)
    assert [d.date() for d in dropped] == [nxt.date()]
    assert out.index[-1] == px.index[-1]
    assert out.equals(px), "and nothing else is touched"

    # The numbers, so a caller can say 12-of-100 rather than "a handful".
    # "A handful" cannot be told apart from a guard that is simply too
    # strict, and that doubt is what sends somebody refreshing all morning.
    assert cover[nxt] == (12, px.shape[1])


def test_a_complete_frame_is_returned_unchanged():
    """The guard must be invisible on every normal day, or it becomes a
    second source of missing sessions."""
    from qbs.data import drop_partial_bars

    px = _wide()
    out, dropped, cover = drop_partial_bars(px)
    assert dropped == [] and out is px

    # A handful of names legitimately absent is not a torn bar -- names get
    # delisted, and a threshold tight enough to catch that would eat real
    # sessions.
    gappy = px.copy()
    gappy.iloc[-1, :5] = np.nan
    out, dropped, cover = drop_partial_bars(gappy)
    assert dropped == [], "95% coverage is a session, not a tear"


def test_consecutive_torn_bars_all_go():
    """Two reruns inside the settling window leave two of them."""
    from qbs.data import drop_partial_bars

    px = _wide()
    torn = px.copy()
    for i, k in enumerate((12, 5), start=1):
        day = px.index[-1] + pd.Timedelta(days=i)
        torn.loc[day] = np.nan
        torn.loc[day, list(px.columns[:k])] = 100.0

    out, dropped, cover = drop_partial_bars(torn)
    assert len(dropped) == 2
    assert out.index[-1] == px.index[-1]


def test_early_history_is_not_mistaken_for_a_tear():
    """Measured against the recent median, not the column count. A universe
    grows over years, so a fixed fraction of the frame's width would delete
    its own early history."""
    from qbs.data import drop_partial_bars

    px = _wide(n_rows=120, n_cols=100)
    # The first 60 sessions only ever had ten names listed.
    px.iloc[:60, 10:] = np.nan

    out, dropped, cover = drop_partial_bars(px)
    assert dropped == [], "thin history is history, not a torn bar"
    assert len(out) == len(px)


def test_drop_partial_bars_survives_a_frame_too_short_to_judge():
    from qbs.data import drop_partial_bars

    one = _wide(n_rows=1)
    assert drop_partial_bars(one)[1] == []
    assert drop_partial_bars(pd.DataFrame())[1] == []


def test_breadth_over_a_torn_bar_reports_zero_movers():
    """Why the guard is at the data layer and not in the chart.

    Nothing downstream can tell a torn bar from a flat one: the 4%-mover
    counts come back 0 and 0 on a day the market rose, which is exactly what
    a quiet session looks like.
    """
    from qbs.breadth import daily_breadth
    from qbs.data import drop_partial_bars

    px = _wide(n_rows=80)
    qqq = px.mean(axis=1)

    nxt = px.index[-1] + pd.Timedelta(days=1)
    torn = px.copy()
    torn.loc[nxt] = np.nan
    torn.loc[nxt, list(px.columns[:12])] = px.iloc[-1][:12] * 1.012
    qqq2 = pd.concat([qqq, pd.Series({nxt: qqq.iloc[-1] * 1.012})])

    # Breadth now finds that row itself (`thin_rows`) and leaves it out, so
    # the symptom -- 0 and 0 movers on a day the market rose -- cannot reach
    # the table even when a caller forgets to trim.
    res = daily_breadth(torn, qqq=qqq2)
    assert nxt in res.gaps and res.table.index[-1] == px.index[-1]

    # With the row dropped first, the table simply ends at the last real one.
    clean, _, _ = drop_partial_bars(torn)
    good = daily_breadth(clean, qqq=qqq2).table
    assert good.index[-1] == px.index[-1]
    assert (good["n_stocks"] == 100).all()


def test_a_bar_is_not_collectable_the_moment_the_bell_rings():
    """The settling margin. The close is when the session ends, not when a
    provider has finished publishing it -- fetching at 16:01 is how the torn
    bar got in."""
    from qbs.data import BAR_SETTLE, last_market_close

    assert BAR_SETTLE > pd.Timedelta(0)
    friday = pd.Timestamp("2026-09-18").date()
    monday = pd.Timestamp("2026-09-21").date()

    assert last_market_close(_et("2026-09-21 16:01")).date() == friday
    assert last_market_close(_et("2026-09-21 16:59")).date() == friday
    assert last_market_close(_et("2026-09-21 17:00")).date() == monday

    # The exchange close itself is still available, and is a different
    # question from what a provider has published.
    assert last_market_close(_et("2026-09-21 16:01"),
                             settle=pd.Timedelta(0)).date() == monday


def test_sessions_behind_counts_weekdays_only():
    from qbs.data import sessions_behind

    now = _et("2026-09-15 17:30")   # a Tuesday, past the settling window
    assert sessions_behind(pd.Timestamp("2026-09-15"), now) == 0
    assert sessions_behind(pd.Timestamp("2026-09-14"), now) == 1
    assert sessions_behind(pd.Timestamp("2026-09-11"), now) == 2   # Fri -> Mon,Tue
    assert sessions_behind(pd.Timestamp("2026-09-08"), now) == 5


def test_sessions_behind_is_measured_from_the_last_close():
    """Before the close, yesterday's bar IS the newest one published.

    The old reading called that "1 session behind, which is normal during a
    session" -- a fudge that could not tell a missing bar from one that did
    not exist yet, and that made every caller treat 1 as ambiguous.
    """
    from qbs.data import sessions_behind

    monday = pd.Timestamp("2026-09-14")
    tuesday = pd.Timestamp("2026-09-15")
    # Tuesday, holding Monday's close: nothing newer is collectable yet.
    # That stays true through the close itself and the settling window after
    # it -- see `test_a_bar_is_not_collectable_the_moment_the_bell_rings`.
    assert sessions_behind(monday, _et("2026-09-15 09:30")) == 0
    assert sessions_behind(monday, _et("2026-09-15 15:59")) == 0
    assert sessions_behind(monday, _et("2026-09-15 16:30")) == 0
    # Once it settles, Monday's cache is genuinely one bar short.
    assert sessions_behind(monday, _et("2026-09-15 17:00")) == 1
    assert sessions_behind(tuesday, _et("2026-09-15 17:00")) == 0


def test_sessions_behind_does_not_depend_on_the_utc_date():
    """The regression. At 22:00 in New York the UTC date has already rolled
    over, and comparing against it reported a cache holding that very
    afternoon's close as a session behind -- so the banner said "not in yet"
    about a bar six hours old and the loader re-downloaded on every rerun
    trying to fetch a bar it already had."""
    from qbs.data import sessions_behind

    evening = _et("2026-09-21 22:50")          # Monday night = Tuesday UTC
    assert evening.tz_convert("UTC").date() > evening.date(), "fixture premise"
    assert sessions_behind(pd.Timestamp("2026-09-21"), evening) == 0

    # And the same instant expressed in UTC has to give the same answer.
    assert sessions_behind(pd.Timestamp("2026-09-21"),
                           evening.tz_convert("UTC")) == 0
    # A naive timestamp is read as UTC, which is what the stamp file holds.
    assert sessions_behind(pd.Timestamp("2026-09-21"),
                           pd.Timestamp("2026-09-22 02:50")) == 0


def test_sessions_behind_ignores_the_weekend():
    """Saturday and Sunday are not missing sessions."""
    from qbs.data import sessions_behind

    friday = pd.Timestamp("2026-09-11")
    assert sessions_behind(friday, _et("2026-09-11 17:30")) == 0   # Fri, post
    assert sessions_behind(friday, _et("2026-09-12 12:00")) == 0   # Sat
    assert sessions_behind(friday, _et("2026-09-13 12:00")) == 0   # Sun
    assert sessions_behind(friday, _et("2026-09-14 09:30")) == 0   # Mon, pre
    assert sessions_behind(friday, _et("2026-09-14 17:30")) == 1   # Mon, post


def test_sessions_behind_never_goes_negative():
    from qbs.data import sessions_behind

    assert sessions_behind(pd.Timestamp("2026-09-15"),
                           _et("2026-09-10 17:30")) == 0


def test_freshness_note_levels():
    from qbs.data import freshness_note

    now = _et("2026-09-15 17:30")
    assert freshness_note(pd.Timestamp("2026-09-15"), now)[1] == "ok"
    assert freshness_note(pd.Timestamp("2026-09-14"), now)[1] == "info"
    assert freshness_note(pd.Timestamp("2026-09-08"), now)[1] == "warn"
    # The n == 1 message no longer calls a missing bar normal.
    msg = freshness_note(pd.Timestamp("2026-09-14"), now)[2]
    assert "missing" in msg and "normal" not in msg


def test_freshness_note_carries_no_remedy():
    """The fix depends on why it is stale, and only the caller knows that --
    telling someone to go online while their download is the thing failing is
    worse than saying nothing."""
    from qbs.data import freshness_note

    _, _, msg = freshness_note(pd.Timestamp("2026-09-08"),
                               _et("2026-09-15 17:30"))
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


# ---- the once-per-published-bar auto-fetch gate ---------------------------

def test_the_market_close_helpers_track_dst_and_the_weekend():
    from qbs.data import last_market_close, next_market_close

    # The close is 20:00 UTC in summer and 21:00 in winter, which is why the
    # boundary is written in exchange time and not in UTC.
    summer = last_market_close(_et("2026-09-21 17:30"))
    winter = last_market_close(_et("2026-01-21 17:30"))
    assert str(summer.tz_convert("UTC").time()) == "20:00:00"
    assert str(winter.tz_convert("UTC").time()) == "21:00:00"

    # Before the day's bar has settled, the newest one is the previous day's.
    assert last_market_close(_et("2026-09-21 15:59")).date() \
        == pd.Timestamp("2026-09-18").date()
    assert last_market_close(_et("2026-09-21 17:00")).date() \
        == pd.Timestamp("2026-09-21").date()

    # Weekends resolve back to, and forward from, the weekday closes.
    assert last_market_close(_et("2026-09-20 12:00")).date() \
        == pd.Timestamp("2026-09-18").date()
    assert next_market_close(_et("2026-09-19 12:00")).date() \
        == pd.Timestamp("2026-09-21").date()
    assert next_market_close(_et("2026-09-21 17:00")).date() \
        == pd.Timestamp("2026-09-22").date()


def test_no_strftime_directive_is_glibc_only():
    """`%-d` and `%#d` are platform extensions, not strftime.

    The no-padding modifier is `%-d` on glibc and `%#d` on Windows, and each
    is a hard error on the other: the Windows C runtime raises "Invalid
    format string" rather than ignoring it. This repo is developed on Linux
    and run on Windows, so a directive that works here and not there is a
    crash nobody sees until it is in somebody else's hands -- which is
    exactly how one shipped, in a date inside a status line that took the
    whole dashboard down.

    Scanned rather than exercised, because the failure only appears on the
    platform the test is not running on. The pattern deliberately only looks
    inside strftime calls and f-string format specs, so prose like
    "the 4%-mover count" in a docstring does not trip it.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    pattern = re.compile(
        r"""(?:strftime\(\s*["'][^"']*|\{[^{}]*:[^{}]*)%[-#][a-zA-Z]""")

    offenders = []
    for path in sorted(list((root / "qbs").rglob("*.py"))
                       + list((root / "dashboard").rglob("*.py"))):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")

    assert not offenders, (
        "platform-specific strftime directive(s) — build the value in Python "
        "instead, e.g. f\"{ts:%b} {ts.day}\":\n  " + "\n  ".join(offenders))


def _hk(stamp):
    """A moment in the reader's zone, which is the clock the gate uses."""
    return pd.Timestamp(stamp, tz="Asia/Hong_Kong")


def test_the_fetch_gate_allows_the_first_run(tmp_path):
    from qbs.finviz import due_for_fetch

    due, why = due_for_fetch(str(tmp_path / "stamp.txt"),
                             now=_hk("2026-09-22 15:00"))
    assert due and "no automatic fetch" in why


def test_the_gate_opens_only_at_the_scheduled_slot(tmp_path):
    """The rule: once a day at the configured hour, manual at every other.

    The gate counts ATTEMPTS, not data age. Streamlit re-runs the script on
    every widget interaction, so a data-based test would start a 2,600-name
    download on every rerun and never stop.
    """
    from qbs.finviz import due_for_fetch, record_fetch_attempt

    stamp = str(tmp_path / "stamp.txt")

    # Before the slot, nothing -- and the message points at the manual route.
    for t in ("2026-09-22 00:01", "2026-09-22 09:00", "2026-09-22 14:59"):
        due, why = due_for_fetch(stamp, now=_hk(t))
        assert not due, t
        assert "before today's 15:00" in why and "Refresh now" in why

    # On the dot, and once.
    due, _ = due_for_fetch(stamp, now=_hk("2026-09-22 15:00"))
    assert due
    record_fetch_attempt(stamp, now=_hk("2026-09-22 15:00"))

    for t in ("2026-09-22 15:01", "2026-09-22 18:00", "2026-09-22 23:59"):
        due, why = due_for_fetch(stamp, now=_hk(t))
        assert not due, t
        assert "already fetched today" in why
        assert "Wed 15:00" in why, "it has to say when it will try again"


def test_a_missed_slot_is_not_collected_late(tmp_path):
    """"At 15:00, manual otherwise" means the morning after a slot nobody was
    there for starts no download either. It costs only freshness: the
    download is the full history each time, so the next slot picks up both
    days."""
    from qbs.finviz import due_for_fetch, record_fetch_attempt

    stamp = str(tmp_path / "stamp.txt")
    record_fetch_attempt(stamp, now=_hk("2026-09-22 15:00"))

    # Wednesday's slot passes unattended...
    assert not due_for_fetch(stamp, now=_hk("2026-09-24 09:00"))[0], \
        "Thursday morning must not collect Wednesday's missed slot"
    assert "before today's" in due_for_fetch(stamp,
                                             now=_hk("2026-09-24 09:00"))[1]
    # ...and Thursday's own slot runs normally.
    assert due_for_fetch(stamp, now=_hk("2026-09-24 15:00"))[0]


def test_the_slot_does_not_spend_a_download_on_a_dead_weekend(tmp_path):
    """At 15:00 UTC+8 on a Sunday the newest bar is still Friday's, which
    Saturday's slot already collected."""
    from qbs.finviz import due_for_fetch, record_fetch_attempt

    stamp = str(tmp_path / "stamp.txt")

    # Saturday's slot is the first one after Friday's close, so it runs.
    assert due_for_fetch(stamp, now=_hk("2026-09-26 15:00"))[0]
    record_fetch_attempt(stamp, now=_hk("2026-09-26 15:00"))

    due, why = due_for_fetch(stamp, now=_hk("2026-09-27 15:00"))
    assert not due, "Sunday has nothing new to fetch"
    assert "nothing new since" in why


def test_the_gate_compares_the_stamp_in_the_right_zone(tmp_path):
    """The stamp is written in naive UTC and the slot is in the reader's
    zone. Comparing them without converting is a whole class of bug this
    file has had before, one layer down."""
    from qbs.finviz import due_for_fetch, record_fetch_attempt

    stamp = str(tmp_path / "stamp.txt")
    at = _hk("2026-09-22 18:00")           # after the slot, so it can fire

    # 07:00 UTC on 9/22 IS 15:00 in UTC+8 -- the slot itself, so this counts
    # as today's fetch. Read as naive UTC against a UTC+8 slot it would look
    # like 07:00 local, hours early, and the gate would fire a second time.
    record_fetch_attempt(stamp, now=pd.Timestamp("2026-09-22 07:00"))
    due, why = due_for_fetch(stamp, now=at)
    assert not due and "already fetched today" in why

    # 19:00 UTC on 9/21 is 03:00 UTC+8 on the 22nd: before the slot, and also
    # before the Sep 21 close landed in this zone at 04:00. Both conditions
    # therefore pass, which is what isolates the comparison being tested --
    # a stamp at 14:00 local would be before the slot but AFTER the close,
    # and would be held by the other condition for an unrelated reason.
    record_fetch_attempt(stamp, now=pd.Timestamp("2026-09-21 19:00"))
    assert due_for_fetch(stamp, now=at)[0]


def test_the_schedule_is_configurable_and_survives_a_typo():
    """Read on a dashboard's start-up path, so a bad value costs a line on
    screen rather than a page that will not load -- and a SILENT fallback
    would have someone waiting all afternoon for a fetch scheduled at an hour
    they think they changed."""
    from qbs.finviz import DEFAULT_FETCH_AT, DEFAULT_FETCH_TZ, fetch_schedule

    hh, mm, tz, note = fetch_schedule({})
    assert (hh, mm) == tuple(int(x) for x in DEFAULT_FETCH_AT.split(":"))
    assert tz == DEFAULT_FETCH_TZ and note is None

    hh, mm, tz, note = fetch_schedule({"QBS_FETCH_AT": "06:30",
                                       "QBS_FETCH_TZ": "Europe/London"})
    assert (hh, mm, tz) == (6, 30, "Europe/London") and note is None

    hh, mm, tz, note = fetch_schedule({"QBS_FETCH_AT": "25:99",
                                       "QBS_FETCH_TZ": "Mars/Olympus"})
    assert (hh, mm, tz) == (15, 0, DEFAULT_FETCH_TZ), "falls back"
    assert note and "25:99" in note and "Mars/Olympus" in note


def test_the_epoch_key_flips_at_the_slot_not_the_close():
    """What the dashboard's caches are keyed on, and it has to answer to the
    same clock as the gate.

    Key it on the close and an app opened at 10:00 memoises "not due yet"
    under a key that will not change again until the next close -- which is
    AFTER the 15:00 slot -- so the scheduled fetch is memoised away and never
    happens at all.
    """
    from qbs.finviz import fetch_epoch

    pre = fetch_epoch(_hk("2026-09-22 09:00"))
    assert pre == fetch_epoch(_hk("2026-09-22 14:59")), "steady all morning"
    # The US close lands at 04:00 UTC+8; the key must NOT move for it.
    assert pre == fetch_epoch(_hk("2026-09-22 05:00"))

    post = fetch_epoch(_hk("2026-09-22 15:00"))
    assert post != pre, "the slot is the only thing that moves it"
    assert post == fetch_epoch(_hk("2026-09-22 23:59"))
    assert post == fetch_epoch(_hk("2026-09-23 09:00")), "and overnight"
    assert fetch_epoch(_hk("2026-09-23 15:00")) != post


def test_an_unwritable_stamp_does_not_break_the_fetch(tmp_path):
    """Worst case it costs one extra attempt tomorrow. Raising here would turn
    a read-only data directory into a dashboard that will not start."""
    from qbs.finviz import last_fetch_attempt, record_fetch_attempt

    bad = str(tmp_path / "nope" / "\x00" / "stamp.txt")
    record_fetch_attempt(bad)             # must not raise
    assert last_fetch_attempt(bad) is None


def test_a_corrupt_stamp_reads_as_never_fetched(tmp_path):
    from qbs.finviz import due_for_fetch

    stamp = tmp_path / "stamp.txt"
    stamp.write_text("not a timestamp")
    # Pinned past the slot: this is about the unreadable STAMP, and left on
    # the wall clock it would pass in the afternoon and fail in the morning.
    assert due_for_fetch(str(stamp), now=_hk("2026-09-22 15:00"))[0], \
        "unreadable means unknown means try"


def test_a_stale_bars_cache_stops_counting_as_a_hit(tmp_path):
    """A cache hit that never asks how old it is pins the app to whatever was
    on disk when it started. With `stale_after` the hit expires; without it the
    old behaviour is kept for callers that manage freshness themselves."""
    import os
    from qbs.finviz import load_universe_bars

    old_idx = pd.bdate_range("2020-01-01", periods=6)     # years behind
    d = str(tmp_path)
    pd.DataFrame({"AAPL": 1.0}, index=old_idx).rename_axis("Date") \
        .to_csv(os.path.join(d, "us_closes.csv"))
    pd.DataFrame({"AAPL": 10}, index=old_idx).rename_axis("Date") \
        .to_csv(os.path.join(d, "us_volumes.csv"))

    # No staleness limit: the stale cache is returned, as before.
    c, _, err = load_universe_bars(["AAPL"], cache_dir=d, verbose=False)
    assert c is not None and err is None

    # With one: the cache is rejected, so the loader goes to the network. There
    # is none here, so it must come back with a REASON rather than silently
    # handing over the stale frame it just declined to trust.
    c2, _, err2 = load_universe_bars(["AAPL"], cache_dir=d, verbose=False,
                                     stale_after=1)
    assert err2, "a rejected cache and a failed download must report something"


def test_a_torn_cached_bar_does_not_pass_as_a_fresh_cache(tmp_path):
    """The bar that made the dashboard sit on Friday's numbers all Monday.

    A torn bar carries the current date, so `c.index.max()` says the cache is
    current, `stale_after` is satisfied, and the download that would replace
    it never runs. Cleaning the frame AFTER this function returns fixes what
    is drawn and not what is decided -- the check has to be asked about the
    last WHOLE bar.
    """
    import os

    from qbs.finviz import load_universe_bars

    idx = pd.bdate_range("2026-09-01", "2026-09-21")
    names = [f"T{i:04d}" for i in range(200)]
    closes = pd.DataFrame(100.0, index=idx, columns=names)
    # Monday's bar arrived for twelve names out of two hundred.
    closes.loc[pd.Timestamp("2026-09-21")] = np.nan
    closes.loc[pd.Timestamp("2026-09-21"), names[:12]] = 100.0

    d = str(tmp_path)
    closes.rename_axis("Date").to_csv(os.path.join(d, "us_closes.csv"))
    (closes * 0 + 1e6).rename_axis("Date").to_csv(
        os.path.join(d, "us_volumes.csv"))

    c, v, err = load_universe_bars(names, cache_dir=d, offline=True,
                                   verbose=False)
    assert c.index.max() == pd.Timestamp("2026-09-18"), "the torn bar is gone"
    assert v.index.max() == pd.Timestamp("2026-09-18"), "volumes follow it"
    assert err and "2026-09-21" in err, "and it is named, not dropped in silence"
    # With the numbers: "a handful" cannot be told apart from a guard that is
    # simply too strict, so it gets read as a bug and refreshed at all morning.
    assert "12 of ~200 names" in err

    # The cache file itself is untouched: those twelve names are the head of a
    # real bar, and the next fetch fills the rest in rather than starting over.
    on_disk = pd.read_csv(os.path.join(d, "us_closes.csv"),
                          parse_dates=["Date"], index_col="Date")
    assert on_disk.index.max() == pd.Timestamp("2026-09-21")

    # And with a staleness limit the cache is now correctly judged one session
    # behind, so the loader goes to the network instead of serving the tear.
    c2, _, err2 = load_universe_bars(names, cache_dir=d, verbose=False,
                                     stale_after=1)
    assert err2, "a torn cache must not satisfy stale_after"


def test_a_complete_cache_is_still_served_untouched(tmp_path):
    """The guard must be invisible on every normal day, or it turns into a
    second source of unnecessary 2,400-name downloads."""
    import os

    from qbs.finviz import load_universe_bars

    idx = pd.bdate_range("2026-09-01", "2026-09-21")
    names = [f"T{i:04d}" for i in range(200)]
    closes = pd.DataFrame(100.0, index=idx, columns=names)
    d = str(tmp_path)
    closes.rename_axis("Date").to_csv(os.path.join(d, "us_closes.csv"))
    (closes * 0 + 1e6).rename_axis("Date").to_csv(
        os.path.join(d, "us_volumes.csv"))

    c, _, err = load_universe_bars(names, cache_dir=d, offline=True,
                                   verbose=False)
    assert c.index.max() == pd.Timestamp("2026-09-21")
    assert err is None


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

    # Both legs this test is about are OFF in the defaults now, so the
    # rejection it pins only exists when they are switched on.
    screen = _screen(within_52w_high_pct=0.10, above_sma=200)
    g = momentum_profile(px, "PULLBACK", screen=screen)["gates"].set_index("Rule")
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

    def gates(**kw):
        return momentum_profile(
            px, "PULLBACK", screen=_screen(**kw))["gates"].set_index("Rule")

    # A leg that is switched OFF must have NO row. A row for a filter nobody
    # is running describes a strategy that does not exist, and the reader has
    # no way to tell that from the table.
    default = gates()
    assert not any("high" in r for r in default.index), \
        "within_52w_high_pct is None by default -- no proximity row"
    assert not any("SMA" in r for r in default.index), \
        "above_sma is 0 by default -- no moving-average row"
    assert not any("quarter up" in r for r in default.index)
    assert any("quarterly gain over 28%" in r for r in default.index), \
        "the leg that IS on must be reported, at its configured threshold"

    # 14% off the high fails a 10% ceiling and clears a 20% one.
    tight = gates(within_52w_high_pct=0.10)
    loose = gates(within_52w_high_pct=0.20)
    assert not tight.loc["within 10% of 252-day high", "Pass"]
    assert loose.loc["within 20% of 252-day high", "Pass"]

    # A configured band floor adds its row; a ceiling-only rule must not claim
    # a floor the screen does not apply.
    assert not any("off the high" in r for r in tight.index)
    banded = gates(min_off_high_pct=0.05)
    assert banded.loc["at least 5% off the high", "Pass"], "14% off clears a 5% floor"

    # And the legs that can be switched back on appear when they are.
    assert any("quarter up" in r for r in gates(require_quarter_up=True).index)
    assert any("above SMA 200" in r for r in gates(above_sma=200).index)


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


# --------------------------------------------------------------------------
# Shadow books
# --------------------------------------------------------------------------

def test_alternative_score_leaves_the_default_path_untouched():
    """The live book must be bit-identical whether or not the feature exists."""
    uni, safe = _mom_fixture()
    p = MomentumParams(n_hold=6, exit_rank=8)
    assert cross_sectional_momentum(uni, safe, p).weights.equals(
        cross_sectional_momentum(uni, safe, p, score=None).weights)


def test_alternative_score_changes_the_order_but_not_the_absolute_filter():
    """Ranking on something else must still refuse names that lose to cash.

    The filter is a statement about a name's own return. Reversing the ordering
    is the strongest possible test: every name the default would rank last is
    now first, and yet nothing failing the hurdle may be held.
    """
    from qbs.shadow import turn_score
    uni, safe = _mom_fixture()
    p = MomentumParams(n_hold=6, exit_rank=8, absolute_filter=True)
    base = cross_sectional_momentum(uni, safe, p)
    flipped = cross_sectional_momentum(uni, safe, p, score=-base.momentum)
    assert not flipped.weights.equals(base.weights), "the score did not reach the ranking"

    look, skip = int(round(p.lookback_months * 21)), int(round(p.skip_months * 21))
    hurdle = safe.shift(skip) / safe.shift(look) - 1.0
    for sig in (base, flipped):
        risk = sig.weights.drop(columns=["BOXX"])
        for dt in risk.index[-40:]:
            if np.isnan(hurdle.loc[dt]):
                continue
            for t in risk.columns[risk.loc[dt] > 0]:
                assert sig.momentum.loc[dt, t] > hurdle.loc[dt], \
                    f"{dt} {t}: held a name that lost to the safe asset"

    # And the tilt is a real reordering, not a no-op dressed up as one.
    tilted = cross_sectional_momentum(uni, safe, p, score=turn_score(uni, 2.0, p))
    assert not tilted.weights.equals(base.weights)


def test_turn_score_compares_rates_not_window_lengths():
    """A constant-growth name must not read as accelerating.

    The 1-0 window is 21 days and the 3-1 window is 42. Subtracting the two
    window returns directly measures the length difference far more than any
    change in pace -- a name compounding at a steady 0.1%/day shows a 2%
    'deceleration' that is pure arithmetic. Dividing each by its own length
    removes it. This is the trap that has already produced one wrong answer in
    this project's research, so it is pinned here.
    """
    idx = pd.bdate_range("2023-01-02", periods=300)
    rates = [0.0005, 0.001, 0.0015, 0.002, 0.003]
    steady = pd.DataFrame(
        {f"S{i}": 100 * (1 + g) ** np.arange(300) for i, g in enumerate(rates)},
        index=idx)
    skip, quarter = 21, 63

    recent = (steady / steady.shift(skip) - 1.0)
    prior = (steady.shift(skip) / steady.shift(quarter) - 1.0)
    raw = (recent - prior).iloc[quarter + 1:]                 # windows not normalised
    rate = (recent / skip - prior / (quarter - skip)).iloc[quarter + 1:]

    assert raw.abs().max().max() > 0.01, "the length artifact should be large"
    # Scale-free, so the bound does not drift if the fixture's growth rates
    # change: normalising kills the artifact by ~three orders of magnitude.
    assert rate.abs().max().max() < raw.abs().max().max() / 500
    assert raw.abs().mean().mean() > 500 * rate.abs().mean().mean()


def test_shadow_books_are_scored_but_never_held():
    from qbs.shadow import shadow_books
    uni, safe = _mom_fixture()
    p = MomentumParams(n_hold=6, exit_rank=8)
    rows = shadow_books(uni, safe, p, weights=[0.0, 2.0])
    assert {r["weight"] for r in rows} == {0.0, 2.0}
    for w in (0.0, 2.0):
        picks = [r for r in rows if r["weight"] == w]
        assert len(picks) <= p.n_hold
        assert len({r["symbol"] for r in picks}) == len(picks), "a name held twice"
        assert all(1 <= r["rank"] <= p.exit_rank for r in picks)
    # w=0 is plain momentum, so it must agree with the live book exactly.
    live = cross_sectional_momentum(uni, safe, p)
    dt = live.weights.index[-1]
    assert {r["symbol"] for r in rows if r["weight"] == 0.0} == \
        set(live.holdings_log[dt])


def test_shadow_books_swallow_a_broken_candidate():
    """A log must not be able to stop a run. A score full of NaN is the most
    likely way a candidate breaks in production -- a name delisted mid-window."""
    from qbs.shadow import shadow_books
    import qbs.shadow as shadow_mod
    uni, safe = _mom_fixture()
    original = shadow_mod.turn_score
    shadow_mod.turn_score = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        assert shadow_books(uni, safe, MomentumParams(), weights=[1.0]) == []
    finally:
        shadow_mod.turn_score = original


# --------------------------------------------------------------------------
# Universe providers (qbs/universe_source.py, qbs/tradingview.py)
# --------------------------------------------------------------------------

def test_the_two_filter_vocabularies_agree():
    """`UniverseFilters` says the same thing twice, in two languages.

    The strings are Finviz's own filter enum, which takes "Over $5" and has
    no general numeric form, so they cannot be derived from the numbers. That
    makes drift possible, and a definition that says one thing to one
    provider and another to the next produces two universes and one number on
    screen.
    """
    from qbs.finviz import UniverseFilters

    f = UniverseFilters()
    assert f.price == f"Over ${f.min_price:g}"
    assert f.avg_volume == f"Over {f.min_avg_volume / 1000:g}K"
    assert "ex-Funds" in f.industry, "both sides exclude funds"
    assert f.include_dr, "and both keep ADRs — the label says so"


def test_the_source_switch_defaults_to_finviz():
    """Every number in this repo has been measured against Finviz, so a
    different provider is a change of measurement and not a preference."""
    from qbs.universe_source import DEFAULT_SOURCE, resolve_source

    assert DEFAULT_SOURCE == "finviz"
    assert resolve_source(environ={}) == "finviz"
    assert resolve_source(environ={"QBS_UNIVERSE_SOURCE": "tradingview"}) \
        == "tradingview"
    # Case and padding are how people actually type it.
    assert resolve_source(environ={"QBS_UNIVERSE_SOURCE": " TradingView "}) \
        == "tradingview"
    # An explicit argument beats the environment.
    assert resolve_source("finviz",
                          environ={"QBS_UNIVERSE_SOURCE": "tradingview"}) \
        == "finviz"


def test_an_unknown_source_falls_back_and_says_so():
    """Read on a dashboard's start-up path, so a typo must cost a line on
    screen rather than a page that will not load."""
    from qbs.universe_source import resolve_source, source_note

    env = {"QBS_UNIVERSE_SOURCE": "tradinview"}      # missing the 'g'
    assert resolve_source(environ=env) == "finviz"
    note = source_note(environ=env)
    assert note and "tradinview" in note and "finviz" in note
    assert source_note(environ={"QBS_UNIVERSE_SOURCE": "tradingview"}) is None
    assert source_note(environ={}) is None


def test_fetch_universe_reports_which_provider_ran(monkeypatch):
    """Two providers are two universes. A breadth count that steps because
    the source changed, on a screen that does not say the source changed,
    reads as a market event."""
    import qbs.universe_source as us

    called = {}

    def fake(filters, **kw):
        called["filters"] = filters
        return pd.DataFrame({"Ticker": ["AAA"], "Sector": ["Tech"]}), None

    monkeypatch.setattr(us, "_provider", lambda name: fake)

    uni, err, src = us.fetch_universe(source="tradingview")
    assert src == "tradingview" and err is None and len(uni) == 1

    # And the typo note rides out on the error channel, not in silence.
    monkeypatch.setenv("QBS_UNIVERSE_SOURCE", "nope")
    uni, err, src = us.fetch_universe()
    assert src == "finviz" and err and "nope" in err


def test_the_tradingview_provider_matches_the_finviz_signature():
    """One seam, two implementations. A caller holds either behind one name,
    so every argument the other takes has to be accepted here -- including
    `sleep_sec`, which paces Finviz's 120 page requests and has nothing to
    pace in a single POST."""
    import inspect

    from qbs import finviz, tradingview

    a = inspect.signature(finviz.fetch_us_universe).parameters
    b = inspect.signature(tradingview.fetch_us_universe).parameters
    assert set(a) == set(b), f"signatures diverged: {set(a) ^ set(b)}"


def test_the_tradingview_provider_needs_no_network_to_fail_politely(tmp_path):
    """`(None, reason)`, never a raise: the caller is a dashboard that falls
    back, and a failure it cannot read off the screen cannot be fixed."""
    from qbs.tradingview import fetch_us_universe

    uni, err = fetch_us_universe(offline=True, verbose=False,
                                 cache_path=str(tmp_path / "none.csv"))
    assert uni is None and err and "offline" in err


def test_the_tradingview_provider_caches_separately():
    """Sharing Finviz's cache file would mean a switch silently reads the
    other provider's answer and reports it as this one's."""
    from qbs.finviz import UNIVERSE_CSV as FINVIZ_CSV
    from qbs.tradingview import UNIVERSE_CSV as TV_CSV

    assert FINVIZ_CSV != TV_CSV
    assert "tradingview" in TV_CSV


def test_the_scanner_frame_is_shaped_like_the_finviz_one():
    """Same columns, same ticker spelling. The two universes have to produce
    keys that match the same yfinance price frame, so BRK.B normalises to
    BRK-B on both sides."""
    from qbs.tradingview import _shape

    raw = pd.DataFrame({
        "name": ["AAPL", "BRK.B", "TSM", "AAPL"],
        "ticker": ["NASDAQ:AAPL", "NYSE:BRK.B", "NYSE:TSM", "NASDAQ:AAPL"],
        "sector": ["Technology", "Finance", "Technology", "Technology"],
        "industry": ["Hardware", "Insurance", "Semis", "Hardware"],
        "country": ["US", "US", "TW", "US"],
    })
    out = _shape(raw)
    assert list(out.columns) == ["Ticker", "Sector", "Industry", "Country"]
    assert list(out["Ticker"]) == ["AAPL", "BRK-B", "TSM"], "dots and dupes"

    # `name` can go missing between versions; the exchange-qualified form is
    # the fallback and has to give the same answer.
    out2 = _shape(raw.drop(columns=["name"]))
    assert list(out2["Ticker"]) == ["AAPL", "BRK-B", "TSM"]

    with pytest.raises(RuntimeError, match="no symbol column"):
        _shape(raw.drop(columns=["name", "ticker"]))


def test_the_tradingview_provider_does_not_apply_the_momentum_rule():
    """`Perf.3M` is a calendar quarter and this package's rule is 63
    SESSIONS. Close enough to look interchangeable, not the same number --
    so the rule stays in `leader_mask`, measured on the bars, and the
    provider's only job is membership."""
    from qbs import tradingview

    src = Path(tradingview.__file__).read_text(encoding="utf-8")
    assert "Perf.3M" not in tradingview.COLUMNS
    assert "col(\"Perf" not in src and "col('Perf" not in src
    # It filters on membership and liquidity only.
    assert "average_volume_90d_calc" in src, "the AVERAGE, as Finviz does"


def test_symbol_normalisation_covers_classes_and_preferreds():
    """Two substitutions, both to a dash, and each has cost a download.

    `.` is a share class (BRK.B). `/` is a preferred series: the screener
    returns ORCL/PD and Yahoo wants ORCL-PD, and left alone it 404s as
    "possibly delisted; no timezone found" -- which reads like a dead company
    rather than a misspelled symbol, so it gets diagnosed as a data problem.
    """
    from qbs.data import normalise_symbols

    got = list(normalise_symbols(
        ["brk.b", "ORCL/PD", "HPE/PC", " aapl ", "BF.B", "MSFT"]))
    assert got == ["BRK-B", "ORCL-PD", "HPE-PC", "AAPL", "BF-B", "MSFT"]
    assert list(normalise_symbols([])) == []


def test_every_universe_provider_normalises_the_same_way():
    """A normalisation only some providers apply produces keys that do not
    match the same price frame, and the symptom is a name silently missing
    from a universe rather than an error."""
    from qbs.data import normalise_symbols
    from qbs.tradingview import _shape

    raw = pd.DataFrame({"name": ["BRK.B", "ORCL/PD", "aapl"],
                        "sector": ["Finance", "Tech", "Tech"],
                        "industry": ["x", "y", "z"],
                        "country": ["US", "US", "US"]})
    assert list(_shape(raw)["Ticker"]) == ["BRK-B", "ORCL-PD", "AAPL"]

    # And nobody keeps a private copy of the chain any more.
    for mod in ("qbs/finviz.py", "qbs/tradingview.py", "qbs/universe.py"):
        src = Path(mod).read_text(encoding="utf-8")
        assert 'str.replace(".", "-"' not in src, f"{mod} still rolls its own"
        assert "normalise_symbols" in src, f"{mod} does not normalise at all"


# --------------------------------------------------------------------------
# Quarantine for symbols the provider will not serve
# --------------------------------------------------------------------------

def test_a_failing_ticker_is_quarantined_then_retried(tmp_path):
    """Yahoo answers an unknown symbol with "possibly delisted; no timezone
    found" — once per ticker, per batch, per run. Thirty of those bury every
    real message in the log and spend a slice of every download rediscovering
    the same answer.

    It EXPIRES on purpose. A permanent blacklist would shrink the universe by
    one name every time the provider had a bad minute, silently and for ever,
    with nothing to put a name back.
    """
    from qbs.data import QUARANTINE_DAYS, load_quarantine, record_failures

    d = str(tmp_path)
    now = pd.Timestamp("2026-09-23 12:00")
    assert load_quarantine(d, now=now) == {}, "nothing banned to begin with"

    record_failures(d, ["HYAC-U", "ORCL-PD"], now=now)
    banned = load_quarantine(d, now=now)
    assert set(banned) == {"HYAC-U", "ORCL-PD"}
    assert banned["HYAC-U"]["fails"] == 1

    # Failing again counts, rather than resetting.
    record_failures(d, ["HYAC-U"], now=now + pd.Timedelta(days=1))
    assert load_quarantine(d, now=now + pd.Timedelta(days=1))["HYAC-U"]["fails"] == 2

    # And it lapses, so a symbol that starts working is tried again without
    # anybody editing a file.
    later = now + pd.Timedelta(days=QUARANTINE_DAYS + 2)
    assert load_quarantine(d, now=later) == {}


def test_the_quarantine_survives_a_corrupt_or_missing_file(tmp_path):
    """It sits on a download's start-up path. A bad file has to cost a noisy
    log, never a download that will not start."""
    from qbs.data import _quarantine_path, clear_quarantine, load_quarantine, \
        record_failures

    d = str(tmp_path)
    assert load_quarantine(d) == {}, "missing file is not an error"

    Path(_quarantine_path(d)).write_text("this is not a csv at all\x00")
    assert load_quarantine(d) == {}, "unreadable file is not an error"

    Path(_quarantine_path(d)).write_text("wrong,columns\n1,2\n")
    assert load_quarantine(d) == {}, "wrong schema is not an error"

    record_failures(d, ["AAA"], now=pd.Timestamp("2026-09-23"))
    assert "AAA" in load_quarantine(d, now=pd.Timestamp("2026-09-23"))
    clear_quarantine(d)
    assert load_quarantine(d, now=pd.Timestamp("2026-09-23")) == {}
    clear_quarantine(d)        # removing a file that is gone is not an error


def test_quarantined_tickers_are_not_requested_again(tmp_path, monkeypatch):
    """The point of the list: the second run does not spend a request finding
    out what the first run already learned."""
    import qbs.finviz as fz
    from qbs.data import record_failures

    d = str(tmp_path)
    record_failures(d, ["DEAD-A", "DEAD-B"], now=pd.Timestamp.now("UTC").tz_localize(None))

    asked = []

    class FakeYF:
        @staticmethod
        def download(batch, **kw):
            asked.extend(batch)
            idx = pd.bdate_range("2026-06-01", periods=80)
            cols = pd.MultiIndex.from_product([["Close", "Volume"], batch])
            vals = np.column_stack([np.full((len(idx), len(batch)), 10.0),
                                    np.full((len(idx), len(batch)), 1e6)])
            return pd.DataFrame(vals, index=idx, columns=cols)

    monkeypatch.setitem(sys.modules, "yfinance", FakeYF)
    c, v, err = fz.load_universe_bars(["GOOD", "DEAD-A", "DEAD-B"],
                                      cache_dir=d, verbose=False)
    assert asked == ["GOOD"], f"the dead ones were asked anyway: {asked}"
    assert c is not None and list(c.columns) == ["GOOD"]
    assert err and "2 ticker(s) skipped" in err, "and it says so, not silently"


# --------------------------------------------------------------------------
# Topping up the newest bar from the screener (qbs/quotes.py)
# --------------------------------------------------------------------------

def _bars(end="2026-09-21", names=10):
    idx = pd.bdate_range("2026-09-01", end)
    cols = [f"T{i}" for i in range(names)]
    return (pd.DataFrame(100.0, index=idx, columns=cols),
            pd.DataFrame(1e6, index=idx, columns=cols))


def _quotes(names=8, close=101.0):
    return pd.DataFrame({"close": [close] * names, "volume": [2e6] * names},
                        index=[f"T{i}" for i in range(names)])


def test_the_front_bar_is_filled_once_the_session_has_closed():
    """The bar yfinance has not published is the one the screener already
    returned, in a request the app was making anyway."""
    from qbs.quotes import fill_last_bar, fill_note

    c, v = _bars()
    out, vol, rep = fill_last_bar(c, v, _quotes(), now=_et("2026-09-22 18:00"))

    day = pd.Timestamp("2026-09-22")
    assert out.index.max() == day, "the missing session is now there"
    assert rep["filled"] == 8 and rep["absent"] == 2
    assert float(out.at[day, "T0"]) == 101.0
    assert float(vol.at[day, "T0"]) == 2e6
    assert pd.isna(out.at[day, "T9"]), "a name the screener lacks stays empty"
    note = fill_note(rep, "finviz")
    assert note and "8 of 10" in note and "finviz" in note


def test_an_open_session_is_refused_not_filled():
    """A screener queried while the market is open returns an INTRADAY print.
    Dropping one into a series of closes makes every return computed across
    it wrong, and nothing downstream can tell."""
    from qbs.quotes import fill_last_bar

    c, v = _bars()
    out, _, rep = fill_last_bar(c, v, _quotes(), now=_et("2026-09-22 11:00"),
                                session=pd.Timestamp("2026-09-22"))
    assert rep["filled"] == 0
    assert "has not closed" in rep["skipped"] and "intraday" in rep["skipped"]
    assert out.index.max() == pd.Timestamp("2026-09-21"), "frame untouched"


def test_the_fill_never_overwrites_a_real_bar():
    """yfinance is the authority for anything it actually published; the
    screener only reaches the gaps."""
    from qbs.quotes import fill_last_bar

    c, v = _bars()
    day = pd.Timestamp("2026-09-22")
    c.loc[day] = np.nan
    c.loc[day, ["T0", "T1", "T2"]] = 99.0          # yfinance got three

    out, _, rep = fill_last_bar(c, v, _quotes(), now=_et("2026-09-22 18:00"))
    assert rep["already"] == 3 and rep["filled"] == 5 and rep["absent"] == 2
    assert list(out.loc[day, ["T0", "T1", "T2"]]) == [99.0, 99.0, 99.0]


def test_a_complete_session_is_left_alone():
    from qbs.quotes import fill_last_bar

    c, v = _bars(end="2026-09-22")
    out, _, rep = fill_last_bar(c, v, _quotes(), now=_et("2026-09-22 18:00"))
    assert rep["filled"] == 0 and "already complete" in rep["skipped"]
    assert out.equals(c)


def test_the_screener_is_not_asked_on_a_normal_day():
    """The top-up is for the hours yfinance is behind, not a second provider
    on every page load."""
    from qbs.quotes import needs_fill

    c, _ = _bars(end="2026-09-22")
    assert not needs_fill(c, _et("2026-09-22 18:00")), "complete: no request"
    assert not needs_fill(c, _et("2026-09-22 11:00")), "mid-session: no request"
    assert needs_fill(c.iloc[:-1], _et("2026-09-22 18:00")), "missing: ask"
    torn = c.copy()
    torn.loc[c.index[-1], "T9"] = np.nan
    assert needs_fill(torn, _et("2026-09-22 18:00")), "torn: ask"


def test_the_top_up_can_be_switched_off():
    """It ships on, because the front edge being unreliable is why it exists
    and a remedy that ships off is one nobody gets. But a series that must be
    purely yfinance has to be reachable."""
    from qbs.quotes import FILL_VAR, fill_disabled

    assert not fill_disabled({}), "on by default"
    for off in ("0", "false", "no", "off", "none", "disabled"):
        assert fill_disabled({FILL_VAR: off}), off
    assert not fill_disabled({FILL_VAR: "1"})


def test_screener_quotes_are_shaped_and_normalised():
    """Same ticker spelling as the price frames, or the fill lands on nothing."""
    from qbs.quotes import _shape_quotes

    out = _shape_quotes(pd.Series(["brk.b", "ORCL/PD", "AAPL", "AAPL", "BAD"]),
                        pd.Series([1.0, 2.0, 3.0, 3.0, None]),
                        pd.Series([10, 20, 30, 30, 40]))
    assert list(out.index) == ["BRK-B", "ORCL-PD", "AAPL"], "normalised, deduped"
    assert float(out.at["AAPL", "close"]) == 3.0
    assert "BAD" not in out.index, "a quote with no price is not a quote"


# ==========================================================================
# The dashboard's third book
# ==========================================================================

def _dashboard_ast():
    """`dashboard/app.py` parsed, never imported.

    Importing it would run a Streamlit script top to bottom -- downloads,
    widgets and all -- so the checks below read the source instead. That is
    enough for the failures they are about, which are all wiring.
    """
    import ast

    root = Path(__file__).resolve().parent.parent
    return ast.parse((root / "dashboard" / "app.py").read_text(encoding="utf-8"))


def _tuple_strings(node):
    import ast

    return [e.value for e in node.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)]


def test_the_residual_band_survives_every_book_size_the_dashboard_offers():
    """The band is a WIDTH, and carrying it over as a rank would be a crash.

    `ResidualMomentumParams` ships (n_hold=6, exit_rank=10): a band four ranks
    wide. The dashboard runs the book ten deep, and reusing the number 10
    there would mean a band of zero -- every name round-tripped the moment it
    wobbled -- while anything deeper than ten would raise outright, taking the
    whole page down. So the dashboard adds the width, and this pins that the
    width works across the entire range its control allows.
    """
    p = ResidualMomentumParams()
    band = p.exit_rank - p.n_hold
    assert band > 0, "a band of zero is a round-trip machine, not a band"

    for n in range(1, 31):          # the number_input's range
        q = ResidualMomentumParams(n_hold=n, exit_rank=n + band)
        assert q.exit_rank - q.n_hold == band, n

    # What carrying the number over instead would do, at the two sizes that
    # show both halves of it: a silent band of zero at ten slots, and a hard
    # ValueError at eleven. Neither is a thing to ship, and the second would
    # take the page down rather than merely trade badly.
    flat = ResidualMomentumParams(n_hold=p.exit_rank, exit_rank=p.exit_rank)
    assert flat.exit_rank - flat.n_hold == 0, "the silent half of the bug"
    try:
        ResidualMomentumParams(n_hold=p.exit_rank + 1, exit_rank=p.exit_rank)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative band should still be rejected")


def test_momentum_label_reads_the_residual_params_too():
    """The two books run different lookbacks, and both headers say which.

    The dashboard names the residual column with `momentum_label` applied to
    `ResidualMomentumParams`, which works because the helper only ever reads
    the two lookback fields. If it grew an isinstance check the column would
    start claiming the total-return book's window, which is the exact kind of
    quiet mislabelling `momentum_label` exists to prevent.
    """
    from qbs.breadth import momentum_label

    resid = momentum_label(ResidualMomentumParams())
    assert resid == "12-1", resid
    assert resid != momentum_label(MomentumParams()), (
        "the two books would be labelled with the same window")


def test_the_picks_tab_and_the_selection_builder_agree_on_the_books():
    """Three lists of strategy keys, one KeyError if they drift.

    `build_selections` writes the frames, `strategy_labels` names them and the
    picks tab iterates them into columns. A key added to one and not the
    others is not caught by anything else here — it surfaces as a KeyError on
    a running dashboard, which is to say on somebody's screen.
    """
    import ast

    tree = _dashboard_ast()

    built = labelled = columns = None
    for node in ast.walk(tree):
        # `for key, sig in (("momentum", mom), ("resmom", res), ...)`.
        # Matched on the loop variables too, so an unrelated tuple-of-tuples
        # loop added later cannot shadow this one and fail the test for a
        # reason its message would not explain.
        if (isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
                and isinstance(node.target, ast.Tuple)
                and [getattr(t, "id", None) for t in node.target.elts]
                    == ["key", "sig"]):
            built = [e.elts[0].value for e in node.iter.elts]
        if (isinstance(node, ast.FunctionDef)
                and node.name == "strategy_labels"):
            ret = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
            labelled = [k.value for k in ret.value.keys]
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
                and any(getattr(t, "id", None) == "books" for t in node.targets)):
            columns = _tuple_strings(node.value)

    assert built and labelled and columns, (built, labelled, columns)
    assert set(built) == set(labelled) == set(columns), (
        f"built {built}, labelled {labelled}, rendered {columns}")
    assert "resmom" in built, "the residual book is not wired in"


# --------------------------------------------------------------------------
# Candles: hammer rules and volume stats
# --------------------------------------------------------------------------

def _candle_bars(rows):
    idx = pd.bdate_range("2026-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def _with_history(last, drift):
    """20 ordinary 2-point-range sessions trending by `drift`, then `last`."""
    rows, c = [], 100.0
    for _ in range(20):
        o, c = c, c + drift
        rows.append([o, max(o, c) + 0.5, min(o, c) - 0.5, c])
    return _candle_bars(rows + [last(c)])


def test_hammer_after_a_decline_is_a_hammer():
    from qbs.candles import hammer_frame
    # Range 4, body 0.4 at the top, lower shadow 3.4, upper 0.2.
    f = hammer_frame(_with_history(lambda c: [c - 0.2, c + 0.2, c - 3.8, c], -0.5))
    last = f.iloc[-1]
    assert last["shape"] and last["hammer"] and not last["hanging_man"]


def test_same_shape_after_a_rise_is_a_hanging_man():
    from qbs.candles import hammer_frame
    f = hammer_frame(_with_history(lambda c: [c - 0.2, c + 0.2, c - 3.8, c], 0.5))
    last = f.iloc[-1]
    assert last["shape"] and last["hanging_man"] and not last["hammer"]


def test_long_upper_shadow_or_tiny_range_is_not_a_hammer():
    from qbs.candles import hammer_frame
    # Upper shadow as long as the lower one: a spinning top.
    top = hammer_frame(_with_history(lambda c: [c, c + 2.0, c - 2.0, c + 0.1], -0.5))
    assert not top.iloc[-1]["shape"]
    # Perfect geometry on a range a tenth of the usual: says nothing.
    tiny = hammer_frame(_with_history(lambda c: [c - 0.01, c + 0.01, c - 0.19, c], -0.5))
    assert tiny.iloc[-1]["lower"] > 0.8 and not tiny.iloc[-1]["shape"]


def test_volume_baseline_excludes_the_last_session():
    from qbs.candles import volume_stats
    idx = pd.bdate_range("2026-01-01", periods=21)
    v = pd.Series([100.0] * 20 + [300.0], index=idx)
    s = volume_stats(v, pd.Series(10.0, index=idx), window=20)
    assert s["last"] == 300 and s["avg"] == 100 and s["ratio"] == 3.0
    assert s["avg_value"] == 1000.0 and s["n"] == 20


# --------------------------------------------------------------------------
# Incremental cache updates
# --------------------------------------------------------------------------

def _truth(names=("AAA", "BBB"), n=30, end="2026-09-18"):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({t: 100.0 + i + np.arange(n) for i, t in enumerate(names)},
                        index=idx).rename_axis("Date")


def _fake_download(truth, calls):
    """A downloader that serves `truth` from `start`, and logs each call."""
    def download(names, start):
        calls.append((tuple(names), start))
        c = truth.loc[pd.Timestamp(start):, [t for t in names if t in truth.columns]]
        return c, c * 1000, [t for t in names if t not in truth.columns]
    return download


def test_an_agreeing_overlap_appends_the_new_rows():
    from qbs.incremental import merge_recent
    truth = _truth()
    cached = truth.iloc[:-2]
    merged, rebased, _ = merge_recent(cached, truth.iloc[-8:])
    assert rebased == []
    pd.testing.assert_frame_equal(merged, truth, check_freq=False)


def test_a_rebased_history_is_not_appended_to():
    """A dividend re-bases every earlier close; appending would fake a return."""
    from qbs.incremental import merge_recent
    truth = _truth()
    cached = truth.iloc[:-2]
    recent = truth.iloc[-8:].copy()
    recent.loc[:, "BBB"] *= 0.995          # yfinance re-adjusted BBB's history
    merged, rebased, _ = merge_recent(cached, recent)
    assert rebased == ["BBB"]
    assert merged["BBB"].dropna().index.max() == cached.index.max(), \
        "a re-based name must be left for a full download, not appended to"
    assert merged["AAA"].index.max() == truth.index.max()


def test_a_provisional_cell_is_not_evidence_and_is_replaced():
    from qbs.incremental import merge_recent
    truth = _truth()
    cached = truth.copy()
    last = truth.index[-1]
    cached.loc[last, "AAA"] = 999.0        # a raw screener print
    marks = {(last, "AAA")}
    merged, rebased, left = merge_recent(cached, truth.iloc[-6:], marks)
    assert rebased == [], "a raw print disagreeing is not a re-basing"
    assert merged.at[last, "AAA"] == truth.at[last, "AAA"]
    assert left == set(), "the mark goes once yfinance has the value"


def test_refresh_incremental_downloads_full_history_only_where_needed():
    from qbs.incremental import refresh_incremental, window_start
    truth = _truth(("AAA", "BBB", "NEW"))
    cached = truth[["AAA", "BBB"]].iloc[:-3].copy()
    cached["BBB"] *= 1.01                  # BBB was re-based since the cache
    calls = []
    closes, vols, marks, info = refresh_incremental(
        cached, None, set(), ["AAA", "BBB", "NEW"], "2000-01-01",
        _fake_download(truth, calls))
    since = f"{window_start(cached, ()):%Y-%m-%d}"
    assert calls[0] == (("AAA", "BBB"), since), "one recent window first"
    assert set(calls[1][0]) == {"BBB", "NEW"} and calls[1][1] == "2000-01-01"
    pd.testing.assert_frame_equal(closes[["AAA", "BBB", "NEW"]], truth,
                                  check_freq=False)
    assert info["rebased"] == ["BBB"] and info["recent"] == 1 and info["full"] == 2


def test_the_window_starts_before_the_oldest_provisional_cell():
    from qbs.incremental import window_start
    truth = _truth()
    old = truth.index[-4]
    assert window_start(truth, {(old, "AAA")}, overlap=5) < old
    assert window_start(truth, (), overlap=5) < truth.index[-1]


def test_persist_fill_never_overwrites_yfinance(tmp_path):
    from qbs.incremental import persist_fill, read_marks
    truth = _truth()
    path, vpath = str(tmp_path / "c.csv"), str(tmp_path / "v.csv")
    disk = truth.iloc[:-1].copy()
    session = truth.index[-1]
    disk.loc[session] = [55.0, np.nan]     # yfinance already has AAA
    disk.to_csv(path)
    (disk * 0).to_csv(vpath)
    n = persist_fill(path, session, pd.Series({"AAA": 1.0, "BBB": 2.0}),
                     vpath, pd.Series({"AAA": 10.0, "BBB": 20.0}))
    back = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    assert n == 1 and back.at[session, "AAA"] == 55.0 and back.at[session, "BBB"] == 2.0
    assert read_marks(path) == {(session, "BBB")}
    vback = pd.read_csv(vpath, parse_dates=["Date"], index_col="Date")
    assert vback.at[session, "BBB"] == 20.0


def test_universe_prices_update_incrementally_and_full_refresh_clears_marks(
        tmp_path, monkeypatch):
    import qbs.universe as U
    from qbs.incremental import read_marks, write_marks
    truth = _truth(("AAA", "BBB"), n=40)
    calls = []
    fake = _fake_download(truth, calls)
    monkeypatch.setattr(U, "_download_universe",
                        lambda ts, start, end, bs=40: fake(ts, start))
    start = f"{truth.index[0]:%Y-%m-%d}"
    d = str(tmp_path)
    U.load_universe_prices(["AAA", "BBB"], start, cache_dir=d, refresh=True,
                           verbose=False)
    assert calls[-1][1] == start
    # Pretend the cache is two sessions old with a provisional front cell.
    cache = os.path.join(d, "universe_prices.csv")
    stale = truth.iloc[:-2].copy()
    stale.loc[truth.index[-3], "AAA"] = 1.0
    stale.to_csv(cache)
    write_marks(cache, {(truth.index[-3], "AAA")})
    calls.clear()
    got = U.load_universe_prices(["AAA", "BBB"], start, cache_dir=d,
                                 incremental=True, verbose=False)
    assert len(calls) == 1 and calls[0][1] > start, "a window, not the history"
    pd.testing.assert_frame_equal(got, truth, check_freq=False)
    assert read_marks(cache) == set()
    write_marks(cache, {(truth.index[-1], "AAA")})
    U.load_universe_prices(["AAA", "BBB"], start, cache_dir=d, refresh=True,
                           verbose=False)
    assert read_marks(cache) == set(), "a full download is yfinance end to end"


def test_market_bars_update_incrementally(tmp_path, monkeypatch):
    import yfinance
    from qbs.finviz import load_universe_bars
    truth = _truth(("AAA", "BBB", "CCC"), n=40)
    calls = []

    def download(batch, start=None, **_):
        calls.append((tuple(batch), start))
        c = truth.loc[pd.Timestamp(start):, list(batch)]
        return pd.concat({"Close": c, "Volume": c * 1000}, axis=1)

    monkeypatch.setattr(yfinance, "download", download)
    start = f"{truth.index[0]:%Y-%m-%d}"
    d = str(tmp_path)
    truth[["AAA", "BBB"]].iloc[:-2].to_csv(os.path.join(d, "us_closes.csv"))
    (truth[["AAA", "BBB"]].iloc[:-2] * 1000).to_csv(os.path.join(d, "us_volumes.csv"))
    c, v, err = load_universe_bars(["AAA", "BBB", "CCC"], start=start,
                                   cache_dir=d, verbose=False, stale_after=1,
                                   incremental=True)
    assert calls[0][0] == ("AAA", "BBB") and calls[0][1] > start
    assert calls[1] == (("CCC",), start), "only the new name in full"
    pd.testing.assert_frame_equal(c[["AAA", "BBB", "CCC"]], truth,
                                  check_freq=False, check_names=False)


def test_a_session_closes_the_next_morning_in_asia():
    """The 09-24 bar is current on the morning of 09-25 in Hong Kong."""
    from qbs.data import session_close
    assert str(session_close("2026-09-24", "Asia/Hong_Kong")) == "2026-09-25 04:00:00+08:00"
    # New York is on standard time in January, so the close is an hour later.
    assert str(session_close("2026-01-15", "Asia/Hong_Kong")) == "2026-01-16 05:00:00+08:00"


def test_fill_note_names_a_handful_of_filled_tickers():
    from qbs.quotes import fill_note
    rep = {"session": pd.Timestamp("2026-09-24"), "filled": 3, "already": 2612,
           "absent": 0, "tickers": ["AAA", "BBB", "CCC"]}
    assert "(AAA, BBB, CCC)" in fill_note(rep, "finviz")
    rep.update(filled=40, tickers=[f"T{i}" for i in range(40)])
    assert "T0" not in fill_note(rep, "finviz")


# --------------------------------------------------------------------------
# Thin sessions in the middle of a history
# --------------------------------------------------------------------------

def _wide_days(n_days=120, n_names=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-24", periods=n_days)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (n_days, n_names)), axis=0)),
                        index=idx, columns=[f"T{i}" for i in range(n_names)])


def test_thin_rows_finds_a_gap_in_the_middle():
    from qbs.data import thin_rows
    px = _wide_days()
    px.iloc[-3, 50:] = np.nan
    gaps = thin_rows(px)
    assert list(gaps) == [px.index[-3]] and gaps[px.index[-3]][0] == 50


def test_breadth_skips_a_gap_instead_of_reading_the_survivors():
    """A session most names lack must not shrink weeks of rows to its survivors."""
    from qbs.breadth import daily_breadth
    px = _wide_days()
    whole = daily_breadth(px).table
    holed = px.copy()
    holed.iloc[-3, 50:] = np.nan
    res = daily_breadth(holed)
    t = res.table
    assert px.index[-3] in res.gaps and px.index[-3] not in t.index
    # The row after the gap would be a two-session move: no 4% counts.
    assert np.isnan(t.loc[px.index[-2], "up4"])
    # The averages still run over every name, not the 50 that had the gap day.
    assert abs(t.loc[px.index[-1], "pct_above_fast"]
               - whole.loc[px.index[-1], "pct_above_fast"]) < 3
    assert t.loc[px.index[-1], "up4"] == whole.loc[px.index[-1], "up4"]


def test_qqq_atr_uses_real_high_low_when_given():
    from qbs.breadth import daily_breadth
    px = _wide_days(n_names=40)
    q = px.mean(axis=1)
    ohlc = pd.DataFrame({"Open": q, "High": q * 1.02, "Low": q * 0.98, "Close": q})
    close_only = daily_breadth(px, qqq=q).table["qqq_atr"].abs().mean()
    real = daily_breadth(px, qqq=q, qqq_ohlc=ohlc).table["qqq_atr"].abs().mean()
    assert real < close_only, "a real range is wider, so the distance is smaller"


def test_the_update_window_reaches_back_to_a_recent_gap():
    from qbs.data import thin_rows
    from qbs.incremental import window_start
    px = _wide_days(n_names=100)
    gap = px.index[-10]
    px.loc[gap, px.columns[20:]] = np.nan
    assert gap in thin_rows(px)
    assert window_start(px, ()) < gap


# --------------------------------------------------------------------------
# Bear-market checklist
# --------------------------------------------------------------------------

def _checklist_px(n_days=400, n_names=200, drift=0.0, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-24", periods=n_days)
    steps = rng.normal(drift, 0.02, (n_days, n_names))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx,
                        columns=[f"T{i}" for i in range(n_names)])


def _answers(px, index):
    from qbs.breadth import bear_checklist
    return {r["key"]: r["answer"] for r in bear_checklist(px, index)}


def test_a_long_slide_trips_the_oversold_question():
    px = _checklist_px()
    px.iloc[-30:] = px.iloc[-31].to_numpy() * np.exp(
        np.cumsum(np.full((30, px.shape[1]), -0.01), axis=0))
    assert _answers(px, px.mean(axis=1))["oversold"] is True


def test_a_steady_rise_answers_no_to_the_bearish_questions():
    px = _checklist_px(drift=0.002)
    a = _answers(px, px.mean(axis=1))
    assert a["oversold"] is False and a["divergence"] is False and a["mli"] is False


def test_final_high_needs_the_index_at_a_new_high_and_weak_leaders():
    px = _checklist_px()
    idx = pd.Series(np.linspace(100, 200, len(px)), index=px.index)   # new high today
    # Leaders (up >10% on the quarter) that have since dropped under their 50-day.
    # +60% into the high, then back to +35%: still a leader on the quarter,
    # but under its own 50-day average.
    px.iloc[-63:, :50] = px.iloc[-64, :50].to_numpy() * np.r_[
        np.linspace(1.0, 1.6, 55), np.linspace(1.6, 1.35, 8)][:, None]
    # Everyone else flat, so no random name joins the leaders.
    px.iloc[-63:, 50:] = px.iloc[-64, 50:].to_numpy()
    from qbs.breadth import bear_checklist
    row = {r["key"]: r for r in bear_checklist(px, idx)}["final_high"]
    assert "252-day high" in row["reading"]
    assert row["answer"] is True, row["reading"]
    flat = pd.Series(np.r_[np.linspace(100, 200, len(px) - 10), np.full(10, 150.0)],
                     index=px.index)
    assert _answers(px, flat)["final_high"] is False, "no new high, no pattern"


def test_short_history_answers_none_rather_than_no():
    px = _checklist_px(n_days=30)
    a = _answers(px, px.mean(axis=1))
    assert a["oversold"] is None and a["final_high"] is None



def test_market_bars_report_progress_per_batch(tmp_path, monkeypatch):
    import yfinance
    from qbs.finviz import load_universe_bars
    truth = _truth(tuple(f"T{i}" for i in range(5)), n=40)

    def download(batch, start=None, **_):
        c = truth.loc[pd.Timestamp(start):, list(batch)]
        return pd.concat({"Close": c, "Volume": c * 1000}, axis=1)

    monkeypatch.setattr(yfinance, "download", download)
    seen = []
    load_universe_bars(list(truth.columns), start=f"{truth.index[0]:%Y-%m-%d}",
                       cache_dir=str(tmp_path), refresh=True, verbose=False,
                       batch_size=2, progress=lambda d, t, w: seen.append((d, t, w)))
    assert seen == [(2, 5, "full history"), (4, 5, "full history"),
                    (5, 5, "full history")]

    # A callback that raises must not cost the download.
    def boom(*_):
        raise RuntimeError("ui gone")
    c, _, _ = load_universe_bars(list(truth.columns), start=f"{truth.index[0]:%Y-%m-%d}",
                                 cache_dir=str(tmp_path), refresh=True,
                                 verbose=False, batch_size=2, progress=boom)
    assert c is not None and c.shape[1] == 5
