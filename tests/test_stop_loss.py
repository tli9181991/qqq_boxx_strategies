"""Per-position stops inside the Top-N books (`StopLossParams`)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import MomentumParams, ResidualMomentumParams, StopLossParams  # noqa: E402
from qbs.data import synthetic_prices  # noqa: E402
from qbs.strategies import (  # noqa: E402
    cross_sectional_momentum, residual_momentum, stop_inputs,
)
from qbs.universe import synthetic_universe  # noqa: E402

KINDS = [
    StopLossParams(kind="fixed", stop_pct=0.15),
    StopLossParams(kind="trailing", stop_pct=0.15),
    StopLossParams(kind="chandelier", atr_mult=3.0),
    StopLossParams(kind="residual", resid_mult=2.0),
]


def _fixture():
    uni = synthetic_universe(n=40)
    safe = synthetic_prices(start="2023-06-01")["BOXX"].reindex(uni.index).ffill().bfill()
    mkt = uni.mean(axis=1)
    return uni, safe, mkt


def _held(sig, dt):
    return set(sig.holdings_log[dt])


def test_stop_off_is_bit_identical():
    uni, safe, mkt = _fixture()
    p = MomentumParams(n_hold=6, exit_rank=10)
    base = cross_sectional_momentum(uni, safe, p)
    off = cross_sectional_momentum(uni, safe, p, stop=StopLossParams(), market=mkt)
    pd.testing.assert_frame_equal(base.weights, off.weights)

    rp = ResidualMomentumParams(beta_window=126)
    rbase = residual_momentum(uni, safe, mkt, rp)
    roff = residual_momentum(uni, safe, mkt, rp, stop=StopLossParams())
    pd.testing.assert_frame_equal(rbase.weights, roff.weights)
    assert rbase.params == roff.params


def _crash_setup():
    """A held name that loses 30% overnight, late in the sample."""
    uni, safe, mkt = _fixture()
    p = MomentumParams(n_hold=6, exit_rank=10)
    base = cross_sectional_momentum(uni, safe, p)
    i = len(uni) - 60
    name = sorted(_held(base, uni.index[i]))[0]
    crashed = uni.copy()
    crashed.iloc[i + 1:, crashed.columns.get_loc(name)] *= 0.70
    return crashed, safe, mkt, p, i, name


def test_fixed_stop_sells_the_crash_and_bars_rebuy():
    uni, safe, mkt, p, i, name = _crash_setup()
    # 6-1 momentum skips the last month, so without a stop the crash is
    # invisible to the ranker and the name is still held the next day.
    plain = cross_sectional_momentum(uni, safe, p)
    assert name in _held(plain, uni.index[i + 1])

    stop = StopLossParams(kind="fixed", stop_pct=0.15, cooldown_days=21)
    sig = cross_sectional_momentum(uni, safe, p, stop=stop)
    assert name not in _held(sig, uni.index[i + 1]), "stop did not fire on the close"
    for k in range(i + 1, i + 1 + 21):
        assert name not in _held(sig, uni.index[k]), f"re-bought inside cooldown at {k}"
    ev = sig.events
    stops = ev[ev["reason"].astype(str).str.startswith("stop:")]
    assert ((stops["asset"] == name) & (stops["date"] == uni.index[i + 1])).any()


def test_refill_false_leaves_the_slot_in_cash():
    uni, safe, mkt, p, i, name = _crash_setup()
    dt = uni.index[i + 1]
    on = cross_sectional_momentum(uni, safe, p, stop=StopLossParams(
        kind="fixed", stop_pct=0.15, refill=True))
    off = cross_sectional_momentum(uni, safe, p, stop=StopLossParams(
        kind="fixed", stop_pct=0.15, refill=False))
    assert len(_held(on, dt)) == len(_held(off, dt)) + 1
    assert off.weights.at[dt, "BOXX"] == pytest.approx(on.weights.at[dt, "BOXX"] + 1 / 6)


@pytest.mark.parametrize("stop", KINDS, ids=lambda s: s.kind)
def test_stop_has_no_lookahead(stop):
    uni, safe, mkt = _fixture()
    p = MomentumParams(n_hold=6, exit_rank=10)
    a = cross_sectional_momentum(uni, safe, p, stop=stop, market=mkt)
    cut = len(uni) - 80
    tampered = uni.copy()
    tampered.iloc[cut + 1:] *= np.random.default_rng(0).uniform(
        0.5, 1.5, size=tampered.iloc[cut + 1:].shape)
    b = cross_sectional_momentum(tampered, safe, p, stop=stop, market=mkt)
    pd.testing.assert_frame_equal(a.weights.iloc[:cut + 1], b.weights.iloc[:cut + 1])


def test_residual_stop_ignores_a_fall_the_market_explains():
    idx = pd.bdate_range("2024-01-01", periods=400)
    rng = np.random.default_rng(1)
    rm = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    mkt = 100 * (1 + rm).cumprod()
    beta_name = 100 * (1 + 1.5 * rm).cumprod()          # pure beta, no residual
    px = pd.DataFrame({"B": beta_name})
    level, dist = stop_inputs(px, StopLossParams(kind="residual"), mkt)
    tail = level["B"].iloc[300:]
    assert (tail / tail.cummax() - 1).min() > -1e-6, "a pure-beta name has no residual fall"


def test_chandelier_distance_scales_with_volatility():
    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(2)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, len(idx))))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx))))
    px = pd.DataFrame({"calm": calm, "wild": wild}, index=idx)
    _, dist = stop_inputs(px, StopLossParams(kind="chandelier", atr_mult=3.0))
    assert dist["wild"].iloc[-50:].mean() > 3 * dist["calm"].iloc[-50:].mean()


def test_stop_params_validate():
    for bad in (dict(kind="nope"), dict(stop_pct=0.0), dict(atr_mult=-1),
                dict(cooldown_days=-1)):
        with pytest.raises(ValueError):
            StopLossParams(**bad)


def test_watchlist_stop_levels():
    from qbs.shadow import watchlist_stop_levels
    idx = pd.bdate_range("2024-01-01", periods=100)
    a = pd.Series(np.linspace(50, 100, 100), index=idx)       # at its high
    b = a.copy()
    b.iloc[-1] = 75.0                                        # 25% off a 99.5 high
    lv = watchlist_stop_levels(pd.DataFrame({"A": a, "B": b, "C": np.nan}))
    assert "C" not in lv
    assert lv["A"]["stops"][0.13] == pytest.approx(100 * 0.87)
    assert lv["A"]["room"][0.20] == pytest.approx(1 / 0.80 - 1)
    assert lv["B"]["room"][0.20] < 0, "25% off the high is past the 20% stop"
    # `asof` cuts the history: nothing after it may set the high.
    cut = watchlist_stop_levels(pd.DataFrame({"A": a}), asof=idx[49])
    assert cut["A"]["high"] == pytest.approx(a.iloc[49])
