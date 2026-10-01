"""The per-name exit (`MomentumParams.exit_drop`) and its paper sleeve."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import Config, MomentumParams  # noqa: E402
from qbs.data import synthetic_prices  # noqa: E402
from qbs.live import state as st  # noqa: E402
from qbs.live.config import LiveConfig  # noqa: E402
from qbs.live.signals import compute_targets  # noqa: E402
from qbs.strategies import cross_sectional_momentum  # noqa: E402
from qbs.universe import synthetic_universe  # noqa: E402


def _fixture():
    uni = synthetic_universe(n=40)
    safe = synthetic_prices(start="2023-06-01")["BOXX"].reindex(uni.index).ffill().bfill()
    return uni, safe


def test_exit_drop_off_is_the_fixed_band():
    uni, safe = _fixture()
    p = MomentumParams(n_hold=6, exit_rank=8)
    a = cross_sectional_momentum(uni, safe, p)
    b = cross_sectional_momentum(uni, safe, replace(p, exit_drop=None))
    pd.testing.assert_frame_equal(a.weights, b.weights)


def test_exit_drop_sells_on_falling_past_entry_rank_plus_drop():
    uni, safe = _fixture()
    p = MomentumParams(n_hold=6, exit_rank=8, exit_drop=4)
    sig = cross_sectional_momentum(uni, safe, p)
    ev = sig.events
    buys = ev[ev["action"] == "buy"].set_index(["asset", "date"])["rank"]
    sells = ev[(ev["action"] == "sell") & ev["reason"].str.startswith("rank ")]
    assert len(sells) > 0
    for _, s in sells.iterrows():
        # The most recent buy of that name before this sale sets its line.
        b = ev[(ev["action"] == "buy") & (ev["asset"] == s["asset"])
               & (ev["date"] < s["date"])].iloc[-1]
        line = b["rank"] + 4
        assert s["rank"] > line, (s["asset"], s["rank"], line)
        assert s["reason"] == f"rank {s['rank']:.0f} > {line:.0f}"
    assert len(buys) > 0


def test_exit_drop_must_not_be_negative():
    with pytest.raises(ValueError):
        MomentumParams(exit_drop=-1)


def _live_frame():
    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    frame = uni.copy()
    frame["BOXX"] = px["BOXX"]
    frame["QQQ"] = px["QQQ"]
    return cfg, uni, frame


def test_paper_sleeve_is_logged_but_never_traded():
    cfg, uni, frame = _live_frame()
    off = compute_targets(cfg, frame, requested=list(uni.columns), now=frame.index[-1])
    on = compute_targets(cfg, frame, requested=list(uni.columns), now=frame.index[-1],
                         compare_exit_drop=8)
    assert "momentum6_exitdrop" not in off.strategy_daily_returns
    assert set(on.strategy_daily_returns) == {"momentum", "momentum6",
                                              "momentum6_exitdrop"}
    # Observation only: the targets the broker sees are identical.
    assert on.weights == off.weights
    assert "momentum6_exitdrop" not in on.strategy_weights
    assert cfg.momentum.exit_drop is None, "the live config must not be mutated"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "comparison.csv")
        assert st.upsert_strategy_comparison_csv(path, f"{on.asof:%Y-%m-%d}", on) == 3
        rows = st.strategy_comparison(path)
        assert {r["strategy"] for r in rows} == {"momentum", "momentum6",
                                                 "momentum6_exitdrop"}
        assert all(np.isfinite(r["return"]) for r in rows)


def test_compare_exit_drop_from_env(monkeypatch):
    assert LiveConfig().compare_exit_drop == 8
    monkeypatch.setenv("QBS_COMPARE_EXIT_DROP", "0")
    assert LiveConfig.from_env().compare_exit_drop == 0
    monkeypatch.setenv("QBS_COMPARE_EXIT_DROP", "-1")
    with pytest.raises(ValueError):
        LiveConfig.from_env()
