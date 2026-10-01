"""The per-name exit (`MomentumParams.exit_drop`)."""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import MomentumParams  # noqa: E402
from qbs.data import synthetic_prices  # noqa: E402
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
