"""Fibonacci levels of the last up-leg (`qbs.swing.fib_retracement`)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.swing import fib_retracement  # noqa: E402


def _bars(path):
    """Daily bars along `path`, each bar's high/low 0.5 either side."""
    c = pd.Series(np.asarray(path, dtype=float),
                  index=pd.bdate_range("2026-01-02", periods=len(path)))
    return pd.DataFrame({"Open": c, "High": c + 0.5, "Low": c - 0.5,
                         "Close": c, "Volume": 1e6})


def _leg(*points, step=15):
    """Straight-line segments through `points`, `step` bars each."""
    out = []
    for a, b in zip(points, points[1:]):
        out.extend(np.linspace(a, b, step, endpoint=False))
    out.append(points[-1])
    return out


def test_levels_span_the_higher_low_to_the_swing_high():
    # low 100 -> high 130 -> higher low 110 -> high 150 -> pull back to 140
    f = fib_retracement(_bars(_leg(120, 100, 130, 110, 150, 140)))
    assert f is not None
    assert f["low"] == pytest.approx(109.5)          # the 110 swing's Low
    assert f["high"] == pytest.approx(150.5)
    assert f["higher_low"] and f["prev_low"] == pytest.approx(99.5)
    assert not f["leg_running"]
    span = f["high"] - f["low"]
    lv = {x["label"]: x["price"] for x in f["levels"]}
    assert lv["61.8%"] == pytest.approx(f["high"] - 0.618 * span)
    assert lv["161.8% ext"] == pytest.approx(f["low"] + 1.618 * span)
    assert f["retraced"] == pytest.approx((f["high"] - 140) / span)
    # Support: the nearest retracement under the close (140 sits between the
    # 23.6% and 38.2% levels, so 38.2%). Target: the leg's high, not the
    # 23.6% retracement just above the close.
    assert f["support"]["label"] == "38.2%"
    assert f["target"]["label"] == "High (0%)"
    kinds = [x["kind"] for x in f["levels"]]
    assert kinds[0] == "extension" and kinds[-1] == "low"


def test_a_lower_low_is_flagged():
    # 100 -> 130 -> LOWER low 90 -> high 125 -> 115
    f = fib_retracement(_bars(_leg(115, 100, 130, 90, 125, 115)))
    assert f is not None and not f["higher_low"]


def test_new_high_after_the_swing_extends_the_leg():
    # swing high 130, pullback to 120, then a fresh run to 160 (still rising)
    f = fib_retracement(_bars(_leg(110, 100, 130, 120, 160)))
    assert f is not None
    assert f["leg_running"] and f["high"] == pytest.approx(160.5)
    assert f["retraced"] == pytest.approx(0.5 / (f["high"] - f["low"]), abs=1e-6)


def test_past_the_high_the_target_is_an_extension():
    f = fib_retracement(_bars(_leg(110, 100, 130, 120, 160)))
    assert f["target"]["kind"] == "extension"


def test_no_leg_without_swings():
    assert fib_retracement(_bars(np.linspace(100, 200, 120))) is None
    assert fib_retracement(_bars([100] * 10)) is None
