"""The auto-selecting swing book (`qbs.swing_book`)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.swing_book import (SwingBookParams, commission,  # noqa: E402
                            run_swing_book, trade_table)

DAYS = pd.bdate_range("2026-01-05", periods=30)


def _frame(**paths):
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in paths.items()},
                        index=DAYS)


def _scan(rows):
    """One setup's scan result: {ticker: (close, stop, target, rr)}."""
    return pd.DataFrame([dict(ticker=t, close=c, stop=s, target=g, rr=r)
                         for t, (c, s, g, r) in rows.items()]).set_index("ticker")


FLAT_SAFE = pd.Series(100.0, index=DAYS)
NO_COST = dict(commission_min=0.0, commission_per_share=0.0, slippage_bps=0.0)


def test_commission_is_ib_fixed():
    p = SwingBookParams()
    assert commission(10, 100.0, p) == 1.0              # the $1 minimum
    assert commission(1000, 50.0, p) == 5.0             # $0.005 a share
    assert commission(10, 1.0, p) == pytest.approx(0.1)  # capped at 1% of value


def test_target_exit_and_whole_shares():
    up = [100.0] * 3 + [105, 110, 116] + [116] * 24
    px = _frame(AAA=up)
    scans = {DAYS[2]: {"trend_pullback": _scan({"AAA": (100.0, 95.0, 115.0, 3.0)})}}
    p = SwingBookParams(n_slots=1, slot_usd=1000.0, **NO_COST)
    eq, trades = run_swing_book(px, FLAT_SAFE, scans, p)
    t = trade_table(trades).iloc[0]
    assert t["entry"] == DAYS[2] and t["shares"] == 10       # floor(1000 / 100)
    assert t["reason"] == "target" and t["exit"] == DAYS[5]  # first close >= 115
    assert eq.iloc[-1] == pytest.approx(1000.0 + 10 * 16.0)


def test_stop_exit_and_no_same_day_reentry():
    down = [100.0] * 3 + [97, 94, 93] + [93] * 24
    px = _frame(AAA=down)
    sc = {"trend_pullback": _scan({"AAA": (100.0, 95.0, 115.0, 3.0)})}
    scans = {d: sc for d in DAYS[2:]}         # the screen keeps listing it
    p = SwingBookParams(n_slots=1, **NO_COST)
    _, trades = run_swing_book(px, FLAT_SAFE, scans, p)
    tt = trade_table(trades)
    first = tt.iloc[0]
    assert first["reason"] == "stop" and first["exit"] == DAYS[4]
    # Re-entered the next session, not on the stop-out close.
    assert tt.iloc[1]["entry"] == DAYS[5]


def test_time_stop_uses_the_setups_max_hold():
    px = _frame(AAA=[100.0] * 30)
    scans = {DAYS[0]: {"mean_reversion": _scan({"AAA": (100.0, 90.0, 120.0, 2.0)})}}
    p = SwingBookParams(n_slots=1, **NO_COST)
    _, trades = run_swing_book(px, FLAT_SAFE, scans, p)
    t = trade_table(trades).iloc[0]
    assert t["reason"] == "time" and t["held"] == 7       # mean reversion: 7 days


def test_slots_fill_best_reward_risk_first_and_skip_unaffordable():
    px = _frame(AAA=[100.0] * 30, BBB=[50.0] * 30, CCC=[2000.0] * 30)
    scans = {DAYS[0]: {"trend_pullback": _scan({
        "AAA": (100.0, 95.0, 110.0, 2.0), "BBB": (50.0, 45.0, 80.0, 6.0),
        "CCC": (2000.0, 1900.0, 2500.0, 5.0)})}}
    p = SwingBookParams(n_slots=2, **NO_COST)
    _, trades = run_swing_book(px, FLAT_SAFE, scans, p)
    held = set(trade_table(trades)["ticker"])
    # CCC ranks second but one share costs more than a $1,000 slot.
    assert held == {"BBB", "AAA"}


def test_idle_cash_earns_the_safe_asset():
    px = _frame(AAA=[100.0] * 30)
    safe = pd.Series(np.linspace(100, 103, 30), index=DAYS)
    eq, _ = run_swing_book(px, safe, {}, SwingBookParams(n_slots=2, **NO_COST))
    assert eq.iloc[-1] == pytest.approx(2000.0 * 1.03)


def test_intraday_stop_and_target_with_ohlc():
    px = _frame(AAA=[100.0] * 30, BBB=[100.0] * 30, CCC=[100.0] * 30)
    bars = {t: pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                             "Close": 100.0}, index=DAYS) for t in px}
    bars["AAA"].loc[DAYS[3], ["Low"]] = 94.0          # trades through the stop
    bars["BBB"].loc[DAYS[3], ["Open", "Low"]] = 90.0  # gaps below it
    bars["CCC"].loc[DAYS[3], ["High"]] = 116.0        # reaches the target
    sc = {"trend_pullback": _scan({t: (100.0, 95.0, 115.0, 3.0) for t in px})}
    p = SwingBookParams(n_slots=3, **NO_COST)
    _, trades = run_swing_book(px, FLAT_SAFE, {DAYS[0]: sc}, p, ohlc=bars)
    tt = trade_table(trades).set_index("ticker")
    assert tt.loc["AAA", "reason"] == "stop" and tt.loc["AAA", "exit_px"] == 95.0
    assert tt.loc["BBB", "reason"] == "stop" and tt.loc["BBB", "exit_px"] == 90.0
    assert tt.loc["CCC", "reason"] == "target" and tt.loc["CCC", "exit_px"] == 115.0
    # Without bars, the same closes never touch either level.
    _, plain = run_swing_book(px, FLAT_SAFE, {DAYS[0]: sc}, p)
    assert set(trade_table(plain)["reason"]) == {"time"}


def test_momentum_first_picks_rank_by_momentum_and_widen_stops():
    from qbs.swing_book import momentum_first_picks
    idx = pd.bdate_range("2025-01-01", periods=200)
    # AAA has the strongest 6-1 momentum, CCC the weakest.
    closes = pd.DataFrame({
        "AAA": np.linspace(50, 100, 200),
        "BBB": np.linspace(80, 100, 200),
        "CCC": np.linspace(99, 100, 200),
    }, index=idx)
    last = closes.iloc[-1]
    scan_day = {
        "trend_pullback": _scan({"CCC": (last["CCC"], 95.0, 110.0, 2.0),
                                 "BBB": (last["BBB"], 96.0, 104.0, 1.0)}),
        "mean_reversion": _scan({"AAA": (last["AAA"], 98.0, 103.0, 1.5)}),
    }
    picks = momentum_first_picks(scan_day, closes, n_slots=2, stop_mult=2.0)
    assert list(picks["ticker"]) == ["AAA", "BBB"]          # CCC ranks last
    a = picks.iloc[0]
    assert a["setup"] == "mean_reversion" and a["max_hold"] == 7
    assert a["stop"] == pytest.approx(100.0 - 2 * (100.0 - 98.0))   # 96
    assert a["rr"] == pytest.approx((103.0 - 100.0) / (100.0 - 96.0))
    assert a["shares"] == 10
    # Names one share of which costs more than the slot are skipped.
    scan_day["mean_reversion"] = _scan({"AAA": (200.0, 196.0, 206.0, 1.5)})
    picks = momentum_first_picks(scan_day, closes, n_slots=2, slot_usd=150.0)
    assert list(picks["ticker"]) == ["BBB", "CCC"]
    assert "AAA" not in set(picks["ticker"])
