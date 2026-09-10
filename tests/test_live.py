"""Tests for the live deployment layer.

Everything here runs without IB Gateway, without network, and without a clock.
That is the point of keeping `orders.py` and `signals.py` pure: the code that
decides what to send to a broker unattended is exactly the code you most need
to be able to test on a laptop.

The broker wrapper is covered only where it has real logic of its own -- the
live-port refusal, filtering non-stock positions, the dry-run path -- against a
stub standing in for ib_async's IB. The parts that are a straight translation
into library calls are not mocked, because such a test asserts nothing beyond
"the mock was called".
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import Config
from qbs.data import synthetic_prices
from qbs.live.config import LiveConfig
from qbs.live.orders import (
    GuardTripped, Order, apply_guards, build_orders, diff_positions, target_shares,
)
from qbs.live.signals import SignalError, check_data_quality, compute_targets
from qbs.live import state as st
from qbs.universe import synthetic_universe


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def test_target_shares_rounds_down():
    """Rounding down keeps the book from creeping above the configured notional."""
    got = target_shares({"AAA": 0.5}, {"AAA": 300.0}, notional=1000.0)
    assert got == {"AAA": 1}, "0.5 * 1000 / 300 = 1.67 -> 1"


def test_target_shares_never_exceeds_notional():
    weights = {f"S{i}": 1 / 6 for i in range(6)}
    prices = {f"S{i}": 37.5 + i for i in range(6)}
    shares = target_shares(weights, prices, notional=100_000.0)
    spend = sum(shares[s] * prices[s] for s in shares)
    assert spend <= 100_000.0


def test_target_shares_rejects_a_missing_price():
    with pytest.raises(ValueError, match="no usable price"):
        target_shares({"AAA": 1.0}, {}, notional=1000.0)
    with pytest.raises(ValueError, match="no usable price"):
        target_shares({"AAA": 1.0}, {"AAA": 0.0}, notional=1000.0)


# --------------------------------------------------------------------------
# Diffing
# --------------------------------------------------------------------------

def test_diff_emits_nothing_when_already_on_target():
    assert diff_positions({"AAA": 10}, {"AAA": 10}, {"AAA": 50.0}) == []


def test_diff_closes_a_dropped_name_in_full():
    orders = diff_positions({"AAA": 0}, {"AAA": 7}, {"AAA": 50.0})
    assert len(orders) == 1
    o = orders[0]
    assert (o.action, o.symbol, o.quantity, o.reason) == ("SELL", "AAA", 7, "exit")


def test_diff_sells_before_buys():
    """Cash from an exit must be available for the entry that replaces it."""
    orders = diff_positions({"BBB": 10, "AAA": 0}, {"AAA": 10},
                            {"AAA": 50.0, "BBB": 50.0})
    assert [o.action for o in orders] == ["SELL", "BUY"]


def test_diff_skips_dust_but_never_an_exit():
    prices = {"AAA": 100.0, "BBB": 100.0}
    orders = diff_positions({"AAA": 101, "BBB": 0}, {"AAA": 100, "BBB": 3},
                            prices, min_notional=500.0)
    syms = {o.symbol for o in orders}
    assert "AAA" not in syms, "a 1-share/$100 rebalance is below the dust floor"
    assert "BBB" in syms, "an exit must not be skipped as dust"


def test_diff_can_exit_without_a_price():
    """A position we cannot price must still be closable."""
    orders = diff_positions({"AAA": 0}, {"AAA": 5}, {})
    assert len(orders) == 1 and orders[0].action == "SELL"


def test_diff_raises_when_buying_without_a_price():
    with pytest.raises(ValueError, match="no usable price"):
        diff_positions({"AAA": 5}, {}, {})


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def _orders(*specs):
    return [Order(sym, act, qty, px) for sym, act, qty, px in specs]


def test_guard_blocks_an_oversized_single_order():
    orders = _orders(("AAA", "BUY", 1000, 100.0))     # $100k
    with pytest.raises(GuardTripped, match="max_order_notional"):
        apply_guards(orders, notional=100_000, target={"AAA": 1000},
                     max_order_notional=40_000, max_gross_turnover=2.0,
                     max_positions=12)


def test_guard_blocks_a_runaway_turnover_session():
    orders = _orders(("AAA", "BUY", 100, 100.0), ("BBB", "BUY", 100, 100.0),
                     ("CCC", "BUY", 100, 100.0))     # $30k on a $10k book
    with pytest.raises(GuardTripped, match="above the"):
        apply_guards(orders, notional=10_000, target={"AAA": 100},
                     max_order_notional=50_000, max_gross_turnover=1.6,
                     max_positions=12)


def test_guard_blocks_too_many_positions():
    target = {f"S{i}": 10 for i in range(20)}
    with pytest.raises(GuardTripped, match="above the 12 limit"):
        apply_guards(_orders(("S0", "BUY", 10, 1.0)), notional=100_000, target=target,
                     max_order_notional=50_000, max_gross_turnover=2.0,
                     max_positions=12)


def test_guard_exempts_the_cash_leg_from_the_per_name_cap():
    """A de-risked book puts most of the notional in BOXX; that is not a runaway order.

    Regression test for a real bug: with the vol scalar at 0.36 the cash leg is
    64% of a $100k book, which tripped a $40k per-order cap on an entirely
    normal session.
    """
    orders = _orders(("BOXX", "BUY", 638, 100.0))     # $63.8k of the cash leg
    out = apply_guards(orders, notional=100_000, target={"BOXX": 638},
                       max_order_notional=40_000, max_gross_turnover=1.6,
                       max_positions=12, safe_asset="BOXX")
    assert out == orders


def test_guard_still_blocks_a_cash_leg_bigger_than_the_book():
    orders = _orders(("BOXX", "BUY", 2000, 100.0))    # $200k on a $100k book
    with pytest.raises(GuardTripped, match="cannot be larger than"):
        apply_guards(orders, notional=100_000, target={"BOXX": 2000},
                     max_order_notional=40_000, max_gross_turnover=9.0,
                     max_positions=12, safe_asset="BOXX")


def test_guard_still_blocks_an_oversized_risk_name():
    """The exemption must be for the safe asset only, not a blanket loosening."""
    orders = _orders(("NVDA", "BUY", 1000, 100.0))
    with pytest.raises(GuardTripped, match="max_order_notional"):
        apply_guards(orders, notional=100_000, target={"NVDA": 1000},
                     max_order_notional=40_000, max_gross_turnover=9.0,
                     max_positions=12, safe_asset="BOXX")


def test_guard_passes_a_normal_session_unchanged():
    orders = _orders(("AAA", "SELL", 50, 100.0), ("BBB", "BUY", 40, 100.0))
    out = apply_guards(orders, notional=100_000, target={"BBB": 40},
                       max_order_notional=40_000, max_gross_turnover=1.6,
                       max_positions=12)
    assert out == orders


def test_guards_are_all_or_nothing():
    """One bad order must block the whole list, not be dropped from it."""
    orders = _orders(("AAA", "BUY", 10, 100.0), ("BBB", "BUY", 1000, 100.0))
    with pytest.raises(GuardTripped):
        apply_guards(orders, notional=100_000, target={"AAA": 10, "BBB": 1000},
                     max_order_notional=40_000, max_gross_turnover=5.0,
                     max_positions=12)


# --------------------------------------------------------------------------
# build_orders end to end
# --------------------------------------------------------------------------

def test_build_orders_leaves_foreign_positions_alone():
    """A position this strategy never opened must not be liquidated by it."""
    weights = {"AAA": 0.5, "BOXX": 0.5}
    prices = {"AAA": 100.0, "BOXX": 100.0}
    actual = {"AAA": 100, "TSLA": 999}      # TSLA is someone else's
    orders, target = build_orders(
        weights, prices, actual, notional=100_000,
        max_order_notional=1e9, max_gross_turnover=9.0, max_positions=12)
    assert "TSLA" not in {o.symbol for o in orders}
    assert "TSLA" not in target


def test_build_orders_can_hold_cash_instead_of_the_safe_asset():
    weights = {"AAA": 0.4, "BOXX": 0.6}
    prices = {"AAA": 100.0, "BOXX": 100.0}
    orders, target = build_orders(
        weights, prices, {"BOXX": 600}, notional=100_000,
        max_order_notional=1e9, max_gross_turnover=9.0, max_positions=12,
        hold_safe_asset=False, safe_asset="BOXX")
    assert target["BOXX"] == 0
    sells = [o for o in orders if o.symbol == "BOXX"]
    assert len(sells) == 1 and sells[0].action == "SELL" and sells[0].quantity == 600


def test_build_orders_from_flat_buys_the_whole_book():
    weights = {f"S{i}": 1 / 6 * 0.58 for i in range(6)}
    weights["BOXX"] = 1 - sum(weights.values())
    prices = {**{f"S{i}": 100.0 for i in range(6)}, "BOXX": 100.0}
    orders, target = build_orders(
        weights, prices, {}, notional=100_000,
        max_order_notional=1e9, max_gross_turnover=9.0, max_positions=12)
    assert all(o.action == "BUY" for o in orders)
    spend = sum(o.notional for o in orders)
    assert spend <= 100_000 * 1.001


# --------------------------------------------------------------------------
# Data-quality gate
# --------------------------------------------------------------------------

def _good_frame(n_days=400, n_tickers=30, end="2026-09-09"):
    idx = pd.bdate_range(end=end, periods=n_days)
    cols = [f"S{i}" for i in range(n_tickers)] + ["BOXX"]
    rng = np.random.default_rng(3)
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(cols))), axis=0))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_quality_gate_rejects_stale_prices():
    px = _good_frame(end="2026-08-01")
    with pytest.raises(SignalError, match="stale"):
        check_data_quality(px, [c for c in px.columns if c != "BOXX"], "BOXX",
                           max_staleness_days=5, min_coverage=0.85,
                           now=pd.Timestamp("2026-09-09"))


def test_quality_gate_rejects_a_half_downloaded_universe():
    px = _good_frame(n_tickers=10)
    requested = [f"S{i}" for i in range(100)]
    with pytest.raises(SignalError, match="below the"):
        check_data_quality(px, requested, "BOXX", max_staleness_days=5,
                           min_coverage=0.85, now=pd.Timestamp("2026-09-09"))


def test_quality_gate_rejects_a_partial_last_bar():
    px = _good_frame()
    cols = [c for c in px.columns if c != "BOXX"]
    px.loc[px.index[-1], cols[5:]] = np.nan      # only a few names printed
    with pytest.raises(SignalError, match="partial download"):
        check_data_quality(px, cols, "BOXX", max_staleness_days=5,
                           min_coverage=0.85, now=pd.Timestamp("2026-09-09"))


def test_quality_gate_rejects_a_missing_safe_asset():
    px = _good_frame().drop(columns=["BOXX"])
    with pytest.raises(SignalError, match="safe asset"):
        check_data_quality(px, [c for c in px.columns], "BOXX",
                           max_staleness_days=5, min_coverage=0.85,
                           now=pd.Timestamp("2026-09-09"))


def test_quality_gate_passes_a_healthy_frame():
    px = _good_frame()
    cols = [c for c in px.columns if c != "BOXX"]
    d = check_data_quality(px, cols, "BOXX", max_staleness_days=5, min_coverage=0.85,
                           now=pd.Timestamp("2026-09-09"))
    assert d["coverage"] == 1.0 and d["n_tickers"] == len(cols)


# --------------------------------------------------------------------------
# The live signal must equal the backtest's last row
# --------------------------------------------------------------------------

def test_live_signal_matches_the_backtest_final_weights():
    """The whole point of the live layer: same code, same answer.

    If this ever fails, the live book has silently forked from the strategy
    the backtest reports on -- which is the single worst failure mode this
    deployment has.
    """
    from qbs.strategies import book_vol_target, cross_sectional_momentum

    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    frame = uni.copy()
    frame["BOXX"] = px["BOXX"]

    # The backtest path.
    mom = cross_sectional_momentum(uni, px["BOXX"], cfg.momentum)
    vt = book_vol_target(mom, frame, cfg.book_vol, lag=cfg.execution_lag)
    expected = vt.weights.iloc[-1]

    # The live path.
    book = compute_targets(cfg, frame, requested=list(uni.columns),
                           max_staleness_days=10_000, min_coverage=0.5,
                           now=frame.index[-1])

    for sym, w in book.weights.items():
        assert abs(w - float(expected[sym])) < 1e-12, f"{sym} diverged"
    assert abs(sum(book.weights.values()) - 1.0) < 1e-9


def test_live_signal_reports_the_scalar_and_holdings():
    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    frame = uni.copy()
    frame["BOXX"] = px["BOXX"]

    book = compute_targets(cfg, frame, requested=list(uni.columns),
                           max_staleness_days=10_000, min_coverage=0.5,
                           now=frame.index[-1])
    assert 0.0 <= book.scalar <= cfg.book_vol.max_weight
    assert len(book.raw_holdings) <= cfg.momentum.n_hold
    assert 0.0 <= book.risk_weight <= 1.0 + 1e-9
    assert all(p > 0 for p in book.prices.values())


# --------------------------------------------------------------------------
# Config and state
# --------------------------------------------------------------------------

def test_live_config_refuses_a_bad_notional():
    with pytest.raises(ValueError, match="notional"):
        LiveConfig(notional=0)


def test_live_config_knows_the_paper_ports():
    assert LiveConfig(ib_port=4002).is_paper_port
    assert LiveConfig(ib_port=7497).is_paper_port
    assert not LiveConfig(ib_port=4001).is_paper_port
    assert not LiveConfig(ib_port=7496).is_paper_port


def test_live_config_rejects_unknown_json_keys():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "live.json")
        json.dump({"notionall": 5}, open(p, "w"))
        with pytest.raises(ValueError, match="unknown keys"):
            LiveConfig.from_env(p)


def test_live_config_env_overrides_the_file():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "live.json")
        json.dump({"notional": 50_000.0}, open(p, "w"))
        os.environ["QBS_NOTIONAL"] = "12345"
        try:
            assert LiveConfig.from_env(p).notional == 12345.0
        finally:
            del os.environ["QBS_NOTIONAL"]


def test_config_redacts_the_account_number():
    cfg = LiveConfig(ib_account="DU1234567")
    assert cfg.redacted()["ib_account"] == "DU12***"


def test_state_survives_a_corrupt_file():
    """A half-written state file after an instance stop must not wedge the runner."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "state.json")
        open(p, "w").write("{not json")
        assert st.load_state(p) == {"schema": st.SCHEMA_VERSION, "runs": []}


def test_state_records_and_trims_runs():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "state.json")
        for i in range(12):
            st.record_run(p, "trade", "ok", {"n": i}, keep=5)
        data = st.load_state(p)
        assert len(data["runs"]) == 5
        assert data["runs"][-1]["n"] == 11
        assert data["last_trade"]["n"] == 11


def test_kill_switch_detection():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "HALT")
        assert not st.kill_switch_engaged(p)
        open(p, "w").close()
        assert st.kill_switch_engaged(p)


def test_orders_csv_gets_a_header_once():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "orders.csv")
        o = [Order("AAA", "BUY", 10, 100.0, "entry")]
        st.log_orders(p, pd.Timestamp("2026-09-09"), o, {"AAA": "Submitted"})
        st.log_orders(p, pd.Timestamp("2026-09-10"), o, {"AAA": "Submitted"})
        lines = open(p).read().strip().split("\n")
        assert len(lines) == 3, "header + two rows"
        assert lines[0].startswith("at,asof,symbol")


# --------------------------------------------------------------------------
# Broker: only the logic that is genuinely its own
# --------------------------------------------------------------------------

class _Ctr:
    def __init__(self, symbol, sec_type="STK", con_id=1):
        self.symbol, self.secType, self.conId = symbol, sec_type, con_id


class _Pos:
    def __init__(self, symbol, qty, sec_type="STK"):
        self.contract, self.position = _Ctr(symbol, sec_type), qty


class _Val:
    def __init__(self, tag, value, currency="USD"):
        self.tag, self.value, self.currency = tag, value, currency


class _Status:
    def __init__(self, status):
        self.status = status


class _IBOrder:
    def __init__(self, order_id=1, action="BUY", qty=1):
        self.orderId, self.action, self.totalQuantity = order_id, action, qty


class _Trade:
    def __init__(self, status="Submitted", order_id=1):
        self.orderStatus, self.order, self.log = _Status(status), _IBOrder(order_id), []
        self.contract = _Ctr("X")


class StubIB:
    """Stands in for ib_async.IB. Records what the broker asked it to do."""
    instances = []

    def __init__(self):
        self.connected = False
        self.placed = []
        self.cancelled = []
        self._positions = []
        self._values = []
        self._status = "Submitted"
        StubIB.instances.append(self)

    def connect(self, host, port, clientId=None, timeout=None):
        self.connected = True

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False

    def managedAccounts(self):
        return ["DU111"]

    def positions(self, account=None):
        return self._positions

    def accountValues(self, account=None):
        return self._values

    def qualifyContracts(self, *cs):
        for c in cs:
            c.conId = 42
        return list(cs)

    def placeOrder(self, contract, order):
        self.placed.append((contract.symbol, order.action, order.totalQuantity,
                            order.orderType))
        return _Trade(self._status)

    def cancelOrder(self, order):
        self.cancelled.append(order)

    def openTrades(self):
        return []

    def fills(self):
        return []

    def sleep(self, _):
        pass


@pytest.fixture
def stub_ib(monkeypatch):
    import ib_async
    StubIB.instances = []
    monkeypatch.setattr(ib_async, "IB", StubIB)
    return StubIB


def test_broker_refuses_a_live_port(stub_ib):
    from qbs.live.broker import BrokerError, IBBroker
    with pytest.raises(BrokerError, match="LIVE trading port"):
        IBBroker(LiveConfig(ib_port=4001)).connect()
    assert not stub_ib.instances, "must refuse before constructing a connection"


def test_broker_allows_a_live_port_only_when_told_to(stub_ib):
    from qbs.live.broker import IBBroker
    IBBroker(LiveConfig(ib_port=4001, allow_live_account=True)).connect()
    assert stub_ib.instances[0].connected


def test_broker_ignores_non_stock_positions(stub_ib):
    """An option or future in the account belongs to something else entirely."""
    from qbs.live.broker import IBBroker
    b = IBBroker(LiveConfig())
    b.connect()
    b.ib._positions = [_Pos("AMD", 10), _Pos("SPY", 5, sec_type="OPT"),
                       _Pos("EUR", 1000, sec_type="CASH")]
    assert b.positions() == {"AMD": 10}


def test_broker_drops_flat_positions(stub_ib):
    from qbs.live.broker import IBBroker
    b = IBBroker(LiveConfig())
    b.connect()
    b.ib._positions = [_Pos("AMD", 10), _Pos("MU", 0)]
    assert b.positions() == {"AMD": 10}


def test_broker_reads_usd_net_liquidation(stub_ib):
    from qbs.live.broker import IBBroker
    b = IBBroker(LiveConfig())
    b.connect()
    b.ib._values = [_Val("NetLiquidation", "250000", "EUR"),
                    _Val("BuyingPower", "1000"),
                    _Val("NetLiquidation", "123456.78")]
    assert b.net_liquidation() == pytest.approx(123456.78)


def test_broker_sends_moc_orders(stub_ib):
    from qbs.live.broker import IBBroker
    b = IBBroker(LiveConfig())
    b.connect()
    b.submit_moc([Order("AMD", "BUY", 11, 505.0), Order("MU", "SELL", 6, 1000.0)])
    assert b.ib.placed == [("AMD", "BUY", 11, "MOC"), ("MU", "SELL", 6, "MOC")]


def test_broker_dry_run_sends_nothing(stub_ib):
    from qbs.live.broker import IBBroker
    b = IBBroker(LiveConfig(dry_run=True))
    b.connect()
    out = b.submit_moc([Order("AMD", "BUY", 11, 505.0)])
    assert b.ib.placed == [], "dry run must not reach placeOrder"
    assert [f.status for f in out] == ["DryRun"]


def test_broker_raises_when_ib_rejects_an_order(stub_ib):
    """A rejection must stop the run, not be logged and stepped over."""
    from qbs.live.broker import BrokerError, IBBroker
    b = IBBroker(LiveConfig())
    b.connect()
    b.ib._status = "Inactive"
    with pytest.raises(BrokerError, match="rejected"):
        b.submit_moc([Order("AMD", "BUY", 11, 505.0)])


def test_broker_rejects_an_account_the_gateway_does_not_serve(stub_ib):
    from qbs.live.broker import BrokerError, IBBroker
    with pytest.raises(BrokerError, match="not served by this Gateway"):
        IBBroker(LiveConfig(ib_account="DU999")).connect()


def test_broker_qualifies_each_symbol_once(stub_ib):
    from qbs.live.broker import IBBroker
    b = IBBroker(LiveConfig())
    b.connect()
    first = b.qualify(["AMD", "MU"])
    again = b.qualify(["AMD", "MU", "INTC"])
    assert set(first) == {"AMD", "MU"}
    assert set(again) == {"AMD", "MU", "INTC"}
