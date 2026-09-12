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
from qbs.live import store
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
    assert LiveConfig(ib_port=4004).is_paper_port, "the gnzsnz image serves paper here"
    assert not LiveConfig(ib_port=4001).is_paper_port
    assert not LiveConfig(ib_port=7496).is_paper_port


def test_paper_ports_are_configurable():
    """A containerised Gateway can publish the paper API anywhere."""
    cfg = LiveConfig(ib_port=5555, paper_ports=[5555])
    assert cfg.is_paper_port
    assert not LiveConfig(ib_port=5555).is_paper_port


def test_paper_account_prefixes():
    cfg = LiveConfig()
    assert cfg.looks_like_paper_account("DU1234567")
    assert cfg.looks_like_paper_account("DF1234567")
    assert not cfg.looks_like_paper_account("U1234567"), "live individual"
    assert not cfg.looks_like_paper_account("F1234567"), "live advisor"


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
    with pytest.raises(BrokerError, match="not a known paper port"):
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


def test_broker_refuses_a_live_account_even_on_a_paper_port(stub_ib):
    """The port is convention; the account number is what IB actually says.

    A Gateway misconfigured to serve a live account on the paper port would
    otherwise be traded against silently -- the one mistake here that cannot be
    undone.
    """
    from qbs.live.broker import BrokerError, IBBroker

    class LiveIB(StubIB):
        def managedAccounts(self):
            return ["U7654321"]

    import ib_async
    monkey = LiveIB
    ib_async.IB = monkey
    b = IBBroker(LiveConfig(ib_port=4002))
    with pytest.raises(BrokerError, match="do not look like"):
        b.connect()
    assert not b.ib.isConnected(), "must disconnect, not stay attached to a live account"


def test_broker_accepts_a_live_account_when_explicitly_allowed(stub_ib):
    from qbs.live.broker import IBBroker

    class LiveIB(StubIB):
        def managedAccounts(self):
            return ["U7654321"]

    import ib_async
    ib_async.IB = LiveIB
    b = IBBroker(LiveConfig(ib_port=4002, allow_live_account=True))
    b.connect()
    assert b.ib.isConnected()


def test_broker_connects_on_a_nonstandard_paper_port(stub_ib):
    """Port 4004, as the gnzsnz container image serves it."""
    from qbs.live.broker import IBBroker
    b = IBBroker(LiveConfig(ib_port=4004))
    b.connect()
    assert b.ib.connected


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


# --------------------------------------------------------------------------
# The run log
# --------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "qbs.db")


def test_store_creates_all_tables_on_demand(db):
    assert store.summary(db) == {"trade_events": 0, "selection_events": 0,
                                 "position_closes": 0, "portfolio_nav": 0,
                                 "signal_runs": 0}


def test_store_refuses_a_newer_schema(db):
    import sqlite3
    with store.connect(db):
        pass
    con = sqlite3.connect(db)
    con.execute(f"PRAGMA user_version={store.SCHEMA_VERSION + 5}")
    con.commit()
    con.close()
    with pytest.raises(RuntimeError, match="Upgrade the code"):
        store.summary(db)


def test_trade_events_are_append_only(db):
    """A phase that ran twice really did submit twice; the log must say so."""
    o = [Order("AMD", "BUY", 11, 505.0, "entry")]
    store.log_orders(db, "2026-09-09", "trade", o, {"AMD": "Submitted"})
    store.log_orders(db, "2026-09-09", "trade", o, {"AMD": "Submitted"})
    rows = store.trades_on(db, "2026-09-09")
    assert len(rows) == 2, "the event log must not deduplicate"
    assert rows[0]["notional"] == pytest.approx(11 * 505.0)
    assert rows[0]["event"] == "submitted"


def test_daily_snapshots_upsert_instead_of_duplicating(db):
    """Re-running reconcile for one session must correct the row, not double it."""
    for mv in (1000.0, 1234.0):
        store.log_position_closes(db, "2026-09-09", [
            {"symbol": "AMD", "shares": 11, "close_price": mv / 11,
             "market_value": mv, "unrealized_pnl": 0.0}])
        store.log_portfolio_nav(db, "2026-09-09", total_market_value=mv,
                                n_positions=1, source="ib")
    closes = store.closes_on(db, "2026-09-09")
    nav = store.nav_history(db)
    assert len(closes) == 1 and closes[0]["market_value"] == 1234.0
    assert len(nav) == 1 and nav[0]["total_market_value"] == 1234.0


def test_selection_events_upsert_per_event_type(db):
    """One name can exit on one date and enter on another; both must survive."""
    store.log_selection(db, "2026-09-09", [
        {"symbol": "MRVL", "event": "entry", "rank": 4, "score": 0.51},
        {"symbol": "NVDA", "event": "exit", "rank": 14, "score": 0.10},
        {"symbol": "AMD", "event": "hold", "rank": 2, "score": 0.63},
    ])
    store.log_selection(db, "2026-09-10", [
        {"symbol": "MRVL", "event": "exit", "rank": 12, "score": 0.08}])
    rows = store.selection_history(db)
    assert len(rows) == 4
    mrvl = sorted([r for r in rows if r["symbol"] == "MRVL"],
                  key=lambda r: r["session_date"])
    assert [r["event"] for r in mrvl] == ["entry", "exit"]


def test_selection_rerun_is_idempotent(db):
    """The ranking is deterministic, so logging it twice must change nothing."""
    rows = [{"symbol": "AMD", "event": "hold", "rank": 2, "score": 0.63}]
    store.log_selection(db, "2026-09-09", rows)
    store.log_selection(db, "2026-09-09", rows)
    assert store.summary(db)["selection_events"] == 1


def test_nav_history_comes_back_oldest_first(db):
    for d, mv in (("2026-09-07", 1.0), ("2026-09-09", 3.0), ("2026-09-08", 2.0)):
        store.log_portfolio_nav(db, d, total_market_value=mv)
    assert [r["total_market_value"] for r in store.nav_history(db)] == [1.0, 2.0, 3.0]


def test_guard_trips_and_skips_are_recorded_as_events(db):
    """A refused session must leave a trace, or the log shows a silent no-op day."""
    store.log_note(db, "2026-09-09", "trade", "guard_tripped", "turnover 300%")
    rows = store.trades_on(db, "2026-09-09")
    assert len(rows) == 1
    assert rows[0]["event"] == "guard_tripped" and "300%" in rows[0]["reason"]


def test_fills_are_logged_with_average_price(db):
    from qbs.live.broker import Fill
    store.log_fills(db, "2026-09-09", [Fill("AMD", "BUY", 11, 504.25, "Filled", 77)])
    r = store.trades_on(db, "2026-09-09")[0]
    assert (r["event"], r["price"], r["order_id"]) == ("filled", 504.25, 77)


def test_dry_run_orders_are_flagged_in_the_log(db):
    store.log_orders(db, "2026-09-09", "trade", [Order("AMD", "BUY", 11, 505.0)],
                     dry_run=True)
    assert store.trades_on(db, "2026-09-09")[0]["dry_run"] == 1


def test_store_rejects_an_unknown_table_name(db):
    with pytest.raises(ValueError, match="unknown table"):
        store.to_frame(db, "trade_events; DROP TABLE trade_events")


def test_to_frame_round_trips_through_pandas(db):
    store.log_orders(db, "2026-09-09", "trade", [Order("AMD", "BUY", 11, 505.0)])
    df = store.to_frame(db, "trade_events")
    assert list(df["symbol"]) == ["AMD"] and df["quantity"].iloc[0] == 11


def test_signal_runs_keep_one_row_per_phase(db):
    """Preflight and trade both keep a row, so you can see the intraday drift."""
    class _B:
        asof = pd.Timestamp("2026-09-09")
        scalar, book_vol, risk_weight = 0.36, 0.62, 0.36
        n_rankable, universe_size = 88, 99
        raw_holdings = ["LRCX", "MU"]
    store.log_signal_run(db, "2026-09-09", "preflight", _B())
    store.log_signal_run(db, "2026-09-09", "trade", _B())
    rows = store.to_frame(db, "signal_runs")
    assert set(rows["phase"]) == {"preflight", "trade"}
    assert rows["holdings"].iloc[0] == "LRCX,MU"


# --------------------------------------------------------------------------
# Selection rows come from the strategy, not from re-derivation
# --------------------------------------------------------------------------

def _book_with_selection():
    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    frame = uni.copy()
    frame["BOXX"] = px["BOXX"]
    return compute_targets(cfg, frame, requested=list(uni.columns),
                           max_staleness_days=10_000, min_coverage=0.5,
                           now=frame.index[-1]), cfg


def test_selection_covers_every_held_name():
    book, cfg = _book_with_selection()
    covered = {r["symbol"] for r in book.selection if r["event"] in ("entry", "hold")}
    assert set(book.raw_holdings) <= covered, "every held name needs a selection row"
    assert all(r["event"] in ("entry", "exit", "hold") for r in book.selection)


def test_selection_ranks_are_the_strategy_own_ranks():
    """Holds carry the rank the strategy ranked them at, not a re-derived one."""
    from qbs.strategies import cross_sectional_momentum

    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    mom = cross_sectional_momentum(uni, px["BOXX"], cfg.momentum)
    asof = mom.weights.index[-1]
    truth = (mom.held_ranks or {}).get(asof, {})

    frame = uni.copy()
    frame["BOXX"] = px["BOXX"]
    book = compute_targets(cfg, frame, requested=list(uni.columns),
                           max_staleness_days=10_000, min_coverage=0.5, now=asof)
    for r in book.selection:
        if r["event"] == "hold" and r["rank"] is not None:
            assert r["rank"] == int(truth[r["symbol"]]), f"{r['symbol']} rank drifted"


def test_momentum_events_expose_rank_and_score_as_columns():
    """Structured, so the run log never has to parse them back out of prose."""
    from qbs.strategies import cross_sectional_momentum

    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    ev = cross_sectional_momentum(uni, px["BOXX"], cfg.momentum).events
    assert {"rank", "score"} <= set(ev.columns)
    buys = ev[ev["action"] == "buy"]
    assert (buys["rank"] >= 1).all() and buys["score"].notna().all()


def test_selection_survives_a_round_trip_through_the_store(db):
    book, _ = _book_with_selection()
    store.log_selection(db, book.asof, book.selection)
    rows = store.selection_history(db, limit=500)
    assert len(rows) == len(book.selection)
    assert {r["symbol"] for r in rows} == {r["symbol"] for r in book.selection}


# --------------------------------------------------------------------------
# The state directory has to be writable before anything else is attempted
# --------------------------------------------------------------------------

def test_writable_state_dir_passes_and_is_created():
    from qbs.live.runner import check_state_dir

    with tempfile.TemporaryDirectory() as d:
        nested = os.path.join(d, "var")
        assert check_state_dir(nested) is None
        assert os.path.isdir(nested)      # created, not merely accepted


def test_unwritable_state_dir_is_reported_before_any_work():
    """The message has to name the fix, not just the errno.

    This failure shows up in production as a PermissionError raised inside an
    exception handler, where it replaces the error being handled. Whoever reads
    that log at 15:30 sees a tempfile traceback and no mention of ownership.
    """
    from qbs.live.runner import check_state_dir

    # A path *under a regular file* is unusable for every uid, root included,
    # so this asserts the same branch the container hits without needing the
    # suite to run unprivileged.
    with tempfile.TemporaryDirectory() as d:
        blocker = os.path.join(d, "not-a-dir")
        open(blocker, "w").close()
        locked = os.path.join(blocker, "var")
        msg = check_state_dir(locked)

    assert msg is not None
    assert locked in msg
    assert "chown" in msg                 # the actual remedy
    assert "QBS_UID" in msg               # and where the uid comes from


def test_main_refuses_to_run_a_phase_on_an_unwritable_state_dir(monkeypatch):
    from qbs.live import runner

    monkeypatch.setattr(runner, "check_state_dir", lambda d: "nope")
    called = []
    monkeypatch.setattr(runner, "phase_preflight",
                        lambda *a, **k: called.append("ran") or 0)

    rc = runner.main(["preflight"])

    assert rc == runner.EXIT_CONFIG
    assert not called, "the phase ran despite an unusable state directory"


# --------------------------------------------------------------------------
# The run log must never read the same for a dry run as for a real one
# --------------------------------------------------------------------------

def test_report_marks_dry_run_orders_as_not_sent(db, capsys, monkeypatch):
    from qbs.live import runner
    from qbs.live.orders import Order

    orders = [Order(symbol="MU", action="BUY", quantity=6, price_hint=974.27)]
    store.log_orders(db, "2026-09-11", "trade", orders, {"MU": "DryRun"},
                     dry_run=True)

    live = LiveConfig(state_dir=os.path.dirname(db))
    monkeypatch.setattr(type(live), "db_path", property(lambda self: db))
    runner.phase_report(live, days=5)
    out = capsys.readouterr().out

    assert "MU" in out
    assert "no (dry run)" in out, "a dry run reads as a real submission"
    assert "DRY RUN and never" in out


def test_report_marks_real_orders_as_sent(db, capsys, monkeypatch):
    from qbs.live import runner
    from qbs.live.orders import Order

    orders = [Order(symbol="MU", action="BUY", quantity=6, price_hint=974.27)]
    store.log_orders(db, "2026-09-11", "trade", orders, {"MU": "Submitted"},
                     dry_run=False)

    live = LiveConfig(state_dir=os.path.dirname(db))
    monkeypatch.setattr(type(live), "db_path", property(lambda self: db))
    runner.phase_report(live, days=5)
    out = capsys.readouterr().out

    assert "no (dry run)" not in out
    assert "DRY RUN and never" not in out


# --------------------------------------------------------------------------
# Exits must actually leave the book
# --------------------------------------------------------------------------

def _exit_scenario(**kw):
    from qbs.live.orders import build_orders
    return build_orders(
        weights={"NVDA": 0.36, "BOXX": 0.64},
        prices={"NVDA": 100.0, "BOXX": 118.0, "MU": 974.0},
        actual={"MU": 6, "BOXX": 539, "TSLA": 40},
        notional=100_000, max_order_notional=40_000,
        max_gross_turnover=1.6, max_positions=12, **kw)


def test_a_name_that_left_the_target_is_sold():
    """The regression that matters: zero weights are not carried.

    `compute_targets` drops weights of zero, so a name the strategy has exited
    is absent from `weights` entirely -- indistinguishable, without the
    universe, from a holding that was never the strategy's. Read as the latter
    it is never sold, and the book buys replacements while keeping every name
    it ever held.
    """
    orders, _ = _exit_scenario(universe=["NVDA", "MU", "AMD", "BOXX"])
    sells = {o.symbol: o for o in orders if o.action == "SELL"}

    assert "MU" in sells, "an exited holding was left in the book"
    assert sells["MU"].quantity == 6
    assert sells["MU"].reason == "exit"
    assert sells["MU"].notional > 0, "the exit should be priced, not logged as $0"


def test_holdings_outside_the_universe_are_still_left_alone():
    orders, _ = _exit_scenario(universe=["NVDA", "MU", "AMD", "BOXX"])
    assert not any(o.symbol == "TSLA" for o in orders)


def test_exits_are_ordered_before_the_entries_they_fund():
    orders, _ = _exit_scenario(universe=["NVDA", "MU", "AMD", "BOXX"])
    actions = [o.action for o in orders]
    assert actions.index("SELL") < actions.index("BUY")


def test_the_target_book_carries_its_universe():
    """Without this the runner has nothing to pass, and the bug returns."""
    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    frame = uni.copy()
    frame[cfg.momentum.safe_asset] = px[cfg.momentum.safe_asset]

    book = compute_targets(cfg, frame, requested=list(uni.columns))

    assert cfg.momentum.safe_asset in book.universe
    assert set(book.weights) <= set(book.universe)
    assert len(book.universe) > len(book.weights), \
        "the universe should be wider than today's targets"
    # Every universe name is priced, so an exit can be logged with a notional.
    assert set(book.universe) <= set(book.prices)


# --------------------------------------------------------------------------
# Sharing an account with holdings the strategy must not touch
# --------------------------------------------------------------------------

def test_the_strategy_sees_only_its_own_share_of_a_shared_position():
    from qbs.live.orders import strategy_positions

    mine, short = strategy_positions({"TSM": 27, "MRVL": 125, "VOO": 30},
                                     {"TSM": 22, "MRVL": 100, "VOO": 30})
    assert mine == {"TSM": 5, "MRVL": 25}, "your shares leaked into the book"
    assert not short


def test_an_exit_sells_only_the_strategy_shares():
    """The failure this exists to prevent: selling 17 of someone else's shares."""
    from qbs.live.orders import build_orders, strategy_positions

    account, external = {"TSM": 27}, {"TSM": 22}
    mine, _ = strategy_positions(account, external)

    orders, _ = build_orders(
        weights={"BOXX": 1.0}, prices={"BOXX": 118.0, "TSM": 200.0},
        actual=mine, notional=100_000, max_order_notional=200_000,
        max_gross_turnover=1.6, max_positions=12, universe=["TSM", "BOXX"])

    sells = [o for o in orders if o.symbol == "TSM"]
    assert len(sells) == 1
    assert sells[0].action == "SELL"
    assert sells[0].quantity == 5, "the strategy tried to sell shares it does not own"


def test_a_stale_baseline_stops_the_run_rather_than_buying_forever():
    from qbs.live.orders import GuardTripped, check_external_baseline, strategy_positions

    mine, short = strategy_positions({"TSM": 5}, {"TSM": 22})
    assert mine == {}
    with pytest.raises(GuardTripped, match="stale"):
        check_external_baseline(short, universe=["TSM"])


def test_a_stale_baseline_outside_the_universe_is_only_a_warning():
    from qbs.live.orders import check_external_baseline, strategy_positions

    _, short = strategy_positions({"VOO": 5}, {"VOO": 30})
    check_external_baseline(short, universe=["TSM", "BOXX"])   # must not raise


def test_marks_are_reduced_to_the_strategy_share():
    from qbs.live.orders import attribute_marks

    marks = [
        dict(symbol="TSM", shares=27.0, close_price=200.0, market_value=5400.0,
             avg_cost=150.0, unrealized_pnl=1350.0, source="ib"),
        dict(symbol="AMD", shares=11.0, close_price=516.0, market_value=5676.0,
             avg_cost=500.0, unrealized_pnl=176.0, source="ib"),
    ]
    out = {m["symbol"]: m for m in attribute_marks(marks, {"TSM": 22})}

    assert out["TSM"]["shares"] == 5
    assert out["TSM"]["market_value"] == pytest.approx(1000.0)
    # Cost basis is blended across both owners, so no split of it is truthful.
    assert np.isnan(out["TSM"]["unrealized_pnl"])
    assert np.isnan(out["TSM"]["avg_cost"])
    assert out["AMD"] == marks[1], "a position you do not share must pass through"


def test_a_position_wholly_yours_drops_out_of_the_marks():
    from qbs.live.orders import attribute_marks

    marks = [dict(symbol="VOO", shares=30.0, close_price=500.0, market_value=15000.0,
                  avg_cost=400.0, unrealized_pnl=3000.0, source="ib")]
    assert attribute_marks(marks, {"VOO": 30}) == []


def test_baseline_round_trips_and_normalises_case(tmp_path):
    path = str(tmp_path / "external_positions.json")
    st.save_external_positions(path, {"tsm": 22, "MRVL": 100, "nothing": 0})
    assert st.load_external_positions(path) == {"TSM": 22, "MRVL": 100}


def test_a_corrupt_baseline_refuses_to_load_rather_than_reading_as_empty(tmp_path):
    """Empty means 'the whole account is mine', which is the dangerous reading."""
    path = tmp_path / "external_positions.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="could not read"):
        st.load_external_positions(str(path))


def test_a_negative_baseline_is_rejected(tmp_path):
    path = tmp_path / "external_positions.json"
    path.write_text('{"positions": {"TSM": -5}}')
    with pytest.raises(ValueError, match="negative"):
        st.load_external_positions(str(path))


# --------------------------------------------------------------------------
# Skipping a name you already hold, and taking the next one down
# --------------------------------------------------------------------------

def _excluded_book(exclude):
    cfg = Config()
    cfg.momentum.min_history = 200
    px = synthetic_prices()
    uni = synthetic_universe(n=30, start="2023-06-01").reindex(px.index).ffill()
    frame = uni.copy()
    frame[cfg.momentum.safe_asset] = px[cfg.momentum.safe_asset]
    return compute_targets(cfg, frame, requested=list(uni.columns),
                           exclude=exclude)


def test_an_excluded_name_is_replaced_not_left_empty():
    """A skipped slot goes to the next name down; the book stays six wide."""
    cfg = Config()
    base = _excluded_book(None)
    held = [t for t in base.raw_holdings]
    assert held, "the fixture produced no holdings"

    after = _excluded_book([held[0]])

    assert held[0] not in after.raw_holdings
    assert len(after.raw_holdings) == len(held), "the slot was dropped, not refilled"
    assert set(after.raw_holdings) - set(held), "no new name took the slot"


def test_an_excluded_name_stays_sellable():
    """The trap: dropping it from the tradeable set would strand the position.

    An excluded name the strategy already holds must still be closable, or the
    order builder reads it as somebody else's holding and never sells it.
    """
    base = _excluded_book(None)
    held = base.raw_holdings[0]
    after = _excluded_book([held])

    assert held not in after.weights, "an excluded name should have no target"
    assert held in after.universe, "an excluded name must stay tradeable to be sold"

    from qbs.live.orders import build_orders
    orders, _ = build_orders(
        after.weights, after.prices, {held: 10},
        notional=100_000, max_order_notional=200_000, max_gross_turnover=99,
        max_positions=12, universe=after.universe)
    sells = [o for o in orders if o.symbol == held]
    assert sells and sells[0].action == "SELL" and sells[0].quantity == 10


def test_excluding_everything_fails_loudly_rather_than_trading_a_thin_book():
    """Better to stop than to run a book narrower than the strategy specifies."""
    from qbs.universe import synthetic_universe

    everything = list(synthetic_universe(n=30, start="2023-06-01").columns)
    with pytest.raises(SignalError, match="after exclusions"):
        _excluded_book(everything)


def test_exclusions_come_from_config_and_optionally_the_baseline(tmp_path):
    from qbs.live.runner import _excluded_names

    live = LiveConfig(state_dir=str(tmp_path), exclude_tickers=["nvda"])
    st.save_external_positions(live.external_positions_path, {"MRVL": 100})

    assert _excluded_names(live) == ["NVDA"], "the baseline leaked in unasked"

    live.exclude_own_holdings = True
    assert _excluded_names(live) == ["MRVL", "NVDA"]


def test_the_book_csv_shows_both_owners_side_by_side(tmp_path):
    path = str(tmp_path / "strategy_book.csv")
    st.write_book_csv(path,
                      account={"MRVL": 125, "VOO": 30},
                      external={"MRVL": 100, "VOO": 30},
                      strategy={"MRVL": 25},
                      prices={"MRVL": 236.56},
                      target={"MRVL": 25},
                      notional=100_000, asof="2026-09-11")

    import csv as _csv
    rows = {r["symbol"]: r for r in _csv.DictReader(open(path))}

    assert rows["MRVL"]["strategy_shares"] == "25"
    assert rows["MRVL"]["account_shares"] == "125"
    assert rows["MRVL"]["yours"] == "100"
    assert float(rows["MRVL"]["market_value"]) == pytest.approx(25 * 236.56)
    # A holding wholly yours still appears, so the file explains the account.
    assert rows["VOO"]["strategy_shares"] == "0"


# --------------------------------------------------------------------------
# The trade ledger: the strategy's own fills as the position of record
# --------------------------------------------------------------------------

def _fill(symbol, action, qty, px, order_id=1, exec_id=""):
    from qbs.live.broker import Fill
    return Fill(symbol, action, qty, px, "Filled", order_id, exec_id)


def test_the_ledger_nets_buys_and_sells(tmp_path):
    from qbs.live import ledger

    path = str(tmp_path / "strategy_trades.csv")
    ledger.append_fills(path, "2026-09-15", [_fill("MU", "BUY", 6, 974.0, 1, "a")])
    ledger.append_fills(path, "2026-09-16", [_fill("MU", "SELL", 2, 980.0, 2, "b")])

    assert ledger.positions(path) == {"MU": 4}


def test_a_fully_closed_name_leaves_the_ledger_positions(tmp_path):
    from qbs.live import ledger

    path = str(tmp_path / "strategy_trades.csv")
    ledger.append_fills(path, "2026-09-15", [_fill("MU", "BUY", 6, 974.0, 1, "a")])
    ledger.append_fills(path, "2026-09-16", [_fill("MU", "SELL", 6, 980.0, 2, "b")])

    assert ledger.positions(path) == {}


def test_rerunning_reconcile_does_not_double_count(tmp_path):
    """Reconcile is re-runnable by design, so appending must be idempotent."""
    from qbs.live import ledger

    path = str(tmp_path / "strategy_trades.csv")
    fills = [_fill("MU", "BUY", 6, 974.0, 1, "exec-a"),
             _fill("MRVL", "BUY", 25, 236.0, 2, "exec-b")]

    assert ledger.append_fills(path, "2026-09-15", fills) == 2
    assert ledger.append_fills(path, "2026-09-15", fills) == 0
    assert ledger.positions(path) == {"MU": 6, "MRVL": 25}


def test_partial_fills_of_one_order_are_both_recorded(tmp_path):
    """Two executions share an order id; only the execution id separates them."""
    from qbs.live import ledger

    path = str(tmp_path / "strategy_trades.csv")
    ledger.append_fills(path, "2026-09-15", [
        _fill("MU", "BUY", 4, 974.0, order_id=7, exec_id="x1"),
        _fill("MU", "BUY", 2, 974.5, order_id=7, exec_id="x2"),
    ])
    assert ledger.positions(path) == {"MU": 6}


def test_dry_run_results_never_enter_the_ledger(tmp_path):
    from qbs.live import ledger
    from qbs.live.broker import Fill

    path = str(tmp_path / "strategy_trades.csv")
    n = ledger.append_fills(path, "2026-09-15",
                            [Fill("MU", "BUY", 6, 974.0, "DryRun", 0, "")])
    assert n == 0
    assert ledger.positions(path) == {}


def test_the_residual_is_what_you_hold_yourself(tmp_path):
    from qbs.live import ledger

    residual, over = ledger.reconcile_against_account(
        {"MU": 6, "MRVL": 25}, {"MU": 6, "MRVL": 125, "VOO": 30})

    assert residual == {"MRVL": 100, "VOO": 30}
    assert not over


def test_your_own_buying_does_not_move_the_strategy_book(tmp_path):
    """The whole reason to tally rather than derive."""
    from qbs.live import ledger

    path = str(tmp_path / "strategy_trades.csv")
    ledger.append_fills(path, "2026-09-15", [_fill("MRVL", "BUY", 25, 236.0, 1, "a")])

    before = ledger.positions(path)
    # You buy 50 more MRVL yourself; the account changes, the ledger does not.
    residual, over = ledger.reconcile_against_account(before, {"MRVL": 175})

    assert before == {"MRVL": 25}
    assert residual == {"MRVL": 150}
    assert not over


def test_a_ledger_claiming_more_than_the_account_holds_is_flagged():
    from qbs.live import ledger

    _, over = ledger.reconcile_against_account({"MU": 6}, {"MU": 2})
    assert over == {"MU": 4}


def test_position_source_must_be_one_of_the_three():
    with pytest.raises(ValueError, match="position_source"):
        LiveConfig(position_source="guess")
