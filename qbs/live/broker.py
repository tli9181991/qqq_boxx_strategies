"""The ib_insync layer. Everything that talks to IB Gateway lives here.

Kept deliberately thin and free of strategy logic, so the parts that decide
what to trade (`signals.py`, `orders.py`) stay testable without a Gateway.

Order type
----------
MOC (market-on-close). The backtest decides on a price near the close and
assumes the fill happens in the closing auction -- that gap is exactly what
`slippage_bps = 5` is modelling. MOC is the order type that reproduces it.
IB requires MOC orders in well before the auction (the cutoff is around
15:45-15:50 ET for US stocks), which is why the job submits at 15:40.

A note on ib_insync: it is unmaintained upstream since 2024. It works, and it
is what this deployment targets, but pin the version -- an incompatible
pandas or eventkit release is the likely way this breaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import LiveConfig
from .orders import Order

log = logging.getLogger(__name__)

MOC_ORDER_TYPE = "MOC"


class BrokerError(RuntimeError):
    pass


@dataclass
class Fill:
    symbol: str
    action: str
    quantity: float
    avg_price: float
    status: str
    order_id: int = 0


class IBBroker:
    """Thin ib_insync wrapper with a context-manager lifecycle.

    Usage::

        with IBBroker(cfg) as broker:
            positions = broker.positions()
            broker.submit_moc(orders)
    """

    def __init__(self, cfg: LiveConfig):
        self.cfg = cfg
        self.ib = None
        self._contracts: Dict[str, object] = {}

    # ---- lifecycle -------------------------------------------------------
    def __enter__(self) -> "IBBroker":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def connect(self) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BrokerError(
                "ib_insync is not installed. `pip install -r requirements-live.txt`"
            ) from exc

        if not self.cfg.is_paper_port and not self.cfg.allow_live_account:
            raise BrokerError(
                f"port {self.cfg.ib_port} is a LIVE trading port. Refusing to connect. "
                "Set allow_live_account=true (or QBS_ALLOW_LIVE=1) if that is deliberate.")

        self.ib = IB()
        log.info("connecting to IB at %s:%s (clientId=%s)",
                 self.cfg.ib_host, self.cfg.ib_port, self.cfg.ib_client_id)
        try:
            self.ib.connect(self.cfg.ib_host, self.cfg.ib_port,
                            clientId=self.cfg.ib_client_id,
                            timeout=self.cfg.connect_timeout)
        except Exception as exc:
            raise BrokerError(
                f"could not connect to IB Gateway at {self.cfg.ib_host}:{self.cfg.ib_port} "
                f"({type(exc).__name__}: {exc}). Is the Gateway running and logged in?"
            ) from exc

        accounts = self.ib.managedAccounts()
        log.info("connected; managed accounts: %s", accounts)
        if self.cfg.ib_account and self.cfg.ib_account not in accounts:
            raise BrokerError(
                f"configured account {self.cfg.ib_account} is not served by this Gateway "
                f"(it has {accounts})")

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
            log.info("disconnected from IB")

    # ---- reads -----------------------------------------------------------
    @property
    def account(self) -> str:
        if self.cfg.ib_account:
            return self.cfg.ib_account
        accts = self.ib.managedAccounts()
        if not accts:
            raise BrokerError("Gateway reports no managed accounts")
        return accts[0]

    def positions(self) -> Dict[str, int]:
        """Current share counts by symbol, for STK positions only.

        Non-stock positions (options, futures, FX) are ignored rather than
        coerced: this strategy only ever holds equities, so anything else in
        the account belongs to something other than this system and must not
        be netted into its view of the book.
        """
        out: Dict[str, int] = {}
        for p in self.ib.positions(self.account):
            c = p.contract
            if getattr(c, "secType", None) != "STK":
                log.info("ignoring non-stock position: %s %s", c.secType, c.symbol)
                continue
            out[c.symbol] = int(out.get(c.symbol, 0) + p.position)
        return {k: v for k, v in out.items() if v != 0}

    def net_liquidation(self) -> float:
        """Reported for the log only -- sizing uses the configured notional."""
        for v in self.ib.accountValues(self.account):
            if v.tag == "NetLiquidation" and v.currency == "USD":
                return float(v.value)
        return float("nan")

    def open_orders(self) -> List[object]:
        return list(self.ib.reqAllOpenOrders())

    # ---- contracts -------------------------------------------------------
    def qualify(self, symbols: List[str]) -> Dict[str, object]:
        """Resolve symbols to IB contracts, once per session.

        SMART routing with USD currency and a primary exchange hint: without
        the hint, ambiguous symbols raise rather than resolving, which is the
        behaviour we want -- an unqualified symbol must stop the run, not
        route somewhere unexpected.
        """
        from ib_insync import Stock

        todo = [s for s in symbols if s not in self._contracts]
        if not todo:
            return {s: self._contracts[s] for s in symbols}

        wanted = [Stock(s, "SMART", "USD") for s in todo]
        qualified = self.ib.qualifyContracts(*wanted)
        by_symbol = {c.symbol: c for c in qualified if getattr(c, "conId", 0)}

        missing = [s for s in todo if s not in by_symbol]
        if missing:
            raise BrokerError(f"IB could not qualify these symbols: {missing}")

        self._contracts.update(by_symbol)
        return {s: self._contracts[s] for s in symbols}

    # ---- writes ----------------------------------------------------------
    def submit_moc(self, orders: List[Order]) -> List[Fill]:
        """Place every order as MOC and wait briefly for acknowledgement.

        Returns one row per order with whatever status IB reported. MOC orders
        do not fill until the auction, so a `Submitted` (or `PreSubmitted`)
        status here is the expected success case -- the actual fill is picked
        up by the reconcile phase after the close.
        """
        from ib_insync import Order as IBOrder

        if not orders:
            log.info("no orders to submit")
            return []

        contracts = self.qualify([o.symbol for o in orders])
        results: List[Fill] = []

        for o in orders:
            ib_order = IBOrder(
                action=o.action,
                orderType=MOC_ORDER_TYPE,
                totalQuantity=int(o.quantity),
                tif="DAY",
                account=self.account,
                transmit=True,
            )
            if self.cfg.dry_run:
                log.info("DRY RUN, not sending: %s", o)
                results.append(Fill(o.symbol, o.action, o.quantity, o.price_hint,
                                    "DryRun"))
                continue

            trade = self.ib.placeOrder(contracts[o.symbol], ib_order)
            self.ib.sleep(0.25)   # let the event loop deliver the ack
            status = trade.orderStatus.status if trade.orderStatus else "Unknown"
            log.info("sent %s -> %s (orderId=%s)", o, status,
                     trade.order.orderId if trade.order else "?")

            if status in ("Inactive", "ApiCancelled", "Cancelled"):
                logs = "; ".join(e.message for e in (trade.log or [])[-3:])
                raise BrokerError(
                    f"IB rejected {o.action} {o.quantity} {o.symbol}: {status}. {logs}")

            results.append(Fill(o.symbol, o.action, o.quantity, o.price_hint, status,
                                order_id=trade.order.orderId if trade.order else 0))

        # Give the Gateway a moment to settle all acknowledgements.
        self.ib.sleep(1.0)
        return results

    def cancel_all_open(self) -> int:
        """Cancel any of our still-open orders. Used by the reconcile phase.

        A working MOC that somehow survived the auction must not be left to
        fire into the next session, when the signal that produced it is a day
        stale.
        """
        n = 0
        for trade in self.ib.openTrades():
            status = trade.orderStatus.status if trade.orderStatus else ""
            if status in ("PendingSubmit", "PreSubmitted", "Submitted"):
                log.warning("cancelling still-open order: %s %s %s",
                            trade.order.action, trade.order.totalQuantity,
                            trade.contract.symbol)
                if not self.cfg.dry_run:
                    self.ib.cancelOrder(trade.order)
                n += 1
        if n:
            self.ib.sleep(1.0)
        return n

    def todays_fills(self) -> List[Fill]:
        """Executions reported this session, for the reconcile log."""
        out: List[Fill] = []
        for f in self.ib.fills():
            ex, c = f.execution, f.contract
            out.append(Fill(
                symbol=c.symbol,
                action="BUY" if ex.side.upper().startswith("B") else "SELL",
                quantity=float(ex.shares),
                avg_price=float(ex.price),
                status="Filled",
                order_id=int(ex.orderId),
            ))
        return out
