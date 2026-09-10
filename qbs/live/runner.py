#!/usr/bin/env python3
"""The live runner. Three phases, one per systemd timer.

    preflight   08:45 ET  Gateway is up, data downloads, signal computes.
                          Runs seven hours early so a broken morning is a
                          fixable morning rather than a missed auction.
    trade       15:30 ET  Rank, build the order list, submit MOC by 15:40.
    reconcile   16:15 ET  Fills, final positions, cancel any straggler, then
                          the instance stops.

Each phase is a separate process. Nothing is carried in memory between them,
and nothing in the state file feeds the signal, so a phase that fails is a
phase you can simply re-run.

Exit codes
----------
    0   fine (including "market is closed today, nothing to do")
    1   something went wrong and no orders were sent
    2   configuration or connection problem
    3   a safety guard refused the order list

A non-zero exit makes the systemd unit fail, which is what surfaces the
problem. Never swallow one to keep the timer green.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Allow `python qbs/live/runner.py` as well as `python -m qbs.live.runner`.
if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from qbs.config import Config
from qbs.live.config import LiveConfig
from qbs.live.orders import GuardTripped, build_orders, format_order_table
from qbs.live.signals import compute_targets, load_live_prices
from qbs.live import state as st

log = logging.getLogger("qbs.live")

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_GUARD = 0, 1, 2, 3


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def setup_logging(verbose: bool = False, logfile: Optional[str] = None) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format=fmt, handlers=handlers, force=True)
    logging.getLogger("ib_async").setLevel(logging.WARNING)


def market_today(tz: str) -> datetime:
    """Now, in exchange local time."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz))
    except Exception:  # pragma: no cover - tzdata missing
        log.warning("could not load timezone %s; falling back to system local time", tz)
        return datetime.now()


def is_weekend(tz: str) -> bool:
    return market_today(tz).weekday() >= 5


def past_moc_cutoff(tz: str, cutoff_hhmm: str) -> bool:
    """Is it already too late for the closing auction?

    IB stops accepting MOC for US stocks a few minutes before 16:00. Submitting
    after that does not half-work -- the orders are rejected, and the book
    silently stays where it was while the log claims a successful run. Better
    to fail the unit loudly and let the next session correct it: the strategy
    recomputes from scratch daily, so one missed auction is self-healing.
    """
    hh, mm = (int(x) for x in cutoff_hhmm.split(":"))
    now = market_today(tz)
    return (now.hour, now.minute) >= (hh, mm)


# --------------------------------------------------------------------------
# Shared work
# --------------------------------------------------------------------------

def _load_and_compute(cfg: Config, live: LiveConfig, refresh: bool = True,
                      offline: bool = False):
    """Download prices and compute the target book. Shared by preflight and trade.

    `offline` reads the CSV cache and skips the network entirely. It exists for
    debugging on the box -- the trade phase never uses it, because ranking on
    yesterday's cache would submit an order list the strategy did not ask for.
    """
    from qbs.universe import load_universe

    tickers = load_universe(fetch=not offline, warn=False)
    log.info("universe: %d tickers", len(tickers))

    px = load_live_prices(cfg, tickers=tickers, fetch_universe=not offline,
                          refresh=refresh and not offline, extra=live.extra_tickers,
                          offline=offline)
    log.info("prices: %d rows x %d cols, last bar %s",
             len(px), px.shape[1], px.index.max().date())

    book = compute_targets(
        cfg, px, requested=tickers,
        max_staleness_days=live.max_price_staleness_days,
        min_coverage=live.min_universe_coverage,
    )
    return px, book


def _bar_is_today(book, tz: str) -> bool:
    return book.asof.date() == market_today(tz).date()


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------

def phase_preflight(cfg: Config, live: LiveConfig) -> int:
    """Prove the whole path works, hours before it matters. Sends nothing."""
    log.info("=== PREFLIGHT ===")
    log.info("config: %s", live.redacted())

    if st.kill_switch_engaged(live.kill_switch):
        log.error("kill switch present at %s -- the trade phase will refuse to run",
                  live.kill_switch)
        st.record_run(live.state_path, "preflight", "halted",
                      {"reason": "kill switch"})
        return EXIT_ERROR

    # 1. Data and signal.
    try:
        _, book = _load_and_compute(cfg, live, refresh=True)
    except Exception as exc:
        log.error("signal failed: %s: %s", type(exc).__name__, exc)
        st.record_run(live.state_path, "preflight", "error",
                      {"reason": f"{type(exc).__name__}: {exc}"})
        return EXIT_ERROR

    log.info("target book:\n%s", book.describe())

    # 2. Broker: connect, read positions, and dry-build the order list.
    from qbs.live.broker import BrokerError, IBBroker
    try:
        with IBBroker(live) as broker:
            positions = broker.positions()
            nlv = broker.net_liquidation()
            log.info("account %s: NetLiquidation $%s", broker.account, f"{nlv:,.0f}")
            log.info("current positions: %s", positions or "(flat)")

            orders, target = build_orders(
                book.weights, book.prices, positions,
                notional=live.notional,
                max_order_notional=live.max_order_notional,
                max_gross_turnover=live.max_gross_turnover,
                max_positions=live.max_positions,
                min_shares=live.min_order_shares,
                min_notional=live.min_order_notional,
                hold_safe_asset=live.hold_safe_asset,
                safe_asset=live.safe_asset,
            )
            log.info("orders the trade phase would send:\n%s",
                     format_order_table(orders, target, positions))
            broker.qualify(sorted({o.symbol for o in orders} | set(target)))
            log.info("all symbols qualified with IB")
    except GuardTripped as exc:
        log.error("GUARD would trip at trade time: %s", exc)
        st.record_run(live.state_path, "preflight", "guard", {"reason": str(exc)})
        return EXIT_GUARD
    except BrokerError as exc:
        log.error("broker problem: %s", exc)
        st.record_run(live.state_path, "preflight", "error", {"reason": str(exc)})
        return EXIT_CONFIG

    st.record_run(live.state_path, "preflight", "ok", {
        "asof": f"{book.asof:%Y-%m-%d}",
        "scalar": round(book.scalar, 4),
        "book_vol": round(book.book_vol, 4),
        "holdings": book.raw_holdings,
        "n_orders": len(orders),
    })
    log.info("preflight OK")
    return EXIT_OK


def phase_trade(cfg: Config, live: LiveConfig, force: bool = False) -> int:
    """Compute, build, submit MOC. This is the phase that moves money."""
    log.info("=== TRADE ===")

    if st.kill_switch_engaged(live.kill_switch):
        log.error("kill switch present at %s -- refusing to trade", live.kill_switch)
        st.record_run(live.state_path, "trade", "halted", {"reason": "kill switch"})
        return EXIT_ERROR

    if is_weekend(live.market_tz) and not force:
        log.info("weekend in %s -- nothing to do", live.market_tz)
        return EXIT_OK

    try:
        _, book = _load_and_compute(cfg, live, refresh=True)
    except Exception as exc:
        log.error("signal failed: %s: %s", type(exc).__name__, exc)
        st.record_run(live.state_path, "trade", "error",
                      {"reason": f"{type(exc).__name__}: {exc}"})
        return EXIT_ERROR

    # The auction we are aiming at is *today's*. If the feed has no bar for
    # today the market is shut (holiday) or the data is late; either way,
    # submitting MOC against a stale ranking is not what the backtest models.
    if not _bar_is_today(book, live.market_tz) and not force:
        log.warning("last bar is %s, not today (%s) -- market closed or data late. "
                    "Not trading.", book.asof.date(), market_today(live.market_tz).date())
        st.record_run(live.state_path, "trade", "skipped",
                      {"reason": "no bar for today", "asof": f"{book.asof:%Y-%m-%d}"})
        return EXIT_OK

    log.info("target book:\n%s", book.describe())

    from qbs.live.broker import BrokerError, IBBroker
    try:
        with IBBroker(live) as broker:
            positions = broker.positions()
            log.info("current positions: %s", positions or "(flat)")

            orders, target = build_orders(
                book.weights, book.prices, positions,
                notional=live.notional,
                max_order_notional=live.max_order_notional,
                max_gross_turnover=live.max_gross_turnover,
                max_positions=live.max_positions,
                min_shares=live.min_order_shares,
                min_notional=live.min_order_notional,
                hold_safe_asset=live.hold_safe_asset,
                safe_asset=live.safe_asset,
            )
            log.info("order list:\n%s", format_order_table(orders, target, positions))

            # Checked here, immediately before sending, not at the top of the
            # phase: the download and ranking take real time, and it is the
            # submission that has to beat the cutoff.
            if orders and past_moc_cutoff(live.market_tz, live.moc_cutoff_hhmm) \
                    and not force:
                raise BrokerError(
                    f"past the {live.moc_cutoff_hhmm} MOC cutoff in {live.market_tz} "
                    f"(now {market_today(live.market_tz):%H:%M}); not submitting. "
                    "Tomorrow's run recomputes from scratch and will correct the book.")

            if not orders:
                st.record_run(live.state_path, "trade", "flat", {
                    "asof": f"{book.asof:%Y-%m-%d}", "n_orders": 0,
                    "scalar": round(book.scalar, 4)})
                return EXIT_OK

            results = broker.submit_moc(orders)
            statuses = {r.symbol: r.status for r in results}
            st.log_orders(live.orders_log_path, book.asof, orders, statuses,
                          dry_run=live.dry_run)

    except GuardTripped as exc:
        log.error("GUARD TRIPPED, nothing sent: %s", exc)
        st.record_run(live.state_path, "trade", "guard", {"reason": str(exc)})
        return EXIT_GUARD
    except BrokerError as exc:
        log.error("broker problem: %s", exc)
        st.record_run(live.state_path, "trade", "error", {"reason": str(exc)})
        return EXIT_CONFIG

    st.record_run(live.state_path, "trade", "submitted", {
        "asof": f"{book.asof:%Y-%m-%d}",
        "scalar": round(book.scalar, 4),
        "book_vol": round(book.book_vol, 4),
        "holdings": book.raw_holdings,
        "n_orders": len(orders),
        "gross_notional": round(sum(o.notional for o in orders), 2),
        "statuses": statuses,
        "dry_run": live.dry_run,
    })
    log.info("submitted %d MOC orders", len(orders))
    return EXIT_OK


def phase_reconcile(cfg: Config, live: LiveConfig) -> int:
    """After the close: what filled, what the book is now, cancel any straggler."""
    log.info("=== RECONCILE ===")

    from qbs.live.broker import BrokerError, IBBroker
    try:
        with IBBroker(live) as broker:
            cancelled = broker.cancel_all_open()
            if cancelled:
                log.warning("cancelled %d order(s) still open after the close", cancelled)

            fills = broker.todays_fills()
            st.log_fills(live.fills_log_path, fills)
            for f in fills:
                log.info("fill: %s %s %s @ %.4f", f.action, f.quantity, f.symbol,
                         f.avg_price)
            if not fills:
                log.info("no fills reported this session")

            positions = broker.positions()
            nlv = broker.net_liquidation()
            log.info("end-of-day positions: %s", positions or "(flat)")
            log.info("NetLiquidation: $%s", f"{nlv:,.0f}")

            # Did the auction actually leave us where the signal wanted? A
            # symbol the trade phase sent that is still not at its target is
            # an unfilled or partially-filled MOC, which is worth seeing in
            # the morning rather than discovering from a P&L discrepancy.
            last = st.load_state(live.state_path).get("last_trade") or {}
            if last.get("status") == "submitted" and not last.get("dry_run"):
                expected = last.get("holdings") or []
                missing = [t for t in expected if positions.get(t, 0) == 0]
                if missing:
                    log.warning("signal wanted %s but the book holds none of: %s",
                                expected, missing)
    except BrokerError as exc:
        log.error("broker problem: %s", exc)
        st.record_run(live.state_path, "reconcile", "error", {"reason": str(exc)})
        return EXIT_CONFIG

    st.record_run(live.state_path, "reconcile", "ok", {
        "n_fills": len(fills),
        "cancelled": cancelled,
        "positions": positions,
        "net_liquidation": round(nlv, 2) if nlv == nlv else None,
    })
    log.info("reconcile OK")
    return EXIT_OK


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["preflight", "trade", "reconcile", "signal"],
                   help="which phase to run ('signal' prints the target book and exits, "
                        "touching no broker)")
    p.add_argument("--config", default=None,
                   help="path to the live JSON config (or set QBS_LIVE_CONFIG)")
    p.add_argument("--notional", type=float, default=None,
                   help="override the configured book size for this run")
    p.add_argument("--dry-run", action="store_true",
                   help="compute and log orders but send nothing")
    p.add_argument("--offline", action="store_true",
                   help="signal phase only: use the CSV cache, no network")
    p.add_argument("--force", action="store_true",
                   help="trade even if the feed has no bar for today (debugging only)")
    p.add_argument("--logfile", default=None, help="also write the run log here")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.logfile)

    try:
        live = LiveConfig.from_env(args.config)
    except (ValueError, OSError) as exc:
        log.error("bad live config: %s", exc)
        return EXIT_CONFIG

    if args.notional is not None:
        live.notional = args.notional
    if args.dry_run:
        live.dry_run = True
    os.makedirs(live.state_dir, exist_ok=True)

    cfg = Config()   # strategy parameters: identical to the backtest's defaults

    if live.dry_run:
        log.warning("DRY RUN: no orders will be sent")

    if args.phase == "signal":
        try:
            _, book = _load_and_compute(cfg, live, refresh=not args.offline,
                                        offline=args.offline)
        except Exception as exc:
            log.error("signal failed: %s: %s", type(exc).__name__, exc)
            return EXIT_ERROR
        print(book.describe())
        return EXIT_OK
    if args.phase == "preflight":
        return phase_preflight(cfg, live)
    if args.phase == "trade":
        return phase_trade(cfg, live, force=args.force)
    return phase_reconcile(cfg, live)


if __name__ == "__main__":
    sys.exit(main())
