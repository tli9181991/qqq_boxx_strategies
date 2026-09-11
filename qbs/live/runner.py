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
import tempfile
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
from qbs.live import store

log = logging.getLogger("qbs.live")

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_GUARD = 0, 1, 2, 3


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def check_state_dir(state_dir: str) -> Optional[str]:
    """Return an actionable message if the state directory is unusable.

    Every phase writes here: the state file, the run-log database, the fill
    records. When it is a bind mount owned by a different uid than the
    container runs as, the first write fails deep inside whatever was already
    happening -- often inside an *error handler*, where a PermissionError
    traceback then buries the problem the handler was reporting. Probing up
    front costs one file create and turns that into one sentence.
    """
    try:
        os.makedirs(state_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=state_dir, suffix=".probe")
        os.close(fd)
        os.unlink(tmp)
        return None
    except OSError as exc:
        return (
            f"state directory {state_dir} is not writable as uid {os.getuid()}:"
            f"{os.getgid()} ({type(exc).__name__}: {exc}). Every phase records "
            "its run and its orders here, so nothing can run until this is "
            "fixed. Under Docker this path is a bind mount of var/ in the "
            "checkout and the container runs as QBS_UID:QBS_GID from .env, so "
            "on the host: sudo chown -R $(id -u):$(id -g) var data -- and check "
            "that QBS_UID and QBS_GID match your own id -u and id -g.")


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

    session = market_today(live.market_tz).date()
    store.log_signal_run(live.db_path, session, "preflight", book)
    store.log_selection(live.db_path, book.asof, book.selection)
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
        store.log_signal_run(live.db_path, market_today(live.market_tz).date(),
                             "trade", book, "skipped", "no bar for today")
        st.record_run(live.state_path, "trade", "skipped",
                      {"reason": "no bar for today", "asof": f"{book.asof:%Y-%m-%d}"})
        return EXIT_OK

    log.info("target book:\n%s", book.describe())

    session = market_today(live.market_tz).date()
    # Logged before anything is sent: the selection is what the strategy
    # decided, and it stays true whether or not the orders make it out.
    store.log_signal_run(live.db_path, session, "trade", book)
    n_sel = store.log_selection(live.db_path, book.asof, book.selection)
    log.info("logged %d selection events for %s", n_sel, book.asof.date())

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
                store.log_note(live.db_path, session, "trade", "no_change",
                               "live book already matches the target")
                st.record_run(live.state_path, "trade", "flat", {
                    "asof": f"{book.asof:%Y-%m-%d}", "n_orders": 0,
                    "scalar": round(book.scalar, 4)})
                return EXIT_OK

            results = broker.submit_moc(orders)
            statuses = {r.symbol: r.status for r in results}
            store.log_orders(live.db_path, session, "trade", orders, statuses,
                             dry_run=live.dry_run)

    except GuardTripped as exc:
        log.error("GUARD TRIPPED, nothing sent: %s", exc)
        # A refused session is a trading event: without a row here the log
        # shows a day that simply did nothing, with no record of why.
        store.log_note(live.db_path, session, "trade", "guard_tripped", str(exc))
        store.log_signal_run(live.db_path, session, "trade", book, "guard", str(exc))
        st.record_run(live.state_path, "trade", "guard", {"reason": str(exc)})
        return EXIT_GUARD
    except BrokerError as exc:
        log.error("broker problem: %s", exc)
        store.log_note(live.db_path, session, "trade", "error", str(exc))
        store.log_signal_run(live.db_path, session, "trade", book, "error", str(exc))
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
    """After the close: what filled, what the book is worth, cancel any straggler.

    This is the phase that writes the end-of-day marks, so it is the one that
    builds the equity curve. Prices come from IB's own portfolio marks rather
    than a re-fetch: at 16:15 the official close may not have reached a free
    feed yet, and a mark that disagrees with the account it describes is worse
    than no mark.
    """
    log.info("=== RECONCILE ===")
    session = market_today(live.market_tz).date()

    from qbs.live.broker import BrokerError, IBBroker
    try:
        with IBBroker(live) as broker:
            cancelled = broker.cancel_all_open()
            if cancelled:
                log.warning("cancelled %d order(s) still open after the close", cancelled)
                store.log_note(live.db_path, session, "reconcile", "cancelled",
                               f"{cancelled} order(s) still open after the close")

            fills = broker.todays_fills()
            store.log_fills(live.db_path, session, fills)
            for f in fills:
                log.info("fill: %s %s %s @ %.4f", f.action, f.quantity, f.symbol,
                         f.avg_price)
            if not fills:
                log.info("no fills reported this session")

            marks = broker.portfolio_marks()
            positions = broker.positions()
            nlv = broker.net_liquidation()
            cash = broker.cash_balance()

            total_mv = sum(m["market_value"] for m in marks)
            total_pnl = sum(m["unrealized_pnl"] for m in marks
                            if m["unrealized_pnl"] == m["unrealized_pnl"])

            # Actual weights are measured against the book we intended to run,
            # not against NLV: the whole point of a fixed notional is that
            # target and actual are comparable on the same denominator.
            denom = live.notional or total_mv or 1.0
            last = st.load_state(live.state_path).get("last_trade") or {}
            for m in marks:
                m["actual_weight"] = m["market_value"] / denom
            store.log_position_closes(live.db_path, session, marks)

            risk_mv = sum(m["market_value"] for m in marks
                          if m["symbol"] != live.safe_asset)
            store.log_portfolio_nav(
                live.db_path, session,
                total_market_value=total_mv,
                net_liquidation=None if nlv != nlv else nlv,
                cash=None if cash != cash else cash,
                n_positions=len(marks),
                risk_weight=risk_mv / denom,
                scalar=last.get("scalar"),
                book_vol=last.get("book_vol"),
                unrealized_pnl=total_pnl,
                source="ib",
            )

            log.info("end-of-day positions: %s", positions or "(flat)")
            log.info("marked book $%s across %d positions (risk %.0f%%), "
                     "NetLiquidation $%s",
                     f"{total_mv:,.0f}", len(marks), 100 * risk_mv / denom,
                     f"{nlv:,.0f}")

            # Did the auction leave us where the signal wanted? A symbol the
            # trade phase sent that is still not held is an unfilled MOC, worth
            # seeing in the morning rather than from a P&L discrepancy.
            if last.get("status") == "submitted" and not last.get("dry_run"):
                expected = last.get("holdings") or []
                missing = [t for t in expected if positions.get(t, 0) == 0]
                if missing:
                    log.warning("signal wanted %s but the book holds none of: %s",
                                expected, missing)
                    store.log_note(live.db_path, session, "reconcile", "unfilled",
                                   f"no position in {', '.join(missing)}")
    except BrokerError as exc:
        log.error("broker problem: %s", exc)
        store.log_note(live.db_path, session, "reconcile", "error", str(exc))
        st.record_run(live.state_path, "reconcile", "error", {"reason": str(exc)})
        return EXIT_CONFIG

    st.record_run(live.state_path, "reconcile", "ok", {
        "n_fills": len(fills),
        "cancelled": cancelled,
        "positions": positions,
        "net_liquidation": round(nlv, 2) if nlv == nlv else None,
        "total_market_value": round(total_mv, 2),
    })
    log.info("reconcile OK")
    return EXIT_OK


def phase_report(live: LiveConfig, days: int = 10) -> int:
    """Print the run log. No broker, no network -- just the tables."""
    counts = store.summary(live.db_path)
    print(f"\nrun log: {live.db_path}")
    print("  " + "   ".join(f"{k}={v}" for k, v in counts.items()) + "\n")

    nav = store.nav_history(live.db_path, limit=days)
    if nav:
        print("PORTFOLIO CLOSES")
        print(f"  {'date':<12}{'market value':>14}{'NAV':>14}{'risk':>7}"
              f"{'scalar':>8}{'positions':>11}{'unreal P&L':>13}")
        for r in nav:
            print(f"  {r['session_date']:<12}"
                  f"{(r['total_market_value'] or 0):>14,.0f}"
                  f"{(r['net_liquidation'] or 0):>14,.0f}"
                  f"{(r['risk_weight'] or 0):>7.0%}"
                  f"{(r['scalar'] or 0):>8.2f}"
                  f"{(r['n_positions'] or 0):>11}"
                  f"{(r['unrealized_pnl'] or 0):>13,.0f}")
        print()

    sel = store.selection_history(live.db_path, limit=40)
    if sel:
        print("STOCK SELECTION")
        print(f"  {'date':<12}{'symbol':<8}{'event':<7}{'rank':>6}{'12-1 mom':>11}"
              f"  reason")
        for r in sel:
            rank = "" if r["rank"] is None else f"{r['rank']:.0f}"
            score = "" if r["score"] is None else f"{r['score']:+.1%}"
            print(f"  {r['session_date']:<12}{r['symbol']:<8}{r['event']:<7}"
                  f"{rank:>6}{score:>11}  {r['reason'] or ''}")
        print()

    trades = store.recent_trades(live.db_path, limit=40)
    if trades:
        print("TRADING EVENTS (most recent first)")
        print(f"  {'date':<12}{'phase':<11}{'event':<14}{'symbol':<8}{'side':<6}"
              f"{'qty':>8}{'price':>11}{'notional':>12}")
        for r in trades:
            print(f"  {r['session_date']:<12}{r['phase']:<11}{r['event']:<14}"
                  f"{(r['symbol'] or ''):<8}{(r['action'] or ''):<6}"
                  f"{(r['quantity'] or 0):>8.0f}"
                  f"{(r['price'] or 0):>11,.2f}{(r['notional'] or 0):>12,.0f}"
                  + (f"  {r['reason']}" if r["event"] in
                     ("guard_tripped", "error", "unfilled", "no_change") else ""))
        print()
    if not any((nav, sel, trades)):
        print("(the run log is empty -- no phase has written to it yet)")
    return EXIT_OK


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase",
                   choices=["preflight", "trade", "reconcile", "signal", "report"],
                   help="which phase to run. 'signal' prints the target book and "
                        "exits, touching no broker; 'report' prints the run log "
                        "and touches neither broker nor network")
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
    p.add_argument("--days", type=int, default=10,
                   help="report: how many sessions of closes to show")
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
    problem = check_state_dir(live.state_dir)
    if problem:
        log.error("%s", problem)
        return EXIT_CONFIG

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
    if args.phase == "report":
        return phase_report(live, days=args.days)
    if args.phase == "preflight":
        return phase_preflight(cfg, live)
    if args.phase == "trade":
        return phase_trade(cfg, live, force=args.force)
    return phase_reconcile(cfg, live)


if __name__ == "__main__":
    sys.exit(main())
