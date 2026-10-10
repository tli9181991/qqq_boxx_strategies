"""The session loop: activate, analyse each due candle, shut down at the close.

    python -m qbs.trading_agent run

A separate process from the dashboard, for the reason the dashboard's own
docs give about its fetch gate: Streamlit runs code when someone interacts
with it and at no other time, so a schedule that must fire every fifteen
minutes cannot live there. The dashboard WRITES the prompt, the
authorization and pause/stop requests to the agent database; this loop
READS them between requests.

Lifecycle
---------
start     load config -> `session.activate` (feature switch, config, trading
          day, crash recovery, today's unused authorization, prompt) -> READY
loop      RUNNING: for each due slot not yet analysed, one engine cycle;
          between cycles adopt an explicitly re-authorized prompt, honour
          pause / resume / stop, write a heartbeat
shutdown  at the close, on SIGTERM/SIGINT, on revocation or on error: stop
          new requests (the in-flight one finishes or times out), mark the
          session STOPPED or ERROR, reset `agent_enabled` to False. Every row
          is committed as it is written, so there is nothing left to flush.

Only the newest due slot is analysed on start-up; a runner started at 11:00
does not backfill 09:45-10:45 with decisions nobody could have acted on.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from .config import AgentConfig
from .engine import TradingAgentEngine
from .llm import build_provider
from .market_context import MarketContextBuilder, YFinanceProvider
from .session import ERROR, PAUSED, RUNNING, STOPPED, AgentSession, activate
from .session_calendar import due_slot, next_slot, session_bounds

log = logging.getLogger(__name__)

POLL_S = 15.0       # how often controls are re-read while waiting


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def run_session(cfg: AgentConfig, provider=None, data_provider=None, portfolio=None,
                clock: Callable[[], datetime] = _utcnow,
                sleep: Callable[[float], None] = time.sleep,
                stop_flag: Optional[threading.Event] = None,
                install_signals: bool = True) -> Optional[AgentSession]:
    """Run one session to its end. Returns the session, or None if the agent
    stayed disabled (the reason is logged and recorded in state_events)."""
    session, reason = activate(cfg, clock())
    if session is None:
        log.warning("agent not started: %s", reason)
        return None
    stop = stop_flag or threading.Event()
    if install_signals and threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: stop.set())
            except (ValueError, OSError):  # pragma: no cover - platform specific
                pass

    try:
        provider = provider or build_provider(cfg.provider)
        builder = MarketContextBuilder(data_provider or YFinanceProvider(),
                                       cfg.market_context, cfg.extra_early_closes)
        engine = TradingAgentEngine(cfg, session, provider, builder, portfolio,
                                    sleep=sleep, stop_flag=stop)
    except Exception as exc:  # noqa: BLE001 -- e.g. provider SDK not installed
        session.transition(ERROR, f"could not start: {type(exc).__name__}: {exc}")
        cfg.agent_enabled = False
        return session

    session.transition(RUNNING, "session started")
    end_reason = "session closed"
    done = set()
    kw = dict(interval_minutes=cfg.analysis_interval_minutes,
              delay_seconds=cfg.data_delay_seconds,
              include_close=cfg.include_closing_candle,
              extra_holidays=cfg.extra_holidays,
              extra_early_closes=cfg.extra_early_closes)
    try:
        while True:
            if stop.is_set():
                end_reason = "shutdown signal"
                break
            engine.may_continue()          # applies pause / resume / stop
            if session.state in (STOPPED, ERROR):
                break
            now = clock()
            _, close = session_bounds(session.session_date, cfg.extra_holidays,
                                      cfg.extra_early_closes)
            due = due_slot(now, **kw)
            # At the close the session ends. The one exception is the closing
            # candle itself, when configured, and only while RUNNING: a paused
            # session, or a slot left over from before the pause, must never
            # keep the runner alive (and its heartbeat blocking another one)
            # or be analysed after the bell.
            closing = close if cfg.include_closing_candle else None
            if now >= close and not (closing is not None and session.state == RUNNING
                                     and closing not in done):
                end_reason = "end of regular session"
                break
            if now >= close and due != closing:
                due = None
            if session.state == RUNNING:
                note = session.adopt_new_prompt()
                if note:
                    log.info(note)
                if due is not None and due not in done:
                    done.add(due)
                    engine.run_cycle(due)
                    if session.state in (STOPPED, ERROR):
                        break
            session.heartbeat()
            nxt = next_slot(clock(), **kw)
            wait = POLL_S
            if nxt is not None:
                until = (nxt + timedelta(seconds=cfg.data_delay_seconds) - clock()).total_seconds()
                wait = max(0.5, min(POLL_S, until))
            else:
                wait = max(0.5, min(POLL_S, (close - clock()).total_seconds()))
            sleep(wait)
    except Exception as exc:  # noqa: BLE001 -- fail closed, never resume
        log.exception("agent loop failed")
        if session.state not in (STOPPED, ERROR):
            session.transition(ERROR, f"{type(exc).__name__}: {exc}")
    finally:
        if session.state not in (STOPPED, ERROR):
            if session.state == PAUSED:
                session.transition(STOPPED, f"{end_reason} (while paused)")
            else:
                session.transition(STOPPED, end_reason)
        cfg.agent_enabled = False
    return session
