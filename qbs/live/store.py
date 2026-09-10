"""The run log: a SQLite database of what the strategy decided, traded and held.

Why a database and not the CSVs this replaces
---------------------------------------------
The questions you actually ask of this data are joins across time -- "what did
we hold the day that name was dropped", "what was the book worth each close
last month", "how often does a name round-trip within a fortnight". Appending
to three CSVs makes every one of those a pandas exercise; a single file with
four tables makes them one query. sqlite3 is in the standard library, needs no
server, and survives the instance stopping every night.

Four tables, two shapes
-----------------------
`trade_events` and nothing else is **append-only**: it is a log of things that
happened, and a phase that ran twice really did submit twice. The daily
snapshots -- `selection_events`, `position_closes`, `portfolio_nav`,
`signal_runs` -- are **upserted on their natural key**, because re-running
reconcile for the same session must correct the row rather than double it.

Nothing here feeds the signal. As with `state.py`, deleting this database
changes what you can see, never what gets traded.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT    NOT NULL,
    session_date  TEXT    NOT NULL,
    phase         TEXT    NOT NULL,
    event         TEXT    NOT NULL,
    symbol        TEXT,
    action        TEXT,
    quantity      REAL,
    price         REAL,
    notional      REAL,
    order_id      INTEGER,
    status        TEXT,
    reason        TEXT,
    dry_run       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_trade_events_date   ON trade_events(session_date);
CREATE INDEX IF NOT EXISTS ix_trade_events_symbol ON trade_events(symbol);

CREATE TABLE IF NOT EXISTS selection_events (
    session_date  TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    event         TEXT NOT NULL,
    rank          INTEGER,
    score         REAL,
    hurdle        REAL,
    reason        TEXT,
    ts_utc        TEXT NOT NULL,
    PRIMARY KEY (session_date, symbol, event)
);
CREATE INDEX IF NOT EXISTS ix_selection_symbol ON selection_events(symbol);

CREATE TABLE IF NOT EXISTS position_closes (
    session_date   TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    shares         REAL,
    close_price    REAL,
    market_value   REAL,
    avg_cost       REAL,
    unrealized_pnl REAL,
    target_weight  REAL,
    actual_weight  REAL,
    source         TEXT,
    ts_utc         TEXT NOT NULL,
    PRIMARY KEY (session_date, symbol)
);
CREATE INDEX IF NOT EXISTS ix_position_closes_symbol ON position_closes(symbol);

CREATE TABLE IF NOT EXISTS portfolio_nav (
    session_date       TEXT PRIMARY KEY,
    total_market_value REAL,
    net_liquidation    REAL,
    cash               REAL,
    n_positions        INTEGER,
    risk_weight        REAL,
    scalar             REAL,
    book_vol           REAL,
    unrealized_pnl     REAL,
    source             TEXT,
    ts_utc             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_runs (
    session_date   TEXT NOT NULL,
    phase          TEXT NOT NULL,
    asof           TEXT,
    scalar         REAL,
    book_vol       REAL,
    risk_weight    REAL,
    n_rankable     INTEGER,
    universe_size  INTEGER,
    holdings       TEXT,
    status         TEXT,
    reason         TEXT,
    ts_utc         TEXT NOT NULL,
    PRIMARY KEY (session_date, phase)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _day(value: Any) -> str:
    """Normalise anything date-like to YYYY-MM-DD."""
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


@contextmanager
def connect(path: str):
    """Open the database, creating and migrating it if needed.

    WAL plus a busy timeout: the phases never overlap by design, but a manual
    `sqlite3` session left open while a timer fires should block briefly rather
    than fail the run.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        _migrate(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        log.info("initialised run-log schema v%d", SCHEMA_VERSION)
    elif version > SCHEMA_VERSION:
        raise RuntimeError(
            f"run log is schema v{version} but this code understands v{SCHEMA_VERSION}. "
            "Upgrade the code rather than downgrading the database.")
    else:
        # Idempotent: every statement is CREATE ... IF NOT EXISTS, so this also
        # adds tables introduced by a later version to an older file.
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

def log_trade_events(path: str, rows: Sequence[Dict[str, Any]]) -> int:
    """Append to the trading event log. Never deduplicates -- it is a log."""
    if not rows:
        return 0
    now = utc_now()
    payload = [(
        r.get("ts_utc") or now,
        _day(r.get("session_date")),
        r.get("phase", ""),
        r.get("event", ""),
        r.get("symbol"),
        r.get("action"),
        r.get("quantity"),
        r.get("price"),
        r.get("notional"),
        r.get("order_id"),
        r.get("status"),
        r.get("reason"),
        int(bool(r.get("dry_run", False))),
    ) for r in rows]
    with connect(path) as conn:
        conn.executemany(
            "INSERT INTO trade_events (ts_utc, session_date, phase, event, symbol, "
            "action, quantity, price, notional, order_id, status, reason, dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", payload)
    return len(payload)


def log_orders(path: str, session_date, phase: str, orders: Iterable,
               statuses: Optional[Dict[str, str]] = None,
               dry_run: bool = False, event: str = "submitted") -> int:
    """Record an order list as trade events."""
    statuses = statuses or {}
    return log_trade_events(path, [{
        "session_date": session_date,
        "phase": phase,
        "event": event,
        "symbol": o.symbol,
        "action": o.action,
        "quantity": o.quantity,
        "price": o.price_hint,
        "notional": o.notional,
        "status": statuses.get(o.symbol),
        "reason": o.reason,
        "dry_run": dry_run,
    } for o in orders])


def log_fills(path: str, session_date, fills: Iterable, phase: str = "reconcile") -> int:
    return log_trade_events(path, [{
        "session_date": session_date,
        "phase": phase,
        "event": "filled",
        "symbol": f.symbol,
        "action": f.action,
        "quantity": f.quantity,
        "price": f.avg_price,
        "notional": (f.quantity or 0) * (f.avg_price or 0),
        "order_id": f.order_id,
        "status": f.status,
    } for f in fills])


def log_note(path: str, session_date, phase: str, event: str, reason: str,
             symbol: Optional[str] = None, dry_run: bool = False) -> int:
    """One non-order event: a guard trip, a skipped session, a cancellation."""
    return log_trade_events(path, [{
        "session_date": session_date, "phase": phase, "event": event,
        "symbol": symbol, "reason": reason, "dry_run": dry_run,
    }])


def log_selection(path: str, session_date, rows: Sequence[Dict[str, Any]]) -> int:
    """Upsert today's stock-selection decisions.

    Keyed on (date, symbol, event) so recomputing the same session overwrites
    rather than duplicating -- the ranking is deterministic given the prices,
    so a second run for the same day should leave the table unchanged.
    """
    if not rows:
        return 0
    now = utc_now()
    payload = [(
        _day(session_date), r["symbol"], r["event"],
        r.get("rank"), r.get("score"), r.get("hurdle"), r.get("reason"), now,
    ) for r in rows]
    with connect(path) as conn:
        conn.executemany(
            "INSERT INTO selection_events (session_date, symbol, event, rank, score, "
            "hurdle, reason, ts_utc) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(session_date, symbol, event) DO UPDATE SET "
            "rank=excluded.rank, score=excluded.score, hurdle=excluded.hurdle, "
            "reason=excluded.reason, ts_utc=excluded.ts_utc", payload)
    return len(payload)


def log_position_closes(path: str, session_date, rows: Sequence[Dict[str, Any]]) -> int:
    """Upsert the end-of-day mark for every position held."""
    if not rows:
        return 0
    now = utc_now()
    payload = [(
        _day(session_date), r["symbol"], r.get("shares"), r.get("close_price"),
        r.get("market_value"), r.get("avg_cost"), r.get("unrealized_pnl"),
        r.get("target_weight"), r.get("actual_weight"), r.get("source", "ib"), now,
    ) for r in rows]
    with connect(path) as conn:
        conn.executemany(
            "INSERT INTO position_closes (session_date, symbol, shares, close_price, "
            "market_value, avg_cost, unrealized_pnl, target_weight, actual_weight, "
            "source, ts_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(session_date, symbol) DO UPDATE SET "
            "shares=excluded.shares, close_price=excluded.close_price, "
            "market_value=excluded.market_value, avg_cost=excluded.avg_cost, "
            "unrealized_pnl=excluded.unrealized_pnl, "
            "target_weight=excluded.target_weight, "
            "actual_weight=excluded.actual_weight, source=excluded.source, "
            "ts_utc=excluded.ts_utc", payload)
    return len(payload)


def log_portfolio_nav(path: str, session_date, **fields: Any) -> None:
    """Upsert one row of book-level end-of-day state."""
    cols = ["total_market_value", "net_liquidation", "cash", "n_positions",
            "risk_weight", "scalar", "book_vol", "unrealized_pnl", "source"]
    values = [_day(session_date)] + [fields.get(c) for c in cols] + [utc_now()]
    sets = ", ".join(f"{c}=excluded.{c}" for c in cols) + ", ts_utc=excluded.ts_utc"
    with connect(path) as conn:
        conn.execute(
            f"INSERT INTO portfolio_nav (session_date, {', '.join(cols)}, ts_utc) "
            f"VALUES ({','.join('?' * (len(cols) + 2))}) "
            f"ON CONFLICT(session_date) DO UPDATE SET {sets}", values)


def log_signal_run(path: str, session_date, phase: str, book=None,
                   status: str = "ok", reason: str = "") -> None:
    """Upsert what the signal said this session, per phase.

    Keyed by phase as well as date so the morning preflight and the afternoon
    trade run both keep their row: comparing them shows how much the book moved
    between 08:50 and the auction.
    """
    fields = dict(asof=None, scalar=None, book_vol=None, risk_weight=None,
                  n_rankable=None, universe_size=None, holdings=None)
    if book is not None:
        fields.update(
            asof=_day(book.asof), scalar=book.scalar, book_vol=book.book_vol,
            risk_weight=book.risk_weight, n_rankable=book.n_rankable,
            universe_size=book.universe_size,
            holdings=",".join(book.raw_holdings))
    cols = list(fields) + ["status", "reason"]
    values = ([_day(session_date), phase] + list(fields.values())
              + [status, reason, utc_now()])
    sets = ", ".join(f"{c}=excluded.{c}" for c in cols) + ", ts_utc=excluded.ts_utc"
    with connect(path) as conn:
        conn.execute(
            f"INSERT INTO signal_runs (session_date, phase, {', '.join(cols)}, ts_utc) "
            f"VALUES ({','.join('?' * (len(cols) + 3))}) "
            f"ON CONFLICT(session_date, phase) DO UPDATE SET {sets}", values)


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------

def _rows(path: str, sql: str, args: Sequence = ()) -> List[Dict[str, Any]]:
    with connect(path) as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def recent_trades(path: str, limit: int = 50) -> List[Dict[str, Any]]:
    return _rows(path, "SELECT * FROM trade_events ORDER BY id DESC LIMIT ?", (limit,))


def trades_on(path: str, session_date) -> List[Dict[str, Any]]:
    return _rows(path, "SELECT * FROM trade_events WHERE session_date=? ORDER BY id",
                 (_day(session_date),))


def selection_history(path: str, limit: int = 100) -> List[Dict[str, Any]]:
    return _rows(path, "SELECT * FROM selection_events "
                       "ORDER BY session_date DESC, event, rank LIMIT ?", (limit,))


def nav_history(path: str, limit: int = 250) -> List[Dict[str, Any]]:
    rows = _rows(path, "SELECT * FROM portfolio_nav ORDER BY session_date DESC LIMIT ?",
                 (limit,))
    return list(reversed(rows))


def closes_on(path: str, session_date) -> List[Dict[str, Any]]:
    return _rows(path, "SELECT * FROM position_closes WHERE session_date=? "
                       "ORDER BY market_value DESC", (_day(session_date),))


def holding_periods(path: str) -> List[Dict[str, Any]]:
    """Each name's entries and exits, newest first -- the round-trip view."""
    return _rows(path,
                 "SELECT symbol, session_date, event, rank, score FROM selection_events "
                 "WHERE event IN ('entry','exit') ORDER BY symbol, session_date")


def to_frame(path: str, table: str):
    """The whole table as a DataFrame, for the notebook."""
    import pandas as pd
    allowed = {"trade_events", "selection_events", "position_closes",
               "portfolio_nav", "signal_runs"}
    if table not in allowed:
        raise ValueError(f"unknown table {table!r}; expected one of {sorted(allowed)}")
    with connect(path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def summary(path: str) -> Dict[str, int]:
    with connect(path) as conn:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("trade_events", "selection_events", "position_closes",
                          "portfolio_nav", "signal_runs")}
