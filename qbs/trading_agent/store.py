"""The agent's audit log: one SQLite file, separate from the live run log.

Separate on purpose. `qbs.live.store` is the trading host's record of what
was ORDERED; a dry-run recommendation written next to real fills is one bad
query away from being read as one. This file lives at var/agent/agent.db
(gitignored, like var/ already is) and nothing outside this package opens it.

Tables
------
prompts         every Daily User Prompt, versioned per session date
authorizations  one-time daily tokens: created, consumed, revoked
sessions        one row per runner session, its state and runtime flag
state_events    every state transition, with a reason
cycles          (session_date, symbol, candle) -- UNIQUE, which is what
                makes a duplicate analysis impossible, not just unlikely
llm_requests    one row per attempted analysis: tokens, cost, latency,
                retries, validation, error
decisions       the validated (or rejected) output for that request
snapshots       the normalized market context the decision was made on
memory_plans    layer 1 of the trading memory: the authorized daily plan
memory_symbol_state  layer 2: one row per (session date, symbol)
memory_events   every proposed memory update, ACCEPTED or REJECTED, and why
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date    TEXT NOT NULL,
    version         INTEGER NOT NULL,
    text            TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    symbols_json    TEXT NOT NULL,
    constraints_json TEXT NOT NULL,
    warnings_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (session_date, version)
);
CREATE TABLE IF NOT EXISTS authorizations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date        TEXT NOT NULL,
    prompt_version      INTEGER NOT NULL,
    prompt_sha256       TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    created_by          TEXT,
    consumed_at         TEXT,
    consumed_by_session TEXT,
    revoked_at          TEXT,
    revoked_reason      TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT PRIMARY KEY,
    session_date            TEXT NOT NULL,
    prompt_version          INTEGER,
    authorization_id        INTEGER,
    authorization_timestamp TEXT,
    state                   TEXT NOT NULL,
    agent_enabled           INTEGER NOT NULL DEFAULT 0,
    requested_state         TEXT,
    started_at              TEXT NOT NULL,
    heartbeat_at            TEXT,
    closed_at               TEXT,
    close_reason            TEXT,
    provider                TEXT,
    model                   TEXT,
    execution_mode          TEXT NOT NULL DEFAULT 'DRY_RUN',
    config_json             TEXT,
    pid                     INTEGER
);
CREATE TABLE IF NOT EXISTS state_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    ts          TEXT NOT NULL,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    reason      TEXT
);
CREATE TABLE IF NOT EXISTS cycles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    session_date  TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    candle_ts     TEXT NOT NULL,
    status        TEXT NOT NULL,
    detail        TEXT,
    data_latency_ms REAL,
    created_at    TEXT NOT NULL,
    finished_at   TEXT,
    UNIQUE (session_date, symbol, candle_ts)
);
CREATE TABLE IF NOT EXISTS llm_requests (
    request_id            TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL,
    session_date          TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    candle_ts             TEXT NOT NULL,
    started_at            TEXT NOT NULL,
    finished_at           TEXT,
    provider              TEXT,
    model                 TEXT,
    prompt_version        INTEGER,
    system_prompt_version TEXT,
    context_version       TEXT,
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    cached_input_tokens   INTEGER,
    reasoning_tokens      INTEGER,
    total_tokens          INTEGER,
    memory_chars          INTEGER,
    memory_tokens_estimated INTEGER,
    memory_input_tokens_attributed INTEGER,
    usage_source          TEXT,
    estimated_input_tokens INTEGER,
    input_cost            REAL,
    output_cost           REAL,
    total_cost            REAL,
    cost_status           TEXT,
    latency_ms            REAL,
    data_latency_ms       REAL,
    success               INTEGER NOT NULL DEFAULT 0,
    retry_count           INTEGER NOT NULL DEFAULT 0,
    timeout_events        INTEGER NOT NULL DEFAULT 0,
    validation_status     TEXT,
    action                TEXT,
    error                 TEXT,
    prompt_text           TEXT,
    response_text         TEXT
);
CREATE INDEX IF NOT EXISTS ix_req_session ON llm_requests(session_id);
CREATE INDEX IF NOT EXISTS ix_req_symbol  ON llm_requests(symbol);
CREATE TABLE IF NOT EXISTS decisions (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id             TEXT,
    session_id             TEXT NOT NULL,
    session_date           TEXT NOT NULL,
    symbol                 TEXT NOT NULL,
    candle_ts              TEXT NOT NULL,
    status                 TEXT NOT NULL,
    action                 TEXT,
    strategy               TEXT,
    confidence             REAL,
    market_condition       TEXT,
    entry_price            REAL,
    stop_loss              REAL,
    take_profit            REAL,
    suggested_quantity     INTEGER,
    reason                 TEXT,
    invalidation_condition TEXT,
    risk_flags_json        TEXT,
    errors_json            TEXT,
    warnings_json          TEXT,
    last_price             REAL,
    created_at             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_dec_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS ix_dec_symbol  ON decisions(symbol, candle_ts);
CREATE TABLE IF NOT EXISTS memory_plans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    prompt_version  INTEGER NOT NULL,
    plan_json       TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    UNIQUE (session_id, prompt_version)
);
CREATE TABLE IF NOT EXISTS memory_symbol_state (
    session_date          TEXT NOT NULL,
    symbol                TEXT NOT NULL,
    session_id            TEXT NOT NULL,
    prompt_version        INTEGER NOT NULL,
    plan_sha256           TEXT NOT NULL,
    state                 TEXT NOT NULL,
    strategy              TEXT,
    state_since           TEXT,
    candle_ts             TEXT,
    stop_reference        REAL,
    invalidation_condition TEXT,
    key_levels_json       TEXT NOT NULL DEFAULT '[]',
    facts_json            TEXT NOT NULL DEFAULT '{}',
    interpretations_json  TEXT NOT NULL DEFAULT '[]',
    last_decision_json    TEXT,
    revision              INTEGER NOT NULL DEFAULT 0,
    updated_at            TEXT NOT NULL,
    PRIMARY KEY (session_date, symbol)
);
CREATE TABLE IF NOT EXISTS memory_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    session_date    TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    candle_ts       TEXT,
    prompt_version  INTEGER,
    request_id      TEXT,
    source          TEXT NOT NULL,
    status          TEXT NOT NULL,
    from_state      TEXT,
    to_state        TEXT,
    reasons_json    TEXT
);
CREATE INDEX IF NOT EXISTS ix_mem_events ON memory_events(session_date, symbol);
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    candle_ts       TEXT NOT NULL,
    request_id      TEXT,
    context_version TEXT,
    data_source     TEXT,
    freshness       TEXT,
    snapshot_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: str) -> Iterator[sqlite3.Connection]:
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        # WAL so the dashboard can read while the runner writes.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        yield conn
    finally:
        conn.close()


def rows(path: str, sql: str, args: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    if path != ":memory:" and not os.path.exists(path):
        return []
    with connect(path) as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def one(path: str, sql: str, args: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
    out = rows(path, sql, args)
    return out[0] if out else None


def insert(conn: sqlite3.Connection, table: str, values: Dict[str, Any]) -> int:
    cols = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cur = conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})",
                       tuple(values.values()))
    return int(cur.lastrowid)


def dumps(v: Any) -> str:
    return json.dumps(v, default=str, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def save_prompt(path: str, session_date: str, parsed) -> int:
    """Store a prompt as the next version for `session_date`. Returns it."""
    with connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute("SELECT COALESCE(MAX(version), 0) FROM prompts "
                               "WHERE session_date = ?", (session_date,))
            version = int(cur.fetchone()[0]) + 1
            insert(conn, "prompts", dict(
                session_date=session_date, version=version, text=parsed.text,
                sha256=parsed.sha256, symbols_json=dumps(parsed.symbols),
                constraints_json=dumps(parsed.constraints_dict()),
                warnings_json=dumps(parsed.warnings), created_at=utc_now()))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return version


def latest_prompt(path: str, session_date: str) -> Optional[Dict[str, Any]]:
    return one(path, "SELECT * FROM prompts WHERE session_date = ? "
                     "ORDER BY version DESC LIMIT 1", (session_date,))


def get_prompt(path: str, session_date: str, version: int) -> Optional[Dict[str, Any]]:
    return one(path, "SELECT * FROM prompts WHERE session_date = ? AND version = ?",
               (session_date, version))


# --------------------------------------------------------------------------
# Reads for the dashboard and the evaluation
# --------------------------------------------------------------------------

def recent_decisions(path: str, limit: int = 200, session_id: Optional[str] = None,
                     symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    where, args = [], []
    if session_id:
        where.append("session_id = ?")
        args.append(session_id)
    if symbol:
        where.append("symbol = ?")
        args.append(symbol)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    return rows(path, f"SELECT * FROM decisions {clause} "
                      f"ORDER BY candle_ts DESC, id DESC LIMIT ?", (*args, limit))


def requests_for(path: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if session_id:
        return rows(path, "SELECT * FROM llm_requests WHERE session_id = ? "
                          "ORDER BY started_at", (session_id,))
    return rows(path, "SELECT * FROM llm_requests ORDER BY started_at")


def sessions(path: str, limit: int = 50) -> List[Dict[str, Any]]:
    return rows(path, "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))


def state_events(path: str, session_id: Optional[str] = None,
                 limit: int = 100) -> List[Dict[str, Any]]:
    if session_id:
        return rows(path, "SELECT * FROM state_events WHERE session_id = ? "
                          "ORDER BY id DESC LIMIT ?", (session_id, limit))
    return rows(path, "SELECT * FROM state_events ORDER BY id DESC LIMIT ?", (limit,))
