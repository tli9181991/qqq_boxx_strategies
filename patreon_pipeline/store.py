"""The job queue: one SQLite file holding everything the pipeline remembers.

Why a queue at all
------------------
The email watcher and the downloader run at wildly different speeds. A
notification arrives in milliseconds; the post behind it takes twenty minutes
to fetch and another ten to upload. Doing the work inline in the watcher means
a burst of three posts either downloads three at once -- which is exactly the
traffic pattern Patreon's rate limiting is built to notice -- or blocks the
IMAP connection until it finishes.

So the watcher only ever writes a row. A single worker drains the table. That
one indirection buys serialised downloads, retry with backoff, crash recovery,
and a queue you can inspect with `sqlite3` when something looks wrong.

Idempotency
-----------
`post_url` is UNIQUE, and every path into the queue goes through `enqueue()`,
which is an upsert that reports whether the row was new. This is what makes the
cron sweep safe to run alongside the email trigger: both discover the same
post, the second one is a no-op. Without it the backstop would double every
download.

The canonical URL matters here -- see `mail.canonical_post_url`. Patreon links
the same post as `/posts/some-title-12345678` and `/posts/12345678`, and a
dedupe key that treats those as different is not a dedupe key.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Job states. A job is in exactly one of these at all times.
#   queued  -- waiting for a worker; `next_attempt_at` may hold it back
#   running -- claimed by a worker (reset to queued on worker startup)
#   done    -- uploaded to Drive
#   failed  -- attempts exhausted; needs a human, will not be retried
#   skipped -- deliberately not wanted (no media in the post, or a manual skip)
STATES = ("queued", "running", "done", "failed", "skipped")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    post_url        TEXT    NOT NULL UNIQUE,
    post_id         TEXT,
    source          TEXT    NOT NULL,
    state           TEXT    NOT NULL DEFAULT 'queued',
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    creator         TEXT,
    title           TEXT,
    upload_date     TEXT,
    local_path      TEXT,
    size_bytes      INTEGER,
    drive_file_id   TEXT,
    drive_link      TEXT,
    last_error      TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_state ON jobs(state, next_attempt_at);

-- Where the IMAP watcher got to. Keyed by folder because a second watcher on a
-- different label must not inherit the first one's position.
CREATE TABLE IF NOT EXISTS mail_state (
    folder       TEXT PRIMARY KEY,
    uidvalidity  INTEGER,
    last_uid     INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT    NOT NULL
);

-- Small key/value corner for cross-process signals. `auth_alert` is the one
-- that matters: see worker.py.
CREATE TABLE IF NOT EXISTS flags (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open the database, creating it and its directory on first use.

    WAL because two processes hold this open -- the watcher writing and the
    worker reading-and-writing -- and the default rollback journal makes them
    block each other for the length of a transaction.
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),))
        conn.commit()
        yield conn
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Queue
# ----------------------------------------------------------------------

def enqueue(conn: sqlite3.Connection,
            post_url: str,
            *,
            source: str = "email",
            post_id: Optional[str] = None,
            creator: Optional[str] = None,
            title: Optional[str] = None) -> tuple:
    """Add a post to the queue. Returns (job_id, created).

    `created` is False when the post was already known in any state, including
    `done` and `failed`. Callers use it only for logging: re-discovering a
    finished post is the normal case, not a problem.
    """
    now = utcnow()
    cur = conn.execute(
        """INSERT INTO jobs (post_url, post_id, source, state, creator, title,
                             created_at, updated_at)
           VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
           ON CONFLICT(post_url) DO NOTHING""",
        (post_url, post_id, source, creator, title, now, now))
    if cur.rowcount:
        conn.commit()
        return int(cur.lastrowid), True
    row = conn.execute("SELECT id FROM jobs WHERE post_url = ?",
                       (post_url,)).fetchone()
    return int(row["id"]), False


def claim_next(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Atomically take the oldest due job and mark it running.

    BEGIN IMMEDIATE takes the write lock before the SELECT, so two workers
    cannot both see the same row as queued. There is only meant to be one
    worker, but "meant to be" is not a guarantee when systemd restarts a unit
    that has not finished dying.
    """
    now = utcnow()
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            """SELECT * FROM jobs
               WHERE state = 'queued'
                 AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
               ORDER BY id LIMIT 1""",
            (now,)).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE jobs SET state='running', updated_at=? WHERE id=?",
            (now, row["id"]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()


def reset_running(conn: sqlite3.Connection) -> int:
    """Return orphaned `running` jobs to the queue. Returns how many.

    Called once at worker startup. A job left running is one whose worker died
    mid-download -- the machine rebooted, the unit was killed. Nothing else can
    free it, because the only process that knew about it is gone.
    """
    now = utcnow()
    cur = conn.execute(
        "UPDATE jobs SET state='queued', updated_at=? WHERE state='running'",
        (now,))
    conn.commit()
    return cur.rowcount


def mark_done(conn: sqlite3.Connection, job_id: int, **fields: Any) -> None:
    _update(conn, job_id, state="done", last_error=None, **fields)


def mark_skipped(conn: sqlite3.Connection, job_id: int, reason: str) -> None:
    _update(conn, job_id, state="skipped", last_error=reason)


def mark_retry(conn: sqlite3.Connection,
               job_id: int,
               error: str,
               *,
               max_attempts: int,
               backoff_base: float,
               backoff_cap: float) -> str:
    """Record a failed attempt. Returns the resulting state.

    Exponential backoff, capped. Once `max_attempts` is spent the job goes to
    `failed` and stays there: a post that has failed four times is not going to
    succeed on the fifth, and a queue that retries forever hides the failure
    instead of surfacing it.
    """
    row = conn.execute("SELECT attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
    attempts = int(row["attempts"]) + 1 if row else 1
    if attempts >= max_attempts:
        _update(conn, job_id, state="failed", attempts=attempts, last_error=error)
        return "failed"
    delay = min(backoff_base * (2 ** (attempts - 1)), backoff_cap)
    nxt = (datetime.now(timezone.utc) + timedelta(seconds=delay)
           ).isoformat(timespec="seconds")
    _update(conn, job_id, state="queued", attempts=attempts,
            next_attempt_at=nxt, last_error=error)
    return "queued"


def requeue(conn: sqlite3.Connection, job_id: int) -> None:
    """Put a job back without spending an attempt.

    Used when the failure is not the job's fault -- an expired cookie fails
    every job identically, and burning four attempts on each one just empties
    the queue into `failed` while the real problem goes unfixed.
    """
    _update(conn, job_id, state="queued", next_attempt_at=None)


def _update(conn: sqlite3.Connection, job_id: int, **fields: Any) -> None:
    allowed = {"state", "attempts", "next_attempt_at", "creator", "title",
               "upload_date", "local_path", "size_bytes", "drive_file_id",
               "drive_link", "last_error", "post_id"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"cannot update {sorted(bad)}")
    fields["updated_at"] = utcnow()
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE jobs SET {sets} WHERE id=?",
                 (*fields.values(), job_id))
    conn.commit()


def counts(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state").fetchall()
    out = {s: 0 for s in STATES}
    for r in rows:
        out[r["state"]] = int(r["n"])
    return out


def recent(conn: sqlite3.Connection, limit: int = 20,
           state: Optional[str] = None) -> List[sqlite3.Row]:
    if state:
        return conn.execute(
            "SELECT * FROM jobs WHERE state=? ORDER BY id DESC LIMIT ?",
            (state, limit)).fetchall()
    return conn.execute(
        "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def retry_failed(conn: sqlite3.Connection) -> int:
    """Move every `failed` job back to `queued` with its attempt count reset."""
    cur = conn.execute(
        "UPDATE jobs SET state='queued', attempts=0, next_attempt_at=NULL, "
        "updated_at=? WHERE state='failed'", (utcnow(),))
    conn.commit()
    return cur.rowcount


# ----------------------------------------------------------------------
# Mail cursor
# ----------------------------------------------------------------------

def get_mail_state(conn: sqlite3.Connection, folder: str) -> Dict[str, int]:
    row = conn.execute(
        "SELECT uidvalidity, last_uid FROM mail_state WHERE folder=?",
        (folder,)).fetchone()
    if row is None:
        return {"uidvalidity": 0, "last_uid": 0}
    return {"uidvalidity": int(row["uidvalidity"] or 0),
            "last_uid": int(row["last_uid"] or 0)}


def set_mail_state(conn: sqlite3.Connection, folder: str,
                   uidvalidity: int, last_uid: int) -> None:
    conn.execute(
        """INSERT INTO mail_state (folder, uidvalidity, last_uid, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(folder) DO UPDATE SET
             uidvalidity=excluded.uidvalidity,
             last_uid=excluded.last_uid,
             updated_at=excluded.updated_at""",
        (folder, uidvalidity, last_uid, utcnow()))
    conn.commit()


# ----------------------------------------------------------------------
# Flags
# ----------------------------------------------------------------------

def set_flag(conn: sqlite3.Connection, key: str, value: Optional[str]) -> None:
    if value is None:
        conn.execute("DELETE FROM flags WHERE key=?", (key,))
    else:
        conn.execute(
            """INSERT INTO flags (key, value, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                 value=excluded.value, updated_at=excluded.updated_at""",
            (key, value, utcnow()))
    conn.commit()


def get_flag(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM flags WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None
