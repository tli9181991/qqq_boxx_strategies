"""Daily authorization and the session state machine.

Fail closed, by construction
----------------------------
A persistent `enabled=True` is the wrong shape for permission to spend
money every fifteen minutes: it survives the night, a crash and a reboot,
and the next start obeys it. So the permission is a **one-time token**:

* `authorize()` writes a token for ONE session date and ONE prompt version.
  Only the newest unconsumed token for the date stays live.
* `activate()` -- called once by a starting runner -- CONSUMES it atomically
  (`UPDATE ... WHERE consumed_at IS NULL`). The token is spent before the
  first LLM request is made.
* A crash leaves the token consumed. Restarting finds no live token and stays
  DISABLED. The orphaned session row is marked ERROR with the reason.
* Yesterday's token never matches today's date; a valid prompt alone creates
  no token; `QBS_AGENT_ENABLED` off refuses both authorizing and activating.

The runtime flag the spec calls `agent_enabled` is `sessions.agent_enabled`:
1 while a session is live, reset to 0 at shutdown and by crash recovery.

States: DISABLED -> READY -> RUNNING <-> PAUSED -> STOPPED | ERROR.
STOPPED and ERROR are terminal for that session; every transition is logged
to `state_events` with a reason.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

from . import store
from .config import AgentConfig
from .prompt import ParsedPrompt, parse_prompt
from .session_calendar import is_trading_day, session_bounds, session_date as et_date

log = logging.getLogger(__name__)

DISABLED, READY, RUNNING, PAUSED, STOPPED, ERROR = (
    "DISABLED", "READY", "RUNNING", "PAUSED", "STOPPED", "ERROR")
STATES = (DISABLED, READY, RUNNING, PAUSED, STOPPED, ERROR)
TRANSITIONS = {
    DISABLED: {READY, STOPPED, ERROR},
    READY: {RUNNING, STOPPED, ERROR},
    RUNNING: {PAUSED, STOPPED, ERROR},
    PAUSED: {RUNNING, STOPPED, ERROR},
    STOPPED: set(),
    ERROR: set(),
}
OPEN_STATES = (READY, RUNNING, PAUSED)

# A session whose runner has not written a heartbeat for this long is
# treated as dead by the next runner that starts.
HEARTBEAT_STALE = timedelta(minutes=5)


def _now(now: Optional[datetime]) -> datetime:
    return now or datetime.now(timezone.utc)


def _log_state(conn, session_id: Optional[str], from_state: Optional[str],
               to_state: str, reason: str) -> None:
    store.insert(conn, "state_events", dict(
        session_id=session_id, ts=store.utc_now(), from_state=from_state,
        to_state=to_state, reason=reason))
    log.info("agent state %s -> %s (%s)", from_state, to_state, reason)


# --------------------------------------------------------------------------
# Prompts and authorization -- what the dashboard and CLI call
# --------------------------------------------------------------------------

def save_daily_prompt(cfg: AgentConfig, text: str,
                      now: Optional[datetime] = None) -> Tuple[Optional[int], ParsedPrompt]:
    """Validate and store today's prompt as a new version.

    Storing never authorizes anything. An invalid prompt is not stored.
    """
    day = et_date(_now(now))
    parsed = parse_prompt(text, day)
    if not parsed.ok:
        return None, parsed
    return store.save_prompt(cfg.db_path, day.isoformat(), parsed), parsed


def authorize(cfg: AgentConfig, now: Optional[datetime] = None,
              created_by: str = "user") -> Tuple[bool, str]:
    """Issue today's one-time token for the newest prompt version."""
    now = _now(now)
    day = et_date(now)
    if not cfg.agent_enabled:
        return False, ("the agent feature is off (QBS_AGENT_ENABLED); "
                       "nothing can be authorized")
    errs = cfg.validate()
    if errs:
        return False, "invalid configuration: " + "; ".join(errs)
    if not is_trading_day(day, cfg.extra_holidays):
        return False, f"{day} is not a US trading day"
    row = store.latest_prompt(cfg.db_path, day.isoformat())
    if row is None:
        return False, f"no daily prompt saved for {day}"
    parsed = parse_prompt(row["text"], day)
    if not parsed.ok:
        return False, "the saved prompt no longer validates: " + "; ".join(parsed.errors)
    with store.connect(cfg.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE authorizations SET revoked_at = ?, revoked_reason = ? "
                     "WHERE session_date = ? AND consumed_at IS NULL AND revoked_at IS NULL",
                     (store.utc_now(), "superseded by a newer authorization",
                      day.isoformat()))
        store.insert(conn, "authorizations", dict(
            session_date=day.isoformat(), prompt_version=row["version"],
            prompt_sha256=row["sha256"], created_at=now.isoformat(timespec="seconds"),
            created_by=created_by))
        _log_state(conn, None, None, "AUTHORIZED",
                   f"{day} prompt v{row['version']} authorized by {created_by}")
        conn.execute("COMMIT")
    return True, f"authorized {day}, prompt v{row['version']}"


def revoke(cfg: AgentConfig, reason: str = "revoked by user",
           now: Optional[datetime] = None) -> int:
    """Revoke live tokens for today and ask any open session to stop.

    Returns how many tokens and sessions were affected.
    """
    day = et_date(_now(now)).isoformat()
    with store.connect(cfg.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        n = conn.execute("UPDATE authorizations SET revoked_at = ?, revoked_reason = ? "
                         "WHERE session_date = ? AND revoked_at IS NULL",
                         (store.utc_now(), reason, day)).rowcount
        n += conn.execute("UPDATE sessions SET requested_state = ? WHERE "
                          "closed_at IS NULL", (STOPPED,)).rowcount
        _log_state(conn, None, None, "REVOKED", reason)
        conn.execute("COMMIT")
    return n


def request_state(cfg: AgentConfig, session_id: str, state: str) -> bool:
    """Ask the runner to PAUSE, resume (RUNNING) or STOP between requests."""
    if state not in (PAUSED, RUNNING, STOPPED):
        raise ValueError(f"cannot request {state}")
    with store.connect(cfg.db_path) as conn:
        return conn.execute("UPDATE sessions SET requested_state = ? WHERE "
                            "session_id = ? AND closed_at IS NULL",
                            (state, session_id)).rowcount == 1


def live_authorization(cfg: AgentConfig, day: date):
    return store.one(cfg.db_path,
                     "SELECT * FROM authorizations WHERE session_date = ? AND "
                     "consumed_at IS NULL AND revoked_at IS NULL "
                     "ORDER BY id DESC LIMIT 1", (day.isoformat(),))


def open_session(cfg: AgentConfig):
    return store.one(cfg.db_path, "SELECT * FROM sessions WHERE closed_at IS NULL "
                                  "ORDER BY started_at DESC LIMIT 1")


# --------------------------------------------------------------------------
# The runner's side
# --------------------------------------------------------------------------

def recover_crashed(cfg: AgentConfig, now: Optional[datetime] = None) -> Tuple[int, Optional[str]]:
    """Close sessions a dead process left open. `(n_closed, blocker)`.

    `blocker` is set when an open session still has a fresh heartbeat --
    another runner is alive, and starting a second one would double every
    request.
    """
    now = _now(now)
    closed = 0
    with store.connect(cfg.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for r in conn.execute("SELECT * FROM sessions WHERE closed_at IS NULL").fetchall():
            beat = r["heartbeat_at"] or r["started_at"]
            try:
                age = now - datetime.fromisoformat(beat)
            except (TypeError, ValueError):
                age = HEARTBEAT_STALE * 2
            if age < HEARTBEAT_STALE and r["pid"] != os.getpid():
                conn.execute("ROLLBACK")
                return closed, (f"session {r['session_id']} (pid {r['pid']}) is still "
                                f"alive; refusing to start a second runner")
            conn.execute("UPDATE sessions SET state = ?, agent_enabled = 0, "
                         "closed_at = ?, close_reason = ? WHERE session_id = ?",
                         (ERROR, store.utc_now(),
                          "crash recovery: the process ended without a shutdown; "
                          "its authorization stays consumed", r["session_id"]))
            _log_state(conn, r["session_id"], r["state"], ERROR,
                       "crash recovery on runner start; not resumed")
            closed += 1
        conn.execute("COMMIT")
    return closed, None


@dataclass
class AgentSession:
    cfg: AgentConfig
    session_id: str
    session_date: date
    prompt_version: int
    prompt: ParsedPrompt
    authorization_id: int
    authorization_timestamp: str
    state: str = READY

    def transition(self, to: str, reason: str) -> None:
        if to == self.state:
            return
        if to not in TRANSITIONS[self.state]:
            raise ValueError(f"illegal transition {self.state} -> {to}")
        with store.connect(self.cfg.db_path) as conn:
            fields = "state = ?"
            args = [to]
            if to in (STOPPED, ERROR):
                fields += ", agent_enabled = 0, closed_at = ?, close_reason = ?"
                args += [store.utc_now(), reason]
            conn.execute(f"UPDATE sessions SET {fields} WHERE session_id = ?",
                         (*args, self.session_id))
            _log_state(conn, self.session_id, self.state, to, reason)
        self.state = to

    @property
    def active(self) -> bool:
        return self.state in (READY, RUNNING)

    def heartbeat(self) -> None:
        with store.connect(self.cfg.db_path) as conn:
            conn.execute("UPDATE sessions SET heartbeat_at = ? WHERE session_id = ?",
                         (store.utc_now(), self.session_id))

    def requested(self) -> Optional[str]:
        """A control request from the dashboard/CLI, or a revocation."""
        row = store.one(self.cfg.db_path,
                        "SELECT s.requested_state, a.revoked_at FROM sessions s "
                        "LEFT JOIN authorizations a ON a.id = s.authorization_id "
                        "WHERE s.session_id = ?", (self.session_id,))
        if row is None:
            return STOPPED
        if row["revoked_at"]:
            return STOPPED
        return row["requested_state"]

    def adopt_new_prompt(self) -> Optional[str]:
        """Switch to a newer, explicitly re-authorized prompt version.

        Called between cycles only, so an in-flight decision is never made
        on a half-changed instruction. Returns a description, or None.
        """
        auth = live_authorization(self.cfg, self.session_date)
        if auth is None or auth["prompt_version"] <= self.prompt_version:
            return None
        row = store.get_prompt(self.cfg.db_path, self.session_date.isoformat(),
                               auth["prompt_version"])
        if row is None or row["sha256"] != auth["prompt_sha256"]:
            return None
        parsed = parse_prompt(row["text"], self.session_date)
        if not parsed.ok:
            return None
        with store.connect(self.cfg.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            took = conn.execute("UPDATE authorizations SET consumed_at = ?, "
                                "consumed_by_session = ? WHERE id = ? AND "
                                "consumed_at IS NULL AND revoked_at IS NULL",
                                (store.utc_now(), self.session_id, auth["id"])).rowcount
            if took != 1:
                conn.execute("ROLLBACK")
                return None
            conn.execute("UPDATE sessions SET prompt_version = ?, authorization_id = ?, "
                         "authorization_timestamp = ? WHERE session_id = ?",
                         (auth["prompt_version"], auth["id"], auth["created_at"],
                          self.session_id))
            msg = (f"prompt revalidated: v{self.prompt_version} -> "
                   f"v{auth['prompt_version']}")
            _log_state(conn, self.session_id, self.state, self.state, msg)
            conn.execute("COMMIT")
        self.prompt_version = auth["prompt_version"]
        self.prompt = parsed
        self.authorization_id = auth["id"]
        self.authorization_timestamp = auth["created_at"]
        return msg


def activate(cfg: AgentConfig, now: Optional[datetime] = None
             ) -> Tuple[Optional[AgentSession], str]:
    """Startup: every activation condition, in order. `(session, reason)`.

    Returns `(None, reason)` -- the agent stays DISABLED -- on the first
    condition that fails. Nothing here calls an LLM or fetches data.
    """
    now = _now(now)
    day = et_date(now)

    def _refuse(reason: str):
        with store.connect(cfg.db_path) as conn:
            _log_state(conn, None, None, DISABLED, reason)
        return None, reason

    if not cfg.agent_enabled:
        return None, "agent_enabled is False (QBS_AGENT_ENABLED); the agent stays disabled"
    errs = cfg.validate()
    if errs:
        return _refuse("invalid configuration: " + "; ".join(errs))
    bounds = session_bounds(day, cfg.extra_holidays, cfg.extra_early_closes)
    if bounds is None:
        return _refuse(f"{day} is not a US trading day")
    if now >= bounds[1]:
        return _refuse(f"the {day} session closed at {bounds[1]:%H:%M} ET; the "
                       f"authorization is left unused")
    _, blocker = recover_crashed(cfg, now)
    if blocker:
        return _refuse(blocker)
    auth = live_authorization(cfg, day)
    if auth is None:
        return _refuse(f"no unused authorization for {day}; authorize today's "
                       f"session explicitly (a previous one is never reused)")
    row = store.get_prompt(cfg.db_path, day.isoformat(), auth["prompt_version"])
    if row is None or row["sha256"] != auth["prompt_sha256"]:
        return _refuse("the authorized prompt version is missing or was altered")
    parsed = parse_prompt(row["text"], day)
    if not parsed.ok:
        return _refuse("the authorized prompt does not validate: " + "; ".join(parsed.errors))

    session_id = f"{day.isoformat()}-{uuid.uuid4().hex[:8]}"
    with store.connect(cfg.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        took = conn.execute("UPDATE authorizations SET consumed_at = ?, consumed_by_session = ? "
                            "WHERE id = ? AND consumed_at IS NULL AND revoked_at IS NULL",
                            (store.utc_now(), session_id, auth["id"])).rowcount
        if took != 1:
            conn.execute("ROLLBACK")
            return _refuse("the authorization was consumed or revoked concurrently")
        store.insert(conn, "sessions", dict(
            session_id=session_id, session_date=day.isoformat(),
            prompt_version=auth["prompt_version"], authorization_id=auth["id"],
            authorization_timestamp=auth["created_at"], state=READY,
            agent_enabled=1, started_at=now.isoformat(timespec="seconds"),
            heartbeat_at=now.isoformat(timespec="seconds"),
            provider=cfg.provider, model=cfg.model, execution_mode=cfg.execution_mode,
            config_json=json.dumps(cfg.to_public_dict(), default=str), pid=os.getpid()))
        _log_state(conn, session_id, DISABLED, READY,
                   f"authorization #{auth['id']} consumed; prompt v{auth['prompt_version']}")
        conn.execute("COMMIT")
    return AgentSession(cfg=cfg, session_id=session_id, session_date=day,
                        prompt_version=auth["prompt_version"], prompt=parsed,
                        authorization_id=auth["id"],
                        authorization_timestamp=auth["created_at"]), "ready"
