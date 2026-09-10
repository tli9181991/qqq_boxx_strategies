"""Run state and audit trail.

State here is for *observability*, never for the signal. The strategy weights
are recomputed from price history on every run (see `signals.py`), so nothing
in this file can change what gets traded -- if the state file were deleted the
next run would still produce exactly the right orders. That is intentional: a
state file that fed back into the signal would be a way for one bad run to
poison every run after it.

What it does hold: what happened, when, so that a morning post-mortem does not
require reading systemd logs.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: str, text: str) -> None:
    """Write via a temp file in the same directory, then rename.

    A half-written state file after an instance stop is worse than no state
    file, because it looks valid until it is parsed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"schema": SCHEMA_VERSION, "runs": []}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("state file %s unreadable (%s); starting a fresh one", path, exc)
        return {"schema": SCHEMA_VERSION, "runs": []}
    data.setdefault("schema", SCHEMA_VERSION)
    data.setdefault("runs", [])
    return data


def save_state(path: str, state: Dict[str, Any]) -> None:
    state["updated_at"] = utc_now_iso()
    _atomic_write(path, json.dumps(state, indent=2, default=str))


def record_run(
    path: str,
    phase: str,
    status: str,
    detail: Dict[str, Any],
    keep: int = 400,
) -> Dict[str, Any]:
    """Append one run record, trimming the history so the file stays small."""
    state = load_state(path)
    state["runs"].append({
        "at": utc_now_iso(),
        "phase": phase,
        "status": status,
        **detail,
    })
    state["runs"] = state["runs"][-keep:]
    state["last_" + phase] = {"at": utc_now_iso(), "status": status, **detail}
    save_state(path, state)
    return state


def append_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Append rows to a CSV, writing the header if the file is new."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


ORDER_FIELDS = ["at", "asof", "symbol", "action", "quantity", "price_hint",
                "notional", "reason", "status", "order_id", "dry_run"]
FILL_FIELDS = ["at", "symbol", "action", "quantity", "avg_price", "order_id"]


def log_orders(path: str, asof, orders, statuses: Optional[Dict[str, str]] = None,
               dry_run: bool = False) -> None:
    statuses = statuses or {}
    now = utc_now_iso()
    append_csv(path, [{
        "at": now,
        "asof": f"{asof:%Y-%m-%d}" if hasattr(asof, "strftime") else str(asof),
        "symbol": o.symbol,
        "action": o.action,
        "quantity": o.quantity,
        "price_hint": round(o.price_hint, 4),
        "notional": round(o.notional, 2),
        "reason": o.reason,
        "status": statuses.get(o.symbol, ""),
        "order_id": "",
        "dry_run": dry_run,
    } for o in orders], ORDER_FIELDS)


def log_fills(path: str, fills) -> None:
    now = utc_now_iso()
    append_csv(path, [{
        "at": now,
        "symbol": f.symbol,
        "action": f.action,
        "quantity": f.quantity,
        "avg_price": round(f.avg_price, 4),
        "order_id": f.order_id,
    } for f in fills], FILL_FIELDS)


def kill_switch_engaged(path: str) -> bool:
    return bool(path) and os.path.exists(path)
