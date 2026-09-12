"""Run state and audit trail.

State here is for *observability*, never for the signal. The strategy weights
are recomputed from price history on every run (see `signals.py`), so nothing
in this file can change what gets traded -- if the state file were deleted the
next run would still produce exactly the right orders. That is intentional: a
state file that fed back into the signal would be a way for one bad run to
poison every run after it.

What it does hold: a small rolling record of run outcomes, enough for a phase
to see what the previous one did. The *analytical* record -- every trade,
selection and end-of-day mark -- lives in the SQLite run log (`store.py`),
which is the thing to query when you want to know what the strategy has been
doing rather than whether last night's run succeeded.
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


def kill_switch_engaged(path: str) -> bool:
    return bool(path) and os.path.exists(path)


# --------------------------------------------------------------------------
# External holdings baseline
#
# The one piece of state that *does* affect orders, and deliberately so: when
# the strategy shares an IB account with positions you manage yourself, there
# is no way to tell them apart at the broker. This records which shares are
# yours so the strategy can subtract them.
#
# It is a fixed snapshot, not a running tally. The strategy never writes to it
# when it trades -- its own position is always derived as (account - baseline),
# with the broker's number authoritative. That is what keeps a missed fill or a
# deleted run log from corrupting it, and it is why this does not reintroduce
# the feedback loop the module docstring warns about.
# --------------------------------------------------------------------------

def load_external_positions(path: str) -> Dict[str, int]:
    """Shares held outside the strategy. Missing file means an empty account."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"could not read the external-holdings baseline at {path}: {exc}. "
            "Refusing to guess -- an unreadable baseline means the strategy "
            "cannot tell its own shares from yours.") from exc

    raw = data.get("positions", data)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of symbol -> shares")
    out: Dict[str, int] = {}
    for sym, qty in raw.items():
        try:
            n = int(qty)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {sym!r} has non-integer shares {qty!r}") from exc
        if n < 0:
            raise ValueError(f"{path}: {sym!r} has negative shares {n}")
        if n:
            out[str(sym).upper()] = n
    return out


def save_external_positions(path: str, positions: Dict[str, int],
                            note: str = "") -> None:
    payload = {
        "captured_at": utc_now_iso(),
        "note": note or ("shares held outside the strategy; it will never sell "
                         "these, and derives its own position as account minus "
                         "this baseline"),
        # Upper-cased on the way in as well as out, so a hand-edited "tsm"
        # cannot become a second, silently ignored entry alongside "TSM".
        "positions": {k.upper(): int(v)
                      for k, v in sorted(positions.items()) if int(v)},
    }
    _atomic_write(path, json.dumps(payload, indent=2))
    log.info("wrote external-holdings baseline to %s: %s", path,
             payload["positions"] or "(empty)")


# --------------------------------------------------------------------------
# A human-readable snapshot of the strategy's own book
# --------------------------------------------------------------------------

BOOK_CSV_COLUMNS = ["asof", "symbol", "strategy_shares", "account_shares",
                    "yours", "price", "market_value", "target_shares",
                    "target_weight"]


def write_book_csv(path: str,
                   account: Dict[str, int],
                   external: Dict[str, int],
                   strategy: Dict[str, int],
                   prices: Optional[Dict[str, float]] = None,
                   target: Optional[Dict[str, int]] = None,
                   notional: float = 0.0,
                   asof: str = "") -> None:
    """Write what the strategy holds, next to what the account holds.

    A *report*, deliberately not a source of truth. Every number here is
    recomputed from the broker and the baseline on the next run, so editing
    this file changes nothing and losing it costs nothing -- which is exactly
    why it is safe to keep. A CSV that the strategy read back as its position
    of record would drift the first time a fill was missed, and nothing in the
    price history could ever repair it.
    """
    prices = prices or {}
    target = target or {}
    asof = asof or utc_now_iso()[:10]

    rows = []
    for sym in sorted(set(account) | set(strategy) | set(target)):
        shares = int(strategy.get(sym, 0))
        px = float(prices.get(sym, 0.0) or 0.0)
        rows.append({
            "asof": asof,
            "symbol": sym,
            "strategy_shares": shares,
            "account_shares": int(account.get(sym, 0)),
            "yours": int(external.get(sym, 0)),
            "price": f"{px:.4f}" if px else "",
            "market_value": f"{shares * px:.2f}" if px else "",
            "target_shares": int(target[sym]) if sym in target else "",
            "target_weight": (f"{target[sym] * px / notional:.4f}"
                              if sym in target and px and notional else ""),
        })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=BOOK_CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    log.info("wrote the strategy book to %s (%d rows)", path, len(rows))
