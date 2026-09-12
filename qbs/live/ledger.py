"""The strategy's own trade record, as a CSV.

Why this exists
---------------
IB reports one position per symbol per account. When the strategy shares an
account with holdings you manage yourself, it cannot see its own book -- and
the two ways to recover it fail in opposite directions:

  derived   strategy = account - a baseline of what is yours.
            Robust to a missed fill, because the broker's number is
            authoritative. Fragile to *your* trading: buy more of a name the
            strategy holds and the baseline silently understates what is
            yours, so the strategy may sell your shares.

  tallied   strategy = the sum of its own fills, recorded here.
            Robust to your trading, because nothing you do touches this file.
            Fragile to a missed fill, since there is no price history that
            could ever reconstruct one.

Tallying is the better trade when you never sell what the strategy bought: your
own activity is unbounded and silent, while a missed fill is bounded, loud (the
reconcile unit fails) and cross-checkable. So the tally is the position of
record, and every run checks it against the broker: the account must hold at
least what the ledger claims. When it does not, the ledger is wrong in the one
direction that matters -- claiming shares that are not there -- and the run
stops rather than trying to sell them.

Rows are appended from IB's own execution records, never inferred from orders
sent. An order that IB never filled leaves no row, which is the point.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from typing import Dict, Iterable, List, Optional

from .state import utc_now_iso

log = logging.getLogger(__name__)

COLUMNS = ["timestamp", "session_date", "symbol", "side", "quantity", "price",
           "order_id", "exec_id"]


def read_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def positions(path: str) -> Dict[str, int]:
    """Net shares per symbol, from the fills recorded here."""
    out: Dict[str, int] = {}
    for r in read_rows(path):
        try:
            qty = int(round(float(r["quantity"])))
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: unreadable quantity in row {r}")
        sign = 1 if str(r.get("side", "")).upper().startswith("B") else -1
        sym = str(r["symbol"]).upper()
        out[sym] = out.get(sym, 0) + sign * qty
    return {k: v for k, v in out.items() if v}


def append_fills(path: str, session_date: str, fills: Iterable) -> int:
    """Add executions not already recorded. Returns how many were new.

    Keyed on IB's execution id so that re-running reconcile -- which is a
    supported thing to do after a failure -- cannot double-count. Executions
    without an id fall back to a composite key, which is weaker but still
    catches the common case of the same reconcile running twice.
    """
    seen = {_key(r["exec_id"], r["session_date"], r["symbol"], r["side"],
                 r["quantity"], r["order_id"])
            for r in read_rows(path)}

    new: List[Dict[str, object]] = []
    for f in fills:
        if str(getattr(f, "status", "")).lower() == "dryrun":
            continue
        key = _key(f.exec_id, session_date, f.symbol, f.action,
                   f.quantity, f.order_id)
        if key in seen:
            continue
        seen.add(key)
        new.append({
            "timestamp": utc_now_iso(),
            "session_date": session_date,
            "symbol": f.symbol.upper(),
            "side": f.action.upper(),
            "quantity": f"{float(f.quantity):g}",
            "price": f"{float(f.avg_price):.4f}",
            "order_id": f.order_id,
            "exec_id": f.exec_id,
        })

    if not new:
        return 0

    fresh = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh:
            w.writeheader()
        w.writerows(new)
    log.info("ledger: recorded %d new execution(s) in %s", len(new), path)
    return len(new)


def _key(exec_id, session_date, symbol, side, quantity, order_id) -> str:
    if exec_id:
        return f"id:{exec_id}"
    return (f"c:{session_date}|{str(symbol).upper()}|{str(side).upper()}|"
            f"{float(quantity):g}|{order_id}")


def reconcile_against_account(
    ledger: Dict[str, int],
    account: Dict[str, int],
) -> tuple[Dict[str, int], Dict[str, int]]:
    """Compare the tally with the broker. Returns (residual, overclaimed).

    `residual` is account minus ledger: the shares the strategy does not claim,
    which should be exactly what you hold yourself. `overclaimed` names symbols
    where the ledger claims more than the account holds -- impossible if every
    fill was recorded and nobody sold the strategy's shares, so it means one of
    those two things happened and the caller must not trade the name.
    """
    residual: Dict[str, int] = {}
    over: Dict[str, int] = {}
    for sym in set(ledger) | set(account):
        have = int(account.get(sym, 0))
        claim = int(ledger.get(sym, 0))
        if claim > have:
            over[sym] = claim - have
        elif have - claim:
            residual[sym] = have - claim
    return residual, over
