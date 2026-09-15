"""Mirror the run log into a Google Sheet.

Why a mirror and not a store
----------------------------
The SQLite run log stays the record. This pushes a copy somewhere you can read
from a phone, share, and chart by hand -- and nothing reads it back. That
matters because the Sheets API is the least reliable dependency in the system:
a network blip, a quota, an expired key or a revoked share all fail here, and
none of them must be able to touch an order.

Two things enforce that. The push runs as its own systemd unit after reconcile,
so a failure fails only itself; and the whole of gspread is imported lazily, so
a box without the package still trades.

Auth
----
A service account, because nothing else works unattended: the OAuth consent
flow needs a browser, and a refresh token would eventually need a human. Create
a service account, download its JSON key to `var/google-sa.json` (var/ is
gitignored and bind-mounted, so it never enters the image), then share the
sheet with the service account's email address as an Editor. The account owns
nothing and can reach nothing else in your Drive.

Writing
-------
Each tab is cleared and rewritten from the database every run, rather than
appended to. Appending would need to know what it had already sent, which is
another piece of state that can drift; a rewrite of a few hundred rows costs
one API call per tab and is correct no matter how many times it runs, or how
many runs were missed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from .store import connect

log = logging.getLogger(__name__)

# Tab name -> (table, order-by clause, row cap). The caps keep a rewrite to one
# API call per tab and stop a year of running turning into a slow daily push.
DEFAULT_TABS: Dict[str, tuple] = {
    "trades":    ("trade_events",     "id DESC",            2000),
    "selection": ("selection_events", "session_date DESC, rank", 4000),
    "closes":    ("position_closes",  "session_date DESC",  4000),
    "nav":       ("portfolio_nav",    "session_date DESC",  1000),
    "runs":      ("signal_runs",      "session_date DESC",  1000),
}

ALLOWED_TABLES = {t for t, _, _ in DEFAULT_TABS.values()}


class SheetsError(RuntimeError):
    pass


def build_tabs(db_path: str,
               tabs: Optional[Dict[str, tuple]] = None) -> Dict[str, List[List[Any]]]:
    """Read the run log into {tab name: [header, *rows]}. No network.

    Values are coerced to what Sheets accepts -- numbers and strings, with None
    as an empty cell. Everything else would arrive as the string "None", which
    then poisons any formula written against the column.
    """
    tabs = tabs or DEFAULT_TABS
    out: Dict[str, List[List[Any]]] = {}
    with connect(db_path) as conn:
        for name, (table, order, cap) in tabs.items():
            if table not in ALLOWED_TABLES:
                raise SheetsError(f"unknown table {table!r}")
            cur = conn.execute(f"SELECT * FROM {table} ORDER BY {order} LIMIT {int(cap)}")
            rows = [dict(r) for r in cur.fetchall()]
            if not rows:
                cols = [d[0] for d in cur.description] if cur.description else []
                out[name] = [cols] if cols else []
                continue
            header = list(rows[0].keys())
            out[name] = [header] + [[_cell(r[c]) for c in header] for r in rows]
    return out


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


def push(spreadsheet_id: str, key_file: str,
         tabs: Dict[str, List[List[Any]]]) -> Dict[str, int]:
    """Replace each named worksheet's contents. Returns rows written per tab."""
    if not spreadsheet_id:
        raise SheetsError("no spreadsheet id configured (QBS_SHEETS_ID)")
    if not os.path.exists(key_file):
        raise SheetsError(
            f"service-account key not found at {key_file}. Download the JSON key "
            "for the service account and put it there, then share the sheet with "
            "that account's email as an Editor.")

    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError as exc:   # pragma: no cover - environment dependent
        raise SheetsError(
            "gspread is not installed. `pip install -r requirements-live.txt`, "
            "or leave the sheets phase disabled -- nothing else needs it."
        ) from exc

    creds = Credentials.from_service_account_file(
        key_file, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    book = gspread.authorize(creds).open_by_key(spreadsheet_id)

    written: Dict[str, int] = {}
    for name, values in tabs.items():
        if not values:
            continue
        ws = _worksheet(book, name, rows=len(values) + 10, cols=len(values[0]) + 2)
        ws.clear()
        ws.update(values, "A1")
        written[name] = len(values) - 1      # not counting the header
        log.info("sheets: wrote %d row(s) to tab %r", written[name], name)
    return written


def _worksheet(book, name: str, rows: int, cols: int):
    """The tab, created if this is the first push."""
    try:
        return book.worksheet(name)
    except Exception:
        log.info("sheets: creating tab %r", name)
        return book.add_worksheet(title=name, rows=max(rows, 100), cols=max(cols, 10))
