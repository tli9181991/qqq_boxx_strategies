"""The newest close, from the screener that already returned it.

Why this exists
---------------
Every price in this package comes from yfinance, and yfinance is reliable for
settled bars and unreliable at the front edge: hours after a close it will
serve the newest session for a dozen names out of two and a half thousand and
nothing for the rest. A bar like that is not a session -- breadth counted over
a dozen names reports zero 4% movers on a day the market rose -- so
`drop_partial_bars` throws it away and the dashboard sits one session behind
until the provider catches up.

Meanwhile the universe request the app has ALREADY made returns that exact
number. Finviz's screener carries `Price` and `Volume`; TradingView's scanner
carries `close` and `volume`. Both were being discarded.

So: yfinance keeps supplying history, and the screener tops up the front edge.
No extra network, because the request is the one being made anyway.

What this is careful about
--------------------------
* **Never during the session.** A screener queried while the market is open
  returns an intraday print, not a close. Dropping one into a series of closes
  is a silent corruption -- every return computed across it is wrong and
  nothing looks wrong. `fill_last_bar` refuses unless the session has closed.
* **Never over a real bar.** yfinance is the authority for anything it has
  actually published; the fill only reaches names whose newest bar is missing.
* **Never more than one bar.** A snapshot knows today and nothing else.
* **Never silently.** The count of filled names comes back so a caller can say
  so on screen. A breadth reading sourced half from one provider and half from
  another, with nothing saying so, is the failure this module is supposed to
  be preventing.

The adjustment basis, which looks like a problem and is not
-----------------------------------------------------------
yfinance serves split- and dividend-adjusted closes; a screener serves the raw
last price. Mixing the two would normally put a step in the series at every
distribution. It is safe HERE because of two facts together: the adjustment
factor for the most recent bar is 1.0 -- adjustments are applied backwards, to
history, never to today -- and a filled cell never outlives the next download.
The dashboard saves filled cells to the cache MARKED PROVISIONAL
(`qbs.incremental.persist_fill`), so the next load needs no network; the next
download window always starts before the oldest provisional cell, replaces it
with yfinance's adjusted value, and leaves it out of the re-basing check it
runs on the overlap. Neither fact alone is enough, which is why this paragraph
exists.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .data import MARKET_TZ, last_market_close, normalise_symbols
from .universe_source import resolve_source

FILL_VAR = "QBS_FILL_LAST_BAR"


def latest_quotes(source: Optional[str] = None,
                  **kwargs) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """`(quotes, error)` -- last price and volume per ticker, from the screener.

    Indexed by ticker, columns `close` and `volume`. Returns `(None, reason)`
    rather than raising: this is a top-up on data that already loaded, and a
    failure to improve it must never take down what was already there.
    """
    name = resolve_source(source)
    try:
        frame = (_tradingview_quotes(**kwargs) if name == "tradingview"
                 else _finviz_quotes(**kwargs))
    except Exception as exc:  # noqa: BLE001
        return None, f"{name}: {type(exc).__name__}: {exc}"
    if frame is None or frame.empty:
        return None, f"{name}: the screener returned no quotes"
    return frame, None


def _tradingview_quotes(limit: int = 20_000, **_) -> pd.DataFrame:
    from tradingview_screener import Query, col

    from .finviz import UniverseFilters
    from .tradingview import STOCK_TYPES

    f = UniverseFilters()
    types = list(STOCK_TYPES) if f.include_dr else [STOCK_TYPES[0]]
    _, df = (Query().select("name", "close", "volume")
             .where(col("close") > f.min_price,
                    col("average_volume_90d_calc") > f.min_avg_volume,
                    col("type").isin(types))
             .set_markets("america").limit(limit).get_scanner_data())
    if df is None or df.empty:
        return pd.DataFrame()
    sym = df["name"] if "name" in df.columns else \
        df["ticker"].astype(str).str.rsplit(":", n=1).str[-1]
    return _shape_quotes(sym, df.get("close"), df.get("volume"))


def _finviz_quotes(sleep_sec: int = 1, **_) -> pd.DataFrame:
    from finvizfinance.screener.overview import Overview

    from .finviz import UniverseFilters

    view = Overview()
    view.set_filter(filters_dict=UniverseFilters().as_dict())
    df = view.screener_view(order="Ticker", verbose=0, sleep_sec=sleep_sec)
    if df is None or df.empty or "Ticker" not in df.columns:
        return pd.DataFrame()
    # The Overview table names them Price and Volume; both have moved before,
    # so a missing one is reported by shape rather than by KeyError.
    return _shape_quotes(df["Ticker"], df.get("Price"), df.get("Volume"))


def _shape_quotes(symbols, close, volume) -> pd.DataFrame:
    if close is None:
        raise RuntimeError("the screener returned no price column")
    out = pd.DataFrame({
        "ticker": normalise_symbols(symbols).values,
        "close": pd.to_numeric(pd.Series(close).values, errors="coerce"),
        "volume": (pd.to_numeric(pd.Series(volume).values, errors="coerce")
                   if volume is not None else np.nan),
    })
    out = out[out["ticker"].str.len() > 0].dropna(subset=["close"])
    return out.drop_duplicates(subset="ticker").set_index("ticker")


def fill_disabled(environ: Optional[Dict[str, str]] = None) -> bool:
    """Is the front-bar top-up switched off?

    On by default -- it exists because the front edge is unreliable, and a
    remedy that ships off is a remedy nobody gets. Set QBS_FILL_LAST_BAR to
    0/false/no/off/none to keep the series purely yfinance.
    """
    import os

    env = os.environ if environ is None else environ
    raw = (env.get(FILL_VAR) or "").strip().lower()
    return raw in ("0", "false", "no", "off", "none", "disabled")


def needs_fill(closes: pd.DataFrame,
               now: Optional[pd.Timestamp] = None) -> bool:
    """Is the newest closed session missing or incomplete?

    Asked BEFORE the screener is queried, so a normal day costs no request at
    all. The top-up is for the hours when yfinance is behind, not a second
    provider on every page load.
    """
    if closes is None or closes.empty:
        return False
    now = (pd.Timestamp.now(MARKET_TZ) if now is None else pd.Timestamp(now))
    now = (now.tz_localize("UTC") if now.tzinfo is None
           else now).tz_convert(MARKET_TZ)
    session = last_market_close(now).tz_localize(None).normalize()
    if session not in closes.index:
        return True
    return bool(closes.loc[session].isna().any())


def fill_last_bar(
    closes: pd.DataFrame,
    volumes: Optional[pd.DataFrame],
    quotes: pd.DataFrame,
    now: Optional[pd.Timestamp] = None,
    session: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    """Top up the newest session from a screener snapshot.

    Returns `(closes, volumes, report)`. `report` carries `session`, `filled`,
    `already`, `absent`, `skipped` and `tickers` (the names filled) -- everything a caller needs to say what
    happened on screen, because this mixes two providers in one series and
    that must never be invisible.

    `session` defaults to the most recent CLOSED session. Passing one that is
    still open is refused rather than honoured: the screener would be
    answering with an intraday print, and a series of closes with one
    intraday value in it is wrong in a way that nothing downstream can detect.
    """
    report = {"session": None, "filled": 0, "already": 0, "absent": 0,
              "skipped": None, "tickers": []}
    if closes is None or closes.empty or quotes is None or quotes.empty:
        report["skipped"] = "nothing to fill from"
        return closes, volumes, report

    now = (pd.Timestamp.now(MARKET_TZ) if now is None else pd.Timestamp(now))
    now = (now.tz_localize("UTC") if now.tzinfo is None
           else now).tz_convert(MARKET_TZ)
    closed = last_market_close(now)
    session = (closed.tz_localize(None).normalize() if session is None
               else pd.Timestamp(session).normalize())
    report["session"] = session

    if session > closed.tz_localize(None).normalize():
        report["skipped"] = (f"{session:%Y-%m-%d} has not closed yet — a "
                             "screener would answer with an intraday price")
        return closes, volumes, report
    if session in closes.index and bool(closes.loc[session].notna().all()):
        report["skipped"] = "that session is already complete"
        return closes, volumes, report

    closes = closes.copy()
    if session not in closes.index:
        closes.loc[session] = np.nan
        closes = closes.sort_index()
    if volumes is not None and not volumes.empty:
        volumes = volumes.reindex(index=closes.index, columns=closes.columns)

    have = closes.loc[session]
    for t in closes.columns:
        if pd.notna(have.get(t)):
            report["already"] += 1
        elif t in quotes.index:
            closes.loc[session, t] = float(quotes.at[t, "close"])
            if volumes is not None and "volume" in quotes.columns \
                    and pd.notna(quotes.at[t, "volume"]):
                volumes.loc[session, t] = float(quotes.at[t, "volume"])
            report["filled"] += 1
            report["tickers"].append(t)
        else:
            report["absent"] += 1
    return closes, volumes, report


def fill_note(report: Dict, source: Optional[str] = None) -> Optional[str]:
    """One sentence naming what was filled and from where, or None."""
    if not report or not report.get("filled"):
        return None
    name = resolve_source(source)
    total = report["filled"] + report["already"] + report["absent"]
    return (f"{report['session']:%Y-%m-%d} topped up from **{name}** — "
            f"{report['filled']} of {total} closes came from the screener "
            f"because yfinance had not published them yet"
            + (f", {report['absent']} still missing" if report["absent"] else ""))
