"""Fundamental data from yfinance, cached, with the one caveat that matters.

What this is for
----------------
Every other number in this package is derived from price, which means it can
be recomputed on any past date and therefore backtested. Fundamentals are not
like that. `yfinance`'s `Ticker.info` is a snapshot of **right now**: today's
trailing P/E, today's analyst target, today's short interest. There is no
history in it and no way to ask it what the P/E was in March.

So this module exists to give an *analyst* -- human or LLM -- context on a
name the strategies have already picked on price. It is emphatically NOT a
signal source. Feeding any of these fields into a backtest would date
today's balance sheet back over a decade of trading and report a return no
one could have earned.

That is not a footnote, it is the whole reason the payload carries
`backtest_safe=False` and `as_of`: whatever reads this downstream should have
to look at those before using a number.

The other honest limit: `info` is scraped, not licensed. Fields come and go,
names change between yfinance versions, and a missing key means "not
available today", never "zero". Missing stays missing here -- nothing is
filled in with a default that would read as a real measurement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "fundamentals")


# --------------------------------------------------------------------------
# What we ask for
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Field:
    key: str            # the yfinance `info` key
    label: str          # what a human calls it
    group: str          # which table it belongs in
    unit: str           # "x" multiple, "pct" fraction, "usd", "raw"


FIELDS: Tuple[Field, ...] = (
    # ---- what you pay ---------------------------------------------------
    Field("trailingPE", "Trailing P/E", "Valuation", "x"),
    Field("forwardPE", "Forward P/E", "Valuation", "x"),
    Field("priceToSalesTrailing12Months", "Price / sales", "Valuation", "x"),
    Field("priceToBook", "Price / book", "Valuation", "x"),
    Field("enterpriseToEbitda", "EV / EBITDA", "Valuation", "x"),
    Field("pegRatio", "PEG", "Valuation", "x"),
    # ---- what you get ---------------------------------------------------
    Field("grossMargins", "Gross margin", "Profitability", "pct"),
    Field("operatingMargins", "Operating margin", "Profitability", "pct"),
    Field("profitMargins", "Net margin", "Profitability", "pct"),
    Field("returnOnEquity", "Return on equity", "Profitability", "pct"),
    Field("returnOnAssets", "Return on assets", "Profitability", "pct"),
    Field("freeCashflow", "Free cash flow", "Profitability", "usd"),
    # ---- whether it is getting bigger -----------------------------------
    Field("revenueGrowth", "Revenue growth (yoy)", "Growth", "pct"),
    Field("earningsGrowth", "Earnings growth (yoy)", "Growth", "pct"),
    Field("earningsQuarterlyGrowth", "Earnings growth (qoq)", "Growth", "pct"),
    # ---- whether it survives a bad year ---------------------------------
    Field("debtToEquity", "Debt / equity", "Balance sheet", "raw"),
    Field("currentRatio", "Current ratio", "Balance sheet", "x"),
    Field("quickRatio", "Quick ratio", "Balance sheet", "x"),
    Field("totalCash", "Total cash", "Balance sheet", "usd"),
    Field("totalDebt", "Total debt", "Balance sheet", "usd"),
    # ---- what everyone else thinks --------------------------------------
    Field("targetMeanPrice", "Analyst target (mean)", "Sentiment", "usd"),
    Field("numberOfAnalystOpinions", "Analysts covering", "Sentiment", "raw"),
    Field("recommendationMean", "Recommendation (1 buy - 5 sell)", "Sentiment", "raw"),
    Field("shortPercentOfFloat", "Short interest (% float)", "Sentiment", "pct"),
    Field("heldPercentInstitutions", "Institutional ownership", "Sentiment", "pct"),
    # ---- the name itself -------------------------------------------------
    Field("marketCap", "Market cap", "Profile", "usd"),
    Field("sector", "Sector", "Profile", "raw"),
    Field("industry", "Industry", "Profile", "raw"),
    Field("beta", "Beta", "Profile", "raw"),
    Field("fullTimeEmployees", "Employees", "Profile", "raw"),
)

GROUPS = ("Profile", "Valuation", "Profitability", "Growth", "Balance sheet",
          "Sentiment")


@dataclass
class Snapshot:
    """One name's fundamentals, as of the moment they were fetched.

    `as_of` is when the data was *retrieved*, not the period it describes --
    yfinance does not say which quarter a trailing figure covers, so claiming
    a period here would be inventing one.
    """
    ticker: str
    as_of: str
    values: Dict[str, object] = field(default_factory=dict)
    source: str = "yfinance"
    stale_days: float = 0.0
    backtest_safe: bool = False     # never true; see the module docstring

    @property
    def missing(self) -> List[str]:
        return [f.label for f in FIELDS if self.values.get(f.key) is None]

    def to_frame(self) -> pd.DataFrame:
        """Long form: one row per field, grouped, with missing rows kept.

        Kept, not dropped: "we asked and Yahoo had nothing" is a different
        statement from "we never asked", and a reader of the table cannot
        tell them apart if the row silently disappears.
        """
        rows = []
        for f in FIELDS:
            rows.append({"Group": f.group, "Measure": f.label,
                         "Value": self.values.get(f.key), "Unit": f.unit})
        out = pd.DataFrame(rows)
        out["Group"] = pd.Categorical(out["Group"], GROUPS, ordered=True)
        return out.sort_values(["Group", "Measure"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Fetch / cache
# --------------------------------------------------------------------------

def _cache_path(ticker: str, cache_dir: Optional[str] = None) -> str:
    return os.path.join(cache_dir or CACHE_DIR, f"{ticker.upper()}.json")


def _read_cache(ticker: str, cache_dir: Optional[str] = None) -> Optional[Snapshot]:
    path = _cache_path(ticker, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except (OSError, ValueError):
        return None                     # a corrupt cache is a cache miss
    as_of = blob.get("as_of", "")
    stale = 0.0
    if as_of:
        stale = max(0.0, (pd.Timestamp.now("UTC").tz_convert(None)
                          - pd.Timestamp(as_of)).total_seconds() / 86400.0)
    return Snapshot(ticker=ticker.upper(), as_of=as_of,
                    values=blob.get("values", {}),
                    source=blob.get("source", "yfinance"),
                    stale_days=stale)


def _write_cache(snap: Snapshot, cache_dir: Optional[str] = None) -> None:
    path = _cache_path(snap.ticker, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"as_of": snap.as_of, "source": snap.source,
                   "values": snap.values}, fh, indent=1, default=str)


def fetch_fundamentals(
    ticker: str,
    refresh: bool = False,
    offline: bool = False,
    max_age_days: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Tuple[Optional[Snapshot], Optional[str]]:
    """One name's fundamentals. Returns `(snapshot, error)`, never raises.

    The `(data, error)` shape is the one `qbs.finviz` already uses, and for
    the same reason: a caller that gets `None` back needs to be able to say
    WHY on screen. Swallowing the exception and returning an empty frame
    produces a UI that can only report that something went wrong.

    A cached snapshot younger than `max_age_days` is returned without a
    network call. When the network fails but a cache exists, the cache comes
    back WITH the error string -- stale data plus an explanation beats no
    data, as long as the caller is told which it is holding.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return None, "no ticker given"

    cached = _read_cache(ticker, cache_dir)
    if cached is not None and not refresh and cached.stale_days <= max_age_days:
        return cached, None
    if offline:
        if cached is not None:
            return cached, None
        return None, (f"offline and nothing cached for {ticker} -- run once "
                      f"with offline=False to populate {_cache_path(ticker, cache_dir)}")

    try:
        import yfinance as yf
    except ImportError:
        return cached, "yfinance is not installed (pip install yfinance)"

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:              # noqa: BLE001 -- network, parsing, rate limits
        # Deliberately broad: yfinance raises whatever its HTTP stack raises,
        # and the caller only needs to know it failed and what it said.
        msg = f"yfinance failed for {ticker}: {type(exc).__name__}: {exc}"
        return cached, msg

    wanted = {f.key for f in FIELDS}
    values = {k: v for k, v in info.items() if k in wanted and v is not None}
    if not values:
        return cached, (f"yfinance returned no recognised fields for {ticker} "
                        f"-- delisted, wrong symbol, or a schema change")

    snap = Snapshot(ticker=ticker,
                    as_of=pd.Timestamp.now("UTC").tz_convert(None).isoformat(timespec="seconds"),
                    values=values)
    try:
        _write_cache(snap, cache_dir)
    except OSError as exc:
        return snap, f"fetched but could not cache: {exc}"
    return snap, None


def fetch_many(
    tickers: List[str],
    pause: float = 0.0,
    **kwargs,
) -> Tuple[Dict[str, Snapshot], Dict[str, str]]:
    """`fetch_fundamentals` over a list. Returns `(snapshots, errors)`.

    One name failing must not lose the others, so errors are collected per
    ticker rather than raised. `pause` throttles between calls: Yahoo rate
    limits, and a 20-name watchlist fetched flat out is how you get blocked.
    """
    snaps: Dict[str, Snapshot] = {}
    errors: Dict[str, str] = {}
    for i, t in enumerate(tickers):
        snap, err = fetch_fundamentals(t, **kwargs)
        if snap is not None:
            snaps[snap.ticker] = snap
        if err:
            errors[t.upper()] = err
        if pause and i < len(tickers) - 1:
            time.sleep(pause)
    return snaps, errors


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def format_value(value: object, unit: str) -> str:
    """One field, printed the way its unit deserves."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if unit == "pct":
        return f"{v:+.1%}"
    if unit == "x":
        return f"{v:.1f}x"
    if unit == "usd":
        for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(v) >= cut:
                return f"${v / cut:,.1f}{suffix}"
        return f"${v:,.0f}"
    return f"{v:,.2f}" if abs(v) < 1e4 else f"{v:,.0f}"


def to_text(snap: Snapshot, errors: str = "") -> str:
    """The snapshot as plain text, for a prompt or a terminal.

    The header carries `as_of` and the backtest warning on purpose: this
    string is what an LLM sees, and a model handed a bare table of ratios
    will cheerfully reason about "the P/E at the time of the signal".
    """
    lines = [f"FUNDAMENTALS — {snap.ticker}",
             f"retrieved {snap.as_of} UTC from {snap.source}"
             + (f" ({snap.stale_days:.1f} days ago)" if snap.stale_days >= 1 else ""),
             "NOTE: a snapshot of today only. These figures have no history and "
             "must not be used to explain a past signal or entered into a "
             "backtest.", ""]
    frame = snap.to_frame()
    for group in GROUPS:
        rows = frame[frame["Group"] == group]
        shown = [(r.Measure, format_value(r.Value, r.Unit))
                 for r in rows.itertuples() if r.Value is not None]
        if not shown:
            continue
        lines.append(f"[{group}]")
        width = max(len(m) for m, _ in shown)
        lines += [f"  {m.ljust(width)}  {v}" for m, v in shown]
        lines.append("")
    if snap.missing:
        lines.append("Not available today: " + ", ".join(snap.missing))
    if errors:
        lines.append(f"WARNING: {errors}")
    return "\n".join(lines).strip()
