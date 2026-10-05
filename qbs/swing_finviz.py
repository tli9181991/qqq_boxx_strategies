"""Finviz pre-screens for the swing-trade tab.

The six setups in `qbs.swing` are exact rules over daily bars, but running
them over the whole US market means downloading ~2,400 histories. Finviz does
the coarse cut first: a liquidity base every swing candidate needs, plus a few
filters per setup that keep the names the setup could plausibly fire on. The
survivors (usually tens to a few hundred) get real daily bars and the full
screens.

Every filter here is Finviz's own vocabulary -- the screener takes "Over $10",
not 10 -- and `validate_filters` checks each one against the option list the
`finvizfinance` library ships, so a typo fails loudly instead of being sent.

The pull goes through `qbs.finviz.screener_pull`: paced, and refused during
the cool-down after a block. Results are cached per filter set, so a re-run of
the dashboard re-reads the file rather than re-crawling.

The presets are a coarse funnel, not the rule: a name can pass its preset and
fail the setup (most do), and a name a preset drops could have qualified --
e.g. "Week Down" drops a pullback that bounced today.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .data import CACHE_DIR, normalise_symbols

SCREEN_DIR = os.path.join(CACHE_DIR, "swing_finviz")

# Liquid enough to swing in and out of: mid cap and up, over $10, a million
# shares a day, US listed common stock.
BASE_FILTERS: Dict[str, str] = {
    "Industry": "Stocks only (ex-Funds)",
    "Country": "USA",
    "Market Cap.": "+Mid (over $2bln)",
    "Price": "Over $10",
    "Average Volume": "Over 1M",
}

# Per setup, added on top of the base. Each line is the coarse version of a
# rule in the matching `qbs.swing` screen.
PRESETS: Dict[str, Dict[str, str]] = {
    # uptrend, cooled off this week, still up on the quarter
    "trend_pullback": {
        "200-Day Simple Moving Average": "Price above SMA200",
        "50-Day Simple Moving Average": "SMA50 above SMA200",
        "RSI (14)": "Not Overbought (<60)",
        "Performance": "Week Down",
        "Performance 2": "Quarter +10%",
    },
    # near the highs after a breakout, dipping back this week
    "breakout_retest": {
        "200-Day Simple Moving Average": "Price above SMA200",
        "52-Week High/Low": "0-10% below High",
        "Performance": "Week Down",
        "Performance 2": "Quarter Up",
    },
    # Finviz's own horizontal-range detection, in the lower half by RSI
    "range_bounce": {
        "Pattern": "Horizontal S/R",
        "RSI (14)": "Not Overbought (<50)",
    },
    # long-term uptrend, short-term oversold
    "mean_reversion": {
        "200-Day Simple Moving Average": "Price above SMA200",
        "RSI (14)": "Oversold (40)",
        "Performance": "Week Down",
    },
    # trending near the highs on quiet volume -- Finviz has no low-volatility
    # filter (its Volatility options are all "Over"), so a below-average
    # relative volume stands in for the dry-up
    "volatility_contraction": {
        "50-Day Simple Moving Average": "Price above SMA50",
        "200-Day Simple Moving Average": "SMA200 below SMA50",
        "52-Week High/Low": "0-10% below High",
        "Relative Volume": "Under 1",
    },
    # leaders: big quarter, at the highs, above the 50-day
    "relative_strength": {
        "Performance": "Quarter +20%",
        "52-Week High/Low": "0-5% below High",
        "50-Day Simple Moving Average": "Price above SMA50",
    },
}

# The filters the tab lets you edit, in display order.
EDITABLE = (
    "Market Cap.", "Price", "Average Volume", "Relative Volume",
    "Performance", "Performance 2", "RSI (14)",
    "50-Day Simple Moving Average", "200-Day Simple Moving Average",
    "52-Week High/Low", "Pattern", "Sector",
)

# Downloading bars costs one request per name, so the candidate list is cut
# to the most liquid names past this.
MAX_CANDIDATES = 150


def preset_filters(key: Optional[str]) -> Dict[str, str]:
    """The base plus one setup's preset (`None` for the base alone)."""
    return {**BASE_FILTERS, **(PRESETS.get(key, {}) if key else {})}


def filter_options() -> Dict[str, List[str]]:
    """Finviz's option list per filter name, from `finvizfinance`. Empty if
    the library is not installed."""
    try:
        from finvizfinance.constants import filter_dict
    except ImportError:
        return {}
    return {k: list(v["option"].keys()) for k, v in filter_dict.items()}


def validate_filters(filters: Dict[str, str]) -> List[str]:
    """Problems with a filter set, one string each; empty when it is valid.
    "Any" is dropped before sending, so it is always valid."""
    opts = filter_options()
    if not opts:
        return ["finvizfinance is not installed (pip install -r "
                "requirements-dashboard.txt)"]
    errs = []
    for k, v in filters.items():
        if k not in opts:
            errs.append(f"unknown filter {k!r}")
        elif v not in opts[k]:
            errs.append(f"{k}: {v!r} is not one of Finviz's options")
    return errs


def clean_filters(filters: Dict[str, str]) -> Dict[str, str]:
    """Without the "Any" entries, which filter nothing."""
    return {k: v for k, v in filters.items() if v and v != "Any"}


def cache_path(filters: Dict[str, str], directory: str = SCREEN_DIR) -> str:
    key = json.dumps(clean_filters(filters), sort_keys=True)
    return os.path.join(directory,
                        hashlib.sha1(key.encode()).hexdigest()[:12] + ".csv")


def load_cached(filters: Dict[str, str], directory: str = SCREEN_DIR
                ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Timestamp]]:
    """`(rows, fetched_at)` from the last pull of exactly these filters, or
    `(None, None)`."""
    path = cache_path(filters, directory)
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None, None
    return df, pd.Timestamp(os.path.getmtime(path), unit="s", tz="UTC")


def run_screen(filters: Dict[str, str], directory: str = SCREEN_DIR,
               sleep_sec: Optional[float] = None
               ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Pull the screen from Finviz and cache it. `(rows, error)`.

    Returns the error rather than raising -- a block, the cool-down after
    one, or a filter typo -- because the caller is a dashboard button.
    """
    errs = validate_filters(filters)
    if errs:
        return None, "; ".join(errs)
    from .finviz import FinvizCoolingDown, screener_pull
    try:
        df = screener_pull(clean_filters(filters), sleep_sec)
    except FinvizCoolingDown as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    if df is None or df.empty:
        df = pd.DataFrame(columns=["Ticker"])
    keep = [c for c in ("Ticker", "Company", "Sector", "Industry",
                        "Market Cap", "Price", "Change", "Volume")
            if c in df.columns]
    out = df[keep].copy()
    if not out.empty:
        out["Ticker"] = normalise_symbols(out["Ticker"]).values
        out = out.drop_duplicates(subset="Ticker").reset_index(drop=True)
    os.makedirs(directory, exist_ok=True)
    out.to_csv(cache_path(filters, directory), index=False)
    return out, None


def candidates(rows: pd.DataFrame, n: int = MAX_CANDIDATES) -> List[str]:
    """The tickers to download, most traded first, at most `n`."""
    if rows is None or rows.empty:
        return []
    if "Volume" in rows.columns:
        vol = pd.to_numeric(rows["Volume"], errors="coerce")
        rows = rows.assign(_v=vol).sort_values("_v", ascending=False)
    return list(rows["Ticker"].astype(str).head(n))
