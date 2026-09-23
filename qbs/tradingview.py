"""The broad US universe, sourced from TradingView's scanner.

A second provider for exactly what `qbs.finviz.fetch_us_universe` returns:
`Ticker`, `Sector`, `Industry`, `Country` for every liquid US stock and ADR.
Same contract, same cache shape, same "return the reason, never raise"
discipline -- so the caller picks a source and nothing else changes.

Why a second one at all
-----------------------
Not because TradingView is more trustworthy. Both are undocumented endpoints
that can be reshaped or blocked without notice, and swapping one for the
other would be a lateral move on that risk. Having BOTH is the gain: when one
stops answering, the dashboard has somewhere else to go, and the breadth tab
falls back to a 99-name index only when neither works.

What it is cheaper at is real, though. Finviz's screener paginates at 20 rows
a page, so ~2,600 names is ~120 requests with a polite sleep between them --
minutes. The scanner answers the same question in one POST.

What this does NOT do
---------------------
Apply the momentum rule. The scanner offers `Perf.3M`, and this package's
high-momentum definition is a 63-SESSION gain computed from prices -- close
enough to look interchangeable and not the same number. The rule stays where
it is, in `qbs.breadth.leader_mask`, measured on the bars. This module's only
job is membership: which tickers are in the pool, and what sector each is in.

Nor is it point-in-time. Like the Finviz screener it returns today's
membership, so a universe fetched today cannot describe what was liquid in
2019. That limitation is unchanged and unfixable from either source.

The endpoint is `scanner.tradingview.com`, which is TradingView's internal
scanner rather than a published API. It needs no key. Their terms restrict
automated access; that is a judgement for whoever runs this, which is why
this is opt-in via `QBS_UNIVERSE_SOURCE` and not the default.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .data import normalise_symbols
from .finviz import CACHE_DIR, UniverseFilters

UNIVERSE_CSV = os.path.join(CACHE_DIR, "tradingview_universe.csv")

# Its own cache file, deliberately. Sharing one with Finviz would mean a
# switch between providers silently reads the other's answer and reports it
# as this one's -- and the whole point of two sources is telling them apart.

# The scanner's instrument types. "stock" is common equity; "dr" is a
# depositary receipt, which is how ADRs arrive and which the Finviz
# "Stocks only (ex-Funds)" filter also keeps. Funds, ETFs, structured
# products and bonds are excluded by not being on this list.
STOCK_TYPES = ("stock", "dr")

COLUMNS = ("name", "close", "volume", "average_volume_90d_calc",
           "sector", "industry", "country", "type", "typespecs")


def fetch_us_universe(
    filters: Optional[UniverseFilters] = None,
    refresh: bool = False,
    offline: bool = False,
    cache_path: str = UNIVERSE_CSV,
    max_age_days: int = 1,
    sleep_sec: int = 1,          # accepted and unused: one request, nothing to pace
    verbose: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """`(universe, error)` -- the same frame `qbs.finviz` returns, from TradingView.

    Signature matches `qbs.finviz.fetch_us_universe` argument for argument,
    including the ones this provider has no use for, so a caller can hold
    either behind one name without special-casing. `sleep_sec` is the honest
    example: Finviz needs it to pace 120 page requests and this needs nothing
    to pace.

    Returns `(None, reason)` rather than raising -- the caller is a dashboard
    that falls back, and a failure it cannot read off the screen is a failure
    it cannot fix.
    """
    filters = filters or UniverseFilters()

    cached, age = None, None
    if os.path.exists(cache_path):
        try:
            cached = pd.read_csv(cache_path)
            age = (pd.Timestamp.now("UTC").tz_localize(None)
                   - pd.Timestamp(os.path.getmtime(cache_path), unit="s")).days
        except Exception:  # noqa: BLE001
            cached = None

    if offline:
        if cached is not None and not cached.empty:
            return cached, None
        return None, ("offline and no cached TradingView universe on disk — "
                      "run once with Source set to Online to build it")
    if (cached is not None and not cached.empty and not refresh
            and (age or 0) <= max_age_days):
        if verbose:
            print(f"[tradingview] universe cache hit: {len(cached)} tickers "
                  f"({age}d old)")
        return cached, None

    try:
        out = _scan(filters)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out.to_csv(cache_path, index=False)
        if verbose:
            print(f"[tradingview] {len(out)} tickers · "
                  f"{out['Sector'].nunique()} sectors")
        return out, None
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"[tradingview] universe fetch failed ({reason})")
        if cached is not None and not cached.empty:
            return cached, (f"using a stale cached universe — live fetch "
                            f"failed ({reason})")
        return None, reason


def _scan(filters: UniverseFilters) -> pd.DataFrame:
    """One POST to the scanner, translated into this package's frame."""
    try:
        from tradingview_screener import Query, col
    except ImportError as exc:
        raise RuntimeError(
            "tradingview-screener is not installed in the environment running "
            "this app. Install it with `pip install tradingview-screener` "
            "(installing it in a notebook or on Colab does not help here — it "
            f"has to be the same interpreter running Streamlit). [{exc}]"
        ) from exc

    types = list(STOCK_TYPES) if filters.include_dr else [STOCK_TYPES[0]]

    # The liquidity test is AVERAGE volume, not the latest session's. They are
    # different tests and the Finviz side filters on the average, so filtering
    # on `volume` here would hand the two providers different universes and
    # blame the difference on the market.
    query = (Query()
             .select(*COLUMNS)
             .where(col("close") > filters.min_price,
                    col("average_volume_90d_calc") > filters.min_avg_volume,
                    col("type").isin(types))
             .set_markets("america")
             .limit(20_000))

    count, df = query.get_scanner_data()
    if df is None or df.empty:
        raise RuntimeError(f"scanner returned no rows (count={count})")
    return _shape(df)


def _shape(df: pd.DataFrame) -> pd.DataFrame:
    """The scanner's frame, as `Ticker / Sector / Industry / Country`.

    `name` is the bare symbol and `ticker` is exchange-qualified
    ("NASDAQ:AAPL"); this takes `name` and falls back to the tail of
    `ticker`, because which of the two a version returns has moved before.

    Symbols go through `normalise_symbols`, the same one the Finviz side
    applies -- dots and slashes both become dashes -- because that is the
    spelling yfinance wants and the two universes have to produce keys that
    match the same price frame.
    """
    out = pd.DataFrame()
    if "name" in df.columns:
        out["Ticker"] = df["name"].astype(str)
    elif "ticker" in df.columns:
        out["Ticker"] = df["ticker"].astype(str).str.rsplit(":", n=1).str[-1]
    else:
        raise RuntimeError(f"no symbol column in {list(df.columns)[:8]}")

    out["Ticker"] = normalise_symbols(out["Ticker"]).values
    for src, dst in (("sector", "Sector"), ("industry", "Industry"),
                     ("country", "Country")):
        if src in df.columns:
            out[dst] = df[src].astype(str).replace({"nan": ""})

    out = out[out["Ticker"].str.len() > 0]
    return out.drop_duplicates(subset="Ticker").reset_index(drop=True)


def diagnose(verbose: bool = True) -> Dict[str, str]:
    """Check every link in the chain and say which one is broken.

    `python -m qbs.tradingview`

    Mirrors `qbs.finviz.diagnose`, for the same reason: Streamlit swallows
    tracebacks, so "the US universe is unavailable" could be a missing
    package, a blocked network or an empty result, and the screen cannot tell
    you which.
    """
    import sys

    steps: Dict[str, str] = {"python": sys.executable}

    try:
        import tradingview_screener  # noqa: F401
        # The package exposes no `__version__`, so ask the installer. A
        # diagnostic whose version line reads "v?" cannot answer the one
        # question it exists for when the endpoint's shape changes.
        try:
            from importlib.metadata import version
            v = version("tradingview-screener")
        except Exception:  # noqa: BLE001
            v = "?"
        steps["tradingview-screener"] = f"OK (v{v})"
    except ImportError as exc:
        steps["tradingview-screener"] = (
            f"MISSING — {exc}. Fix: pip install tradingview-screener "
            "using THIS interpreter")
        if verbose:
            _print_steps(steps)
        return steps

    f = UniverseFilters()
    steps["filters"] = (f"close > {f.min_price:g} · avg vol > "
                        f"{f.min_avg_volume:,.0f} · type in "
                        f"{list(STOCK_TYPES) if f.include_dr else [STOCK_TYPES[0]]}")

    uni, err = fetch_us_universe(refresh=True, verbose=False)
    if uni is None:
        steps["scanner"] = f"FAILED — {err}"
        if verbose:
            _print_steps(steps)
        return steps
    steps["scanner"] = (f"OK — {len(uni)} tickers"
                        + (f", {uni['Sector'].nunique()} sectors"
                           if "Sector" in uni.columns else ", NO Sector column"))
    steps["sample"] = ", ".join(uni["Ticker"].head(8))
    if verbose:
        _print_steps(steps)
    return steps


def _print_steps(steps: Dict[str, str]) -> None:
    width = max(len(k) for k in steps)
    print("\nTradingView universe diagnostics")
    print("-" * (width + 40))
    for k, v in steps.items():
        print(f"  {k:<{width}}  {v}")
    print()


if __name__ == "__main__":
    diagnose()
