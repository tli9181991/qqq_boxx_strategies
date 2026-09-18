"""The broad US universe, sourced from the Finviz screener.

The breadth tab was measuring ~99 Nasdaq-100 constituents and calling it
market breadth, which it is not: a "names up 4%" count out of 99 mega-caps is
a different measurement from the same count out of 2,400, not a smaller
version of it. This module fetches the universe the reference dashboards
actually use.

The universe definition
-----------------------
US common stocks and ADRs, excluding funds, priced over $5, trading over
300k shares a day -- the same liquidity floor those dashboards quote. In
Finviz's own filter vocabulary:

    Industry        "Stocks only (ex-Funds)"     -- drops ETFs and CEFs
    Price           "Over $5"
    Average Volume  "Over 300K"

The screener returns `Sector` in the same response, which is what finally
makes the sector-concentration panel computable: it needs a ticker -> sector
map, and there was never one in this package.

What this costs, because it is not free
---------------------------------------
* **The screener paginates at 20 rows a page.** ~2,400 names is ~120 requests
  with a polite sleep between them -- minutes, not seconds. It is cached to
  CSV and keyed by the filters, so it is a once-a-day cost, not once a page
  load.
* **Prices for 2,400 names is a real download.** `load_universe_bars` batches
  it, but the first run takes a while and the cache runs to tens of megabytes.
  Volume comes back in the same yfinance response as the closes, so keeping it
  costs no extra network -- only disk -- and it is what lets the momentum
  screen apply its 300k-share volume test instead of skipping it.
* **This 300k floor and the leader rule's now measure the same thing**, which
  makes the leader leg close to non-binding here: a name in this universe
  already AVERAGES over 300k shares, so it fails the leg only on an unusually
  quiet session. That is fine -- it is a sanity check rather than a filter --
  but do not read the leader count as liquidity-screened beyond what this
  universe filter already did.
* **Finviz is a scrape, not an API.** It rate-limits, and the page layout is
  not a contract. Every entry point here returns None or raises a clear error
  rather than half a universe, because a breadth reading computed over a
  truncated sample is wrong in a way that looks plausible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DOWNLOAD_START

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
UNIVERSE_CSV = os.path.join(CACHE_DIR, "finviz_universe.csv")
BARS_DIR = os.path.join(CACHE_DIR, "universe")


@dataclass
class UniverseFilters:
    """The screener filters defining "the US market" for breadth purposes."""
    industry: str = "Stocks only (ex-Funds)"   # excludes ETFs and closed-end funds
    price: str = "Over $5"
    avg_volume: str = "Over 300K"

    def as_dict(self) -> Dict[str, str]:
        return {"Industry": self.industry, "Price": self.price,
                "Average Volume": self.avg_volume}

    @property
    def label(self) -> str:
        return f"US common + ADR, ex-funds · {self.price} · avg vol {self.avg_volume}"


def fetch_us_universe(
    filters: Optional[UniverseFilters] = None,
    refresh: bool = False,
    offline: bool = False,
    cache_path: str = UNIVERSE_CSV,
    max_age_days: int = 1,
    sleep_sec: int = 1,
    verbose: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """`(universe, error)` -- Ticker / Sector / Industry / Country for the US market.

    Returns `(None, reason)` rather than raising: the caller is a dashboard
    that falls back to the cached index universe. But it returns the REASON,
    which the first version did not -- it only printed when `verbose`, and the
    dashboard calls it with `verbose=False`, so every failure surfaced as the
    same useless "not available" message no matter what actually went wrong.
    A failure you cannot diagnose from the screen is a failure you cannot fix.

    The cache is reused while it is younger than `max_age_days`. Membership of
    "every liquid US common stock" moves slowly; re-scraping 120 pages on
    every app start to catch one delisting is not a trade worth making.
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
        return None, ("offline and no cached universe on disk — run once with "
                      "Source set to Online to build it")
    if cached is not None and not cached.empty and not refresh and (age or 0) <= max_age_days:
        if verbose:
            print(f"[finviz] universe cache hit: {len(cached)} tickers ({age}d old)")
        return cached, None

    try:
        try:
            from finvizfinance.screener.overview import Overview
        except ImportError as exc:
            raise RuntimeError(
                "finvizfinance is not installed in the environment running this "
                "app. Install it with `pip install -r requirements-dashboard.txt` "
                "(installing it in a notebook or Colab does not help here -- it "
                f"has to be the same interpreter running Streamlit). [{exc}]"
            ) from exc

        view = Overview()
        view.set_filter(filters_dict=filters.as_dict())
        df = view.screener_view(order="Ticker", verbose=0, sleep_sec=sleep_sec)
        if df is None or df.empty:
            raise RuntimeError("screener returned no rows")

        keep = [c for c in ("Ticker", "Company", "Sector", "Industry", "Country")
                if c in df.columns]
        if "Ticker" not in keep:
            raise RuntimeError(f"no Ticker column in {list(df.columns)[:8]}")

        out = df[keep].copy()
        out["Ticker"] = (out["Ticker"].astype(str).str.strip().str.upper()
                         .str.replace(".", "-", regex=False))
        out = out.drop_duplicates(subset="Ticker").reset_index(drop=True)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        out.to_csv(cache_path, index=False)
        if verbose:
            print(f"[finviz] {len(out)} tickers · {out['Sector'].nunique()} sectors"
                  if "Sector" in out.columns else f"[finviz] {len(out)} tickers")
        return out, None
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if verbose:
            print(f"[finviz] universe fetch failed ({reason})")
        if cached is not None and not cached.empty:
            return cached, f"using a stale cached universe — live fetch failed ({reason})"
        return None, reason


def sector_map(universe: Optional[pd.DataFrame]) -> Dict[str, str]:
    """`ticker -> sector`, empty when the screener did not supply sectors.

    Empty is deliberate and must stay that way: `breadth.sector_breakdown`
    returns an empty table for an empty map rather than bucketing everything
    into "Unclassified", which would render as a finding.
    """
    if universe is None or universe.empty or "Sector" not in universe.columns:
        return {}
    pairs = universe.dropna(subset=["Sector"])
    return dict(zip(pairs["Ticker"], pairs["Sector"]))


def load_universe_bars(
    tickers: List[str],
    start: str = DOWNLOAD_START,
    end: Optional[str] = None,
    cache_dir: str = BARS_DIR,
    prefix: str = "us",
    refresh: bool = False,
    offline: bool = False,
    batch_size: int = 100,
    verbose: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """`(closes, volumes, error)` for a wide universe, cached as two CSVs.

    Volume is kept because it arrives in the same yfinance response as the
    closes -- no extra network -- and it is the only thing standing between
    the momentum screen and its volume test. `universe.load_universe_prices`
    deliberately discards it, which is right for the ranking strategies and
    wrong here.

    A batch that fails is reported and skipped rather than aborting the run:
    losing 40 names out of 2,400 is a slightly smaller sample, and the row
    still carries `n_stocks` so the reading stays honest. Losing all of them
    returns the reason, so the UI can say what went wrong instead of only
    that something did.
    """
    os.makedirs(cache_dir, exist_ok=True)
    c_path = os.path.join(cache_dir, f"{prefix}_closes.csv")
    v_path = os.path.join(cache_dir, f"{prefix}_volumes.csv")

    def _read():
        if not (os.path.exists(c_path) and os.path.exists(v_path)):
            return None, None
        try:
            c = pd.read_csv(c_path, parse_dates=["Date"], index_col="Date")
            v = pd.read_csv(v_path, parse_dates=["Date"], index_col="Date")
            return c, v
        except Exception:  # noqa: BLE001
            return None, None

    if offline or not refresh:
        c, v = _read()
        if offline:
            err = None if c is not None else (
                "offline and no cached price frames on disk — run once with "
                "Source set to Online to build them")
            return c, v, err
        if c is not None and not c.empty:
            return c, v, None

    tickers = sorted({t for t in tickers if t})
    closes, volumes, failed = [], [], []
    last_error = "unknown"
    try:
        import yfinance as yf
    except ImportError as exc:
        if verbose:
            print("[finviz] yfinance is not installed")
        c, v = _read()
        return c, v, f"yfinance is not installed ({exc})"

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(batch, start=start, end=end, auto_adjust=True,
                              progress=False, actions=False, group_by="column",
                              threads=True)
            if raw is None or raw.empty:
                raise RuntimeError("empty response")
            if isinstance(raw.columns, pd.MultiIndex):
                c = raw["Close"]
                v = raw["Volume"] if "Volume" in raw.columns.get_level_values(0) else None
            else:
                c = raw[["Close"]].rename(columns={"Close": batch[0]})
                v = raw[["Volume"]].rename(columns={"Volume": batch[0]}) if "Volume" in raw else None
            closes.append(c)
            if v is not None:
                volumes.append(v)
        except Exception as exc:  # noqa: BLE001
            failed.extend(batch)
            last_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[finviz] batch {i // batch_size + 1} failed ({exc})")
        if verbose and (i // batch_size) % 5 == 0:
            print(f"[finviz] {min(i + batch_size, len(tickers))}/{len(tickers)} tickers")

    if not closes:
        if verbose:
            print("[finviz] no price data for any ticker")
        c, v = _read()
        return c, v, (f"no price data returned for any of {len(tickers)} tickers "
                      f"(every batch failed; last reason: {last_error})")

    cdf = pd.concat(closes, axis=1).sort_index()
    cdf = cdf.loc[:, ~cdf.columns.duplicated()]
    cdf.index = pd.to_datetime(cdf.index).tz_localize(None).normalize()
    cdf.index.name = "Date"
    cdf = cdf.drop(columns=[c for c in cdf.columns if cdf[c].notna().sum() == 0],
                   errors="ignore")

    vdf = None
    if volumes:
        vdf = pd.concat(volumes, axis=1).sort_index()
        vdf = vdf.loc[:, ~vdf.columns.duplicated()]
        vdf.index = pd.to_datetime(vdf.index).tz_localize(None).normalize()
        vdf.index.name = "Date"
        vdf = vdf.reindex(columns=cdf.columns)

    cdf.to_csv(c_path)
    if vdf is not None:
        vdf.to_csv(v_path)
    if verbose:
        print(f"[finviz] {cdf.shape[1]} tickers, {len(cdf)} rows"
              + (f" · no data for {len(set(failed))}" if failed else ""))
    warn = (f"{len(set(failed))} of {len(tickers)} tickers returned no data"
            if failed else None)
    return cdf, vdf, warn


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def diagnose(verbose: bool = True) -> Dict[str, str]:
    """Check every link in the chain and say which one is broken.

    `python -m qbs.finviz`

    Streamlit swallows tracebacks, so "the US universe is unavailable" on the
    dashboard could be a missing package, a blocked network, a rate limit or
    an empty result, and the screen cannot tell you which. This runs the same
    steps outside Streamlit and names the failure.

    The most common cause is the dullest: `finvizfinance` installed in a
    notebook or on Colab is not installed for the interpreter running the app.
    """
    import sys

    steps: Dict[str, str] = {}

    steps["python"] = sys.executable
    try:
        import finvizfinance
        steps["finvizfinance"] = f"OK (v{getattr(finvizfinance, '__version__', '?')})"
    except ImportError as exc:
        steps["finvizfinance"] = (
            f"MISSING — {exc}. Fix: pip install -r requirements-dashboard.txt "
            "using THIS interpreter")
        if verbose:
            _print_steps(steps)
        return steps

    try:
        from finvizfinance.screener.overview import Overview
        v = Overview()
        v.set_filter(filters_dict=UniverseFilters().as_dict())
        steps["filters"] = f"OK — {v.request_params.get('f')}"
    except Exception as exc:  # noqa: BLE001
        steps["filters"] = f"FAILED — {type(exc).__name__}: {exc}"
        if verbose:
            _print_steps(steps)
        return steps

    uni, err = fetch_us_universe(refresh=True, verbose=False)
    if uni is None:
        steps["screener"] = f"FAILED — {err}"
        if verbose:
            _print_steps(steps)
        return steps
    steps["screener"] = (f"OK — {len(uni)} tickers"
                         + (f", {uni['Sector'].nunique()} sectors"
                            if "Sector" in uni.columns else ", NO Sector column"))
    steps["sector_map"] = f"{len(sector_map(uni))} tickers mapped to a sector"

    try:
        import yfinance as yf
        probe = yf.download(uni["Ticker"].iloc[0], period="5d", progress=False,
                            auto_adjust=True)
        steps["yfinance"] = ("OK" if probe is not None and not probe.empty
                             else "FAILED — empty response for a probe ticker")
    except Exception as exc:  # noqa: BLE001
        steps["yfinance"] = f"FAILED — {type(exc).__name__}: {exc}"

    if verbose:
        _print_steps(steps)
    return steps


def _print_steps(steps: Dict[str, str]) -> None:
    width = max(len(k) for k in steps)
    print("\nFinviz universe diagnostics")
    print("-" * (width + 40))
    for k, v in steps.items():
        print(f"  {k:<{width}}  {v}")
    print()


if __name__ == "__main__":
    diagnose()
