"""Price data: download via yfinance, cache to CSV, or generate synthetic series.

The loader is deliberately boring. It returns one wide DataFrame of
split/dividend-adjusted closes indexed by date, columns = tickers, with no
gaps. Everything downstream assumes that shape.

Offline use
-----------
`load_prices(..., offline=True)` reads only the CSV cache and never touches the
network. `synthetic_prices()` builds a reproducible fake market so the whole
pipeline can be exercised (and unit-tested) with no data feed at all.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import DOWNLOAD_START, TICKERS

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# --------------------------------------------------------------------------
# Download / cache
# --------------------------------------------------------------------------

def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.csv")


def _read_cache(ticker: str) -> Optional[pd.Series]:
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if "Close" not in df.columns or df.empty:
        return None
    s = df["Close"].astype(float)
    s.name = ticker
    return s


def _write_cache(ticker: str, s: pd.Series) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    s.rename("Close").to_frame().rename_axis("Date").to_csv(_cache_path(ticker))


def _download_one(ticker: str, start: str, end: Optional[str]) -> pd.Series:
    import yfinance as yf  # imported lazily so offline use needs no yfinance

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,      # closes already adjusted for splits AND dividends
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")

    # yfinance hands back a MultiIndex column frame for some versions/queries.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    s = raw["Close"].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = ticker
    return s


def load_prices(
    tickers: Iterable[str] = TICKERS,
    start: str = DOWNLOAD_START,
    end: Optional[str] = None,
    use_cache: bool = True,
    refresh: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """Return adjusted closes for `tickers` as a wide, gap-free DataFrame.

    Parameters
    ----------
    use_cache : write each download to ``data/<TICKER>.csv`` and prefer it on
        the next call. Keeps repeat notebook runs instant and offline-safe.
    refresh : ignore the cache and re-download.
    offline : never hit the network; raise if the cache is missing.
    """
    tickers = list(tickers)
    series: List[pd.Series] = []
    missing: List[str] = []

    for t in tickers:
        s = None
        if use_cache and not refresh:
            s = _read_cache(t)
        if s is None:
            if offline:
                missing.append(t)
                continue
            s = _download_one(t, start, end)
            if use_cache:
                _write_cache(t, s)
        series.append(s)

    if missing:
        raise FileNotFoundError(
            f"offline=True but no cached data for {missing}. "
            f"Run once with offline=False to populate {CACHE_DIR}."
        )

    px = pd.concat(series, axis=1).sort_index()
    px.index = pd.to_datetime(px.index).normalize()
    px.index.name = "Date"

    # Union of calendars: hold the last known price across a ticker's
    # non-trading day rather than dropping the whole row.
    px = px.ffill()
    px = px.dropna(how="any")  # trims the leading stretch before the youngest ETF listed

    if end is not None:
        px = px.loc[:pd.Timestamp(end)]
    return px


# --------------------------------------------------------------------------
# Synthetic market (for tests and for exercising the pipeline offline)
# --------------------------------------------------------------------------

def load_vix(
    start: str = DOWNLOAD_START,
    end: Optional[str] = None,
    symbol: str = "^VIX",
    use_cache: bool = True,
    refresh: bool = False,
    offline: bool = False,
) -> pd.Series:
    """Daily closing level of the VIX (or any index symbol).

    Cached under the same ``data/`` directory as the ETFs. The circuit breaker
    reads the *close*, matching the rest of the package: decide on today's
    close, trade tomorrow.
    """
    key = symbol.replace("^", "_")
    if use_cache and not refresh:
        s = _read_cache(key)
        if s is not None:
            s.name = symbol
            return s
    if offline:
        raise FileNotFoundError(
            f"offline=True but no cached data for {symbol}. "
            f"Run once with offline=False to populate {CACHE_DIR}."
        )
    s = _download_one(symbol, start, end)
    if use_cache:
        _write_cache(key, s)
    s.name = symbol
    return s


def synthetic_vix(
    prices: pd.DataFrame,
    risk_asset: str = "QQQ",
    seed: int = 21,
    target_median: float = 17.5,
) -> pd.Series:
    """A VIX-shaped series derived from the synthetic market's own volatility.

    Real VIX is implied vol on the S&P, which spikes harder and faster than it
    decays and sits above realised vol most of the time. This reproduces the
    asymmetric spike-and-decay shape from the synthetic market's own
    volatility, then **rescales the whole series so its median matches
    `target_median`**.

    That rescaling is deliberate and matters for testing. VIX's long-run
    median is near 17-18, and a circuit breaker's behaviour depends entirely
    on where its trigger sits relative to that distribution. A synthetic VIX
    derived naively from a 23%-vol synthetic Nasdaq lands near 30, which would
    trip any sane threshold on day one and park forever -- exercising none of
    the state machine. Calibrating the median makes the offline test
    representative. It is still not a forecast of anything.
    """
    rng = np.random.default_rng(seed)
    rets = prices[risk_asset].pct_change()
    rv = rets.ewm(halflife=10, min_periods=5).std() * np.sqrt(252)

    # The gap between implied and realised widens when vol is low.
    premium = 1.0 + 0.55 * np.exp(-8 * rv.fillna(0.2))
    level = (100 * rv * premium).to_numpy(dtype=float)

    # Spikes arrive fast and decay slowly: a one-sided smoother.
    out = np.full(len(level), np.nan)
    prev = np.nan
    for i, x in enumerate(level):
        if np.isnan(x):
            continue
        prev = x if np.isnan(prev) else (x if x > prev else 0.90 * prev + 0.10 * x)
        out[i] = prev

    s = pd.Series(out, index=prices.index).bfill()
    med = float(s.median())
    if med > 0:
        s = s * (target_median / med)

    noise = rng.normal(0, 0.45, len(s))
    return pd.Series(np.clip(s.to_numpy() + noise, 9.0, 90.0),
                     index=prices.index, name="^VIX")


def synthetic_prices(
    start: str = "2023-06-01",
    end: str = "2026-09-08",
    seed: int = 7,
    tickers: Iterable[str] = TICKERS,
) -> pd.DataFrame:
    """A reproducible fake market with a QQQ-like, VEU-like and BOXX-like series.

    BOXX is generated as a near-deterministic upward drift (a box-spread ETF
    tracks T-bills), which is what makes it usable as the risk-free leg.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)

    spec = {
        "QQQ":  dict(mu=0.14, sigma=0.24, p0=380.0),
        "VEU":  dict(mu=0.07, sigma=0.16, p0=58.0),
        "BOXX": dict(mu=0.049, sigma=0.0015, p0=105.0),
    }

    # A shared market factor so QQQ and VEU are correlated, as real equities are.
    market = rng.standard_normal(n)

    out = {}
    for t in tickers:
        s = spec.get(t, dict(mu=0.08, sigma=0.20, p0=100.0))
        idio = rng.standard_normal(n)
        beta = 0.0 if t == "BOXX" else (1.0 if t == "QQQ" else 0.75)
        shock = beta * market + np.sqrt(max(1e-9, 1 - beta ** 2)) * idio
        dt = 1 / 252
        rets = (s["mu"] - 0.5 * s["sigma"] ** 2) * dt + s["sigma"] * np.sqrt(dt) * shock
        # Inject a drawdown regime so the defensive strategies have something to dodge.
        if t in ("QQQ", "VEU"):
            crash = (idx >= pd.Timestamp("2025-03-15")) & (idx <= pd.Timestamp("2025-05-10"))
            rets = np.where(crash, rets - 0.0035, rets)
        out[t] = s["p0"] * np.exp(np.cumsum(rets))

    px = pd.DataFrame(out, index=idx)
    px.index.name = "Date"
    return px
