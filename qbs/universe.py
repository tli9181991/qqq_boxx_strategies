"""The Nasdaq-100 ranking universe.

READ THIS BEFORE TRUSTING A BACKTEST BUILT ON `NDX_FALLBACK`
------------------------------------------------------------
Ranking against *today's* index membership is survivorship bias, and for a
momentum strategy it is the worst kind: today's Nasdaq-100 is, almost by
definition, a list of stocks that went up. A backtest that may only ever buy
from that list has been quietly told the answer in advance.

Over a 20-month window the damage is smaller than over 20 years, but the
Nasdaq-100 reconstitutes every December plus ad-hoc changes for M&A and
delistings, so it is not zero. Three ways to source the universe, best first:

1. **Point-in-time membership** -- `load_pit_universe(path)` reads a CSV of
   `date,ticker` rows and returns, for any date, the members as of that date.
   This is the only bias-free option. Vendors sell this data; some free
   reconstructions exist. If you are going to trade real money on this, buy it.
2. **Live fetch** -- `fetch_ndx()` scrapes the current constituent table.
   Current membership, applied to the whole history: still biased, but at least
   it is *today's* truth rather than a stale snapshot.
3. **`NDX_FALLBACK`** -- the hardcoded list below. Offline convenience only.
   It is approximate, it will drift out of date, and it is not verified against
   an official source.

`membership_mask()` turns any of these into a date x ticker boolean frame that
the strategy uses to mask ineligible names, so switching to point-in-time data
later changes one line and nothing else.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Offline fallback list
# --------------------------------------------------------------------------
# APPROXIMATE. Not verified against an official source. Prefer fetch_ndx() or
# a point-in-time file. Kept sorted so diffs are readable when you update it.

NDX_FALLBACK_ASOF = "2026-09"

NDX_FALLBACK: List[str] = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG",
    "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC",
    "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR", "MCHP",
    "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA",
    "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR",
    "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS", "TSLA",
    "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]

UNIVERSE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "universe"
)


# --------------------------------------------------------------------------
# Sourcing the constituent list
# --------------------------------------------------------------------------

def fetch_ndx(timeout: int = 20) -> List[str]:
    """Scrape the current Nasdaq-100 constituents.

    Needs network and `lxml`. Raises rather than silently returning a stale
    list -- a backtest that quietly fell back to hardcoded tickers is worse
    than one that stopped and told you.
    """
    import pandas as pd  # noqa: F811

    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        for key in ("ticker", "symbol"):
            if key in cols:
                col = [c for c in t.columns if str(c).strip().lower() == key][0]
                syms = (t[col].astype(str).str.strip().str.upper()
                        .str.replace(".", "-", regex=False))
                syms = [s for s in syms if s.isascii() and 1 <= len(s) <= 6]
                if len(syms) >= 90:
                    return sorted(set(syms))
    raise RuntimeError("could not locate a constituents table with >=90 tickers")


def load_universe(
    path: Optional[str] = None,
    fetch: bool = False,
    warn: bool = True,
) -> List[str]:
    """Return the ranking universe as a flat ticker list.

    `path` -- a text/CSV file with one ticker per line (or a `ticker` column).
    `fetch` -- try the live scrape, falling back only if it fails.
    """
    if path:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            return sorted(set(df[col].astype(str).str.strip().str.upper()))
        with open(path) as f:
            return sorted({ln.strip().upper() for ln in f if ln.strip()})

    if fetch:
        try:
            return fetch_ndx()
        except Exception as exc:  # noqa: BLE001
            if warn:
                print(f"[universe] live fetch failed ({exc}); using NDX_FALLBACK "
                      f"(as of {NDX_FALLBACK_ASOF}) -- results carry survivorship bias.")

    if warn:
        print(f"[universe] using NDX_FALLBACK (as of {NDX_FALLBACK_ASOF}). "
              "This is today's membership applied to all history: survivorship bias. "
              "See qbs/universe.py for how to supply point-in-time membership.")
    return sorted(NDX_FALLBACK)


def load_pit_universe(path: str) -> pd.DataFrame:
    """Read point-in-time membership from a CSV of `date,ticker` rows.

    Each row asserts "this ticker was a member on this date". Returns a tidy
    frame; pass it to `membership_mask` to get the boolean grid.
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "ticker" not in cols:
        raise ValueError("point-in-time file needs 'date' and 'ticker' columns")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[cols["date"]]).dt.normalize(),
        "ticker": df[cols["ticker"]].astype(str).str.strip().str.upper(),
    })
    return out.drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)


def membership_mask(
    index: pd.DatetimeIndex,
    tickers: Sequence[str],
    pit: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """date x ticker boolean grid of "was this name eligible to be ranked?".

    With `pit=None` every listed ticker is eligible on every date -- the
    survivorship-biased case. With a point-in-time frame, eligibility is
    forward-filled from each membership observation, so a name added in
    March 2025 is simply not rankable before then.
    """
    tickers = list(tickers)
    if pit is None or pit.empty:
        return pd.DataFrame(True, index=index, columns=tickers)

    grid = pd.DataFrame(False, index=sorted(set(pit["date"])), columns=tickers)
    for dt, grp in pit.groupby("date"):
        present = [t for t in grp["ticker"] if t in grid.columns]
        grid.loc[dt, present] = True

    return (grid.reindex(index.union(grid.index)).ffill()
            .reindex(index).fillna(False).astype(bool))


# --------------------------------------------------------------------------
# Prices for a wide universe
# --------------------------------------------------------------------------

def load_universe_prices(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
    cache_dir: str = UNIVERSE_DIR,
    refresh: bool = False,
    batch_size: int = 40,
    min_coverage: float = 0.6,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download adjusted closes for many tickers, cached as one parquet/CSV.

    Downloads in batches because yfinance gets unreliable with 100 symbols in
    one request. Tickers that come back empty (delisted, renamed, or simply
    unavailable) are reported and dropped rather than silently becoming NaN
    columns that the ranker would treat as missing data.
    """
    tickers = sorted(set(tickers))
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "universe_prices.csv")

    if os.path.exists(cache) and not refresh:
        px = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        have = [t for t in tickers if t in px.columns]
        if len(have) >= min_coverage * len(tickers):
            if verbose:
                print(f"[universe] cache hit: {len(have)}/{len(tickers)} tickers, "
                      f"{px.index.min():%Y-%m-%d} to {px.index.max():%Y-%m-%d}")
            return px[have].sort_index()

    import yfinance as yf

    frames, failed = [], []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        raw = yf.download(batch, start=start, end=end, auto_adjust=True,
                          progress=False, actions=False, group_by="column",
                          threads=True)
        if raw is None or raw.empty:
            failed.extend(batch)
            continue
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(batch[0])
        frames.append(close)
        failed.extend([t for t in batch if t not in close.columns
                       or close[t].notna().sum() == 0])

    if not frames:
        raise RuntimeError("no price data returned for any ticker in the universe")

    px = pd.concat(frames, axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()]
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px.index.name = "Date"
    px = px.drop(columns=[c for c in px.columns if px[c].notna().sum() == 0], errors="ignore")

    if verbose:
        print(f"[universe] {px.shape[1]} tickers, {len(px)} rows, "
              f"{px.index.min():%Y-%m-%d} to {px.index.max():%Y-%m-%d}")
        if failed:
            print(f"[universe] no data for {len(set(failed))}: {sorted(set(failed))}")

    px.to_csv(cache)
    return px


def synthetic_universe(
    n: int = 100,
    start: str = "2023-06-01",
    end: str = "2026-09-08",
    seed: int = 11,
) -> pd.DataFrame:
    """A fake 100-name universe with a genuine cross-sectional momentum effect.

    Each name gets a persistent drift that slowly mean-reverts, so past winners
    really do tend to keep winning for a while. Without that, a momentum
    backtest on random walks tests nothing -- it would rank pure noise, and a
    strategy that "works" on it would be a bug.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, end=end)
    T = len(idx)

    market = rng.standard_normal(T) * 0.011
    # Slow-moving per-name drift: an AR(1) on the annualised drift itself.
    drift = np.zeros((T, n))
    drift[0] = rng.normal(0.08, 0.22, n)
    phi = 0.995
    for t in range(1, T):
        drift[t] = phi * drift[t - 1] + rng.normal(0, 0.02, n)

    betas = rng.uniform(0.6, 1.5, n)
    idio = rng.uniform(0.16, 0.42, n)
    dt = 1 / 252
    shocks = rng.standard_normal((T, n))
    rets = (drift * dt
            + betas * market[:, None]
            + idio * np.sqrt(dt) * shocks)

    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    names = [f"SY{i:03d}" for i in range(n)]
    return pd.DataFrame(prices, index=idx, columns=names).rename_axis("Date")
