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

    from .data import normalise_symbols

    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for t in tables:
        cols = {str(c).strip().lower() for c in t.columns}
        for key in ("ticker", "symbol"):
            if key in cols:
                col = [c for c in t.columns if str(c).strip().lower() == key][0]
                syms = normalise_symbols(t[col])
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


def pit_tickers(pit: pd.DataFrame) -> List[str]:
    """Every ticker that was EVER a member, across the whole point-in-time file.

    This is the list to download prices for, and it is not the same list as
    today's index. A name that was a member in 2024 and dropped in 2025 has to
    be priced and rankable for its 2024 dates, or the backtest still cannot
    see it -- which is the survivorship bias the membership file was bought to
    remove. Masking today's constituents to their join dates fixes only the
    half of the problem where a name is ranked before it joined.
    """
    return sorted(set(pit["ticker"].astype(str).str.strip().str.upper()))


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

def _download_universe(tickers: List[str], start: str, end: Optional[str],
                       batch_size: int = 40):
    """`(closes, volumes, failed)` for `tickers` from `start`, batched.

    Volume arrives in the same response, so caching it costs disk and no
    network. The ranking strategies ignore it; research that needs a
    liquidity or participation test would otherwise have to re-download the
    whole universe to get a column already in hand.
    """
    import yfinance as yf

    frames, vframes, failed = [], [], []
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
        if list(close.columns) == ["Close"]:
            close = close.rename(columns={"Close": batch[0]})
        frames.append(close)
        if "Volume" in (raw.columns.get_level_values(0)
                        if isinstance(raw.columns, pd.MultiIndex) else raw.columns):
            v = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]
            if isinstance(v, pd.Series):
                v = v.to_frame(batch[0])
            if list(v.columns) == ["Volume"]:
                v = v.rename(columns={"Volume": batch[0]})
            vframes.append(v)
        failed.extend([t for t in batch if t not in close.columns
                       or close[t].notna().sum() == 0])

    def _frame(parts):
        if not parts:
            return None
        df = pd.concat(parts, axis=1).sort_index()
        df = df.loc[:, ~df.columns.duplicated()]
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "Date"
        return df

    px = _frame(frames)
    if px is not None:
        px = px.drop(columns=[c for c in px.columns if px[c].notna().sum() == 0],
                     errors="ignore")
    return px, _frame(vframes), failed


def load_universe_prices(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
    cache_dir: str = UNIVERSE_DIR,
    refresh: bool = False,
    batch_size: int = 40,
    min_coverage: float = 0.6,
    verbose: bool = True,
    incremental: bool = False,
) -> pd.DataFrame:
    """Download adjusted closes for many tickers, cached as one parquet/CSV.

    Downloads in batches because yfinance gets unreliable with 100 symbols in
    one request. Tickers that come back empty (delisted, renamed, or simply
    unavailable) are reported and dropped rather than silently becoming NaN
    columns that the ranker would treat as missing data.

    `incremental=True` (with `refresh=False`) turns a cache hit into an
    UPDATE: a short recent window is downloaded and folded in, and only names
    whose history the provider re-based -- or that the cache does not hold --
    are downloaded in full. See `qbs.incremental` for why a plain append is
    not safe. `refresh=True` is always a full download.
    """
    from .incremental import clear_marks, read_marks, refresh_incremental, write_marks

    tickers = sorted(set(tickers))
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "universe_prices.csv")
    vol_cache = os.path.join(cache_dir, "universe_volumes.csv")

    if os.path.exists(cache) and not refresh:
        px = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")
        have = [t for t in tickers if t in px.columns]
        covered = len(have) >= min_coverage * len(tickers)
        if covered and not incremental:
            if verbose:
                print(f"[universe] cache hit: {len(have)}/{len(tickers)} tickers, "
                      f"{px.index.min():%Y-%m-%d} to {px.index.max():%Y-%m-%d}")
            return px[have].sort_index()
        # An incremental update needs a cache that reaches back to `start`;
        # one built from a later start would never be filled in behind.
        if (covered and end is None and not px.empty
                and px.index.min() <= pd.Timestamp(start) + pd.Timedelta(days=10)):
            from .data import load_quarantine, record_failures

            banned = load_quarantine(cache_dir)
            wanted = [t for t in tickers if t not in banned]
            vol = (pd.read_csv(vol_cache, parse_dates=["Date"], index_col="Date")
                   if os.path.exists(vol_cache) else None)
            px, vol, marks, info = refresh_incremental(
                px, vol, read_marks(cache), wanted, start,
                lambda ts, since: _download_universe(ts, since, None, batch_size))
            if verbose:
                print(f"[universe] incremental since {info['since']}: "
                      f"{info['recent']} updated, {info['full']} re-downloaded "
                      f"in full" + (f" (re-based: {', '.join(info['rebased'])})"
                                    if info["rebased"] else ""))
            if info["failed"]:
                record_failures(cache_dir, info["failed"])
            px.to_csv(cache)
            if vol is not None:
                vol.reindex(columns=px.columns).to_csv(vol_cache)
            write_marks(cache, marks)
            return px[[t for t in tickers if t in px.columns]].sort_index()

    from .data import (QUARANTINE_DAYS, load_quarantine,
                       record_failures)

    # Symbols the provider refused recently are not asked again until the
    # quarantine expires -- see `qbs.data.load_quarantine` for why it expires
    # rather than being a permanent blacklist.
    banned = load_quarantine(cache_dir)
    skipped = [t for t in tickers if t in banned]
    if skipped:
        tickers = [t for t in tickers if t not in banned]
        if verbose:
            print(f"[universe] skipping {len(skipped)} quarantined: "
                  f"{', '.join(skipped[:10])}"
                  + (" ..." if len(skipped) > 10 else ""))

    px, vol, failed = _download_universe(tickers, start, end, batch_size)
    if px is None:
        raise RuntimeError("no price data returned for any ticker in the universe")

    if verbose:
        print(f"[universe] {px.shape[1]} tickers, {len(px)} rows, "
              f"{px.index.min():%Y-%m-%d} to {px.index.max():%Y-%m-%d}")
        if failed:
            print(f"[universe] no data for {len(set(failed))}: {sorted(set(failed))}")
            print(f"[universe] quarantined for {QUARANTINE_DAYS} days — they "
                  "will not be requested again until then")
    if failed:
        record_failures(cache_dir, failed)

    px.to_csv(cache)
    # A full download is yfinance from end to end: no screener cell survives it.
    clear_marks(cache)
    if vol is not None:
        vol.reindex(columns=px.columns).to_csv(vol_cache)
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


def load_universe_volumes(
    tickers: Optional[Iterable[str]] = None,
    cache_dir: str = UNIVERSE_DIR,
) -> Optional[pd.DataFrame]:
    """Share volumes for the universe, or None if they were never cached.

    Written as a side effect of `load_universe_prices`, which gets them free in
    the same yfinance response. Returns None rather than raising when the cache
    predates that -- a missing volume column should degrade a liquidity test to
    "not applied", never fail a price download that succeeded.
    """
    path = os.path.join(cache_dir, "universe_volumes.csv")
    if not os.path.exists(path):
        return None
    vol = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if tickers is not None:
        have = [t for t in tickers if t in vol.columns]
        if not have:
            return None
        vol = vol[have]
    return vol.sort_index()
