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
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DOWNLOAD_START, TICKERS, VOL_INDEX_3M

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


OHLC_CACHE = os.path.join(CACHE_DIR, "ohlc")


def load_daily_ohlc(
    ticker: str,
    start: str = DOWNLOAD_START,
    end: Optional[str] = None,
    refresh: bool = False,
    offline: bool = False,
    cache_dir: str = OHLC_CACHE,
) -> Optional[pd.DataFrame]:
    """Daily Open/High/Low/Close/Volume for ONE ticker, cached per name.

    Separate from `load_prices`, which is deliberately closes-only and one
    wide frame. A candlestick needs the other three columns, and it only ever
    draws one name at a time, so this fetches one name at a time rather than
    quadrupling the size of the universe cache for a chart.

    Returns None rather than raising when the data cannot be had: the caller
    is a chart that can fall back to a close line, and a missing candle is not
    worth taking the page down for.

    Never synthesise the missing columns from closes. `closes_to_bars` exists
    for indicator maths where a close-to-close envelope is a defensible
    stand-in for range; on a candlestick it would draw a body spanning
    previous-close to close with no wick at all, for every bar, which reads as
    a factual claim about the session's high and low that is simply untrue.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker.upper()}.csv")

    cached = None
    if os.path.exists(path) and not refresh:
        try:
            cached = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        except Exception:  # noqa: BLE001
            cached = None
    if offline:
        return cached if cached is not None and not cached.empty else None

    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                          progress=False, actions=False)
        if raw is None or raw.empty:
            raise RuntimeError("no rows returned")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume")
                if c in raw.columns]
        raw = raw[cols].dropna(subset=["Close"])
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        raw.index.name = "Date"
        fresh = raw if cached is None else pd.concat([cached, raw])
        fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
        fresh.to_csv(path)
        return fresh
    except Exception:  # noqa: BLE001
        return cached if cached is not None and not cached.empty else None


MARKET_TZ = "America/New_York"
MARKET_CLOSE = (16, 0)            # 16:00 ET, the regular-session close

# How long after the close before the day's bar counts as collectable.
#
# The bell is when the session ends, not when a provider has finished
# publishing it. Fetching at 16:01 returns a row that a handful of tickers
# have and the rest do not, and a union of per-ticker series turns that into
# a session-shaped row with twelve names in it -- see `drop_partial_bars`,
# which catches the ones that get through. This margin is the cheaper half of
# the fix: ask later and mostly do not get a torn read at all.
#
# An hour is a guess at a provider's settling time, not a documented SLA. It
# is a knob for that reason, and the coverage check is what actually
# guarantees correctness.
BAR_SETTLE = pd.Timedelta(minutes=60)


def normalise_symbols(symbols) -> "pd.Series":
    """Provider spellings turned into the one yfinance wants.

    Two substitutions, both to a dash, and each has cost a download:

    * ``.`` -- a share class. Finviz and Wikipedia write ``BRK.B``; Yahoo
      wants ``BRK-B``.
    * ``/`` -- a preferred series. The screener writes ``ORCL/PD``; Yahoo
      wants ``ORCL-PD``. Left alone it 404s as "possibly delisted; no
      timezone found", which reads like a dead company rather than a
      misspelled symbol, and there is one of these per issuer so they arrive
      in batches.

    One function rather than the same two-line chain in every provider,
    because a normalisation that only three of four callers apply produces
    keys that do not match the same price frame -- and the symptom is a name
    silently missing from a universe, not an error.
    """
    return (pd.Series(list(symbols), dtype="object").astype(str).str.strip()
            .str.upper()
            .str.replace(".", "-", regex=False)
            .str.replace("/", "-", regex=False))


QUARANTINE_DAYS = 14


def _quarantine_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "bad_tickers.csv")


def load_quarantine(cache_dir: str, days: int = QUARANTINE_DAYS,
                    now: Optional[pd.Timestamp] = None) -> Dict[str, dict]:
    """Symbols the provider has refused recently, and how often.

    Yahoo answers a symbol it does not know with "possibly delisted; no
    timezone found" -- once per ticker, per batch, per run. A universe with
    thirty of them buries every real message in the log and spends a slice of
    every download finding out the same thing again.

    Quarantine EXPIRES, deliberately. A permanent blacklist would shrink the
    universe by one name every time the provider had a bad minute, silently
    and for ever, and nothing would ever put a name back. A fortnight is long
    enough to stop the noise and short enough that a symbol which starts
    working is retried without anybody editing a file.
    """
    path = _quarantine_path(cache_dir)
    if not os.path.exists(path):
        return {}
    now = now or pd.Timestamp.now("UTC").tz_localize(None)
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return {}
    if "ticker" not in df.columns or "last_failed" not in df.columns:
        return {}
    out: Dict[str, dict] = {}
    for row in df.itertuples():
        try:
            last = pd.Timestamp(row.last_failed)
        except Exception:  # noqa: BLE001
            continue
        if (now - last).days < days:
            out[str(row.ticker)] = {
                "last_failed": last,
                "fails": int(getattr(row, "fails", 1) or 1),
            }
    return out


def record_failures(cache_dir: str, failed: Iterable[str],
                    now: Optional[pd.Timestamp] = None) -> None:
    """Add or refresh quarantine entries. Never raises.

    An unwritable cache costs a noisy log tomorrow, which is a great deal
    better than a download that will not start.
    """
    failed = sorted({str(t).strip().upper() for t in failed if str(t).strip()})
    if not failed:
        return
    now = now or pd.Timestamp.now("UTC").tz_localize(None)
    path = _quarantine_path(cache_dir)
    existing = load_quarantine(cache_dir, days=10 ** 6, now=now)
    for t in failed:
        prior = existing.get(t, {})
        existing[t] = {"last_failed": now,
                       "fails": int(prior.get("fails", 0)) + 1}
    try:
        os.makedirs(cache_dir, exist_ok=True)
        pd.DataFrame([{"ticker": t, "last_failed": v["last_failed"].isoformat(
            timespec="seconds"), "fails": v["fails"]}
            for t, v in sorted(existing.items())]).to_csv(path, index=False)
    except (OSError, ValueError):
        pass


def clear_quarantine(cache_dir: str) -> None:
    """Forget every quarantined symbol, so the next run retries them all."""
    try:
        os.remove(_quarantine_path(cache_dir))
    except OSError:
        pass


def drop_partial_bars(frame: pd.DataFrame, min_coverage: float = 0.5,
                      lookback: int = 20) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    """Remove trailing rows the provider had not finished publishing.

    Returns `(frame, dropped_dates, coverage)`, where `coverage` maps each
    dropped date to `(names_that_carried_it, names_a_normal_bar_has)`. The
    caller needs those two numbers: "a handful" cannot be told apart from a
    guard that is simply too strict, and that doubt is what sends somebody
    refreshing all morning.

    A wide download is a union of per-ticker series, so one ticker carrying
    today's bar puts today's DATE in the index for all of them -- everyone
    else NaN. Fetch just after the close, while the provider is still
    settling the tape, and you get a row with twelve names in it out of two
    and a half thousand.

    Nothing downstream can tell that row from a real session. Breadth counts
    the 4% movers among the twelve and reports zero, which reads as a flat
    tape rather than an empty one; the ranker ranks twelve names and calls it
    the universe. Both produce a confident number from almost no data, which
    is worse than producing none.

    Trailing rows only, and measured against the median of the `lookback`
    rows before them rather than against the frame's width. A universe grows
    and shrinks over years, so early history legitimately has fewer names
    than today and a fixed fraction of the column count would delete it. The
    thing being detected is a cliff at the end, not a thin patch.

    `min_coverage` is deliberately loose. A real session has essentially
    every name; the failure this catches is a factor of a hundred, not a
    borderline call, and a tight threshold would start eating half-holidays.
    """
    dropped: List[pd.Timestamp] = []
    coverage: Dict[pd.Timestamp, Tuple[int, int]] = {}
    if frame is None or frame.empty or len(frame) < 2:
        return frame, dropped, coverage
    # Filled in below so a caller can say 12-of-2610 rather than "a handful".
    # "A handful" is not a number anybody can act on: it cannot be told from
    # a guard that is too strict, which is exactly the doubt it creates.

    covered = frame.notna().sum(axis=1)
    end = len(frame)
    while end > 1:
        ref = covered.iloc[max(0, end - 1 - lookback):end - 1].median()
        if ref and covered.iloc[end - 1] < min_coverage * ref:
            day = frame.index[end - 1]
            dropped.append(day)
            coverage[day] = (int(covered.iloc[end - 1]), int(ref))
            end -= 1
        else:
            break
    if not dropped:
        return frame, dropped, coverage
    out = list(reversed(dropped))
    return frame.iloc[:end], out, {d: coverage[d] for d in out}


def last_market_close(now: Optional[pd.Timestamp] = None,
                      settle: Optional[pd.Timedelta] = None) -> pd.Timestamp:
    """The newest daily bar that should be collectable, as its close.

    Returns the 16:00 ET timestamp of that bar, so `.date()` is the bar's
    date, but the bar does not count as available until `settle` after it --
    see `BAR_SETTLE`. Pass `settle=pd.Timedelta(0)` for the exchange close
    itself, which is a different question and has no callers here.

    Weekdays only, so it over-reports around a holiday in the harmless
    direction: the fetch runs and finds nothing new, rather than not running
    when there is.
    """
    settle = BAR_SETTLE if settle is None else settle
    now = (pd.Timestamp.now(MARKET_TZ) if now is None
           else pd.Timestamp(now))
    now = (now.tz_localize("UTC") if now.tzinfo is None
           else now).tz_convert(MARKET_TZ)

    close = now.normalize() + pd.Timedelta(hours=MARKET_CLOSE[0],
                                           minutes=MARKET_CLOSE[1])
    # Walk back to the last weekday whose bar has had time to be published.
    while close + settle > now or close.weekday() >= 5:
        close -= pd.Timedelta(days=1)
    return close


def next_market_close(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """The next weekday 16:00 ET strictly after `now`, tz-aware.

    Only used to tell someone when the app will try again. A "nothing to do"
    message that does not say until when is the one that gets read as a
    fault.
    """
    now = (pd.Timestamp.now(MARKET_TZ) if now is None else pd.Timestamp(now))
    now = (now.tz_localize("UTC") if now.tzinfo is None
           else now).tz_convert(MARKET_TZ)

    close = now.normalize() + pd.Timedelta(hours=MARKET_CLOSE[0],
                                           minutes=MARKET_CLOSE[1])
    while close <= now or close.weekday() >= 5:
        close += pd.Timedelta(days=1)
    return close


def sessions_behind(last: pd.Timestamp,
                    now: Optional[pd.Timestamp] = None) -> int:
    """How many PUBLISHED daily bars are missing after `last`.

    0 means the newest bar the exchange has published is in the data. 1 or
    more means a bar exists and is not here, which is a real gap and not a
    time of day.

    Measured against the last close in EXCHANGE time. It used to compare
    against the UTC calendar date, which made "behind" a function of where
    you were standing: at 22:00 in New York the UTC date has already rolled
    over, so a cache holding that very afternoon's close reported itself one
    session behind, the banner said "the latest session is not in yet" about
    a bar that had been in for six hours, and the caller re-downloaded on
    every rerun trying to fetch a bar it already had.

    That also retires the old "1 is normal during a session" fudge, which
    existed only because the old reading could not tell a missing bar from a
    bar that did not exist yet. This one can: before the close, the newest
    published bar is yesterday's, so a cache holding it is current and says
    so.

    Weekdays only -- there is no exchange holiday calendar here. Around a
    market holiday this OVER-reports by a day, which is the safe direction
    for a staleness warning: it nags early rather than staying quiet while
    the data rots. Do not use it to decide whether a session existed.
    """
    last = pd.Timestamp(last)
    last = (last.tz_convert(MARKET_TZ) if last.tzinfo is not None
            else last).tz_localize(None).normalize()
    # The date of the newest bar the exchange has published.
    latest = last_market_close(now).tz_localize(None).normalize()
    if latest <= last:
        return 0
    return max(0, len(pd.bdate_range(last + pd.Timedelta(days=1), latest)))


def freshness_note(last: pd.Timestamp,
                   now: Optional[pd.Timestamp] = None) -> Tuple[int, str, str]:
    """`(sessions_behind, level, message)` for a UI banner.

    `level` is one of "ok", "info", "warn" so the caller picks the styling
    without re-deriving the rule.
    """
    n = sessions_behind(last, now)
    stamp = pd.Timestamp(last).strftime("%Y-%m-%d")
    if n == 0:
        return n, "ok", f"Data current through {stamp}."
    if n == 1:
        # Not "normal before the close" any more: 0 covers that case now, so
        # reaching 1 means a bar the exchange has published is genuinely not
        # here. Still only a caption -- the commonest cause is opening the
        # app in the minutes after a close, before the fetch has run.
        return n, "info", (f"Data through {stamp} — **the last published "
                           "session is missing**.")
    # Deliberately no remedy here: what to do about staleness depends on why
    # it happened, and the caller is the only one that knows. Telling someone
    # to "switch to Online" while their online download is failing is worse
    # than saying nothing.
    return n, "warn", f"Data through {stamp} — **{n} sessions behind**."


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


def load_vix3m(
    start: str = DOWNLOAD_START,
    end: Optional[str] = None,
    use_cache: bool = True,
    refresh: bool = False,
    offline: bool = False,
) -> pd.Series:
    """Daily close of the 3-month VIX -- the long leg of the term structure.

    Thin wrapper over `load_vix`, which already takes a symbol; it exists so
    callers do not have to remember the ticker and so the cache key is written
    the same way every time (`data/_VIX3M.csv`).
    """
    return load_vix(start=start, end=end, symbol=VOL_INDEX_3M,
                    use_cache=use_cache, refresh=refresh, offline=offline)


def synthetic_vix3m(
    vix: pd.Series,
    long_run: Optional[float] = None,
    damp: float = 0.55,
    halflife: int = 20,
    target_backwardation: float = 0.08,
    seed: int = 22,
) -> pd.Series:
    """A VIX3M-shaped companion to whatever VIX you hand it.

    Three-month implied vol is not a smoothed VIX, but for exercising a term
    structure signal the property that matters is the one that makes the curve
    informative: **the long leg barely moves when the short leg spikes**. So
    this damps VIX's deviation from its own long-run level and lags it, which
    puts the ratio above 1.0 in a spike and below it the rest of the time.

    Calibration, and why it is not optional
    ---------------------------------------
    The level is then scaled so the curve is inverted on about
    `target_backwardation` of days -- 8%, which is roughly what ^VIX/^VIX3M
    has done since 2010. This is the same device as `synthetic_vix`'s
    `target_median`, for the same reason: a term-structure trigger's entire
    behaviour is decided by where it sits in the signal's distribution, so an
    uncalibrated fixture does not exercise the strategy, it replaces it.

    It matters more here than it looks. Fed the REAL cached VIX with a premium
    tuned on the SYNTHETIC one, this produced inversion on 24% of days instead
    of 8% -- tripping three times too often and making the strategy look far
    worse than its rules are. Calibrating against the series actually supplied
    removes that particular way of being wrong.

    `long_run` defaults to the given series' own median, so the fixture does
    not assume its input is calibrated to anything in particular.

    ⚠️ It is still a fixture, not a forecast, and it does not reproduce the
    DEPTH of a real inversion: the ratio here tops out well short of the
    1.3-1.4 a real crisis reaches. Numbers computed on it say what the code
    does, never what the market does.
    """
    rng = np.random.default_rng(seed)
    v = vix.astype(float)
    base = float(v.median()) if long_run is None else float(long_run)

    smooth = v.ewm(halflife=halflife, min_periods=1).mean()
    # Pull the smoothed level back toward the long-run mean: a 3-month
    # contract prices a spike as mostly transitory.
    level = base + (smooth - base) * damp

    # Solve for the multiplicative premium that lands the inversion frequency
    # on target. ratio = v / (level * k) > 1  <=>  k < v / level, so the
    # required k is a quantile of that ratio -- one line, no search.
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = (v / level.where(level > 0)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(raw) and 0.0 < target_backwardation < 1.0:
        k = float(raw.quantile(1.0 - target_backwardation))
    else:
        k = 1.0
    level = level * max(k, 1e-6)

    noise = rng.normal(0.0, 0.25, len(level))
    out = np.clip(level.to_numpy() + noise, 9.0, 90.0)
    return pd.Series(out, index=vix.index, name=VOL_INDEX_3M)


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
