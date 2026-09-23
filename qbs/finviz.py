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
from .data import (MARKET_CLOSE, MARKET_TZ, drop_partial_bars,
                   last_market_close, next_market_close,
                   sessions_behind)

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
UNIVERSE_CSV = os.path.join(CACHE_DIR, "finviz_universe.csv")
BARS_DIR = os.path.join(CACHE_DIR, "universe")
FETCH_STAMP = os.path.join(BARS_DIR, "us_last_fetch.txt")


# --------------------------------------------------------------------------
# "Once per published bar" -- an ATTEMPT counter, not a success counter
# --------------------------------------------------------------------------
# Streamlit re-runs the whole script on every widget interaction, and the app
# is expected to refresh itself when the data is old. Those two together are a
# trap: at 10am the last bar is yesterday's, so "is the data stale?" says yes,
# a 2,400-name download runs, today's bar still does not exist because the
# market has not closed, and the next rerun asks the same question and gets the
# same answer. The app would download all day.
#
# So the gate is on the attempt, not on the data, and it is stamped whether
# the attempt succeeds or fails. The manual refresh button ignores it.
#
# What the gate counts, and why it is not a calendar day
# -----------------------------------------------------
# It was one attempt per UTC calendar day, and that quietly lost a session.
# Open the dashboard at 09:00 in UTC+8 -- 01:00 UTC -- and the day's only
# automatic attempt is spent hours before the close it was meant to collect.
# The bar for that session appears at 20:00 UTC and the gate refuses to fetch
# it until the UTC date rolls over. The app would report itself one session
# behind all evening with no way forward but the button.
#
# The gate is now the last CLOSE, in exchange time: an automatic attempt is
# due when the last attempt predates the most recently published bar. That is
# the honest form of the rule the calendar day was approximating -- fetch when
# there is something new to fetch, at most once per new bar -- and it fixes
# both failure modes at once. Before the close, "the most recent bar" is
# yesterday's, so a fetch that already has it does not run again; after the
# close it is today's, so the first interaction after 16:00 ET collects it.
#
# Exchange time, not UTC, so the boundary does not move with the seasons: the
# close is 20:00 UTC in summer and 21:00 UTC in winter, and a rule written in
# UTC is wrong for half the year.
#
# The cost of stamping failures too: if the network is down after the close it
# will not retry by itself until the next one. That is deliberate -- a silent
# retry loop on a broken connection is worse than a stale number with a button
# next to it -- and the UI says when the next automatic attempt is due.
#
# No holiday calendar here, same as `sessions_behind`. On a holiday the
# boundary still moves at 16:00 ET, so one attempt is spent finding out that
# no new bar exists. One wasted fetch a holiday is the right side to err on.
#
# `last_market_close` and `next_market_close` live in `qbs.data` beside
# `sessions_behind`, which measures staleness against the same boundary and
# is imported by this module -- putting them here would have made the two
# files import each other.


def fetch_epoch(now: Optional[pd.Timestamp] = None) -> str:
    """A key that changes exactly when a new bar becomes collectable.

    For a caller that memoises a load -- the dashboard's `st.cache_data`
    wrappers -- passing this in makes the memo expire on the schedule instead
    of living for the life of the process. A cache keyed only on its
    arguments pins a long-running app to whatever it read on start-up, and
    then the gate below is never even consulted, because the function holding
    it does not run.

    Keyed on the SLOT, not on the close, and the two are not
    interchangeable. Key it on the close and an app opened at 10:00 memoises
    "not due yet" under a key that will not change again until the next
    close -- which is AFTER the 15:00 slot -- so the scheduled fetch is
    memoised away and never happens. The gate and the key have to answer to
    the same clock.
    """
    return last_fetch_slot(now).isoformat(timespec="minutes")


def last_fetch_attempt(stamp_path: str = FETCH_STAMP) -> Optional[pd.Timestamp]:
    """When the app last TRIED to refresh the US universe, or None."""
    try:
        with open(stamp_path) as fh:
            return pd.Timestamp(fh.read().strip())
    except (OSError, ValueError):
        return None


def record_fetch_attempt(stamp_path: str = FETCH_STAMP,
                         now: Optional[pd.Timestamp] = None) -> None:
    """Stamp an attempt. Never raises -- an unwritable stamp must not break a
    fetch that otherwise worked; it only costs one extra attempt."""
    now = now or pd.Timestamp.now("UTC").tz_convert(None)
    try:
        os.makedirs(os.path.dirname(stamp_path), exist_ok=True)
        with open(stamp_path, "w") as fh:
            fh.write(now.isoformat(timespec="seconds"))
    except (OSError, ValueError):
        # ValueError too: `os.makedirs` raises it, not OSError, on a path the
        # OS will not even look at (an embedded null, say). Catching only
        # OSError turns an unusable stamp path into a dashboard that will not
        # start, which is a much worse failure than fetching twice.
        pass


def due_for_fetch(stamp_path: str = FETCH_STAMP,
                  now: Optional[pd.Timestamp] = None) -> Tuple[bool, str]:
    """`(due, why)` -- is an automatic refresh allowed right now?

    Two conditions, both required:

    1. the day's scheduled slot has passed since the last attempt, and
    2. a US session has closed since the last attempt.

    (1) is the rule: once a day, at the configured hour, and never at any
    other -- see the schedule section for why a gate and not a scheduler.
    (2) stops the slot spending a 2,600-name download on a day with nothing
    new in it: at 15:00 UTC+8 on a Sunday the newest bar is still Friday's,
    which Saturday's slot already collected.

    `why` is written to be rendered next to a stale date, so it says which of
    the two is holding and when the app will try again. A "nothing to do"
    that does not say until when is the one that gets read as a fault.
    """
    hh, mm, tz, cfg_note = fetch_schedule()
    now = pd.Timestamp.now(tz) if now is None else pd.Timestamp(now)
    now = (now.tz_localize("UTC") if now.tzinfo is None else now).tz_convert(tz)

    slot = last_fetch_slot(now)
    close = last_market_close(now)
    suffix = f" ({cfg_note})" if cfg_note else ""

    # Today's slot, or nothing. A missed slot is NOT collected late: opening
    # the app at 09:00 must not start a download, because "at 15:00, and
    # manually otherwise" is the whole rule. Skipping one costs only
    # freshness -- the download is the full history every time, not an
    # increment, so the next slot picks up both days.
    if slot.normalize() != now.normalize():
        return False, (f"before today's {hh:02d}:{mm:02d} {tz} — automatic "
                       f"fetches run then and at no other time; use Refresh "
                       f"now for one immediately" + suffix)

    last = last_fetch_attempt(stamp_path)
    if last is None:
        return True, "no automatic fetch has run yet" + suffix

    # The stamp is written in naive UTC and the slot is in the reader's zone.
    # Comparing them without converting is a whole class of bug this file has
    # had before, one layer down.
    last_tz = (last.tz_localize("UTC") if last.tzinfo is None
               else last).tz_convert(tz)
    nxt = next_fetch_slot(now)

    if last_tz >= slot:
        return False, (f"already fetched today at {last_tz:%H:%M} — next "
                       f"automatic attempt {nxt:%a} {nxt:%H:%M} {tz}, or use "
                       f"Refresh now" + suffix)
    if last_tz >= close.tz_convert(tz):
        return False, (f"nothing new since the {close:%b} {close.day} close, "
                       f"already fetched at {last_tz:%H:%M} — next automatic "
                       f"attempt {nxt:%a} {nxt:%H:%M} {tz}, or use Refresh now"
                       + suffix)
    return True, (f"scheduled fetch for {slot:%a} {slot:%H:%M} {tz}; last "
                  f"attempt {last_tz:%Y-%m-%d %H:%M}" + suffix)


@dataclass
class UniverseFilters:
    """The screener filters defining "the US market" for breadth purposes.

    Two vocabularies for one definition, because the providers do not speak
    the same language. `min_price` and `min_avg_volume` are the DEFINITION --
    numbers, provider-neutral, what the TradingView scanner is handed and
    what anything new should read. The three strings are Finviz's own filter
    enum, which takes "Over $5" and not 5.0 and has no general numeric form,
    so they cannot be derived.

    They must agree, and `test_the_two_filter_vocabularies_agree` is what
    keeps them agreeing: a definition that says one thing to one provider and
    another to the next produces two different universes and one number on
    screen.
    """
    industry: str = "Stocks only (ex-Funds)"   # excludes ETFs and closed-end funds
    price: str = "Over $5"
    avg_volume: str = "Over 300K"

    # The same three rules as numbers. `include_dr` keeps ADRs, which the
    # Finviz "Stocks only (ex-Funds)" industry filter also keeps -- the label
    # has always said "US common + ADR".
    min_price: float = 5.0
    min_avg_volume: float = 300_000.0
    include_dr: bool = True

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


# --------------------------------------------------------------------------
# When the automatic fetch is allowed to run
# --------------------------------------------------------------------------
# A wall-clock slot in the reader's own timezone, once a day, and nothing at
# any other hour. The default is 15:00 in UTC+8: eleven hours after the
# previous US close, so the tape has long settled, and the middle of the
# afternoon for somebody in Asia rather than the middle of their night.
#
# This is a GATE, not a scheduler. Streamlit runs code when something
# interacts with it and at no other time, so "fetch at 15:00" means "the
# first run at or after 15:00 fetches, and the rest of the day does not". Open
# the app at 18:00 and it fetches then; open it at 10:00 and it will not. For
# a download that happens whether or not anyone is looking, point cron or
# Task Scheduler at the app's own fetch -- a dashboard cannot do it, because
# a dashboard nobody has open is not running.
#
# Both conditions have to hold, and the second is why Sunday does not spend a
# download: the slot has to have passed since the last attempt, AND a US
# session has to have closed since the last attempt. At 15:00 UTC+8 on a
# Sunday the newest bar is still Friday's, which Saturday's slot already
# collected.

FETCH_AT_VAR = "QBS_FETCH_AT"
FETCH_TZ_VAR = "QBS_FETCH_TZ"
DEFAULT_FETCH_AT = "15:00"
DEFAULT_FETCH_TZ = "Asia/Hong_Kong"   # UTC+8; Singapore/Taipei/Shanghai are identical


def fetch_schedule(environ: Optional[Dict[str, str]] = None
                   ) -> Tuple[int, int, str, Optional[str]]:
    """`(hour, minute, tz, complaint)` for the daily automatic fetch.

    A bad value falls back to the default and is REPORTED rather than raised.
    This is read on a dashboard's start-up path, so a typo in a `.env` should
    cost a line on screen, not a page that will not load -- and a silent
    fallback would have someone waiting all afternoon for a fetch scheduled
    at an hour they think they changed.
    """
    env = os.environ if environ is None else environ
    bad: List[str] = []

    raw = (env.get(FETCH_AT_VAR) or DEFAULT_FETCH_AT).strip()
    try:
        hh, mm = (int(x) for x in raw.split(":", 1))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(raw)
    except (ValueError, TypeError):
        bad.append(f"{FETCH_AT_VAR}={raw!r} is not HH:MM")
        hh, mm = (int(x) for x in DEFAULT_FETCH_AT.split(":"))

    tz = (env.get(FETCH_TZ_VAR) or DEFAULT_FETCH_TZ).strip()
    try:
        pd.Timestamp("2026-01-01", tz=tz)
    except Exception:  # noqa: BLE001
        bad.append(f"{FETCH_TZ_VAR}={tz!r} is not a timezone pandas knows")
        tz = DEFAULT_FETCH_TZ

    note = ("; ".join(bad) + f" — using {hh:02d}:{mm:02d} {tz}") if bad else None
    return hh, mm, tz, note


def last_fetch_slot(now: Optional[pd.Timestamp] = None,
                    environ: Optional[Dict[str, str]] = None) -> pd.Timestamp:
    """The most recent scheduled fetch moment at or before `now`, tz-aware."""
    hh, mm, tz, _ = fetch_schedule(environ)
    now = pd.Timestamp.now(tz) if now is None else pd.Timestamp(now)
    now = (now.tz_localize("UTC") if now.tzinfo is None else now).tz_convert(tz)

    slot = now.normalize() + pd.Timedelta(hours=hh, minutes=mm)
    if slot > now:
        slot -= pd.Timedelta(days=1)
    return slot


def next_fetch_slot(now: Optional[pd.Timestamp] = None,
                    environ: Optional[Dict[str, str]] = None) -> pd.Timestamp:
    """The next scheduled fetch moment strictly after `now`, tz-aware.

    Only used to tell someone when the app will try again. A "nothing to do"
    message that does not say until when is the one that gets read as a fault.
    """
    return last_fetch_slot(now, environ) + pd.Timedelta(days=1)


def _torn_note(torn: List[pd.Timestamp]) -> Optional[str]:
    """What to say about a bar the provider had not finished publishing.

    Named rather than dropped in silence: a session that vanishes with no
    explanation reads as a failed download, and the remedy for the two is
    not the same.
    """
    if not torn:
        return None
    days = ", ".join(f"{d:%Y-%m-%d}" for d in sorted(set(torn)))
    return (f"dropped {days} — the download landed before the provider had "
            f"published most of the universe, so that bar held only a handful "
            f"of names. It is not counted as a session and does not count as "
            f"a fresh cache; the next fetch will pick it up, or press "
            f"**Refresh now**.")


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
    stale_after: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[str]]:
    """`(closes, volumes, error)` for a wide universe, cached as two CSVs.

    `stale_after` is how many trading sessions behind the cache may be before
    a hit stops counting and the download runs anyway. Without it a cache hit
    is returned whatever its age, which pins a long-running app to whatever
    happened to be on disk when it started.

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
    torn: List[pd.Timestamp] = []

    def _whole(c, v):
        """Trailing rows the provider had not finished publishing, removed.

        Applied BEFORE the staleness check below, not after, and that
        ordering is the whole point. A torn bar carries the current date, so
        a cache holding one reports itself current, the download that would
        replace it never runs, and the app sits on the last good session
        until somebody presses the button. The check has to be asked about
        the last WHOLE bar, not the last row.
        """
        if c is None or c.empty:
            return c, v
        c, dropped = drop_partial_bars(c)
        if dropped:
            torn.extend(dropped)
            if v is not None and not v.empty:
                v = v.loc[:c.index.max()]
        return c, v

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
            c, v = _whole(c, v)
            err = None if c is not None else (
                "offline and no cached price frames on disk — run once with "
                "Source set to Online to build them")
            return c, v, err or _torn_note(torn)
        c, v = _whole(c, v)
        if c is not None and not c.empty:
            # A cache HIT that never asks how old it is pins the app to
            # whatever is on disk for ever. `stale_after` is the number of
            # trading sessions behind at which the hit stops counting; None
            # keeps the old behaviour for callers that manage freshness
            # themselves.
            if stale_after is None:
                return c, v, _torn_note(torn)
            behind = sessions_behind(c.index.max())
            if behind < stale_after:
                return c, v, _torn_note(torn)
            if verbose:
                print(f"[finviz] price cache is {behind} session(s) behind "
                      f"(limit {stale_after}) — re-downloading")

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

    # The cache keeps what arrived; the caller gets what is whole. Writing
    # the trimmed frame would throw away the dozen names that DID report,
    # and they are the head of a real bar -- the next fetch fills the rest
    # in rather than starting over.
    cdf.to_csv(c_path)
    if vdf is not None:
        vdf.to_csv(v_path)
    cdf, vdf = _whole(cdf, vdf)
    if verbose:
        print(f"[finviz] {cdf.shape[1]} tickers, {len(cdf)} rows"
              + (f" · no data for {len(set(failed))}" if failed else ""))
    warn = (f"{len(set(failed))} of {len(tickers)} tickers returned no data"
            if failed else None)
    return cdf, vdf, "; ".join(x for x in (warn, _torn_note(torn)) if x) or None


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
