"""Streamlit dashboard: daily strategy selections, and a market breadth monitor.

    streamlit run dashboard/app.py

Four tabs:

* **Daily picks** -- what each of the three selection strategies held on each
  day, with the entries and exits that changed it. The momentum book ranks the
  universe and is always invested; the residual book ranks the same universe
  on what the market cannot explain; the high-momentum screen applies an
  absolute bar and can hold almost nothing.
* **Market overview** -- the breadth monitor: 4% movers, percent holding the
  moving averages, index stretch in ATR units, and the momentum-leader group.
* **News & sentiment** -- the last 12 hours of market headlines, always; plus
  a model's read of them when one is configured and switched on.
* **Analyst** -- the price panel again, and a chat over the lab's own tools.

READ THE BANNER AT THE TOP OF THE MARKET TAB
--------------------------------------------
Breadth is a statement about a universe. The dashboard this mirrors samples
~2,400 US common stocks and ADRs; this package ships ~99 Nasdaq-100 daily
closes. The arithmetic is identical and the readings are not comparable -- a
"4% up" count out of 99 mega-caps is a different measurement from one out of
2,432 names, not a smaller version of it. The app states its own sample size
on every screen for that reason, and leaves blank the panels it cannot
honestly fill (sector concentration needs a sector map; the turnover leg of
the leadership screen needs volume; the S&P column needs an index this
package does not cache).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.breadth import (BreadthParams, atr_class, bear_checklist,
                         daily_breadth, ma_class,
                         ma_fast_cell, momentum_label, momentum_profile,
                         pulse_cell, pulse_class, sector_breakdown,
                         sector_leaders)
from qbs.breakout import closes_to_bars, levels_in_view, sr_levels
from qbs.config import (BreakoutParams, Config, FinvizScreenParams,
                        MomentumParams, ResidualMomentumParams)
from qbs.data import (MARKET_TZ, drop_partial_bars, freshness_note,
                      load_daily_ohlc, load_prices, session_close,
                      sessions_behind)
from qbs.finviz import (BARS_DIR, UniverseFilters, due_for_fetch, fetch_epoch,
                        fetch_schedule, load_universe_bars,
                        record_fetch_attempt, sector_map)
from qbs.quotes import (fill_disabled, fill_last_bar, fill_note,
                        latest_quotes, needs_fill)
from qbs.universe_source import (SOURCE_VAR, available_sources, fetch_universe,
                                 resolve_source)
from qbs.screens import finviz_momentum_screen
from qbs.candles import HammerRules, hammer_frame, volume_stats
from qbs.shadow import (parse_watchlist, watchlist_residual_ranks,
                        watchlist_rows)
from qbs.strategies import cross_sectional_momentum, residual_momentum
from qbs.incremental import persist_fill
from qbs.universe import UNIVERSE_DIR, load_universe, load_universe_prices

st.set_page_config(page_title="Strategy picks / Market overview", layout="wide",
                   initial_sidebar_state="expanded")

MUTED = "#898781"
UP, UP_STRONG = "#a9d7bd", "#1b7a4b"
DN, DN_STRONG = "#f0b2b2", "#b02525"
CELL = {"extreme_low": "#f6c9c9", "low": "#fbe6e6", "mid": "",
        "high": "#ddefe4", "extreme_high": "#a9d7bd", "none": "",
        "stretched": "#f6c9c9", "oversold": "#c9e6d5", "normal": ""}

# The four-band scale for the daily monitor's 4%-mover columns. Green is
# bullish on BOTH, so a quiet down-4% count shades green like a heavy up-4%
# one -- `breadth.pulse_cell` owns which band a value lands in.
PULSE_CELL = {"dark_green": UP_STRONG, "light_green": UP,
              "light_red": DN, "dark_red": DN_STRONG, "none": ""}

# How many slots the residual book runs here. The research default is six,
# the same as its total-return sibling, because the point of that comparison
# is that ONLY the score differs. This dashboard runs it deeper on purpose:
# the residual score is a risk-adjusted one and its whole claim is that the
# names it picks are less alike, so ten slots of it is not ten times the same
# bet the way ten momentum slots would be. See docs/RESIDUAL_MOMENTUM.md.
RESID_N_HOLD = 10
# The band, kept as a WIDTH rather than an absolute rank. `exit_rank` is how
# far a held name may slip before it is sold, and the sweep that validated it
# varied the pair together -- carrying the number 10 over to a ten-name book
# would be a band of zero, i.e. a round-trip every time a name wobbles.
RESID_BAND = (ResidualMomentumParams().exit_rank
              - ResidualMomentumParams().n_hold)


def strategy_labels(n_mom: int, n_screen: int, n_resid: int) -> Dict[str, str]:
    """The three book names, counting the slots each one is actually running.

    Every count is derived, never written out, for the same reason the
    lookback is: each of these has already moved once, and a header claiming
    the old value is a quiet lie on every screenshot. They read the live
    sidebar values rather than the config defaults, because the sidebar is
    what built the book on screen.
    """
    return {
        "momentum": f"Top-{n_mom} NDX momentum ({momentum_label()})",
        "resmom": (f"Top-{n_resid} residual momentum "
                   f"({momentum_label(ResidualMomentumParams())})"),
        "finviz": f"Top-{n_screen} high momentum screen",
    }


# The defaults, so anything importing this module before the sidebar renders
# still has a name for each book. Rebuilt from the sidebar further down.
STRATEGY_LABELS = strategy_labels(
    Config().momentum.n_hold, FinvizScreenParams().n_hold, RESID_N_HOLD)
PULSE_LABELS = {"up_strong": f"Up 4% ≥ {BreadthParams().pulse_strong}",
                "up": f"Up 4% < {BreadthParams().pulse_strong}",
                "down": f"Down 4% < {BreadthParams().pulse_strong}",
                "down_strong": f"Down 4% ≥ {BreadthParams().pulse_strong}"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

# The on-disk caches a screener top-up is saved into: (closes, volumes).
PICKS_CACHE = (os.path.join(UNIVERSE_DIR, "universe_prices.csv"),
               os.path.join(UNIVERSE_DIR, "universe_volumes.csv"))
MARKET_CACHE = (os.path.join(BARS_DIR, "us_closes.csv"),
                os.path.join(BARS_DIR, "us_volumes.csv"))


def _top_up_front_bar(closes, volumes, online: bool, persist=None):
    """The newest close from the screener, when yfinance has not published it.

    Returns `(closes, volumes, note)`. yfinance stays the authority for every
    settled bar; this only reaches names whose newest one is missing, and only
    once the session has closed -- see `qbs.quotes` for why an open session is
    refused rather than filled.

    Asked only when a bar is actually short, so a normal day costs no screener
    request at all.

    `persist` is a `(closes_csv, volumes_csv)` pair: the filled cells are
    written there, marked provisional, so the next load finds the session on
    disk and needs no download at all. The next download that does run
    replaces them with yfinance's values -- see `qbs.incremental`.
    """
    if not online or fill_disabled() or not needs_fill(closes):
        return closes, volumes, None
    quotes, err = latest_quotes()
    if quotes is None:
        return closes, volumes, f"could not top up the newest bar — {err}"
    closes, volumes, report = fill_last_bar(closes, volumes, quotes)
    if persist and report.get("tickers"):
        names, session = report["tickers"], report["session"]
        vol = (quotes["volume"].reindex(names)
               if "volume" in quotes.columns else None)
        persist_fill(persist[0], session, closes.loc[session, names],
                     persist[1], vol)
    return closes, volumes, fill_note(report)


def _read_cache(download_start: str, fetch_members: bool = False):
    """Whatever is on disk, without touching the network.

    `fetch_members` is about the INDEX membership list, not the US universe
    that `qbs.universe_source.fetch_universe` supplies -- it was called
    `fetch_universe` and shadowed that import, which is a rename waiting to
    turn into a bug the day someone uses the function here.
    """
    tickers = load_universe(fetch=fetch_members, warn=False)
    uni = load_universe_prices(tickers, start=download_start, verbose=False)
    px = load_prices(["QQQ", "VEU", "BOXX"], start=download_start, offline=True)
    return uni, px


@st.cache_data(show_spinner="Loading prices…")
def load_data(download_start: str, online: bool, force: bool, bar_epoch: str,
              _token: int):
    """Closes for the ranking universe plus the core ETFs.

    Returns `(uni, px, status)` where `status` explains what actually
    happened, because "online" and "fresh" are not the same thing and the UI
    must not imply otherwise.

    Why online mode is not simply `offline=False`
    ---------------------------------------------
    `load_universe_prices` returns a cache HIT without asking how old it is,
    so the download only happens when `refresh=True`. Flipping the offline
    flag alone would have left the app pinned to whatever the cache held --
    which is exactly the bug this is fixing. So online mode reads the cache
    first, measures how far behind it is, and re-downloads only when there is
    something to fetch. A fresh cache costs no network at all.

    A failed download falls back to the cache and says so. Showing stale
    numbers is fine; showing them while claiming to be live is not.
    """
    status = {"mode": "online" if online else "offline", "downloaded": False,
              "error": None, "partial": [], "filled": None, "carried": None}

    uni, px = None, None
    try:
        uni, px = _read_cache(download_start)
    except Exception as exc:  # noqa: BLE001
        if not online:
            raise
        status["error"] = f"no usable cache ({exc})"

    # Behind is asked of the UNIVERSE's last whole bar, which counts a
    # screener top-up saved on an earlier load: once today's closes are on
    # disk, a reload needs no network. The ETFs are never topped up, so a
    # one-session lag there is carried forward below (and said so) rather
    # than treated as a reason to download.
    behind_u = (sessions_behind(drop_partial_bars(uni)[0].index.max())
                if uni is not None and not uni.empty else 99)
    behind_px = sessions_behind(px.index.max()) if px is not None else 99
    if online and (force or uni is None or behind_u >= 1 or behind_px >= 2):
        try:
            # "Refresh now" is a full download; anything else is an update
            # that fetches a recent window and re-downloads only the names
            # the provider re-based (`qbs.incremental`).
            tickers = load_universe(fetch=True, warn=False)
            uni = load_universe_prices(tickers, start=download_start,
                                       refresh=bool(force), incremental=True,
                                       verbose=False)
            px = load_prices(["QQQ", "VEU", "BOXX"], start=download_start,
                             refresh=bool(force), incremental=True,
                             offline=False)
            status["downloaded"] = True
            status["error"] = None
        except Exception as exc:  # noqa: BLE001
            status["error"] = str(exc)
            if uni is None or px is None:
                raise

    # BEFORE the reindex-and-ffill, which would paper a torn bar over with
    # yesterday's prices and make every name look unchanged on the day.
    uni, torn, cover = drop_partial_bars(uni)
    status["partial"] = [
        f"{d:%Y-%m-%d}" + (f" ({cover[d][0]} of ~{cover[d][1]} names)"
                           if d in cover else "")
        for d in torn]
    # The bar yfinance dropped is the one the screener already has. Done
    # after the torn one is gone, so the fill lands on a clean frame rather
    # than beside a dozen stragglers.
    uni, _, status["filled"] = _top_up_front_bar(uni, None, online,
                                                 persist=PICKS_CACHE)
    if uni.index.max() < px.index.max():
        px = px.loc[:uni.index.max()]
    elif uni.index.max() > px.index.max():
        # The screener filled a session the ETFs do not have. Reindexing onto
        # `px.index` here would throw that bar straight back out -- the fill
        # would appear to work, say so on screen, and change nothing. The
        # ETFs are carried forward instead, and it is reported: BOXX is a
        # T-bill proxy so one stale session moves the absolute filter by
        # basis points, but QQQ carried forward reads as an unchanged day
        # rather than an unknown one, and that is worth saying out loud.
        status["carried"] = f"{uni.index.max():%Y-%m-%d}"
        px = px.reindex(px.index.union(uni.index)).ffill()

    uni = uni.reindex(px.index).ffill()
    uni = uni.loc[:, uni.notna().sum() >= 260]
    return uni, px, status


@st.cache_data(show_spinner="Building selections…")
def build_selections(_uni: pd.DataFrame, _safe: pd.Series, _market: pd.Series,
                     n_hold: int, exit_rank: int, n_screen: int,
                     n_resid: int,
                     bar_epoch: str = "") -> Dict[str, pd.DataFrame]:
    """Daily holdings for each strategy, plus whether the volume leg ran.

    Returns `(frames, volume_applied)`. The screen RAISES on a volume leg it
    cannot apply rather than skipping one, so this either supplies volumes or
    opts out explicitly -- and the caller has to be told which, because
    without the leg the screen is more permissive than its own definition.

    `_market` is the factor the residual book regresses against -- QQQ, the
    same series the breadth monitor uses. It is passed in rather than read
    here so this function keeps taking every price it needs from its caller,
    which is what makes the cache key honest.
    """
    from qbs.config import MomentumParams
    from qbs.universe import load_universe_volumes

    out: Dict[str, pd.DataFrame] = {}

    mom = cross_sectional_momentum(
        _uni, _safe, MomentumParams(n_hold=n_hold, exit_rank=exit_rank))

    # Same universe, same safe asset, same slot machinery, same absolute
    # filter against BOXX -- only the score it sorts on differs. That is the
    # whole point of the strategy, and it is also why the two columns can be
    # read side by side: anything they disagree about is the market component
    # of the ranking, and nothing else.
    res = residual_momentum(
        _uni, _safe, _market,
        ResidualMomentumParams(n_hold=n_resid,
                               exit_rank=n_resid + RESID_BAND))

    vols = load_universe_volumes(list(_uni.columns))
    volume_applied = vols is not None and not vols.empty
    screen = FinvizScreenParams(n_hold=n_screen)
    if volume_applied:
        vols = vols.reindex(index=_uni.index).ffill()
    else:
        vols, screen = None, FinvizScreenParams(n_hold=n_screen, min_volume=None)
    fin = finviz_momentum_screen(_uni, _safe, screen, volumes=vols)

    for key, sig in (("momentum", mom), ("resmom", res), ("finviz", fin)):
        ev = sig.events
        rows = []
        for d, names in sig.holdings_log.items():
            day = ev[ev["date"] == d] if not ev.empty else ev
            rows.append({
                "date": d,
                "holdings": ", ".join(names),
                "n": len(names),
                "buys": ", ".join(day.loc[day["action"] == "buy", "asset"]) if len(day) else "",
                "sells": ", ".join(day.loc[day["action"] == "sell", "asset"]) if len(day) else "",
            })
        out[key] = pd.DataFrame(rows).set_index("date")

    return out, volume_applied


@st.cache_data(show_spinner="Loading watchlist prices…")
def load_watch_prices(tickers: tuple, download_start: str, online: bool,
                      through: str, _token: int):
    """Closes for watched names that are NOT in the ranking universe.

    Returns `(frame, errors)`. Each name is fetched on its own and a failure
    is reported by name rather than raised: a typo in a watchlist is the
    common case, and it should cost that one row, not the panel.

    The cache is read first and re-downloaded only when it ends before
    `through` (the universe's last bar). A watched name a week behind the
    book would be ranked on a week-old price against today's constituents,
    which is a comparison of two different days dressed up as one.
    """
    rows: Dict[str, pd.Series] = {}
    errors: Dict[str, str] = {}
    last = pd.Timestamp(through)
    for t in tickers:
        s, err = None, None
        try:
            s = load_prices([t], start=download_start, offline=True)[t]
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        if online and (s is None or s.index.max() < last):
            try:
                # An update when there is a cache to update, a full
                # download when there is not.
                s = load_prices([t], start=download_start,
                                refresh=s is None, incremental=True,
                                offline=False)[t]
                err = None
            except Exception as exc:  # noqa: BLE001
                if s is None:
                    err = f"{type(exc).__name__}: {exc}"
        if s is None:
            errors[t] = err or "no data"
        else:
            rows[t] = s
    return (pd.DataFrame(rows) if rows else pd.DataFrame()), errors


@st.cache_data(show_spinner="Ranking the watchlist…")
def watch_rows(_uni: pd.DataFrame, _safe: pd.Series, _market: pd.Series,
               _watch: pd.DataFrame, names: str, n_hold: int, exit_rank: int,
               n_resid: int, asof: pd.Timestamp):
    """`watchlist_rows` behind a cache, with the residual book's rank added.

    `names` is in the signature only to be hashed -- the frames are passed
    with a leading underscore, so without it the cache key would not change
    when the watchlist does and editing the box would show the old ranking.

    `resid_rank` is the same name placed in the residual momentum book's
    ranking (`_market` is its factor, QQQ, as in `build_selections`). A
    separate ranking, not a re-sort of this one: the two books disagree on
    exactly the names whose strength is mostly market beta.
    """
    rows = watchlist_rows(_uni, _safe, _watch,
                          MomentumParams(n_hold=n_hold, exit_rank=exit_rank),
                          asof=asof)
    resid = watchlist_residual_ranks(
        _uni, _safe, _market, _watch,
        ResidualMomentumParams(n_hold=n_resid,
                               exit_rank=n_resid + RESID_BAND),
        asof=asof)
    for r in rows:
        r["resid_rank"] = resid.get(r["symbol"], float("nan"))
    return rows


@st.cache_data(show_spinner="Computing breadth…")
def build_breadth(_uni: pd.DataFrame, _qqq: pd.Series, note: str,
                  _volumes: Optional[pd.DataFrame] = None,
                  bar_epoch: str = "", _qqq_ohlc: Optional[pd.DataFrame] = None,
                  _spy: Optional[pd.Series] = None,
                  _spy_ohlc: Optional[pd.DataFrame] = None):
    return daily_breadth(_uni, qqq=_qqq, volumes=_volumes, universe_note=note,
                         qqq_ohlc=_qqq_ohlc, spy=_spy, spy_ohlc=_spy_ohlc)


@st.cache_data(show_spinner=False)
def spy_close_for(download_start: str, online: bool, bar_epoch: str):
    """SPY closes, for when its OHLC could not be had. `(series, error)`.

    Through the per-ticker close cache (`data/SPY.csv`): read when present,
    updated online, downloaded the first time. Offline with no cache there
    is nothing to read, and the error says so.
    """
    try:
        return load_prices(["SPY"], start=download_start, offline=not online,
                           incremental=online)["SPY"], None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner="Checking the checklist…")
def build_checklist(_uni: pd.DataFrame, _qqq: pd.Series, note: str,
                    _volumes: Optional[pd.DataFrame] = None,
                    bar_epoch: str = ""):
    """`bear_checklist` behind a cache; `note` and `bar_epoch` key it."""
    return bear_checklist(_uni, _qqq, volumes=_volumes)


# The checklist, in its source's words. `auto` rows are answered from the
# data by `qbs.breadth.bear_checklist`; the rest are the reader's to tick.
# `weight` 2 is the source's "(Weighted double)".
CHECKLIST = [
    dict(key="oversold", auto=True, weight=1,
         q="Has the percentage of NYSE stocks above their 40-day moving "
           "average stayed below 20% for over 10 trading days after dropping "
           "below 20%?",
         yes="Oversold condition has turned into a trend; historically bearish."),
    dict(key="final_high", auto=True, weight=1,
         q="On the day the index reaches a new high, is the percentage of "
           "momentum stocks above their 50-day moving average still below 30%?",
         yes="\"Final New High\" pattern."),
    dict(key="divergence", auto=True, weight=1,
         q="Is the index rising while component new lows still outnumber new "
           "highs?",
         yes="Divergence has not been resolved."),
    dict(key="mli", auto=True, weight=1,
         q="Is the number of momentum stocks (MLiN) continuing to decline "
           "without stabilizing?",
         yes="Market leaders are still receding / pulling back."),
    dict(key="fedwatch", auto=False, weight=1,
         q="Are FedWatch rate hike expectations for October and December "
           "continuing to rise?",
         yes="\"Fuel\" is not yet exhausted; the market bottom has not arrived."),
    dict(key="stops", auto=False, weight=2,
         q="Have a large number of current positions simultaneously hit "
           "stop-loss levels? (Weighted double)",
         yes="Your own portfolio has confirmed the trend/risk."),
]


SENTIMENT_TINT = {"bullish": UP_STRONG, "leaning bullish": UP,
                  "mixed": "", "unclear": "",
                  "leaning bearish": DN, "bearish": DN_STRONG}


@st.cache_data(ttl=1800, show_spinner="Fetching the last 12 hours of news…")
def load_news(as_of: str, hours: int, model: str, refresh_token: int,
              _summarise: bool):
    """`(feed, summary, from_cache)` for the news tab.

    The feed is always fetched; the summary only when `_summarise` is set.
    That split is the whole design -- the headlines are worth showing with no
    model, no key and no money.

    Cached three ways, because each stops a different kind of waste. The
    30-minute TTL stops a rerun re-searching. `sentiment.read_news` caches the
    SUMMARY to disk, so a restart does not re-bill. And `_summarise` False
    cannot reach the model at all -- the sidebar's off position has to be
    incapable of spending, not merely disinclined to.
    """
    from qbs.agent import sentiment as snt

    return snt.read_news(hours=hours, summarise_it=_summarise,
                         refresh=refresh_token > 0, model=model or None,
                         as_of=as_of)


@st.cache_data(show_spinner="Fetching the US universe from Finviz…")
def load_us_market(download_start: str, online: bool, force: bool,
                   bar_epoch: str, _token: int):
    """The broad US universe for the breadth tab: closes, volumes, sectors.

    Returns `(closes, volumes, sectors, note, error, fetched)`. `error` is not
    fatal -- the caller falls back to the cached index universe and says so on
    screen, because breadth computed over a truncated sample is wrong in a way
    that looks entirely plausible. `fetched` says whether this call went to the
    network, so the UI can tell "fresh" from "cached" rather than implying one.

    Automatic refresh is gated by `due_for_fetch` to one attempt a day, at the
    wall-clock slot `QBS_FETCH_AT` names, and the gate counts ATTEMPTS.
    Streamlit re-runs this script on every widget interaction, so a data-based
    test would start a 2,400-name download on every rerun and never stop.
    **Refresh now** ignores the gate.
    """
    filters = UniverseFilters()
    if not online:
        uni, uni_err, src = fetch_universe(filters, offline=True, verbose=False)
        auto, why = False, "offline"
    else:
        auto, why = due_for_fetch()
        auto = auto or force
        # Stamp BEFORE the work, not after. A download that dies halfway must
        # still count as today's attempt, or a broken feed retries on every
        # rerun -- which is the loop this gate exists to prevent.
        if auto:
            record_fetch_attempt()
        uni, uni_err, src = fetch_universe(filters, refresh=force, offline=False,
                                           verbose=False)
    if uni is None or uni.empty:
        return None, None, {}, f"{filters.label} · via {src}", (
            uni_err or "unknown failure"), False, None

    tickers = uni["Ticker"].tolist()

    # Created on the first batch, not up front: a cache hit downloads nothing
    # and should not flash an empty bar. Made INSIDE this cached function on
    # purpose -- an element on a container created outside it cannot be
    # replayed on a cache hit, and Streamlit raises there rather than skip it.
    bar = None

    def _progress(done: int, total: int, what: str) -> None:
        nonlocal bar
        text = (f"Loading US universe {what} — {done:,} of {total:,} stocks "
                f"({done / total:.0%})")
        if bar is None:
            bar = st.progress(0.0, text=text)
        bar.progress(min(1.0, done / total), text=text)

    try:
        closes, volumes, bars_err = load_universe_bars(
            tickers, start=download_start, refresh=force, offline=not online,
            verbose=False, stale_after=1 if auto else None, incremental=True,
            progress=_progress)
    finally:
        if bar is not None:
            bar.empty()
    if closes is None or closes.empty:
        return None, None, {}, filters.label, (
            f"Finviz listed {len(tickers)} tickers but no prices loaded — "
            f"{bars_err}"), False, None

    # A bar the provider had not finished publishing is dropped inside
    # `load_universe_bars` now, not here -- it has to happen before that
    # function's own staleness check, or a torn bar makes the cache look
    # current and the download that would replace it never runs. Its reason
    # arrives in `bars_err`.
    closes, volumes, fill = _top_up_front_bar(closes, volumes, online,
                                              persist=MARKET_CACHE)
    # The top-up is NOT a warning: it is the fill working as designed, and
    # lumping it in here put a "Partial US universe" banner over a universe
    # that was complete. It goes back separately so the tab can say it the
    # way the picks tab does.
    warn = "; ".join(x for x in (uni_err, bars_err) if x) or None
    # The source is named in the note because two providers apply the same
    # rules to different listings databases and will not agree on the last
    # hundred names. A breadth count that steps when the source changed, on a
    # screen that does not say the source changed, reads as a market event.
    note = f"{filters.label} · via {src} · {why}"
    return closes, volumes, sector_map(uni), note, warn, auto, fill


EMA_SPANS = (10, 20, 50, 200)
EMA_COLOURS = {"EMA 10": "#eb6834", "EMA 20": "#eda100",
               "EMA 50": "#2a78d6", "EMA 200": "#8a63d2"}


CANDLE_UP, CANDLE_DN = "#1b7a4b", "#b02525"


@st.cache_data(show_spinner=False)
def ohlc_for(ticker: str, download_start: str, online: bool, bar_epoch: str):
    """Real daily OHLC for one name, or None if it cannot be had.

    None is a first-class answer: the chart draws a close line instead and
    says why. Faking the missing columns from closes would put a body and no
    wick on every bar, which asserts a high and a low that never happened.
    """
    return load_daily_ohlc(ticker, start=download_start, offline=not online)


@st.cache_data(show_spinner=False)
def chart_frames(_uni: pd.DataFrame, ticker: str, asof: pd.Timestamp,
                 lookback: int, max_levels: int = 8):
    """Price, EMAs and support/resistance for one name, as of one date.

    Levels are derived from history up to `asof` ONLY — the same causal rule
    `qbs.breakout.candidate_trades` uses. Drawing levels from the full series
    would show the chart lines that the strategy could not have seen on the
    date being inspected, which is the look-ahead the breakout port exists to
    remove; a dashboard that quietly reintroduces it is worse than none.
    """
    close = _uni[ticker].dropna().loc[:asof]
    if close.empty:
        return None

    emas = pd.DataFrame({f"EMA {n}": close.ewm(span=n, adjust=False,
                                               min_periods=n).mean()
                         for n in EMA_SPANS})

    levels = sr_levels(closes_to_bars(close.to_frame(ticker))[ticker],
                       BreakoutParams())

    window = close.iloc[-lookback:]
    price = window.rename("close").reset_index()
    price.columns = ["date", "close"]

    ema_long = (emas.loc[window.index].reset_index()
                .melt(id_vars=emas.index.name or "Date",
                      var_name="ema", value_name="value"))
    ema_long.columns = ["date", "ema", "value"]
    ema_long = ema_long.dropna(subset=["value"])

    last = float(window.iloc[-1])
    shown, n_in_view, has_overhead = levels_in_view(
        levels, last, float(window.min()), float(window.max()), max_levels)

    lvl = pd.DataFrame({"level": shown})
    if not lvl.empty:
        lvl["kind"] = np.where(lvl["level"] >= last, "Resistance", "Support")
    else:
        lvl["kind"] = pd.Series(dtype=object)
    return price, ema_long, lvl, last, n_in_view, has_overhead


def session_axis(*frames, date_col: str = "date", n_ticks: int = 6):
    """Put charts on a SESSION index instead of a calendar one.

    Returns `(frames, axis)` with an `n` column added to each frame and an
    Altair axis that still prints dates, at a handful of ticks.

    A temporal axis draws real time, which means it draws the weekend: every
    Saturday and Sunday is a gap the market did not trade through, and on a
    twelve-month chart roughly two days in seven of the width carry no data.
    Holidays add more. Candles end up separated by whitespace that looks like
    a pause in trading and is not one, and a 20-session pulse chart wears four
    gaps that mean nothing.

    Numbering the sessions removes all of it: bar `n` sits next to bar `n+1`
    whatever the calendar did in between. The cost is that the axis no longer
    reads as a ruler of time, which is why the tick LABELS are still dates --
    the spacing is sessions, the labels say when.

    All frames share one index, built from the union of their dates, because
    the price panel layers candles, EMAs and levels on one chart and a layer
    numbered on its own dates would sit a bar or two off the others.
    """
    dates = pd.DatetimeIndex(sorted(set().union(
        *[pd.DatetimeIndex(f[date_col]) for f in frames if f is not None
          and not f.empty])))
    if len(dates) == 0:
        return frames, alt.Axis(title=None)

    index = {d: i for i, d in enumerate(dates)}
    out = []
    for f in frames:
        if f is None or f.empty:
            out.append(f)
            continue
        f = f.copy()
        f["n"] = pd.DatetimeIndex(f[date_col]).map(index)
        out.append(f)

    # A handful of ticks, always including the last session -- the right-hand
    # edge is the one anybody actually looks for.
    step = max(1, (len(dates) - 1) // max(1, n_ticks - 1))
    picks = list(range(0, len(dates), step))
    last = len(dates) - 1
    if picks[-1] != last:
        # Replace rather than append when the final tick would land on top of
        # the previous one: two labels a couple of sessions apart overlap and
        # read as a mistake.
        if last - picks[-1] < step / 2:
            picks[-1] = last
        else:
            picks.append(last)
    spans_years = dates[-1].year != dates[0].year
    fmt_ = "%b %d %y" if spans_years else "%b %d"
    expr = " : ".join(f"datum.value === {i} ? '{dates[i].strftime(fmt_)}'"
                      for i in picks) + " : ''"
    axis = alt.Axis(values=picks, labelExpr=expr, title=None, labelAngle=0,
                    grid=False)
    return out, axis


def fmt(v, spec="{:.1f}", dash="—"):
    return dash if v is None or (isinstance(v, float) and pd.isna(v)) else spec.format(v)


def session_text(session) -> str:
    """A bar date, and when that US session closed in the reader's zone.

    The zone is the fetch schedule's (`QBS_FETCH_TZ`), the one the reader
    already set to their own. Bar dates are New York dates: read in Asia the
    morning after, the newest bar carries YESTERDAY's date and is current.
    """
    tz = fetch_schedule()[2]
    close = session_close(session, tz)
    here = (f"{close:%m-%d %H:%M} {tz}" if tz != MARKET_TZ
            else f"{close:%H:%M} ET")
    return f"{pd.Timestamp(session):%Y-%m-%d} (US session, closed {here})"


def md(text: str) -> str:
    """Escape `$` for Streamlit's markdown, which reads `$...$` as LaTeX.

    Two dollar signs in one caption -- "close > $5 ... turnover > $5M/day" --
    make everything between them a maths span: the dollars vanish and the text
    renders in a serif italic. It looks like a styling quirk rather than a bug,
    which is why it survived a review here, so every string that can carry a
    price goes through this.

    Only needed for markdown-rendered text (`caption`, `markdown`, `warning`).
    `st.dataframe` shows cell values literally and must NOT be escaped, or the
    backslashes appear in the table.
    """
    return text.replace("$", r"\$")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

cfg = Config()
st.sidebar.header("Settings")

st.sidebar.subheader("Data")
online = st.sidebar.radio(
    "Source", [True, False], index=0,
    format_func=lambda v: "Online (download + cache)" if v else "Offline (cache only)",
    help="Online downloads only when the cache is behind, so a fresh cache "
         "costs no network. Offline never reaches the internet.",
) 
st.session_state.setdefault("refresh_token", 0)
if st.sidebar.button("Refresh now", width="stretch",
                     help="Force a re-download even if the cache looks current."):
    st.session_state["refresh_token"] += 1
    st.cache_data.clear()

download_start = st.sidebar.text_input("Data start", cfg.download_start)
n_hold = st.sidebar.number_input("Names held (n_hold)", 1, 20, cfg.momentum.n_hold)
exit_rank = st.sidebar.number_input("Exit rank (band)", int(n_hold), 50,
                                    max(cfg.momentum.exit_rank, int(n_hold)))
n_screen = st.sidebar.number_input(
    "High-momentum screen size", 5, 100, FinvizScreenParams().n_hold,
    help="How many names the high-momentum screen ranks down to. On the "
         "Nasdaq-100 universe fewer than this usually qualify, so the column "
         "shows everyone who passed.")
n_resid = st.sidebar.number_input(
    "Residual book size", 1, 30, RESID_N_HOLD,
    help="Slots in the residual-momentum book. It gets its own control "
         "rather than sharing n_hold because it is run deeper here than its "
         f"total-return sibling: its band follows at +{RESID_BAND}.")

# Now that every slot count is known, name each book after the one it is
# actually running rather than after the config default.
STRATEGY_LABELS = strategy_labels(int(n_hold), int(n_screen), int(n_resid))

# Loaded here, not with the Analyst settings further down, because the
# default below reads the environment and this box is rendered first. The
# load is cached per file, so the later call is free.
from qbs.agent.env import load_env as _load_env
_load_env()
# Deliberately NOT `QBS_WATCHLIST`. That one belongs to the live runner and
# lives in deploy/docker/.env on the trading host; this one belongs to the
# dashboard and lives in the repo root's .env. Different files already, but
# the same NAME would collide the moment both run on one host with the
# runner's env exported into the shell -- and then changing what the
# dashboard charts would quietly change what the runner logs. Two names, no
# fallback between them: a fallback is the coupling this is removing.
WATCHLIST_VAR = "QBS_DASH_WATCHLIST"
watch_raw = st.sidebar.text_input(
    "Watchlist", os.environ.get(WATCHLIST_VAR, ""),
    help="Names to rank beside the book without letting the book buy them — "
         "comma or space separated. Anything outside the index is "
         f"interpolated into the constituents' ranking. Set {WATCHLIST_VAR} "
         "in the repo root's .env to seed this box. The live runner's own "
         "watchlist (QBS_WATCHLIST, in deploy/docker/.env) is separate and "
         "nothing here touches it.")
# The runner's parser, not a second one that agrees with it today. The two
# watchlists are different lists; they should still mean the same thing by
# "TSM, googl".
WATCHLIST = parse_watchlist(watch_raw)

# Every cache below that holds price data takes this as a key. `st.cache_data`
# memoises on arguments alone, and the frames are passed with a leading
# underscore so they are not hashed -- which means a process left running
# across a close serves the numbers it read on start-up for ever, and the
# fetch gate inside the loader is never even consulted because the loader
# does not run. `fetch_epoch` changes exactly when a new bar becomes
# collectable, so the memo expires on the close and on nothing else. A
# time-based TTL would also work and would re-read on a schedule that has
# nothing to do with when the data changes.
#
# It has to be threaded through the DERIVED caches too -- selections, breadth,
# the OHLC bars. Fixing only the loaders would leave them returning yesterday's
# answer over today's frame, which is worse than being uniformly stale: the
# banner would say current and the table would not be.
BAR_EPOCH = fetch_epoch()

# Default to 0, not -1. With -1 the very first page load has 0 > -1, so the
# app force-refreshed on EVERY start -- re-downloading the whole universe
# before it had shown anything. "Refresh now" is the only thing that should
# set this.
force = st.session_state["refresh_token"] > st.session_state.get("applied_token", 0)
try:
    uni, px, data_status = load_data(download_start, bool(online), bool(force),
                                     BAR_EPOCH,
                                     st.session_state["refresh_token"])
    st.session_state["applied_token"] = st.session_state["refresh_token"]
except Exception as exc:  # noqa: BLE001
    st.error(
        f"Could not load data: {exc}\n\n"
        "Offline mode reads only the CSV cache in `data/`. Build it once with "
        "`python run_backtest.py`, or switch the sidebar to **Online**."
    )
    st.stop()

LAST_BAR = uni.index.max()
N_BEHIND, FRESH_LEVEL, FRESH_MSG = freshness_note(LAST_BAR)


def freshness_banner():
    """Say how old the data is, on every tab. A dashboard that is quietly a
    week behind is worse than one that admits it."""
    failed = bool(data_status["error"])
    if failed:
        st.error(f"**Download failed — showing the cached data instead.** "
                 f"{data_status['error']}", icon="🚫")
    if data_status.get("filled"):
        # Two providers in one series must never be invisible: a reader
        # comparing today's numbers with last week's needs to know the newest
        # bar did not come from the same place as the rest.
        st.info(md("🔗 " + data_status["filled"]
                   + (f". QQQ, VEU and BOXX have no bar for "
                      f"{data_status['carried']} either and are **carried "
                      "forward** — the index ATR reads as an unchanged day "
                      "on it." if data_status.get("carried") else "")),
                icon="🧩")
    if data_status.get("partial"):
        # Said out loud rather than quietly dropped. Someone looking for
        # yesterday's session needs to know it was there and was thrown
        # away, or they will read the gap as a missing download.
        days = ", ".join(data_status["partial"])
        st.warning(md(
            f"**{days} dropped — the provider had not finished publishing "
            "it.** A session counted over a fraction of the universe is not a "
            "session: every 4%-mover count and every rank would have come "
            "from those few names. Press **Refresh now** once the data has "
            "settled — the count in brackets is how many carried that bar "
            "against how many a normal one has."), icon="🧩")
    if FRESH_LEVEL == "warn":
        # The remedy depends on why it is stale. Do not tell someone to go
        # online when they already are and the download is what broke.
        if failed:
            remedy = ("Check your connection, then press **Refresh now**. "
                      "Everything below is from the cache.")
        elif online:
            remedy = "Press **Refresh now** in the sidebar to force a re-download."
        else:
            remedy = ("Switch **Source** to Online, or run "
                      "`python run_backtest.py --refresh`.")
        st.warning(f"{FRESH_MSG} {remedy}", icon="🕒")
    elif FRESH_LEVEL == "info":
        st.caption(f"🕒 {FRESH_MSG}")

SRC = "downloaded" if data_status["downloaded"] else "cache"
UNIVERSE_NOTE = f"{uni.shape[1]} Nasdaq-100 constituents (closes only)"
# Which provider the market tab will ask, named before anything is fetched.
# Two sources are two universes; a reader comparing today's breadth with
# last week's needs to know whether the measurement changed underneath it.
_SRC = resolve_source()
_SRC_STATE = available_sources().get(_SRC, "unknown")
st.sidebar.caption(md(f"US universe source: **{_SRC}** (`{SOURCE_VAR}`)"))
if not _SRC_STATE.startswith("installed"):
    st.sidebar.warning(md(
        f"`{_SRC}` is selected but {_SRC_STATE}. The market tab will fall "
        "back to the Nasdaq-100 until it is installed **for this "
        "interpreter**."), icon="🔌")

st.sidebar.caption(f"Picks universe: {UNIVERSE_NOTE}")
st.sidebar.caption("The market tab has its own universe selector.")
st.sidebar.caption(f"Data through {LAST_BAR:%Y-%m-%d} ({SRC})")
st.sidebar.caption({"ok": "✅ current", "info": "🕒 1 session behind",
                    "warn": f"⚠️ {N_BEHIND} sessions behind"}[FRESH_LEVEL])

selections, SCREEN_VOLUME_APPLIED = build_selections(
    uni, px["BOXX"], px["QQQ"], int(n_hold), int(exit_rank), int(n_screen),
    int(n_resid), BAR_EPOCH)
# QQQ's real High/Low for the ATR column. Closes alone make the true range
# close-to-close, which understates it and roughly doubles the reading next to
# a source that uses real bars. None falls back to closes.
QQQ_OHLC = ohlc_for("QQQ", download_start, bool(online), BAR_EPOCH)
# SPY is not one of the core ETFs the books trade, so nothing else loads it.
# Its bars are fetched here, the same way as QQQ's -- real High/Low for the
# ATR -- and cached under data/ohlc/. Closes are the fallback.
SPY_OHLC = ohlc_for("SPY", download_start, bool(online), BAR_EPOCH)
SPY_CLOSE, SPY_ERR = ((SPY_OHLC["Close"], None) if SPY_OHLC is not None
                      else spy_close_for(download_start, bool(online), BAR_EPOCH))
breadth = build_breadth(uni, px["QQQ"], UNIVERSE_NOTE, bar_epoch=BAR_EPOCH,
                        _qqq_ohlc=QQQ_OHLC, _spy=SPY_CLOSE, _spy_ohlc=SPY_OHLC)

def names_on(key: str, when) -> list:
    """The tickers a strategy held on a date, from the prebuilt selections.

    Read from `selections` rather than from whatever the picks tab happened to
    leave in a local: Streamlit runs the script top to bottom, so that would
    work today and break the moment the tabs are reordered.
    """
    frame = selections.get(key)
    if frame is None or when not in frame.index:
        return []
    raw = frame.loc[when, "holdings"]
    return [t for t in str(raw).split(", ") if t]


def rank_table(rows, n_hold: int, exit_rank: int,
               lead: Optional[Dict[str, Dict[str, str]]] = None,
               sort: bool = True):
    """The watchlist / sector-leaders table, as one Styler.

    Two panels show these same seven columns about different sets of names --
    the picks tab's watchlist and the market tab's sector leaders -- and a
    second copy of the formatting would drift from the first the moment
    either is touched. `rows` is `qbs.shadow.watchlist_rows` output.

    `lead` adds columns in FRONT of Ticker, each a `{symbol: value}` map
    rather than a list, so a caller cannot get its extra column out of step
    with the rows it is labelling.

    `sort=False` keeps the caller's order, which the sector panel needs: it
    groups by sector first and ranks inside it, so a global sort by rank
    would shuffle the groups apart.
    """
    wf = pd.DataFrame(rows)
    if sort:
        # NaN last: a name with too little history is unranked, not top.
        wf = wf.sort_values("rank", na_position="last")
    wf = wf.reset_index(drop=True)

    cols: Dict[str, list] = {}
    for label, by_symbol in (lead or {}).items():
        cols[label] = [by_symbol.get(t, "—") for t in wf["symbol"]]
    cols["Ticker"] = list(wf["symbol"])
    cols["Rank"] = [fmt(r, "{:.0f}") for r in wf["rank"]]
    if "resid_rank" in wf:
        # The residual book's ranking, placed the same way: a constituent's
        # standing rank there, an outsider interpolated into it.
        cols["Resid. rank"] = [fmt(r, "{:.0f}") for r in wf["resid_rank"]]
    cols["Placed"] = ["in index" if c else "interpolated"
                      for c in wf["constituent"]]
    cols["Score"] = [fmt(v, "{:+.3f}") for v in wf["score"]]
    cols[f"Book cutoff (#{int(n_hold)})"] = [fmt(v, "{:+.3f}")
                                             for v in wf["book_cutoff"]]
    cols[f"Band cutoff (#{int(exit_rank)})"] = [fmt(v, "{:+.3f}")
                                                for v in wf["band_cutoff"]]
    cols["Beats the book"] = ["✅" if b else "—" for b in wf["beats_book"]]

    show = pd.DataFrame(cols)
    beats = list(wf["beats_book"])
    # The tint is on the rank cell only: it is the one number the row is
    # about, and a whole green row reads as an endorsement of the name.
    styler = show.style.apply(
        lambda _col: [f"background-color: {UP}" if b else "" for b in beats],
        subset=["Rank"])
    return styler, show, wf


def candle_table(names, held_by: Dict[str, str], asof: pd.Timestamp,
                 window: int, rules: HammerRules):
    """One row per name: volume against its average, and the last 3 candles.

    Returns `(frame, hammer_cells, missing)`. `hammer_cells` is a
    `{column: [verdict per row]}` map the caller tints from, so the colour
    follows the rule rather than a re-parse of the cell text; `missing` lists
    the names with no OHLC at all.

    Read from the per-name OHLC cache (`ohlc_for`), cut at `asof` so moving
    the date slider shows what was knowable on that date. Volume falls back to
    the universe volume cache for a name with no OHLC -- the hammer columns
    cannot, since a candle cannot be drawn from closes.
    """
    from qbs.universe import load_universe_volumes

    uvol = load_universe_volumes(list(names))
    rows, missing = [], []
    day_cols = ["Hammer D0", "Hammer D-1", "Hammer D-2"]
    verdicts: Dict[str, list] = {c: [] for c in day_cols}
    for t in names:
        ohlc = ohlc_for(t, download_start, bool(online), BAR_EPOCH)
        bars = None
        if ohlc is not None and not ohlc.empty:
            bars = ohlc.loc[:asof]
            if not {"Open", "High", "Low", "Close"} <= set(bars.columns) or bars.empty:
                bars = None
        if bars is not None and "Volume" in bars.columns:
            vs = volume_stats(bars["Volume"], bars["Close"], window)
            bar_date = bars.index[-1]
        elif uvol is not None and t in uvol.columns:
            v = uvol[t].loc[:asof].dropna()
            close = uni[t] if t in uni.columns else None
            vs = volume_stats(v, close, window)
            bar_date = v.index[-1] if len(v) else None
        else:
            vs, bar_date = volume_stats(pd.Series(dtype=float)), None

        row = {"Ticker": t, "Held by": held_by.get(t, "—"),
               "Bar": f"{bar_date:%m-%d}" if bar_date is not None else "—",
               "Last vol": vs["last"], f"Avg vol ({window}d)": vs["avg"],
               "Vol ratio": vs["ratio"],
               f"Avg $ vol ({window}d)": vs["avg_value"]}
        if bars is None:
            missing.append(t)
            for c in day_cols:
                row[c] = "—"
                verdicts[c].append(None)
        else:
            hf = hammer_frame(bars, rules).tail(3).iloc[::-1]
            for i, c in enumerate(day_cols):
                if i >= len(hf):
                    row[c] = "—"
                    verdicts[c].append(None)
                    continue
                b = hf.iloc[i]
                share = "—" if pd.isna(b["lower"]) else f"{b['lower']:.0%}"
                if b["hammer"]:
                    row[c], v = f"🔨 {share}", "hammer"
                elif b["hanging_man"]:
                    row[c], v = f"⚠️ {share}", "hanging"
                elif b["shape"]:
                    row[c], v = f"◐ {share}", "shape"
                else:
                    row[c], v = share, None
                verdicts[c].append(v)
        rows.append(row)
    return pd.DataFrame(rows), verdicts, missing


# Loaded once, above the tabs, because two of them need it: the picks tab
# ranks the watchlist and the analyst tab charts it. Inside one tab it would
# be a second download the moment the other wanted the same prices.
WATCH_OUTSIDERS = tuple(t for t in WATCHLIST if t not in uni.columns)
WATCH_PX, WATCH_ERR = load_watch_prices(
    WATCH_OUTSIDERS, download_start, bool(online), f"{LAST_BAR:%Y-%m-%d}",
    st.session_state["refresh_token"])

# Every watched name that has prices, on the universe's calendar. A
# constituent is read from the frame that was RANKED rather than
# re-downloaded: the watchlist row has to report the rank the book acted on,
# and a second copy of the same prices is how the two drift apart.
WATCH_FRAME = pd.DataFrame({
    t: (uni[t] if t in uni.columns else WATCH_PX[t].reindex(uni.index).ffill())
    for t in WATCHLIST
    if t in uni.columns or t in WATCH_PX.columns})

# The watched names the universe frame does NOT hold. This is the list that
# needs `extra` prices to be chartable at all, and the list a caption has to
# mark as interpolated rather than ranked.
WATCH_EXTRA = [t for t in WATCH_FRAME.columns if t not in uni.columns]


def price_panel(uni, px, asof, options, n_hold: int, key_prefix: str,
                default_ticker: Optional[str] = None,
                extra: Optional[pd.DataFrame] = None,
                ticker_help: str = "Today's picks come first, then the rest "
                                   "of the universe."):
    """The price / levels / momentum panel, so two tabs can show one panel.

    Extracted rather than copied: it is ~180 lines of chart, level and gate
    logic, and a second copy would drift from the first the moment either is
    touched. `key_prefix` namespaces the Streamlit widget keys, which must be
    unique across the whole app even when the widgets are identical.

    `options` is the ticker list for the combo box, already in the order the
    caller wants it -- this function does not decide what is interesting.

    `extra` carries prices for names that are NOT in `uni` -- the watchlist's
    non-constituents. One such name is joined into the universe frame for the
    duration of its own panel, and only then:

    * a CONSTITUENT is profiled against the untouched universe, exactly as
      before. The join is skipped entirely, so putting a name in the
      watchlist box cannot move a number the picks tab reports.
    * an OUTSIDER is interpolated into that universe: it joins, and its own
      percentile is what the panel reports. The rank means "where it would
      place among the constituents", not "a book holds it" -- the same claim
      `qbs.shadow.watchlist_rows` makes, on a different scale (that one is an
      ordinal position, this one a 1-99 percentile).

    Joined one name at a time, not a whole watchlist at once. Joining them
    together would rank every watched name against the others, so two
    outsiders would shift each other and the number would depend on what
    else happened to be in the box. Adding a 98th name to 97 does still move
    the other constituents' percentiles by a point, but those are not shown
    on the outsider's own panel, and they are back to normal on everyone
    else's.
    """
    st.markdown("**Price & levels**")
    if not options:
        st.info("No name to chart.")
        return None
    index = options.index(default_ticker) if default_ticker in options else 0
    c1, c2, c3 = st.columns([2, 1, 1])
    ticker = c1.selectbox(
        "Ticker", options, index=index, key=f"{key_prefix}_ticker",
        help=ticker_help)

    outsider = ticker not in uni.columns
    if outsider:
        if extra is None or ticker not in extra.columns:
            st.info(f"No prices for {ticker} — it is not in the ranking "
                    "universe and nothing was loaded for it.")
            return ticker
        uni = uni.join(extra[[ticker]], how="left")
    months = c2.selectbox("Window", [3, 6, 12, 24], index=2,
                          format_func=lambda m: f"{m}m", key=f"{key_prefix}_win")
    n_lvl = c3.number_input("Levels", 0, 30, 8, key=f"{key_prefix}_levels",
                            help="Nearest N support/resistance levels "
                                 "to the last price. 0 hides them.")
    frames = chart_frames(uni, ticker, asof, int(months * 21), int(n_lvl))
    if frames is None:
        st.info(f"No price history for {ticker} up to this date.")
    else:
        price, ema_long, lvl, last, n_levels, has_overhead = frames

        # Candles need real Open/High/Low. When they cannot be had the
        # chart falls back to a close line and says so, rather than
        # drawing a wickless body per bar off the close series -- that
        # would assert a session high and low that never happened.
        ohlc = ohlc_for(ticker, download_start, bool(online), BAR_EPOCH)
        bars = None
        if ohlc is not None and not ohlc.empty:
            win = ohlc.loc[:asof].tail(int(months * 21))
            if len(win) > 2 and {"Open", "High", "Low"} <= set(win.columns):
                bars = win.reset_index()
                bars.columns = [str(c).lower() for c in bars.columns]

        # One index across every layer here. Numbered per layer, a candle at
        # session 40 and an EMA point at session 40 would be different days
        # whenever the two frames start on different dates.
        (price, ema_long, bars), xaxis = session_axis(price, ema_long, bars)
        xenc = alt.X("n:Q", axis=xaxis, title=None,
                     scale=alt.Scale(nice=False, zero=False))

        yscale = alt.Scale(zero=False, nice=True)
        if bars is not None:
            body_colour = alt.condition(
                "datum.open <= datum.close",
                alt.value(CANDLE_UP), alt.value(CANDLE_DN))
            cbase = alt.Chart(bars).encode(
                x=xenc, color=body_colour,
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("open:Q", format=".2f"),
                         alt.Tooltip("high:Q", format=".2f"),
                         alt.Tooltip("low:Q", format=".2f"),
                         alt.Tooltip("close:Q", format=".2f")])
            wick = cbase.mark_rule(size=1).encode(
                y=alt.Y("low:Q", title=None, scale=yscale),
                y2=alt.Y2("high:Q"))
            body = cbase.mark_bar(size=max(1.5, 380 / len(bars))).encode(
                y=alt.Y("open:Q", scale=yscale), y2=alt.Y2("close:Q"))
            line = wick + body
        else:
            line = alt.Chart(price).mark_line(
                color="#0b0b0b", size=1.7).encode(
                x=xenc,
                y=alt.Y("close:Q", title=None, scale=yscale),
                tooltip=[alt.Tooltip("date:T", title="Date"),
                         alt.Tooltip("close:Q", title="Close", format=".2f")])
        emas = alt.Chart(ema_long).mark_line(size=1.1, opacity=0.9).encode(
            x=xenc,
            y=alt.Y("value:Q", scale=alt.Scale(zero=False, nice=True)),
            color=alt.Color("ema:N", title=None, scale=alt.Scale(
                domain=list(EMA_COLOURS), range=list(EMA_COLOURS.values())),
                legend=alt.Legend(orient="top", direction="horizontal")),
            tooltip=[alt.Tooltip("date:T", title="Date"),
                     alt.Tooltip("ema:N", title="Line"),
                     alt.Tooltip("value:Q", title="Value", format=".2f")])
        layers = [line, emas]
        if not lvl.empty:
            rules = alt.Chart(lvl).mark_rule(
                strokeDash=[5, 4], size=1.1, opacity=0.85).encode(
                y=alt.Y("level:Q", scale=alt.Scale(zero=False, nice=True)),
                color=alt.Color("kind:N", title=None, scale=alt.Scale(
                    domain=["Resistance", "Support"],
                    range=["#d03b3b", "#0ca30c"]),
                    legend=alt.Legend(orient="top", direction="horizontal")),
                tooltip=[alt.Tooltip("kind:N", title="Level"),
                         alt.Tooltip("level:Q", title="Price", format=".2f")])
            layers.append(rules)
        st.altair_chart(
            alt.layer(*layers).resolve_scale(color="independent")
            .properties(height=430), use_container_width=True)
        if bars is None:
            st.caption(
                "📉 Close line, not candles — no Open/High/Low for "
                f"**{ticker}**. The universe cache holds closes only; "
                "switch **Source** to Online so the panel can fetch "
                "real OHLC for the selected name. Candles are never "
                "drawn from closes, because a wickless body would "
                "assert a high and low that never happened."
            )

        above = [f"EMA {n}" for n in EMA_SPANS
                 if not ema_long[ema_long["ema"] == f"EMA {n}"].empty
                 and last > ema_long[ema_long["ema"] == f"EMA {n}"]["value"].iloc[-1]]
        st.caption(
            f"**{ticker}** {last:,.2f} · above "
            f"{', '.join(above) if above else 'none'} · "
            f"showing {len(lvl)} of {n_levels} levels in view"
        )
        if not has_overhead:
            st.warning(
                "**No resistance overhead in this window.** The name has "
                "already cleared every level its chart shows, so there "
                "is nothing above it to break through — it is in price "
                "discovery.",
                icon="⚠️")
        st.caption(
            "Levels are re-derived from history **up to the selected date "
            "only** — the same causal rule the breakout strategy uses, so "
            "the chart never shows a level the strategy could not have seen."
        )

        # ---- the numbers behind the picture -----------------------
        # Same params the picks table above was screened with, so the
        # gate rows report the screen actually running, not a default.
        prof = momentum_profile(uni, ticker, asof=asof, safe=px["BOXX"],
                                screen=FinvizScreenParams(n_hold=int(n_hold)))
        if not prof["returns"].empty:
            st.markdown("###### Momentum")
            ret = prof["returns"].copy()
            st.dataframe(
                ret.style.format({"Return": "{:+.1%}",
                                  "Universe median": "{:+.1%}",
                                  "Rank": "{:.0f}"}, na_rep="—")
                .background_gradient(subset=["Rank"], cmap="RdYlGn",
                                     vmin=1, vmax=99),
                hide_index=True, width="stretch",
                height=45 + 35 * len(ret))
            st.caption(
                "**Rank is a percentile within this universe on this date**, "
                "1–99. A twelve-month return means nothing on its own — the "
                "question a momentum strategy asks is relative, so the number "
                "only counts next to what every other candidate did. "
                "*Universe median* is that comparison in one column."
                + (f" **{ticker} is not in the index**, so it is interpolated "
                   "into the constituents' ranking — nothing else moved to "
                   "make room, and a strong rank means it *would* place there "
                   "if it were a constituent, not that a book holds it."
                   if outsider else "")
            )

            tr = prof["trend"].copy()
            tr["Value"] = [
                ("—" if pd.isna(v) else
                 f"{v:+.2f} ATR" if u == "ATR" else f"{v:+.1%}")
                for v, u in zip(tr["Value"], tr["Unit"])]
            st.dataframe(tr[["Measure", "Value"]], hide_index=True,
                         width="stretch", height=45 + 35 * len(tr))

            st.markdown("###### Which strategies would take it, and why")
            g = prof["gates"].copy()
            # `off` is a distance below the high, so it never carries
            # a "+" -- 17.6% there means 17.6% WORSE than the high.
            g["Reading"] = [
                ("—" if pd.isna(v) else
                 f"${v:,.2f}" if f == "price" else
                 f"{v:.1%} off high" if f == "off" else f"{v:+.1%}")
                for v, f in zip(g["Value"], g["Fmt"])]
            g["✓"] = ["—" if x is None else ("✅" if x else "❌")
                      for x in g["Pass"]]
            st.dataframe(g[["Strategy", "Rule", "Reading", "✓"]],
                         hide_index=True, width="stretch",
                         height=45 + 35 * len(g))
            failed = g[g["Pass"] == False]          # noqa: E712
            if not failed.empty:
                note = ("Blocked by: "
                        + "; ".join(f"**{r.Strategy}** — {r.Rule} "
                                    f"({r.Reading})"
                                    for r in failed.itertuples()) + ".")
                # Only claim the overlap finding when it is the
                # proximity rule doing the blocking -- that is the
                # specific disagreement it describes.
                if any("high" in r.Rule and r.Strategy == "Finviz screen"
                       for r in failed.itertuples()):
                    note += (" This is the per-name version of the overlap "
                             "finding: a name can rank at the very top on "
                             "momentum and still fail the proximity test, "
                             "which is why the two screens rarely agree.")
                st.caption(md(note))
    return st.session_state.get(f"{key_prefix}_ticker")


# --------------------------------------------------------------------------
# Analyst settings
# --------------------------------------------------------------------------
# Defined here, NOT inside the Analyst tab: the market tab renders first and
# needs `model_name` for its news summary, and a sidebar control created in a
# later tab does not exist yet when an earlier one reads it.

from qbs.agent.analyst import (DEFAULT_MODEL, DEFAULT_SUMMARY_MODEL,
                               _default_thinking_budget, analyse,
                               check_requirements)
from qbs.agent.env import (DISABLE_CHAT_VAR, DISABLE_NEWS_ANALYSIS_VAR,
                           DISABLE_NEWS_READ_VAR, RETIRED_VARS, chat_disabled,
                           load_env, news_analysis_disabled,
                           news_read_disabled, retired_vars_in_use)
from qbs.agent.evidence import Book
from qbs.agent.news import available_backends, backend_note
from qbs.agent.sentiment import parse_published as snt_parse_published

env_load = load_env()
chat_off = chat_disabled()
news_off = news_analysis_disabled()
fetch_off = news_read_disabled()
# Two blockers, because the three features answer to three switches and no
# master: the chat being off says nothing about the news read, or the other
# way round.
chat_blocker = check_requirements(role="chat")
news_blocker = check_requirements(role="summary")
blocker = chat_blocker                    # the Analyst tab's own gate
backends = available_backends()
retired = retired_vars_in_use()

with st.sidebar:
    st.markdown("---")
    st.markdown("**Analyst**")
    # The controls below stay editable on purpose -- you can line the model and
    # the toggles up while something is off -- but without this they read as an
    # analyst that is simply misbehaving.
    if chat_off:
        st.caption(f"⏸️ chat switched off by `{DISABLE_CHAT_VAR}`. These "
                   "settings are saved for when it is switched back on; the "
                   "news read answers to its own switch.")
    if news_off:
        st.caption(f"⏸️ news analysis off. `{DISABLE_NEWS_ANALYSIS_VAR}=0` "
                   "switches the Gemini read on; headlines show either way.")
    # A retired switch someone is still setting is the one thing here worth
    # a warning rather than a caption: it LOOKS like it is holding the bill
    # down, and it is not holding anything.
    for _var in retired:
        st.warning(md(f"`{_var}` is retired and no longer read — "
                      + RETIRED_VARS[_var]), icon="⚠️")
    # Two models, because they are not the same job: the chat reasons over
    # tool output turn after turn (~17x the news panel's token usage), while
    # the news read is one call a day. Cheap-and-thinking for the first,
    # strong for the second.
    model_name = st.text_input("Chat model", DEFAULT_MODEL,
                               help="The Analyst tab's chat. Reasons over tool "
                                    "output every turn, so it is the expensive "
                                    "one. Override here or set QBS_GEMINI_MODEL.")
    thinking = st.number_input(
        "Thinking budget", min_value=-1, max_value=32768,
        value=int(_default_thinking_budget() or -1), step=512,
        help="Chat model only. −1 lets it decide, 0 turns thinking off, a "
             "positive number caps it in tokens. The accepted range is "
             "model-specific — the API rejects a bad one, this app does not "
             "second-guess it. Or set QBS_THINKING_BUDGET.")
    summary_model = st.text_input(
        "News summary model", DEFAULT_SUMMARY_MODEL,
        help="The News tab. One call a day over ~30 headlines, so the stronger "
             "model costs pennies a month. No thinking budget is sent — this "
             "is extraction into a fixed shape, not multi-step reasoning. "
             "Or set QBS_SUMMARY_MODEL.")
    allow_web = st.checkbox("Allow web search", value=bool(backends),
                            disabled=not backends,
                            help=("Search backends found: "
                                  + (", ".join(backends) or "none — "
                                     "pip install ddgs")))
    live_fundamentals = st.checkbox(
        "Fetch fundamentals live", value=True,
        help="Off reads only what is already cached in data/fundamentals/.")
    # `value=not news_blocker`, not `value=True`. The box is the in-app
    # mirror of the switch, and a ticked box over a switched-off feature is
    # the UI telling you the opposite of what is happening.
    daily_news = st.checkbox(
        "News sentiment analysis", value=not news_blocker,
        disabled=bool(news_blocker),
        help=f"One Gemini call per day on the News tab, cached to "
             f"data/sentiment/. Off still shows the headlines — only the "
             f"model's read of them goes away. Off by default: set "
             f"{DISABLE_NEWS_ANALYSIS_VAR}=0 in your .env to allow it.")


tab_picks, tab_market, tab_news, tab_analyst = st.tabs(
    ["📋 Daily picks", "📊 Market overview", "📰 News & sentiment",
     "🤖 Analyst"])


# ==========================================================================
# Tab 1 -- daily selections
# ==========================================================================

with tab_picks:
    freshness_banner()
    st.subheader("Daily picks")
    st.caption(f"{len(STRATEGY_LABELS)} selection strategies · universe {UNIVERSE_NOTE} · "
               f"data through {session_text(LAST_BAR)} · {SRC}")

    dates = [d for d in selections["momentum"].index
             if d >= pd.Timestamp(cfg.backtest_start)]
    asof = st.select_slider("Date", options=dates, value=dates[-1],
                            format_func=lambda d: f"{d:%Y-%m-%d}")

    # Three books, so the panel is narrower than it was -- but each column
    # holds one short ticker list, and splitting them across two rows would
    # put the chart out of eyeshot of the names it is meant to explain.
    books = ("momentum", "resmom", "finviz")
    cols = st.columns([1, 1, 1, 3])
    RP = ResidualMomentumParams()
    # The long "why" lives in a metric tooltip rather than under the header:
    # three columns is narrow enough that a paragraph there would push this
    # book's ticker table a screenful below its neighbours', and the tables
    # are what the panel is for.
    BOOK_HELP = {
        "resmom": (
            "Same universe, same slot machinery, same hysteresis rule and "
            f"the same absolute filter against {RP.safe_asset} as the "
            "momentum book — the only thing that differs is what the "
            "ranking sorts on. So "
            "whatever these two columns disagree about is the market "
            "component of the score, and nothing else. Run deeper than its "
            "sibling because a residual score is meant to pick names that "
            "are less alike; see docs/RESIDUAL_MOMENTUM.md."),
    }
    picks: Dict[str, set] = {}
    for col, key in zip(cols[:3], books):
        frame = selections[key]
        row = frame.loc[asof] if asof in frame.index else None
        raw = row["holdings"] if row is not None and row["holdings"] else ""
        names = [t for t in raw.split(", ") if t]
        picks[key] = set(names)
        with col:
            st.markdown(f"**{STRATEGY_LABELS[key]}**")
            st.metric("Names held", len(names), help=BOOK_HELP.get(key))
            if key == "resmom":
                st.caption(md(
                    f"{int(n_resid)} slots, band +{RESID_BAND} · "
                    f"{momentum_label(RP)} momentum of each return **net of "
                    f"a {RP.beta_window}-day beta to {RP.market_asset}**, "
                    "over that residual's own vol."))
            if key == "finviz":
                # Say it here rather than leaving someone to wonder why a
                # "top-20" shows 6 names.
                short = len(names) < int(n_screen)
                p_scr = FinvizScreenParams()
                vol_note = (
                    f"volume > {p_scr.min_volume/1e3:,.0f}k shares/day ✅"
                    if SCREEN_VOLUME_APPLIED else
                    f"**the volume leg (> {p_scr.min_volume/1e3:,.0f}k "
                    "shares/day) is not applied** — no volumes in this cache, "
                    "so the screen is more permissive than its own definition")
                st.caption(md(
                    (f"{len(names)} of {int(n_screen)} slots — fewer names "
                     "cleared the filter than the screen ranks down to. "
                     if short else "")
                    + f"Filter: close > ${p_scr.min_price:.0f} · "
                    f"quarterly gain > {p_scr.min_quarter_return:.0%} · "
                    + vol_note))
            if names:
                st.dataframe(pd.DataFrame({"Ticker": names}), hide_index=True,
                             width="stretch",
                             height=min(420, 38 + 35 * len(names)))
            else:
                st.info("Nothing held — fully in cash.")
            if row is not None and row["buys"]:
                st.caption(f"🟢 Bought: {row['buys']}")
            if row is not None and row["sells"]:
                st.caption(f"🔴 Sold: {row['sells']}")

    # ---- right-hand panel: price, EMAs and the levels that matter ---------
    with cols[3]:
        universe_names = list(uni.columns)
        picked = sorted(set().union(*picks.values()))
        options = picked + [t for t in universe_names if t not in picked]
        price_panel(uni, px, asof, options, int(n_hold), "picks")

    # ---- what the books agree on -----------------------------------------
    # One "held by both" line stopped being answerable at three books: it no
    # longer says which two. The pairs are listed instead, momentum ∩
    # residual first, because that pair shares its universe, its slots and
    # its engine -- what it does NOT share is exactly what removing the
    # market component from the score changed, and nothing else.
    SHORT = {"momentum": "momentum", "resmom": "residual", "finviz": "screen"}
    st.markdown("**Held in common**")
    ov = st.columns(4)
    for c, (a, b) in zip(ov, (("momentum", "resmom"), ("momentum", "finviz"),
                              ("resmom", "finviz"))):
        both = picks[a] & picks[b]
        smaller = min(len(picks[a]), len(picks[b]))
        with c:
            st.caption(md(
                f"**{SHORT[a]} ∩ {SHORT[b]}** — "
                + (", ".join(sorted(both)) if both else "no overlap")
                + f" ({len(both)} of {smaller or '—'})"))
    with ov[3]:
        all3 = picks["momentum"] & picks["resmom"] & picks["finviz"]
        st.caption(md(
            "**all three** — "
            + (", ".join(sorted(all3)) if all3 else "no overlap")
            + f" ({len(all3)})"))

    # ---- volume and hammer candles for the names on the page -----------
    st.divider()
    st.markdown("#### Volume & hammer candles")
    held_by: Dict[str, str] = {}
    for key in books:
        for t in sorted(picks[key]):
            held_by[t] = (held_by[t] + ", " if t in held_by else "") + SHORT[key]
    for t in WATCHLIST:
        held_by.setdefault(t, "watchlist")
    cand_names = list(held_by)
    if not cand_names:
        st.caption("No held or watched names on this date.")
    else:
        vwin = st.number_input(
            "Average volume window (sessions)", 5, 120, 20, step=5,
            key="candle_vol_window",
            help="Sessions averaged for the baseline. The last session is "
                 "excluded from its own baseline, so a spike is measured "
                 "against days it did not help set.")
        HR = HammerRules()
        ctab, verdicts, no_ohlc = candle_table(
            cand_names, held_by, asof, int(vwin), HR)
        vol_cols = [c for c in ctab.columns if c.startswith(("Last vol", "Avg vol"))]
        val_col = next(c for c in ctab.columns if c.startswith("Avg $ vol"))
        HAMMER_TINT = {"hammer": f"background-color: {UP}",
                       "hanging": f"background-color: {DN}",
                       "shape": ""}
        cstyle = (ctab.style
                  .format({**{c: lambda v: fmt(v, "{:,.0f}") for c in vol_cols},
                           "Vol ratio": lambda v: fmt(v, "{:.2f}×"),
                           val_col: lambda v: fmt(
                               v / 1e6 if v == v else v, "${:,.1f}M")})
                  .background_gradient(subset=["Vol ratio"], cmap="Blues",
                                       vmin=0.5, vmax=2.5))
        for c, vs in verdicts.items():
            cstyle = cstyle.apply(
                lambda _col, vs=vs: [HAMMER_TINT.get(v, "") if v else ""
                                     for v in vs], subset=[c])
        st.dataframe(cstyle, hide_index=True, width="stretch",
                     height=min(620, 38 + 35 * len(ctab)))
        st.caption(md(
            f"Last session's volume against the average of the **{int(vwin)} "
            "sessions before it** (the last one excluded), and the ratio of the "
            "two; *Avg $ vol* is close × shares over the same window. "
            "*Bar* is the date of the last candle read, which can trail the "
            "slider when a name's OHLC is behind the ranking cache. "
            "**Hammer D0 / D-1 / D-2** are the last three candles, newest "
            "first, each showing the lower shadow as a share of the day's "
            "range. A candle is a hammer **shape** when: body ≤ "
            f"{HR.max_body:.0%} of the range · lower shadow ≥ "
            f"{HR.min_lower_to_body:g}× the body **and** ≥ {HR.min_lower:.0%} "
            f"of the range · upper shadow ≤ {HR.max_upper:.0%} of the range · "
            f"range ≥ {HR.min_range_atr:g}× the prior {HR.atr_window}-day ATR "
            "(a tiny range says nothing). 🔨 is that shape **after a "
            f"{HR.trend_days}-session decline** — the reversal pattern. ⚠️ is "
            "the same shape after a rise, a *hanging man*, which reads the "
            "other way. ◐ is the shape with a flat prior trend. "
            "Descriptive only — nothing here changes a ranking or a holding."
            + (f" No OHLC for {', '.join(no_ohlc)}"
               + ("" if online else " — switch **Source** to Online to fetch it")
               + "; volume there, if shown, is from the universe cache."
               if no_ohlc else "")))

    # ---- watchlist: where a name places, without the book buying it -----
    st.divider()
    st.markdown("#### Watchlist rank")
    if not WATCHLIST:
        st.caption(md(
            "Nothing watched. Put tickers in the sidebar box (or set "
            f"`{WATCHLIST_VAR}` in the repo root's `.env`) to see where they "
            "place in the ranking the book acts on. A name outside the index "
            "is interpolated into that ranking without joining it."))
    else:
        if WATCH_ERR:
            # The remedy depends on the mode. Telling someone to go online
            # when they already are, and the download is what failed, sends
            # them to fix the wrong thing.
            remedy = ("Check the spelling — a name the provider does not "
                      "know cannot be ranked."
                      if online else
                      "Offline mode reads only the CSV cache in `data/`. "
                      "Switch **Source** to Online to fetch a name for the "
                      "first time.")
            st.warning(md("No prices for " + ", ".join(
                f"**{t}** ({e})" for t, e in WATCH_ERR.items())
                + ". " + remedy), icon="⚠️")

        rows = (watch_rows(uni, px["BOXX"], px["QQQ"], WATCH_FRAME,
                           ",".join(WATCHLIST), int(n_hold), int(exit_rank),
                           int(n_resid), asof)
                if not WATCH_FRAME.empty else [])
        if not rows:
            st.info("Nothing on the watchlist could be ranked on this date.")
        else:
            styler, show_w, wf = rank_table(rows, int(n_hold), int(exit_rank))
            st.dataframe(styler, hide_index=True, width="stretch",
                         height=min(420, 38 + 35 * len(show_w)))
            n_out = int((~wf["constituent"]).sum())
            # Said only when there is an interpolated row to say it about.
            # "the 0 outsiders are interpolated" explains a distinction the
            # table is not currently making.
            outsider_note = (
                f" The {n_out} interpolated "
                f"{'row shows' if n_out == 1 else 'rows show'} where a "
                f"non-constituent would sit, so a rank inside the top "
                f"{int(n_hold)} means the book *would* hold it **if it were "
                "a constituent** — not that the book should."
                if n_out else "")
            st.caption(md(
                f"Ranked on {asof:%Y-%m-%d} against the {uni.shape[1]} "
                "constituents alone, so two watched names never shift each "
                "other's row, and nothing here changes a holding. A "
                "constituent shows the rank it already has. *Rank* is the "
                "momentum book's ranking; *Resid. rank* places the same name, "
                "the same way, in the **residual momentum** book's ranking "
                "(12-1 momentum net of its beta to QQQ) — a name far better "
                "on *Rank* than on *Resid. rank* owes its strength mostly to "
                "the market."
                + outsider_note
                + " A blank rank means the name was filtered out (too little "
                "history, or it lost to BOXX over the same window), not that "
                "it placed last."))

    st.divider()
    st.markdown("#### Selection history")
    which = st.radio("Strategy", list(STRATEGY_LABELS),
                     format_func=lambda k: STRATEGY_LABELS[k],
                     horizontal=True, key="hist")
    hist = selections[which].loc[selections[which].index <= asof].tail(120).iloc[::-1]
    show = hist.reset_index().rename(columns={
        "date": "Date", "holdings": "Holdings", "n": "N",
        "buys": "Bought", "sells": "Sold"})
    show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, hide_index=True, width="stretch", height=460)
    st.download_button("Download CSV",
                       show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"picks_{which}_{asof:%Y%m%d}.csv",
                       mime="text/csv")


# ==========================================================================
# Tab 2 -- market overview
# ==========================================================================

with tab_market:
    freshness_banner()


    st.subheader("Breadth & momentum monitor")

    use_us = st.toggle(
        "Measure the US market (Finviz universe)", value=True, key="use_us",
        help="Off falls back to the cached Nasdaq-100 constituents, which is "
             "an index, not the market.")

    mkt_closes = mkt_vols = None
    mkt_sectors: Dict[str, str] = {}
    mkt_note, mkt_err, mkt_fetched, mkt_fill = "", None, False, None
    if use_us:
        (mkt_closes, mkt_vols, mkt_sectors, mkt_note, mkt_err,
         mkt_fetched, mkt_fill) = load_us_market(
            download_start, bool(online), bool(force), BAR_EPOCH,
            st.session_state["refresh_token"])

    if use_us and mkt_closes is not None:
        m_uni = mkt_closes
        m_vols = mkt_vols
        universe_label = f"{m_uni.shape[1]} US names · {mkt_note}"
        breadth_m = build_breadth(m_uni, px["QQQ"], universe_label, m_vols,
                                  bar_epoch=BAR_EPOCH, _qqq_ohlc=QQQ_OHLC,
                                  _spy=SPY_CLOSE, _spy_ohlc=SPY_OHLC)
        # Say which of the two happened. "Fetched just now" and "served from a
        # cache built at some point" look identical on screen otherwise, and
        # the difference is the whole reason for the auto-refresh.
        m_behind = sessions_behind(m_uni.index.max())
        age = ("current" if m_behind <= 0 else
               f"{m_behind} session{'s' if m_behind != 1 else ''} behind")
        _, why_not = due_for_fetch()
        st.caption(
            f"🌐 {m_uni.shape[1]:,} names · last bar "
            f"{session_text(m_uni.index.max())} ({age}) · "
            + ("**fetched on this run**" if mkt_fetched else why_not))
        if mkt_fill:
            st.info(md("🔗 " + mkt_fill), icon="🧩")
        if mkt_err:                      # loaded, but not cleanly
            st.warning(f"**Partial US universe.** {mkt_err}", icon="⚠️")
    else:
        if use_us:
            st.error(
                f"**Falling back to the Nasdaq-100** — the numbers below are an "
                f"index, not the market.\n\n**Reason:** {mkt_err}", icon="🚫")
            st.caption(
                "Streamlit hides tracebacks, so if that reason is not enough, run "
                "`python -m qbs.finviz` in the same environment as this app: it "
                "checks the interpreter, the package, the filters, the screener "
                "and yfinance in order and names the step that breaks. The most "
                "common cause is `finvizfinance` being installed in a notebook or "
                "on Colab rather than for the interpreter running Streamlit.")
        else:
            st.warning(
                f"**Measuring {uni.shape[1]} Nasdaq-100 constituents, not the US "
                "market.** A count of names up 4% out of 99 mega-caps is a "
                "different measurement from the same count out of ~2,400 — not a "
                "smaller version of it. Turn the toggle on for the real universe.",
                icon="⚠️")
        m_uni, m_vols, mkt_sectors = uni, None, {}
        universe_label = UNIVERSE_NOTE
        breadth_m = breadth

    breadth = breadth_m
    tbl = breadth.table
    tbl = tbl.loc[tbl.index >= pd.Timestamp(cfg.backtest_start)]
    if tbl.empty:
        st.error("Breadth table is empty — not enough history.")
        st.stop()
    last = tbl.iloc[-1]
    prev = tbl.iloc[-2] if len(tbl) > 1 else None

    st.caption(
        f"Close {tbl.index[-1]:%Y-%m-%d} · universe {universe_label} · "
        f"sample {int(last['n_stocks'])} names · source {SRC}"
    )

    # ---- KPI row ---------------------------------------------------------
    # `delta` is reserved for actual session-on-session change, so it keeps its
    # arrow and its colour. Everything else is a descriptor and goes in a
    # caption -- a green up-arrow on the words "SPY not cached here", or on a
    # NEGATIVE leaders reading, is just a lie told in punctuation.
    k = st.columns(6)
    ratio = last["up4"] / last["dn4"] if last["dn4"] else np.nan
    k[0].metric("Up 4% / Down 4%",
                f"{fmt(last['up4'], '{:.0f}')} / {fmt(last['dn4'], '{:.0f}')}")
    k[0].caption(fmt(ratio, "up:down ratio {:.2f}"))

    d_fast = (last["pct_above_fast"] - prev["pct_above_fast"]) if prev is not None else np.nan
    k[1].metric("% above 20-day", fmt(last["pct_above_fast"]), fmt(d_fast, "{:+.1f}"))
    k[1].caption("vs prior session")

    d_slow = (last["pct_above_slow"] - prev["pct_above_slow"]) if prev is not None else np.nan
    k[2].metric("% above 50-day", fmt(last["pct_above_slow"]), fmt(d_slow, "{:+.1f}"))
    k[2].caption("vs prior session")

    k[3].metric("SPY vs 50D EMA", fmt(last["spy_atr"], "{:+.2f}"))
    if SPY_CLOSE is None:
        k[3].caption(
            "no SPY data — " + ("the download failed" if online else
                                "offline and not cached; switch Source to "
                                "Online to fetch it once"))
    else:
        k[3].caption("in units of 14-day ATR"
                     + ("" if SPY_OHLC is not None else
                        " · close-only range, reads high"))

    k[4].metric("QQQ vs 50D EMA", fmt(last["qqq_atr"], "{:+.2f}"))
    k[4].caption("in units of 14-day ATR"
                 + ("" if QQQ_OHLC is not None else
                    " · close-only range, reads high"))

    k[5].metric("Leaders (MLI)", fmt(last["mli_pct"], "{:+.2f}%"))
    k[5].caption(f"{int(last['mli_n'])} members · "
                 f"{fmt(last['mli_up_pct'], '{:.1f}%')} advancing")

    # ---- 4% pulse --------------------------------------------------------
    st.markdown("#### 4% momentum pulse")
    st.caption("Above the axis: names closing up 4% or more versus the prior session. "
               f"Below: names down 4% or more. Dark = count ≥ {BreadthParams().pulse_strong}.")
    pulse_n = st.slider("Sessions shown", 10, 120, 20, key="pulse_days")
    pulse = tbl.tail(pulse_n)
    bars = pd.concat([
        pd.DataFrame({"date": pulse.index, "value": pulse["up4"],
                      "cls": [PULSE_LABELS[pulse_class(v, "up")] for v in pulse["up4"]]}),
        pd.DataFrame({"date": pulse.index, "value": -pulse["dn4"],
                      "cls": [PULSE_LABELS[pulse_class(v, "down")] for v in pulse["dn4"]]}),
    ]).dropna(subset=["value"])
    (bars,), pulse_axis = session_axis(bars)
    chart = alt.Chart(bars).mark_bar(
        size=max(2.0, 620 / max(len(pulse), 1))).encode(
        x=alt.X("n:Q", axis=pulse_axis, title=None,
                scale=alt.Scale(nice=False, zero=False)),
        y=alt.Y("value:Q", title="Number of stocks"),
        color=alt.Color("cls:N", scale=alt.Scale(
            domain=[PULSE_LABELS[k_] for k_ in
                    ("up_strong", "up", "down", "down_strong")],
            range=[UP_STRONG, UP, DN, DN_STRONG]), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("value:Q", title="Count")],
    ).properties(height=260)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED).encode(y="y:Q")
    st.altair_chart(chart + zero, use_container_width=True)

    # ---- the daily monitor table -----------------------------------------
    st.markdown("#### Daily monitor")
    st.caption("The SPY and QQQ ATR columns are (close − 50-day EMA) ÷ 14-day "
               "ATR — how many ATRs each index sits from its own 50-day line.")
    n_rows = st.slider("Sessions shown", 10, 250, 20, key="table_days")
    view = tbl.tail(n_rows).iloc[::-1]
    disp = pd.DataFrame({
        "Date": view.index.strftime("%Y-%m-%d"),
        "Up 4%": view["up4"],
        "Dn 4%": view["dn4"],
        "% > 20D": view["pct_above_fast"],
        "% > 50D": view["pct_above_slow"],
        "SPY ATR": view["spy_atr"],
        "QQQ ATR": view["qqq_atr"],
        "MLI %": view["mli_pct"],
        "MLI adv%": view["mli_up_pct"],
        "MLI N": view["mli_n"].astype(int),
        "Sample": view["n_stocks"].astype(int),
    })

    def _style(col: pd.Series):
        name = col.name
        if name in ("Up 4%", "Dn 4%"):
            which = "up" if name.startswith("Up") else "down"
            return [f"background-color: {PULSE_CELL[pulse_cell(v, which)]}"
                    for v in col]
        if name == "% > 20D":
            # `enumerate` over the column, not the index: `disp` is built
            # newest-first, so position 0 IS the latest session. Reading the
            # date instead would break the moment the sort order changed.
            return [f"background-color: {PULSE_CELL[ma_fast_cell(v, i)]}"
                    for i, v in enumerate(col)]
        if name == "% > 50D":
            return [f"background-color: {CELL[ma_class(v, 'slow')]}" for v in col]
        if name in ("SPY ATR", "QQQ ATR"):
            return [f"background-color: {CELL[atr_class(v)]}" for v in col]
        return ["" for _ in col]

    styled = (disp.style
              .apply(_style, axis=0)
              .format({"Up 4%": "{:.0f}", "Dn 4%": "{:.0f}",
                       "% > 20D": "{:.1f}", "% > 50D": "{:.1f}",
                       "SPY ATR": "{:+.2f}", "QQQ ATR": "{:+.2f}",
                       "MLI %": "{:+.2f}", "MLI adv%": "{:.1f}"}, na_rep="—"))
    st.dataframe(styled, hide_index=True, width="stretch",
                 height=min(720, 45 + 35 * len(disp)))
    if breadth.gaps:
        gap_txt = ", ".join(f"**{d:%Y-%m-%d}** ({n:,} of ~{ref:,} names)"
                            for d, (n, ref) in sorted(breadth.gaps.items())[-5:])
        st.warning(md(
            f"**Left out — most names have no close:** {gap_txt}. Kept in, a "
            "day like that makes the next day's return undefined for every "
            "missing name and the 20- and 50-day averages undefined for weeks "
            "after it, so the table would be read over a few hundred names. "
            "The row after it shows its 4% counts as **—** because its move "
            "spans two sessions. The next update re-fetches a gap in the last "
            "60 sessions; **Refresh now** rebuilds the whole cache. If it "
            "stays, the provider has no data for that day."), icon="🕳️")
    _bp = BreadthParams()
    _u, _d = _bp.up4_bands, _bp.dn4_bands
    st.caption(
        "**Green is bullish in both 4% columns**, not \u201cgreen means "
        "up\u201d — a quiet down-4% count is a good day and shades green "
        "like a heavy up-4% one. "
        f"Up 4%: dark red ≤{_u[0]:.0f} · light red ≤{_u[1]:.0f} · light green "
        f"≤{_u[2]:.0f} · dark green above. "
        f"Dn 4%: dark green ≤{_d[0]:.0f} · light green ≤{_d[1]:.0f} · light "
        f"red ≤{_d[2]:.0f} · dark red above. "
        f"**% > 20D** is shaded on the **last {_bp.ma_fast_recent} sessions "
        f"only** — green above {_bp.ma_fast_green:.0f}%, red at or below. It "
        "reads the tape now, and a shaded year of it is wallpaper. "
        "% > 50D keeps the full-history scale: red below 20%, green above 80%. "
        "ATR shades red beyond ±5. The bar chart above keeps the plain "
        "up-is-green convention, since a signed bar already shows direction."
    )
    # The bands are absolute counts, sized for the ~2,400-name US universe. On
    # the 99-name fallback no session can reach 301 up, so the column goes
    # uniformly dark red and reads as a crash that is not happening. Say so
    # rather than let the colour be believed.
    _sample = int(tbl["n_stocks"].iloc[-1]) if "n_stocks" in tbl else 0
    if _sample and _sample < _u[2]:
        st.warning(
            f"**These colours are calibrated for the broad US universe.** The "
            f"bands are absolute counts (dark green needs {_u[2]:.0f}+ names up "
            f"4%), and this sample is **{_sample} names** — no session here can "
            f"reach that, so Up 4% shades dark red throughout and means nothing. "
            "Turn the US universe on above for the scale to apply.", icon="🎨")

    # ---- checklist ---------------------------------------------------------
    st.divider()
    st.markdown("#### Checklist")
    auto = {r["key"]: r for r in build_checklist(
        m_uni, px["QQQ"], universe_label, m_vols, bar_epoch=BAR_EPOCH)}
    manual: Dict[str, bool] = {}
    mc = st.columns(2)
    for col, item in zip(mc, [i for i in CHECKLIST if not i["auto"]]):
        manual[item["key"]] = col.checkbox(
            item["q"], key=f"check_{item['key']}",
            help="Not in this data — tick it yourself. Kept for this session "
                 "only.")

    rows, answers = [], []
    for item in CHECKLIST:
        if item["auto"]:
            r = auto.get(item["key"], {})
            ans, reading = r.get("answer"), r.get("reading", "—")
        else:
            ans = manual.get(item["key"], False)
            reading = "ticked by you" if ans else "not ticked"
        answers.append(ans)
        rows.append({"Question": item["q"],
                     'Answering "Yes" means': item["yes"],
                     "Answer": "—" if ans is None else ("Yes" if ans else "No"),
                     "Reading": reading})
    ck = pd.DataFrame(rows)
    st.dataframe(
        ck.style.apply(lambda _c: [f"background-color: {DN}" if a else ""
                                   for a in answers], subset=["Answer"]),
        hide_index=True, width="stretch",
        column_config={"Question": st.column_config.TextColumn(width="large"),
                       'Answering "Yes" means': st.column_config.TextColumn(width="medium"),
                       "Reading": st.column_config.TextColumn(width="medium")},
        height=45 + 35 * len(ck))
    score = sum(i["weight"] for i, a in zip(CHECKLIST, answers) if a)
    total = sum(i["weight"] for i in CHECKLIST)
    unknown = sum(1 for a in answers if a is None)
    st.metric("Checklist score", f"{score} / {total}",
              help="Each Yes counts 1; the stop-loss question counts 2.")
    st.caption(md(
        f"The first four are answered from the data on **{tbl.index[-1]:%Y-%m-%d}**, "
        f"over **this universe ({m_uni.shape[1]:,} names)** rather than the NYSE, "
        "with **QQQ** as the index. The thresholds: Q1 counts consecutive "
        "sessions with under 20% of names above their 40-day average (Yes above "
        "10). Q2 asks only when QQQ closes at a 52-week high, and measures the "
        "momentum leaders (the MLI set) against their 50-day average. Q3 is "
        "QQQ up over 5 sessions while more names sit at a 52-week low than a "
        "52-week high. Q4 is fewer leaders than 5 sessions ago **and** the "
        "fewest in 10 — still falling, no floor yet. FedWatch and your own "
        "stop-losses are not in this data, so they are yours to tick."
        + (f" {unknown} row{'s' if unknown != 1 else ''} could not be "
           "answered for lack of history." if unknown else "")))

    # ---- momentum leaders -------------------------------------------------
    st.divider()
    st.markdown("#### Momentum leaders")
    p = BreadthParams()
    note = (f"Rules: US stock or ADR · close > ${p.leader_min_price:.0f} · "
            f"quarterly gain > {p.leader_min_quarter_return:.0%} "
            f"({p.leader_quarter_days} sessions)")
    if breadth.has_volume:
        note += f" · volume > {p.leader_min_volume/1e3:,.0f}k shares/day ✅"
    else:
        note += (f" · ⚠️ the volume test (> {p.leader_min_volume/1e3:,.0f}k "
                 "shares/day) cannot run without volume, so this leader count "
                 "is an over-estimate")
    st.caption(md(note))

    c = st.columns(3)
    prev_n = int(prev["mli_n"]) if prev is not None else None
    c[0].metric("Leaders", int(last["mli_n"]),
                f"{int(last['mli_n']) - prev_n:+d}" if prev_n is not None else None)
    c[0].caption("vs prior session")
    c[1].metric("Share of analysed names",
                fmt(100.0 * last["mli_n"] / last["n_stocks"], "{:.1f}%"))
    c[1].caption(f"out of {int(last['n_stocks'])} names")
    c[2].metric("Leaders' average move", fmt(last["mli_pct"], "{:+.2f}%"))
    c[2].caption(fmt(last["mli_up_pct"], "{:.1f}% advancing"))

    trend = (100.0 * tbl["mli_n"] / tbl["n_stocks"]).rename("pct").reset_index()
    trend.columns = ["date", "pct"]
    (trend_n,), trend_axis = session_axis(trend.tail(250))
    st.altair_chart(
        alt.Chart(trend_n).mark_line(color="#2a4b8d").encode(
            x=alt.X("n:Q", axis=trend_axis, title=None,
                    scale=alt.Scale(nice=False, zero=False)),
            y=alt.Y("pct:Q", title="Leaders as % of sample"),
            tooltip=[alt.Tooltip("date:T", title="Date"),
                     alt.Tooltip("pct:Q", title="%", format=".1f")],
        ).properties(height=240),
        use_container_width=True)

    # ---- sector concentration --------------------------------------------
    st.markdown("##### Sector composition")
    if not mkt_sectors:
        st.info(
            "**Not shown — no `ticker → sector` map.** The Nasdaq-100 cache carries "
            "no sector classification, so the panel stays blank rather than "
            "bucketing everything into one label and rendering that as a finding. "
            "Turn on the US universe above: the Finviz screener returns Sector in "
            "the same response, which is what fills this in.", icon="ℹ️")
    else:
        sect = sector_breakdown(m_uni, mkt_sectors, asof=tbl.index[-1],
                                volumes=m_vols)
        if sect.empty:
            st.info("No momentum leaders on this date, so there is nothing to "
                    "break down by sector.", icon="ℹ️")
        else:
            top3 = sect["share_pct"].head(3).sum()
            sc = st.columns(3)
            sc[0].metric("Sectors represented", int(sect["sector"].nunique()))
            sc[1].metric("Top-3 concentration", f"{top3:.1f}%")
            sc[2].metric("Strongest sector", sect.iloc[0]["sector"])
            sc[2].caption(f"{sect.iloc[0]['share_pct']:.1f}% of leaders · "
                          f"{sect.iloc[0]['excess_pp']:+.1f}pp vs its own weight")

            show = sect.rename(columns={
                "sector": "Sector", "n": "N", "share_pct": "Share %",
                "pool_pct": "Pool %", "penetration": "Penetration %",
                "excess_pp": "Excess pp"})
            st.dataframe(
                show.style
                .format({"Share %": "{:.1f}", "Pool %": "{:.1f}",
                         "Penetration %": "{:.1f}", "Excess pp": "{:+.1f}"})
                .background_gradient(subset=["Excess pp"], cmap="RdYlGn"),
                hide_index=True, width="stretch",
                height=min(560, 45 + 35 * len(show)))
            st.caption(
                "**Excess pp is the column that carries information.** A sector "
                "holding 20% of the leaders is unremarkable if it is 20% of the "
                "universe; the same 20% from a sector that is 5% of the universe "
                "is the finding. Pool % is that sector's own weight — the bar it "
                "has to beat. Penetration is leaders ÷ analysed names in the sector."
            )

        # ---- who the leadership actually is -----------------------------
        # The table above says WHERE the leadership sits. A concentration
        # reading nobody can name the members of is a number to nod at
        # rather than act on, so this names them, in the same sector order.
        st.markdown("##### High-momentum names by sector")
        per_sector = st.number_input(
            "Names per sector", 1, 20, 5, key="sec_leaders_n",
            help="The strongest N leaders in each sector by 6-1 momentum. "
                 "The US universe throws up leaders in the hundreds, so the "
                 "list is capped; the sector label says how many it has.")
        lead_rows = sector_leaders(m_uni, mkt_sectors, asof=tbl.index[-1],
                                   volumes=m_vols,
                                   per_sector=int(per_sector))
        if lead_rows.empty:
            st.info("No leader on this date could be scored over the "
                    f"{momentum_label()} window — that needs more history "
                    "than this cache holds for them.", icon="ℹ️")
        else:
            # Ranked against the Nasdaq-100 book, NOT against the US
            # universe these names came from, and deliberately: the two
            # cutoff columns are the score holding slot n_hold and slot
            # exit_rank of the book that actually trades. "Beats the book"
            # has to mean the same thing here as it does on the picks tab,
            # or the same words carry two readings one scroll apart.
            in_book = uni.index[uni.index <= tbl.index[-1]]
            rank_asof = in_book[-1] if len(in_book) else None
            lead_names = [t for t in lead_rows["symbol"] if t in m_uni.columns]
            lead_frame = pd.DataFrame({
                t: m_uni[t].reindex(uni.index).ffill() for t in lead_names})
            rows_l = (watch_rows(uni, px["BOXX"], px["QQQ"], lead_frame,
                                 ",".join(lead_names), int(n_hold),
                                 int(exit_rank), int(n_resid), rank_asof)
                      if lead_names and rank_asof is not None else [])
            if not rows_l:
                st.info("These names could not be placed in the book's "
                        "ranking on this date.", icon="ℹ️")
            else:
                # The sector panel's own order, kept: grouped by sector,
                # strongest first inside it. A global sort by rank would
                # shuffle the groups apart, which is the one thing this
                # table is for.
                order = {t: i for i, t in enumerate(lead_rows["symbol"])}
                rows_l.sort(key=lambda r: order.get(r["symbol"], 1 << 30))
                sec_of = {r.symbol: f"{r.sector} ({int(r.n_sector)})"
                          for r in lead_rows.itertuples()}
                styler_l, show_l, _ = rank_table(
                    rows_l, int(n_hold), int(exit_rank),
                    lead={"Sector": sec_of}, sort=False)
                st.dataframe(styler_l, hide_index=True, width="stretch",
                             height=min(620, 38 + 35 * len(show_l)))
                shown = len(show_l)
                total = int(lead_rows["n_sector"].groupby(
                    lead_rows["sector"]).first().sum())
                st.caption(md(
                    f"The strongest {int(per_sector)} per sector — "
                    f"**{shown} of {total}** leaders on "
                    f"{tbl.index[-1]:%Y-%m-%d}. The number beside each sector "
                    "is how many leaders it has in total. "
                    f"**Ranked against the {uni.shape[1]} Nasdaq-100 "
                    "constituents**, the universe the book trades — not "
                    "against the US universe these names were screened from. "
                    "A leader outside the index is *interpolated* into that "
                    "ranking, so **Beats the book** means it would place in "
                    f"the top {int(n_hold)} **if it were a constituent**, not "
                    "that anything holds it."
                    # A leader with no rank is the one cell here that looks
                    # like a bug and is not: the two rules disagree on
                    # purpose, and the row is listed because the leader rule
                    # passed it.
                    + " A **blank rank** is a name the leader rule passed and "
                    "the ranker then filtered out — it lost to BOXX over the "
                    f"same {momentum_label()} window, or has too little "
                    "history. The leader rule tests a quarterly gain against "
                    "a threshold; the ranker tests the same name against "
                    "cash. Neither is a stricter version of the other."
                    + (f" Leaders read on {tbl.index[-1]:%Y-%m-%d}, ranks on "
                       f"{rank_asof:%Y-%m-%d} — the two caches are not "
                       "equally fresh."
                       if rank_asof != tbl.index[-1] else "")))


# ==========================================================================
# Tab 3 -- the LLM analyst
# ==========================================================================
# A Gemini agent that reads the tabs above through tools and writes about
# them. Everything it can quote is computed by this package; it has no
# arithmetic of its own and no way to change a parameter.
#
# The tool trace below every answer is not a debug view. It is how a reader
# checks a number against the call it came from, which is the only thing that
# separates a research note from a fluent guess.

with tab_news:
    freshness_banner()
    st.subheader("News & sentiment")

    st.session_state.setdefault("news_token", 0)
    _now = pd.Timestamp.now("UTC").tz_convert(None)
    _today = _now.strftime("%Y-%m-%d")
    NEWS_HOURS = 12

    # The model is optional here, and that is the point of this tab: the
    # headlines are worth reading with no key and no spend. `run_llm` gates
    # only the read on top.
    run_llm = bool(daily_news) and not news_blocker

    if fetch_off:
        # Nothing below this point can run without a feed, so it stops here
        # rather than rendering an empty page and blaming the search. Note
        # `news_read_disabled` returns None whenever the analysis is on, so
        # this branch cannot strand a read that was asked for.
        st.info(md(f"**No news is being fetched** — {fetch_off}"), icon="⏸️")
        st.caption(
            "Streamlit reads the environment once at start-up, so **restart "
            "the app** after changing this — a rerun alone will not pick it "
            "up.")

    # `else`, not `st.stop()`: every tab renders in one script run, so
    # stopping here would take the Analyst tab down with it.
    else:
        top = st.columns([3, 1])
        with top[1]:
            if st.button("Refresh news", width="stretch",
                         help="Re-search now. Also re-reads with the model when "
                              "the sentiment analysis is on."):
                st.session_state["news_token"] += 1
                load_news.clear()
                st.rerun()

        feed, summary, from_cache = load_news(
            _today, NEWS_HOURS, summary_model.strip(),
            st.session_state["news_token"], run_llm)

        # ---- the read, when there is a model to do it ------------------------
        with top[0]:
            if not run_llm:
                why = ("switched off in the sidebar" if not daily_news
                       else news_blocker)
                st.info(md(
                    f"**Headlines only — no sentiment analysis.** {why}\n\n"
                    "Everything below is the news itself, which needs no model "
                    "and costs nothing to read."), icon="📰")
            elif summary is None or summary.error:
                st.warning(
                    f"**No sentiment read.** "
                    f"{summary.error if summary else 'not run yet'}", icon="📰")
            else:
                tint = SENTIMENT_TINT.get(summary.label, "")
                st.markdown(
                    f"<div style='padding:.55rem .9rem;border-radius:.4rem;"
                    f"background:{tint or '#00000010'};display:inline-block'>"
                    f"<b>{summary.label.upper()}</b></div>",
                    unsafe_allow_html=True)
                if summary.headline:
                    st.markdown(f"**{summary.headline}**")
                for b in summary.bullets:
                    cites = " ".join(f"`[{n}]`" for n in b.get("sources", []))
                    st.markdown(f"- {b['point']} {cites}")

        if run_llm and summary is not None and not summary.error:
            # A cached read of a DIFFERENT set of stories is still what the model
            # said -- but saying so beats letting it pass as current.
            if summary.fingerprint and summary.fingerprint != feed.fingerprint():
                st.caption(
                    "🔁 This read covers an earlier set of stories than the "
                    "headlines below — the feed has moved on since. "
                    "**Refresh news** re-reads it.")
            if summary.warnings:
                with st.expander(f"⚠️ {len(summary.warnings)} thing(s) dropped "
                                 "from this summary"):
                    for w in summary.warnings:
                        st.markdown(f"- {w}")
                    st.caption(
                        "Bullets without a citation, and citations pointing "
                        "outside the headline list, are removed before you see "
                        "them — an unsourced claim in a finance summary cannot be "
                        "told apart from a remembered one.")
            st.caption(
                f"🤖 {summary.model} over {summary.n_articles} headlines"
                + (" · served from cache" if from_cache else " · read just now")
                + ". **This is a read of what was written, not a signal.** Nothing "
                "here is backtested and nothing enters a strategy; every bullet "
                "points back to a numbered headline below.")

        st.divider()

        # ---- the headlines, always ------------------------------------------
        st.markdown(f"#### Headlines — last {feed.hours} hours")
        # Which backend actually served this, not which one is configured. The
        # two differ silently when the key is set and the package is not, and
        # this panel's whole window rests on timestamps only one of them sends.
        _note = backend_note()
        if _note:
            st.warning(md(_note), icon="🔍")
        if feed.backends:
            # `feed.backends`, not the result's `source` -- that one is the
            # OUTLET (Reuters, CNBC), so reading backends off it printed
            # publishers where search engines belonged.
            st.caption("Searched with " + ", ".join(f"**{b}**" for b in feed.backends)
                       + (" (both, merged and de-duplicated)"
                          if len(feed.backends) > 1 else "")
                       + ". No model is involved in fetching these.")
        if not feed.headlines:
            st.warning(
                "**No headlines came back.** "
                + ("; ".join(feed.errors) if feed.errors else
                   "the search returned nothing for any query."), icon="🔍")
            st.caption(
                "Run `python -m qbs.agent --check` in this app's environment to "
                "see which search backends resolve. With none installed, "
                "`pip install ddgs` adds the keyless one.")
        else:
            # Say what the window really cost. The backends' narrowest filter is
            # one DAY, so the 12-hour window is applied here on each headline's
            # own timestamp -- and a headline without one cannot be checked.
            bits = [f"**{feed.total}** stories",
                    f"{feed.n_dated} timestamped inside the window"]
            if feed.n_undated:
                bits.append(f"**{feed.n_undated} undated** (kept, but the window "
                            f"could not be checked)")
            if feed.n_dropped:
                bits.append(f"{feed.n_dropped} older than {feed.hours}h, dropped")
            st.caption(" · ".join(bits))

            for i, r in enumerate(feed.headlines, 1):
                when = snt_parse_published(r.published)
                age = ""
                if when is not None:
                    mins = max(0, int((_now - when).total_seconds() // 60))
                    age = (f"{mins}m ago" if mins < 60 else
                           f"{mins // 60}h {mins % 60:02d}m ago")
                title = f"[{r.title}]({r.url})" if r.url else r.title
                st.markdown(f"`[{i}]` **{title}**")
                meta = " · ".join(x for x in (r.source, age or "no timestamp") if x)
                st.caption(meta + ("" if age else
                                   " — this one could not be checked against the "
                                   "12-hour window"))
                if r.snippet:
                    st.caption(r.snippet)

            if feed.errors:
                with st.expander(f"⚠️ {len(feed.errors)} search query "
                                 "returned an error"):
                    for e in feed.errors:
                        st.markdown(f"- {e}")
                    st.caption(
                        "Queries overlap on purpose, so one failing leaves a "
                        "thinner feed rather than an empty one.")
            st.caption(f"Fetched {feed.fetched_at.replace('T', ' ')} UTC · "
                       "re-searched at most every 30 minutes.")


with tab_analyst:
    freshness_banner()
    st.subheader("Ask the analyst")

    # The picks tab's date slider drives this tab too -- two sliders for one
    # date is a way to have the chart and the chat disagree about "today".
    asof_analyst = asof
    screen_names = names_on("finviz", asof_analyst)
    momentum_names = names_on("momentum", asof_analyst)

    if chat_off:
        # A chat switched off on purpose is not a misconfiguration, and the
        # "install this, paste a key there" advice below would send someone to
        # fix something that is not broken.
        st.info(
            f"**The chat is switched off.** `{DISABLE_CHAT_VAR}` is set, so "
            "this tab makes no Gemini call. **The News tab's read answers to "
            f"its own switch** (`{DISABLE_NEWS_ANALYSIS_VAR}`) and is "
            "unaffected — that is what separate switches are for, bringing "
            "one feature up at a time.", icon="⏸️")
        st.markdown(
            "```bash\n"
            f"unset {DISABLE_CHAT_VAR}          # or set it to 0\n"
            "python -m qbs.agent --check\n"
            "```")
        st.caption(
            "Streamlit reads the environment once at start-up, so **restart "
            "the app** after changing this — a rerun alone will not pick it up.")
    elif blocker:
        st.warning(
            f"**The analyst is not configured.** {blocker}\n\n"
            "Everything else in this dashboard works without it — the analyst "
            "reads results, it never produces them.", icon="🔌")
        st.markdown(
            "```bash\n"
            "pip install -r requirements-agent.txt\n"
            "cp .env.example .env && chmod 600 .env   # then paste your key in\n"
            "python -m qbs.agent --check\n"
            "```")
        st.caption(
            "Streamlit does not re-import a package that is already loaded, "
            "so **restart the app** after creating `.env` — a rerun alone "
            "will not pick the key up."
        )
    # Key names and where they came from. Never a value: this renders in a
    # browser and lands in screenshots.
    st.caption(f"🔑 `.env`: {env_load.summary()}")

    st.caption(
        "The analyst can read the current picks, any name's momentum profile, "
        "market breadth, the breakout funnel and trade log, company "
        "fundamentals from yfinance, and the web. It is told that every figure "
        "must come from one of those tools, and every call it made is listed "
        "under each answer so you can check the figures against their source."
    )

    a_cols = st.columns([1.35, 1])

    # ---- left: the same panel the picks tab shows ------------------------
    with a_cols[0]:
        # Your watchlist first, then the high-momentum screen, then the
        # momentum book, then everyone else. The watchlist leads because it
        # is the shortest list and the only one you typed yourself -- a name
        # you went out of your way to watch is the one you came here to ask
        # about, and it would otherwise be buried in ~100 constituents.
        watched = [t for t in WATCH_FRAME.columns]
        hi = [t for t in screen_names
              if t in uni.columns and t not in watched]
        mom = [t for t in momentum_names
               if t in uni.columns and t not in watched and t not in hi]
        rest = [t for t in uni.columns
                if t not in watched and t not in hi and t not in mom]
        a_options = watched + hi + mom + rest
        bits = []
        if watched:
            bits.append(f"**{len(watched)} watched** "
                        + (f"({len(WATCH_EXTRA)} outside the index) "
                           if WATCH_EXTRA else "")
                        + "first")
        bits.append(f"{len(hi)} high-momentum name{'s' if len(hi) != 1 else ''}")
        bits.append(f"{len(mom)} from the momentum book")
        bits.append(f"then the rest of the {len(a_options)} names")
        st.caption(md(", ".join(bits) + "."))
        # `extra` is what makes a non-constituent chartable here at all: its
        # prices are not in the ranking universe, so without it the panel
        # would offer the name and then report no history for it.
        chart_ticker = price_panel(
            uni, px, asof_analyst, a_options, int(n_hold), "analyst",
            # WATCH_FRAME, not the raw download: it is already on the
            # universe's calendar, so a name that does not trade on exactly
            # the same days joins without punching holes in the series.
            extra=WATCH_FRAME,
            ticker_help="Your watchlist first, then today's high-momentum "
                        "screen, then the momentum book, then the rest of "
                        "the universe. A watched name outside the index is "
                        "charted from its own prices and interpolated into "
                        "the constituents' ranking.")

    # ---- right: the chat -------------------------------------------------
    with a_cols[1]:
        st.markdown("**Finance assistant**")
        st.session_state.setdefault("chat", [])

        if st.session_state["chat"]:
            if st.button("Clear conversation", key="chat_clear"):
                st.session_state["chat"] = []
                st.rerun()

        # A fixed-height box so the chart beside it does not jump every time a
        # message lands, and so the input stays where the eye expects it.
        with st.container(height=520, border=True):
            if not st.session_state["chat"]:
                st.caption(
                    "Ask about the name on the left, the current books, "
                    "breadth, or the backtest. Follow-ups work — the "
                    "conversation is sent with each message, so "
                    "\u201cwhat about its fundamentals?\u201d knows what "
                    "\u201cit\u201d is.")
            for turn in st.session_state["chat"]:
                with st.chat_message(turn["role"]):
                    st.markdown(turn["content"])
                    for i, call in enumerate(turn.get("tool_calls", []), 1):
                        args = ", ".join(f"{k}={v!r}"
                                         for k, v in call["args"].items())
                        with st.expander(f"{i}. `{call['name']}({args})`"):
                            st.code(call["result"] or "(no output)",
                                    language="text")

        prompt = st.chat_input(
            f"Ask about {chart_ticker}…" if chart_ticker else "Ask the analyst…",
            key="chat_in", disabled=bool(blocker))
        if blocker:
            st.caption("💬 The chat needs the analyst configured — see above.")

        if prompt:
            # The agent gets the frames this app already loaded rather than
            # re-reading the cache: a three-tool answer would otherwise spend a
            # minute rebuilding a universe that is sitting in memory.
            book = Book(universe=uni, prices=px, cfg=cfg, note=UNIVERSE_NOTE)
            # The selected ticker rides along as context, so "is it extended?"
            # means the name on screen rather than whatever was mentioned last.
            asked = (f"[the chart on screen is showing {chart_ticker}] {prompt}"
                     if chart_ticker else prompt)
            history = [{"role": t["role"], "content": t["content"]}
                       for t in st.session_state["chat"]]
            st.session_state["chat"].append({"role": "user", "content": prompt})

            with st.spinner(f"Asking {model_name}…"):
                answer = analyse(
                    asked, model=model_name.strip() or None, history=history,
                    thinking_budget=int(thinking),
                    book=book, n_hold=int(n_hold), allow_web=bool(allow_web),
                    offline_fundamentals=not live_fundamentals)

            if answer.disabled:
                text = f"⏸️ **Switched off.** {answer.text}"
            elif answer.error:
                text = f"🚫 **Could not answer.** {answer.text}"
            elif not answer.tool_calls:
                # Say it in the transcript, not in a banner that the next
                # message scrolls away from the answer it is about.
                text = (answer.text + "\n\n---\n⚠️ *Answered without calling a "
                        "tool, so nothing above is sourced from your data. "
                        "Treat it as opinion.*")
            else:
                text = answer.text
            st.session_state["chat"].append(
                {"role": "assistant", "content": text,
                 "tool_calls": answer.tool_calls})
            st.rerun()

        st.caption(
            "Every number should appear in one of the tool calls folded under "
            "the answer. One that does not is a fabrication. Only the last 20 "
            "turns are re-sent, and tool output is never replayed — the model "
            "calls the tool again when it needs the numbers twice."
        )
