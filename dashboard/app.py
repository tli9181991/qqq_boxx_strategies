"""Streamlit dashboard: daily strategy selections, and a market breadth monitor.

    streamlit run dashboard/app.py

Two tabs:

* **Daily picks** -- what each of the three selection strategies held on each
  day, with the entries and exits that changed it.
* **Market overview** -- the breadth monitor: 4% movers, percent holding the
  moving averages, index stretch in ATR units, and the momentum-leader group.

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

from qbs.breadth import (BreadthParams, atr_class, daily_breadth, ma_class,
                         momentum_profile, pulse_class, sector_breakdown)
from qbs.breakout import closes_to_bars, levels_in_view, sr_levels
from qbs.config import BreakoutParams, Config, FinvizScreenParams
from qbs.data import (freshness_note, load_daily_ohlc, load_prices,
                      sessions_behind)
from qbs.finviz import (UniverseFilters, fetch_us_universe, load_universe_bars,
                        sector_map)
from qbs.screens import finviz_momentum_screen
from qbs.strategies import cross_sectional_momentum
from qbs.universe import load_universe, load_universe_prices

st.set_page_config(page_title="Strategy picks / Market overview", layout="wide",
                   initial_sidebar_state="expanded")

MUTED = "#898781"
UP, UP_STRONG = "#a9d7bd", "#1b7a4b"
DN, DN_STRONG = "#f0b2b2", "#b02525"
CELL = {"extreme_low": "#f6c9c9", "low": "#fbe6e6", "mid": "",
        "high": "#ddefe4", "extreme_high": "#a9d7bd", "none": "",
        "stretched": "#f6c9c9", "oversold": "#c9e6d5", "normal": ""}

STRATEGY_LABELS = {
    "momentum": "Top-6 NDX momentum (12-1)",
    "finviz": "Top-6 Finviz screen",
    "breakout": "Weekly breakout watchlist",
}
PULSE_LABELS = {"up_strong": f"Up 4% ≥ {BreadthParams().pulse_strong}",
                "up": f"Up 4% < {BreadthParams().pulse_strong}",
                "down": f"Down 4% < {BreadthParams().pulse_strong}",
                "down_strong": f"Down 4% ≥ {BreadthParams().pulse_strong}"}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _read_cache(download_start: str, fetch_universe: bool = False):
    """Whatever is on disk, without touching the network."""
    tickers = load_universe(fetch=fetch_universe, warn=False)
    uni = load_universe_prices(tickers, start=download_start, verbose=False)
    px = load_prices(["QQQ", "VEU", "BOXX"], start=download_start, offline=True)
    return uni, px


@st.cache_data(show_spinner="Loading prices…")
def load_data(download_start: str, online: bool, force: bool, _token: int):
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
              "error": None}

    uni, px = None, None
    try:
        uni, px = _read_cache(download_start)
    except Exception as exc:  # noqa: BLE001
        if not online:
            raise
        status["error"] = f"no usable cache ({exc})"

    behind = (sessions_behind(px.index.max()) if px is not None else 99)
    if online and (force or behind >= 1 or uni is None):
        try:
            tickers = load_universe(fetch=True, warn=False)
            uni = load_universe_prices(tickers, start=download_start,
                                       refresh=True, verbose=False)
            px = load_prices(["QQQ", "VEU", "BOXX"], start=download_start,
                             refresh=True, offline=False)
            status["downloaded"] = True
            status["error"] = None
        except Exception as exc:  # noqa: BLE001
            status["error"] = str(exc)
            if uni is None or px is None:
                raise

    uni = uni.reindex(px.index).ffill()
    uni = uni.loc[:, uni.notna().sum() >= 260]
    return uni, px, status


@st.cache_data(show_spinner="Building selections…")
def build_selections(_uni: pd.DataFrame, _safe: pd.Series, n_hold: int,
                     exit_rank: int, n_watch: int) -> Dict[str, pd.DataFrame]:
    """Daily holdings for each strategy, as date x (tickers, buys, sells)."""
    from qbs.breakout import finviz_watchlists
    from qbs.config import MomentumParams

    out: Dict[str, pd.DataFrame] = {}

    mom = cross_sectional_momentum(
        _uni, _safe, MomentumParams(n_hold=n_hold, exit_rank=exit_rank))
    fin = finviz_momentum_screen(_uni, _safe, FinvizScreenParams(n_hold=n_hold))

    for key, sig in (("momentum", mom), ("finviz", fin)):
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

    # The breakout strategy selects weekly, so its "daily" view is the
    # watchlist standing for that week -- not a held book.
    wl = finviz_watchlists(_uni, _safe, n_watch=n_watch)
    rows, current = [], ""
    for d in _uni.index:
        if d in wl:
            current = ", ".join(wl[d])
        rows.append({"date": d, "holdings": current,
                     "n": len(current.split(", ")) if current else 0,
                     "buys": "", "sells": ""})
    out["breakout"] = pd.DataFrame(rows).set_index("date")
    return out


@st.cache_data(show_spinner="Computing breadth…")
def build_breadth(_uni: pd.DataFrame, _qqq: pd.Series, note: str,
                  _volumes: Optional[pd.DataFrame] = None):
    return daily_breadth(_uni, qqq=_qqq, volumes=_volumes, universe_note=note)


@st.cache_data(show_spinner="Fetching the US universe from Finviz…")
def load_us_market(download_start: str, online: bool, force: bool, _token: int):
    """The broad US universe for the breadth tab: closes, volumes, sectors.

    Returns `(closes, volumes, sectors, note, error)`. `error` is not fatal --
    the caller falls back to the cached index universe and says so on screen,
    because breadth computed over a truncated sample is wrong in a way that
    looks entirely plausible.
    """
    filters = UniverseFilters()
    uni, uni_err = fetch_us_universe(filters, refresh=force, offline=not online,
                                     verbose=False)
    if uni is None or uni.empty:
        return None, None, {}, filters.label, uni_err or "unknown failure"

    tickers = uni["Ticker"].tolist()
    closes, volumes, bars_err = load_universe_bars(
        tickers, start=download_start, refresh=force, offline=not online,
        verbose=False)
    if closes is None or closes.empty:
        return None, None, {}, filters.label, (
            f"Finviz listed {len(tickers)} tickers but no prices loaded — {bars_err}")

    # A partial fetch is usable; a silent one is not. Carry the warning up.
    warn = "; ".join(x for x in (uni_err, bars_err) if x) or None
    return closes, volumes, sector_map(uni), filters.label, warn


EMA_SPANS = (10, 20, 50, 200)
EMA_COLOURS = {"EMA 10": "#eb6834", "EMA 20": "#eda100",
               "EMA 50": "#2a78d6", "EMA 200": "#8a63d2"}


CANDLE_UP, CANDLE_DN = "#1b7a4b", "#b02525"


@st.cache_data(show_spinner=False)
def ohlc_for(ticker: str, download_start: str, online: bool, _token: int):
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


def fmt(v, spec="{:.1f}", dash="—"):
    return dash if v is None or (isinstance(v, float) and pd.isna(v)) else spec.format(v)


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
n_watch = st.sidebar.number_input("Breakout watchlist size", 5, 50, 20)

# Default to 0, not -1. With -1 the very first page load has 0 > -1, so the
# app force-refreshed on EVERY start -- re-downloading the whole universe
# before it had shown anything. "Refresh now" is the only thing that should
# set this.
force = st.session_state["refresh_token"] > st.session_state.get("applied_token", 0)
try:
    uni, px, data_status = load_data(download_start, bool(online), bool(force),
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
st.sidebar.caption(f"Picks universe: {UNIVERSE_NOTE}")
st.sidebar.caption("The market tab has its own universe selector.")
st.sidebar.caption(f"Data through {LAST_BAR:%Y-%m-%d} ({SRC})")
st.sidebar.caption({"ok": "✅ current", "info": "🕒 1 session behind",
                    "warn": f"⚠️ {N_BEHIND} sessions behind"}[FRESH_LEVEL])

selections = build_selections(uni, px["BOXX"], int(n_hold), int(exit_rank), int(n_watch))
breadth = build_breadth(uni, px["QQQ"], UNIVERSE_NOTE)

tab_picks, tab_market, tab_analyst = st.tabs(
    ["📋 Daily picks", "📊 Market overview", "🤖 Analyst"])


# ==========================================================================
# Tab 1 -- daily selections
# ==========================================================================

with tab_picks:
    freshness_banner()
    st.subheader("Daily picks")
    st.caption(f"Three selection strategies · universe {UNIVERSE_NOTE} · "
               f"data through {LAST_BAR:%Y-%m-%d} ({SRC})")

    dates = [d for d in selections["momentum"].index
             if d >= pd.Timestamp(cfg.backtest_start)]
    asof = st.select_slider("Date", options=dates, value=dates[-1],
                            format_func=lambda d: f"{d:%Y-%m-%d}")

    cols = st.columns([1, 1, 1, 2.6])
    picks: Dict[str, set] = {}
    for col, key in zip(cols[:3], ("momentum", "finviz", "breakout")):
        frame = selections[key]
        row = frame.loc[asof] if asof in frame.index else None
        raw = row["holdings"] if row is not None and row["holdings"] else ""
        names = [t for t in raw.split(", ") if t]
        picks[key] = set(names)
        with col:
            st.markdown(f"**{STRATEGY_LABELS[key]}**")
            st.metric("Watchlist size" if key == "breakout" else "Names held",
                      len(names))
            if key == "breakout":
                st.caption("A watchlist, not a book — a name is only bought "
                           "once a breakout confirms.")
            if names:
                st.dataframe(pd.DataFrame({"Ticker": names}), hide_index=True,
                             width="stretch",
                             height=min(250, 38 + 35 * len(names)))
            else:
                st.info("Nothing held — fully in cash.")
            if row is not None and row["buys"]:
                st.caption(f"🟢 Bought: {row['buys']}")
            if row is not None and row["sells"]:
                st.caption(f"🔴 Sold: {row['sells']}")

    # ---- right-hand panel: price, EMAs and the levels that matter ---------
    with cols[3]:
        st.markdown("**Price & levels**")
        universe_names = list(uni.columns)
        picked = sorted(set().union(*picks.values()))
        options = picked + [t for t in universe_names if t not in picked]
        if not options:
            st.info("No name to chart.")
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            ticker = c1.selectbox(
                "Ticker", options, index=0, key="chart_ticker",
                help="Today's picks come first, then the rest of the universe.")
            months = c2.selectbox("Window", [3, 6, 12, 24], index=2,
                                  format_func=lambda m: f"{m}m", key="chart_win")
            n_lvl = c3.number_input("Levels", 0, 30, 8, key="chart_levels",
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
                ohlc = ohlc_for(ticker, download_start, bool(online),
                                st.session_state["refresh_token"])
                bars = None
                if ohlc is not None and not ohlc.empty:
                    win = ohlc.loc[:asof].tail(int(months * 21))
                    if len(win) > 2 and {"Open", "High", "Low"} <= set(win.columns):
                        bars = win.reset_index()
                        bars.columns = [str(c).lower() for c in bars.columns]

                yscale = alt.Scale(zero=False, nice=True)
                if bars is not None:
                    body_colour = alt.condition(
                        "datum.open <= datum.close",
                        alt.value(CANDLE_UP), alt.value(CANDLE_DN))
                    cbase = alt.Chart(bars).encode(
                        x=alt.X("date:T", title=None), color=body_colour,
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
                        x=alt.X("date:T", title=None),
                        y=alt.Y("close:Q", title=None, scale=yscale),
                        tooltip=[alt.Tooltip("date:T", title="Date"),
                                 alt.Tooltip("close:Q", title="Close", format=".2f")])
                emas = alt.Chart(ema_long).mark_line(size=1.1, opacity=0.9).encode(
                    x="date:T",
                    y=alt.Y("value:Q", scale=alt.Scale(zero=False, nice=True)),
                    color=alt.Color("ema:N", title=None, scale=alt.Scale(
                        domain=list(EMA_COLOURS), range=list(EMA_COLOURS.values())),
                        legend=alt.Legend(orient="top", direction="horizontal")),
                    tooltip=[alt.Tooltip("ema:N", title="Line"),
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
                        "already cleared every level its chart shows, so a "
                        "breakout entry has nothing to fire on. This is the state "
                        "52% of Finviz picks are in — see the funnel notebook.",
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
                        st.caption(note)

    common = picks["momentum"] & picks["finviz"]
    st.markdown(
        f"**Momentum ∩ Finviz:** {', '.join(sorted(common)) if common else 'no overlap'} "
        f"({len(common)}/{len(picks['momentum']) or '—'})"
    )

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
    mkt_note, mkt_err = "", None
    if use_us:
        mkt_closes, mkt_vols, mkt_sectors, mkt_note, mkt_err = load_us_market(
            download_start, bool(online), bool(force),
            st.session_state["refresh_token"])

    if use_us and mkt_closes is not None:
        m_uni = mkt_closes
        m_vols = mkt_vols
        universe_label = f"{m_uni.shape[1]} US names · {mkt_note}"
        breadth_m = build_breadth(m_uni, px["QQQ"], universe_label, m_vols)
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
    k[0].metric("Up 4% / Down 4%", f"{int(last['up4'])} / {int(last['dn4'])}")
    k[0].caption(fmt(ratio, "up:down ratio {:.2f}"))

    d_fast = (last["pct_above_fast"] - prev["pct_above_fast"]) if prev is not None else np.nan
    k[1].metric("% above 20-day", fmt(last["pct_above_fast"]), fmt(d_fast, "{:+.1f}"))
    k[1].caption("vs prior session")

    d_slow = (last["pct_above_slow"] - prev["pct_above_slow"]) if prev is not None else np.nan
    k[2].metric("% above 50-day", fmt(last["pct_above_slow"]), fmt(d_slow, "{:+.1f}"))
    k[2].caption("vs prior session")

    k[3].metric("SPY vs 50D EMA", "—")
    k[3].caption("SPY is not cached in this package")

    k[4].metric("QQQ vs 50D EMA", fmt(last["qqq_atr"], "{:+.2f}"))
    k[4].caption("in units of 14-day ATR")

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
        pd.DataFrame({"date": pulse.index, "value": pulse["up4"].astype(int),
                      "cls": [PULSE_LABELS[pulse_class(v, "up")] for v in pulse["up4"]]}),
        pd.DataFrame({"date": pulse.index, "value": -pulse["dn4"].astype(int),
                      "cls": [PULSE_LABELS[pulse_class(v, "down")] for v in pulse["dn4"]]}),
    ])
    chart = alt.Chart(bars).mark_bar().encode(
        x=alt.X("date:T", title=None),
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
    st.caption("The ATR column is (close − 50-day EMA) ÷ 14-day ATR — how many ATRs "
               "the index sits from its own 50-day line.")
    n_rows = st.slider("Sessions shown", 10, 250, 20, key="table_days")
    view = tbl.tail(n_rows).iloc[::-1]
    disp = pd.DataFrame({
        "Date": view.index.strftime("%Y-%m-%d"),
        "Up 4%": view["up4"].astype(int),
        "Dn 4%": view["dn4"].astype(int),
        "% > 20D": view["pct_above_fast"],
        "% > 50D": view["pct_above_slow"],
        "QQQ ATR": view["qqq_atr"],
        "MLI %": view["mli_pct"],
        "MLI adv%": view["mli_up_pct"],
        "MLI N": view["mli_n"].astype(int),
        "Sample": view["n_stocks"].astype(int),
    })

    def _style(col: pd.Series):
        name = col.name
        if name == "Up 4%":
            return [f"background-color: {UP_STRONG if pulse_class(v,'up')=='up_strong' else UP}"
                    for v in col]
        if name == "Dn 4%":
            return [f"background-color: {DN_STRONG if pulse_class(v,'down')=='down_strong' else DN}"
                    for v in col]
        if name in ("% > 20D", "% > 50D"):
            which_ma = "fast" if "20" in name else "slow"
            return [f"background-color: {CELL[ma_class(v, which_ma)]}" for v in col]
        if name == "QQQ ATR":
            return [f"background-color: {CELL[atr_class(v)]}" for v in col]
        return ["" for _ in col]

    styled = (disp.style
              .apply(_style, axis=0)
              .format({"% > 20D": "{:.1f}", "% > 50D": "{:.1f}", "QQQ ATR": "{:+.2f}",
                       "MLI %": "{:+.2f}", "MLI adv%": "{:.1f}"}, na_rep="—"))
    st.dataframe(styled, hide_index=True, width="stretch",
                 height=min(720, 45 + 35 * len(disp)))
    st.caption(
        f"Green = names up 4% (dark ≥ {BreadthParams().pulse_strong}) · "
        f"red = down 4% · moving-average columns shade red below 10/20% and "
        "green above 90/80% · ATR shades red beyond ±5."
    )

    # ---- momentum leaders -------------------------------------------------
    st.divider()
    st.markdown("#### Momentum leaders")
    p = BreadthParams()
    note = (f"Rules: close ≥ ${p.leader_min_price:.0f} · "
            f"quarterly gain ≥ {p.leader_min_quarter_return:.0%}")
    if breadth.has_volume:
        note += f" · turnover ≥ ${p.leader_min_turnover/1e6:.0f}M/day ✅"
    else:
        note += (f" · ⚠️ the turnover test (≥ ${p.leader_min_turnover/1e6:.0f}M/day) "
                 "cannot run without volume, so this leader count is an over-estimate")
    st.caption(note)

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
    st.altair_chart(
        alt.Chart(trend.tail(250)).mark_line(color="#2a4b8d").encode(
            x=alt.X("date:T", title=None),
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

with tab_analyst:
    freshness_banner()
    st.subheader("Ask the analyst")

    from qbs.agent.analyst import DEFAULT_MODEL, analyse, check_requirements
    from qbs.agent.env import load_env
    from qbs.agent.evidence import Book
    from qbs.agent.news import available_backends

    env_load = load_env()
    blocker = check_requirements()
    backends = available_backends()

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Analyst**")
        model_name = st.text_input("Gemini model", DEFAULT_MODEL,
                                   help="Model names move faster than this app. "
                                        "Override here or set QBS_GEMINI_MODEL.")
        allow_web = st.checkbox("Allow web search", value=bool(backends),
                                disabled=not backends,
                                help=("Search backends found: "
                                      + (", ".join(backends) or "none — "
                                         "pip install ddgs")))
        live_fundamentals = st.checkbox(
            "Fetch fundamentals live", value=True,
            help="Off reads only what is already cached in data/fundamentals/.")

    if blocker:
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

    EXAMPLES = [
        "Why is the momentum book holding names the Finviz screen rejects?",
        "Profile the top momentum pick and say what would take it out of the book.",
        "Is the market broad or narrow right now, and what does that imply "
        "for a breakout strategy?",
        "Pull fundamentals and recent news for the strongest pick, and say "
        "where they disagree with the price signal.",
    ]
    picked = st.selectbox("Start from an example, or write your own", 
                          ["— write my own —"] + EXAMPLES)
    default_q = "" if picked.startswith("—") else picked
    question = st.text_area("Question", value=default_q, height=90,
                            placeholder="Ask about the picks, a name, breadth, "
                                        "or the backtest results…")

    ask = st.button("Analyse", type="primary", disabled=bool(blocker) or not question.strip())

    if ask:
        # The agent gets the frames this app already loaded rather than
        # re-reading the cache: a three-tool answer would otherwise spend a
        # minute rebuilding a universe that is sitting in memory.
        book = Book(universe=uni, prices=px, cfg=cfg, note=UNIVERSE_NOTE)
        with st.spinner(f"Asking {model_name}…"):
            answer = analyse(
                question.strip(), model=model_name.strip() or None,
                book=book, n_hold=int(n_hold), allow_web=bool(allow_web),
                offline_fundamentals=not live_fundamentals)

        if answer.error:
            st.error(f"**The analyst could not answer.** {answer.text}", icon="🚫")
        else:
            st.markdown(answer.text)

        if answer.tool_calls:
            st.markdown("---")
            st.markdown(f"###### Evidence — {len(answer.tool_calls)} tool "
                        f"call{'s' if len(answer.tool_calls) != 1 else ''}")
            for i, call in enumerate(answer.tool_calls, 1):
                args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                with st.expander(f"{i}. `{call['name']}({args})`"):
                    st.code(call["result"] or "(no output)", language="text")
            st.caption(
                "Every number in the answer above should appear in one of "
                "these. One that does not is a fabrication, and worth telling "
                "the model about in a follow-up.")
        elif ask and not answer.error:
            st.warning(
                "The analyst answered without calling a single tool, which "
                "means nothing in that answer is sourced from your data. "
                "Treat it as opinion and ask again more specifically.",
                icon="⚠️")
