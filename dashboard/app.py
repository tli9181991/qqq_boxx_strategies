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
from typing import Dict

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.breadth import BreadthParams, atr_class, daily_breadth, ma_class, pulse_class
from qbs.breakout import closes_to_bars, levels_in_view, sr_levels
from qbs.config import BreakoutParams, Config, FinvizScreenParams
from qbs.data import freshness_note, load_prices, sessions_behind
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
def build_breadth(_uni: pd.DataFrame, _qqq: pd.Series, note: str):
    return daily_breadth(_uni, qqq=_qqq, universe_note=note)


EMA_SPANS = (10, 20, 50, 200)
EMA_COLOURS = {"EMA 10": "#eb6834", "EMA 20": "#eda100",
               "EMA 50": "#2a78d6", "EMA 200": "#8a63d2"}


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

force = st.session_state["refresh_token"] > st.session_state.get("applied_token", -1)
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
st.sidebar.caption(f"Universe: {UNIVERSE_NOTE}")
st.sidebar.caption(f"Data through {LAST_BAR:%Y-%m-%d} ({SRC})")
st.sidebar.caption({"ok": "✅ current", "info": "🕒 1 session behind",
                    "warn": f"⚠️ {N_BEHIND} sessions behind"}[FRESH_LEVEL])

selections = build_selections(uni, px["BOXX"], int(n_hold), int(exit_rank), int(n_watch))
breadth = build_breadth(uni, px["QQQ"], UNIVERSE_NOTE)

tab_picks, tab_market = st.tabs(["📋 Daily picks", "📊 Market overview"])


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
                y = alt.Y("close:Q", title=None,
                          scale=alt.Scale(zero=False, nice=True))
                line = alt.Chart(price).mark_line(color="#0b0b0b", size=1.7).encode(
                    x=alt.X("date:T", title=None), y=y,
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

    st.warning(
        f"**This tab samples {uni.shape[1]} Nasdaq-100 constituents, not the US market.** "
        "The reference version samples ~2,400 US common stocks and ADRs. The arithmetic "
        "is the same but the readings are not comparable: a count of names up 4% out of "
        "99 mega-caps measures something different from the same count out of 2,432 — "
        "it is not a smaller version of the same number. Real readings need a "
        "full-market feed.",
        icon="⚠️",
    )

    tbl = breadth.table
    tbl = tbl.loc[tbl.index >= pd.Timestamp(cfg.backtest_start)]
    if tbl.empty:
        st.error("Breadth table is empty — not enough history.")
        st.stop()
    last = tbl.iloc[-1]
    prev = tbl.iloc[-2] if len(tbl) > 1 else None

    st.caption(
        f"Close {tbl.index[-1]:%Y-%m-%d} · universe {UNIVERSE_NOTE} · "
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
    if not breadth.has_volume:
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

    st.info(
        "**The sector-composition table is not shown.** It needs a `ticker → sector` "
        "map to compute share, pool weight, penetration and excess pp. This package "
        "stores no sector classification, so the panel is left blank rather than "
        "filled with a guess — `qbs.breadth.sector_breakdown()` is written and will "
        "produce the table as soon as you pass it a sector map.",
        icon="ℹ️")
