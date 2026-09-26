"""The dashboard's Market overview tab, as text an analyst can read.

`evidence.breadth_report` measures participation over the ranking universe
-- ~100 Nasdaq-100 names. The Market overview tab measures the broad US
universe (~2,400 names from the Finviz screener) when it can, and carries
more than breadth: the SPY and QQQ stretch in ATR units, the momentum-leader
group, the bear-market checklist, and where the leadership sits by sector.
`Market` holds those inputs, and the reports below read them the way the tab
does -- the same functions from `qbs.breadth`, over the same frames -- so a
number the analyst quotes is a number on that tab.

When the dashboard hands in what it already loaded, nothing is recomputed.
Anything missing is computed once, lazily, and cached on the object: tools
get called several times in one run and a 2,400-name breadth table is not
free.

Pure pandas. Nothing here imports LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..breadth import (BreadthParams, BreadthResult, atr_class, bear_checklist,
                       daily_breadth, leader_mask, ma_class, momentum_label,
                       pulse_cell, sector_breakdown, sector_leaders)
from ..data import sessions_behind
from .evidence import Book

# The checklist's automatic rows, in short. The dashboard shows the long
# wording; these are the same questions, keyed by `bear_checklist`'s keys.
CHECKLIST_QUESTIONS = {
    "oversold": "Has % above the 40-day stayed under 20% for over 10 "
                "sessions (oversold turned into a trend)?",
    "final_high": "On an index new high, are under 30% of momentum leaders "
                  "above their 50-day (a 'final new high')?",
    "divergence": "Is the index rising while new lows outnumber new highs?",
    "mli": "Is the number of momentum leaders still falling, with no floor?",
}
MANUAL_ROWS = ("FedWatch rate-hike expectations rising", "a large share of "
               "your own positions hitting stops (weighted double)")


@dataclass
class Market:
    """The Market overview tab's inputs, and what it derived from them."""
    closes: pd.DataFrame
    volumes: Optional[pd.DataFrame] = None
    sectors: Dict[str, str] = field(default_factory=dict)
    note: str = ""
    qqq: Optional[pd.Series] = None
    spy: Optional[pd.Series] = None
    qqq_ohlc: Optional[pd.DataFrame] = None
    spy_ohlc: Optional[pd.DataFrame] = None
    # True when this is the Nasdaq-100 fallback, not the US market.
    index_fallback: bool = False
    # Already computed by the caller (the dashboard) -- used as given.
    breadth: Optional[BreadthResult] = None
    checklist: Optional[List[Dict]] = None
    _leaders: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def asof(self) -> pd.Timestamp:
        return self.closes.index[-1]

    def get_breadth(self) -> BreadthResult:
        if self.breadth is None:
            self.breadth = daily_breadth(
                self.closes, qqq=self.qqq, spy=self.spy, volumes=self.volumes,
                universe_note=self.note, qqq_ohlc=self.qqq_ohlc,
                spy_ohlc=self.spy_ohlc)
        return self.breadth

    def get_checklist(self) -> List[Dict]:
        if self.checklist is None:
            self.checklist = (bear_checklist(self.closes, self.qqq,
                                             volumes=self.volumes)
                              if self.qqq is not None else [])
        return self.checklist

    def leaders(self) -> pd.DataFrame:
        if self._leaders is None:
            self._leaders = leader_mask(self.closes, self.volumes)
        return self._leaders

    def header(self) -> str:
        behind = sessions_behind(self.asof)
        age = ("current" if behind <= 0 else
               f"{behind} trading session{'s' if behind != 1 else ''} behind")
        what = ("NASDAQ-100 FALLBACK — an index, not the market"
                if self.index_fallback else "US market universe")
        return (f"{what}: {self.closes.shape[1]:,} names, last bar "
                f"{self.asof:%Y-%m-%d} ({age})"
                + (f"; {self.note}" if self.note else ""))


def market_from_book(book: Book, spy: Optional[pd.Series] = None) -> Market:
    """The fallback the tab itself uses when the US universe is unavailable."""
    return Market(closes=book.universe, qqq=book.prices.get("QQQ"), spy=spy,
                  note=book.note, index_fallback=True)


_BAND = {"dark_green": "dark green (strongly bullish)",
         "light_green": "light green (bullish)",
         "light_red": "light red (bearish)",
         "dark_red": "dark red (strongly bearish)", "none": "n/a"}


def _f(v, spec: str = "{:.1f}") -> str:
    return "n/a" if v is None or pd.isna(v) else spec.format(v)


# --------------------------------------------------------------------------
# The tab's top half: KPIs, the daily monitor, the checklist
# --------------------------------------------------------------------------

def overview_report(market: Market, sessions: int = 10,
                    p: Optional[BreadthParams] = None) -> str:
    """The Market overview tab's readings, in the tab's order."""
    p = p or BreadthParams()
    res = market.get_breadth()
    tbl = res.table
    if tbl is None or tbl.empty:
        return "Market breadth could not be computed -- not enough history."
    last = tbl.iloc[-1]
    prev = tbl.iloc[-2] if len(tbl) > 1 else None
    sessions = max(1, min(int(sessions), 60))

    lines = ["MARKET OVERVIEW — " + market.header(),
             f"Sample on the last row: {int(last['n_stocks']):,} names", ""]
    if market.index_fallback:
        lines += ["WARNING: these readings are over ~100 Nasdaq-100 mega-caps, "
                  "not the US market. The 4% counts and their colour bands are "
                  "calibrated for ~2,400 names and are not comparable.", ""]

    ratio = last["up4"] / last["dn4"] if last["dn4"] else np.nan

    def _d(col):
        return (last[col] - prev[col]) if prev is not None else np.nan

    lines.append("[Today — the tab's headline metrics]")
    lines.append(f"  Up 4% / Down 4%:   {_f(last['up4'], '{:.0f}')} / "
                 f"{_f(last['dn4'], '{:.0f}')}  (ratio {_f(ratio, '{:.2f}')}; "
                 "the monitor's colour bands, where green is bullish on BOTH "
                 f"columns: up {_BAND[pulse_cell(last['up4'], 'up', p)]}, "
                 f"down {_BAND[pulse_cell(last['dn4'], 'down', p)]})")
    lines.append(f"  % above 20-day:    {_f(last['pct_above_fast'])}% "
                 f"({_f(_d('pct_above_fast'), '{:+.1f}')} vs prior session; "
                 f"{ma_class(last['pct_above_fast'], 'fast', p)})")
    lines.append(f"  % above 50-day:    {_f(last['pct_above_slow'])}% "
                 f"({_f(_d('pct_above_slow'), '{:+.1f}')} vs prior session; "
                 f"{ma_class(last['pct_above_slow'], 'slow', p)})")
    for col, name, ohlc in (("spy_atr", "SPY", market.spy_ohlc),
                            ("qqq_atr", "QQQ", market.qqq_ohlc)):
        v = last.get(col, np.nan)
        if pd.isna(v):
            lines.append(f"  {name} vs 50D EMA:    n/a (no {name} series in "
                         "this run)")
            continue
        lines.append(f"  {name} vs 50D EMA:    {v:+.2f} ATR(14) "
                     f"({atr_class(v, p)}; beyond ±{p.atr_stretched:.0f} is "
                     "stretched)"
                     + ("" if ohlc is not None else
                        " -- close-only range, so this reads high"))
    lines.append(f"  Momentum leaders:  {int(last['mli_n'])} names "
                 f"({100.0 * last['mli_n'] / last['n_stocks']:.1f}% of the "
                 f"sample; {_f(_d('mli_n'), '{:+.0f}')} vs prior session), "
                 f"average move {_f(last['mli_pct'], '{:+.2f}')}%, "
                 f"{_f(last['mli_up_pct'])}% advancing")
    share = 100.0 * tbl["mli_n"] / tbl["n_stocks"]
    trend = []
    for n in (5, 20, 60):
        if len(share) > n:
            trend.append(f"{n} sessions ago {share.iloc[-1 - n]:.1f}%")
    if trend:
        lines.append(f"  Leader share now {share.iloc[-1]:.1f}% vs "
                     + ", ".join(trend))
    lines.append(
        "  Leader rule: close > $"
        f"{p.leader_min_price:.0f}, up > {p.leader_min_quarter_return:.0%} on "
        f"the quarter"
        + (f", > {p.leader_min_volume / 1e3:,.0f}k shares/day"
           if res.has_volume else
           " -- the volume leg could not run, so the count is an over-estimate"))

    view = tbl.tail(sessions).iloc[::-1]
    lines += ["", f"[Daily monitor — last {len(view)} sessions, newest first]",
              "  date        up4   dn4  %>20D  %>50D  SPY ATR  QQQ ATR  "
              "MLI N  MLI %  MLI adv%"]
    for d, r in view.iterrows():
        lines.append(
            f"  {d:%Y-%m-%d} {_f(r['up4'], '{:>5.0f}')} {_f(r['dn4'], '{:>5.0f}')} "
            f"{_f(r['pct_above_fast'], '{:>6.1f}')} "
            f"{_f(r['pct_above_slow'], '{:>6.1f}')} "
            f"{_f(r.get('spy_atr'), '{:+.2f}'):>8} "
            f"{_f(r.get('qqq_atr'), '{:+.2f}'):>8} "
            f"{int(r['mli_n']):>6} {_f(r['mli_pct'], '{:>+6.2f}')} "
            f"{_f(r['mli_up_pct'], '{:>8.1f}')}")
    if res.gaps:
        lines.append(f"  ({len(res.gaps)} session(s) left out because most "
                     "names had no close that day.)")

    checks = market.get_checklist()
    lines += ["", "[Bear-market checklist — the four rows the data answers; "
              "Yes is the BEARISH reading]"]
    if not checks:
        lines.append("  Not available: no index series to answer it against.")
    yes = 0
    for c in checks:
        ans = c.get("answer")
        yes += bool(ans)
        mark = "n/a" if ans is None else ("YES" if ans else "no")
        lines.append(f"  [{mark:>3}] {CHECKLIST_QUESTIONS.get(c['key'], c['key'])}"
                     f" -- {c.get('reading', '')}")
    if checks:
        lines.append(f"  {yes} of {len(checks)} automatic rows read bearish. "
                     f"Not in the data, so unanswered here: "
                     f"{'; '.join(MANUAL_ROWS)}.")

    lines += ["",
              "Breadth is participation, not direction: a rising index on "
              "narrowing breadth and one on broadening breadth look the same "
              "on a price chart and are not the same market."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The tab's bottom half: sectors
# --------------------------------------------------------------------------

def sector_report(market: Market, per_sector: int = 3, top: int = 8) -> str:
    """Where the momentum leadership sits, and who leads each sector."""
    if not market.sectors:
        return ("No sector map in this run: the Nasdaq-100 cache carries no "
                "sector classification, and the tab leaves this panel blank "
                "rather than inventing one. The US universe (Finviz) supplies "
                "it.")
    sect = sector_breakdown(market.closes, market.sectors, asof=market.asof,
                            volumes=market.volumes)
    if sect.empty:
        return (f"No momentum leaders on {market.asof:%Y-%m-%d}, so there is "
                "no sector breakdown.")
    lines = ["SECTOR LEADERSHIP — " + market.header(), "",
             "[Composition — Excess pp is share of leaders minus the "
             "sector's own share of the universe; that is the column that "
             "carries information]",
             f"  {'sector':<24} {'leaders':>7} {'share%':>7} {'pool%':>6} "
             f"{'penetr%':>8} {'excess pp':>9}"]
    for r in sect.itertuples():
        lines.append(f"  {str(r.sector)[:24]:<24} {r.n:>7} {r.share_pct:>7.1f} "
                     f"{r.pool_pct:>6.1f} {_f(r.penetration, '{:>8.1f}')} "
                     f"{r.excess_pp:>+9.1f}")
    lines.append(f"  Top-3 concentration {sect['share_pct'].head(3).sum():.1f}% "
                 f"of all leaders across {sect['sector'].nunique()} sectors")

    lead = sector_leaders(market.closes, market.sectors, asof=market.asof,
                          volumes=market.volumes, per_sector=int(per_sector))
    if not lead.empty:
        lines += ["", f"[Strongest {per_sector} leaders per sector by "
                  f"{momentum_label()} momentum, largest sectors first]"]
        for sec, g in list(lead.groupby("sector", sort=False))[:max(1, int(top))]:
            names = ", ".join(f"{r.symbol} {r.score:+.0%}" for r in g.itertuples())
            lines.append(f"  {str(sec)[:24]:<24} ({int(g['n_sector'].iloc[0])}): "
                         f"{names}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# One stock against the market it trades in
# --------------------------------------------------------------------------

def stock_context_report(market: Market, ticker: str,
                         book: Optional[Book] = None) -> str:
    """One name placed in the overview's universe: sector, leader status,
    and where its recent returns rank against the market and its sector."""
    ticker = ticker.upper().strip()
    px = market.closes
    s = px[ticker] if ticker in px.columns else None
    if (s is None or s.dropna().empty) and book is not None:
        s = book.closes(ticker)
        joined = s is not None
    else:
        joined = False
    if s is None or s.dropna().empty:
        return (f"{ticker} is not in the market universe of this run "
                f"({px.shape[1]:,} names) and no prices are held for it.")
    if joined:
        px = px.join(s.rename(ticker).reindex(px.index).ffill(), how="left")

    asof = market.asof
    lines = [f"{ticker} IN THE MARKET — " + market.header()]
    if joined:
        lines.append(f"{ticker} is not in this universe; it is joined in for "
                     "this report only, so its percentiles say where it WOULD "
                     "place.")
    sector = market.sectors.get(ticker)
    lines.append(f"Sector: {sector or 'not classified in this run'}")

    in_sec = ([t for t, v in market.sectors.items() if v == sector and t in px]
              if sector else [])
    lines += ["", "[Return percentile, 1-99, against every name in the "
              "universe" + (" and within its sector" if sector else "")
              + " on the same sessions]"]
    for label, n in (("1 day", 1), ("1 week", 5), ("1 month", 21),
                     ("3 months", 63), ("6 months", 126)):
        if len(px) <= n:
            continue
        r = (px.iloc[-1] / px.iloc[-1 - n] - 1.0).dropna()
        if ticker not in r:
            continue
        v = float(r[ticker])
        pct = float((r < v).mean() * 98 + 1)
        row = (f"  {label:<9} {v:+7.1%}  market pct {pct:3.0f}  "
               f"market median {r.median():+.1%}")
        if len(in_sec) >= 5:
            rs = r.reindex(in_sec).dropna()
            spct = float((rs < v).mean() * 98 + 1)
            row += (f"  |  sector pct {spct:3.0f} ({len(rs)} names)  "
                    f"sector median {rs.median():+.1%}")
        lines.append(row)

    p = BreadthParams()
    if ticker in market.closes.columns:
        is_leader = bool(market.leaders().loc[asof].get(ticker, False))
    else:
        is_leader = bool(leader_mask(px[[ticker]], None, p).loc[asof, ticker])
    q = (px[ticker].iloc[-1] / px[ticker].iloc[-1 - p.leader_quarter_days] - 1.0
         if len(px) > p.leader_quarter_days else np.nan)
    lines += ["", f"Momentum leader today: {'YES' if is_leader else 'no'} "
              f"(quarterly return {_f(q, '{:+.1%}')} against the "
              f"{p.leader_min_quarter_return:.0%} bar, price "
              f"{px[ticker].iloc[-1]:,.2f} against ${p.leader_min_price:.0f})"]
    if sector and market.sectors:
        sect = sector_breakdown(market.closes, market.sectors, asof=asof,
                                volumes=market.volumes)
        row = sect[sect["sector"] == sector]
        if row.empty:
            lines.append(f"{sector} has no momentum leaders today.")
        else:
            r = row.iloc[0]
            rank = int(row.index[0]) + 1
            lines.append(f"{sector}: {int(r['n'])} leaders, {r['share_pct']:.1f}% "
                         f"of all leaders vs {r['pool_pct']:.1f}% of the "
                         f"universe ({r['excess_pp']:+.1f}pp) -- "
                         f"#{rank} of {len(sect)} sectors by leader count")
    return "\n".join(lines)
