"""The lab's own results, rendered as text an analyst can read.

Every function here answers one question with numbers this package computed,
and attaches the caveat that decides how far the number can be pushed. That
pairing is the whole design. An LLM handed "expectancy +1.37R" will call it
an expectation; handed "expectancy +1.37R over 11 closed trades, of which the
top 3 are 39% of all R" it has no room to.

So each report carries its own sample size, its window, and the specific way
it can mislead. None of that is decoration -- it is the part that stops a
fluent summary from being a wrong one.

Nothing in this module talks to an LLM, and nothing imports langchain. It is
the data layer under `tools.py`, and it is directly useful on its own: every
report prints fine in a terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..breadth import (BreadthParams, atr_class, daily_breadth, ma_class,
                       momentum_label, momentum_profile)
from ..config import Config, FinvizScreenParams, MomentumParams, SAFE_ASSET
from ..data import load_prices, sessions_behind
from ..screens import finviz_momentum_screen
from ..strategies import cross_sectional_momentum
from ..universe import load_universe, load_universe_prices

MIN_TRADES_TO_TRUST = 30       # below this, trade statistics are anecdote


@dataclass
class Book:
    """Everything the reports need, loaded once.

    Tools get called repeatedly inside one agent run, and reloading a 1500-row
    universe per call would turn a three-tool answer into a minute of waiting.
    """
    universe: pd.DataFrame                  # ranking universe closes
    prices: pd.DataFrame                    # QQQ / VEU / BOXX
    cfg: Config
    note: str = ""

    @property
    def asof(self) -> pd.Timestamp:
        return self.universe.index[-1]

    @property
    def safe(self) -> pd.Series:
        return self.prices[SAFE_ASSET]

    @property
    def stale_sessions(self) -> int:
        return sessions_behind(self.asof)

    def header(self) -> str:
        behind = self.stale_sessions
        age = ("current" if behind <= 0 else
               f"{behind} trading session{'s' if behind != 1 else ''} behind")
        return (f"Data as of {self.asof:%Y-%m-%d} ({age}), "
                f"{self.universe.shape[1]} names, "
                f"{len(self.universe)} sessions{'; ' + self.note if self.note else ''}")


def load_book(offline: bool = True, download_start: Optional[str] = None,
              cfg: Optional[Config] = None) -> Book:
    """Load the cached universe and core ETFs. Offline by default.

    Offline is the default deliberately: an agent run that silently kicks off
    a multi-hundred-ticker download because a tool was called is a surprise
    nobody asked for. Refresh the cache with `run_backtest.py` or the
    dashboard, then analyse it.
    """
    cfg = cfg or Config()
    start = download_start or cfg.download_start
    tickers = load_universe(fetch=not offline, warn=False)
    uni = load_universe_prices(tickers, start=start, verbose=False,
                               refresh=not offline)
    px = load_prices(["QQQ", "VEU", "BOXX"], start=start, offline=offline)
    uni = uni.reindex(px.index).ffill()
    uni = uni.loc[:, uni.notna().sum() >= 260]
    return Book(universe=uni, prices=px, cfg=cfg,
                note="names with under 260 sessions dropped as unrankable")


# --------------------------------------------------------------------------
# What the strategies hold right now
# --------------------------------------------------------------------------

def current_picks(book: Book, n_hold: int = 6) -> Dict[str, List[str]]:
    """Each selection strategy's book on the last bar in the cache."""
    mom = cross_sectional_momentum(book.universe, book.safe,
                                   MomentumParams(n_hold=n_hold))
    fin = finviz_momentum_screen(book.universe, book.safe,
                                 FinvizScreenParams(n_hold=n_hold))
    out = {}
    for key, sig in (("momentum", mom), ("finviz", fin)):
        log = sig.holdings_log
        out[key] = list(log[max(log)]) if log else []
    return out


def picks_report(book: Book, n_hold: int = 6) -> str:
    """The current books, and how much they agree.

    The overlap line is the point. These two screens were measured against
    each other over the whole sample and mostly do not intersect, so a reader
    comparing them needs the count in front of them, not the impression that
    two momentum strategies must be picking the same names.
    """
    picks = current_picks(book, n_hold=n_hold)
    mom, fin = set(picks["momentum"]), set(picks["finviz"])
    both = sorted(mom & fin)
    # The lookback is named from the config, never written out -- it has
    # already moved from 12-1 to 6-1 once, and a report that keeps saying
    # 12-1 while the ranker scores 6-1 is the failure this whole layer is
    # meant to prevent.
    label = momentum_label(MomentumParams(n_hold=n_hold))
    lines = [f"CURRENT PICKS — {book.header()}", ""]
    left = [f"Top-{n_hold} NDX momentum ({label}):", f"Top-{n_hold} Finviz screen:"]
    width = max(len(x) for x in left)
    lines.append(f"{left[0].ljust(width)}  "
                 + (", ".join(picks["momentum"]) or "cash"))
    lines.append(f"{left[1].ljust(width)}  "
                 + (", ".join(picks["finviz"]) or "cash"))
    lines.append("")
    lines.append(f"Held by both: {', '.join(both) if both else 'none'} "
                 f"({len(both)} of {n_hold})")
    lines.append(
        f"The two screens select on different things -- one on relative "
        f"{label} rank, the other on proximity to the 52-week high -- so low "
        f"overlap is the normal state, not a bug or a data problem.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# One name, in numbers
# --------------------------------------------------------------------------

def name_report(book: Book, ticker: str, n_hold: int = 6) -> str:
    """Returns, rank, location and every strategy gate for one name."""
    ticker = ticker.upper().strip()
    if ticker not in book.universe.columns:
        near = [c for c in book.universe.columns if c.startswith(ticker[:2])][:8]
        return (f"{ticker} is not in the cached universe "
                f"({book.universe.shape[1]} names). "
                + (f"Nearest by prefix: {', '.join(near)}." if near else ""))

    prof = momentum_profile(book.universe, ticker, safe=book.safe,
                            screen=FinvizScreenParams(n_hold=n_hold),
                            momentum=MomentumParams(n_hold=n_hold))
    if prof["returns"].empty:
        return f"{ticker} has too little history in the cache to profile."

    lines = [f"MOMENTUM PROFILE — {ticker}", book.header(), "",
             "[Returns — Rank is a percentile in THIS universe on THIS date, "
             "1-99; momentum is a relative question and the rank is the answer]"]
    # By column name, not itertuples position: "Universe median" has a space
    # in it, so itertuples renames it to `_4` and any column added ahead of
    # it silently shifts what `_4` means.
    for _, r in prof["returns"].iterrows():
        rank = "n/a" if pd.isna(r["Rank"]) else f"{r['Rank']:.0f}"
        lines.append(f"  {r['Horizon']:<16} {r['Return']:+8.1%}   rank {rank:>3}   "
                     f"universe median {r['Universe median']:+.1%}")

    lines += ["", "[Trend and location]"]
    for r in prof["trend"].itertuples():
        val = ("n/a" if pd.isna(r.Value) else
               f"{r.Value:+.2f} ATR" if r.Unit == "ATR" else f"{r.Value:+.1%}")
        lines.append(f"  {r.Measure:<26} {val}")

    lines += ["", "[Gates each strategy applies]"]
    blocked = []
    for r in prof["gates"].itertuples():
        reading = ("n/a" if pd.isna(r.Value) else
                   f"${r.Value:,.2f}" if r.Fmt == "price" else
                   f"{r.Value:.1%} off high" if r.Fmt == "off" else f"{r.Value:+.1%}")
        mark = "?" if r.Pass is None else ("PASS" if r.Pass else "FAIL")
        lines.append(f"  [{mark:>4}] {r.Strategy:<16} {r.Rule:<40} {reading}")
        if r.Pass is False:
            blocked.append(f"{r.Strategy}: {r.Rule} ({reading})")
    lines.append("")
    lines.append("Blocked by — " + ("; ".join(blocked) if blocked
                                    else "nothing; it clears every rule shown"))
    lines.append(
        "Two gates are missing because this universe carries closes only: the "
        "screen's average-volume filter and the leader rule's share-volume "
        "leg. "
        "Clearing every rule above is necessary, not sufficient.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Backtest results
# --------------------------------------------------------------------------

def performance_report(results: Optional[Dict] = None,
                       rf: Optional[pd.Series] = None) -> str:
    """The strategy comparison table, with the caveats that bound it.

    `results` is a dict of `BacktestResult` as `pipeline.run()` returns. When
    it is None the report says so rather than quietly computing a different
    thing.
    """
    if not results:
        return ("No backtest results were passed in. Run "
                "`python run_backtest.py` (or `qbs.pipeline.run()`) and hand "
                "the results in; this tool does not invent a backtest.")
    from ..metrics import format_summary, summary_table
    from ..config import STRATEGY_LABELS

    table = summary_table(results, rf=rf, labels=STRATEGY_LABELS)
    pretty = format_summary(table)
    lines = ["STRATEGY PERFORMANCE", "", pretty.to_string(), "",
             "Reading this honestly:",
             "- These are IN-SAMPLE results on the window the parameters were "
             "chosen over. Treat the ranking as a hypothesis, not a measurement.",
             "- Sharpe here is excess of BOXX where a risk-free series was "
             "supplied; without one it is a total-return Sharpe and flatters "
             "everything equally.",
             "- Max drawdown is daily close to close. Intraday pain was worse.",
             "- The universe is today's index membership unless a "
             "point-in-time mask was used, which biases every cross-sectional "
             "result upward by excluding names that were dropped."]
    return "\n".join(lines)


def funnel_report(funnel: Optional[pd.DataFrame] = None) -> str:
    """Selection -> breakout conversion, stage by stage."""
    if funnel is None or funnel.empty:
        return ("No funnel was passed in. Build one with "
                "`qbs.breakout.breakout_funnel(...)`; this tool does not "
                "invent one.")
    from ..breakout import funnel_summary

    summary = funnel_summary(funnel)
    lines = ["SELECTION -> BREAKOUT FUNNEL", f"{len(funnel)} picks", ""]
    for r in summary.itertuples():
        prev = "" if pd.isna(r.of_previous) else f"  ({r.of_previous:.0%} of the stage above)"
        lines.append(f"  {r.stage:<28} {r.n:>5}  {r.of_selected:>6.1%} of picks{prev}")
    lines += ["",
              "Where the attrition is concentrated is the finding. A large "
              "drop at 'had resistance overhead' is structural: a screen that "
              "demands a name be within 10% of its 52-week high selects names "
              "with nothing above them to break out through."]
    return "\n".join(lines)


def trades_report(trades: Optional[pd.DataFrame] = None) -> str:
    """Trade statistics in R, with the concentration check that guards them.

    Expectancy over a few dozen trades is dominated by its best few. So the
    report states what share of total R the top three trades are. Where that
    share is large, the mean is a description of those three trades and not
    of the strategy, and no amount of confident prose changes it.
    """
    if trades is None or trades.empty:
        return ("No trades were passed in. Build a book with "
                "`qbs.breakout.weekly_breakout_book(...)`.")
    from ..breakout import trade_stats

    st = trade_stats(trades)
    closed = trades[trades["exit_time"].notna()]
    r = closed["R_multiple"].dropna().sort_values(ascending=False)
    lines = ["BREAKOUT TRADE STATISTICS", ""]
    lines.append(f"  trades           {st['n_trades']:.0f} "
                 f"({st['n_closed']:.0f} closed)")
    for key, label, fmt in (("hit_rate", "hit rate", "{:.1%}"),
                            ("expectancy_R", "expectancy", "{:+.2f}R"),
                            ("avg_win_R", "average win", "{:+.2f}R"),
                            ("avg_loss_R", "average loss", "{:+.2f}R"),
                            ("pct_stop", "exited at stop", "{:.0%}"),
                            ("pct_take_profit", "exited at target", "{:.0%}"),
                            ("pct_time_or_ema", "exited on time/EMA", "{:.0%}")):
        v = st.get(key, float("nan"))
        lines.append(f"  {label:<16} " + ("n/a" if pd.isna(v) else fmt.format(v)))

    lines.append("")
    if len(r) >= 3 and r.abs().sum() > 0:
        top3 = float(r.head(3).sum() / r.abs().sum())
        lines.append(f"  top 3 trades are {top3:.0%} of all R across "
                     f"{len(r)} closed trades")
    if st["n_closed"] < MIN_TRADES_TO_TRUST:
        lines.append(
            f"  SAMPLE WARNING: {st['n_closed']:.0f} closed trades is below "
            f"{MIN_TRADES_TO_TRUST}. Expectancy at this count is an "
            f"observation about the trades that happened, NOT an expected "
            f"value for the next one. Do not size positions off it.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The market around the picks
# --------------------------------------------------------------------------

def breadth_report(book: Book, p: Optional[BreadthParams] = None,
                   lookback: int = 5) -> str:
    """Participation, not the index: what the average name is doing."""
    p = p or BreadthParams()
    res = daily_breadth(book.universe, qqq=book.prices.get("QQQ"), p=p,
                        universe_note=book.note)
    table = res.table
    if table is None or table.empty:
        return "Breadth could not be computed from the cached universe."
    recent = table.tail(lookback)
    last = recent.iloc[-1]
    lines = [f"MARKET BREADTH — {book.header()}", ""]
    # `pct_above_*` already comes back on a 0-100 scale, so it is printed with
    # an explicit "%" -- a `:.1%` here would multiply by 100 a second time and
    # report breadth of 4387%, which is wrong in a way that still parses.
    for d, row in recent.iterrows():
        lines.append(
            f"  {d:%Y-%m-%d}  up4 {row.get('up4', float('nan')):>4.0f} / "
            f"dn4 {row.get('dn4', float('nan')):>4.0f}   "
            f"above 20d {row.get('pct_above_fast', float('nan')):>5.1f}%   "
            f"above 50d {row.get('pct_above_slow', float('nan')):>5.1f}%   "
            f"leaders {row.get('mli_n', float('nan')):>4.0f}")
    # `ma_class` takes the band name second: the 20- and 50-day readings have
    # different thresholds, and passing `p` positionally here would silently
    # classify both against the slow band.
    lines += ["", "Classification on the last row: "
              f"20-day {ma_class(last.get('pct_above_fast', np.nan), 'fast', p)}, "
              f"50-day {ma_class(last.get('pct_above_slow', np.nan), 'slow', p)}, "
              f"QQQ stretch {atr_class(last.get('qqq_atr', np.nan), p)}"]
    lines.append(
        "Breadth is participation, not direction. A rising index on "
        "narrowing breadth and one on broadening breadth look identical on a "
        "price chart and are not the same market.")
    return "\n".join(lines)
