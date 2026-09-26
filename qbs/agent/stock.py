"""One stock's recent price action, as the dashboard computes it, in text.

The Analyst tab's price panel shows a chart, the EMAs, the nearest support
and resistance, and a momentum table. `name_report` in `evidence.py` already
covers the momentum table. This module covers the rest -- the part a reader
takes off the chart by eye -- so the analyst can say "recent performance"
with numbers rather than with an impression of a picture it cannot see:

* returns over the last day, week, month, quarter, half and year, next to
  QQQ's and SPY's over the same sessions, so "up 8% this month" arrives with
  "QQQ was up 6%" attached;
* where price sits against the 10/20/50/200-day EMAs (the chart's four
  lines), and whether each is rising;
* the 52-week and 20-day range, the drawdown from the recent peak;
* volatility: ATR(14) as a share of price, 20-day realised vol, beta to QQQ;
* the nearest support and resistance, re-derived from history up to the
  as-of date only -- the chart's causal rule;
* volume against its own 20-day average, when OHLCV is available;
* the last N sessions, one line each.

Same contract as `evidence.py`: numbers this package computed, each with the
caveat that bounds it; nothing here imports LangChain or talks to a model.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..breakout import atr, closes_to_bars, levels_in_view, sr_levels
from ..candles import volume_stats
from ..config import BreakoutParams
from .evidence import Book

EMA_SPANS = (10, 20, 50, 200)          # the price panel's four lines
HORIZONS = (("1 day", 1), ("1 week", 5), ("1 month", 21), ("3 months", 63),
            ("6 months", 126), ("12 months", 252))
BETA_WINDOW = 126
VOL_WINDOW = 20

OhlcLoader = Callable[[str], Optional[pd.DataFrame]]


def _ret(s: Optional[pd.Series], n: int) -> float:
    if s is None or len(s) <= n:
        return float("nan")
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0)


def _pct(v: float, signed: bool = True) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _usable_ohlc(ohlc: Optional[pd.DataFrame], asof: pd.Timestamp
                 ) -> Optional[pd.DataFrame]:
    """Real OHLC through `asof`, or None. Never synthesised from closes."""
    if ohlc is None or ohlc.empty:
        return None
    if not {"Open", "High", "Low", "Close"} <= set(ohlc.columns):
        return None
    out = ohlc.sort_index().loc[:asof].dropna(subset=["High", "Low", "Close"])
    return out if len(out) > 20 else None


def price_action_report(
    book: Book,
    ticker: str,
    sessions: int = 10,
    ohlc: Optional[pd.DataFrame] = None,
    spy: Optional[pd.Series] = None,
) -> str:
    """Recent performance, trend, range, volatility, levels and volume."""
    ticker = ticker.upper().strip()
    close = book.closes(ticker)
    if close is None:
        near = [c for c in book.universe.columns if c.startswith(ticker[:2])][:8]
        return (f"No prices for {ticker}: it is not in the cached universe "
                f"({book.universe.shape[1]} names) and not on the watchlist. "
                + (f"Nearest by prefix: {', '.join(near)}." if near else ""))
    if len(close) < 30:
        return (f"{ticker} has only {len(close)} sessions of history through "
                f"{book.asof:%Y-%m-%d} -- too little to describe a trend.")

    asof = close.index[-1]
    last = float(close.iloc[-1])
    sessions = max(1, min(int(sessions), 60))
    bars = _usable_ohlc(ohlc, asof)
    qqq = book.prices["QQQ"].loc[:asof].dropna() if "QQQ" in book.prices else None
    spy = spy.loc[:asof].dropna() if spy is not None else None

    lines = [f"PRICE ACTION — {ticker}", book.header()]
    if asof < book.asof:
        lines.append(f"{ticker}'s own last close is {asof:%Y-%m-%d}, before "
                     "the book's -- every figure below stands on that date.")
    if book.is_outsider(ticker):
        lines.append(f"{ticker} is a watchlist name outside the ranking "
                     "universe; its closes are its own download, aligned to "
                     "the universe's calendar.")
    lines += ["", f"Last close {last:,.2f} on {asof:%Y-%m-%d}", ""]

    # ---- returns, against the two indices --------------------------------
    uni = book.universe.loc[:asof]
    lines.append("[Returns — the name, QQQ and SPY over the same sessions; "
                 "'vs QQQ' is the difference in percentage points]")
    lines.append(f"  {'horizon':<10} {ticker:>9} {'QQQ':>9} {'SPY':>9} "
                 f"{'vs QQQ':>9} {'univ. median':>13}")
    for label, n in HORIZONS:
        r, q, s = _ret(close, n), _ret(qqq, n), _ret(spy, n)
        med = (float((uni.iloc[-1] / uni.iloc[-1 - n] - 1.0).median())
               if len(uni) > n else float("nan"))
        diff = "n/a" if pd.isna(r) or pd.isna(q) else f"{(r - q) * 100:+.1f}pp"
        lines.append(f"  {label:<10} {_pct(r):>9} {_pct(q):>9} {_pct(s):>9} "
                     f"{diff:>9} {_pct(med):>13}")
    prior = close.loc[close.index.year < asof.year]
    if len(prior):
        lines.append(f"  {'YTD':<10} {_pct(last / float(prior.iloc[-1]) - 1):>9}")
    if spy is None:
        lines.append("  (SPY was not supplied to this run, so its column is n/a.)")

    # ---- trend: the chart's four EMAs ------------------------------------
    lines += ["", "[Trend — the price panel's EMAs; 'rising' compares each "
              "EMA with its value 5 sessions earlier]"]
    ema_last = {}
    for n in EMA_SPANS:
        ema = close.ewm(span=n, adjust=False, min_periods=n).mean()
        if pd.isna(ema.iloc[-1]):
            lines.append(f"  EMA {n:<4} n/a (under {n} sessions)")
            continue
        e = float(ema.iloc[-1])
        ema_last[n] = e
        slope = ("rising" if len(ema) > 5 and e > float(ema.iloc[-6]) else
                 "falling")
        side = "above" if last > e else "below"
        lines.append(f"  EMA {n:<4} {e:>10,.2f}   price {last / e - 1:+6.1%} "
                     f"({side})   {slope}")
    if len(ema_last) == len(EMA_SPANS):
        vals = [ema_last[n] for n in EMA_SPANS]
        if all(a > b for a, b in zip(vals, vals[1:])):
            order = "stacked bullish (10 > 20 > 50 > 200)"
        elif all(a < b for a, b in zip(vals, vals[1:])):
            order = "stacked bearish (10 < 20 < 50 < 200)"
        else:
            order = "mixed -- the averages are not in trend order"
        lines.append(f"  EMA order: {order}")

    # ---- range and drawdown ----------------------------------------------
    hi_src = bars["High"] if bars is not None else close
    lo_src = bars["Low"] if bars is not None else close
    y_hi, y_lo = float(hi_src.tail(252).max()), float(lo_src.tail(252).min())
    m_hi, m_lo = float(hi_src.tail(20).max()), float(lo_src.tail(20).min())
    peak_3m = float(close.tail(63).max())
    lines += ["", "[Range]" + ("" if bars is not None else
                               " (closes only -- intraday extremes unknown)")]
    lines.append(f"  52-week high {y_hi:,.2f} ({last / y_hi - 1:+.1%} from it) · "
                 f"low {y_lo:,.2f} ({last / y_lo - 1:+.1%} above it)"
                 + ("" if len(close) >= 252 else
                    f" -- only {len(close)} sessions of history"))
    lines.append(f"  20-day high {m_hi:,.2f} · low {m_lo:,.2f} · position in "
                 f"that range {(last - m_lo) / (m_hi - m_lo):.0%}"
                 if m_hi > m_lo else "  20-day range is flat")
    lines.append(f"  Drawdown from the 3-month closing peak "
                 f"({peak_3m:,.2f}): {last / peak_3m - 1:+.1%}")

    # ---- volatility --------------------------------------------------------
    daily = close.pct_change().dropna()
    lines += ["", "[Volatility]"]
    if bars is not None:
        a = float(atr(bars, 14).iloc[-1])
        atr_note = "ATR(14) from real highs and lows"
    else:
        a = float(atr(closes_to_bars(close.to_frame(ticker))[ticker], 14).iloc[-1])
        atr_note = ("ATR(14) from closes only -- understates the true range")
    lines.append(f"  {atr_note}: {a:,.2f} ({a / last:.1%} of price)")
    if len(daily) > 1 and not pd.isna(a) and a > 0:
        move = float(close.iloc[-1] - close.iloc[-2])
        lines.append(f"  Last session's move: {move:+,.2f} = {move / a:+.2f} ATR")
    rv = float(daily.tail(VOL_WINDOW).std() * np.sqrt(252))
    qv = (float(qqq.pct_change().tail(VOL_WINDOW).std() * np.sqrt(252))
          if qqq is not None and len(qqq) > VOL_WINDOW else float("nan"))
    lines.append(f"  {VOL_WINDOW}-day realised vol (annualised): {rv:.0%}"
                 + ("" if pd.isna(qv) else f" · QQQ {qv:.0%}"))
    if qqq is not None:
        j = pd.concat([daily, qqq.pct_change()], axis=1, join="inner"
                      ).dropna().tail(BETA_WINDOW)
        if len(j) >= 60 and j.iloc[:, 1].var() > 0:
            beta = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
            corr = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
            lines.append(f"  Beta to QQQ over {len(j)} sessions: {beta:.2f} "
                         f"(correlation {corr:.2f})")

    # ---- support / resistance ---------------------------------------------
    lvl_bars = bars if bars is not None else closes_to_bars(
        close.to_frame(ticker))[ticker]
    levels = sr_levels(lvl_bars.tail(252 * 2), BreakoutParams())
    window = close.tail(252)
    _, n_view, overhead = levels_in_view(levels, last, float(window.min()),
                                         float(window.max()), 30)
    above = sorted(L for L in levels if L > last)
    below = sorted((L for L in levels if L <= last), reverse=True)
    lines += ["", "[Support / resistance — swing levels from history up to "
              f"{asof:%Y-%m-%d} only, the chart's causal rule]"]

    def _lvl(L: float) -> str:
        d = f"{L / last - 1:+.1%}"
        return f"{L:,.2f} ({d}" + (f", {(L - last) / a:+.1f} ATR)"
                                   if a and not pd.isna(a) else ")")
    lines.append("  Nearest resistance: "
                 + (", ".join(_lvl(L) for L in above[:2]) or "none overhead"))
    lines.append("  Nearest support:    "
                 + (", ".join(_lvl(L) for L in below[:2]) or "none below"))
    if not overhead:
        lines.append("  No resistance in the 12-month window: the name has "
                     "cleared every level its chart shows (price discovery).")

    # ---- volume ------------------------------------------------------------
    vol = (bars["Volume"].astype(float) if bars is not None
           and "Volume" in bars.columns and bars["Volume"].notna().any()
           else None)
    lines += ["", "[Volume]"]
    if vol is None:
        lines.append("  No volume for this name in this run (closes-only "
                     "cache; the dashboard's Online source fetches OHLCV).")
    else:
        vs = volume_stats(vol, bars["Close"], window=20)
        lines.append(f"  Last session {vs['last']:,.0f} shares = "
                     f"{vs['ratio']:.2f}x its prior 20-day average "
                     f"({vs['avg']:,.0f}); average $ volume "
                     f"${vs['avg_value'] / 1e6:,.1f}m/day")
        chg = bars["Close"].diff().tail(20)
        v20 = vol.reindex(chg.index)
        up_v, dn_v = float(v20[chg > 0].sum()), float(v20[chg < 0].sum())
        if dn_v > 0:
            lines.append(f"  Last 20 sessions: volume on up days / down days "
                         f"= {up_v / dn_v:.2f} ({int((chg > 0).sum())} up, "
                         f"{int((chg < 0).sum())} down) -- above 1 reads as "
                         "accumulation, below 1 as distribution")

    # ---- the last N sessions ---------------------------------------------
    lines += ["", f"[Last {sessions} sessions]"]
    avg_vol = vol.shift(1).rolling(20, min_periods=10).mean() if vol is not None else None
    for d in close.index[-sessions:][::-1]:
        c = float(close.loc[d])
        p = close.shift(1).loc[d]
        row = f"  {d:%Y-%m-%d}  {c:>10,.2f}  {_pct(c / p - 1 if p == p else np.nan):>7}"
        if bars is not None and d in bars.index:
            b = bars.loc[d]
            row += f"  range {(b['High'] - b['Low']) / c:5.1%}"
            if vol is not None and avg_vol is not None and not pd.isna(avg_vol.get(d)):
                row += f"  vol {float(vol.loc[d]) / float(avg_vol.loc[d]):.2f}x avg"
        lines.append(row)

    lines += ["",
              "Describes what HAS happened, through the date above. None of "
              "it is a forecast: trend and levels are how the chart reads, "
              "not where price must go next."]
    return "\n".join(lines)
