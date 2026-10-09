"""An auto-selecting swing book: ~6 slots of fixed dollars, filled from the
swing screens in `qbs.swing`, each trade managed by its own stop, target and
time limit.

How it trades, on each session's close:

1. **Exits first.** Each open trade is closed when the close is at or below
   its stop, at or above its target, or it has been held for its setup's
   maximum holding period (the upper end of the setup's "typical hold").
2. **Then entries.** Free slots are filled from that session's swing scan
   (computed on prices up to that close only), best candidates first, one
   trade per ticker. A name exited today is not re-bought the same day.

Fills are at the close the decision is made on -- the lab's convention
(decide at ~15:30, fill in the closing auction). Stops are checked on closes
only, because the cache holds closes: a real stop order fills intraday, often
better on a slow drift and worse on a gap.

Costs are explicit, because they bite at $1,000 a trade:

* commission, IB fixed pricing -- $0.005 a share, minimum $1, maximum 1% of
  the trade value -- so a $1,000 round trip costs at least $2 (0.2%);
* slippage, a few basis points of each fill;
* whole shares by default: a $505 stock fills 1 share of a $1,000 slot and
  leaves $495 idle. Set `whole_shares=False` if the account trades fractions.

Idle cash earns the safe asset's daily return (BOXX), as in the rest of the
lab. This is a research backtest: nothing here reaches an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Upper end of each setup's typical hold, in sessions (qbs.swing.SETUPS).
DEFAULT_MAX_HOLD = {
    "trend_pullback": 15, "breakout_retest": 20, "range_bounce": 10,
    "mean_reversion": 7, "volatility_contraction": 15, "relative_strength": 20,
}


@dataclass
class SwingBookParams:
    n_slots: int = 6
    slot_usd: float = 1000.0
    setups: Tuple[str, ...] = tuple(DEFAULT_MAX_HOLD)
    max_hold: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_HOLD))
    # How to choose among the day's candidates for the free slots:
    #   "rr"        reward/risk to the screen's target (no target -> last)
    #   "momentum"  6-1 momentum, the Top-6 book's own ranking
    rank_by: str = "rr"
    min_rr: Optional[float] = None    # skip candidates below this reward/risk
    need_target: bool = False         # skip candidates with no target at all
    # Widen each screen's stop: entry - stop_mult x (entry - screen stop).
    # The screens' stops sit 1-2 ATRs under the close, and on close-only data
    # that ATR understates the real range -- so 1.0 is tight.
    stop_mult: float = 1.0
    whole_shares: bool = True
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    commission_max_pct: float = 0.01
    slippage_bps: float = 5.0

    def __post_init__(self):
        if self.n_slots < 1 or self.slot_usd <= 0:
            raise ValueError("n_slots and slot_usd must be positive")
        if self.rank_by not in ("rr", "momentum"):
            raise ValueError("rank_by must be 'rr' or 'momentum'")


def commission(shares: float, price: float, p: SwingBookParams) -> float:
    """IB fixed pricing: per share, with a per-order minimum and maximum."""
    if shares <= 0:
        return 0.0
    fee = max(p.commission_min, p.commission_per_share * shares)
    return min(fee, p.commission_max_pct * shares * price)


@dataclass
class Trade:
    ticker: str
    setup: str
    entry_date: pd.Timestamp
    entry_px: float
    shares: float
    stop: Optional[float]
    target: Optional[float]
    max_hold: int
    exit_date: Optional[pd.Timestamp] = None
    exit_px: Optional[float] = None
    reason: str = ""
    costs: float = 0.0
    held: int = 0

    @property
    def pnl(self) -> float:
        if self.exit_px is None:
            return float("nan")
        return (self.exit_px - self.entry_px) * self.shares - self.costs

    @property
    def ret(self) -> float:
        return self.pnl / (self.entry_px * self.shares)


def candidates_for(scan_day: Dict[str, pd.DataFrame], p: SwingBookParams,
                   mom: Optional[pd.Series]) -> pd.DataFrame:
    """One row per ticker from that day's scan: the setup it qualifies for
    (its best by reward/risk if several), with stop, target and rr."""
    rows = []
    for key in p.setups:
        df = scan_day.get(key)
        if df is None or df.empty:
            continue
        for t, r in df.iterrows():
            rr = r.get("rr")
            rr = float(rr) if rr is not None and rr == rr else float("nan")
            tgt = r.get("target")
            tgt = float(tgt) if tgt is not None and tgt == tgt else None
            stop = r.get("stop")
            stop = float(stop) if stop is not None and stop == stop else None
            rows.append(dict(ticker=t, setup=key, close=float(r["close"]),
                             stop=stop, target=tgt, rr=rr))
    if not rows:
        return pd.DataFrame(columns=["ticker", "setup", "close", "stop",
                                     "target", "rr", "score"])
    c = pd.DataFrame(rows)
    if p.need_target:
        c = c[c["target"].notna()]
    if p.min_rr is not None:
        c = c[c["rr"] >= p.min_rr]
    # A ticker in several setups trades once, under its best reward/risk.
    c = (c.sort_values("rr", ascending=False, na_position="last")
          .drop_duplicates("ticker"))
    if p.rank_by == "momentum" and mom is not None:
        c["score"] = c["ticker"].map(mom)
    else:
        c["score"] = c["rr"]
    return c.sort_values("score", ascending=False, na_position="last")


def run_swing_book(
    closes: pd.DataFrame,
    safe: pd.Series,
    scans: Dict[pd.Timestamp, Dict[str, pd.DataFrame]],
    p: Optional[SwingBookParams] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    momentum: Optional[pd.DataFrame] = None,
    ohlc: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.Series, List[Trade]]:
    """Simulate the book. Returns (equity in dollars by session, trades).

    With `ohlc` (ticker -> daily Open/High/Low/Close), stops and targets fill
    INTRADAY, as resting orders would: a stop fills when the day's low
    reaches it -- at the stop, or at the open if the stock gapped below it --
    and a target when the day's high reaches it (at the target, or the open
    on a gap above). A day that touches both is counted as a stop, the
    conservative order. Names without bars, and every exit without `ohlc`,
    are checked on the close as before.

    `scans` maps each session to `qbs.swing.scan` output computed on prices up
    to that session only. `momentum` (date x ticker 6-1 momentum) is needed
    only for `rank_by="momentum"`.
    """
    p = p or SwingBookParams()
    px = closes.sort_index()
    days = px.index
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    safe_ret = safe.reindex(px.index).ffill().pct_change().fillna(0.0)
    slip = p.slippage_bps / 1e4

    cash = p.n_slots * p.slot_usd
    open_: Dict[str, Trade] = {}
    trades: List[Trade] = []
    equity = {}

    for d in days:
        cash *= 1.0 + float(safe_ret.get(d, 0.0))
        row = px.loc[d]

        # 1. exits, on today's close
        exited_today = set()
        for t, tr in list(open_.items()):
            c = row.get(t)
            if c is None or c != c:
                continue
            tr.held += 1
            reason, px_out = "", c
            bar = None
            if ohlc is not None and t in ohlc and d in ohlc[t].index:
                bar = ohlc[t].loc[d]
            if bar is not None:
                o, hi, lo = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
                if tr.stop is not None and lo <= tr.stop:
                    reason, px_out = "stop", min(o, tr.stop)
                elif tr.target is not None and hi >= tr.target:
                    reason, px_out = "target", max(o, tr.target)
            elif tr.stop is not None and c <= tr.stop:
                reason = "stop"
            elif tr.target is not None and c >= tr.target:
                reason = "target"
            if not reason and tr.held >= tr.max_hold:
                reason, px_out = "time", c
            if reason:
                fill = px_out * (1 - slip)
                fee = commission(tr.shares, fill, p)
                cash += tr.shares * fill - fee
                tr.costs += fee + tr.shares * px_out * slip
                tr.exit_date, tr.exit_px, tr.reason = d, px_out, reason
                trades.append(tr)
                del open_[t]
                exited_today.add(t)

        # 2. entries into free slots
        free = p.n_slots - len(open_)
        scan_day = scans.get(d)
        if free > 0 and scan_day:
            mom = momentum.loc[d] if momentum is not None and d in momentum.index else None
            cands = candidates_for(scan_day, p, mom)
            for _, r in cands.iterrows():
                if free <= 0:
                    break
                t = r["ticker"]
                if t in open_ or t in exited_today:
                    continue
                c = row.get(t)
                if c is None or c != c:
                    continue
                fill = c * (1 + slip)
                budget = min(p.slot_usd, cash)
                shares = (math.floor(budget / fill) if p.whole_shares
                          else budget / fill)
                if shares <= 0:
                    continue          # one share costs more than the slot
                fee = commission(shares, fill, p)
                if shares * fill + fee > cash:
                    continue
                cash -= shares * fill + fee
                stop = r["stop"]
                if stop is not None and p.stop_mult != 1.0:
                    stop = c - p.stop_mult * (c - stop)
                open_[t] = Trade(
                    ticker=t, setup=r["setup"], entry_date=d, entry_px=c,
                    shares=shares, stop=stop, target=r["target"],
                    max_hold=int(p.max_hold.get(r["setup"], 15)),
                    costs=fee + shares * c * slip)
                free -= 1

        held_value = sum(tr.shares * float(row.get(t, np.nan))
                         for t, tr in open_.items()
                         if row.get(t) == row.get(t))
        equity[d] = cash + held_value

    for tr in open_.values():
        tr.reason = "open"
        trades.append(tr)
    return pd.Series(equity, name="equity"), trades


def trade_table(trades: List[Trade]) -> pd.DataFrame:
    return pd.DataFrame([dict(
        ticker=t.ticker, setup=t.setup, entry=t.entry_date, exit=t.exit_date,
        entry_px=t.entry_px, exit_px=t.exit_px, shares=t.shares,
        stop=t.stop, target=t.target, held=t.held, reason=t.reason,
        pnl=t.pnl, ret=t.ret if t.exit_px is not None else float("nan"),
        costs=t.costs) for t in trades])


def momentum_first_picks(
    scan_day: Dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    n_slots: int = 6,
    stop_mult: float = 2.0,
    slot_usd: float = 1000.0,
    lookback: int = 126,
    skip: int = 21,
) -> pd.DataFrame:
    """Today's picks for the momentum-first swing book.

    The variant that held up in docs/SWING_BOOK.md: every name passing one of
    the six swing screens today, ranked by 6-1 momentum (the Top-6 book's
    own score), best `n_slots` first. Each pick keeps its screen's target and
    setup time limit, with the stop widened to `stop_mult` x the screen's
    distance below the close -- the screens' stops sit 1-2 ATRs under the
    close, which the backtest found too tight.

    `scan_day` is `qbs.swing.scan` output; `closes` the same universe's closes
    up to today. Picks are candidates for new trades, not a held book: the
    dashboard has no record of what was bought on earlier days.
    """
    p = SwingBookParams(n_slots=n_slots, slot_usd=slot_usd, rank_by="momentum",
                        stop_mult=stop_mult)
    c = closes.dropna(how="all")
    if len(c) <= lookback:
        mom = pd.Series(dtype=float)
    else:
        mom = c.iloc[-1 - skip] / c.iloc[-1 - lookback] - 1.0
    cands = candidates_for(scan_day, p, mom)
    # As in the backtest, a name one share of which costs more than a slot is
    # skipped and the next one takes its place.
    cands = cands[cands["score"].notna() & (cands["close"] <= slot_usd)].head(n_slots)
    rows = []
    for rank, r in enumerate(cands.itertuples(index=False), 1):
        close = float(r.close)
        stop = None if r.stop is None else close - stop_mult * (close - float(r.stop))
        target = r.target
        rr = ((target - close) / (close - stop)
              if stop is not None and target is not None and target > close > stop
              else float("nan"))
        rows.append(dict(
            rank=rank, ticker=r.ticker, setup=r.setup, close=close, stop=stop,
            target=target, rr=rr, momentum=float(r.score),
            max_hold=int(p.max_hold.get(r.setup, 15)),
            shares=int(math.floor(slot_usd / close)) if close > 0 else 0,
        ))
    return pd.DataFrame(rows, columns=["rank", "ticker", "setup", "close", "stop",
                                       "target", "rr", "momentum", "max_hold",
                                       "shares"])
