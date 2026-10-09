#!/usr/bin/env python3
"""Intraday / overnight add-on strategies, tested on IB bars.

    python scripts/intraday_study.py --data data/ib

Can a 15-minute monitor run beside the momentum book make money on its own?
Two data sets, both from the IB connector (OHLCV, regular hours, as traded):

* **daily QQQ, ~4 years** -- for the strategies that only need each day's
  open and close: overnight vs intraday, gap fades.
* **hourly bars, ~7 months, QQQ + 16 large Nasdaq names** -- for the
  strategies that need the intraday path. IB serves at most 1,000 bars a
  request and no end date, so 15-minute history reaches back only ~38 days;
  hourly is the finest bar with a usable sample. The 15-minute QQQ file is
  used as a cross-check on the same rules.

Every rule decides on a bar's CLOSE and fills at the NEXT bar's OPEN (or at
the session's close for an end-of-day exit), so no trade uses a price it
could not have seen. Costs are charged per side: `--cost-bps` on top of the
traded price (spread + commission; 2 bp is IB's $1 minimum on a $5,000
order plus half a cent of spread on a liquid name).

Each strategy is reported per trade (count, win rate, average gross and net
return, t-statistic of the net) and as a sleeve: K equal slots, trades taking
a free slot in time order, idle slots in cash at 0%.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

NY = "America/New_York"


def load(path: str) -> pd.DataFrame:
    d = json.load(open(path))
    t = pd.to_datetime(d["time"]).tz_convert(NY)
    df = pd.DataFrame({k: d[k] for k in ("open", "high", "low", "close", "volume")},
                      index=t).astype(float)
    return df[~df.index.duplicated()].sort_index()


# --------------------------------------------------------------------------
# trades -> statistics
# --------------------------------------------------------------------------

Trade = Tuple[pd.Timestamp, pd.Timestamp, float, float, str]   # in, out, px_in, px_out, sym


def summarise(trades: List[Trade], cost_bps: float, slots: int, years: float) -> Dict:
    if not trades:
        return dict(Trades=0)
    tr = pd.DataFrame(trades, columns=["t_in", "t_out", "p_in", "p_out", "sym"])
    tr["gross"] = tr["p_out"] / tr["p_in"] - 1.0
    tr["net"] = tr["gross"] - 2 * cost_bps / 1e4
    n = len(tr)
    sd = tr["net"].std(ddof=1) if n > 1 else np.nan
    # Sleeve: K slots, each trade takes one if free, at 1/K of current equity.
    tr = tr.sort_values("t_in")
    eq, busy = 1.0, []          # busy: list of (t_out, stake, net)
    for r in tr.itertuples():
        done = [b for b in busy if b[0] <= r.t_in]
        for b in done:
            eq += b[1] * b[2]
        busy = [b for b in busy if b[0] > r.t_in]
        if len(busy) < slots:
            busy.append((r.t_out, eq / slots, r.net))
    for b in busy:
        eq += b[1] * b[2]
    return dict(
        Trades=n, **{"Trades/yr": n / years},
        Win=float((tr["net"] > 0).mean()),
        **{"Avg gross (bp)": tr["gross"].mean() * 1e4,
           "Avg net (bp)": tr["net"].mean() * 1e4,
           "t(net)": tr["net"].mean() / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan,
           "Sleeve ann.": eq ** (1 / years) - 1 if eq > 0 else -1.0},
    )


# --------------------------------------------------------------------------
# daily rules (QQQ, ~4 years)
# --------------------------------------------------------------------------

def daily_rules(d: pd.DataFrame) -> Dict[str, List[Trade]]:
    o, c = d["open"].to_numpy(), d["close"].to_numpy()
    out: Dict[str, List[Trade]] = {k: [] for k in (
        "overnight: buy close, sell next open",
        "intraday: buy open, sell close",
        "gap down > 0.5%: buy open, sell close",
        "gap up > 0.5%: buy open, sell close",
        "after a down day: buy close, sell next open",
        "after a down day: buy close, hold 1 day")}
    idx = d.index
    for i in range(1, len(d) - 1):
        t, tn = idx[i], idx[i + 1]
        out["overnight: buy close, sell next open"].append((t, tn, c[i], o[i + 1], "QQQ"))
        out["intraday: buy open, sell close"].append((t, t, o[i], c[i], "QQQ"))
        gap = o[i] / c[i - 1] - 1
        if gap < -0.005:
            out["gap down > 0.5%: buy open, sell close"].append((t, t, o[i], c[i], "QQQ"))
        if gap > 0.005:
            out["gap up > 0.5%: buy open, sell close"].append((t, t, o[i], c[i], "QQQ"))
        if c[i] < o[i]:
            out["after a down day: buy close, sell next open"].append(
                (t, tn, c[i], o[i + 1], "QQQ"))
            out["after a down day: buy close, hold 1 day"].append(
                (t, tn, c[i], c[i + 1], "QQQ"))
    return out


# --------------------------------------------------------------------------
# intraday rules (hourly or 15-minute bars)
# --------------------------------------------------------------------------

def rsi(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def intraday_rules(sym: str, b: pd.DataFrame, bars_per_day: int,
                   allowed: Dict = None) -> Dict[str, List[Trade]]:
    """`allowed` maps a date to the set of names the momentum book held."""
    out: Dict[str, List[Trade]] = {k: [] for k in (
        "RSI2 < 10 in uptrend, exit RSI2 > 70 or 2 days",
        "opening-range breakout, exit at close",
        "first hour down > 1%: buy, exit at close",
        "last-hour momentum (first 30 min up)",
        "RSI2 < 10 dip in a momentum-book name")}
    o, h, l, c = b["open"].values, b["high"].values, b["low"].values, b["close"].values
    t = b.index
    day = t.normalize()
    r2 = rsi(b["close"], 2).values
    sma = b["close"].rolling(5 * bars_per_day, min_periods=5 * bars_per_day).mean().values
    n = len(b)
    max_hold = 2 * bars_per_day

    # RSI(2) dips in an uptrend -- and the same, only in momentum-book names.
    for key, gate in (("RSI2 < 10 in uptrend, exit RSI2 > 70 or 2 days", None),
                      ("RSI2 < 10 dip in a momentum-book name", allowed)):
        i = 0
        while i < n - 1:
            ok = r2[i] < 10 and not np.isnan(sma[i]) and c[i] > sma[i]
            if gate is not None:
                held = gate.get(day[i].tz_localize(None))
                ok = ok and held is not None and sym in held
            if ok:
                j = i + 1
                entry = o[j]
                k = j
                while k < n - 1 and r2[k] <= 70 and (k - j) < max_hold:
                    k += 1
                # exit at the next bar's open after the exit signal
                ex = o[k + 1] if k + 1 < n else c[k]
                out[key].append((t[j], t[min(k + 1, n - 1)], entry, ex, sym))
                i = k + 1
            else:
                i += 1

    # Per-day rules.
    for d, g in b.groupby(day):
        if len(g) < bars_per_day - 1:
            continue
        go, gh, gl, gc = g["open"].values, g["high"].values, g["low"].values, g["close"].values
        gt = g.index
        first = 1 if bars_per_day == 7 else 2          # bars in the first 30 minutes
        or_high = gh[:first].max()
        # Opening-range breakout: the first later bar closing above the
        # opening range's high; buy the next open, sell the close.
        for k in range(first, len(g) - 1):
            if gc[k] > or_high:
                out["opening-range breakout, exit at close"].append(
                    (gt[k + 1], gt[-1], go[k + 1], gc[-1], sym))
                break
        # First hour down more than 1% (to 10:30): buy then, sell the close.
        fh = 2 if bars_per_day == 7 else 4
        if len(g) > fh and gc[fh - 1] / go[0] - 1 < -0.01:
            out["first hour down > 1%: buy, exit at close"].append(
                (gt[fh], gt[-1], go[fh], gc[-1], sym))
        # Last-hour momentum: first 30 minutes up -> hold the last hour.
        last = 1 if bars_per_day == 7 else 4
        if gc[first - 1] > go[0]:
            out["last-hour momentum (first 30 min up)"].append(
                (gt[-last], gt[-1], go[-last], gc[-1], sym))
    return out


def momentum_holdings(start, end) -> Dict:
    """Date -> names the shipped Top-6 momentum book held (cached closes)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qbs.config import MomentumParams
    from qbs.strategies import cross_sectional_momentum
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    u = pd.read_csv(os.path.join(root, "data/universe/universe_prices.csv"),
                    index_col=0, parse_dates=True)
    b = pd.read_csv(os.path.join(root, "data/BOXX.csv"), index_col=0,
                    parse_dates=True).iloc[:, 0].reindex(u.index).ffill().bfill()
    u = u.loc[:, u.notna().sum() >= MomentumParams().min_history]
    sig = cross_sectional_momentum(u, b, MomentumParams())
    # The book decided at yesterday's close is what is held today.
    log = sig.holdings_log
    dates = sorted(log)
    return {dates[i + 1]: set(log[dates[i]]) for i in range(len(dates) - 1)}


def table(rows: List[Dict]) -> str:
    df = pd.DataFrame(rows)
    f = {"Win": "{:.0%}", "Avg gross (bp)": "{:+.1f}", "Avg net (bp)": "{:+.1f}",
         "t(net)": "{:+.2f}", "Sleeve ann.": "{:+.1%}", "Trades/yr": "{:.0f}"}
    for c, s in f.items():
        if c in df:
            df[c] = df[c].map(lambda v, s=s: "—" if v != v else s.format(v))
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder of IB JSON bar files")
    ap.add_argument("--cost-bps", type=float, default=2.0,
                    help="per side: commission + half-spread (default 2)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    parts = []

    # ---- daily QQQ ------------------------------------------------------
    d = load(os.path.join(a.data, "QQQ_1d.json"))
    d.index = d.index.normalize()
    yrs = len(d) / 252
    rows = []
    bh = d["close"].iloc[-1] / d["close"].iloc[0]
    for name, tr in daily_rules(d).items():
        rows.append(dict(Strategy=name, **summarise(tr, a.cost_bps, 1, yrs)))
    parts.append(f"## QQQ daily bars, {d.index[0].date()} to {d.index[-1].date()} "
                 f"({yrs:.1f} years; buy & hold {bh ** (1 / yrs) - 1:+.1%} a year)\n")
    parts.append(table(rows) + "\n")

    # ---- hourly, QQQ + stocks -------------------------------------------
    files = sorted(glob.glob(os.path.join(a.data, "*_1h.json")))
    bars = {os.path.basename(f).split("_")[0]: load(f) for f in files}
    first = min(b.index[0] for b in bars.values())
    last = max(b.index[-1] for b in bars.values())
    yrs_h = len(bars["QQQ"].index.normalize().unique()) / 252
    try:
        held = momentum_holdings(first, last)
    except Exception as exc:               # noqa: BLE001
        print("momentum holdings unavailable:", exc)
        held = {}
    agg: Dict[str, List[Trade]] = {}
    qqq_only: Dict[str, List[Trade]] = {}
    for sym, b in bars.items():
        res = intraday_rules(sym, b, 7, held)
        for k, v in res.items():
            (qqq_only if sym == "QQQ" else agg).setdefault(k, []).extend(v)
    q = bars["QQQ"]
    qbh = (q["close"].iloc[-1] / q["open"].iloc[0]) ** (1 / yrs_h) - 1
    parts.append(f"## Hourly bars, {first.date()} to {last.date()} "
                 f"({yrs_h:.2f} years; QQQ buy & hold {qbh:+.1%} a year)\n")
    parts.append("### QQQ alone (one slot)\n")
    parts.append(table([dict(Strategy=k, **summarise(v, a.cost_bps, 1, yrs_h))
                        for k, v in qqq_only.items()
                        if k != "RSI2 < 10 dip in a momentum-book name"]) + "\n")
    parts.append(f"### {len(bars) - 1} stocks (six slots)\n")
    held_end = max(held) if held else None
    parts.append(table([dict(Strategy=k, **summarise(v, a.cost_bps, 6, yrs_h))
                        for k, v in agg.items()]) + "\n")
    if held_end is not None:
        parts.append(f"Momentum-book holdings come from the cached closes, which "
                     f"end {held_end.date()}; that rule only trades up to then.\n")

    # ---- 15-minute QQQ cross-check --------------------------------------
    f15 = os.path.join(a.data, "QQQ_15m.json")
    if os.path.exists(f15):
        q15 = load(f15)
        yrs15 = len(q15.index.normalize().unique()) / 252
        res = intraday_rules("QQQ", q15, 26, {})
        parts.append(f"## Cross-check: QQQ 15-minute bars, {q15.index[0].date()} "
                     f"to {q15.index[-1].date()} ({len(q15.index.normalize().unique())} days)\n")
        parts.append(table([dict(Strategy=k, **summarise(v, a.cost_bps, 1, yrs15))
                            for k, v in res.items()
                            if k != "RSI2 < 10 dip in a momentum-book name"]) + "\n")

    text = "\n".join(parts)
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
