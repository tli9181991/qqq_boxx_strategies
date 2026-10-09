#!/usr/bin/env python3
"""Backtest the auto-selecting swing book (`qbs.swing_book`) against the
Top-6 momentum book and QQQ, all at $1,000 a slot with real trade costs.

    python scripts/swing_book_study.py --scans data/swing_scans.pkl   # reuse scans
    python scripts/swing_book_study.py --build-scans                  # ~10 min, 4 cores

The swing screens are re-run as of every session on prices up to that
session only (`--build-scans`, cached to a pickle), so no entry sees the
future. Windows are the lab's default (2025-01-20 on) and the extended one
(2022-01-03 on, a 0% cash proxy before BOXX listed), as in the other studies.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qbs.config import MomentumParams  # noqa: E402
from qbs.strategies import cross_sectional_momentum  # noqa: E402
from qbs.swing_book import (SwingBookParams, Trade, commission,  # noqa: E402
                            run_swing_book, trade_table)

WINDOWS = {"default": "2025-01-20", "extended": "2022-01-03"}
END = None
HIST = 400


def load():
    u = pd.read_csv(os.path.join(ROOT, "data/universe/universe_prices.csv"),
                    index_col=0, parse_dates=True)
    q = pd.read_csv(os.path.join(ROOT, "data/QQQ.csv"), index_col=0,
                    parse_dates=True).iloc[:, 0]
    b = pd.read_csv(os.path.join(ROOT, "data/BOXX.csv"), index_col=0,
                    parse_dates=True).iloc[:, 0]
    b = b.reindex(u.index)
    b.loc[:b.first_valid_index()] = b.loc[b.first_valid_index()]  # 0% before listing
    return u, q, b.ffill()


def _scan_one(args):
    d, u, q = args
    from qbs.swing import scan
    out = scan(u.loc[:d].tail(HIST), q.loc[:d])
    return d, {k: v for k, v in out.items() if not v.empty}


def build_scans(u, q, start, path):
    from multiprocessing import Pool
    dates = [d for d in u.index if d >= pd.Timestamp(start)]
    res = {}
    with Pool(os.cpu_count() or 2) as pool:
        for i, (d, r) in enumerate(pool.imap_unordered(
                _scan_one, [(d, u, q) for d in dates], chunksize=4)):
            res[d] = r
            if i % 200 == 0:
                print(f"  scans {i}/{len(dates)}", flush=True)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return res


def momentum_trades(u, safe, start, p: SwingBookParams):
    """The Top-6 momentum book traded like the swing book: $1,000 a slot,
    whole shares, IB commission. Buys when a name joins, sells when it
    leaves; no stop or target."""
    sig = cross_sectional_momentum(u, safe, MomentumParams())
    log = sig.holdings_log
    safe_ret = safe.pct_change().fillna(0.0)
    slip = p.slippage_bps / 1e4
    cash = p.n_slots * p.slot_usd
    pos, eq, trades = {}, {}, []
    for d in u.index[u.index >= pd.Timestamp(start)]:
        cash *= 1 + float(safe_ret.get(d, 0.0))
        row = u.loc[d]
        want = set(log.get(d, []))
        for t in list(pos):
            if t not in want:
                tr = pos.pop(t)
                c = float(row[t])
                fee = commission(tr.shares, c, p)
                cash += tr.shares * c * (1 - slip) - fee
                tr.costs += fee + tr.shares * c * slip
                tr.exit_date, tr.exit_px, tr.reason = d, c, "rank"
                tr.held = int(((u.index > tr.entry_date) & (u.index <= d)).sum())
                trades.append(tr)
        for t in want:
            if t in pos or len(pos) >= p.n_slots:
                continue
            c = float(row[t])
            fill = c * (1 + slip)
            sh = (math.floor(min(p.slot_usd, cash) / fill) if p.whole_shares
                  else min(p.slot_usd, cash) / fill)
            if sh <= 0:
                continue
            fee = commission(sh, fill, p)
            cash -= sh * fill + fee
            pos[t] = Trade(t, "momentum", d, c, sh, None, None, 10**6,
                           costs=fee + sh * c * slip)
        eq[d] = cash + sum(tr.shares * float(row[t]) for t, tr in pos.items())
    for tr in pos.values():
        tr.reason = "open"
        trades.append(tr)
    return pd.Series(eq), trades


def stats(eq: pd.Series, trades, capital: float) -> dict:
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (eq.iloc[-1] / capital) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    tt = trade_table(trades)
    closed = tt[tt["reason"] != "open"] if not tt.empty else tt
    traded = float((tt["entry_px"] * tt["shares"]).sum() * 2) if not tt.empty else 0.0
    return {
        "CAGR": cagr, "Vol": float(r.std() * np.sqrt(252)), "MaxDD": dd,
        "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
        "Trades/yr": len(closed) / yrs if yrs else np.nan,
        "Win rate": float((closed["pnl"] > 0).mean()) if len(closed) else np.nan,
        "Avg trade": float(closed["ret"].mean()) if len(closed) else np.nan,
        "Avg hold": float(closed["held"].mean()) if len(closed) else np.nan,
        "Costs/yr": float(tt["costs"].sum() / capital / yrs) if len(tt) else 0.0,
        "Turnover": traded / capital / yrs if yrs else np.nan,
    }


def fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("CAGR", "Vol", "MaxDD", "Win rate", "Avg trade", "Costs/yr"):
        if c in out:
            out[c] = out[c].map(lambda v: "—" if v != v else f"{v:.1%}")
    for c in ("Calmar",):
        out[c] = out[c].map(lambda v: "—" if v != v else f"{v:.2f}")
    for c in ("Trades/yr", "Avg hold"):
        out[c] = out[c].map(lambda v: "—" if v != v else f"{v:.0f}")
    out["Turnover"] = out["Turnover"].map(lambda v: f"{v:.1f}x")
    try:
        return out.to_markdown(index=False)
    except ImportError:
        return out.to_string(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scans", default=os.path.join(ROOT, "data", "swing_scans.pkl"))
    ap.add_argument("--build-scans", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    u, q, safe = load()
    u = u.loc[:, u.notna().sum() >= MomentumParams().min_history]
    if a.build_scans or not os.path.exists(a.scans):
        print("building swing scans (once; cached to", a.scans, ")")
        scans = build_scans(u, q, "2021-12-01", a.scans)
    else:
        with open(a.scans, "rb") as f:
            scans = pickle.load(f)

    look, skip = 126, 21
    mom = u.shift(skip) / u.shift(look) - 1.0

    base = SwingBookParams()
    variants = {
        "swing: all setups, best R:R first": base,
        "swing: all setups, best momentum first": replace(base, rank_by="momentum"),
        "swing: R:R >= 2 only": replace(base, min_rr=2.0, need_target=True),
        "swing: fractional shares": replace(base, whole_shares=False),
        "swing: stops 2x wider": replace(base, stop_mult=2.0),
        "swing: $3k slots": replace(base, slot_usd=3000.0),
        "swing: NO costs (signal only)": replace(
            base, whole_shares=False, commission_min=0.0,
            commission_per_share=0.0, slippage_bps=0.0),
        "swing, momentum first: stops 2x wider": replace(
            base, rank_by="momentum", stop_mult=2.0),
        "swing, momentum first: NO costs": replace(
            base, rank_by="momentum", whole_shares=False, commission_min=0.0,
            commission_per_share=0.0, slippage_bps=0.0),
    }
    for key in base.setups:
        variants[f"swing: {key} only"] = replace(base, setups=(key,))

    parts = []
    all_trades = {}
    for window, start in WINDOWS.items():
        capital = base.n_slots * base.slot_usd
        rows = []
        for name, p in variants.items():
            eq, tr = run_swing_book(u, safe, scans, p, start=start, momentum=mom)
            rows.append(dict(Book=name, **stats(eq, tr, p.n_slots * p.slot_usd)))
            if name == "swing: all setups, best R:R first":
                all_trades[window] = trade_table(tr)
        eq, tr = momentum_trades(u, safe, start, base)
        rows.append(dict(Book="Top-6 momentum, $1k slots", **stats(eq, tr, capital)))
        qq = q.reindex(u.index).ffill().loc[start:]
        qeq = capital * qq / qq.iloc[0]
        rows.append(dict(Book="QQQ buy & hold", **stats(qeq, [], capital)))
        parts.append(f"## {window} window ({start} to {u.index[-1].date()})\n")
        parts.append(fmt(pd.DataFrame(rows)) + "\n")

        tt = all_trades[window]
        closed = tt[tt["reason"] != "open"]
        by = closed.groupby("reason").agg(n=("ret", "size"), avg=("ret", "mean"),
                                          hold=("held", "mean"))
        parts.append(f"### {window}: how trades end (all setups, best R:R first)\n")
        parts.append(by.to_string(formatters={"avg": "{:+.1%}".format,
                                              "hold": "{:.1f}".format}) + "\n")

    text = "\n".join(parts)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
