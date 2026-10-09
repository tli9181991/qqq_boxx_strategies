#!/usr/bin/env python3
"""Two-year backtest of the Swing trades tab's six setups on REAL daily bars.

    python scripts/swing_ohlc_study.py --ib path/to/ib_daily_json

`scripts/swing_book_study.py` ran the swing book on the lab's cached closes,
where a day's range is the close-to-close envelope and a stop is only seen
at the close. This runs the same book on real daily Open/High/Low/Close from
IB (one `SYMBOL_1d.json` per name, collected with the IB connector), so

* the screens size their ATR, stops and level merges from true daily ranges
  (`qbs.swing.scan(..., ohlc=...)`, as the tab does with Source -> Online), and
* stops and targets fill intraday (`run_swing_book(..., ohlc=...)`): at the
  stop when the low reaches it, at the open on a gap through it.

Same universe in every arm -- the names with IB bars -- so the comparison
isolates the data, not the stock list:

  A. closes only     scans on closes, exits on the close (the earlier study)
  B. real bars       scans on OHLC,   exits intraday
  C. real scans      scans on OHLC,   exits on the close (separates the two)

IB's daily bars are split-adjusted but not dividend-adjusted; each day's
Open/High/Low is scaled by that day's cached-close / IB-close ratio, so all
prices sit on the lab's adjusted basis and the closes are identical in every
arm. Window: the two years to the cache's last session.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import replace
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from qbs.swing_book import SwingBookParams, run_swing_book, trade_table  # noqa: E402
from swing_book_study import fmt, load, momentum_trades, stats  # noqa: E402

HIST = 400


def ib_ohlc(path: str, closes: pd.Series) -> pd.DataFrame:
    d = json.load(open(path))
    idx = (pd.to_datetime(d["time"]).tz_convert("America/New_York")
           .normalize().tz_localize(None))
    raw = pd.DataFrame({k.capitalize(): d[k] for k in ("open", "high", "low", "close")},
                       index=idx).astype(float)
    raw = raw[~raw.index.duplicated()].reindex(closes.index).dropna()
    ratio = closes.reindex(raw.index) / raw["Close"]
    out = raw[["Open", "High", "Low"]].mul(ratio, axis=0)
    out["Close"] = closes.reindex(raw.index)
    # A bar whose scaled range does not contain its close is a bad print.
    out["High"] = out[["High", "Close", "Open"]].max(axis=1)
    out["Low"] = out[["Low", "Close", "Open"]].min(axis=1)
    return out


_U = _Q = _O = None


def _init(u, q, o):
    global _U, _Q, _O
    _U, _Q, _O = u, q, o


def _scan(d):
    from qbs.swing import scan
    c = _U.loc[:d].tail(HIST)
    out = scan(c, _Q.loc[:d], ohlc=_O)
    return d, {k: v for k, v in out.items() if not v.empty}


def build_scans(u, q, ohlc, dates):
    res = {}
    with Pool(os.cpu_count() or 2, initializer=_init, initargs=(u, q, ohlc)) as pool:
        for i, (d, r) in enumerate(pool.imap_unordered(_scan, dates, chunksize=4)):
            res[d] = r
            if i % 100 == 0:
                print(f"  scans {i}/{len(dates)}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ib", required=True, help="folder of SYMBOL_1d.json files")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--cache", default=None, help="pickle the scans here (reused)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    u_all, q, safe = load()
    names = sorted(f[:-8] for f in os.listdir(a.ib) if f.endswith("_1d.json")
                   and f[:-8] in u_all.columns)
    u = u_all[names]
    ohlc = {t: ib_ohlc(os.path.join(a.ib, f"{t}_1d.json"), u[t].dropna()) for t in names}
    end = u.index[-1]
    start = u.index[u.index >= end - pd.DateOffset(years=a.years)][0]
    dates = [d for d in u.index if d >= start]
    print(f"{len(names)} names with IB bars; window {start.date()} to {end.date()}")

    if a.cache and os.path.exists(a.cache):
        scans_c, scans_o = pickle.load(open(a.cache, "rb"))
    else:
        print("scanning on closes ...")
        scans_c = build_scans(u, q, None, dates)
        print("scanning on real bars ...")
        scans_o = build_scans(u, q, ohlc, dates)
        if a.cache:
            pickle.dump((scans_c, scans_o), open(a.cache, "wb"))

    mom = u.shift(21) / u.shift(126) - 1.0
    base = SwingBookParams()
    variants = {
        "all setups, best R:R first": base,
        "all setups, best momentum first": replace(base, rank_by="momentum"),
        "momentum first, stops 2x wider": replace(base, rank_by="momentum", stop_mult=2.0),
    }
    for key in base.setups:
        variants[f"{key} only"] = replace(base, setups=(key,))
    arms = {
        "A. closes only": (scans_c, None),
        "B. real bars (scans + intraday exits)": (scans_o, ohlc),
        "C. real-bar scans, exits on close": (scans_o, None),
    }

    capital = base.n_slots * base.slot_usd
    rows, how = [], []
    for arm, (scans, bars) in arms.items():
        for name, p in variants.items():
            eq, tr = run_swing_book(u, safe, scans, p, start=str(start.date()),
                                    momentum=mom, ohlc=bars)
            rows.append(dict(Arm=arm, Book=name, **stats(eq, tr, capital)))
            if name == "all setups, best R:R first":
                tt = trade_table(tr)
                closed = tt[tt["reason"] != "open"]
                for reason, g in closed.groupby("reason"):
                    how.append(dict(Arm=arm, Exit=reason, Trades=len(g),
                                    Avg=f"{g['ret'].mean():+.1%}",
                                    Hold=f"{g['held'].mean():.1f}"))
    eq, tr = momentum_trades(u, safe, str(start.date()), base)
    bench = [dict(Book=f"Top-6 momentum on these {len(names)} names, $1k slots",
                  **stats(eq, tr, capital))]
    qq = q.reindex(u.index).ffill().loc[start:]
    bench.append(dict(Book="QQQ buy & hold", **stats(capital * qq / qq.iloc[0], [], capital)))

    parts = [f"## Swing book on real daily bars: {len(names)} names, "
             f"{start.date()} to {end.date()}\n",
             "Names: " + ", ".join(names) + "\n"]
    df = pd.DataFrame(rows)
    for arm in arms:
        parts.append(f"### {arm}\n")
        parts.append(fmt(df[df["Arm"] == arm].drop(columns="Arm")) + "\n")
    parts.append("### Benchmarks, same window\n")
    parts.append(fmt(pd.DataFrame(bench)) + "\n")
    parts.append("### How trades end (all setups, best R:R first)\n")
    try:
        parts.append(pd.DataFrame(how).to_markdown(index=False) + "\n")
    except ImportError:
        parts.append(pd.DataFrame(how).to_string(index=False) + "\n")
    text = "\n".join(parts)
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
