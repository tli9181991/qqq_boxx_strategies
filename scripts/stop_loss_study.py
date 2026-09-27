#!/usr/bin/env python3
"""Reproduce the tables in docs/STOP_LOSS.md.

    python scripts/stop_loss_study.py              # both windows + paired grid
    python scripts/stop_loss_study.py --quick      # skip the paired grid

Runs offline from the CSV cache. Two windows:

  default   2025-01-20 onward, the lab's reporting window, BOXX as cash.
  extended  2022-01-03 onward, which adds the 2022 bear market -- the one
            episode a stop is actually for. BOXX only lists from 2022-12-28,
            so before that the cash leg is a FLAT 0% proxy. T-bills paid
            roughly 0-2% over those months, so this understates the cash leg
            slightly and flatters nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import Config, SAFE_ASSET, StopLossParams  # noqa: E402
from qbs.pipeline import run, sweep_stops  # noqa: E402

EXT_START = "2022-01-03"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def extended_prices(cfg: Config) -> pd.DataFrame:
    # Read the caches directly: load_prices aligns on the common calendar,
    # which would cut everything back to BOXX's listing date.
    cols = {t: _read_cache(t) for t in cfg.tickers}
    px = pd.DataFrame(cols).dropna(subset=["QQQ"]).sort_index()
    first = px[SAFE_ASSET].first_valid_index()
    px.loc[:first, SAFE_ASSET] = px.at[first, SAFE_ASSET]    # 0% before listing
    return px.ffill().dropna()


def _read_cache(ticker: str) -> pd.Series:
    path = os.path.join(DATA, ticker.replace("^", "_") + ".csv")
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


def lab_for(window: str):
    cfg = Config()
    kw = dict(offline=True, fetch_universe=False, with_finviz=False, with_vix=False,
              with_book_vt=False)
    if window == "extended":
        cfg = replace(cfg, backtest_start=EXT_START, download_start="2020-06-01")
        return run(cfg, prices=extended_prices(cfg), **kw)
    return run(cfg, **kw)


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:          # tabulate is optional
        return df.to_string(index=False)


def fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in out.columns:
        if c in ("CAGR", "Ann. vol", "Max drawdown", "CAGR vs none",
                 "Ann. vol vs none", "Max drawdown vs none", "Avg exposure"):
            out[c] = out[c].map(lambda v: f"{v:+.1%}" if "vs" in c else f"{v:.1%}")
        elif c in ("Ann. turnover", "Ann. turnover vs none"):
            out[c] = out[c].map(lambda v: f"{v:.1f}x")
        elif c in ("Sharpe", "Calmar", "Calmar vs none"):
            out[c] = out[c].map(lambda v: f"{v:.2f}")
    return _md(out)


# The candidates carried into the paired test: each family at a middle and an
# edge distance, plus the book-level stops.
PAIRED = [
    StopLossParams(kind="fixed", stop_pct=0.10),
    StopLossParams(kind="fixed", stop_pct=0.20),
    StopLossParams(kind="trailing", stop_pct=0.10),
    StopLossParams(kind="trailing", stop_pct=0.20),
    StopLossParams(kind="chandelier", atr_mult=3.0),
    StopLossParams(kind="chandelier", atr_mult=6.0),
    StopLossParams(kind="residual", resid_mult=1.0),
    StopLossParams(kind="residual", resid_mult=2.0),
]
CELLS = [(4, 6), (4, 8), (6, 8), (6, 10), (6, 12), (8, 10), (8, 12), (8, 14)]


def paired(lab, start) -> pd.DataFrame:
    """Each stop against its own no-stop book, cell by cell."""
    rows = []
    base_cfg = lab.config
    for n, x in CELLS:
        lab.config = replace(
            base_cfg,
            momentum=replace(base_cfg.momentum, n_hold=n, exit_rank=x),
            resmom=replace(base_cfg.resmom, n_hold=n, exit_rank=x),
        )
        df = sweep_stops(lab, variants=[StopLossParams()] + PAIRED,
                         book_stops=(0.13, 0.20), start=start)
        df["cell"] = f"({n},{x})"
        rows.append(df)
    lab.config = base_cfg
    df = pd.concat(rows)
    df = df[df["Stop"] != "no stop"]
    g = df.groupby(["Book", "Stop"], sort=False)
    return pd.DataFrame({
        "Cells": g.size(),
        "Lower max DD": g.apply(lambda d: f"{(d['Max drawdown vs none'] > 0).sum()}/{len(d)}"),
        "Median ΔMaxDD": g["Max drawdown vs none"].median().map(lambda v: f"{v:+.1%}"),
        "Higher Calmar": g.apply(lambda d: f"{(d['Calmar vs none'] > 0).sum()}/{len(d)}"),
        "Median ΔCAGR": g["CAGR vs none"].median().map(lambda v: f"{v:+.1%}"),
        "Median ΔVol": g["Ann. vol vs none"].median().map(lambda v: f"{v:+.1%}"),
        "Median Δturnover": g["Ann. turnover vs none"].median().map(lambda v: f"{v:+.1f}x"),
    }).reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the paired grid")
    ap.add_argument("--out", default=None, help="also write the markdown here")
    a = ap.parse_args(argv)

    parts = []
    for window in ("default", "extended"):
        lab = lab_for(window)
        start = lab.config.backtest_start
        df = sweep_stops(lab)
        cols = ["Book", "Stop", "CAGR", "Ann. vol", "Sharpe", "Max drawdown",
                "Calmar", "Ann. turnover", "Stops"]
        parts.append(f"## {window} window ({start} to {lab.combined.index[-1].date()})\n")
        parts.append(fmt(df[cols]) + "\n")

        # Cooldown and refill: does it matter what happens after the stop?
        extra = []
        for kind_kw in (dict(kind="trailing", stop_pct=0.10),
                        dict(kind="residual", resid_mult=2.0)):
            for cd in (5, 21, 63):
                for refill in (True, False):
                    extra.append(StopLossParams(cooldown_days=cd, refill=refill, **kind_kw))
        dc = sweep_stops(lab, variants=[StopLossParams()] + extra, book_stops=())
        parts.append(f"### {window}: cooldown and refill\n")
        parts.append(fmt(dc[cols]) + "\n")

        if not a.quick:
            parts.append(f"### {window}: paired across {len(CELLS)} (n_hold, exit_rank) cells\n")
            parts.append(_md(paired(lab, start)) + "\n")

    text = "\n".join(parts)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
