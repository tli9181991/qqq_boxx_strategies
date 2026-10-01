#!/usr/bin/env python3
"""Which six to hold from the momentum top-20? Reproduces docs/TOP20_SELECTION.md.

    python scripts/top20_selection_study.py
    python scripts/top20_selection_study.py --out docs_tables.md

The Top-6 book holds the six highest 6-1 momentum names. Here the same
ranking (same 6-1 score, same "beat BOXX" filter) only defines a POOL -- the
top N, 20 by default -- and a second rule picks the six held from inside it.
A held name is sold when it leaves the pool, or (with a band) when it falls
past `exit` on the second rule. Everything else -- slot weighting, the cash
leg, costs, the one-day lag -- is the shipped book's, because the second rule
is passed to `cross_sectional_momentum` as its `score`.

Windows and the 0% cash proxy before BOXX listed are those of
`stop_loss_study.py`; read its docstring.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import MomentumParams, ResidualMomentumParams, SAFE_ASSET  # noqa: E402
from qbs.engine import run_backtest  # noqa: E402
from qbs.metrics import summarise  # noqa: E402
from qbs.strategies import cross_sectional_momentum, residual_momentum_score  # noqa: E402
from stop_loss_study import lab_for  # noqa: E402

RULES = {
    "top 6 by momentum": "mom",
    "ranks 7+ by momentum": "skip6",
    "lowest 60d vol": "lowvol",
    "lowest beta to QQQ": "lowbeta",
    "best residual momentum": "resid",
    "nearest 52w high": "near_high",
    "strongest last month": "recent",
}


def momentum_rank(uni: pd.DataFrame, safe: pd.Series, p: MomentumParams) -> tuple:
    """The shipped ranking, rebuilt: 6-1 momentum among names that beat BOXX."""
    look, skip = int(round(p.lookback_months * 21)), int(round(p.skip_months * 21))
    mom = uni.shift(skip) / uni.shift(look) - 1.0
    s = safe.reindex(uni.index).ffill()
    safe_mom = s.shift(skip) / s.shift(look) - 1.0
    ok = mom.notna() & (uni.notna().cumsum() >= p.min_history)
    if p.absolute_filter:
        ok &= mom.gt(safe_mom, axis=0)
    return mom, mom.where(ok).rank(axis=1, ascending=False)


def secondary(key: str, uni: pd.DataFrame, mom: pd.DataFrame, rank: pd.DataFrame,
              qqq: pd.Series) -> pd.DataFrame:
    """Higher is better. Only the pool mask is applied by the caller."""
    r = uni.pct_change()
    if key in ("mom", "skip6"):
        if key == "mom":
            return mom
        # Ranks 1-6 sort LAST rather than vanish: never bought, but a held
        # name that climbs into the top 6 is kept, not sold for its strength.
        return mom.where(rank > 6, -1e9)
    if key == "lowvol":
        return -r.rolling(60, min_periods=30).std()
    if key == "lowbeta":
        rm = qqq.reindex(uni.index).ffill().pct_change()
        beta = r.rolling(252, min_periods=126).cov(rm).div(
            rm.rolling(252, min_periods=126).var(), axis=0)
        return -beta
    if key == "resid":
        return residual_momentum_score(uni, qqq, ResidualMomentumParams())
    if key == "near_high":
        return uni / uni.rolling(252, min_periods=126).max()
    if key == "recent":
        return uni / uni.shift(21) - 1.0
    raise ValueError(key)


def run_rule(lab, key: str, pool: int, exit_rank: int, n_hold: int = 6):
    cfg = lab.config
    uni = lab.combined.drop(columns=[SAFE_ASSET])
    safe, qqq = lab.prices[SAFE_ASSET], lab.prices["QQQ"]
    mom, rank = momentum_rank(uni, safe, cfg.momentum)
    score = secondary(key, uni, mom, rank, qqq).where(rank <= pool)
    p = MomentumParams(**{**cfg.momentum.__dict__, "n_hold": n_hold,
                          "exit_rank": max(exit_rank, n_hold)})
    sig = cross_sectional_momentum(uni, safe, p, score=score, name=key)
    res = run_backtest(lab.combined, sig, start=cfg.backtest_start,
                       end=cfg.backtest_end, lag=cfg.execution_lag,
                       cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps)
    s = summarise(res, rf=lab.rf)
    return dict(CAGR=s["CAGR"], Vol=s["Ann. vol"], Sharpe=s["Sharpe (vs BOXX)"],
                MaxDD=s["Max drawdown"], Calmar=s["Calmar"],
                Turnover=s["Ann. turnover"], Exposure=s["Avg risk exposure"])


def baseline(lab):
    res = lab.results["momentum"]
    s = summarise(res, rf=lab.rf)
    return dict(CAGR=s["CAGR"], Vol=s["Ann. vol"], Sharpe=s["Sharpe (vs BOXX)"],
                MaxDD=s["Max drawdown"], Calmar=s["Calmar"],
                Turnover=s["Ann. turnover"], Exposure=s["Avg risk exposure"])


def fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("CAGR", "Vol", "MaxDD", "Exposure"):
        if c in out:
            out[c] = out[c].map(lambda v: f"{v:.1%}")
    for c in ("Sharpe", "Calmar"):
        if c in out:
            out[c] = out[c].map(lambda v: f"{v:.2f}")
    if "Turnover" in out:
        out["Turnover"] = out["Turnover"].map(lambda v: f"{v:.1f}x")
    try:
        return out.to_markdown(index=False)
    except ImportError:
        return out.to_string(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    parts, grid = [], []
    for window in ("default", "extended"):
        lab = lab_for(window)
        start = lab.config.backtest_start
        rows = [dict(Rule="shipped Top-6 (exit rank 8)", Pool="-", Exit="8",
                     **baseline(lab))]
        rows.append(dict(Rule="top-20 equal weight (20 slots)", Pool="20",
                         Exit="20", **run_rule(lab, "mom", 20, 20, n_hold=20)))
        for label, key in RULES.items():
            for ex in (20, 10):
                rows.append(dict(Rule=label, Pool="20", Exit=str(ex),
                                 **run_rule(lab, key, 20, ex)))
        df = pd.DataFrame(rows)
        parts.append(f"## {window} window ({start} to {lab.combined.index[-1].date()})\n")
        parts.append("Pool = top 20 by 6-1 momentum. Exit 20 = sold only on "
                     "leaving the pool; exit 10 = also sold past 10th on the rule.\n")
        parts.append(fmt(df) + "\n")

        # Robustness: the same rules at other pool sizes, exit = pool.
        for pool in (15, 20, 30):
            for label, key in RULES.items():
                r = run_rule(lab, key, pool, pool)
                grid.append(dict(Window=window, Pool=pool, Rule=label, **r))

    g = pd.DataFrame(grid)
    base = g[g["Rule"] == "top 6 by momentum"].set_index(["Window", "Pool"])
    g = g.join(base[["Calmar", "CAGR", "Turnover"]], on=["Window", "Pool"],
               rsuffix="_ref")
    summary = g.groupby("Rule", sort=False).apply(lambda d: pd.Series({
        "Cells": len(d),
        "Calmar > top-6 in pool": f"{(d['Calmar'] > d['Calmar_ref']).sum()}/{len(d)}",
        "Median CAGR": f"{d['CAGR'].median():.1%}",
        "Median Calmar": f"{d['Calmar'].median():.2f}",
        "Median MaxDD": f"{d['MaxDD'].median():.1%}",
        "Median turnover": f"{d['Turnover'].median():.1f}x",
    })).reset_index()
    parts.append("## Robustness: pools of 15 / 20 / 30, both windows (exit = pool)\n")
    parts.append(summary.to_markdown(index=False) + "\n")
    parts.append("### every cell\n")
    parts.append(fmt(g.drop(columns=["Calmar_ref", "CAGR_ref", "Turnover_ref"])) + "\n")

    text = "\n".join(parts)
    print(text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
