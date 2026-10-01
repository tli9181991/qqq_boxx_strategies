#!/usr/bin/env python3
"""Sandbox: the live Top-6 momentum book vs the same book with a per-name exit.

    python scripts/exit_drop_sim.py                       # after preflight
    python scripts/exit_drop_sim.py --since 2026-10-01    # paper trial start
    python scripts/exit_drop_sim.py --drop 8 --days 30

Run it after preflight. It reads the price cache preflight just refreshed
(`data/`, mounted into the container as /app/data) and never touches the
network, the broker, `var/` or any live setting. Nothing it computes reaches
an order.

Two books, both six momentum picks in equal slots with no vol scaling --
the "momentum6" line of the live comparison:

  fixed band   the live rule: sold once its rank passes `exit_rank` (8)
  exit drop    sold once its rank passes the rank it was bought at + --drop

It prints today's holdings side by side, each book's net return since
`--since` (or over the last `--days` sessions), and the full-history
backtest for context. Costs are the configured commission + slippage on
turnover, the same as the live comparison.

Names excluded from the live ranking (QBS_EXCLUDE_TICKERS) can be passed with
--exclude so the sandbox ranks the same universe preflight did.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.config import Config  # noqa: E402
from qbs.engine import run_backtest  # noqa: E402
from qbs.live.signals import load_live_prices  # noqa: E402
from qbs.metrics import summarise  # noqa: E402
from qbs.strategies import cross_sectional_momentum  # noqa: E402
from qbs.universe import load_universe  # noqa: E402


def net_returns(sig, combined: pd.DataFrame, cfg: Config) -> pd.Series:
    """Daily net model return, computed the way the live comparison does."""
    w = sig.weights.reindex(combined.index).ffill().shift(cfg.execution_lag).fillna(0.0)
    rets = combined.pct_change().fillna(0.0)
    cols = [c for c in w.columns if c in rets.columns]
    gross = (w[cols] * rets[cols]).sum(axis=1)
    cost = w.diff().abs().sum(axis=1) * (cfg.cost_bps + cfg.slippage_bps) / 1e4
    return gross - cost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop", type=int, default=8,
                    help="sell once rank > entry rank + this (default 8)")
    ap.add_argument("--since", default=None,
                    help="paper-trial start date; returns are compounded from here")
    ap.add_argument("--days", type=int, default=21,
                    help="without --since, compound the last N sessions (default 21)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated names preflight excludes from ranking")
    a = ap.parse_args(argv)

    cfg = Config()
    safe = cfg.momentum.safe_asset
    tickers = load_universe(fetch=False, warn=False)
    px = load_live_prices(cfg, tickers=tickers, fetch_universe=False,
                          refresh=False, offline=True)
    excluded = {t.strip().upper() for t in a.exclude.split(",") if t.strip()}
    names = [t for t in tickers if t in px.columns and t not in excluded]
    uni = px[names]
    uni = uni.loc[:, uni.notna().sum() >= cfg.momentum.min_history]
    combined = uni.copy()
    combined[safe] = px[safe]
    asof = uni.index[-1]

    books = {
        f"fixed band (exit rank {cfg.momentum.exit_rank})": cfg.momentum,
        f"exit drop (entry rank + {a.drop})": replace(cfg.momentum, exit_drop=a.drop),
    }
    sigs = {k: cross_sectional_momentum(uni, px[safe], p) for k, p in books.items()}

    print(f"prices: {len(uni.columns)} names, last bar {asof:%Y-%m-%d} "
          f"(read from the cache preflight refreshed; offline)\n")

    print(f"HOLDINGS ON {asof:%Y-%m-%d}")
    held = {k: list((s.holdings_log or {}).get(asof, [])) for k, s in sigs.items()}
    ranks = {k: (s.held_ranks or {}).get(asof, {}) for k, s in sigs.items()}
    for k in sigs:
        cells = [f"{t} (#{ranks[k].get(t, float('nan')):.0f})" for t in held[k]]
        print(f"  {k:<32} {', '.join(cells) or '(all BOXX)'}")
    a_set, b_set = (set(v) for v in held.values())
    if a_set != b_set:
        k1, k2 = list(held)
        print(f"  only in {k1.split(' (')[0]}: {', '.join(sorted(a_set - b_set)) or '-'}")
        print(f"  only in {k2.split(' (')[0]}: {', '.join(sorted(b_set - a_set)) or '-'}")
    else:
        print("  same six names today")
    print()

    rets = {k: net_returns(s, combined, cfg).loc[:asof] for k, s in sigs.items()}
    if a.since:
        window = {k: r.loc[pd.Timestamp(a.since):] for k, r in rets.items()}
        label = f"since {a.since}"
    else:
        window = {k: r.tail(a.days) for k, r in rets.items()}
        label = f"last {a.days} sessions"
    first = next(iter(window.values()))
    print(f"NET RETURN, {label} ({len(first)} sessions, "
          f"{first.index[0]:%Y-%m-%d} to {first.index[-1]:%Y-%m-%d})")
    out = {k: float((1 + r).prod() - 1) for k, r in window.items()}
    for k, v in out.items():
        print(f"  {k:<32} {v:+8.2%}")
    k1, k2 = list(out)
    edge = out[k2] - out[k1]
    print(f"  exit drop vs fixed band: {edge:+.2%} "
          f"({'exit drop' if edge > 0 else 'fixed band'} ahead)\n")

    print(f"FULL HISTORY BACKTEST ({cfg.backtest_start} to {asof:%Y-%m-%d}, for context)")
    print(f"  {'':<32} {'CAGR':>7} {'Vol':>7} {'Max DD':>8} {'Turnover':>9}")
    for k, s in sigs.items():
        res = run_backtest(combined, s, start=cfg.backtest_start, lag=cfg.execution_lag,
                           cost_bps=cfg.cost_bps, slippage_bps=cfg.slippage_bps)
        m = summarise(res)
        print(f"  {k:<32} {m['CAGR']:>7.1%} {m['Ann. vol']:>7.1%} "
              f"{m['Max drawdown']:>8.1%} {m['Ann. turnover']:>8.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
