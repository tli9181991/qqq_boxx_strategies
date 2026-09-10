#!/usr/bin/env python3
"""Head-to-head: the notebook's trend-template screen vs 12-1 momentum.

    python compare_selection.py --offline        # from the cached CSVs
    python compare_selection.py                  # live download
    python compare_selection.py --benchmark VEU  # how much does the benchmark matter?

Both strategies run through the *same* engine, on the same universe, over the
same window, with the same execution lag and the same commission and slippage.
The only thing that differs is how names are chosen -- which is the point.

Read the output in this order:

1. The summary table tells you which produced more return, and at what risk.
2. The deployment split tells you whether a difference in return came from
   picking different names or from being in the market a different amount of
   the time. On this sample it is almost entirely the latter, and that changes
   what the headline numbers mean.
3. The episode drawdowns show where each one earned its keep.
4. The criterion counts show which of the screen's filters actually binds.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from qbs.config import Config, SAFE_ASSET
from qbs.engine import run_backtest
from qbs.metrics import summarise
from qbs.pipeline import run
from qbs.screens import TrendScreenParams, trend_template_screen
from qbs.strategies import book_vol_target

# The two drawdown episodes the earlier analysis identified. They are different
# in kind: one is a market-wide selloff, the other a concentration blow-up in a
# calm index, and the two strategies handle them very differently.
EPISODES = {
    "2025 market selloff": ("2025-02-19", "2025-08-05"),
    "2026 concentration": ("2026-07-01", None),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--offline", action="store_true", help="use only the CSV cache")
    p.add_argument("--synthetic", action="store_true", help="generated prices, no network")
    p.add_argument("--benchmark", default="QQQ",
                   help="the index the screen's relative-strength test uses "
                        "(default QQQ; the notebook used ^GSPC)")
    p.add_argument("--n-hold", type=int, default=6,
                   help="names the screen holds; 0 = every name that passes")
    p.add_argument("--rebalance", default="daily", choices=["daily", "ME", "W-FRI"])
    p.add_argument("--no-fetch-universe", dest="fetch_universe", action="store_false",
                   default=True, help="use the hardcoded NDX list instead of scraping")
    p.add_argument("--csv", default=None, help="also write the summary table here")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()

    try:
        lab = run(cfg, offline=args.offline, use_synthetic=args.synthetic,
                  fetch_universe=args.fetch_universe)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not build the backtest: {exc}", file=sys.stderr)
        return 1

    uni = lab.combined.drop(columns=[SAFE_ASSET])
    safe = lab.prices[SAFE_ASSET]
    book = lab.combined
    if args.benchmark not in lab.prices.columns:
        print(f"benchmark {args.benchmark} not in {list(lab.prices.columns)}",
              file=sys.stderr)
        return 2
    bench = lab.prices[args.benchmark]

    def bt(sig):
        return run_backtest(book, sig, start=cfg.backtest_start, end=cfg.backtest_end,
                            lag=cfg.execution_lag, cost_bps=cfg.cost_bps,
                            slippage_bps=cfg.slippage_bps)

    def row(label, res):
        s = summarise(res, rf=lab.rf)
        return {"Strategy": label, "CAGR": s["CAGR"], "Vol": s["Ann. vol"],
                "Sharpe": s["Sharpe (vs BOXX)"], "MaxDD": s["Max drawdown"],
                "Calmar": s["Calmar"], "Turnover": s["Ann. turnover"],
                "Cost drag": s["Cost drag (ann.)"], "Exposure": s["Avg risk exposure"]}

    # ---- build both books -------------------------------------------------
    sp = TrendScreenParams(n_hold=args.n_hold, rebalance=args.rebalance)
    screen = trend_template_screen(uni, bench, safe, sp, name="screen")
    r_screen = bt(screen)
    r_mom = lab.results["momentum"]
    r_qqq = lab.results["bh_qqq"]

    screen_vt = bt(book_vol_target(screen, book, cfg.book_vol, lag=cfg.execution_lag))

    rows = [
        row(f"Notebook screen, top {args.n_hold or 'all'} (vs {args.benchmark})", r_screen),
        row("Notebook screen + vol target 25%", screen_vt),
        row("Top-6 12-1 momentum", r_mom),
        row("Top-6 vol-targeted 25%", lab.results["momentum_vt"]),
        row("Buy & hold QQQ", r_qqq),
        row("Buy & hold BOXX", lab.results["bh_boxx"]),
    ]
    df = pd.DataFrame(rows)

    banner = (f"Selection comparison — {r_mom.start:%Y-%m-%d} to {r_mom.end:%Y-%m-%d}"
              f"{'  [SYNTHETIC]' if args.synthetic else ''}")
    print("\n" + banner)
    print("=" * len(banner) + "\n")
    fmt = {"CAGR": "{:.1%}".format, "Vol": "{:.1%}".format, "Sharpe": "{:.2f}".format,
           "MaxDD": "{:.1%}".format, "Calmar": "{:.2f}".format,
           "Turnover": "{:.1f}x".format, "Cost drag": "{:.2%}".format,
           "Exposure": "{:.0%}".format}
    print(df.to_string(index=False, formatters=fmt))

    # ---- 2. selection quality vs deployment -------------------------------
    # The headline gap could be either. Splitting returns by whether the screen
    # was deployed separates them, and the answer decides what to fix.
    risky = r_screen.weights.drop(columns=[SAFE_ASSET]).sum(axis=1)
    inv = risky > 0.01
    rs = r_screen.returns
    rm = r_mom.returns.reindex(rs.index)
    rq = r_qqq.returns.reindex(rs.index)

    print(f"\n--- Was the difference stock picking, or time in the market? ---")
    print(f"The screen was invested on {inv.sum()} of {len(inv)} days ({inv.mean():.0%}).\n")
    print(f"  {'':<12}{'screen-invested days':>22}{'screen-cash days':>20}")
    for lbl, r in (("screen", rs), ("momentum", rm), (args.benchmark, rq)):
        on = (1 + r[inv]).prod() - 1
        off = (1 + r[~inv]).prod() - 1
        print(f"  {lbl:<12}{on:>22.1%}{off:>20.1%}")
    print("\n  If the two strategies earn about the same on invested days, the gap is\n"
          "  deployment, not selection -- and widening the screen would help more\n"
          "  than changing what it ranks on.")

    # ---- 3. drawdown by episode -------------------------------------------
    def dd(res, a, b):
        e = res.equity.loc[a:b]
        return float((e / e.cummax() - 1).min()) if len(e) else float("nan")

    print(f"\n--- Max drawdown within each episode ---")
    print(f"  {'episode':<24}{'screen':>10}{'momentum':>11}{args.benchmark:>9}")
    for k, (a, b) in EPISODES.items():
        print(f"  {k:<24}{dd(r_screen, a, b):>10.1%}"
              f"{dd(r_mom, a, b):>11.1%}{dd(r_qqq, a, b):>9.1%}")

    # ---- 4. which criterion binds -----------------------------------------
    d = screen.diagnostics.loc[r_screen.start:r_screen.end]
    n_uni = uni.shape[1]
    print(f"\n--- How selective is the screen? (mean names/day of {n_uni}) ---")
    print(f"  stage 2 (> rising SMA150 and SMA200)   {d['n_stage2'].mean():>6.1f}")
    print(f"  relative strength vs {args.benchmark:<18} {d['n_rs_ok'].mean():>6.1f}")
    print(f"  within 25% of the 52-week high         {d['n_within_high'].mean():>6.1f}")
    print(f"  all criteria together                  {d['n_passing'].mean():>6.1f}")
    print(f"\n  days with nothing passing (fully in {SAFE_ASSET}): "
          f"{(d['n_passing'] == 0).sum()} of {len(d)} ({(d['n_passing'] == 0).mean():.0%})")
    if not screen.params.get("sector_filter_applied"):
        print(f"  NOTE: the sector-ETF criterion is not applied (no sector data supplied),\n"
              f"        so the screen here is MORE permissive than the notebook's.")

    last = d.index[-1]
    print(f"\n--- Holdings on {last:%Y-%m-%d} ---")
    print(f"  screen  : {', '.join(screen.holdings_log[last]) or '(all cash)'}")
    print(f"  momentum: {', '.join(lab.signals['momentum'].holdings_log[last])}")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nsummary written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
