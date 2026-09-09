#!/usr/bin/env python3
"""Command-line runner: fetch, backtest, print the table, save the charts.

    python run_backtest.py                        # live data via yfinance
    python run_backtest.py --synthetic            # no network needed
    python run_backtest.py --start 2024-09-01 --target-vol 0.12
    python run_backtest.py --no-charts --csv out/
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=None, help="backtest start (default: 2025-01-20)")
    p.add_argument("--end", default=None, help="backtest end (default: today)")
    p.add_argument("--download-start", default=None,
                   help="how far back to pull data for indicator warm-up")
    p.add_argument("--cost-bps", type=float, default=None, help="commission per 100%% turnover")
    p.add_argument("--slippage-bps", type=float, default=None,
                   help="15:30 decision vs 16:00 MOC fill, per 100%% turnover")
    p.add_argument("--target-vol", type=float, default=None, help="e.g. 0.15 for 15%%")
    p.add_argument("--rsi-entry", type=float, default=None, help="RSI(2) buy threshold")
    p.add_argument("--max-leverage", type=float, default=None,
                   help="cap on the vol-targeted weight (1.0 = never lever)")
    p.add_argument("--n-hold", type=int, default=None, help="momentum: names to hold")
    p.add_argument("--exit-rank", type=int, default=None,
                   help="momentum: hysteresis band -- sell once rank passes this")
    p.add_argument("--rebalance", default=None, choices=["daily", "ME", "W-FRI"],
                   help="momentum rebalance frequency")
    p.add_argument("--no-momentum", dest="momentum", action="store_false", default=True,
                   help="skip the Nasdaq-100 strategy (avoids the ~100-ticker download)")
    p.add_argument("--no-fetch-universe", dest="fetch_universe", action="store_false",
                   default=True, help="use the hardcoded NDX list instead of scraping")
    p.add_argument("--pit-membership", default=None,
                   help="CSV of date,ticker point-in-time index membership")
    p.add_argument("--sweep-band", action="store_true",
                   help="also print the hysteresis-band sensitivity table")
    p.add_argument("--vix-exit", type=float, default=None,
                   help="VIX close above this switches the book to cash")
    p.add_argument("--vix-entry", type=float, default=None,
                   help="VIX below this resumes the base strategy")
    p.add_argument("--vix-park-after", type=int, default=None,
                   help="days in cash before converting to the safe asset")
    p.add_argument("--vix-min-cash", type=int, default=None,
                   help="minimum days in cash before re-entry is allowed")
    p.add_argument("--no-vix", dest="vix", action="store_false", default=True,
                   help="skip the VIX circuit-breaker strategy")
    p.add_argument("--sweep-vix", action="store_true",
                   help="also print the VIX trigger-level sensitivity table")
    p.add_argument("--target-vol-book", type=float, default=None,
                   help="annualised vol target for the whole momentum book "
                        "(default 0.25; this is the drawdown control that works)")
    p.add_argument("--no-book-vt", dest="book_vt", action="store_false", default=True,
                   help="skip the vol-targeted variant of the momentum book")
    p.add_argument("--sweep-target-vol", action="store_true",
                   help="also print the book vol-target sensitivity table")
    p.add_argument("--synthetic", action="store_true", help="use generated prices, no network")
    p.add_argument("--offline", action="store_true", help="use only the CSV cache")
    p.add_argument("--refresh", action="store_true", help="re-download, ignoring the cache")
    p.add_argument("--charts", dest="charts", action="store_true", default=True)
    p.add_argument("--no-charts", dest="charts", action="store_false")
    p.add_argument("--outdir", default="output", help="where charts and CSVs are written")
    p.add_argument("--csv", action="store_true", help="also write equity curves and trades to CSV")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from qbs.config import BACKTEST_START, Config, STRATEGY_LABELS
    from qbs.metrics import trade_log
    from qbs.pipeline import run, sweep_band
    from qbs.universe import load_pit_universe

    cfg = Config()
    if args.start:
        cfg.backtest_start = args.start
    if args.end:
        cfg.backtest_end = args.end
    if args.download_start:
        cfg.download_start = args.download_start
    if args.cost_bps is not None:
        cfg.cost_bps = args.cost_bps
    if args.slippage_bps is not None:
        cfg.slippage_bps = args.slippage_bps
    if args.target_vol is not None:
        cfg.vol.target_vol = args.target_vol
    if args.max_leverage is not None:
        cfg.vol.max_weight = args.max_leverage
    if args.rsi_entry is not None:
        cfg.rsi2.entry_threshold = args.rsi_entry
    if args.n_hold is not None:
        cfg.momentum.n_hold = args.n_hold
    if args.exit_rank is not None:
        cfg.momentum.exit_rank = args.exit_rank
    if args.rebalance is not None:
        cfg.momentum.rebalance = args.rebalance
    if args.vix_exit is not None:
        cfg.vix.exit_level = args.vix_exit
    if args.vix_entry is not None:
        cfg.vix.entry_level = args.vix_entry
    if args.vix_park_after is not None:
        cfg.vix.park_after_days = args.vix_park_after
    if args.vix_min_cash is not None:
        cfg.vix.min_cash_days = args.vix_min_cash
    if args.target_vol_book is not None:
        cfg.book_vol.target_vol = args.target_vol_book
    if cfg.momentum.exit_rank < cfg.momentum.n_hold:
        print("exit-rank must be >= n-hold (the band cannot be negative)", file=sys.stderr)
        return 2
    if cfg.vix.entry_level > cfg.vix.exit_level:
        print("vix-entry must be <= vix-exit (the band cannot be inverted)", file=sys.stderr)
        return 2

    pit = load_pit_universe(args.pit_membership) if args.pit_membership else None

    try:
        lab = run(cfg, offline=args.offline, refresh=args.refresh,
                  use_synthetic=args.synthetic, with_momentum=args.momentum,
                  with_vix=args.vix, with_book_vt=args.book_vt,
                  fetch_universe=args.fetch_universe, pit_membership=pit)
    except ImportError:
        print("yfinance is not installed. Either `pip install yfinance` or run "
              "with --synthetic to exercise the pipeline without a data feed.",
              file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Could not build the backtest: {exc}", file=sys.stderr)
        return 1

    first = next(iter(lab.results.values()))
    banner = (f"QQQ / BOXX strategy lab — {first.start:%Y-%m-%d} to {first.end:%Y-%m-%d}"
              f"{'  [SYNTHETIC DATA]' if args.synthetic else ''}")
    print("\n" + banner)
    print("=" * len(banner) + "\n")

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(lab.summary_pretty.to_string())

    tl = trade_log(lab.results["rsi2"])
    if not tl.empty:
        print(f"\nRSI(2) round trips: {len(tl)}   "
              f"win rate {(tl['ret'] > 0).mean():.0%}   "
              f"avg {tl['ret'].mean():+.2%}   "
              f"avg hold {tl['days'].mean():.1f} trading days")

    if "momentum" in lab.signals:
        log = lab.signals["momentum"].holdings_log or {}
        res = lab.results["momentum"]
        inwin = {d: v for d, v in log.items() if res.start <= d <= res.end}
        if inwin:
            last = max(inwin)
            print(f"\nMomentum holdings as of {last:%Y-%m-%d}: {', '.join(inwin[last])}")
            print(f"  distinct names touched: {len({t for v in inwin.values() for t in v})}"
                  f"   band: top {cfg.momentum.n_hold}, exit past rank {cfg.momentum.exit_rank}")

    if "momentum_vix" in lab.signals:
        d = lab.signals["momentum_vix"].diagnostics
        rv = lab.results["momentum_vix"]
        d = d.loc[rv.start:rv.end]
        ev = lab.signals["momentum_vix"].events
        pct_in = float((d["regime"] == "INVESTED").mean())
        print(f"\nVIX breaker (exit {cfg.vix.exit_level:g} / re-enter {cfg.vix.entry_level:g}): "
              f"invested {pct_in:.0%} of the window, "
              f"{int((ev['action'] == 'sell').sum())} trips to cash, "
              f"{int((ev['action'] == 'park').sum())} parked in {cfg.vix.safe_asset}")
        if pct_in < 0.6:
            print(f"  ** the trigger sits inside the VIX distribution -- this is a "
                  f"mostly-out-of-market strategy, not a crash filter. Run --sweep-vix. **")

    if "momentum_vt" in lab.signals:
        d = lab.signals["momentum_vt"].diagnostics
        rv = lab.results["momentum_vt"]
        d = d.loc[rv.start:rv.end]
        base_dd = lab.results["momentum"].drawdown.min()
        print(f"\nBook vol target {cfg.book_vol.target_vol:.0%}: "
              f"avg book weight {d['risk_weight'].mean():.0%} "
              f"(book vol averaged {d['book_vol'].mean():.0%}), "
              f"max drawdown {rv.drawdown.min():.1%} vs {base_dd:.1%} unscaled")

    if args.sweep_target_vol and "momentum_vt" in lab.signals:
        from qbs.pipeline import sweep_target_vol as _sweep_tv
        stv = _sweep_tv(lab)
        print("\nBook vol-target sensitivity (expect a smooth dial, not a peak):")
        with pd.option_context("display.width", 200):
            print(stv.round(4).to_string(index=False))

    if args.sweep_vix and "momentum_vix" in lab.signals:
        from qbs.pipeline import sweep_vix as _sweep_vix
        svx = _sweep_vix(lab)
        print("\nVIX trigger sensitivity (read 'Time invested' first):")
        with pd.option_context("display.width", 200):
            print(svx.round(4).to_string(index=False))

    if args.sweep_band and "momentum" in lab.signals:
        sw = sweep_band(lab, n_holds=[cfg.momentum.n_hold],
                        exit_ranks=[6, 8, 10, 12, 15, 20, 25, 30])
        print("\nHysteresis-band sensitivity (look for a plateau, not a peak):")
        with pd.option_context("display.width", 200):
            print(sw.round(4).to_string(index=False))

    if args.charts or args.csv:
        os.makedirs(args.outdir, exist_ok=True)

    if args.charts:
        import matplotlib
        matplotlib.use("Agg")
        from qbs import plotting as P
        P.use_style()
        s, e = first.start, first.end
        figs = {
            "01_equity": P.plot_equity_curves(lab.results).figure,
            "02_drawdown": P.plot_drawdowns(lab.results).figure,
            "03_rsi2_signals": P.plot_rsi2_signals(lab.signals["rsi2"], s, e),
            "04_gem_signals": P.plot_gem_signals(lab.prices, lab.signals["gem"], s, e),
            "05_voltarget_signals": P.plot_voltarget_signals(
                lab.signals["voltarget"], cfg.vol.target_vol, s, e),
            "06_risk_return": P.plot_return_scatter(lab.results, rf=lab.rf).figure,
        }
        if "momentum" in lab.signals:
            figs["07_momentum_holdings"] = P.plot_momentum_holdings(
                lab.signals["momentum"], s, e)
            if args.sweep_band:
                figs["08_band_sensitivity"] = P.plot_band_sensitivity(
                    sw, n_hold=cfg.momentum.n_hold)
        if "momentum_vix" in lab.signals:
            figs["09_vix_regimes"] = P.plot_vix_regimes(
                lab.signals["momentum_vix"], lab.results, s, e)
            if args.sweep_vix:
                figs["10_vix_sweep"] = P.plot_vix_sweep(
                    svx, baseline_cagr=lab.summary.loc[
                        STRATEGY_LABELS["momentum"], "CAGR"])
        if "momentum_vt" in lab.signals:
            figs["11_book_voltarget"] = P.plot_book_voltarget(
                lab.signals["momentum_vt"], cfg.book_vol.target_vol, s, e, vix=lab.vix)
            if args.sweep_target_vol:
                figs["12_target_vol_sweep"] = P.plot_target_vol_sweep(
                    stv, baseline=lab.summary.loc[STRATEGY_LABELS["momentum"]].to_dict())
        for name, fig in figs.items():
            path = os.path.join(args.outdir, f"{name}.png")
            fig.savefig(path, bbox_inches="tight", dpi=140)
        print(f"\nCharts written to {os.path.abspath(args.outdir)}/")

    if args.csv:
        from qbs.engine import equity_frame
        from qbs.metrics import holdings_runs
        equity_frame(lab.results).to_csv(os.path.join(args.outdir, "equity_curves.csv"))
        lab.summary.to_csv(os.path.join(args.outdir, "summary.csv"))
        if not tl.empty:
            tl.to_csv(os.path.join(args.outdir, "rsi2_trades.csv"), index=False)
        if "momentum" in lab.results:
            hr = holdings_runs(lab.results["momentum"])
            if not hr.empty:
                hr.to_csv(os.path.join(args.outdir, "momentum_holdings.csv"), index=False)
                print(f"  momentum_holdings.csv -- drop this into the holdings timeline viewer")
        print(f"CSVs written to {os.path.abspath(args.outdir)}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
