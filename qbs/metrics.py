"""Performance statistics.

Risk-adjusted numbers are computed against BOXX's *realised* return, not
against zero. Over a window where cash paid ~4-5%, a Sharpe ratio measured
against zero flatters everything and flatters the defensive strategies most,
because a lot of what they earn is simply the cash rate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .engine import BacktestResult
from .indicators import drawdown


def _annualise_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    total = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1.0 / years) - 1.0


def summarise(
    result: BacktestResult,
    rf: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """One row of statistics for a single backtest.

    `rf` is a daily risk-free return series (BOXX's own daily return). If it
    is None, excess return is measured against zero and the Sharpe is a
    total-return Sharpe -- flattering, and labelled as such by the caller.
    """
    r = result.returns.dropna()
    if r.empty:
        return {}

    rf_aligned = (rf.reindex(r.index).fillna(0.0) if rf is not None
                  else pd.Series(0.0, index=r.index))
    excess = r - rf_aligned

    ann_ret = _annualise_return(r)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (float(excess.mean()) * TRADING_DAYS / ann_vol) if ann_vol > 0 else np.nan

    downside = excess.clip(upper=0.0)
    dd_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS))
    sortino = (float(excess.mean()) * TRADING_DAYS / dd_dev) if dd_dev > 0 else np.nan

    dd = drawdown(result.equity)
    max_dd = float(dd.min())

    # Time actually exposed to something other than the safe asset.
    risky_cols = [c for c in result.weights.columns if c.upper() != "BOXX"]
    risky_w = result.weights[risky_cols].sum(axis=1) if risky_cols else pd.Series(0.0, index=r.index)

    return {
        "Total return": float((1.0 + r).prod() - 1.0),
        "CAGR": ann_ret,
        "Ann. vol": ann_vol,
        "Sharpe (vs BOXX)": sharpe,
        "Sortino": sortino,
        "Max drawdown": max_dd,
        "Calmar": (ann_ret / abs(max_dd)) if max_dd < 0 else np.nan,
        "Best day": float(r.max()),
        "Worst day": float(r.min()),
        "Hit rate": float((r > 0).mean()),
        "Avg risk exposure": float(risky_w.mean()),
        "Days in risk": float((risky_w > 0.01).mean()),
        "Ann. turnover": float(result.turnover.sum() / (len(r) / TRADING_DAYS)),
        "Cost drag (ann.)": float(result.costs.sum() / (len(r) / TRADING_DAYS)),
        "of which slippage": (float(result.slippage.sum() / (len(r) / TRADING_DAYS))
                              if result.slippage is not None else 0.0),
    }


def summary_table(
    results: Dict[str, BacktestResult],
    rf: Optional[pd.Series] = None,
    labels: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    rows = {}
    for key, res in results.items():
        label = (labels or {}).get(key, key)
        rows[label] = summarise(res, rf=rf)
    df = pd.DataFrame(rows).T
    return df


PERCENT_ROWS = [
    "Total return", "CAGR", "Ann. vol", "Max drawdown", "Best day",
    "Worst day", "Hit rate", "Avg risk exposure", "Days in risk",
    "Cost drag (ann.)", "of which slippage",
]


def format_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable copy of `summary_table` -- percentages as percentages."""
    out = df.copy()
    for col in out.columns:
        if col in PERCENT_ROWS:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif col == "Ann. turnover":
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.1f}x")
        else:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return out


def monthly_returns(result: BacktestResult) -> pd.Series:
    return (1.0 + result.returns).resample("ME").prod() - 1.0


def monthly_return_matrix(result: BacktestResult) -> pd.DataFrame:
    """Year x month grid of returns, for the heatmap."""
    m = monthly_returns(result)
    df = pd.DataFrame({
        "year": m.index.year,
        "month": m.index.month,
        "ret": m.values,
    })
    return df.pivot(index="year", columns="month", values="ret")


def rolling_sharpe(
    result: BacktestResult,
    window: int = 63,
    rf: Optional[pd.Series] = None,
) -> pd.Series:
    r = result.returns
    rf_aligned = rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0
    excess = r - rf_aligned
    mean = excess.rolling(window, min_periods=window).mean() * TRADING_DAYS
    vol = r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return mean / vol.replace(0.0, np.nan)


def holdings_runs(result: BacktestResult) -> pd.DataFrame:
    """Collapse the daily holdings log into one row per continuous holding.

    Each row is "this name was held from entry to exit" -- which is what a
    tenure chart plots, and a far more useful export than 427 rows of daily
    membership. Feed the CSV to the holdings-timeline viewer.
    """
    sig = result.signals
    log = getattr(sig, "holdings_log", None) if sig is not None else None
    if not log:
        return pd.DataFrame()

    log = {d: v for d, v in log.items() if result.start <= d <= result.end}
    dates = sorted(log)
    if not dates:
        return pd.DataFrame()

    # Entry rank and momentum, parsed back out of the event reasons.
    import re
    entry_meta = {}
    if sig.events is not None and not sig.events.empty:
        for _, row in sig.events[sig.events["action"] == "buy"].iterrows():
            m = re.search(r"rank (\d+), 12-1 mom ([+-][\d.]+)%", str(row["reason"]))
            if m:
                entry_meta[(row["asset"], pd.Timestamp(row["date"]))] = (
                    int(m.group(1)), float(m.group(2)))

    rows = []
    for t in sorted({n for v in log.values() for n in v}):
        on, start_dt, prev = False, None, None
        spans = []
        for d in dates:
            here = t in log[d]
            if here and not on:
                on, start_dt = True, d
            elif not here and on:
                on = False
                spans.append((start_dt, prev, False))
            if here:
                prev = d
        if on:
            spans.append((start_dt, dates[-1], True))

        for s, e, still in spans:
            rank, mom = entry_meta.get((t, s), (np.nan, np.nan))
            rows.append(dict(
                ticker=t, entry=s, exit=e,
                trading_days=sum(1 for d in dates if s <= d <= e),
                still_open=still, entry_rank=rank, entry_mom=mom,
            ))

    return pd.DataFrame(rows).sort_values(["entry", "ticker"]).reset_index(drop=True)


def trade_log(result: BacktestResult) -> pd.DataFrame:
    """Round-trip trades reconstructed from the held-weight path of the risk asset."""
    sig = result.signals
    if sig is None or sig.events is None or sig.events.empty:
        return pd.DataFrame()
    ev = sig.events.copy()
    ev = ev[(ev["date"] >= result.start) & (ev["date"] <= result.end)]
    if ev.empty:
        return pd.DataFrame()

    trades = []
    open_row = None
    for _, row in ev.iterrows():
        if row["action"] == "buy" and open_row is None:
            open_row = row
        elif row["action"] == "sell" and open_row is not None:
            trades.append(dict(
                entry=open_row["date"], exit=row["date"], asset=row["asset"],
                entry_px=open_row["price"], exit_px=row["price"],
                ret=row["price"] / open_row["price"] - 1.0,
                days=int(np.busday_count(open_row["date"].date(), row["date"].date())),
                exit_reason=row["reason"],
            ))
            open_row = None
    if open_row is not None:
        trades.append(dict(
            entry=open_row["date"], exit=pd.NaT, asset=open_row["asset"],
            entry_px=open_row["price"], exit_px=np.nan, ret=np.nan,
            days=np.nan, exit_reason="still open",
        ))
    return pd.DataFrame(trades)
