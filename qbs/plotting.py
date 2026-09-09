"""Charts.

Design rules applied throughout, so the figures read as one system:

* **One y-axis per panel, never two.** Where two quantities of different scale
  belong together (price and RSI; price, vol and exposure) they get stacked
  panels sharing an x-axis instead of a twinned axis. A dual-axis chart lets
  the author choose the story by choosing the scaling; stacked panels do not.
* **Colour-vision-safe categorical palette**, three slots, validated all-pairs.
  Benchmarks sit outside that set in neutral grey so they never compete with a
  strategy for identity.
* **Identity is never colour alone.** Every multi-series chart carries a legend
  *and* direct end-labels; buy/sell markers differ in shape as well as hue.
* **Recessive chrome.** Hairline grid, no top/right spines, muted tick labels --
  the data is the darkest thing on the page.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .config import PALETTE, STRATEGY_LABELS
from .engine import BacktestResult
from .metrics import monthly_return_matrix, rolling_sharpe
from .strategies import StrategySignals

# Dates worth marking on a 2025-2026 chart.
REGIME_MARKERS = {
    "Inauguration": "2025-01-20",
}


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

def use_style() -> None:
    """Apply the chart style. Call once per notebook."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.labelcolor": PALETTE["ink_2"],
        "axes.edgecolor": PALETTE["axis"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "figure.dpi": 110,
    })


def _color(key: str) -> str:
    return PALETTE.get(key, PALETTE["muted"])


def _label(key: str) -> str:
    return STRATEGY_LABELS.get(key, key)


def _tidy_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_xlabel("")


def _direct_label(ax, x, y, text, color, dx=6) -> None:
    """End-of-line label. Required relief for the palette's lower-contrast slot,
    and it removes the legend round-trip for the reader."""
    ax.annotate(
        text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
        color=color, fontsize=9, fontweight="600",
        va="center", ha="left", annotation_clip=False,
    )


def _direct_labels_spread(ax, items: Sequence[tuple], x, dx=6, min_gap_frac=0.045) -> None:
    """Place several end-of-line labels without letting them overlap.

    `items` is a sequence of (y, text, color). Two strategies that finish within
    a whisker of each other -- which is exactly the interesting case -- would
    otherwise print their labels on top of one another. Nudge them apart in
    data space, keeping the original order, and accept the small offset: the
    line itself still carries the exact value.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    ordered = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in ordered]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    # If the stack overshot the top, slide the whole run back down.
    overshoot = ys[-1] - hi
    if overshoot > 0:
        ys = [y - overshoot for y in ys]
    for (y_orig, text, color), y in zip(ordered, ys):
        ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="600",
                    va="center", ha="left", annotation_clip=False)


def _mark_regimes(ax, markers: Optional[Dict[str, str]] = REGIME_MARKERS) -> None:
    """Vertical rules for dates worth naming. Labelled along the bottom, where
    a chart's legend is not."""
    if not markers:
        return
    for label, date in markers.items():
        d = pd.Timestamp(date)
        lo, hi = ax.get_xlim()
        if not (lo <= mdates.date2num(d) <= hi):
            continue
        ax.axvline(d, color=PALETTE["axis"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        ax.annotate(label, xy=(d, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 6), textcoords="offset points",
                    color=PALETTE["muted"], fontsize=8, ha="left", va="bottom")


# --------------------------------------------------------------------------
# Comparison charts
# --------------------------------------------------------------------------

def plot_equity_curves(
    results: Dict[str, BacktestResult],
    title: str = "Growth of $1",
    ax=None,
    log: bool = False,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.2))

    curves = {k: r.equity / r.equity.iloc[0] for k, r in results.items()}
    # Draw benchmarks first so strategies sit on top of them.
    order = ([k for k in curves if k.startswith("bh_")] +
             [k for k in curves if not k.startswith("bh_")])

    end_labels = []
    for key in order:
        s = curves[key]
        is_bench = key.startswith("bh_")
        ax.plot(s.index, s.values, color=_color(key),
                linewidth=1.6 if is_bench else 2.2,
                linestyle=(0, (5, 2)) if is_bench else "-",
                label=_label(key), zorder=2 if is_bench else 3)
        end_labels.append((float(s.iloc[-1]), f"{_label(key)}  {s.iloc[-1]:.2f}x", _color(key)))

    ax.set_title(title)
    ax.set_ylabel("Value of $1 invested")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}x"))
    if log:
        ax.set_yscale("log")
    ax.axhline(1.0, color=PALETTE["axis"], linewidth=0.8, zorder=1)
    _tidy_dates(ax)
    _mark_regimes(ax)
    ax.legend(loc="upper left", ncols=2)
    ax.margins(x=0.02)
    _direct_labels_spread(ax, end_labels, x=list(curves.values())[0].index[-1])
    ax.get_figure().subplots_adjust(right=0.76)
    return ax


def plot_drawdowns(results: Dict[str, BacktestResult], title: str = "Drawdown from running peak", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.0))

    for key, res in results.items():
        dd = res.drawdown
        is_bench = key.startswith("bh_")
        ax.plot(dd.index, dd.values, color=_color(key),
                linewidth=1.4 if is_bench else 2.0,
                linestyle=(0, (5, 2)) if is_bench else "-",
                label=_label(key))
        if not is_bench:
            ax.fill_between(dd.index, dd.values, 0, color=_color(key), alpha=0.10, linewidth=0)

    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    _tidy_dates(ax)
    _mark_regimes(ax)
    ax.legend(loc="lower left", ncols=2)
    ax.margins(x=0.02)
    return ax


def plot_rolling_sharpe(
    results: Dict[str, BacktestResult],
    window: int = 63,
    rf: Optional[pd.Series] = None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.8))
    for key, res in results.items():
        if key.startswith("bh_boxx"):
            continue  # a near-riskless series has a meaningless rolling Sharpe
        rs = rolling_sharpe(res, window=window, rf=rf).dropna()
        if rs.empty:
            continue
        ax.plot(rs.index, rs.values, color=_color(key), label=_label(key),
                linewidth=1.4 if key.startswith("bh_") else 2.0,
                linestyle=(0, (5, 2)) if key.startswith("bh_") else "-")
    ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax.set_title(f"Rolling {window}-day Sharpe (excess of BOXX)")
    ax.set_ylabel("Sharpe")
    _tidy_dates(ax)
    ax.legend(loc="upper left", ncols=3)
    ax.margins(x=0.02)
    return ax


def plot_exposure(result: BacktestResult, title: Optional[str] = None, ax=None,
                  max_series: int = 6):
    """Stacked area of held weights -- what the book actually owned, day by day.

    A wide book (the 100-name momentum universe) is collapsed to
    "risk assets vs safe asset": a hundred stacked bands would be an unreadable
    smear, and the question this chart answers is how invested the book was,
    not which specific names.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.0))
    w = result.weights.loc[:, (result.weights.abs().sum() > 0)]

    if w.shape[1] > max_series:
        safe = [c for c in w.columns if c.upper() == "BOXX"]
        risk = [c for c in w.columns if c not in safe]
        w = pd.DataFrame({"Risk assets": w[risk].sum(axis=1),
                          "BOXX": w[safe].sum(axis=1) if safe else 0.0},
                         index=w.index)

    asset_colors = {"QQQ": PALETTE["rsi2"], "VEU": PALETTE["gem"],
                    "BOXX": PALETTE["voltarget"], "Risk assets": PALETTE["momentum"]}
    cols = [c for c in ["QQQ", "VEU", "Risk assets", "BOXX"] if c in w.columns] + \
           [c for c in w.columns if c not in ("QQQ", "VEU", "BOXX", "Risk assets")]
    ax.stackplot(w.index, *[w[c].values for c in cols],
                 labels=cols,
                 colors=[asset_colors.get(c, PALETTE["muted"]) for c in cols],
                 alpha=0.55, edgecolor=PALETTE["surface"], linewidth=0.5)
    ax.set_ylim(0, max(1.0, float(w.sum(axis=1).max())) * 1.02)
    ax.set_title(title or f"Holdings — {_label(result.name)}")
    ax.set_ylabel("Weight")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", visible=False)
    _tidy_dates(ax)
    ax.legend(loc="lower left", ncols=len(cols))
    ax.margins(x=0)
    return ax


# --------------------------------------------------------------------------
# Signal charts -- one per strategy
# --------------------------------------------------------------------------

def _plot_trade_markers(ax, events: pd.DataFrame, lo=None, hi=None) -> None:
    if events is None or events.empty:
        return
    ev = events.copy()
    if lo is not None:
        ev = ev[ev["date"] >= pd.Timestamp(lo)]
    if hi is not None:
        ev = ev[ev["date"] <= pd.Timestamp(hi)]
    for action, marker, color, offset in (
        ("buy", "^", PALETTE["buy"], -0.9),
        ("sell", "v", PALETTE["sell"], 0.9),
    ):
        sub = ev[ev["action"] == action]
        if sub.empty:
            continue
        span = ax.get_ylim()
        pad = (span[1] - span[0]) * 0.02
        ax.scatter(sub["date"], sub["price"] + offset * pad,
                   marker=marker, s=64, color=color,
                   edgecolor=PALETTE["surface"], linewidth=1.2, zorder=6,
                   label=f"{action.capitalize()} signal")


def plot_rsi2_signals(
    signals: StrategySignals,
    start=None,
    end=None,
    title: str = "Connors RSI(2) — entries and exits",
):
    """Price with the trend filter and trade markers, RSI(2) beneath.

    The two panels share an x-axis rather than sharing one y-axis with two
    scales, so a spike in RSI can never be made to look like a move in price.
    """
    d = signals.diagnostics.copy()
    if start is not None:
        d = d.loc[pd.Timestamp(start):]
    if end is not None:
        d = d.loc[:pd.Timestamp(end)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True,
        gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.12),
    )

    trend_col = [c for c in d.columns if c.startswith("sma") and c != "sma5"]
    ax1.plot(d.index, d["close"], color=PALETTE["ink"], linewidth=1.6, label="QQQ close", zorder=4)
    if trend_col:
        ax1.plot(d.index, d[trend_col[0]], color=PALETTE["muted"], linewidth=1.4,
                 linestyle=(0, (5, 2)), label=trend_col[0].upper(), zorder=3)
    if "sma5" in d.columns:
        ax1.plot(d.index, d["sma5"], color=PALETTE["rsi2"], linewidth=1.2,
                 alpha=0.9, label="SMA5 (exit)", zorder=3)

    # Shade the stretches where the trend filter blocks any new long. Anchored
    # to the axes in y, so the band always spans the full panel height no
    # matter how the price limits end up.
    blocked = (d["above_trend"] < 0.5).to_numpy()
    ax1.fill_between(d.index, 0, 1, where=blocked,
                     transform=ax1.get_xaxis_transform(),
                     color=PALETTE["shade_safe"], zorder=0, linewidth=0,
                     label="Below SMA200 — no new longs")

    _plot_trade_markers(ax1, signals.events, d.index.min(), d.index.max())
    ax1.set_title(title)
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="lower left", ncols=3)
    _mark_regimes(ax1)

    ax2.plot(d.index, d["rsi"], color=PALETTE["rsi2"], linewidth=1.4, label="RSI(2)")
    entry = signals.params.get("entry_threshold", 5)
    exit_rsi = signals.params.get("exit_rsi", 70)
    ax2.axhline(entry, color=PALETTE["buy"], linewidth=1.0, linestyle=(0, (3, 3)))
    ax2.axhline(exit_rsi, color=PALETTE["sell"], linewidth=1.0, linestyle=(0, (3, 3)))
    ax2.fill_between(d.index, 0, entry, color=PALETTE["buy"], alpha=0.10, linewidth=0)
    ax2.annotate(f"buy below {entry:g}", xy=(0.005, entry + 3), xycoords=("axes fraction", "data"),
                 color=PALETTE["buy"], fontsize=8, va="bottom")
    ax2.annotate(f"sell above {exit_rsi:g}", xy=(0.005, exit_rsi + 3), xycoords=("axes fraction", "data"),
                 color=PALETTE["sell"], fontsize=8, va="bottom")
    ax2.set_ylim(-2, 102)
    ax2.set_ylabel("RSI(2)")
    ax2.set_title("Oversold trigger", fontsize=10)
    _tidy_dates(ax2)
    for ax in (ax1, ax2):
        ax.margins(x=0.01)
    return fig


def plot_gem_signals(
    prices: pd.DataFrame,
    signals: StrategySignals,
    start=None,
    end=None,
    title: str = "GEM — which sleeve momentum picked",
):
    """Rebased prices with regime shading, and the 12-month momentum scores below."""
    px = prices.copy()
    if start is not None:
        px = px.loc[pd.Timestamp(start):]
    if end is not None:
        px = px.loc[:pd.Timestamp(end)]
    px = px / px.iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True,
        gridspec_kw=dict(height_ratios=[1.6, 1.0], hspace=0.16),
    )

    asset_colors = {"QQQ": PALETTE["rsi2"], "VEU": PALETTE["gem"], "BOXX": PALETTE["voltarget"]}

    # Regime shading first, so the price lines read on top of it. Contiguous
    # runs of the same holding become one band with one label -- a band per
    # trading day would be unreadable and would print the same name 250 times.
    holding = signals.holding
    if holding is not None:
        h = holding.reindex(px.index).ffill().bfill()
        run_id = h.ne(h.shift()).cumsum()
        for _, seg in h.groupby(run_id):
            asset = seg.iloc[0]
            lo_dt, hi_dt = seg.index[0], seg.index[-1]
            shade = PALETTE["shade_risk"] if asset in ("QQQ", "VEU") else PALETTE["shade_safe"]
            ax1.axvspan(lo_dt, hi_dt, color=shade, zorder=0, linewidth=0)
            # Only name a band wide enough for the text to fit inside it.
            if (hi_dt - lo_dt).days >= 0.06 * max(1, (px.index[-1] - px.index[0]).days):
                ax1.annotate(str(asset), xy=(lo_dt + (hi_dt - lo_dt) / 2, 0.02),
                             xycoords=("data", "axes fraction"),
                             ha="center", va="bottom", fontsize=8, fontweight="600",
                             color=PALETTE["ink_2"], zorder=5,
                             bbox=dict(boxstyle="round,pad=0.25", linewidth=0,
                                       facecolor=PALETTE["surface"], alpha=0.85))

    end_labels = []
    for col in [c for c in ["QQQ", "VEU", "BOXX"] if c in px.columns]:
        ax1.plot(px.index, px[col], color=asset_colors[col], linewidth=2.0, label=col, zorder=3)
        end_labels.append((float(px[col].iloc[-1]), f"{col} {px[col].iloc[-1]:.2f}x",
                           asset_colors[col]))

    ax1.set_title(title)
    ax1.set_ylabel("Rebased to 1.0")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax1.legend(loc="upper left", ncols=3)
    ax1.axhline(1.0, color=PALETTE["axis"], linewidth=0.8, zorder=1)
    _mark_regimes(ax1)
    _direct_labels_spread(ax1, end_labels, x=px.index[-1])

    diag = signals.diagnostics
    if not diag.empty:
        dsub = diag.loc[(diag.index >= px.index[0]) & (diag.index <= px.index[-1])]
        mom_labels = []
        for col in [c for c in dsub.columns if c.startswith("mom_")]:
            asset = col.replace("mom_", "")
            ax2.plot(dsub.index, dsub[col], marker="o", markersize=5,
                     color=asset_colors.get(asset, PALETTE["muted"]),
                     linewidth=1.8, label=asset,
                     markeredgecolor=PALETTE["surface"], markeredgewidth=1.0)
            if len(dsub):
                mom_labels.append((float(dsub[col].iloc[-1]), asset,
                                   asset_colors.get(asset, PALETTE["muted"])))
        if mom_labels:
            _direct_labels_spread(ax2, mom_labels, x=dsub.index[-1])
    ax2.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax2.set_title("12-month trailing total return, measured at each month end", fontsize=10)
    ax2.set_ylabel("Trailing 12m return")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.legend(loc="upper left", ncols=3)
    _tidy_dates(ax2)
    for ax in (ax1, ax2):
        ax.margins(x=0.01)
    fig.subplots_adjust(right=0.86)
    return fig


def plot_voltarget_signals(
    signals: StrategySignals,
    target_vol: float,
    start=None,
    end=None,
    title: str = "Vol targeting — exposure follows realised volatility",
):
    """Three stacked panels: price, realised vol against the target, exposure."""
    d = signals.diagnostics.copy()
    if start is not None:
        d = d.loc[pd.Timestamp(start):]
    if end is not None:
        d = d.loc[:pd.Timestamp(end)]

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 8.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.6, 1.0, 1.0], hspace=0.16),
    )
    ax1, ax2, ax3 = axes

    ax1.plot(d.index, d["close"], color=PALETTE["ink"], linewidth=1.6, label="QQQ close")
    _plot_trade_markers(ax1, signals.events, d.index.min(), d.index.max())
    ax1.set_title(title)
    ax1.set_ylabel("Price ($)")
    handles, labels = ax1.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); hh.append(h); ll.append(l)
    ax1.legend(hh, ll, loc="upper left", ncols=3)
    _mark_regimes(ax1)

    ax2.plot(d.index, d["realized_vol"], color=PALETTE["voltarget"], linewidth=2.0,
             label="Realised vol (EWMA, annualised)")
    ax2.axhline(target_vol, color=PALETTE["ink_2"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax2.annotate(f"target {target_vol:.0%}", xy=(0.005, target_vol),
                 xycoords=("axes fraction", "data"), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=PALETTE["ink_2"])
    ax2.fill_between(d.index, target_vol, d["realized_vol"],
                     where=d["realized_vol"] > target_vol,
                     color=PALETTE["sell"], alpha=0.12, linewidth=0,
                     label="Above target — de-risk")
    ax2.set_ylabel("Annualised vol")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_title("Volatility forecast vs target", fontsize=10)
    ax2.legend(loc="upper left", ncols=2)

    ax3.plot(d.index, d["raw_weight"], color=PALETTE["muted"], linewidth=1.2,
             linestyle=(0, (3, 3)), label="Unbanded target")
    ax3.step(d.index, d["risk_weight"], where="post", color=PALETTE["voltarget"],
             linewidth=2.0, label="Weight actually held")
    ax3.fill_between(d.index, 0, d["risk_weight"], step="post",
                     color=PALETTE["voltarget"], alpha=0.14, linewidth=0)
    ax3.set_ylim(-0.02, max(1.02, float(d["raw_weight"].max()) * 1.05))
    ax3.set_ylabel("QQQ weight")
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax3.set_title("Exposure — the rest sits in BOXX", fontsize=10)
    ax3.legend(loc="upper left", ncols=2)
    _tidy_dates(ax3)
    for ax in axes:
        ax.margins(x=0.01)
    return fig


def plot_momentum_holdings(
    signals: StrategySignals,
    start=None,
    end=None,
    title: str = "Top-6 NDX momentum — who was held, and when",
    max_names: int = 40,
):
    """A timeline of the book: one row per stock, a bar for every stretch held.

    This is the chart the summary table cannot give you. Turnover of "8x a
    year" is an abstraction; six rows that keep their bars for months while
    the rest churn at the bottom is the same fact in a form you can act on.
    """
    log = signals.holdings_log or {}
    dates = sorted(d for d in log
                   if (start is None or d >= pd.Timestamp(start))
                   and (end is None or d <= pd.Timestamp(end)))
    if not dates:
        raise ValueError("no holdings in the requested window")

    # Order rows by first appearance so the chart reads top-left to bottom-right.
    first_seen, days_held = {}, {}
    for d in dates:
        for t in log[d]:
            first_seen.setdefault(t, d)
            days_held[t] = days_held.get(t, 0) + 1

    names = sorted(first_seen, key=lambda t: (first_seen[t], t))
    trimmed = len(names) - max_names
    if trimmed > 0:
        names = sorted(days_held, key=lambda t: -days_held[t])[:max_names]
        names = sorted(names, key=lambda t: (first_seen[t], t))

    fig, ax = plt.subplots(figsize=(11, max(3.2, 0.26 * len(names) + 1.6)))
    row = {t: i for i, t in enumerate(names)}
    held_set = {d: set(log[d]) for d in dates}

    for t in names:
        runs, run_start, prev = [], None, None
        for d in dates:
            on = t in held_set[d]
            if on and run_start is None:
                run_start = d
            elif not on and run_start is not None:
                runs.append((run_start, prev))
                run_start = None
            if on:
                prev = d
        if run_start is not None:
            runs.append((run_start, dates[-1]))
        for a, b in runs:
            width = max((b - a).days, 1)
            ax.barh(row[t], width=width, left=a, height=0.62,
                    color=PALETTE["momentum"], edgecolor=PALETTE["surface"],
                    linewidth=0.6, zorder=3)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylim(len(names) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    if trimmed > 0:
        ax.annotate(f"showing the {max_names} most-held of {len(first_seen)} names",
                    xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", fontsize=8, color=PALETTE["muted"])
    _tidy_dates(ax)
    ax.margins(x=0.01)
    return fig


def plot_vix_regimes(
    signals: StrategySignals,
    results: Optional[Dict[str, BacktestResult]] = None,
    start=None,
    end=None,
    title: str = "VIX circuit breaker — when the book was switched off",
):
    """VIX against its thresholds, the regime it produced, and what it cost.

    Three stacked panels on one x-axis. The top panel is the decision input,
    the middle is the decision, the bottom is the consequence -- read
    downwards and the causal chain is visible without cross-referencing.
    """
    d = signals.diagnostics.copy()
    if start is not None:
        d = d.loc[pd.Timestamp(start):]
    if end is not None:
        d = d.loc[:pd.Timestamp(end)]
    p = signals.params

    n = 3 if results else 2
    fig, axes = plt.subplots(
        n, 1, figsize=(11, 3.0 * n + 1.4), sharex=True,
        gridspec_kw=dict(height_ratios=[1.5, 0.85] + ([1.5] if results else []), hspace=0.18),
    )
    ax1, ax2 = axes[0], axes[1]

    # --- regime shading, drawn first so the VIX line sits on top ---------
    regime = d["regime"]
    runs = regime.ne(regime.shift()).cumsum()
    shade = {"INVESTED": None,
             "CASH": PALETTE["shade_safe"],
             "PARKED": PALETTE["shade_risk"]}
    for _, seg in regime.groupby(runs):
        col = shade.get(seg.iloc[0])
        if col is None:
            continue
        for ax in (ax1, ax2):
            ax.axvspan(seg.index[0], seg.index[-1], color=col, zorder=0, linewidth=0)

    ax1.plot(d.index, d["vix"], color=PALETTE["ink"], linewidth=1.5, label="VIX close", zorder=4)
    ax1.axhline(p["exit_level"], color=PALETTE["sell"], linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=3)
    ax1.axhline(p["entry_level"], color=PALETTE["buy"], linewidth=1.2,
                linestyle=(0, (4, 3)), zorder=3)
    ax1.annotate(f"exit  {p['exit_level']:g}", xy=(0.004, p["exit_level"]),
                 xycoords=("axes fraction", "data"), xytext=(0, 4),
                 textcoords="offset points", fontsize=8.5, color=PALETTE["sell"],
                 bbox=dict(boxstyle="round,pad=0.18", linewidth=0,
                           facecolor=PALETTE["surface"], alpha=0.9))
    ax1.annotate(f"re-enter  {p['entry_level']:g}", xy=(0.004, p["entry_level"]),
                 xycoords=("axes fraction", "data"), xytext=(0, -12),
                 textcoords="offset points", fontsize=8.5, color=PALETTE["buy"],
                 bbox=dict(boxstyle="round,pad=0.18", linewidth=0,
                           facecolor=PALETTE["surface"], alpha=0.9))
    ax1.set_ylabel("VIX")
    ax1.set_title(title)
    ax1.legend(loc="upper right")
    _mark_regimes(ax1)

    # --- the regime itself, as a filled step -----------------------------
    ax2.fill_between(d.index, 0, d["invested"], step="post",
                     color=PALETTE["momentum"], alpha=0.55, linewidth=0)
    ax2.step(d.index, d["invested"], where="post",
             color=PALETTE["momentum"], linewidth=1.6)
    ax2.set_ylim(-0.06, 1.12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["out", "in"])
    pct_in = float(d["invested"].mean())
    ax2.set_title(f"Invested {pct_in:.0%} of the window "
                  f"(shaded: light = cash, blue = parked in the safe asset)", fontsize=10)
    ax2.grid(axis="y", visible=False)

    # --- what it cost -----------------------------------------------------
    if results:
        ax3 = axes[2]
        end_labels = []
        for key in ("momentum", "momentum_vix", "bh_boxx"):
            if key not in results:
                continue
            eq = results[key].equity
            eq = eq / eq.iloc[0]
            bench = key.startswith("bh_")
            ax3.plot(eq.index, eq.values, color=_color(key),
                     linewidth=1.5 if bench else 2.2,
                     linestyle=(0, (5, 2)) if bench else "-", label=_label(key))
            end_labels.append((float(eq.iloc[-1]),
                               f"{_label(key)}  {eq.iloc[-1]:.2f}x", _color(key)))
        ax3.axhline(1.0, color=PALETTE["axis"], linewidth=0.8)
        ax3.set_ylabel("Value of $1")
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}x"))
        ax3.set_title("Protection has a price — compare against the unprotected book",
                      fontsize=10)
        ax3.legend(loc="upper left", ncols=3)
        _direct_labels_spread(ax3, end_labels, x=eq.index[-1])
        _tidy_dates(ax3)
        fig.subplots_adjust(right=0.78)
    else:
        _tidy_dates(ax2)

    for ax in axes:
        ax.margins(x=0.01)
    return fig


def plot_vix_sweep(sweep: pd.DataFrame, baseline_cagr: Optional[float] = None):
    """Trigger level against time invested and against return.

    The point of this chart is the top panel: a trigger set inside the body of
    the VIX distribution switches the strategy off most of the time, and
    everything else follows from that.
    """
    df = sweep.sort_values("exit")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 6.8), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.30),
    )

    ax1.plot(df["exit"], df["Time invested"], marker="o", markersize=7,
             linewidth=2.0, color=PALETTE["momentum_vix"],
             markeredgecolor=PALETTE["surface"], markeredgewidth=1.2)
    ax1.set_ylabel("Share of window invested")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_title("Where the trigger sits in the VIX distribution decides everything")

    ax2.plot(df["exit"], df["CAGR"], marker="o", markersize=7, linewidth=2.0,
             color=PALETTE["momentum_vix"], label="with breaker",
             markeredgecolor=PALETTE["surface"], markeredgewidth=1.2)
    if baseline_cagr is not None:
        ax2.axhline(baseline_cagr, color=PALETTE["momentum"], linewidth=1.8,
                    linestyle=(0, (5, 2)), label="unprotected momentum")
    ax2.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax2.set_ylabel("CAGR")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_xlabel("VIX exit trigger")
    ax2.set_title("…and the return it gives up to get there", fontsize=10)
    ax2.legend(loc="best")

    for ax in (ax1, ax2):
        ax.margins(x=0.06)
    return fig


def plot_band_sensitivity(sweep: pd.DataFrame, n_hold: Optional[int] = None):
    """Turnover and Sharpe against band width, as stacked panels sharing an x-axis.

    Deliberately NOT a dual-axis chart. Turnover falls smoothly and reliably
    with a wider band; Sharpe wanders. Plotting them on twinned y-scales would
    invite the eye to read a relationship between the two curves that the data
    does not support.
    """
    df = sweep.copy()
    if n_hold is not None:
        df = df[df["n_hold"] == n_hold]
    df = df.sort_values("band")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 6.8), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.30),
    )

    for ax, col, label, fmt in (
        (ax1, "Ann. turnover", "Annual turnover", lambda v: f"{v:.0f}x"),
        (ax2, "Sharpe", "Sharpe vs BOXX", lambda v: f"{v:.2f}"),
    ):
        for n, grp in df.groupby("n_hold"):
            ax.plot(grp["band"], grp[col], marker="o", markersize=7,
                    linewidth=2.0, color=PALETTE["momentum"] if n_hold else None,
                    markeredgecolor=PALETTE["surface"], markeredgewidth=1.2,
                    label=f"hold {n}")
        ax.set_ylabel(label)
        if len(df["n_hold"].unique()) > 1:
            ax.legend(loc="best", ncols=4)

    ax1.set_title("What the hysteresis band buys you")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}x"))
    ax2.axhline(0.0, color=PALETTE["axis"], linewidth=0.8)
    ax2.set_title("…and what it costs in risk-adjusted return", fontsize=10)
    ax2.set_xlabel("Band width  (exit_rank − n_hold)")
    for ax in (ax1, ax2):
        ax.margins(x=0.06)
    return fig


# --------------------------------------------------------------------------
# Return distribution
# --------------------------------------------------------------------------

def plot_monthly_heatmap(result: BacktestResult, title: Optional[str] = None, ax=None):
    """Diverging blue<->red around a neutral midpoint: the polarity is the point."""
    m = monthly_return_matrix(result)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 1.0 + 0.55 * max(1, len(m))))

    vmax = float(np.nanmax(np.abs(m.values))) if m.size else 0.05
    vmax = max(vmax, 1e-6)
    cmap = plt.get_cmap("RdBu")   # red<->blue diverging, neutral centre
    im = ax.imshow(m.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels([month_names[c - 1] for c in m.columns])
    ax.set_yticks(range(len(m.index)))
    ax.set_yticklabels(m.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Every cell is labelled -- the grid is small, and a colour alone would
    # leave the reader estimating magnitudes off a ramp.
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.1%}", ha="center", va="center", fontsize=8,
                    color=PALETTE["ink"] if abs(v) < vmax * 0.55 else "#ffffff")
    ax.set_title(title or f"Monthly returns — {_label(result.name)}")
    return ax


def plot_return_scatter(results: Dict[str, BacktestResult], rf: Optional[pd.Series] = None, ax=None):
    """Risk vs return. Three categorical slots only -- past three, this form's
    all-pairs colour separation stops holding, so benchmarks go grey."""
    from .metrics import summarise
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.2))
    for key, res in results.items():
        s = summarise(res, rf=rf)
        if not s:
            continue
        ax.scatter(s["Ann. vol"], s["CAGR"], s=150, color=_color(key),
                   edgecolor=PALETTE["surface"], linewidth=1.5, zorder=3)
        ax.annotate(_label(key), xy=(s["Ann. vol"], s["CAGR"]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, va="center", color=PALETTE["ink_2"])
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("CAGR")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axhline(0, color=PALETTE["axis"], linewidth=0.8)
    ax.set_title("Risk taken vs return earned")
    ax.margins(0.18)
    return ax
