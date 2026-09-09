"""Backtest engine.

One function does the accounting for every strategy, so the execution
assumptions -- lag, costs, when turnover is charged -- are identical across
the comparison. A strategy that looked good only because it had a different
lag would be a bug, not a finding.

Convention
----------
`weights.loc[t]` is what the strategy decided while looking at the close of
day t. With `lag=1` that weight earns day t+1's return. Set `lag=0` only to
measure how much of an edge is same-close execution (it is not tradeable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import COST_BPS, EXECUTION_LAG
from .indicators import drawdown
from .strategies import StrategySignals


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    returns: pd.Series          # net of costs
    gross_returns: pd.Series
    weights: pd.DataFrame       # as actually held (lag applied)
    turnover: pd.Series
    costs: pd.Series            # commission + slippage combined
    drawdown: pd.Series
    signals: Optional[StrategySignals] = None
    commission: Optional[pd.Series] = None
    slippage: Optional[pd.Series] = None

    @property
    def start(self) -> pd.Timestamp:
        return self.equity.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.equity.index[-1]


def run_backtest(
    prices: pd.DataFrame,
    signals: StrategySignals,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lag: int = EXECUTION_LAG,
    cost_bps: float = COST_BPS,
    slippage_bps: float = 0.0,
    initial: float = 1.0,
) -> BacktestResult:
    """Run one strategy and return its equity curve and diagnostics.

    Costs are charged on the day a position actually changes, at
    `cost_bps` per 100% of one-way turnover summed across legs. A full switch
    from QQQ to BOXX is |Δ| = 1 + 1 = 2, i.e. two legs, which is right: you
    pay to sell one and to buy the other.

    `slippage_bps` is tracked separately from commission because it models a
    different thing: the gap between the price the signal was computed on
    (~15:30, when the decision job runs) and the price the order actually
    fills at (the 16:00 closing auction). It is not a fee -- it is execution
    uncertainty, it is roughly unbiased, and keeping it separate lets you see
    how much of the result depends on assuming it away.
    """
    px = prices.ffill()
    w_decided = signals.weights.reindex(px.index).ffill().fillna(0.0)

    # The lag lives here and nowhere else.
    w_held = w_decided.shift(lag).fillna(0.0)

    rets = px.pct_change().fillna(0.0)

    # Trim to the reporting window AFTER the shift, so the first backtest day
    # inherits the position decided on the last day before the window.
    if start is not None:
        mask = px.index >= pd.Timestamp(start)
        w_held, rets, w_decided = w_held[mask], rets[mask], w_decided[mask]
    if end is not None:
        mask = w_held.index <= pd.Timestamp(end)
        w_held, rets, w_decided = w_held[mask], rets[mask], w_decided[mask]

    if w_held.empty:
        raise ValueError("no rows left after applying the start/end window")

    gross = (w_held * rets[w_held.columns]).sum(axis=1)

    turnover = w_held.diff().abs().sum(axis=1)
    turnover.iloc[0] = w_held.iloc[0].abs().sum()   # cost of putting the book on
    commission = turnover * (cost_bps / 1e4)
    slippage = turnover * (slippage_bps / 1e4)
    costs = commission + slippage

    net = gross - costs
    equity = initial * (1.0 + net).cumprod()

    return BacktestResult(
        name=signals.name,
        equity=equity,
        returns=net,
        gross_returns=gross,
        weights=w_held,
        turnover=turnover,
        costs=costs,
        drawdown=drawdown(equity),
        signals=signals,
        commission=commission,
        slippage=slippage,
    )


def run_all(
    prices: pd.DataFrame,
    signal_set: Dict[str, StrategySignals],
    **kwargs,
) -> Dict[str, BacktestResult]:
    return {k: run_backtest(prices, s, **kwargs) for k, s in signal_set.items()}


def equity_frame(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """Equity curves side by side, all rebased to 1.0 at the common start."""
    df = pd.DataFrame({k: r.equity for k, r in results.items()}).dropna(how="all")
    return df / df.iloc[0]
