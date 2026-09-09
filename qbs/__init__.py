"""qbs -- QQQ/BOXX strategy lab.

Three strategies, one backtest engine, one chart system.

    from qbs import data, strategies, engine, metrics, plotting

Quick start::

    from qbs.pipeline import run
    res = run()                 # downloads, backtests, prints the table
"""

from .config import (
    Config, GEMParams, RSI2Params, VolTargetParams,
    RISK_ASSET, INTL_ASSET, SAFE_ASSET, TICKERS,
    BACKTEST_START, BACKTEST_END, PALETTE, STRATEGY_LABELS,
)

__all__ = [
    "Config", "GEMParams", "RSI2Params", "VolTargetParams",
    "RISK_ASSET", "INTL_ASSET", "SAFE_ASSET", "TICKERS",
    "BACKTEST_START", "BACKTEST_END", "PALETTE", "STRATEGY_LABELS",
]

__version__ = "1.0.0"
