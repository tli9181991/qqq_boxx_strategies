"""qbs -- QQQ/BOXX strategy lab.

Six strategies, one backtest engine, one chart system.

    from qbs import data, strategies, engine, metrics, plotting

Quick start::

    from qbs.pipeline import run
    res = run()                 # downloads, backtests, prints the table
"""

from .config import (
    BookVolTargetParams, Config, GEMParams, MomentumParams, RSI2Params,
    VixBreakerParams, VolTargetParams,
    RISK_ASSET, INTL_ASSET, SAFE_ASSET, TICKERS,
    BACKTEST_START, BACKTEST_END, PALETTE, STRATEGY_LABELS,
)

__all__ = [
    "BookVolTargetParams", "Config", "GEMParams", "MomentumParams",
    "RSI2Params", "VixBreakerParams", "VolTargetParams",
    "RISK_ASSET", "INTL_ASSET", "SAFE_ASSET", "TICKERS",
    "BACKTEST_START", "BACKTEST_END", "PALETTE", "STRATEGY_LABELS",
]

__version__ = "1.0.0"
