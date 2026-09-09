"""Indicator primitives. Pure functions on a price Series -- no state, no I/O.

Every function here returns a Series aligned to its input index, with NaN in
the warm-up region. Nothing forward-fills across the warm-up: a NaN means the
indicator genuinely is not defined yet, and the strategy layer must treat it
as "no signal" rather than as zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def wilder_rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """Wilder's RSI -- the definition Connors' RSI(2) rules assume.

    Wilder smooths average gain/loss with an exponential filter of
    alpha = 1/period, seeded by the simple mean of the first `period` changes.
    Using a plain EWM without that seed drifts for the first few dozen bars;
    with period=2 the drift washes out fast, but the seed is cheap and correct.
    """
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)

    if len(close) <= period:
        return pd.Series(np.nan, index=close.index, name=f"RSI{period}")

    g = gain.to_numpy()
    l = loss.to_numpy()
    ag = np.full(len(close), np.nan)
    al = np.full(len(close), np.nan)

    # Seed at index `period` with the simple mean of changes 1..period.
    ag[period] = np.nanmean(g[1:period + 1])
    al[period] = np.nanmean(l[1:period + 1])
    a = 1.0 / period
    for i in range(period + 1, len(close)):
        ag[i] = ag[i - 1] * (1 - a) + g[i] * a
        al[i] = al[i - 1] * (1 - a) + l[i] * a

    avg_gain[:] = ag
    avg_loss[:] = al

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # All-gain window: RS is infinite, RSI is 100 by definition.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # Utterly flat window: neither up nor down -> neutral.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    rsi.name = f"RSI{period}"
    return rsi


def sma(close: pd.Series, window: int) -> pd.Series:
    out = close.rolling(window, min_periods=window).mean()
    out.name = f"SMA{window}"
    return out


def realized_vol(
    returns: pd.Series,
    halflife: int = 20,
    min_periods: int = 20,
    annualise: bool = True,
) -> pd.Series:
    """EWMA volatility forecast.

    An exponentially weighted estimate reacts to a vol spike within days and
    decays smoothly, where a 20-day rolling window jumps the moment a big
    return falls out the back of the window.
    """
    vol = returns.ewm(halflife=halflife, min_periods=min_periods).std()
    if annualise:
        vol = vol * np.sqrt(TRADING_DAYS)
    vol.name = "realized_vol"
    return vol


def total_return(close: pd.Series, periods: int) -> pd.Series:
    """Trailing total return over `periods` observations of `close`."""
    return close / close.shift(periods) - 1.0


def drawdown(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (0 at a new high, negative below)."""
    peak = equity.cummax()
    return equity / peak - 1.0
