"""S&P 500 rebound monitor: is the index bouncing off a moving average?

Ported from `sp500-rebound-monitor` as written -- the same moving averages,
the same hammer rule and the same support tests -- so the dashboard reads
exactly what that script reads. It runs on the S&P 500 ETF's daily bars (VOO
by default).

Note the hammer here is the script's own, looser rule, and deliberately NOT
`qbs.candles.HammerRules`: lower wick >= 2x the body, upper wick <= 1.5x the
body, close in the top 40% of the range. No prior-trend condition -- the
support test supplies the context (a dip to a moving average).

Per moving average (SMA50 / SMA100 / SMA200), with `tolerance` (1%):

* touch   -- the day's low came within `tolerance` of the average
* reclaim -- the low got to (or under) the average + `tolerance` and the
             close finished back ABOVE it: a dip into the line that held
* support -- a reclaim on a hammer day: the strongest of the three, the
             line rejected the sell-off
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MAS = ("SMA50", "SMA100", "SMA200")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """SMA20/50/100/200 of the close and a 14-day ATR."""
    df = df.copy()
    for n in (20, 50, 100, 200):
        df[f"SMA{n}"] = df["Close"].rolling(n).mean()
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Body, wicks and the script's hammer flag."""
    df = df.copy()
    body = (df["Close"] - df["Open"]).abs()
    lower_wick = np.minimum(df["Open"], df["Close"]) - df["Low"]
    upper_wick = df["High"] - np.maximum(df["Open"], df["Close"])
    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body"] = body
    df["lower_wick"] = lower_wick
    df["upper_wick"] = upper_wick
    df["hammer"] = ((lower_wick >= 2 * body)
                    & (upper_wick <= body * 1.5)
                    & ((df["Close"] - df["Low"]) / candle_range >= 0.60))
    return df


def add_support_signals(df: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """touch / reclaim / support per moving average (see the module docstring)."""
    df = df.copy()
    for ma in MAS:
        df[f"{ma}_touch"] = (df["Low"] - df[ma]).abs() / df[ma] <= tolerance
        df[f"{ma}_reclaim"] = ((df["Low"] <= df[ma] * (1 + tolerance))
                               & (df["Close"] > df[ma]))
        df[f"{ma}_support"] = df[f"{ma}_reclaim"] & df["hammer"]
    return df


def rebound_frame(ohlc: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """All three steps on daily OHLC bars, plus each close's distance from
    each average (`<MA>_dist`, a fraction: +0.02 is 2% above it)."""
    df = ohlc[["Open", "High", "Low", "Close"]].astype(float).dropna().sort_index()
    df = add_support_signals(add_candle_features(add_indicators(df)), tolerance)
    for ma in MAS:
        df[f"{ma}_dist"] = df["Close"] / df[ma] - 1.0
    return df


def ma_status(row: pd.Series, ma: str) -> Optional[str]:
    """The strongest signal one session gives for one average, or None."""
    if row.get(f"{ma}_support"):
        return "support"
    if row.get(f"{ma}_reclaim"):
        return "reclaim"
    if row.get(f"{ma}_touch"):
        return "touch"
    return None
