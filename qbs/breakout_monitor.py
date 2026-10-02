"""Breakout monitor: how close is a name to a breakout, scored out of 100?

Ported from the `breakout-checking` script -- the same indicators, the same
scoring table and the same status bands. Swing detection has since moved from
the script's standard-deviation prominence to an ATR-based one (1.5x the
latest 14-day ATR), and support/resistance now comes from the Daily picks
chart's engine (`qbs.breakout.sr_levels`) so the two tabs agree. Everything
but the levels runs on one year of daily bars per name (the script's
`period="1y"`).

Not to be confused with `qbs.breakout`, the M6 weekly-breakout STRATEGY whose
level engine it borrows; it has hourly entries and trades. This is a read-only screen over a
watchlist; nothing trades on it.

Steps, per name:

* swings     -- `scipy.signal.find_peaks` on the highs and the (negated)
                lows, 5 bars apart, prominence 1.5x the latest 14-day ATR
* structure  -- HH: the last swing high above the one before it; HL: the
                same for the swing lows
* levels     -- the Daily picks chart's support/resistance (`qbs.breakout.
                sr_levels` on the close history): the three nearest above the
                close (resistance) and below it (support)
* score      -- structure 25, moving averages 15, momentum 13, distance to
                the nearest resistance 15, volume vs its 20-day average 15;
                capped at 100
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

# The script's starting list. The dashboard seeds its box from this unless
# QBS_DASH_BREAKOUT_WATCHLIST says otherwise.
DEFAULT_WATCHLIST = ("DOCN", "FTNT", "BTC-USD", "ETH-USD")

# (minimum score, label) -- highest first, as the script checks them.
STATUS_BANDS = ((80, "🔥 STRONG BREAKOUT SETUP"),
                (65, "🟡 BREAKOUT WATCH"),
                (50, "⚪ DEVELOPING"))
NO_SETUP = "❌ NO SETUP"


def last_year(ohlc: pd.DataFrame) -> pd.DataFrame:
    """The final 365 calendar days of bars: the script's `period="1y"`.

    Trimming matters beyond speed -- the swings are found over the whole
    window, so a longer history can reach older levels.
    """
    df = ohlc.sort_index()
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df]
    df = df[cols].astype(float).dropna()
    if df.empty:
        return df
    return df[df.index > df.index[-1] - pd.Timedelta(days=365)]


# Swing prominence in ATRs: a peak must stand out by this many days of
# normal range to count.
SWING_ATR_MULT = 1.5


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """`ATR`: the simple `period`-day average of the true range. In place."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close],
                           axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(period).mean()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """SMA20/50/200, 20-day average volume, 14-day RSI (simple averages),
    MACD(12, 26, 9) and 14-day ATR."""
    df = calculate_atr(df.copy())
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()

    delta = df["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()
    return df


def find_swings(df: pd.DataFrame, distance: int = 5,
                atr_mult: float = SWING_ATR_MULT) -> Tuple[pd.Series, pd.Series]:
    """Swing highs and swing lows, each a Series of price by date.

    Prominence is `atr_mult` x the latest ATR, for both. `df` must carry an
    `ATR` column (`add_indicators` adds it).
    """
    from scipy.signal import find_peaks

    prominence = df["ATR"].iloc[-1] * atr_mult
    high_idx, _ = find_peaks(df["High"].values, distance=distance,
                             prominence=prominence)
    low_idx, _ = find_peaks(-df["Low"].values, distance=distance,
                            prominence=prominence)
    return df.iloc[high_idx]["High"], df.iloc[low_idx]["Low"]


def detect_market_structure(swing_highs: pd.Series, swing_lows: pd.Series) -> Dict:
    """HH / HL from the last two swings of each kind, plus the swings used."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"HH": False, "HL": False}
    last_high, prev_high = swing_highs.iloc[-1], swing_highs.iloc[-2]
    last_low, prev_low = swing_lows.iloc[-1], swing_lows.iloc[-2]
    return {"HH": bool(last_high > prev_high), "HL": bool(last_low > prev_low),
            "last_high": float(last_high), "prev_high": float(prev_high),
            "last_low": float(last_low), "prev_low": float(prev_low)}


def get_support_resistance(closes: pd.Series, n: int = 3
                           ) -> Tuple[List[float], List[float]]:
    """The `n` nearest levels below the last close (support) and at or above
    it (resistance), nearest first.

    The Daily picks chart's levels, not this module's swings: `sr_levels` on
    close-to-close bars over the whole history, split at the last close the
    way the chart colours them. Same input, same engine, so the two tabs show
    the same numbers for a name.
    """
    from qbs.breakout import closes_to_bars, sr_levels
    from qbs.config import BreakoutParams

    closes = closes.dropna()
    bars = closes_to_bars(closes.to_frame("x"))["x"]
    levels = sr_levels(bars, BreakoutParams())
    price = float(closes.iloc[-1])
    res = sorted(x for x in levels if x >= price)
    sup = sorted((x for x in levels if x < price), reverse=True)
    return sup[:n], res[:n]


def score_setup(df: pd.DataFrame, structure: Dict, support: List[float],
                resistance: List[float]) -> int:
    """The script's 0-100 setup score. See the module docstring for weights."""
    latest = df.iloc[-1]
    score = 0

    # market structure
    if structure.get("HH"):
        score += 10
    if structure.get("HL"):
        score += 10
    if "last_low" in structure and latest["Close"] > structure["last_low"]:
        score += 5

    # moving averages
    if latest["Close"] > latest["SMA20"]:
        score += 3
    if latest["SMA20"] > latest["SMA50"]:
        score += 4
    if latest["SMA50"] > latest["SMA200"]:
        score += 4
    if df["SMA20"].iloc[-1] > df["SMA20"].iloc[-5]:
        score += 2
    if df["SMA50"].iloc[-1] > df["SMA50"].iloc[-5]:
        score += 2

    # momentum
    if 50 <= latest["RSI"] <= 75:
        score += 5
    if latest["MACD"] > latest["MACD_SIGNAL"]:
        score += 5
    if latest["Close"] / df["Close"].iloc[-20] - 1 > 0:
        score += 3

    # breakout proximity
    if resistance:
        distance = resistance[0] / latest["Close"] - 1
        if distance < 0.02:
            score += 15
        elif distance < 0.04:
            score += 12
        elif distance < 0.07:
            score += 8
        elif distance < 0.10:
            score += 5

    # volume
    volume_ratio = latest["Volume"] / latest["VOL20"]
    if volume_ratio > 1.5:
        score += 15
    elif volume_ratio > 1.2:
        score += 12
    elif volume_ratio > 1:
        score += 8
    else:
        score += 4

    return min(score, 100)


def setup_status(score: float) -> str:
    """The script's label for a score."""
    for floor, label in STATUS_BANDS:
        if score >= floor:
            return label
    return NO_SETUP


def breakout_monitor(ticker: str, ohlc: Optional[pd.DataFrame]) -> Dict:
    """Everything the script prints for one name, as one dict.

    `ohlc` is daily Open/High/Low/Close/Volume; anything past the last year is
    dropped first. A name without bars, or without a Volume column, comes back
    with an `error` instead of a score -- one bad ticker costs its own row.
    """
    if ohlc is None or ohlc.empty:
        return {"ticker": ticker, "error": "no daily bars"}
    if "Volume" not in ohlc:
        return {"ticker": ticker, "error": "no volume in the bars"}
    full_close = ohlc["Close"].astype(float).sort_index()
    df = last_year(ohlc)
    if len(df) < 20:
        return {"ticker": ticker, "error": f"only {len(df)} bars"}

    df = add_indicators(df)
    highs, lows = find_swings(df)
    structure = detect_market_structure(highs, lows)
    support, resistance = get_support_resistance(
        full_close.loc[:df.index[-1]])
    score = score_setup(df, structure, support, resistance)
    latest = df.iloc[-1]
    return {
        "ticker": ticker,
        "asof": df.index[-1],
        "price": float(latest["Close"]),
        "structure": structure,
        "support": support,
        "resistance": resistance,
        "score": score,
        "status": setup_status(score),
        "rsi": float(latest["RSI"]),
        "volume_ratio": float(latest["Volume"] / latest["VOL20"]),
        "to_resistance": (resistance[0] / latest["Close"] - 1
                          if resistance else None),
    }
