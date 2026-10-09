"""The Market Context Builder: completed candles in, a compact snapshot out.

Three rules, each one a way a model gets handed something untrue:

1. **Completed candles only.** A provider serves the bar in progress
   alongside the finished ones. A 15-minute bar is used only if
   `start + 15min <= candle_end`; a 30-minute bar likewise; the daily
   series stops at the PREVIOUS session, and today is described from
   today's completed 15-minute bars instead of a half-built daily bar.
2. **Never fabricate.** An indicator without enough history is `null` with a
   warning, not a guess. If the newest completed 15-minute bar is not the
   one the slot is about, the snapshot is STALE; with no bars it is MISSING.
   Either way the engine makes no LLM request and logs a non-actionable
   record instead.
3. **Compact.** A handful of raw candles and a few summary numbers per
   timeframe -- not hundreds of rows. Token usage is logged per request so
   `MarketContextConfig` can be tuned against what it costs.

Indicators reuse `qbs.indicators.wilder_rsi`; EMA, MACD and ATR are the
one-line pandas definitions, kept here because nothing in `qbs` computes
them on intraday bars.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from ..indicators import wilder_rsi
from .config import MarketContextConfig
from .session_calendar import ET, session_bounds

CONTEXT_VERSION = "ctx-v1"
FRESH, STALE, MISSING = "FRESH", "STALE", "MISSING"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_SPAN = {"15m": timedelta(minutes=15), "30m": timedelta(minutes=30)}


class MarketDataProvider(Protocol):
    """Bars for one symbol. Index: tz-aware bar START times. Columns: OHLCV."""
    name: str

    def bars(self, symbol: str, interval: str) -> Optional[pd.DataFrame]: ...


class YFinanceProvider:
    """Yahoo via yfinance, the repo's existing price source.

    Regular-hours bars only (`prepost=False`). Yahoo's intraday feed is
    delayed and occasionally late to publish a bar -- the freshness check is
    what stops that from becoming a decision on old data.
    """
    name = "yfinance"
    PERIODS = {"15m": "5d", "30m": "1mo", "1d": "1y"}

    def bars(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        import yfinance as yf

        df = yf.Ticker(symbol).history(period=self.PERIODS[interval], interval=interval,
                                       prepost=False, auto_adjust=True, actions=False)
        if df is None or df.empty:
            return None
        return df[[c for c in OHLCV if c in df.columns]]


@dataclass
class Snapshot:
    symbol: str
    candle_end: datetime
    freshness: str
    data: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    data_latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.freshness == FRESH and self.error is None

    @property
    def last_price(self) -> Optional[float]:
        return self.data.get("last_price")


def _r(v: Any, nd: int = 2) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, nd)


def _et_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        # A naive index is ambiguous; yfinance localizes intraday bars to the
        # exchange, so naive here means a test or a provider that dropped it.
        idx = idx.tz_localize(ET)
    df.index = idx.tz_convert(ET)
    return df.sort_index()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(),
                    (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def swing_points(df: pd.DataFrame, k: int = 2, n: int = 3) -> Dict[str, List[float]]:
    """Pivot highs/lows: a bar beyond the `k` bars on each side. Newest `n`."""
    h, l = df["High"].to_numpy(), df["Low"].to_numpy()
    highs, lows = [], []
    for i in range(k, len(df) - k):
        if h[i] == max(h[i - k:i + k + 1]):
            highs.append(_r(h[i]))
        if l[i] == min(l[i - k:i + k + 1]):
            lows.append(_r(l[i]))
    return {"swing_highs": highs[-n:], "swing_lows": lows[-n:]}


def _raw(df: pd.DataFrame, n: int, fmt: str) -> Dict[str, Any]:
    tail = df.tail(n)
    return {"cols": ["t", "o", "h", "l", "c", "v"],
            "rows": [[ts.strftime(fmt), _r(r.Open), _r(r.High), _r(r.Low), _r(r.Close),
                      int(r.Volume) if pd.notna(r.Volume) else None]
                     for ts, r in tail.iterrows()]}


def _last(s: pd.Series, warnings: List[str], label: str, nd: int = 2) -> Optional[float]:
    v = _r(s.iloc[-1], nd) if len(s) else None
    if v is None:
        warnings.append(f"{label} unavailable (not enough completed bars)")
    return v


def completed(df: pd.DataFrame, interval: str, candle_end: datetime) -> pd.DataFrame:
    """Bars whose END is at or before `candle_end`."""
    return df[df.index + _SPAN[interval] <= candle_end]


class MarketContextBuilder:
    def __init__(self, provider: MarketDataProvider,
                 cfg: Optional[MarketContextConfig] = None,
                 extra_early_closes=()):
        self.provider = provider
        self.cfg = cfg or MarketContextConfig()
        self.extra_early_closes = extra_early_closes

    def _fetch(self, symbol: str, interval: str, warnings: List[str]) -> Optional[pd.DataFrame]:
        try:
            df = self.provider.bars(symbol, interval)
        except Exception as exc:  # noqa: BLE001 -- network, provider, parsing
            warnings.append(f"{interval} fetch failed: {type(exc).__name__}: {exc}")
            return None
        if df is None or df.empty or not set(OHLCV) <= set(df.columns):
            warnings.append(f"{interval} bars unavailable")
            return None
        return _et_index(df[OHLCV].dropna(subset=["Close"]))

    def build(self, symbol: str, candle_end: datetime,
              key_levels: Optional[List[float]] = None) -> Snapshot:
        candle_end = candle_end.astimezone(ET)
        warnings: List[str] = []
        t0 = _time.perf_counter()
        b15 = self._fetch(symbol, "15m", warnings)
        b30 = self._fetch(symbol, "30m", warnings)
        bd = self._fetch(symbol, "1d", warnings)
        latency = (_time.perf_counter() - t0) * 1000
        source = getattr(self.provider, "name", type(self.provider).__name__)
        base = {"context_version": CONTEXT_VERSION, "symbol": symbol,
                "data_source": source, "as_of_candle_end": candle_end.isoformat()}

        if b15 is None:
            return Snapshot(symbol, candle_end, MISSING, base, warnings, latency,
                            "no 15-minute bars")
        c15 = completed(b15, "15m", candle_end)
        if c15.empty:
            return Snapshot(symbol, candle_end, MISSING, base, warnings, latency,
                            "no completed 15-minute bars")
        newest_end = c15.index[-1] + _SPAN["15m"]
        base["latest_completed_15m_end"] = newest_end.isoformat()
        if newest_end != candle_end:
            return Snapshot(symbol, candle_end, STALE, base, warnings, latency,
                            f"newest completed 15m bar ends {newest_end:%Y-%m-%d %H:%M} ET, "
                            f"expected {candle_end:%Y-%m-%d %H:%M} ET")

        day = candle_end.date()
        today = c15[c15.index.date == day]
        cfg = self.cfg
        data = dict(base)
        data["freshness"] = FRESH
        bounds = session_bounds(day, extra_early_closes=self.extra_early_closes)
        if bounds:
            o, c = bounds
            data["session"] = {
                "date": day.isoformat(), "open": o.strftime("%H:%M"),
                "close": c.strftime("%H:%M"), "early_close": c.hour < 16,
                "minutes_since_open": int((candle_end - o).total_seconds() // 60),
                "minutes_to_close": int((c - candle_end).total_seconds() // 60),
                "timezone": "America/New_York"}
        last = c15.iloc[-1]
        data["last_price"] = _r(last.Close)
        data["last_price_ts"] = newest_end.isoformat()

        # ---- daily: completed sessions only ---------------------------
        daily: Dict[str, Any] = {}
        prev_close = None
        if bd is not None:
            dd = bd[bd.index.date < day]
            if not dd.empty:
                prev = dd.iloc[-1]
                prev_close = float(prev.Close)
                win = dd.tail(cfg.candles_daily)
                daily = {
                    "raw": _raw(dd, cfg.raw_daily, "%Y-%m-%d"),
                    "prev_high": _r(prev.High), "prev_low": _r(prev.Low),
                    "prev_close": _r(prev.Close),
                    "ema20": _last(ema(dd.Close, 20), warnings, "daily EMA20"),
                    "sma50": _last(dd.Close.rolling(50, min_periods=50).mean(), warnings,
                                   "daily SMA50"),
                    "rsi14": _last(wilder_rsi(dd.Close, 14), warnings, "daily RSI14", 1),
                    "atr14": _last(atr(dd), warnings, "daily ATR14"),
                    "window_high": _r(win.High.max()), "window_low": _r(win.Low.min()),
                    "window_sessions": int(len(win)),
                    "return_window_pct": _r((dd.Close.iloc[-1] / win.Close.iloc[0] - 1) * 100),
                }
            else:
                warnings.append("no completed daily bars before today")
        data["daily"] = daily

        # ---- today, from completed 15m bars ---------------------------
        if not today.empty:
            vol = today.Volume.sum()
            tp = (today.High + today.Low + today.Close) / 3
            data["today"] = {
                "open": _r(today.Open.iloc[0]), "high": _r(today.High.max()),
                "low": _r(today.Low.min()), "last": _r(today.Close.iloc[-1]),
                "volume": int(vol),
                "vwap": _r((tp * today.Volume).sum() / vol) if vol > 0 else None,
                "change_vs_prev_close_pct": (_r((today.Close.iloc[-1] / prev_close - 1) * 100)
                                             if prev_close else None),
                "completed_15m_bars": int(len(today))}
        else:
            warnings.append("no completed 15m bars yet today")

        # ---- 15m: primary --------------------------------------------
        w15 = c15.tail(cfg.candles_15m)
        line, sig, hist = macd(c15.Close)
        vol_base = c15.Volume.iloc[:-1].tail(20)
        tf15: Dict[str, Any] = {
            "raw": _raw(c15, cfg.raw_15m, "%m-%d %H:%M"),
            "ema9": _last(ema(c15.Close, 9), warnings, "15m EMA9"),
            "ema21": _last(ema(c15.Close, 21), warnings, "15m EMA21"),
            "rsi14": _last(wilder_rsi(c15.Close, 14), warnings, "15m RSI14", 1),
            "macd": {"line": _last(line, warnings, "15m MACD", 3),
                     "signal": _r(sig.iloc[-1], 3), "hist": _r(hist.iloc[-1], 3),
                     "hist_prev": _r(hist.iloc[-2], 3) if len(hist) > 1 else None},
            "atr14": _last(atr(c15), warnings, "15m ATR14", 3),
            "rel_volume_vs_prev20": (_r(c15.Volume.iloc[-1] / vol_base.mean())
                                     if len(vol_base) and vol_base.mean() > 0 else None),
            "window_bars": int(len(w15)),
            "window_high": _r(w15.High.max()), "window_low": _r(w15.Low.min()),
            **swing_points(w15),
        }
        data["tf_15m"] = tf15

        # ---- 30m: confirmation ---------------------------------------
        if b30 is not None:
            c30 = completed(b30, "30m", candle_end)
            if not c30.empty:
                e20 = ema(c30.Close, 20)
                data["tf_30m"] = {
                    "raw": _raw(c30, cfg.raw_30m, "%m-%d %H:%M"),
                    "ema20": _last(e20, warnings, "30m EMA20"),
                    "ema20_slope_3bars": (_r(e20.iloc[-1] - e20.iloc[-4])
                                          if len(e20.dropna()) >= 4 else None),
                    "rsi14": _last(wilder_rsi(c30.Close, 14), warnings, "30m RSI14", 1),
                    "window_high": _r(c30.tail(cfg.candles_30m).High.max()),
                    "window_low": _r(c30.tail(cfg.candles_30m).Low.min()),
                }
            else:
                warnings.append("no completed 30m bars")

        if key_levels:
            lp = data["last_price"]
            data["prompt_levels"] = [{"level": _r(x),
                                      "distance_pct": _r((lp / x - 1) * 100) if lp else None}
                                     for x in key_levels]
        data["indicators_used"] = ["EMA9/21 (15m)", "RSI14", "MACD 12/26/9 (15m)", "ATR14",
                                   "relative volume (15m vs prior 20 bars)", "VWAP (today)",
                                   "swing highs/lows (15m window)", "EMA20 (30m, daily)",
                                   "SMA50 (daily)", "prior-day H/L/C", "window high/low"]
        data["warnings"] = warnings
        return Snapshot(symbol, candle_end, FRESH, data, warnings, latency)
