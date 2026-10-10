"""Build a versioned market fixture for the agent test notebooks.

    python scripts/build_agent_fixtures.py                       # v1, synthetic
    python scripts/build_agent_fixtures.py --version v2 --source yfinance --replay-date 2026-10-08

Writes notebooks/agent_tests/fixtures/<version>/ :

    <SYMBOL>_15m.csv, <SYMBOL>_30m.csv, <SYMBOL>_1d.csv   start (UTC ISO), OHLCV
    manifest.json   version, source, seed, replay date, symbols, the daily
                    prompt the scenario is written for, SHA-256 per file

`synthetic` is deterministic (fixed seed): rerunning it reproduces the files
byte for byte, and `testkit.load_fixture` refuses a file whose hash no longer
matches. Its scenario is scripted so the replay day exercises the pipeline:
TEAM breaks out above $200, MRVL (a held position) fades below VWAP in the
afternoon, TSM pulls back and recovers. The prices are NOT real.

`yfinance` downloads real bars (Yahoo keeps ~60 days of 15m history), cut at
the replay date. Commit the result as a NEW version; never overwrite one that
notebooks or results already refer to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.trading_agent.session_calendar import ET, is_trading_day  # noqa: E402
from qbs.trading_agent.testkit import FIXTURE_DIR, sha256_file      # noqa: E402

SYMBOLS = ("TEAM", "MRVL", "TSM")
SEED = 20261008
INTRADAY_SESSIONS = 10
DAILY_SESSIONS = 252

PROMPT = """Date: {day}

Monitor only TEAM, MRVL, and TSM.

TEAM:
Watch for a breakout above $200.
Look for breakout confirmation or a successful retest.
Maximum suggested position: 4 shares.
Reference stop loss: $196.

MRVL:
Monitor my existing 9-share position.
Evaluate whether to hold or take profit.
Do not recommend adding new shares.

TSM:
Monitor for a potential recovery after pullback.
Recovery confirmation above $302.
Maximum suggested position: 3 shares.
Do not force a trade without confirmation.

General:
Use 15-minute candles for primary analysis.
Use 30-minute and daily candles as supporting context.
Avoid unnecessary trading.
No short selling.
"""

# Replay-day path per symbol: (bar index 0..25, price) waypoints.
SCENARIO = {
    "TEAM": (197.0, [(0, 197.4), (4, 199.2), (6, 199.8), (7, 200.7), (10, 200.2),
                     (14, 201.4), (25, 202.3)]),
    "MRVL": (224.0, [(0, 224.5), (6, 227.2), (10, 228.1), (16, 226.0), (19, 223.4),
                     (25, 222.8)]),
    "TSM": (300.0, [(0, 299.5), (6, 294.2), (10, 295.0), (16, 300.5), (20, 302.8),
                    (25, 303.6)]),
}


def trading_days(end: date, n: int):
    out, d = [], end
    while len(out) < n:
        if is_trading_day(d):
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def bars_from_path(day: date, closes: np.ndarray, rng, base_vol: float) -> pd.DataFrame:
    starts = [datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET) + timedelta(minutes=15 * i)
              for i in range(len(closes))]
    opens = np.r_[closes[0] - rng.normal(0, 0.1), closes[:-1]]
    spread = np.abs(rng.normal(0, 0.18, len(closes))) + 0.05
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread[::-1]
    u = np.linspace(-1, 1, len(closes))
    vol = base_vol * (0.6 + 1.2 * u ** 2) * rng.uniform(0.8, 1.2, len(closes))
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": vol.round()}, index=pd.DatetimeIndex(starts, name="start"))


def synthetic(replay: date):
    rng = np.random.default_rng(SEED)
    out = {}
    for sym in SYMBOLS:
        prior_close, waypoints = SCENARIO[sym]
        idays = trading_days(replay, INTRADAY_SESSIONS)
        # Earlier intraday sessions: a mean-reverting walk ending at the prior close.
        frames, level = [], prior_close * 0.985
        for d in idays[:-1]:
            path = level + np.cumsum(rng.normal(0, prior_close * 0.0012, 26))
            path += np.linspace(0, (prior_close - level) * 0.25, 26)
            level = path[-1]
            frames.append(bars_from_path(d, path, rng, 40_000))
        last = frames[-1].index[-1]
        frames[-1].loc[last, "Close"] = prior_close
        frames[-1].loc[last, "High"] = max(frames[-1].loc[last, "High"], prior_close)
        frames[-1].loc[last, "Low"] = min(frames[-1].loc[last, "Low"], prior_close)
        xs, ys = zip(*waypoints)
        path = np.interp(np.arange(26), xs, ys) + rng.normal(0, 0.06, 26)
        frames.append(bars_from_path(idays[-1], path, rng, 40_000))
        b15 = pd.concat(frames).round(2)
        b30 = b15.groupby(b15.index.floor("30min")).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        b30.index = pd.DatetimeIndex(b30.index, name="start")
        # Daily: a year's walk into the intraday window, then the intraday days aggregated.
        ddays = trading_days(idays[0] - timedelta(days=1), DAILY_SESSIONS - INTRADAY_SESSIONS)
        first_open = float(b15.Open.iloc[0])
        # Walk BACKWARDS from the first intraday open, so the year joins it.
        rets = rng.normal(0.0004, 0.018, len(ddays))
        walk = first_open / np.exp(np.cumsum(rets[::-1]))[::-1]
        rows = []
        for d, c in zip(ddays, walk):
            o = c * (1 + rng.normal(0, 0.006))
            rows.append({"Open": o, "High": max(o, c) * (1 + abs(rng.normal(0, 0.008))),
                         "Low": min(o, c) * (1 - abs(rng.normal(0, 0.008))), "Close": c,
                         "Volume": float(round(1_000_000 * rng.uniform(0.7, 1.3)))})
        early = pd.DataFrame(rows, index=pd.DatetimeIndex(
            [datetime(d.year, d.month, d.day, tzinfo=ET) for d in ddays], name="start"))
        late = b15.groupby(b15.index.date).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        late.index = pd.DatetimeIndex([datetime(d.year, d.month, d.day, tzinfo=ET)
                                       for d in late.index], name="start")
        bd = pd.concat([early, late]).round(2)
        out[sym] = {"15m": b15, "30m": b30.round(2), "1d": bd}
    return out


def from_yfinance(replay: date):
    import yfinance as yf

    end = replay + timedelta(days=1)
    out = {}
    for sym in SYMBOLS:
        t = yf.Ticker(sym)
        out[sym] = {}
        for iv, start in (("15m", replay - timedelta(days=20)),
                          ("30m", replay - timedelta(days=40)),
                          ("1d", replay - timedelta(days=380))):
            df = t.history(start=start.isoformat(), end=end.isoformat(), interval=iv,
                           prepost=False, auto_adjust=True, actions=False)
            if df.empty:
                raise SystemExit(f"no {iv} bars for {sym}")
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.DatetimeIndex(df.index.tz_convert(ET), name="start")
            out[sym][iv] = df.round(4)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    ap.add_argument("--replay-date", default="2026-10-08")
    ap.add_argument("--force", action="store_true", help="overwrite an existing version")
    ap.add_argument("--root", default=FIXTURE_DIR, help="fixtures directory")
    args = ap.parse_args(argv)
    replay = date.fromisoformat(args.replay_date)
    if not is_trading_day(replay):
        raise SystemExit(f"{replay} is not a trading day")
    base = os.path.join(args.root, args.version)
    if os.path.exists(os.path.join(base, "manifest.json")) and not args.force:
        raise SystemExit(f"{base} exists; fixtures are immutable -- pick a new --version")
    os.makedirs(base, exist_ok=True)
    data = synthetic(replay) if args.source == "synthetic" else from_yfinance(replay)
    files = {}
    for sym, by in data.items():
        for iv, df in by.items():
            name = f"{sym}_{iv}.csv"
            out = df.copy()
            out.insert(0, "start", out.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
            out.to_csv(os.path.join(base, name), index=False, lineterminator="\n")
            files[name] = sha256_file(os.path.join(base, name))
    manifest = {
        "version": args.version, "source": args.source,
        "seed": SEED if args.source == "synthetic" else None,
        "real_prices": args.source != "synthetic",
        "replay_date": replay.isoformat(), "symbols": list(SYMBOLS),
        "timezone_of_start": "UTC (bar START time)",
        "scenario": ("TEAM breaks out above 200 around 11:15 ET and retests; MRVL (held, "
                     "9 shares) peaks near noon and loses VWAP after 14:00; TSM pulls back "
                     "to ~294 then recovers above 302 late") if args.source == "synthetic"
        else "real bars",
        "daily_prompt": PROMPT.format(day=replay.isoformat()),
        "files": dict(sorted(files.items())),
    }
    with open(os.path.join(base, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"wrote {len(files)} files to {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
