"""Test harness for the agent's pre-deployment notebooks and their CI tests.

Everything the notebooks in notebooks/agent_tests/ need that is NOT agent
logic lives here, so the notebooks call production modules (`engine`,
`runner`, `session`, `memory`, `market_context`, `evaluation`) and only
those, and the same harness is exercised by tests/test_agent_notebooks.py.

    load_fixture()      versioned, hash-checked historical bars
    FixtureBars         a MarketDataProvider over a fixture that reveals only
                        candles COMPLETED by the simulated clock -- no
                        look-ahead, by construction, and it records what it
                        revealed so a test can prove it
    Clock               a controllable clock whose sleep() advances it
    test_mode()         MOCK (default) or REAL_LLM, from QBS_AGENT_TEST_MODE
    make_provider()     MockProvider with a deterministic rule-based analyst,
                        or a real provider behind a CallBudget
    workspace()         an isolated AgentConfig + temp database; never reads
                        or writes the production config or var/agent/agent.db
    save_results()      CSV/JSON into notebooks/agent_tests/results/ (git-ignored)

Paid calls need BOTH `QBS_AGENT_TEST_MODE=REAL_LLM` and
`QBS_AGENT_TEST_ALLOW_PAID=1`, and stop at `QBS_AGENT_TEST_MAX_CALLS`
(default 20). Anything else is MOCK, with zero external requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import REPO_ROOT, AgentConfig
from .llm import LLMError, LLMResponse, MockProvider, build_provider, resolve_model
from .market_context import OHLCV
from .pricing import estimate_cost, load_prices
from .session_calendar import ET

NB_DIR = os.path.join(REPO_ROOT, "notebooks", "agent_tests")
FIXTURE_DIR = os.path.join(NB_DIR, "fixtures")
# Overridable so CI can execute the notebooks without writing into the repo.
RESULTS_DIR = os.environ.get("QBS_AGENT_TEST_RESULTS") or os.path.join(NB_DIR, "results")
DEFAULT_FIXTURE = "v1"

MOCK, REAL_LLM = "MOCK", "REAL_LLM"
MODE_VAR, PAID_VAR, MAX_CALLS_VAR = ("QBS_AGENT_TEST_MODE", "QBS_AGENT_TEST_ALLOW_PAID",
                                     "QBS_AGENT_TEST_MAX_CALLS")
INTERVALS = {"15m": timedelta(minutes=15), "30m": timedelta(minutes=30)}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Fixture:
    version: str
    manifest: Dict[str, Any]
    bars: Dict[Tuple[str, str], pd.DataFrame]       # (symbol, interval) -> OHLCV

    @property
    def replay_date(self) -> date:
        return date.fromisoformat(self.manifest["replay_date"])

    @property
    def symbols(self) -> List[str]:
        return list(self.manifest["symbols"])

    @property
    def prompt(self) -> str:
        return self.manifest["daily_prompt"]


def read_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx = pd.to_datetime(df.pop("start"), utc=True).dt.tz_convert(ET)
    df.index = pd.DatetimeIndex(idx, name="start")
    return df[OHLCV].astype(float)


def load_fixture(version: str = DEFAULT_FIXTURE, root: str = FIXTURE_DIR,
                 verify: bool = True) -> Fixture:
    """Load a fixture and check every file against the manifest's SHA-256.

    A fixture that does not match its manifest is refused: a "reproducible"
    test on silently edited data reproduces nothing.
    """
    base = os.path.join(root, version)
    with open(os.path.join(base, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    bars = {}
    for name, digest in manifest["files"].items():
        path = os.path.join(base, name)
        if verify and sha256_file(path) != digest:
            raise ValueError(f"fixture {version}/{name} does not match its manifest hash")
        sym, interval = name[:-4].rsplit("_", 1)
        bars[(sym, interval)] = read_bars(path)
    return Fixture(version, manifest, bars)


class LookAheadError(AssertionError):
    pass


class FixtureBars:
    """Serves a fixture as a live feed would, minus the future.

    Intraday: only bars whose END is at or before the clock. (A live feed
    also shows the bar in progress -- but a fixture only knows that bar's
    FINAL values, so revealing it would leak the rest of the candle.)
    Daily: only sessions before the clock's date, for the same reason.

    `withhold` drops specific bars to simulate a feed that is late
    ({(symbol, bar_end_iso), ...}); `missing` drops symbols entirely.
    `revealed_max` records the latest bar END ever served, for assertions.
    """

    def __init__(self, fixture: Fixture, clock, withhold=(), missing=()):
        self.fixture = fixture
        self.clock = clock
        self.withhold = set(withhold)
        self.missing = set(missing)
        self.name = f"fixture:{fixture.version}"
        self.revealed_max: Optional[datetime] = None
        self.calls = 0

    def bars(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        self.calls += 1
        if symbol in self.missing:
            return None
        df = self.fixture.bars.get((symbol, interval))
        if df is None:
            return None
        now = self.clock().astimezone(ET)
        if interval == "1d":
            out = df[df.index.date < now.date()]
            ends = out.index + timedelta(days=1)
        else:
            ends = df.index + INTERVALS[interval]
            out = df[ends <= now]
            ends = out.index + INTERVALS[interval]
            if self.withhold:
                keep = [not ((symbol, e.isoformat()) in self.withhold) for e in ends]
                out = out[keep]
                ends = ends[keep]
        if len(out):
            last_end = ends.max().to_pydatetime()
            if interval != "1d" and last_end > now:
                raise LookAheadError(f"{symbol} {interval} bar ending {last_end} served at {now}")
            if interval != "1d":
                self.revealed_max = max(filter(None, [self.revealed_max, last_end]))
        return out.copy()


class Clock:
    """`clock()` reads it, `clock.sleep(s)` advances it. tz-aware."""

    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("Clock needs a tz-aware start")
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def set(self, when: datetime) -> None:
        self.now = when


def et(day: date, hh: int, mm: int) -> datetime:
    return datetime(day.year, day.month, day.day, hh, mm, tzinfo=ET)


# --------------------------------------------------------------------------
# Modes and providers
# --------------------------------------------------------------------------

def test_mode(environ: Optional[Dict[str, str]] = None) -> str:
    """MOCK unless REAL_LLM is asked for AND paid calls are allowed."""
    env = os.environ if environ is None else environ
    want = (env.get(MODE_VAR) or MOCK).strip().upper()
    if want != REAL_LLM:
        return MOCK
    if (env.get(PAID_VAR) or "").strip().lower() not in ("1", "true", "yes"):
        raise RuntimeError(f"{MODE_VAR}=REAL_LLM makes paid API calls; set {PAID_VAR}=1 "
                           f"to confirm (and {MAX_CALLS_VAR} to cap them).")
    return REAL_LLM


def max_calls(environ: Optional[Dict[str, str]] = None) -> int:
    env = os.environ if environ is None else environ
    try:
        return max(0, int(env.get(MAX_CALLS_VAR) or 20))
    except ValueError:
        return 20


class BudgetExceeded(LLMError):
    pass


class CallBudget:
    """Wraps a real provider: refuses call `limit + 1` before it is sent."""

    def __init__(self, inner, limit: int):
        self.inner, self.limit, self.used = inner, limit, 0
        self.name = getattr(inner, "name", "provider")

    def complete(self, system, user, settings):
        if self.used >= self.limit:
            raise BudgetExceeded(f"paid-call budget of {self.limit} reached "
                                 f"({MAX_CALLS_VAR})")
        self.used += 1
        return self.inner.complete(system, user, settings)


def cost_preview(cfg: AgentConfig, n_calls: int, input_tokens: int = 2200,
                 output_tokens: int = 250) -> str:
    """A one-line estimate shown BEFORE any paid run."""
    model = resolve_model(cfg.provider, cfg.model)
    c = estimate_cost(model, input_tokens * n_calls, output_tokens * n_calls,
                      prices=load_prices(cfg.pricing_file))
    money = f"~${c.total_cost:.4f}" if c.total_cost is not None else "unknown (no price)"
    return (f"{n_calls} call(s) to {cfg.provider}/{model} at ~{input_tokens} in / "
            f"~{output_tokens} out tokens each: estimated {money}")


_SYM = re.compile(r'"symbol": "([A-Z.\-]+)"')
_TS = re.compile(r'"candle_timestamp": "([^"]+)"')


def _section(user: str, tag: str) -> Dict[str, Any]:
    m = re.search(rf"<{tag}[^>]*>\n(.*?)\n</{tag}>", user, re.S)
    return json.loads(m.group(1)) if m else {}


def rule_based_reply(system: str, user: str) -> str:
    """A deterministic stand-in analyst for MOCK mode.

    Reads the same request a real model gets and applies fixed rules, so a
    replay produces BUY / HOLD / SELL / NO_TRADE and memory transitions
    without any network. It is a harness, not a strategy: its only job is to
    exercise the production pipeline reproducibly.
    """
    sym = _SYM.search(user).group(1)
    ts = _TS.search(user).group(1)
    ctx = _section(user, "market_context")
    mem = _section(user, "trading_memory")
    m = re.search(r"ENFORCED CONSTRAINTS \(checked in code\)\n(\{.*?\})\n", user)
    cons = json.loads(m.group(1)) if m else {}
    port = re.search(r"PORTFOLIO \(real holdings, read-only; null means unknown\)\n(.*?)\n", user)
    held = json.loads(port.group(1)) if port else None
    last = ctx.get("last_price")
    tf = ctx.get("tf_15m") or {}
    ema9, ema21, vwap = tf.get("ema9"), tf.get("ema21"), (ctx.get("today") or {}).get("vwap")
    levels = [x["level"] for x in ctx.get("prompt_levels") or [] if x.get("level")]
    state = mem.get("state", "WATCHING")
    out = dict(symbol=sym, timestamp=ts, action="NO_TRADE", strategy="NONE", confidence=0.5,
               market_condition="NEUTRAL", entry_price=None, stop_loss=None, take_profit=None,
               suggested_quantity=None, reason="No setup on the latest candle.",
               invalidation_condition="Not applicable.", risk_flags=[], memory_update=None)
    up = None not in (last, ema9, ema21, vwap) and last > ema9 > ema21 and last > vwap
    down = None not in (last, ema9, vwap) and last < ema9 and last < vwap
    out["market_condition"] = "BULLISH" if up else "BEARISH" if down else "NEUTRAL"
    breakout = max(levels) if levels else None
    if held and held.get("quantity"):
        qty = int(held["quantity"])
        if down and state == "MANAGING_POSITION":
            out.update(action="SELL", strategy="POSITION_MANAGEMENT", suggested_quantity=qty // 3 or 1,
                       confidence=0.55, reason="Price lost VWAP and the 15m EMA9; trim.",
                       invalidation_condition="Reclaims VWAP.",
                       memory_update={"proposed_state": "EXIT_SIGNALED",
                                      "observations": ["Lost VWAP on the 15m."],
                                      "key_levels": [], "thesis_invalidated": False})
        else:
            out.update(action="HOLD", strategy="POSITION_MANAGEMENT", confidence=0.6,
                       reason="Position trend intact; no exit signal.",
                       invalidation_condition="15m close below VWAP.")
    elif breakout and last and last > breakout and up and state in ("WATCHING", "SETUP_FORMING",
                                                                     "INVALIDATED"):
        stop = cons.get("reference_stop") or round(breakout * 0.98, 2)
        cap = cons.get("max_position_shares") or 1
        out.update(action="BUY", strategy="BREAKOUT", confidence=0.65, entry_price=last,
                   stop_loss=min(stop, round(last * 0.99, 2)), suggested_quantity=cap,
                   reason=f"15m close above {breakout} with EMA9 > EMA21 and above VWAP.",
                   invalidation_condition=f"15m close back below {breakout}.",
                   memory_update={"proposed_state": "SIGNAL_LONG",
                                  "observations": [f"Broke {breakout} on {ts[11:16]} ET."],
                                  "key_levels": [{"price": breakout, "label": "breakout"}],
                                  "thesis_invalidated": False})
    elif state == "SIGNAL_LONG":
        out.update(action="HOLD", strategy="BREAKOUT", confidence=0.6,
                   reason="Breakout holding.", invalidation_condition="Close below breakout.")
    elif breakout and last and last > breakout * 0.99 and state == "WATCHING":
        out.update(memory_update={"proposed_state": "SETUP_FORMING",
                                  "observations": [f"Within 1% of {breakout}."],
                                  "key_levels": [], "thesis_invalidated": False},
                   reason="Pressing the breakout level; wait for a close above.")
    return json.dumps(out)


def make_provider(mode: str, cfg: AgentConfig, script=None, limit: Optional[int] = None):
    """MOCK: MockProvider (rule-based unless `script` is given). REAL_LLM: the
    configured provider behind a CallBudget."""
    if mode == MOCK:
        return MockProvider(script or [rule_based_reply])
    return CallBudget(build_provider(cfg.provider), max_calls() if limit is None else limit)


# --------------------------------------------------------------------------
# Isolation and results
# --------------------------------------------------------------------------

@dataclass
class Workspace:
    cfg: AgentConfig
    dir: str
    mode: str
    notes: List[str] = field(default_factory=list)


def workspace(mode: str = MOCK, **overrides) -> Workspace:
    """A fresh AgentConfig on a temp database. Built from defaults, NOT from
    `load_config`, so the production config file and var/agent/agent.db are
    never read or written."""
    d = tempfile.mkdtemp(prefix="qbs_agent_test_")
    cfg = AgentConfig(agent_enabled=True, provider="mock" if mode == MOCK else
                      (os.environ.get("QBS_AGENT_PROVIDER") or "gemini"),
                      model="" if mode == MOCK else os.environ.get("QBS_AGENT_MODEL", ""),
                      db_path=os.path.join(d, "agent.db"), data_delay_seconds=0,
                      stale_retry_seconds=0, retry_backoff_s=0.0,
                      request_timeout_s=5.0 if mode == MOCK else 60.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return Workspace(cfg, d, mode)


def save_results(name: str, data, results_dir: str = RESULTS_DIR) -> str:
    """Write a DataFrame (CSV) or dict/list (JSON), stamped, never into fixtures."""
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if isinstance(data, pd.DataFrame):
        path = os.path.join(results_dir, f"{name}_{stamp}.csv")
        data.to_csv(path, index=False)
    else:
        path = os.path.join(results_dir, f"{name}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    return path


def broker_modules_loaded(before: set) -> List[str]:
    """Broker/live-trading modules imported since `before` (a sys.modules snapshot)."""
    import sys

    return sorted(m for m in set(sys.modules) - before
                  if m.split(".")[0] in ("ib_async", "ib_insync", "ibapi")
                  or m.startswith("qbs.live"))


class Check:
    """Collects PASS/FAIL lines so a notebook reports every failure clearly,
    then raises once at the end if any failed."""

    def __init__(self, title: str):
        self.title, self.rows = title, []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": name, "result": "PASS" if ok else "FAIL",
                          "detail": detail})
        return ok

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def raise_if_failed(self) -> None:
        bad = [r for r in self.rows if r["result"] == "FAIL"]
        if bad:
            raise AssertionError(f"{self.title}: {len(bad)} check(s) failed: "
                                 + "; ".join(f"{r['check']} ({r['detail']})" for r in bad))


def response_with_usage(text: str, input_tokens=None, output_tokens=None) -> LLMResponse:
    from .llm import Usage

    return LLMResponse(text, Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                                   total_tokens=(input_tokens + output_tokens)
                                   if None not in (input_tokens, output_tokens) else None,
                                   source="provider" if input_tokens is not None
                                   else "unavailable"), "mock-model", "stop")
