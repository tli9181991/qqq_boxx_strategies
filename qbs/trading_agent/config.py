"""Agent configuration: defaults <- JSON file <- environment.

The same precedence as the rest of the repo: the live runner reads a JSON
file and lets the environment override it, and `qbs.agent.env` loads a
`.env` that never beats the shell. Here:

1. the dataclass defaults below (agent OFF, DRY_RUN, 15 minutes);
2. `var/agent/config.json`, or the file named by `QBS_AGENT_CONFIG`
   (see deploy/agent.example.json);
3. `QBS_AGENT_*` environment variables, including lines in `.env`.

`agent_enabled` is the FEATURE switch. It is necessary and never sufficient:
a session also needs the day's one-time authorization (see `session.py`).
The daily user prompt itself lives in the agent database, versioned per
session date, because it changes every day and must be auditable -- it is
not a config value. `daily_user_prompt` here is only a seed for the CLI.

`execution_mode` accepts exactly one value in Phase 1. Anything else makes
the configuration invalid, and an invalid configuration keeps the agent
disabled rather than guessing what was meant.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

# qbs/trading_agent/config.py -> qbs/trading_agent -> qbs -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAR_DIR = os.path.join(REPO_ROOT, "var", "agent")
DEFAULT_CONFIG_PATH = os.path.join(VAR_DIR, "config.json")
DEFAULT_DB_PATH = os.path.join(VAR_DIR, "agent.db")

DRY_RUN = "DRY_RUN"
EXECUTION_MODES = (DRY_RUN,)        # Phase 1: nothing else exists

PROVIDERS = ("gemini", "anthropic", "openai", "mock")

# Same reading as `qbs.agent.env`: only an explicit on-word turns it on.
ON_VALUES = ("1", "true", "yes", "on", "enabled")


@dataclass
class MarketContextConfig:
    timeframe_primary: str = "15m"
    timeframe_confirmation: str = "30m"
    timeframe_daily: str = "1d"
    # Candles the snapshot SUMMARISES (swing levels, ranges, volume). The
    # indicators are computed over everything fetched, so a 32-bar window
    # still gets a warmed-up MACD.
    candles_15m: int = 32
    candles_30m: int = 20
    candles_daily: int = 30
    # Raw OHLCV rows actually SENT to the model. Token efficiency is a Phase
    # 1 objective: the rest goes in as a few summary numbers.
    raw_15m: int = 8
    raw_30m: int = 4
    raw_daily: int = 5


@dataclass
class AgentConfig:
    # --- the four the spec names ------------------------------------------
    agent_enabled: bool = False
    execution_mode: str = DRY_RUN
    analysis_interval_minutes: int = 15
    daily_user_prompt: str = ""

    # --- the model --------------------------------------------------------
    provider: str = "gemini"            # the provider the repo already uses
    model: str = ""                     # "" -> provider default, see llm.py
    temperature: Optional[float] = 0.0  # None -> do not send one
    max_output_tokens: int = 1024
    request_timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 2.0

    # --- scheduling -------------------------------------------------------
    # Seconds after a candle closes before it is analysed, so the provider
    # has published the bar. A knob because providers differ.
    data_delay_seconds: int = 60
    # One retry for a bar that is not in yet, after this many seconds.
    stale_retry_seconds: int = 30
    # The 16:00 candle closes with the session. Off: a decision made after
    # the bell is about tomorrow, which is not what this dry run measures.
    include_closing_candle: bool = False
    # Consecutive cycles in which EVERY symbol failed (data or LLM) before
    # the session stops itself in ERROR. Fail closed on lost connectivity.
    max_consecutive_failed_cycles: int = 3
    # Extra full-day closures / 13:00 closes as YYYY-MM-DD, for anything the
    # built-in NYSE rules cannot know (a national day of mourning).
    extra_holidays: List[str] = field(default_factory=list)
    extra_early_closes: List[str] = field(default_factory=list)

    # --- context and history ---------------------------------------------
    market_context: MarketContextConfig = field(default_factory=MarketContextConfig)
    decision_history: int = 3           # memory layer 3: decisions per stock sent (3-5)
    data_source: str = "yfinance"

    # --- logging ----------------------------------------------------------
    db_path: str = DEFAULT_DB_PATH
    # Full prompt and response text in the log. Off by default: the snapshot
    # and the parsed decision are always kept, which is what the audit needs.
    log_full_prompts: bool = False
    pricing_file: str = ""              # JSON price table, see pricing.py

    # Not settable: where the values came from, for `status`.
    sources: Dict[str, str] = field(default_factory=dict, repr=False)

    def validate(self) -> List[str]:
        """Every problem, as sentences. Empty means usable."""
        errs: List[str] = []
        if self.execution_mode != DRY_RUN:
            errs.append(f"execution_mode must be {DRY_RUN!r} in Phase 1, got "
                        f"{self.execution_mode!r}; no order execution exists")
        if self.provider not in PROVIDERS:
            errs.append(f"provider must be one of {PROVIDERS}, got {self.provider!r}")
        else:
            from .llm import resolve_model

            if not resolve_model(self.provider, self.model):
                errs.append(f"provider {self.provider!r} has no default model; set "
                            f"model (QBS_AGENT_MODEL)")
        if (self.analysis_interval_minutes < 15
                or self.analysis_interval_minutes % 15):
            errs.append("analysis_interval_minutes must be a multiple of 15 "
                        "(the primary candle) and at least 15")
        if self.market_context.timeframe_primary != "15m":
            errs.append("timeframe_primary must be '15m' in Phase 1")
        if self.request_timeout_s <= 0:
            errs.append("request_timeout_s must be positive")
        if self.max_retries < 0:
            errs.append("max_retries cannot be negative")
        if self.max_output_tokens <= 0:
            errs.append("max_output_tokens must be positive")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            errs.append("temperature must be between 0 and 2")
        if not 3 <= self.decision_history <= 5:
            errs.append("decision_history must be 3 to 5 (the rolling memory window)")
        if self.max_consecutive_failed_cycles < 1:
            errs.append("max_consecutive_failed_cycles must be at least 1")
        mc = self.market_context
        for name in ("candles_15m", "candles_30m", "candles_daily",
                     "raw_15m", "raw_30m", "raw_daily"):
            if getattr(mc, name) < 1:
                errs.append(f"market_context.{name} must be at least 1")
        return errs

    def to_public_dict(self) -> Dict[str, Any]:
        """Everything but the bookkeeping. Holds no secrets: keys are never
        configuration here, they are read from the environment at call time."""
        d = asdict(self)
        d.pop("sources", None)
        return d


def _bool(raw: str) -> bool:
    return raw.strip().lower() in ON_VALUES


def _list(raw: str) -> List[str]:
    return [p for p in raw.replace(",", " ").split() if p]


# Environment variable -> (field, parser). The only names this module reads.
ENV_VARS = {
    "QBS_AGENT_ENABLED": ("agent_enabled", _bool),
    "QBS_AGENT_EXECUTION_MODE": ("execution_mode", str),
    "QBS_AGENT_INTERVAL_MINUTES": ("analysis_interval_minutes", int),
    "QBS_AGENT_PROVIDER": ("provider", lambda s: s.strip().lower()),
    "QBS_AGENT_MODEL": ("model", str),
    "QBS_AGENT_TEMPERATURE": ("temperature",
                              lambda s: None if s.strip().lower() in ("", "none") else float(s)),
    "QBS_AGENT_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
    "QBS_AGENT_TIMEOUT_S": ("request_timeout_s", float),
    "QBS_AGENT_MAX_RETRIES": ("max_retries", int),
    "QBS_AGENT_DATA_DELAY_S": ("data_delay_seconds", int),
    "QBS_AGENT_DECISION_HISTORY": ("decision_history", int),
    "QBS_AGENT_DB": ("db_path", str),
    "QBS_AGENT_LOG_FULL_PROMPTS": ("log_full_prompts", _bool),
    "QBS_AGENT_PRICING_FILE": ("pricing_file", str),
    "QBS_AGENT_EXTRA_HOLIDAYS": ("extra_holidays", _list),
    "QBS_AGENT_EXTRA_EARLY_CLOSES": ("extra_early_closes", _list),
}


def load_config(path: Optional[str] = None,
                environ: Optional[Dict[str, str]] = None,
                load_dotenv: bool = True) -> AgentConfig:
    """The effective configuration. Never raises on a bad file or value.

    A problem is recorded in `sources["error:..."]` and the field keeps its
    previous value -- and since the default of `agent_enabled` is False, a
    broken file can never switch the agent on. Call `cfg.validate()` for the
    remaining checks.
    """
    if load_dotenv and environ is None:
        try:
            from ..agent.env import load_env
            load_env()
        except Exception:  # noqa: BLE001 -- .env is a convenience, not a need
            pass
    env = os.environ if environ is None else environ
    cfg = AgentConfig()
    path = path or env.get("QBS_AGENT_CONFIG") or DEFAULT_CONFIG_PATH

    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            _apply_file(cfg, raw, path)
        except (OSError, ValueError, TypeError) as exc:
            cfg.sources["error:file"] = f"{path}: {exc}"
            cfg.agent_enabled = False

    for var, (name, parse) in ENV_VARS.items():
        raw = env.get(var)
        if raw is None or raw == "":
            continue
        try:
            setattr(cfg, name, parse(raw))
            cfg.sources[name] = f"env {var}"
        except (TypeError, ValueError) as exc:
            cfg.sources[f"error:{var}"] = f"could not read {var}: {exc}"
            if name == "agent_enabled":
                cfg.agent_enabled = False
    if not os.path.isabs(cfg.db_path):
        cfg.db_path = os.path.join(REPO_ROOT, cfg.db_path)
    return cfg


# Fields whose default is None but which hold a number when set.
_OPTIONAL_FLOAT = {"temperature"}


def _set_checked(cfg: AgentConfig, target: Any, name: str, value: Any, label: str) -> bool:
    """Assign a file value only if it has the field's declared type.

    JSON from a hand-edited file is not trusted: "false" is a truthy string,
    "60" makes a later comparison raise. A wrong type is recorded as an error
    and the default stays -- and since every safety default here is the
    disabled position, a bad value can only switch something OFF.
    """
    current = getattr(target, name)
    ok = False
    if name in _OPTIONAL_FLOAT:
        ok = value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
        value = None if value is None else (float(value) if ok else value)
    elif isinstance(current, bool):
        ok = isinstance(value, bool)
    elif isinstance(current, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(current, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif isinstance(current, str):
        ok = isinstance(value, str)
    elif isinstance(current, list):
        ok = isinstance(value, list) and all(isinstance(x, str) for x in value)
    if ok:
        setattr(target, name, value)
    else:
        cfg.sources[f"error:{label}"] = (f"expected {type(current).__name__}, got "
                                         f"{type(value).__name__} {value!r}; ignored")
    return ok


def _apply_file(cfg: AgentConfig, raw: Dict[str, Any], path: str) -> None:
    if not isinstance(raw, dict):
        raise TypeError("the config file must hold a JSON object")
    names = {f.name for f in fields(AgentConfig)} - {"sources", "market_context"}
    for key, value in raw.items():
        if key == "market_context" and isinstance(value, dict):
            mc_names = {f.name for f in fields(MarketContextConfig)}
            history = value.get("history") or {}
            flat = {**{k: v for k, v in value.items() if k != "history"}, **history}
            for k, v in flat.items():
                if k in mc_names:
                    _set_checked(cfg, cfg.market_context, k, v, f"market_context.{k}")
                else:
                    cfg.sources[f"error:market_context.{k}"] = "unknown key, ignored"
            cfg.sources["market_context"] = f"file {path}"
        elif key in names:
            if _set_checked(cfg, cfg, key, value, key):
                cfg.sources[key] = f"file {path}"
        elif not key.startswith("_"):
            cfg.sources[f"error:{key}"] = "unknown key, ignored"
    if not isinstance(cfg.agent_enabled, bool):
        cfg.agent_enabled = False
