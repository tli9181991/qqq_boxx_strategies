"""The LLM interface: one method, several providers, no tools.

    provider.complete(system, user, settings) -> LLMResponse

Every provider sends exactly two messages -- the system prompt and one user
message -- and never a tool list. In Phase 1 the model can only answer; it
cannot call anything, so there is nothing for a prompt injection in a news
snippet or a crafted daily prompt to reach. (`tests/test_trading_agent.py`
asserts this against the call payloads.)

Providers
---------
* `gemini`    -- the repo's existing provider, via langchain-google-genai
                 (requirements-agent.txt). Key: GOOGLE_API_KEY / GEMINI_API_KEY.
* `anthropic` -- the `anthropic` SDK (`pip install anthropic`). Key:
                 ANTHROPIC_API_KEY.
* `openai`    -- the `openai` SDK (`pip install openai`). Key: OPENAI_API_KEY.
                 No built-in default model: set QBS_AGENT_MODEL.
* `mock`      -- scripted replies for tests and offline dry runs.

SDKs are imported lazily, so none of them is needed unless selected. Keys
are read from the environment at call time and never logged.

Usage is normalized into `Usage`. A category the provider did not report is
None ("unavailable"), never 0, and `source` says whether the numbers came
from the provider at all.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-5-5",
                  "openai": "", "mock": "mock-model"}


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    source: str = "unavailable"          # "provider" | "unavailable"


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: Optional[str] = None


@dataclass
class LLMSettings:
    model: str
    temperature: Optional[float] = 0.0
    max_output_tokens: int = 1024
    timeout_s: float = 60.0


class LLMError(RuntimeError):
    pass


def _int(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

class GeminiProvider:
    name = "gemini"

    def complete(self, system: str, user: str, s: LLMSettings) -> LLMResponse:
        from langchain_google_genai import ChatGoogleGenerativeAI

        from ..agent.env import resolve_google_key

        key = resolve_google_key()
        if not key:
            raise LLMError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set")
        kw: Dict[str, Any] = dict(model=s.model, max_output_tokens=s.max_output_tokens,
                                  timeout=s.timeout_s, max_retries=0, google_api_key=key,
                                  response_mime_type="application/json")
        if s.temperature is not None:
            kw["temperature"] = s.temperature
        llm = ChatGoogleGenerativeAI(**kw)
        msg = llm.invoke([("system", system), ("human", user)])
        from ..agent.analyst import _stringify

        um = getattr(msg, "usage_metadata", None) or {}
        usage = Usage(
            input_tokens=_int(um.get("input_tokens")),
            output_tokens=_int(um.get("output_tokens")),
            cached_input_tokens=_int((um.get("input_token_details") or {}).get("cache_read")),
            reasoning_tokens=_int((um.get("output_token_details") or {}).get("reasoning")),
            total_tokens=_int(um.get("total_tokens")),
            source="provider" if um else "unavailable")
        meta = getattr(msg, "response_metadata", None) or {}
        return LLMResponse(_stringify(getattr(msg, "content", "")), usage,
                           meta.get("model_name") or s.model, meta.get("finish_reason"))


class AnthropicProvider:
    name = "anthropic"

    def complete(self, system: str, user: str, s: LLMSettings) -> LLMResponse:
        import anthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic(api_key=key, timeout=s.timeout_s, max_retries=0)
        kw: Dict[str, Any] = dict(model=s.model, max_tokens=s.max_output_tokens,
                                  system=system,
                                  messages=[{"role": "user", "content": user}])
        if s.temperature is not None:
            kw["temperature"] = s.temperature
        r = client.messages.create(**kw)
        text = "".join(getattr(b, "text", "") for b in (r.content or [])
                       if getattr(b, "type", "") == "text")
        u = r.usage
        fresh = _int(_get(u, "input_tokens"))
        read = _int(_get(u, "cache_read_input_tokens"))
        write = _int(_get(u, "cache_creation_input_tokens"))
        out = _int(_get(u, "output_tokens"))
        inp = None if fresh is None else fresh + (read or 0) + (write or 0)
        usage = Usage(input_tokens=inp, output_tokens=out, cached_input_tokens=read,
                      reasoning_tokens=None,
                      total_tokens=None if inp is None or out is None else inp + out,
                      source="provider" if u is not None else "unavailable")
        return LLMResponse(text, usage, getattr(r, "model", s.model),
                           getattr(r, "stop_reason", None))


class OpenAIProvider:
    name = "openai"

    def complete(self, system: str, user: str, s: LLMSettings) -> LLMResponse:
        import openai

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set")
        client = openai.OpenAI(api_key=key, timeout=s.timeout_s, max_retries=0)
        kw: Dict[str, Any] = dict(model=s.model, max_completion_tokens=s.max_output_tokens,
                                  response_format={"type": "json_object"},
                                  messages=[{"role": "system", "content": system},
                                            {"role": "user", "content": user}])
        if s.temperature is not None:
            kw["temperature"] = s.temperature
        r = client.chat.completions.create(**kw)
        u = r.usage
        usage = Usage(
            input_tokens=_int(_get(u, "prompt_tokens")),
            output_tokens=_int(_get(u, "completion_tokens")),
            cached_input_tokens=_int(_get(_get(u, "prompt_tokens_details"), "cached_tokens")),
            reasoning_tokens=_int(_get(_get(u, "completion_tokens_details"),
                                       "reasoning_tokens")),
            total_tokens=_int(_get(u, "total_tokens")),
            source="provider" if u is not None else "unavailable")
        choice = r.choices[0]
        return LLMResponse(choice.message.content or "", usage, r.model,
                           choice.finish_reason)


Script = Union[str, LLMResponse, Exception, Callable[[str, str], Any]]


class MockProvider:
    """Scripted replies, in order; the last one repeats. For tests and for a
    no-cost rehearsal of the whole loop (`QBS_AGENT_PROVIDER=mock`).

    An item may be a string, an `LLMResponse`, an exception (raised), or a
    callable `(system, user) -> any of those`. `delay_s` sleeps first, to
    exercise timeouts. Every call is recorded in `calls`.
    """
    name = "mock"

    def __init__(self, script: Optional[Sequence[Script]] = None, delay_s: float = 0.0,
                 usage: Optional[Usage] = None):
        self.script: List[Script] = list(script or [_hold_reply])
        self.delay_s = delay_s
        self.usage = usage
        self.calls: List[Dict[str, Any]] = []

    def complete(self, system: str, user: str, s: LLMSettings) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "settings": s})
        if self.delay_s:
            time.sleep(self.delay_s)
        i = min(len(self.calls) - 1, len(self.script) - 1)
        item = self.script[i]
        if callable(item) and not isinstance(item, Exception):
            item = item(system, user)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        usage = self.usage or Usage(input_tokens=len(system + user) // 4,
                                    output_tokens=len(str(item)) // 4, cached_input_tokens=None,
                                    reasoning_tokens=None, source="provider")
        if usage.total_tokens is None and usage.input_tokens is not None \
                and usage.output_tokens is not None:
            usage = Usage(**{**usage.__dict__,
                             "total_tokens": usage.input_tokens + usage.output_tokens})
        return LLMResponse(str(item), usage, s.model, "stop")


def _hold_reply(system: str, user: str) -> str:
    """The mock's default: a well-formed NO_TRADE for whatever was asked."""
    import json
    import re

    sym = re.search(r'"symbol":\s*"([A-Z.\-]+)"', user)
    ts = re.search(r'"candle_timestamp":\s*"([^"]+)"', user)
    return json.dumps({
        "symbol": sym.group(1) if sym else "", "timestamp": ts.group(1) if ts else "",
        "action": "NO_TRADE", "strategy": "NONE", "confidence": 0.5,
        "market_condition": "UNCLEAR", "entry_price": None, "stop_loss": None,
        "take_profit": None, "suggested_quantity": None,
        "reason": "Mock provider: no analysis performed.",
        "invalidation_condition": "Not applicable (mock).", "risk_flags": ["mock"],
        "memory_update": None})


def build_provider(name: str):
    providers = {"gemini": GeminiProvider, "anthropic": AnthropicProvider,
                 "openai": OpenAIProvider, "mock": MockProvider}
    if name not in providers:
        raise LLMError(f"unknown provider {name!r}")
    return providers[name]()


def resolve_model(provider: str, model: str) -> str:
    if model:
        return model
    if provider == "gemini":
        return os.environ.get("QBS_GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
    return DEFAULT_MODELS.get(provider, "")


# --------------------------------------------------------------------------
# Timeouts and retries
# --------------------------------------------------------------------------

_RETRYABLE = ("timeout", "ratelimit", "rate_limit", "apiconnection", "connection",
              "serviceunavailable", "internalserver", "resourceexhausted",
              "deadlineexceeded", "overloaded", "toomanyrequests")


def is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (isinstance(exc, (TimeoutError, cf.TimeoutError, ConnectionError))
            or any(k in name for k in _RETRYABLE)
            or any(code in text for code in (" 429", " 500", " 502", " 503", " 504",
                                             "rate limit", "overloaded")))


def is_timeout(exc: BaseException) -> bool:
    return (isinstance(exc, (TimeoutError, cf.TimeoutError))
            or "timeout" in type(exc).__name__.lower()
            or "deadline" in type(exc).__name__.lower())


@dataclass
class CallResult:
    response: Optional[LLMResponse]
    attempts: int
    retries: int
    timeout_events: int
    latency_ms: float
    error: Optional[str] = None


def call_with_retries(provider, system: str, user: str, settings: LLMSettings,
                      max_retries: int = 2, backoff_s: float = 2.0,
                      sleep: Callable[[float], None] = time.sleep,
                      should_continue: Callable[[], bool] = lambda: True) -> CallResult:
    """Call with a hard wall-clock timeout and bounded, backed-off retries.

    The timeout is enforced here as well as by the SDK, because an SDK
    timeout covers one socket read and a stalled stream can outlast it.

    **A timeout is never retried.** Python cannot cancel a running thread, so
    a request that outlived the wall clock may still be in flight -- and may
    still complete and be billed. Retrying it would run a second paid request
    concurrently with the first. The abandoned worker is a daemon thread, so
    it cannot hold up process shutdown either. Other transient errors (rate
    limits, 5xx, connection) came back from the provider, so nothing is in
    flight and they are retried with backoff; a bad key or a bad model name
    fails at once. `should_continue` is checked before each retry so a stop
    request is honoured between attempts.
    """
    t0 = time.perf_counter()
    timeouts = 0
    attempts = 0
    last: Optional[str] = None
    for attempt in range(max_retries + 1):
        attempts += 1
        box: Dict[str, Any] = {}
        done = threading.Event()

        def work() -> None:
            try:
                box["resp"] = provider.complete(system, user, settings)
            except BaseException as e:  # noqa: BLE001 -- handed to the caller
                box["exc"] = e
            finally:
                done.set()

        threading.Thread(target=work, name="llm-request", daemon=True).start()
        if not done.wait(settings.timeout_s):
            timeouts += 1
            last = (f"TimeoutError: no response within {settings.timeout_s}s; not "
                    f"retried, the request may still be in flight")
            break
        if "resp" in box:
            return CallResult(box["resp"], attempts, attempts - 1, timeouts,
                              (time.perf_counter() - t0) * 1000)
        exc = box["exc"]
        if is_timeout(exc):
            timeouts += 1
        last = f"{type(exc).__name__}: {exc}"
        if not is_retryable(exc) or attempt == max_retries or not should_continue():
            break
        sleep(backoff_s * (2 ** attempt))
    return CallResult(None, attempts, attempts - 1, timeouts,
                      (time.perf_counter() - t0) * 1000, last)
