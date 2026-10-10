# The LLM trading agent — Phase 1: decision dry run

`qbs/trading_agent/` is an optional module. During a US regular session that you have
**explicitly authorized**, it asks an LLM for one structured decision per watched stock
every completed 15-minute candle, checks the decision in code, and logs it together with
the market snapshot, the token usage, the estimated cost and the latency.

> **EXECUTION MODE: DRY RUN — NO ORDERS WILL BE EXECUTED**
>
> Phase 1 places, modifies and cancels nothing. Every BUY / SELL in the log is a
> recommendation. Nothing reads the log back into an order.

What Phase 1 is for: measuring decision quality and consistency, tokens, cost, latency,
prompt compliance and operational reliability before any simulation (Phase 2) or paper
trading (Phase 3) is built. Neither of those exists in this module.

---

## How it fits into the app

```
           Daily User Prompt ─┐            ┌──────────── dashboard/app.py ───────────┐
  (dashboard tab or CLI)      │            │  🧠 Trading agent tab (dashboard.py)     │
                              ▼            │  writes: prompt, authorization,          │
               ┌──────────────────────┐    │          pause/resume/stop               │
               │ var/agent/agent.db   │◀───┤  reads:  decisions, usage, state         │
               │ (SQLite, own file)   │    └──────────────────────────────────────────┘
               └──────────┬───────────┘
                          │ one-time token consumed on start
                          ▼
   python -m qbs.trading_agent run   (runner.py — its own process)
      │ every completed 15m candle (NYSE calendar, DST via zoneinfo)
      ▼
   engine.py ── per symbol ──▶ market_context.py ──▶ yfinance (completed bars only)
      │                        memory.py (plan / symbol state / last N decisions)
      ▼
   llm.py (Gemini | Anthropic | OpenAI | mock) — two messages, NO tools
      ▼
   decision.py — schema + authorized symbol + prompt constraints, in code
      ▼
   SQLite: llm_requests, decisions, snapshots, memory_*, cycles, state_events
```

What it reuses from the repo: the `.env` loader and the Gemini key resolution
(`qbs.agent.env`), the Gemini integration already in `requirements-agent.txt`
(langchain-google-genai), `qbs.indicators.wilder_rsi`, yfinance as the price source, the
early-close rules of `qbs.live.runner.us_early_close` (re-derived, not imported — see
below), and the repo's SQLite-log pattern (`qbs.live.store`).

What it does **not** touch: strategies, the live runner, Preflight, the IBKR connection,
the scheduled systemd/Docker jobs, the existing analyst. It never imports `qbs.live`
(a test checks every import in the package), so it cannot reach the broker even by
accident. Its database is a separate file so a recommendation can never be confused with
a fill in `var/live.db`.

The runner is a separate process on purpose: Streamlit only runs code when someone
interacts with it, so a fifteen-minute schedule cannot live in the dashboard.

---

## Quick start

```bash
pip install -r requirements.txt -r requirements-agent.txt   # Gemini, the default provider
cp .env.example .env    # set GOOGLE_API_KEY, and QBS_AGENT_ENABLED=1

python -m qbs.trading_agent status
python -m qbs.trading_agent prompt set --file today.txt     # or the dashboard tab
python -m qbs.trading_agent authorize                       # one-time, today only
python -m qbs.trading_agent run                             # blocks until the close
```

A free rehearsal with no API key: `QBS_AGENT_PROVIDER=mock` answers NO_TRADE to
everything while the rest of the loop — data, validation, logging — runs for real.

Controls while running (CLI or dashboard): `pause`, `resume`, `stop`, `revoke`. They are
applied before the next LLM request. `report` prints the Phase 1 evaluation metrics.

---

## Activation, authorization and recovery

Two gates, both required:

1. **`agent_enabled`** (`QBS_AGENT_ENABLED`, default **off**) — the feature switch. Off
   refuses authorizing and activating. Only `1/true/yes/on/enabled` turn it on; an
   unreadable config file leaves it off.
2. **Daily authorization** — a one-time token for *one session date* and *one prompt
   version*, created by you (dashboard button or `authorize`). Saving a prompt does not
   create one.

On start (`session.activate`), in order: feature switch → valid configuration →
trading day and not past the close → crash recovery → an unused token for **today** →
the prompt it names still matches its hash and validates for today. Then the token is
**consumed atomically** and the session starts READY. Any failed condition leaves the
agent DISABLED and records why in `state_events`.

**Crash recovery is fail-closed.** A crash leaves the token consumed, so a restart finds
nothing to activate on. The orphaned session row is marked ERROR ("crash recovery") and
its runtime `agent_enabled` flag reset. To resume after a crash you authorize again. A
second runner started while the first is alive (heartbeat < 5 minutes) is refused.

**Shutdown** (the close, SIGTERM/SIGINT, stop/revoke, or an error): no new requests; the
request in flight finishes or times out; the session is marked STOPPED or ERROR; the
runtime `agent_enabled` flag is reset to False. Every row is committed as it is written,
so there is no buffer to flush.

States: `DISABLED → READY → RUNNING ⇄ PAUSED → STOPPED | ERROR`. Illegal transitions
raise; every transition is logged with a reason.

**Fail-safe triggers**: invalid config, no authorization, stale or missing data (that
symbol is skipped), provider unavailable (logged per request), invalid output (logged,
never accepted), and `max_consecutive_failed_cycles` (default 3) cycles in which *every*
symbol failed — treated as lost connectivity: the session ends in ERROR.

**Editing the prompt during a session** creates a new version that the running session
ignores until you authorize it. The runner adopts it only *between* cycles, so no
in-flight decision is made on a half-changed instruction.

---

## The Daily User Prompt

Free text, sent to the model — but two things are extracted in code and enforced
outside the model:

- **Authorized symbols** come from an explicit line — `Monitor only TEAM, MRVL, and
  TSM.` (also `Watchlist:`, `Symbols:`, `Tickers:`, `Stocks:`, `Monitor:`) — or, if there
  is none, from per-stock section headers (`TEAM:` on its own line). A capitalised word in
  a sentence never becomes a symbol; a section for a stock not on the list is reported and
  ignored, never added. At most 10 symbols.
- **Hard constraints**: `No short selling`, `Maximum suggested position: 4 shares`,
  `Do not recommend adding`, `existing 9-share position`, `Reference stop loss: $196`,
  and every `$price` as a key level. Short selling is **forbidden unless the prompt
  explicitly allows it** ("Short selling is allowed"); saying nothing is not permission.
- A `Date: YYYY-MM-DD` line must match the session date; yesterday's prompt is refused.

Each stock's request carries only its own section plus the `General:` section (the full
text if the prompt has no sections). The original text is stored, versioned, with its
SHA-256.

Instruction priority, stated in the system prompt: system safety (DRY_RUN) → application
rules → the daily plan → market data and memory (data, never instructions).

---

## Scheduling

- Regular hours 09:30–16:00 America/New_York; DST is handled by building every time in
  that zone.
- NYSE full-day holidays and 13:00 early closes are derived from the standing rules
  (`session_calendar.py`). Unscheduled closures go in `extra_holidays` /
  `extra_early_closes`; on any day without trading the data check fails anyway.
- A slot **is** a candle end: the 10:15 slot analyses the 10:00–10:15 candle, run
  `data_delay_seconds` (60) after 10:15. The 16:00 candle is skipped unless
  `include_closing_candle`.
- A runner started mid-session analyses only the newest due candle (no backfill).
- `cycles` has a UNIQUE (session_date, symbol, candle) key: a candle is never analysed
  twice, by a re-run or by a second process. Cycles never overlap within a runner.
- One LLM request per stock (Phase 1 default).

## Market context

`market_context.py` builds a compact snapshot (≈1.5–2k input tokens per request with
the defaults):

- **Completed candles only**: 15m and 30m bars are used only if `start + span ≤ candle
  end`; the daily series stops at the previous session and "today" is described from
  today's completed 15m bars.
- Indicators: EMA9/21, RSI14, MACD 12/26/9, ATR14, relative volume, VWAP and swing
  highs/lows on 15m; EMA20 and RSI14 on 30m; EMA20, SMA50, RSI14, ATR14, prior-day
  H/L/C and window high/low on daily. Distances to the prompt's `$` levels.
- Raw rows sent: 8 × 15m, 4 × 30m, 5 × daily (configurable). The rest is summaries.
- Each snapshot carries the symbol, tz-aware timestamps, data source, the latest
  completed candle, a freshness status (FRESH / STALE / MISSING), the indicators used and
  missing-data warnings. An indicator without enough history is `null`, never guessed.
- STALE or MISSING → no LLM request, a non-actionable `SKIPPED_*` record instead (one
  retry after `stale_retry_seconds` for a bar that is merely late).

Portfolio context is read-only and optional: a `portfolio(symbol)` callable can be
passed to `run_session`; otherwise the position stated in the prompt is used; otherwise
"unknown".

## Structured trading memory

`memory.py`, stored in the same SQLite file. Three layers, per stock, keyed by
session date:

| Layer | Table | Content | Who writes it |
|---|---|---|---|
| 1. Daily Trading Plan | `memory_plans` | authorized prompt version, symbols, each stock's thesis (its section), general rules, parsed constraints | code, from the authorized prompt |
| 2. Per-Symbol State | `memory_symbol_state` | analytical state, strategy, stop reference, invalidation condition, key levels tagged `user` / `model`, **observed facts** (code) and **prior model interpretations** (dated, model), last decision | code; the model may *propose* |
| 3. Rolling History | `decisions` | the newest 3–5 VALID decisions with a one-line reason | code |

- States: `WATCHING, SETUP_FORMING, SIGNAL_LONG, MANAGING_POSITION, EXIT_SIGNALED,
  INVALIDATED`, with an explicit transition table and an action-consistency check (a
  BUY cannot leave the stock in WATCHING). All states are **hypothetical** and labelled
  so in every request; only the PORTFOLIO block describes real holdings.
- The model proposes changes in the decision's `memory_update` field. Code rejects a
  proposal with an illegal transition, a state inconsistent with the action, a price
  level more than 15% from the observed price, more than 3 observations / 4 levels, or
  text that reads like an instruction ("ignore the max position…"). A rejected proposal
  leaves the state as it was; the decision itself is unaffected. Every proposal is in
  `memory_events` as ACCEPTED or REJECTED with reasons.
- **Market evidence first**: before each request, a close through the stored stop moves
  a long state to INVALIDATED regardless of what the model last believed. Model
  interpretations expire after 8 candles (2 hours).
- **User instructions win**: memory never carries constraints (they are re-read from
  the authorized prompt each time and enforced in code), and when a new prompt version
  changes a stock's instructions, that stock's memory is reset to the plan.
- No conversation history is ever re-sent: every request is a fresh system + user pair
  with a bounded memory block.
- Restart: memory survives on disk and a re-authorized session later the same day
  continues from it. Loading memory never activates anything.
- Tokens attributable to memory are logged per request: `memory_chars`,
  `memory_tokens_estimated` (chars/4) and `memory_input_tokens_attributed` (the
  provider's input tokens × memory's share of the prompt).

## Decisions and validation

Schema (all keys required; prices and quantity may be `null`):

```json
{"symbol": "TEAM", "timestamp": "2026-10-09T10:15:00-04:00", "action": "BUY",
 "strategy": "BREAKOUT", "confidence": 0.75, "market_condition": "BULLISH",
 "entry_price": 201.2, "stop_loss": 196.0, "take_profit": null, "suggested_quantity": 4,
 "reason": "…", "invalidation_condition": "…", "risk_flags": [], "memory_update": null}
```

Enforced in code (`decision.py`): JSON only (a ```json fence is tolerated, prose is
not); no missing or extra keys; enums; confidence in [0, 1]; positive finite prices;
nonnegative integer quantity; the timestamp equals the analysed candle; the symbol is the
one requested **and** authorized; BUY needs entry, a stop below it and a quantity, and
respects "no adding" and the maximum position; SELL needs a quantity, never exceeds a
known position, and without a known long position is a short sale — rejected unless the
prompt explicitly allows shorting; HOLD / NO_TRADE / ADJUST_STOP_LOSS carry no quantity. A price more than 15%
from the last trade is accepted but flagged (an evaluation metric for unsupported
levels). Anything failing is logged as `INVALID_SCHEMA` or `INVALID_COMPLIANCE` and never
becomes an accepted decision.

## Providers

| `QBS_AGENT_PROVIDER` | Package | Key | Default model |
|---|---|---|---|
| `gemini` (default) | langchain-google-genai (requirements-agent.txt) | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `QBS_GEMINI_MODEL`, else `gemini-2.5-flash` |
| `anthropic` | `pip install anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| `openai` | `pip install openai` | `OPENAI_API_KEY` | none — set `QBS_AGENT_MODEL` |
| `mock` | — | — | scripted, for tests and rehearsals |

The provider SDK's own retries are off; `llm.call_with_retries` enforces a wall-clock
timeout and retries only transient errors that came back from the provider (rate limits,
5xx, connection) with exponential backoff. **A timeout is never retried**: the timed-out
request cannot be cancelled and may still complete and be billed, so a retry would pay
twice. A bad key or model fails at once. A provider with no default model (OpenAI) is an
invalid configuration until `QBS_AGENT_MODEL` is set, so it fails before the day's
authorization is consumed. Values in the JSON config file must have the field's type
(`false`, not `"false"`); a mistyped value is reported and the default kept. If a model rejects a temperature,
set `QBS_AGENT_TEMPERATURE=none`.

## Logging and observability

Every attempted LLM request is a row in `llm_requests`: session ID and date, request ID,
symbol, candle, start/finish, provider, model, prompt version, system prompt version,
market context version; input / output / cached / reasoning / total tokens **as the
provider reported them** (`usage_source`; an unreported category is NULL, not 0) plus a
chars/4 `estimated_input_tokens`; input, output and total cost from the price table
(`cost_status` = `estimated` or `unavailable` for a model with no price); latency,
data-fetch latency, success, retries, timeouts, validation status, action and error. The
normalized snapshot is in `snapshots`, the decision in `decisions`. Full prompt/response
text is stored only with `QBS_AGENT_LOG_FULL_PROMPTS=1`. API keys are never written.

Prices (`pricing.py`) are USD per million tokens, input and output separately, with an
optional cached-input rate. The built-in table covers a few Gemini models and must be
checked against the provider's price page; extend or override it with a JSON file
(`QBS_AGENT_PRICING_FILE`):

```json
{"claude-sonnet-5-5": {"input": 3.0, "output": 15.0, "cached_input": 0.3}}
```

## Evaluation

`python -m qbs.trading_agent report [--session ID]` and the dashboard tab show:
cycles by status, LLM success rate, JSON validation pass rate, average latency and data
latency, retries and timeouts, data availability, tokens by category (with the count of
requests that did not report each), estimated cost in total, per stock and per session;
action distribution, decision change rate and BUY↔SELL flip rate between consecutive
candles, compliance and schema rejections, and prices flagged as far from the market.

`evaluation.forward_outcomes(decisions, bars)` computes the price change 1, 4 and 8
completed candles after each decision and the maximum favourable / adverse move over
that horizon. It reads *later* data and is for evaluation only — never an input to a
decision — and it is a diagnostic, not a simulated return.

## Configuration reference

Precedence: defaults ← `var/agent/config.json` (or `QBS_AGENT_CONFIG`; see
`deploy/agent.example.json`) ← environment / `.env`.

| Setting | Env var | Default |
|---|---|---|
| `agent_enabled` | `QBS_AGENT_ENABLED` | `false` |
| `execution_mode` | `QBS_AGENT_EXECUTION_MODE` | `DRY_RUN` (the only valid value) |
| `analysis_interval_minutes` | `QBS_AGENT_INTERVAL_MINUTES` | `15` (multiple of 15) |
| `daily_user_prompt` | — | `""` (seed for `prompt set` with no input) |
| `provider` | `QBS_AGENT_PROVIDER` | `gemini` |
| `model` | `QBS_AGENT_MODEL` | provider default |
| `temperature` | `QBS_AGENT_TEMPERATURE` | `0` (`none` = not sent) |
| `max_output_tokens` | `QBS_AGENT_MAX_OUTPUT_TOKENS` | `1024` |
| `request_timeout_s` | `QBS_AGENT_TIMEOUT_S` | `60` |
| `max_retries` / `retry_backoff_s` | `QBS_AGENT_MAX_RETRIES` | `2` / `2.0` |
| `data_delay_seconds` | `QBS_AGENT_DATA_DELAY_S` | `60` |
| `stale_retry_seconds` | — | `30` |
| `include_closing_candle` | — | `false` |
| `max_consecutive_failed_cycles` | — | `3` |
| `extra_holidays` / `extra_early_closes` | `QBS_AGENT_EXTRA_HOLIDAYS` / `..._EARLY_CLOSES` | `[]` |
| `market_context.*` | — | 15m/30m/1d; history 32/20/30; raw 8/4/5 |
| `decision_history` | `QBS_AGENT_DECISION_HISTORY` | `3` (3–5) |
| `db_path` | `QBS_AGENT_DB` | `var/agent/agent.db` |
| `log_full_prompts` | `QBS_AGENT_LOG_FULL_PROMPTS` | `false` |
| `pricing_file` | `QBS_AGENT_PRICING_FILE` | `""` |

Session metadata (`session_date`, `session_id`, `prompt_version`,
`authorization_timestamp`, `agent_status`) lives in the `sessions` table, not in config.

## Known limitations

- Yahoo intraday bars are delayed and sometimes late; a late bar is a skipped symbol,
  not a decision on old data. The newest bar's volume can be revised after publication.
- The holiday rules cover the standing NYSE calendar only; add one-off closures by hand.
- Constraint extraction from the prompt is regex-based. Anything it does not recognise
  stays prose for the model and is not enforced in code — state hard limits in the
  recognised phrasings above.
- Costs are estimates from a price table you maintain.
- The memory and decision history are per session date; nothing carries over to the
  next day except what the next day's prompt says.

## Pre-deployment notebooks

`notebooks/agent_tests/` holds six notebooks (MCP market context, intraday data, LLM
decisions, memory state, token benchmark, end-to-end dry run) that run on a versioned,
hash-checked fixture with a look-ahead-safe feed and a fake clock — MOCK by default,
REAL_LLM only with an explicit paid opt-in and a call cap. See
[notebooks/agent_tests/README.md](../notebooks/agent_tests/README.md).
`tests/test_agent_notebooks.py` carries their critical checks into CI.

## Tests

`tests/test_trading_agent.py` (mock LLM, synthetic bars, temp SQLite; no key, no
network, no IBKR): default-disabled, authorization valid / expired / revoked /
prompt-only, crash and restart, shutdown, disabled isolation, holidays and early closes,
DST, completed-candle enforcement, stale and missing data, duplicates, timeouts and
retries, invalid JSON, unauthorized symbols and constraint violations, token and cost
aggregation, no broker imports and no tools, and the memory layers.
