"""The LLM trading agent -- Phase 1: a decision DRY RUN.

What this is
------------
An optional module that, during an explicitly authorized US regular session,
asks an LLM for one structured decision per watched stock every completed
15-minute candle, validates it outside the model, and logs the decision, the
market snapshot it was made on, and the request's tokens, cost and latency.

What it is NOT
--------------
A trader. Phase 1 places, modifies and cancels nothing. Nothing in this
package imports `qbs.live`, `ib_async` or any broker client, the LLM is given
no tools at all, and `execution_mode` refuses any value but "DRY_RUN". The
decisions are recommendations written to a SQLite log; nothing reads that log
back into an order. See docs/TRADING_AGENT.md.

Layers, each usable on its own:

    config.py          AgentConfig: defaults <- JSON file <- environment
    session_calendar   NYSE sessions: holidays, early closes, DST, slots
    prompt.py          the Daily User Prompt: symbols, constraints, sections
    store.py           SQLite: prompts, authorizations, sessions, requests,
                       decisions, snapshots -- the audit trail
    session.py         one-time daily authorization and the state machine
    market_context.py  completed-candle snapshot + compact indicators
    llm.py             provider interface (Gemini, Anthropic, OpenAI, mock)
    pricing.py         configurable per-model price table
    decision.py        the output schema and its validation
    engine.py          one analysis cycle: data -> LLM -> validate -> log
    runner.py          the session loop: startup, schedule, shutdown
    evaluation.py      Phase 1 diagnostics over the log
    dashboard.py       the Streamlit tab (imported by dashboard/app.py)

Disabled by default. Nothing runs unless `QBS_AGENT_ENABLED` is on AND the
user has authorized today's session for a specific prompt version, and that
authorization is consumed the moment a runner picks it up -- so a crash or a
restart never resumes on its own.
"""

from .config import AgentConfig, load_config          # noqa: F401
from .decision import ACTIONS, validate_decision     # noqa: F401

EXECUTION_BANNER = "EXECUTION MODE: DRY RUN — NO ORDERS WILL BE EXECUTED"

__all__ = ["AgentConfig", "load_config", "ACTIONS", "validate_decision",
           "EXECUTION_BANNER"]
