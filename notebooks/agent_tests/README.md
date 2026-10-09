# Pre-deployment test notebooks — LLM trading agent, Phase 1

Six notebooks that validate the dry-run agent before it is deployed. Each runs on its
own, in **MOCK mode by default with zero external LLM calls**, on a temporary database.

| Notebook | Validates |
|---|---|
| `01_mcp_market_context.ipynb` | the shared MCP/toolkit data layer: every tool returns text, snapshot consistency, freshness, missing-data refusal, read-only (no order verbs, no broker imports, data files unchanged), and that the agent's model gets no tools |
| `02_intraday_data.ipynb` | completed 15m/30m candles only, OHLC and 15m↔30m alignment, timezone and DST, indicators (null when history is short), late/missing candles → STALE/MISSING, session boundaries, holidays, early closes |
| `03_llm_decision.ipynb` | the decision engine for TEAM, MRVL, TSM on fixed candles: prompt parsing, strict JSON, validation, unauthorized ticker / short sale / oversized BUY rejected, tokens, latency and cost per request |
| `04_memory_state.ipynb` | consecutive-candle replay with persistent memory: legal transitions, bounded history, hypothetical vs real holdings, crash → no resume → re-authorize → memory restored, prompt versioning resets only changed stocks, no look-ahead |
| `05_token_benchmark.ipynb` | context size × memory length × provider: input/output/cached/reasoning tokens, memory tokens, latency, cost, validity — saved for comparison |
| `06_end_to_end_dry_run.ipynb` | a full session through the production runner on a fake clock, with a late candle, LLM timeouts, an invalid reply and a mid-session crash; scheduling, logging, shutdown, and no broker module ever loaded |

## Running

```bash
pip install -r requirements.txt            # includes jupyterlab
jupyter lab notebooks/agent_tests/         # or: jupyter nbconvert --execute ...
```

Notebooks import the production modules (`qbs.trading_agent.*`) and a small harness,
`qbs/trading_agent/testkit.py`, which holds the fixture loader, the look-ahead-safe
feed, the fake clock, the mode switch and result export. They hold no agent logic.

### MOCK and REAL_LLM

| | MOCK (default) | REAL_LLM |
|---|---|---|
| Enable | nothing | `QBS_AGENT_TEST_MODE=REAL_LLM` **and** `QBS_AGENT_TEST_ALLOW_PAID=1` |
| Model | `tk.rule_based_reply`, a deterministic harness | `QBS_AGENT_PROVIDER` / `QBS_AGENT_MODEL` (+ key, package) |
| Cost | $0, no network | estimate printed first; hard cap `QBS_AGENT_TEST_MAX_CALLS` (default 20) |
| Results | reproducible | exploratory observations — not evidence of prediction quality |

`05` also reads `QBS_AGENT_BENCH_PROVIDERS=gemini:gemini-2.5-flash,anthropic:claude-sonnet-5-5`.

## Fixtures and results

- `fixtures/v1/` — committed, versioned, hash-checked (`manifest.json`). **Synthetic**,
  deterministic prices (seed in the manifest) scripted so the replay day has a TEAM
  breakout above $200, a MRVL fade below VWAP and a TSM pullback/recovery. Not real
  prices. Rebuild or add a version with `scripts/build_agent_fixtures.py`
  (`--source yfinance --version v2` for real bars on a machine with Yahoo access).
  Fixtures are immutable: a changed file fails its hash check.
- `results/` — generated CSV/JSON, git-ignored, stamped per run. Override with
  `QBS_AGENT_TEST_RESULTS`.

The fixture feed (`tk.FixtureBars`) reveals only candles that have **ended** by the
simulated clock, and daily bars only for earlier sessions — the in-progress candle's
final values would be future data. It records the latest bar it served so notebooks
04 and 06 can assert no look-ahead.

## Safety

Each notebook builds its config from defaults (`tk.workspace()`), never from
`var/agent/config.json`, and writes only to a temp database. Nothing imports
`qbs.live` or a broker client; 01 and 06 assert it.

## CI

`tests/test_agent_notebooks.py` asserts the critical checks directly (fixture integrity
and determinism, no look-ahead, MOCK default and paid opt-in, call budget, isolation,
reproducible replay, late-candle skip, no broker modules) and, when `nbclient` and
`ipykernel` are installed, executes all six notebooks in MOCK mode.

**Do not deploy Phase 1 until** `pytest tests/test_trading_agent.py
tests/test_agent_notebooks.py` passes and notebooks 02, 04 and 06 pass in MOCK mode.
