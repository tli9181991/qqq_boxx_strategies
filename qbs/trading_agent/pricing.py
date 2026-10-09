"""A configurable model price table, and cost estimates that admit ignorance.

Prices move faster than this repo, so the built-in table is a starting point
and every estimate is labelled an ESTIMATE. Override or extend it with a
JSON file (`pricing_file` / `QBS_AGENT_PRICING_FILE`):

    {"gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached_input": 0.075},
     "my-model":         {"input": 1.00, "output": 4.00}}

USD per million tokens. Input and output are priced separately, and cached
input at its own rate when one is given.

A model missing from the table gets NO cost -- `cost_status="unavailable"`
-- rather than $0, which would read as "free" on the dashboard. Likewise a
token category the provider did not report stays None, never 0.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

# USD per 1M tokens. VERIFY against the provider's pricing page before
# relying on a cost figure; these are list prices at the time of writing for
# prompts under any long-context tier. No cached-input rate is built in, so
# cached tokens are costed at the full input rate -- an overestimate, which is
# the safe direction -- until a pricing file supplies one.
DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
}


def load_prices(path: str = "") -> Dict[str, Dict[str, float]]:
    prices = {k: dict(v) for k, v in DEFAULT_PRICES.items()}
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            extra = json.load(fh)
        for model, p in (extra or {}).items():
            if isinstance(p, dict) and "input" in p and "output" in p:
                prices[model] = {k: float(v) for k, v in p.items()}
    return prices


@dataclass
class Cost:
    input_cost: Optional[float]
    output_cost: Optional[float]
    total_cost: Optional[float]
    status: str                 # "estimated" | "unavailable"


def estimate_cost(model: str, input_tokens: Optional[int], output_tokens: Optional[int],
                  cached_input_tokens: Optional[int] = None,
                  prices: Optional[Dict[str, Dict[str, float]]] = None) -> Cost:
    """Cost from token counts. Reasoning tokens are billed as output by every
    provider supported here and are already inside `output_tokens`."""
    table = prices if prices is not None else DEFAULT_PRICES
    p = table.get(model)
    if p is None or input_tokens is None or output_tokens is None:
        return Cost(None, None, None, "unavailable")
    cached = cached_input_tokens or 0
    rate_cached = p.get("cached_input", p["input"])
    inp = ((input_tokens - cached) * p["input"] + cached * rate_cached) / 1e6
    out = output_tokens * p["output"] / 1e6
    return Cost(round(inp, 8), round(out, 8), round(inp + out, 8), "estimated")
