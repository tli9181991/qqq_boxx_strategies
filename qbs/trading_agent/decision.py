"""The decision schema, and validation that does not trust the model.

Every reply is parsed and checked here before it is called a decision. A
reply that fails ANY check is logged as a validation failure with its
reasons, and never becomes an accepted decision -- there is no "repair"
step that guesses what the model meant, because a guessed field in a
trading decision is exactly the thing an audit must not contain.

Two kinds of failure, reported separately because they mean different
things when evaluating a model:

* **schema** -- not JSON, a missing or extra field, a wrong type, an enum
  value outside the list, a number outside its range;
* **compliance** -- well-formed, but against the rules: a symbol that is
  not authorized today, a short sale under "no short selling", adding to a
  position the prompt said not to add to, more shares than the stated
  maximum, a BUY whose stop is above its entry.

Even an accepted decision is a RECOMMENDATION. Nothing reads it back into
an order in Phase 1.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

ACTIONS = ("BUY", "SELL", "HOLD", "ADJUST_STOP_LOSS", "NO_TRADE")
STRATEGIES = ("MOMENTUM", "BREAKOUT", "SWING", "POSITION_MANAGEMENT", "NONE")
MARKET_CONDITIONS = ("BULLISH", "BEARISH", "NEUTRAL", "VOLATILE", "UNCLEAR")

FIELDS = ("symbol", "timestamp", "action", "strategy", "confidence",
          "market_condition", "entry_price", "stop_loss", "take_profit",
          "suggested_quantity", "reason", "invalidation_condition",
          "risk_flags", "memory_update")

MAX_TEXT = 600
MAX_FLAGS = 10
# A price this far from the last trade is flagged (not rejected) as a
# possibly unsupported level -- one of the Phase 1 evaluation metrics.
FAR_FROM_MARKET = 0.15

_NUM_OR_NULL = {"type": ["number", "null"]}

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(FIELDS),
    "properties": {
        "symbol": {"type": "string"},
        "timestamp": {"type": "string",
                      "description": "ISO-8601 with UTC offset; echo the candle timestamp given"},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "strategy": {"type": "string", "enum": list(STRATEGIES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "market_condition": {"type": "string", "enum": list(MARKET_CONDITIONS)},
        "entry_price": _NUM_OR_NULL,
        "stop_loss": _NUM_OR_NULL,
        "take_profit": _NUM_OR_NULL,
        "suggested_quantity": {"type": ["integer", "null"], "minimum": 0},
        "reason": {"type": "string"},
        "invalidation_condition": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        # Validated separately by `memory.validate_memory_update`: a bad
        # proposal is rejected on its own and never invalidates the decision.
        "memory_update": {"type": ["object", "null"]},
    },
}


@dataclass
class DecisionContext:
    """What a decision is checked against. Built by the engine, never by
    the model."""
    symbol: str
    authorized: Sequence[str]
    candle_ts: datetime
    no_short: bool = False
    max_quantity: Optional[int] = None
    no_add: bool = False
    position_qty: Optional[float] = None     # None = unknown
    last_price: Optional[float] = None


@dataclass
class ValidationResult:
    decision: Optional[Dict[str, Any]]
    schema_errors: List[str] = field(default_factory=list)
    compliance_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.decision is not None and not self.schema_errors
                and not self.compliance_errors)

    @property
    def status(self) -> str:
        if self.ok:
            return "VALID"
        return "INVALID_SCHEMA" if self.schema_errors else "INVALID_COMPLIANCE"

    @property
    def errors(self) -> List[str]:
        return self.schema_errors + self.compliance_errors


def extract_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """`(payload, error)`. Tolerates a ```json fence, nothing else.

    Text around a bare object is NOT tolerated: a reply that wraps its JSON
    in prose did not follow the output contract, and that is worth counting.
    """
    body = (text or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.+?)\s*```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    if not body:
        return None, "empty response"
    try:
        return json.loads(body), None
    except ValueError as exc:
        return None, f"not valid JSON ({exc.msg} at char {exc.pos})"


def _is_number(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def _price(d: Dict[str, Any], key: str, errs: List[str]) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    if not _is_number(v) or v <= 0:
        errs.append(f"{key} must be a positive number or null, got {v!r}")
        return None
    return float(v)


def validate_decision(payload: Any, ctx: DecisionContext) -> ValidationResult:
    """Check one parsed reply against the schema and the day's rules."""
    schema: List[str] = []
    comp: List[str] = []
    warn: List[str] = []
    if not isinstance(payload, dict):
        return ValidationResult(None, ["the reply is not a JSON object"])

    missing = [f for f in FIELDS if f not in payload]
    extra = [k for k in payload if k not in FIELDS]
    if missing:
        schema.append(f"missing fields: {', '.join(missing)}")
    if extra:
        schema.append(f"unexpected fields: {', '.join(map(str, extra))}")

    d = dict(payload)
    symbol = str(d.get("symbol") or "").strip().upper()
    if symbol != ctx.symbol:
        comp.append(f"symbol {d.get('symbol')!r} is not the symbol requested ({ctx.symbol})")
    if symbol not in ctx.authorized:
        comp.append(f"symbol {d.get('symbol')!r} is not authorized for this session")

    ts = d.get("timestamp")
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            schema.append("timestamp has no UTC offset")
        elif parsed != ctx.candle_ts:
            schema.append(f"timestamp {ts} is not the analysed candle "
                          f"({ctx.candle_ts.isoformat()})")
    except (TypeError, ValueError):
        schema.append(f"timestamp {ts!r} is not ISO-8601")

    action = d.get("action")
    if action not in ACTIONS:
        schema.append(f"action {action!r} is not one of {', '.join(ACTIONS)}")
    if d.get("strategy") not in STRATEGIES:
        schema.append(f"strategy {d.get('strategy')!r} is not one of {', '.join(STRATEGIES)}")
    if d.get("market_condition") not in MARKET_CONDITIONS:
        schema.append(f"market_condition {d.get('market_condition')!r} is not one of "
                      f"{', '.join(MARKET_CONDITIONS)}")
    conf = d.get("confidence")
    if not _is_number(conf) or not 0 <= conf <= 1:
        schema.append(f"confidence must be a number in [0, 1], got {conf!r}")

    entry = _price(d, "entry_price", schema)
    stop = _price(d, "stop_loss", schema)
    target = _price(d, "take_profit", schema)

    qty = d.get("suggested_quantity")
    if qty is not None:
        if (isinstance(qty, bool) or not _is_number(qty) or qty < 0
                or float(qty) != int(qty)):
            schema.append(f"suggested_quantity must be a nonnegative integer or null, got {qty!r}")
            qty = None
        else:
            qty = int(qty)

    for key in ("reason", "invalidation_condition"):
        v = d.get(key)
        if not isinstance(v, str) or not v.strip():
            schema.append(f"{key} must be a non-empty string")
        elif len(v) > MAX_TEXT:
            schema.append(f"{key} is longer than {MAX_TEXT} characters")
    flags = d.get("risk_flags")
    if not isinstance(flags, list) or not all(isinstance(x, str) for x in flags):
        schema.append("risk_flags must be a list of strings")
    elif len(flags) > MAX_FLAGS:
        schema.append(f"risk_flags has more than {MAX_FLAGS} entries")
    mu = d.get("memory_update")
    if mu is not None and not isinstance(mu, dict):
        schema.append("memory_update must be an object or null")

    # ---- semantics, only meaningful once the shape is right --------------
    pos = ctx.position_qty
    if action == "BUY":
        if entry is None or stop is None:
            comp.append("BUY needs entry_price and stop_loss")
        elif stop >= entry:
            comp.append(f"BUY stop_loss {stop} is not below entry_price {entry}")
        if entry is not None and target is not None and target <= entry:
            comp.append(f"BUY take_profit {target} is not above entry_price {entry}")
        if not qty:
            comp.append("BUY needs a suggested_quantity of at least 1")
        if ctx.no_add and pos:
            comp.append("BUY adds to an existing position the prompt said not to add to")
        if ctx.max_quantity is not None and qty:
            total = qty + (int(pos) if pos else 0)
            if total > ctx.max_quantity:
                comp.append(f"BUY would take the position to {total} shares, over "
                            f"the prompt's maximum of {ctx.max_quantity}")
    elif action == "SELL":
        if not qty:
            comp.append("SELL needs a suggested_quantity of at least 1")
        if ctx.no_short and not pos:
            comp.append("SELL without a known long position is a short sale, "
                        "and the prompt says no short selling")
        if pos is not None and qty and pos > 0 and qty > pos:
            comp.append(f"SELL of {qty} shares exceeds the {int(pos)}-share position")
    elif action == "ADJUST_STOP_LOSS":
        if stop is None:
            comp.append("ADJUST_STOP_LOSS needs a stop_loss")
        if pos is not None and pos <= 0:
            comp.append("ADJUST_STOP_LOSS with no position to protect")
        if qty:
            comp.append("ADJUST_STOP_LOSS must not carry a suggested_quantity")
    elif action in ("HOLD", "NO_TRADE"):
        if qty:
            comp.append(f"{action} must not carry a suggested_quantity")

    if ctx.last_price and ctx.last_price > 0:
        for key, v in (("entry_price", entry), ("stop_loss", stop), ("take_profit", target)):
            if v is not None and abs(v / ctx.last_price - 1) > FAR_FROM_MARKET:
                warn.append(f"{key} {v} is more than {FAR_FROM_MARKET:.0%} from the "
                            f"last price {ctx.last_price}")

    if schema:
        return ValidationResult(None, schema, comp, warn)
    clean = {k: d[k] for k in FIELDS}
    clean.update(symbol=symbol, suggested_quantity=qty, entry_price=entry,
                 stop_loss=stop, take_profit=target, confidence=float(conf))
    return ValidationResult(clean, schema, comp, warn)


def parse_and_validate(text: str, ctx: DecisionContext) -> ValidationResult:
    payload, err = extract_json(text)
    if err:
        return ValidationResult(None, [err])
    return validate_decision(payload, ctx)
