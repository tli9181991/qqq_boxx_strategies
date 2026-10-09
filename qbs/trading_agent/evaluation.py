"""Phase 1 diagnostics, computed from the agent log.

Two groups, mirroring the spec:

* `operational` -- cycles, failures, latency, tokens, cost, validation pass
  rate, data availability;
* `decisions`  -- action distribution, how often consecutive decisions on
  one stock change (and how often they flip BUY<->SELL), compliance
  rejections, prices flagged as far from the market.

`forward_outcomes` is the optional post-decision look: the price change 1,
4 and 8 completed 15-minute candles after each decision, and the maximum
favourable / adverse move over that horizon. It reads LATER bars, so it is
evaluation only -- the engine never sees it -- and the numbers are
diagnostics of direction, not simulated returns: no fills, no sizing, no
costs.

Token sums skip categories a provider did not report and say how many
requests were missing them, rather than counting a gap as zero.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from . import store


def _sum(rows: List[Dict[str, Any]], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vals) if vals else None), len(rows) - len(vals)


def operational(db_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    reqs = store.requests_for(db_path, session_id)
    clause, args = ("WHERE session_id = ?", (session_id,)) if session_id else ("", ())
    cycles = store.rows(db_path, f"SELECT status, symbol, data_latency_ms FROM cycles {clause}",
                        args)
    n = len(reqs)
    ok = [r for r in reqs if r["success"]]
    valid = [r for r in reqs if r["validation_status"] == "VALID"]
    out: Dict[str, Any] = {
        "analysis_cycles": len(cycles),
        "cycle_status": dict(Counter(c["status"] for c in cycles)),
        "llm_requests": n,
        "llm_successful": len(ok),
        "llm_failed": n - len(ok),
        "success_rate": len(ok) / n if n else None,
        "validation_pass_rate": len(valid) / len(ok) if ok else None,
        "avg_latency_ms": (sum(r["latency_ms"] or 0 for r in reqs) / n) if n else None,
        "avg_data_latency_ms": (sum(c["data_latency_ms"] for c in cycles
                                    if c["data_latency_ms"] is not None)
                                / max(1, sum(c["data_latency_ms"] is not None for c in cycles))
                                if cycles else None),
        "retries": sum(r["retry_count"] or 0 for r in reqs),
        "timeouts": sum(r["timeout_events"] or 0 for r in reqs),
        "data_available_rate": (sum(c["status"] not in ("SKIPPED_STALE", "SKIPPED_MISSING")
                                    for c in cycles) / len(cycles)) if cycles else None,
    }
    for key in ("input_tokens", "output_tokens", "cached_input_tokens",
                "reasoning_tokens", "total_tokens"):
        total, missing = _sum(reqs, key)
        out[key] = total
        out[f"{key}_unreported"] = missing
    out["avg_total_tokens"] = (out["total_tokens"] / (n - out["total_tokens_unreported"])
                               if out["total_tokens"] is not None else None)
    cost, unpriced = _sum(reqs, "total_cost")
    out["estimated_cost_usd"] = cost
    out["requests_without_cost"] = unpriced
    out["input_cost_usd"], _ = _sum(reqs, "input_cost")
    out["output_cost_usd"], _ = _sum(reqs, "output_cost")
    per_stock: Dict[str, Dict[str, Any]] = {}
    for r in reqs:
        s = per_stock.setdefault(r["symbol"], {"requests": 0, "cost_usd": None,
                                               "total_tokens": 0})
        s["requests"] += 1
        if r["total_cost"] is not None:
            s["cost_usd"] = (s["cost_usd"] or 0) + r["total_cost"]
        s["total_tokens"] += r["total_tokens"] or 0
    out["per_stock"] = per_stock
    return out


def decisions(db_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    clause, args = ("WHERE session_id = ?", (session_id,)) if session_id else ("", ())
    rows = store.rows(db_path, f"SELECT * FROM decisions {clause} ORDER BY symbol, candle_ts",
                      args)
    valid = [r for r in rows if r["status"] == "VALID"]
    changes = flips = pairs = 0
    by_symbol: Dict[str, List[str]] = {}
    for r in valid:
        by_symbol.setdefault(r["symbol"], []).append(r["action"])
    for acts in by_symbol.values():
        for a, b in zip(acts, acts[1:]):
            pairs += 1
            changes += a != b
            flips += {a, b} == {"BUY", "SELL"}
    far = sum(any("from the last price" in w for w in json.loads(r["warnings_json"] or "[]"))
              for r in valid)
    return {
        "decisions": len(rows),
        "status": dict(Counter(r["status"] for r in rows)),
        "action_distribution": dict(Counter(r["action"] for r in valid)),
        "consecutive_pairs": pairs,
        "decision_change_rate": changes / pairs if pairs else None,
        "buy_sell_flip_rate": flips / pairs if pairs else None,
        "compliance_rejections": sum(r["status"] == "INVALID_COMPLIANCE" for r in rows),
        "schema_rejections": sum(r["status"] == "INVALID_SCHEMA" for r in rows),
        "price_far_from_market": far,
    }


def forward_outcomes(rows: Iterable[Dict[str, Any]], bars: Dict[str, pd.DataFrame],
                     horizons=(1, 4, 8)) -> pd.DataFrame:
    """Per decision: % change after each horizon of completed 15m candles,
    and the max favourable / adverse move (in % of the decision price) over
    the longest horizon. `bars[symbol]`: 15m OHLCV indexed by tz-aware bar
    START. Direction for MFE/MAE: SELL reads short-side, everything else
    long-side, since HOLD/NO_TRADE are measured as "what if long"."""
    out = []
    for r in rows:
        df = bars.get(r["symbol"])
        if df is None or r.get("last_price") is None:
            continue
        ts = pd.Timestamp(r["candle_ts"])
        after = df[df.index >= ts]          # bars starting at or after the candle end
        rec = {"symbol": r["symbol"], "candle_ts": r["candle_ts"], "action": r["action"]}
        p0 = float(r["last_price"])
        for h in horizons:
            rec[f"chg_{h}c_pct"] = ((float(after.Close.iloc[h - 1]) / p0 - 1) * 100
                                    if len(after) >= h else None)
        win = after.head(max(horizons))
        if len(win):
            up = (float(win.High.max()) / p0 - 1) * 100
            dn = (float(win.Low.min()) / p0 - 1) * 100
            short = r["action"] == "SELL"
            rec["mfe_pct"] = -dn if short else up
            rec["mae_pct"] = -up if short else dn
        out.append(rec)
    return pd.DataFrame(out)
