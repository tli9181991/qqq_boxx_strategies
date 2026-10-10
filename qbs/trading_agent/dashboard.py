"""The dashboard's Trading agent tab. Monitoring and controls only.

The tab never runs an analysis: it writes the daily prompt, the one-time
authorization and pause/resume/stop requests to the agent database, and
reads decisions and usage back. The schedule lives in
`python -m qbs.trading_agent run` (see runner.py for why).

Imported lazily by dashboard/app.py, so a broken or missing piece here can
cost this tab and nothing else.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from . import EXECUTION_BANNER, evaluation, store
from .config import load_config
from .llm import resolve_model
from .prompt import parse_prompt
from .session import (OPEN_STATES, PAUSED, RUNNING, STOPPED, authorize,
                      live_authorization, open_session, request_state, revoke,
                      save_daily_prompt)
from .session_calendar import session_bounds, session_date

PROMPT_TEMPLATE = """Date: {day}

Monitor only TEAM, MRVL, and TSM.

TEAM:
Watch for a breakout above $200.
Maximum suggested position: 4 shares.
Reference stop loss: $196.

MRVL:
Monitor my existing 9-share position.
Do not recommend adding new shares.

TSM:
Monitor for a potential recovery after pullback.

General:
Avoid unnecessary trading.
No short selling.
"""


def _fmt_ts(v: Any) -> str:
    try:
        return pd.Timestamp(v).tz_convert("America/New_York").strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(v)


def _money(v) -> str:
    return "n/a" if v is None else f"${v:,.4f}"


def _num(v, spec="{:,.0f}") -> str:
    return "n/a" if v is None else spec.format(v)


def render() -> None:
    import streamlit as st

    cfg = load_config()
    day = session_date()
    st.error(f"**{EXECUTION_BANNER}**", icon="🛑")
    st.caption("Phase 1 is a decision dry run: every BUY / SELL below is a logged "
               "recommendation, never an order. The analysis loop runs in its own "
               "process: `python -m qbs.trading_agent run`.")

    sess = open_session(cfg)
    prompt_row = store.latest_prompt(cfg.db_path, day.isoformat())
    auth = live_authorization(cfg, day)
    bounds = session_bounds(day, cfg.extra_holidays, cfg.extra_early_closes)

    # ---- control panel ------------------------------------------------
    st.subheader("Agent control panel")
    c = st.columns(4)
    c[0].metric("Feature switch", "ON" if cfg.agent_enabled else "OFF",
                help="QBS_AGENT_ENABLED. Necessary, never sufficient: each session "
                     "also needs today's one-time authorization.")
    c[1].metric("Agent state", sess["state"] if sess else "DISABLED")
    c[2].metric("Session date", day.isoformat(),
                help=("Market hours " + "–".join(b.strftime("%H:%M") for b in bounds) + " ET")
                if bounds else "US market closed today")
    c[3].metric("Agent enabled (this session)",
                "YES" if sess and sess["agent_enabled"] else "NO")
    c = st.columns(4)
    c[0].metric("Model", resolve_model(cfg.provider, cfg.model) or "(unset)")
    c[1].metric("Provider", cfg.provider)
    c[2].metric("Interval", f"{cfg.analysis_interval_minutes} min")
    c[3].metric("Execution mode", cfg.execution_mode)
    errs = cfg.validate()
    if errs:
        st.warning("Configuration invalid — the agent stays disabled:\n\n- "
                   + "\n- ".join(errs))
    if not cfg.agent_enabled:
        st.info("The agent feature is off. Set `QBS_AGENT_ENABLED=1` (environment, "
                "`.env` or var/agent/config.json) and restart to allow sessions.", icon="⏸️")

    # ---- the daily prompt ---------------------------------------------
    st.markdown("#### Daily user prompt")
    running = bool(sess and sess["state"] in OPEN_STATES)
    if running:
        st.warning(f"A session is open on prompt **v{sess['prompt_version']}**. Saving "
                   "creates a new version that the running session does NOT use until "
                   "you authorize it again; it is then adopted between cycles, never "
                   "mid-request.")
    text = st.text_area("Instructions for today", height=300, key="agent_prompt",
                        value=prompt_row["text"] if prompt_row
                        else PROMPT_TEMPLATE.format(day=day.isoformat()))
    parsed = parse_prompt(text, day)
    st.caption("Authorized symbols (enforced in code): **"
               + (", ".join(parsed.symbols) or "none") + "** · no short selling: "
               + ("yes" if parsed.no_short else "not stated"))
    for e in parsed.errors:
        st.error(e)
    for w in parsed.warnings:
        st.warning(w)
    if prompt_row:
        st.caption(f"Saved: v{prompt_row['version']} at {prompt_row['created_at']} UTC")
    b = st.columns(3)
    if b[0].button("Save prompt", disabled=not parsed.ok, key="agent_save"):
        version, _ = save_daily_prompt(cfg, text)
        st.success(f"Saved v{version}. Not authorized yet.")
        st.rerun()
    can_auth = cfg.agent_enabled and prompt_row is not None and not errs
    if b[1].button("Authorize today's session", type="primary", disabled=not can_auth,
                   key="agent_auth",
                   help="One-time token for today and the latest saved prompt version. "
                        "Consumed when the runner starts; a restart needs a new one."):
        if prompt_row and prompt_row["text"] != text:
            st.error("The editor has unsaved changes. Save first, so you authorize "
                     "exactly what you see.")
        else:
            ok, msg = authorize(cfg, created_by="dashboard")
            (st.success if ok else st.error)(msg)
    if b[2].button("Revoke / stop", key="agent_revoke"):
        revoke(cfg, "revoked from the dashboard")
        st.warning("Revoked. A running session stops before its next request.")
    if auth:
        st.caption(f"Unused authorization: prompt v{auth['prompt_version']}, issued "
                   f"{auth['created_at']}. The runner consumes it on start.")
    if sess and sess["state"] in OPEN_STATES:
        p = st.columns(3)
        if p[0].button("Pause", disabled=sess["state"] != RUNNING, key="agent_pause"):
            request_state(cfg, sess["session_id"], PAUSED)
        if p[1].button("Resume", disabled=sess["state"] != PAUSED, key="agent_resume"):
            request_state(cfg, sess["session_id"], RUNNING)
        if p[2].button("Stop session", key="agent_stop"):
            request_state(cfg, sess["session_id"], STOPPED)

    # ---- decisions ------------------------------------------------------
    st.subheader("Decisions")
    sessions = store.sessions(cfg.db_path)
    ids = [s["session_id"] for s in sessions]
    f = st.columns(2)
    pick = f[0].selectbox("Session", ["(all)"] + ids, key="agent_session_pick")
    sid = None if pick == "(all)" else pick
    rows = store.recent_decisions(cfg.db_path, limit=500, session_id=sid)
    symbols = sorted({r["symbol"] for r in rows})
    chosen = f[1].multiselect("Symbols", symbols, default=symbols, key="agent_syms")
    rows = [r for r in rows if r["symbol"] in chosen]
    good = [r for r in rows if r["status"] == "VALID"]
    bad = [r for r in rows if r["status"] != "VALID"]
    if good:
        st.dataframe(pd.DataFrame([{
            "Time (ET)": _fmt_ts(r["candle_ts"]), "Symbol": r["symbol"],
            "Action": r["action"], "Confidence": r["confidence"],
            "Entry": r["entry_price"], "SL": r["stop_loss"], "TP": r["take_profit"],
            "Qty": r["suggested_quantity"], "Reason": r["reason"],
        } for r in good]), hide_index=True, width="stretch")
        st.caption("Recommendations only — none of these was executed.")
    else:
        st.caption("No accepted decisions yet.")
    if bad:
        with st.expander(f"Errors, invalid and skipped ({len(bad)})", expanded=False):
            st.dataframe(pd.DataFrame([{
                "Time (ET)": _fmt_ts(r["candle_ts"]), "Symbol": r["symbol"],
                "Status": r["status"],
                "Errors": "; ".join(json.loads(r["errors_json"] or "[]")),
            } for r in bad]), hide_index=True, width="stretch")

    # ---- tokens and cost -------------------------------------------------
    st.subheader("Tokens, cost and latency")
    op = evaluation.operational(cfg.db_path, sid)
    m = st.columns(4)
    m[0].metric("Input tokens", _num(op["input_tokens"]))
    m[1].metric("Output tokens", _num(op["output_tokens"]))
    m[2].metric("Total tokens", _num(op["total_tokens"]))
    m[3].metric("Avg tokens / request", _num(op["avg_total_tokens"]))
    m = st.columns(4)
    m[0].metric("Est. API cost", _money(op["estimated_cost_usd"]),
                help="Estimated from the configurable price table (pricing.py). "
                     "Requests with no price or no reported usage are left out, "
                     "never counted as $0.")
    m[1].metric("Avg latency", _num(op["avg_latency_ms"], "{:,.0f} ms"))
    m[2].metric("Success rate", _num(op["success_rate"], "{:.0%}"))
    m[3].metric("Validation pass rate", _num(op["validation_pass_rate"], "{:.0%}"))
    if op["requests_without_cost"]:
        st.caption(f"{op['requests_without_cost']} request(s) have no cost estimate "
                   "(model not in the price table, or usage not reported).")
    if op["per_stock"]:
        st.dataframe(pd.DataFrame([{"Symbol": k, "Requests": v["requests"],
                                    "Tokens": v["total_tokens"],
                                    "Est. cost (USD)": v["cost_usd"]}
                                   for k, v in op["per_stock"].items()]),
                     hide_index=True)
    per_session: List[Dict[str, Any]] = []
    for s in sessions[:20]:
        o = evaluation.operational(cfg.db_path, s["session_id"])
        per_session.append({"Session": s["session_id"], "State": s["state"],
                            "Requests": o["llm_requests"], "Tokens": o["total_tokens"],
                            "Est. cost (USD)": o["estimated_cost_usd"]})
    if per_session:
        with st.expander("Cost per session"):
            st.dataframe(pd.DataFrame(per_session), hide_index=True)
    with st.expander("State transitions"):
        ev = store.state_events(cfg.db_path, sid, limit=100)
        if ev:
            st.dataframe(pd.DataFrame(ev)[["ts", "session_id", "from_state", "to_state",
                                           "reason"]], hide_index=True)
        else:
            st.caption("None recorded.")
