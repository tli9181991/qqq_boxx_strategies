"""Command line for the trading agent (Phase 1, dry run).

    python -m qbs.trading_agent status
    python -m qbs.trading_agent prompt set --file today.txt   # or --text "..." / --file -
    python -m qbs.trading_agent prompt show
    python -m qbs.trading_agent authorize                    # today's one-time token
    python -m qbs.trading_agent run                          # the session loop
    python -m qbs.trading_agent pause | resume | stop
    python -m qbs.trading_agent revoke
    python -m qbs.trading_agent report [--session ID]

Nothing here can place an order. `run` exits at once, with the reason, unless
QBS_AGENT_ENABLED is on and today's session has been authorized.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import EXECUTION_BANNER, evaluation, store
from .config import load_config
from .session import (PAUSED, RUNNING, STOPPED, authorize, live_authorization,
                      open_session, request_state, revoke, save_daily_prompt)
from .session_calendar import session_bounds, session_date


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qbs.trading_agent",
                                 description="LLM trading agent — Phase 1 dry run")
    ap.add_argument("--config", help="JSON config file (default var/agent/config.json)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("prompt")
    p.add_argument("action", choices=["set", "show"])
    p.add_argument("--file")
    p.add_argument("--text")
    sub.add_parser("authorize")
    sub.add_parser("revoke")
    for name in ("pause", "resume", "stop"):
        sub.add_parser(name)
    sub.add_parser("run")
    r = sub.add_parser("report")
    r.add_argument("--session")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    print(EXECUTION_BANNER, file=sys.stderr)
    day = session_date()

    if args.cmd == "status":
        bounds = session_bounds(day, cfg.extra_holidays, cfg.extra_early_closes)
        prompt = store.latest_prompt(cfg.db_path, day.isoformat())
        _print({
            "feature_enabled (QBS_AGENT_ENABLED)": cfg.agent_enabled,
            "config_errors": cfg.validate(),
            "config_notes": {k: v for k, v in cfg.sources.items() if k.startswith("error")},
            "session_date": day.isoformat(),
            "market_hours_et": ([b.strftime("%H:%M") for b in bounds] if bounds
                                else "closed today"),
            "provider": cfg.provider, "model": cfg.model or "(provider default)",
            "analysis_interval_minutes": cfg.analysis_interval_minutes,
            "execution_mode": cfg.execution_mode,
            "prompt_version": prompt["version"] if prompt else None,
            "authorized_symbols": json.loads(prompt["symbols_json"]) if prompt else [],
            "unused_authorization": live_authorization(cfg, day),
            "open_session": open_session(cfg),
            "db": cfg.db_path,
        })
        return 0

    if args.cmd == "prompt":
        if args.action == "show":
            row = store.latest_prompt(cfg.db_path, day.isoformat())
            print(row["text"] if row else f"(no prompt saved for {day})")
            return 0
        if args.text is not None:
            text = args.text
        elif args.file == "-":
            text = sys.stdin.read()
        elif args.file:
            with open(args.file, encoding="utf-8") as fh:
                text = fh.read()
        else:
            text = cfg.daily_user_prompt
        version, parsed = save_daily_prompt(cfg, text)
        if version is None:
            print("prompt NOT saved:\n  " + "\n  ".join(parsed.errors), file=sys.stderr)
            return 2
        print(f"saved prompt v{version} for {day}: symbols {', '.join(parsed.symbols)}")
        for w in parsed.warnings:
            print(f"  warning: {w}")
        print("Not authorized yet — run `authorize` to allow today's session.")
        return 0

    if args.cmd == "authorize":
        ok, msg = authorize(cfg, created_by="cli")
        print(msg, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 2

    if args.cmd == "revoke":
        n = revoke(cfg, "revoked from the command line")
        print(f"revoked ({n} token/session rows affected)")
        return 0

    if args.cmd in ("pause", "resume", "stop"):
        s = open_session(cfg)
        if s is None:
            print("no open session", file=sys.stderr)
            return 2
        state = {"pause": PAUSED, "resume": RUNNING, "stop": STOPPED}[args.cmd]
        request_state(cfg, s["session_id"], state)
        print(f"requested {state} for {s['session_id']}; the runner applies it "
              f"before its next request")
        return 0

    if args.cmd == "run":
        from .runner import run_session

        session = run_session(cfg)
        return 0 if session is not None else 1

    if args.cmd == "report":
        _print({"operational": evaluation.operational(cfg.db_path, args.session),
                "decisions": evaluation.decisions(cfg.db_path, args.session)})
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
