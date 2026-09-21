"""CLI: ask the analyst a question, or print one report without an LLM.

    python -m qbs.agent "Why is MU in the momentum book but not the screen?"
    python -m qbs.agent --report picks
    python -m qbs.agent --report name --ticker MU
    python -m qbs.agent --check

`--report` is the escape hatch worth knowing about: it prints exactly what
the agent would read, with no API key, no model and no network. When an
answer looks wrong, diff it against the report rather than re-prompting.

The key comes from a `.env` at the repository root (see `.env.example`) or
from the environment, which wins. `--check` says which, without printing it.

`QBS_DISABLE_ANALYST=1` switches Gemini off without uninstalling anything.
`--report` and everything else in this package carry on working.

`--models` asks the API which model IDs your key can serve, which is the only
reliable way to settle whether a name you read somewhere actually exists.
"""

from __future__ import annotations

import argparse
import os
import sys

REPORTS = ("picks", "name", "breadth", "universe", "fundamentals", "news")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m qbs.agent",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="what to ask the analyst")
    ap.add_argument("--report", choices=REPORTS,
                    help="print one report directly, with no LLM involved")
    ap.add_argument("--ticker", default="", help="for --report name/fundamentals")
    ap.add_argument("--query", default="", help="for --report news")
    ap.add_argument("--model", default=None, help="override the Gemini model")
    ap.add_argument("--no-web", action="store_true",
                    help="build the agent without the search tools")
    ap.add_argument("--online", action="store_true",
                    help="allow the price cache to refresh from the network")
    ap.add_argument("--check", action="store_true",
                    help="report whether the analyst can run, then exit")
    ap.add_argument("--models", action="store_true",
                    help="list the model IDs your key can actually serve")
    ap.add_argument("--quiet", action="store_true",
                    help="print the answer only, without the tool trace")
    args = ap.parse_args(argv)

    if args.check:
        from .analyst import (DEFAULT_MODEL, DEFAULT_SUMMARY_MODEL,
                              _default_thinking_budget, check_requirements)
        from .env import (DISABLE_VAR, KNOWN_KEYS, analyst_disabled, load_env,
                          resolve_google_key)
        from .news import available_backends
        loaded = load_env()
        off = analyst_disabled()
        missing = check_requirements()
        print(f".env:            {loaded.summary()}")
        # Key names and set/unset only. Printing a secret to a terminal puts
        # it in the scrollback and the shell history of whoever ran --check.
        present = [k for k in KNOWN_KEYS if os.environ.get(k)]
        print(f"keys set:        {', '.join(present) or 'none'}")
        print(f"google key:      {'found' if resolve_google_key() else 'MISSING'}")
        budget = _default_thinking_budget()
        print(f"chat model:      {DEFAULT_MODEL}"
              + (f" · thinking budget {budget}"
                 + (" (dynamic)" if budget == -1 else
                    " (off)" if budget == 0 else "")
                 if budget is not None else " · no thinking budget sent"))
        print(f"summary model:   {DEFAULT_SUMMARY_MODEL} (news read, no thinking)")
        print(f"search backends: {', '.join(available_backends()) or 'none'}")
        # The switch gets its own line and its own word. "NOT ready" reads as
        # a misconfiguration and sends someone hunting for one; "disabled on
        # purpose" tells them they already know the cause.
        state = os.environ.get(DISABLE_VAR)
        print(f"{DISABLE_VAR}: {state!r}" if state else
              f"{DISABLE_VAR}: unset (the analyst may call Gemini)")
        if off:
            print("analyst:         DISABLED on purpose — "
                  f"unset {DISABLE_VAR} to turn it back on")
        else:
            print(f"analyst:         {'ready' if not missing else 'NOT ready — ' + missing}")
        return 0 if not missing else 1

    if args.models:
        return _models()

    if args.report:
        return _report(args)

    question = " ".join(args.question).strip()
    if not question:
        ap.print_help()
        return 2

    from .analyst import analyse
    answer = analyse(question, model=args.model, allow_web=not args.no_web)
    if not args.quiet and answer.tool_calls:
        print("Tools called: " + ", ".join(answer.tools_used), file=sys.stderr)
    print(answer.text)
    return 1 if answer.error else 0


def _models() -> int:
    """What this key can serve, straight from the API.

    Model names move faster than any file in this repo, and guessing one is
    how you get a 404 three layers down inside LangChain. This asks the
    source: no memory, no docs, no table that went stale.
    """
    from .env import load_env, resolve_google_key

    load_env()
    key = resolve_google_key()
    if not key:
        print("No API key. Put GOOGLE_API_KEY in .env or export it.",
              file=sys.stderr)
        return 1
    try:
        from google import genai
    except ImportError as exc:
        print(f"google-genai is not installed ({exc}); it ships with "
              f"langchain-google-genai — pip install -r requirements-agent.txt",
              file=sys.stderr)
        return 1

    try:
        models = list(genai.Client(api_key=key).models.list())
    except Exception as exc:                      # noqa: BLE001
        print(f"Could not list models: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    from .analyst import DEFAULT_MODEL, DEFAULT_SUMMARY_MODEL

    configured = {DEFAULT_MODEL, DEFAULT_SUMMARY_MODEL}
    names = []
    for m in models:
        name = str(getattr(m, "name", m)).replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue                              # embeddings and the like
        names.append(name)

    for name in sorted(names):
        mark = "  <- configured" if name in configured else ""
        print(f"  {name}{mark}")
    print(f"\n{len(names)} models generate content with this key.")
    for want in sorted(configured):
        if want not in names:
            print(f"WARNING: {want} is configured but NOT in that list — "
                  f"calls using it will fail.", file=sys.stderr)
    return 0


def _report(args) -> int:
    """The no-LLM path. Imports lazily so `--report` needs no LangChain."""
    from . import evidence as ev

    if args.report == "fundamentals":
        from .fundamentals import fetch_fundamentals, to_text
        if not args.ticker:
            print("--report fundamentals needs --ticker", file=sys.stderr)
            return 2
        snap, err = fetch_fundamentals(args.ticker)
        if snap is None:
            print(f"No fundamentals for {args.ticker.upper()}: {err}", file=sys.stderr)
            return 1
        print(to_text(snap, errors=err or ""))
        return 0

    if args.report == "news":
        from .news import search_web, to_text
        query = args.query or (f"{args.ticker} stock news" if args.ticker else "")
        if not query:
            print("--report news needs --query or --ticker", file=sys.stderr)
            return 2
        results, err = search_web(query, days=30)
        print(to_text(results, query=query, error=err or ""))
        return 1 if err else 0

    book = ev.load_book(offline=not args.online)
    if args.report == "picks":
        print(ev.picks_report(book))
    elif args.report == "breadth":
        print(ev.breadth_report(book))
    elif args.report == "universe":
        print(f"{book.universe.shape[1]} names through {book.asof:%Y-%m-%d}")
        print(", ".join(sorted(book.universe.columns)))
    elif args.report == "name":
        if not args.ticker:
            print("--report name needs --ticker", file=sys.stderr)
            return 2
        print(ev.name_report(book, args.ticker))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
