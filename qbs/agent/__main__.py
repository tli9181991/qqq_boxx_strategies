"""CLI: ask the analyst a question, or print one report without an LLM.

    python -m qbs.agent "Why is MU in the momentum book but not the screen?"
    python -m qbs.agent --report picks
    python -m qbs.agent --report name --ticker MU
    python -m qbs.agent --check

`--report` is the escape hatch worth knowing about: it prints exactly what
the agent would read, with no API key, no model and no network. When an
answer looks wrong, diff it against the report rather than re-prompting.
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--quiet", action="store_true",
                    help="print the answer only, without the tool trace")
    args = ap.parse_args(argv)

    if args.check:
        from .analyst import DEFAULT_MODEL, check_requirements
        from .news import available_backends
        missing = check_requirements()
        print(f"model:           {DEFAULT_MODEL}")
        print(f"search backends: {', '.join(available_backends()) or 'none'}")
        print(f"analyst:         {'ready' if not missing else 'NOT ready — ' + missing}")
        return 0 if not missing else 1

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
