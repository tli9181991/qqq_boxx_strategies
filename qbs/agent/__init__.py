"""An LLM analyst over the lab's results.

Layered so that only the top one needs LangChain:

    env.py          `.env` loading: the shell wins, no value is ever printed
    evidence.py     the lab's own numbers, rendered as text with their caveats
    fundamentals.py yfinance company data, cached  (no LangChain)
    news.py         web search and headlines       (no LangChain)
    tools.py        the three above, as LangChain tools
    analyst.py      a Gemini agent that may use them

The split is the point. `evidence`, `fundamentals` and `news` are ordinary
Python that a notebook, the dashboard or a test can call, and they behave the
same whether or not an LLM is involved. Nothing about the analysis depends on
the model being available -- the model only decides which report to read and
how to summarise it.

    from qbs.agent import analyse
    print(analyse("Why is MU in the momentum book but not the Finviz screen?"))

Needs `pip install -r requirements-agent.txt` and `GOOGLE_API_KEY`. Without
them `check_requirements()` says which is missing, and every layer below
`tools.py` still works.

The key can live in a `.env` at the repository root -- importing this package
reads it, filling in only what the shell has not already set. `ENV` holds
what that load did (key names and the file, never a value).
"""

# `.env` is read HERE, before `.analyst` is imported: that module reads
# QBS_GEMINI_MODEL at import time, so loading any later would pick up the
# default and ignore the file. A real environment variable still wins --
# `load_env` fills in what is missing and never overrides the shell.
from .env import load_env

ENV = load_env()

from .analyst import Answer, analyse, build_analyst, check_requirements  # noqa: E402
from .evidence import Book, load_book                                    # noqa: E402
from .fundamentals import Snapshot, fetch_fundamentals                   # noqa: E402
from .news import search_web, ticker_news                                # noqa: E402

__all__ = ["Answer", "analyse", "build_analyst", "check_requirements",
           "Book", "load_book", "Snapshot", "fetch_fundamentals",
           "search_web", "ticker_news", "ENV", "load_env"]
