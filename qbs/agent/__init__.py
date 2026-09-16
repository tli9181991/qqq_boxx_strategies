"""An LLM analyst over the lab's results.

Four layers, and only the top one needs LangChain:

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
"""

from .analyst import Answer, analyse, build_analyst, check_requirements
from .evidence import Book, load_book
from .fundamentals import Snapshot, fetch_fundamentals
from .news import search_web, ticker_news

__all__ = ["Answer", "analyse", "build_analyst", "check_requirements",
           "Book", "load_book", "Snapshot", "fetch_fundamentals",
           "search_web", "ticker_news"]
