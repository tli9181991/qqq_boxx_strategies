"""The analyst's tools as LangChain tools -- from `toolkit`, or from a server.

The tools themselves live in `qbs.agent.toolkit`, defined once for the MCP
server and for this module. Here they are only wrapped for LangChain, from
one of two sources:

* **in-process** (the default): the toolkit's functions, run in this
  process over frames the caller hands in or the cache.
* **an MCP server** (`mcp_url=`, or `QBS_MCP_URL` in the environment or
  `.env`): the same tools, called over MCP on a server that already has the
  data loaded -- `python -m qbs.agent.mcp_server --http`. The frames passed
  in here are then NOT used; the server reads its own cache, and its
  defaults, not a dashboard sidebar's.

Either way the model sees the same tool names and descriptions, because they
are the same functions. The contract every tool keeps is in `toolkit`.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import pandas as pd

from . import evidence as ev
from . import market as mk
from .toolkit import CONTEXT_TOOLS, RESEARCH_TOOLS, Data
from .toolkit import build_tools as build_plain_tools

# What the CLI analyst gets. `stock_data`, `market_snapshot` and
# `reload_data` are the MCP server's extras: the first two are what the
# dashboard's context mode hands over, the last is for a long-lived server.
ANALYST_TOOLS = ("current_picks", "name_momentum", "price_action",
                 "list_universe", "market_overview", "sector_leadership",
                 "stock_vs_market", "strategy_performance", "breakout_funnel",
                 "breakout_trades", "fundamentals", "search_news",
                 "ticker_headlines", "market_news")


def _require_langchain():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:       # pragma: no cover - import-guard path
        raise ImportError(
            "LangChain is not installed. `pip install -r requirements-agent.txt` "
            "(langchain, langchain-google-genai). The data layer underneath -- "
            "qbs.agent.toolkit, .evidence, .fundamentals, .news -- works "
            "without it."
        ) from exc
    return StructuredTool


def resolve_mcp_url(explicit: Optional[str] = None) -> Optional[str]:
    """The MCP server the analyst should use, or None for in-process."""
    return explicit or os.environ.get("QBS_MCP_URL") or None


def _names(toolset: str, has_stock: bool, allow_web: bool) -> List[str]:
    if toolset == "context":
        names = list(CONTEXT_TOOLS if has_stock else RESEARCH_TOOLS)
    else:
        names = list(ANALYST_TOOLS)
    if not allow_web:
        from .toolkit import WEB_TOOLS
        names = [n for n in names if n not in WEB_TOOLS]
    return names


def build_tools(
    book: Optional[ev.Book] = None,
    results: Optional[Dict] = None,
    rf: Optional[pd.Series] = None,
    trades: Optional[pd.DataFrame] = None,
    funnel: Optional[pd.DataFrame] = None,
    n_hold: int = 6,
    offline_fundamentals: bool = False,
    allow_web: bool = True,
    market: Optional[mk.Market] = None,
    spy: Optional[pd.Series] = None,
    ohlc_loader: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
    news_hours: int = 12,
    toolset: str = "all",
    stock_lookup: Optional[Callable[[str], Dict]] = None,
    mcp_url: Optional[str] = None,
    mcp_token: Optional[str] = None,
) -> List:
    """The tool set, as LangChain tools.

    `toolset="context"` is the dashboard's mode: the market arrives
    pre-computed in the prompt (`qbs.agent.context`), so the model gets only
    what the prompt does not hold -- `stock_data` when a `stock_lookup` is
    passed (or when a server supplies it), then `fundamentals`,
    `ticker_headlines` and `search_news`. No price data is loaded for it.

    `allow_web=False` makes the run provably offline: the search tools are
    absent, not blocked, so the model cannot report having tried.

    With `mcp_url` (or `QBS_MCP_URL`) the tools come from that server and
    every frame argument is ignored -- see the module docstring.
    """
    StructuredTool = _require_langchain()
    if toolset not in ("all", "context"):
        raise ValueError(f"toolset must be 'all' or 'context', not {toolset!r}")

    url = resolve_mcp_url(mcp_url)
    if url:
        from .mcp_client import shared_client
        client = shared_client(url, mcp_token)
        return client.langchain_tools(_names(toolset, True, allow_web))

    data = Data(book=book, market=market, spy=spy, results=results, rf=rf,
                trades=trades, funnel=funnel, ohlc_loader=ohlc_loader,
                stock_lookup=stock_lookup, n_hold=n_hold,
                offline_fundamentals=offline_fundamentals,
                news_hours=news_hours, run_backtest=False)
    names = _names(toolset, stock_lookup is not None, allow_web)
    return [StructuredTool.from_function(func=fn, name=fn.__name__,
                                         description=(fn.__doc__ or "").strip())
            for fn in build_plain_tools(data, allow_web=allow_web, names=names)]


def tool_names(tools: List) -> List[str]:
    """Names of a built tool list -- handy for logging what the agent had."""
    return [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
