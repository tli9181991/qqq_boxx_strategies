"""Tests for the MCP server (`qbs.agent.mcp_server`).

The tools are plain functions over synthetic data -- no cache, no network,
no key. The registration test needs the `mcp` package and skips cleanly
without it, since it is an optional dependency.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qbs.agent import evidence as ev
from qbs.agent import mcp_server as ms
from qbs.agent.context import NOT_IN_FOCUS_MESSAGE
from qbs.config import Config
from qbs.data import synthetic_prices
from qbs.universe import synthetic_universe


def _data(**kw) -> ms.Data:
    px = synthetic_prices()
    uni = synthetic_universe(n=40).reindex(px.index).ffill()
    book = ev.Book(universe=uni, prices=px, cfg=Config(), note="synthetic")
    return ms.Data(book=book, watchlist=kw.pop("watchlist", []),
                   ohlc_loader=lambda t: None, **kw)


def _tools(data=None, allow_web=False):
    return {t.__name__: t for t in ms.build_tools(data or _data(),
                                                  allow_web=allow_web)}


def test_every_offline_tool_returns_text():
    data = _data()
    tools = _tools(data)
    ticker = data.book.universe.columns[0]
    calls = {"current_picks": (), "stock_data": (ticker,),
             "name_momentum": (ticker,), "price_action": (ticker, 3),
             "list_universe": (), "market_overview": (3,),
             "market_snapshot": (), "sector_leadership": (),
             "stock_vs_market": (ticker,)}
    for name, args in calls.items():
        out = tools[name](*args)
        assert isinstance(out, str) and out.strip(), name
        assert "failed:" not in out, f"{name}: {out[:200]}"


def test_web_tools_are_absent_not_blocked():
    """Offline means the model cannot even see a search tool to try."""
    assert not {"search_news", "ticker_headlines", "market_news"} & set(_tools())
    assert {"search_news", "ticker_headlines", "market_news"} <= set(
        _tools(allow_web=True))


def test_a_name_outside_the_focus_list_gets_the_dashboards_refusal():
    assert NOT_IN_FOCUS_MESSAGE in _tools()["stock_data"]("ZZZZ")


def test_nothing_reaches_stdout(capsys):
    """Over stdio, stdout IS the protocol: one print corrupts the stream."""
    def noisy() -> str:
        print("[universe] cache hit")
        return "ok"
    assert ms._quiet(noisy)() == "ok"
    out, err = capsys.readouterr()
    assert out == "" and "cache hit" in err


def test_a_failure_is_text_that_says_what_failed():
    def broken() -> str:
        raise ValueError("no cache")
    out = ms._quiet(broken)()
    assert "broken failed" in out and "no cache" in out


def test_the_backtest_runs_once_and_is_kept(monkeypatch):
    calls = []

    class Lab:
        results, rf = {}, None

    def fake_run(**kw):
        calls.append(kw)
        return Lab()

    import qbs.pipeline
    monkeypatch.setattr(qbs.pipeline, "run", fake_run)
    tools = _tools()
    tools["strategy_performance"]()
    tools["strategy_performance"]()
    assert len(calls) == 1 and calls[0]["offline"] is True


def test_the_web_switch_fails_safe(monkeypatch):
    monkeypatch.delenv(ms.NO_WEB_VAR, raising=False)
    assert ms.web_allowed()
    monkeypatch.setenv(ms.NO_WEB_VAR, "0")
    assert ms.web_allowed()
    for value in ("1", "yes", "disable"):
        monkeypatch.setenv(ms.NO_WEB_VAR, value)
        assert not ms.web_allowed(), value


def test_the_server_registers_every_tool():
    pytest.importorskip("mcp")
    server = ms.build_server(_data(), allow_web=False)
    listed = asyncio.run(server.list_tools())
    assert {t.name for t in listed} == set(_tools())
