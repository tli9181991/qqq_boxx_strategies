"""Tests for the shared toolkit, the MCP server and the MCP client.

The toolkit tests run on synthetic data -- no cache, no network, no key.
The server and client tests need the `mcp` package and skip cleanly without
it, since it is an optional dependency. The round-trip tests start a real
server process and talk to it over stdio and over HTTP.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qbs.agent import evidence as ev
from qbs.agent import mcp_server as ms
from qbs.agent import toolkit as tk
from qbs.agent.context import NOT_IN_FOCUS_MESSAGE
from qbs.config import Config
from qbs.data import synthetic_prices
from qbs.universe import synthetic_universe


def _data(**kw) -> tk.Data:
    px = synthetic_prices()
    uni = synthetic_universe(n=40).reindex(px.index).ffill()
    book = ev.Book(universe=uni, prices=px, cfg=Config(), note="synthetic")
    return tk.Data(book=book, watchlist=kw.pop("watchlist", []),
                   ohlc_loader=lambda t: None, **kw)


def _tools(data=None, allow_web=False):
    return {t.__name__: t for t in tk.build_tools(data or _data(),
                                                  allow_web=allow_web)}


# --------------------------------------------------------------------------
# The toolkit
# --------------------------------------------------------------------------

def test_every_offline_tool_returns_text():
    data = _data()
    tools = _tools(data)
    ticker = data.book.universe.columns[0]
    calls = {"current_picks": (), "stock_data": (ticker,),
             "name_momentum": (ticker,), "price_action": (ticker, 3),
             "list_universe": (), "market_overview": (3,),
             "market_snapshot": (), "sector_leadership": (),
             "stock_vs_market": (ticker,), "breakout_funnel": (),
             "breakout_trades": ()}
    for name, args in calls.items():
        out = tools[name](*args)
        assert isinstance(out, str) and out.strip(), name
        assert "failed" not in out.split("\n")[0], f"{name}: {out[:200]}"


def test_web_tools_are_absent_not_blocked():
    """Offline means the model cannot even see a search tool to try."""
    assert not set(tk.WEB_TOOLS) & set(_tools())
    assert set(tk.WEB_TOOLS) <= set(_tools(allow_web=True))


def test_names_selects_and_orders():
    names = ["fundamentals", "stock_data"]
    got = [t.__name__ for t in tk.build_tools(_data(), names=names)]
    assert got == names


def test_a_name_outside_the_focus_list_gets_the_dashboards_refusal():
    out = _tools()["stock_data"]("ZZZZ")
    assert "NOT IN FOCUS LIST" in out and NOT_IN_FOCUS_MESSAGE in out


def test_json_tools_parse_and_refusals_do_not():
    tools = _tools()
    member = _data().book.universe.columns[0]
    assert isinstance(tk.split_json(tools["stock_data"](member)), dict)
    assert isinstance(tk.split_json(tools["market_snapshot"]()), dict)
    assert isinstance(tk.split_json(tools["stock_data"]("ZZZZ")), str)


def test_a_stock_lookup_replaces_how_stock_data_is_built():
    data = _data(stock_lookup=lambda t: {"ticker": t, "from": "dashboard"})
    assert '"from":"dashboard"' in _tools(data)["stock_data"]("MU")


def test_nothing_reaches_stdout(capsys):
    """Over stdio, stdout IS the protocol: one print corrupts the stream."""
    def noisy() -> str:
        print("[universe] cache hit")
        return "ok"
    assert tk._quiet(noisy)() == "ok"
    out, err = capsys.readouterr()
    assert out == "" and "cache hit" in err


def test_a_failure_is_text_that_says_what_failed():
    def broken(ticker) -> str:
        raise ValueError("no cache")
    out = tk._quiet(broken)("mu")
    assert "broken failed for MU" in out and "no cache" in out


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


def test_a_caller_that_owns_the_backtest_is_not_second_guessed(monkeypatch):
    import qbs.pipeline
    monkeypatch.setattr(qbs.pipeline, "run",
                        lambda **kw: pytest.fail("ran a backtest"))
    out = _tools(_data(run_backtest=False))["strategy_performance"]()
    assert "No backtest results were passed in" in out


def test_the_us_market_falls_back_and_says_why(monkeypatch):
    import qbs.universe_source as us
    monkeypatch.setattr(us, "fetch_universe",
                        lambda *a, **k: (None, "no cache", "finviz"))
    data = _data(us_market=True)
    text = _tools(data)["market_overview"]()
    assert data.market_note and "Nasdaq-100 fallback" in text


def test_the_web_switch_fails_safe(monkeypatch):
    monkeypatch.delenv(ms.NO_WEB_VAR, raising=False)
    assert ms.web_allowed()
    monkeypatch.setenv(ms.NO_WEB_VAR, "0")
    assert ms.web_allowed()
    for value in ("1", "yes", "disable"):
        monkeypatch.setenv(ms.NO_WEB_VAR, value)
        assert not ms.web_allowed(), value


def test_langchain_and_mcp_see_the_same_tools():
    """One definition: the analyst's tools are the toolkit's, by name and
    by description."""
    pytest.importorskip("langchain_core")
    from qbs.agent.tools import ANALYST_TOOLS, build_tools
    lc = {t.name: t.description for t in build_tools(book=_data().book)}
    plain = {t.__name__: (t.__doc__ or "").strip()
             for t in tk.build_tools(_data())}
    assert list(lc) == list(ANALYST_TOOLS)
    for name, desc in lc.items():
        assert desc == plain[name], name


# --------------------------------------------------------------------------
# The server
# --------------------------------------------------------------------------

def test_the_server_registers_every_tool():
    pytest.importorskip("mcp")
    server = ms.build_server(_data(), allow_web=False)
    listed = asyncio.run(server.list_tools())
    assert {t.name for t in listed} == set(_tools())


def test_http_off_localhost_needs_a_token():
    assert ms.serve_http(host="0.0.0.0", token=None) == 2


def test_the_token_gate():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    def call(headers, scope_type="http"):
        sent = []

        async def send(msg):
            sent.append(msg)

        async def receive():
            return {}
        asyncio.run(ms.BearerAuth(app, "s3cret")(
            {"type": scope_type, "headers": headers}, receive, send))
        return sent[0].get("status") if sent else None

    assert call([]) == 401
    assert call([(b"authorization", b"Bearer wrong")]) == 401
    assert call([(b"authorization", b"Bearer s3cret")]) == 200
    assert call([], scope_type="lifespan") == 200


# --------------------------------------------------------------------------
# Client <-> server, for real. These read the repository's price cache.
# --------------------------------------------------------------------------

def _has_cache() -> bool:
    return os.path.exists(os.path.join(ROOT, "data", "universe",
                                       "universe_prices.csv"))


needs_server = pytest.mark.skipif(
    not _has_cache(), reason="needs the cached universe in data/")


@needs_server
def test_stdio_round_trip():
    pytest.importorskip("mcp")
    from qbs.agent.mcp_client import connect
    with connect(url=None, env={ms.NO_WEB_VAR: "1"}) as client:
        assert "current_picks" in client.tool_names()
        assert "search_news" not in client.tool_names()
        assert "CURRENT PICKS" in client.current_picks()
        assert isinstance(client.call_json("market_snapshot"), dict)
        with pytest.raises(AttributeError):
            client.no_such_tool


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@needs_server
def test_http_round_trip_with_a_token():
    pytest.importorskip("mcp")
    from qbs.agent.mcp_client import MCPClient, MCPError
    port = _free_port()
    env = dict(os.environ, QBS_MCP_TOKEN="t0ken", QBS_MCP_NO_WEB="1",
               PYTHONPATH=ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "qbs.agent.mcp_server", "--http",
         "--port", str(port)], cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        with pytest.raises(MCPError):
            MCPClient(url=url, token="wrong", timeout=20)
        with MCPClient(url=url, token="t0ken", timeout=120) as client:
            assert "CURRENT PICKS" in client.call("current_picks")
            if pytest.importorskip("langchain_core"):
                tools = {t.name: t for t in client.langchain_tools(
                    ["name_momentum"])}
                assert "MOMENTUM PROFILE" in tools["name_momentum"].invoke(
                    {"ticker": "NVDA"})
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_a_client_hanging_up_is_not_reported_as_a_crash():
    """Windows logs an abrupt disconnect as a traceback; the server drops
    exactly that case and passes everything else through."""
    seen = []

    class Loop:
        def default_exception_handler(self, context):
            seen.append(context)

    ms._ignore_disconnects(Loop(), {"exception": ConnectionResetError(10054)})
    ms._ignore_disconnects(Loop(), {"exception": ConnectionAbortedError()})
    assert seen == []
    other = {"exception": ValueError("real"), "message": "boom"}
    ms._ignore_disconnects(Loop(), other)
    assert seen == [other]


# --------------------------------------------------------------------------
# The Market overview's sector leaders, in the focus list
# --------------------------------------------------------------------------

def _leader_market():
    """A synthetic US market: the constituents in two sectors plus LEADX,
    an outsider in its own sector that the leader rule has to pass."""
    import numpy as np
    import pandas as pd
    from qbs.agent.market import Market

    data = _data()
    uni = data.book.universe
    n = len(uni)
    g = np.r_[np.linspace(0, 0.5, n - 80), 0.5 + np.linspace(0, 0.6, 80)]
    closes = uni.copy()
    closes["LEADX"] = pd.Series(10 * np.exp(g), index=uni.index)
    sectors = {t: ("Tech" if i % 2 else "Health")
               for i, t in enumerate(uni.columns)}
    sectors["LEADX"] = "Energy"
    return Market(closes=closes, sectors=sectors,
                  qqq=data.book.prices["QQQ"], note="synthetic US")


def _leader_data(**kw):
    px = synthetic_prices()
    uni = synthetic_universe(n=40).reindex(px.index).ffill()
    book = ev.Book(universe=uni, prices=px, cfg=Config(), note="synthetic")
    return tk.Data(book=book, market=_leader_market(),
                   watchlist=kw.pop("watchlist", []),
                   ohlc_loader=lambda t: None, leaders_per_sector=3, **kw)


def test_leader_focus_keeps_the_overview_tables_order():
    from qbs.agent.context import leader_focus
    from qbs.breadth import sector_leaders
    m = _leader_market()
    rows = sector_leaders(m.closes, m.sectors, per_sector=3)
    got = leader_focus(rows)
    assert list(got) == list(rows["symbol"])
    assert got["LEADX"] == {"sector": "Energy", "rank_in_sector": 1,
                            "leaders_in_sector": 1,
                            "score": pytest.approx(float(
                                rows.set_index("symbol").loc["LEADX", "score"]))}
    assert leader_focus(None) == {} and leader_focus(rows.iloc[:0]) == {}


def test_a_leader_outside_the_index_is_in_focus_and_labelled_as_one():
    data = _leader_data()
    assert "LEADX" in data.leaders
    ctx = tk.split_json(_tools(data)["stock_data"]("leadx"))
    assert isinstance(ctx, dict), ctx
    assert ctx["membership"] == "sector_leader_outside_ndx"
    assert ctx["watchlist"] is False
    assert ctx["sector"] == "Energy"
    assert ctx["sector_leader"]["rank_in_sector"] == 1
    # An outsider: placed against the constituents, never held.
    assert "placement_rank_against_ndx" in ctx["normal_momentum"]
    assert ctx["normal_momentum"]["currently_held"] is False


def test_a_constituent_leader_keeps_its_membership():
    data = _leader_data()
    member = next(t for t in data.leaders if t in data.book.universe.columns)
    ctx = tk.split_json(_tools(data)["stock_data"](member))
    assert ctx["membership"] == "Nasdaq-100"
    assert ctx["sector_leader"]["sector"] in ("Tech", "Health")


def test_a_watched_leader_reads_as_the_watchlists():
    """You typed it in: it is your watchlist name first."""
    from qbs.agent.context import stock_context
    data = _leader_data()
    data.leaders                                   # joins LEADX's prices
    ctx = stock_context(data.book, "LEADX", watchlist=["LEADX"],
                        leaders=data.leaders, held={}, ranks={})
    assert ctx["membership"] == "watchlist_outside_ndx"
    assert ctx["watchlist"] is True and ctx["sector_leader"] is not None


def test_a_non_leader_outsider_is_still_refused():
    data = _leader_data()
    out = _tools(data)["stock_data"]("ZZZZ")
    assert "NOT IN FOCUS LIST" in out and NOT_IN_FOCUS_MESSAGE in out


def test_a_leader_with_no_prices_says_so_rather_than_refusing():
    from qbs.agent.context import focus_lookup
    out = focus_lookup(_data().book, "NOPX",
                       leaders={"NOPX": {"sector": "Energy"}})
    assert out["in_focus_list"] is True and "sector leader" in out["error"]


def test_the_leaders_are_listed_and_charted():
    data = _leader_data()
    tools = _tools(data)
    listing = tools["list_universe"]()
    assert "Sector leaders" in listing and "Energy: LEADX" in listing
    assert "PRICE ACTION — LEADX" in tools["price_action"]("LEADX", 3)


def test_no_leaders_on_the_index_fallback():
    data = _data()                                 # no US market
    assert data.leaders == {}
    assert "Sector leaders: none" in _tools(data)["list_universe"]()


def test_the_context_prompt_explains_the_leader_membership():
    from qbs.agent.analyst import system_prompt
    prompt = system_prompt(context="MARKET_CONTEXT\n{}")
    assert "sector_leader_outside_ndx" in prompt and '"sector_leader"' in prompt
