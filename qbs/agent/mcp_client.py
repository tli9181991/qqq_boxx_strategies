"""A synchronous client for the qbs MCP server: notebooks, scripts, the analyst.

    from qbs.agent.mcp_client import connect

    qbs = connect("http://minipc:8765/mcp", token="...")   # a running server
    qbs = connect()          # QBS_MCP_URL / QBS_MCP_TOKEN, else a local server

    qbs.tool_names()                     # what the server offers
    print(qbs.current_picks())           # any tool, as a method
    ctx = qbs.call_json("stock_data", ticker="MU")   # the JSON tools, parsed
    tools = qbs.langchain_tools()        # hand them to a LangChain agent

With no URL, `connect()` starts `python -m qbs.agent.mcp_server` as a child
process over stdio -- the server Claude Desktop runs, with its own data
loaded once for as long as the client lives.

Why a thread
------------
The MCP SDK is async, and Jupyter already runs an event loop, so
`asyncio.run` inside a notebook cell raises. The client runs its own loop on
a daemon thread and blocks on it, which works the same in a notebook, a
script and a Streamlit rerun. The session's transport is opened and closed
inside ONE task on that loop: anyio's cancel scopes refuse to be exited from
a task other than the one that entered them.

Works with mcp 1.x and 2.x.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence

URL_VAR = "QBS_MCP_URL"
TOKEN_VAR = "QBS_MCP_TOKEN"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _field(obj: Any, *names: str, default=None):
    """The same field under its 1.x (camelCase) or 2.x (snake_case) name."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _root_cause(exc: BaseException) -> BaseException:
    """The first leaf of an ExceptionGroup: anyio wraps transport errors in
    one, and "unhandled errors in a TaskGroup" names nothing actionable."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return exc


class MCPError(RuntimeError):
    """The server could not be reached, or a call failed at the protocol
    level. A tool that ran and reported a failure returns text instead --
    that is an answer, and the tool contract says it explains itself."""


class MCPClient:
    """One open session to a qbs MCP server. Use `connect()` to make one."""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None,
                 command: Optional[Sequence[str]] = None,
                 env: Optional[Dict[str, str]] = None,
                 timeout: float = 300.0):
        self.url = url
        self.timeout = timeout
        self._token = token
        self._command = list(command) if command else [
            sys.executable, "-m", "qbs.agent.mcp_server"]
        self._env = env
        self._session = None
        self._tools: List[Any] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever,
                                        name="qbs-mcp-client", daemon=True)
        self._thread.start()
        self._closing: Optional[asyncio.Event] = None
        self._done: Optional[concurrent.futures.Future] = None
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._done = asyncio.run_coroutine_threadsafe(self._main(ready), self._loop)
        try:
            ready.result(timeout=timeout)
        except Exception as exc:
            self._stop_loop()
            where = url or " ".join(self._command)
            cause = _root_cause(exc)
            hint = (" -- check the server is running, the URL ends in /mcp, "
                    "and the token matches the server's QBS_MCP_TOKEN"
                    if url else "")
            raise MCPError(f"Could not open an MCP session to {where}: "
                           f"{type(cause).__name__}: {cause}{hint}") from exc

    # -- the session, held open by one task --------------------------------

    async def _main(self, ready: concurrent.futures.Future) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession

        self._closing = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                if self.url:
                    from mcp.client.streamable_http import (
                        create_mcp_http_client, streamable_http_client)
                    headers = ({"Authorization": f"Bearer {self._token}"}
                               if self._token else None)
                    http = await stack.enter_async_context(
                        create_mcp_http_client(headers=headers))
                    streams = await stack.enter_async_context(
                        streamable_http_client(self.url, http_client=http))
                else:
                    from mcp import StdioServerParameters
                    from mcp.client.stdio import stdio_client
                    env = dict(os.environ)
                    env["PYTHONPATH"] = os.pathsep.join(
                        p for p in (REPO_ROOT, env.get("PYTHONPATH")) if p)
                    env.setdefault("PYTHONUTF8", "1")
                    env.update(self._env or {})
                    params = StdioServerParameters(
                        command=self._command[0], args=self._command[1:],
                        env=env, cwd=REPO_ROOT)
                    streams = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(
                    ClientSession(streams[0], streams[1]))
                await session.initialize()
                self._tools = list((await session.list_tools()).tools)
                self._session = session
                ready.set_result(None)
                await self._closing.wait()
        except BaseException as exc:   # noqa: BLE001 -- handed to the caller
            if not ready.done():
                ready.set_exception(exc)
            else:
                raise
        finally:
            self._session = None

    def _run(self, coro, timeout: Optional[float] = None):
        if self._session is None:
            coro.close()
            raise MCPError("The MCP session is closed.")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout or self.timeout)
        except concurrent.futures.TimeoutError as exc:
            fut.cancel()
            raise MCPError(f"No answer within {timeout or self.timeout:.0f}s") from exc

    # -- the public API ------------------------------------------------------

    def tools(self) -> List[Dict[str, Any]]:
        """Every tool the server offers: name, description, input schema."""
        return [{"name": t.name, "description": t.description or "",
                 "input_schema": _field(t, "input_schema", "inputSchema",
                                        default={}) or {}}
                for t in self._tools]

    def tool_names(self) -> List[str]:
        return [t.name for t in self._tools]

    def call(self, name: str, **arguments) -> str:
        """Call one tool and return its text. Raises `MCPError` only when the
        call could not be made; a tool's own failure comes back as text."""
        if name not in self.tool_names():
            raise MCPError(f"The server has no tool {name!r}. It offers: "
                           + ", ".join(self.tool_names()))
        result = self._run(self._session.call_tool(name, arguments))
        text = "\n".join(getattr(c, "text", "") for c in result.content
                         if getattr(c, "type", "") == "text")
        if _field(result, "is_error", "isError", default=False):
            return f"{name} failed on the server: {text}"
        return text

    def call_json(self, name: str, **arguments):
        """`call`, with the JSON tools (`stock_data`, `market_snapshot`)
        parsed into a dict. A refusal or failure comes back as its text."""
        from .toolkit import split_json
        return split_json(self.call(name, **arguments))

    def __getattr__(self, name: str):
        # Only reached for attributes that do not exist, so the tools become
        # methods -- `qbs.current_picks()` -- without shadowing anything real.
        if name.startswith("_") or name not in self.tool_names():
            raise AttributeError(name)

        def method(**arguments) -> str:
            return self.call(name, **arguments)
        method.__name__ = name
        method.__doc__ = next((t["description"] for t in self.tools()
                               if t["name"] == name), "")
        return method

    def __dir__(self):
        return list(super().__dir__()) + self.tool_names()

    def langchain_tools(self, names: Optional[Sequence[str]] = None) -> List:
        """The server's tools as LangChain `StructuredTool`s, each calling
        back through this session. `names` keeps only those, in that order."""
        from langchain_core.tools import StructuredTool

        specs = {t["name"]: t for t in self.tools()}
        order = list(names) if names is not None else list(specs)
        out = []
        for n in order:
            if n not in specs:
                continue
            spec = specs[n]
            schema = dict(spec["input_schema"]) or {"type": "object",
                                                    "properties": {}}
            schema.setdefault("properties", {})
            out.append(StructuredTool(
                name=n, description=spec["description"], args_schema=schema,
                func=(lambda _n: lambda **kw: self.call(_n, **kw))(n)))
        return out

    # -- lifetime ------------------------------------------------------------

    def close(self) -> None:
        if self._closing is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._closing.set)
            try:
                self._done.result(timeout=10)
            except Exception:          # noqa: BLE001 -- closing anyway
                pass
        self._stop_loop()

    def _stop_loop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        where = self.url or "local stdio server"
        state = "open" if self._session is not None else "closed"
        return f"<MCPClient {where} ({state}, {len(self._tools)} tools)>"


_SHARED: Dict[tuple, MCPClient] = {}
_SHARED_LOCK = threading.Lock()


def shared_client(url: Optional[str] = None,
                  token: Optional[str] = None) -> MCPClient:
    """One open client per (url, token), reused across calls.

    The analyst builds its tools per question, and the dashboard reruns its
    script on every click; opening a session each time would add a
    handshake to every answer. A session that has died is replaced.
    """
    from .env import load_env
    load_env()
    url = url or os.environ.get(URL_VAR) or None
    token = token or os.environ.get(TOKEN_VAR) or None
    key = (url, token)
    with _SHARED_LOCK:
        client = _SHARED.get(key)
        if client is None or client._session is None:
            client = _SHARED[key] = MCPClient(url=url, token=token)
        return client


def connect(url: Optional[str] = None, token: Optional[str] = None,
            **kwargs) -> MCPClient:
    """An open client. `url` and `token` default to `QBS_MCP_URL` and
    `QBS_MCP_TOKEN` (the environment or `.env`); with no URL at all, a local
    server is started as a child process."""
    from .env import load_env
    load_env()
    url = url or os.environ.get(URL_VAR) or None
    token = token or os.environ.get(TOKEN_VAR) or None
    return MCPClient(url=url, token=token, **kwargs)
