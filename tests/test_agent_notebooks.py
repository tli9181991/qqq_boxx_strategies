"""CI checks behind the pre-deployment notebooks (notebooks/agent_tests/).

The critical validations the notebooks demonstrate -- fixture integrity, no
look-ahead, MOCK-by-default with a paid-call opt-in, isolation, reproducible
replays, the full-session lifecycle -- are asserted here directly, so CI does
not depend on a notebook runner. When nbclient and ipykernel are installed
the six notebooks are also executed end to end in MOCK mode.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qbs.trading_agent import session as ss, store, testkit as tk  # noqa: E402
from qbs.trading_agent.config import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH  # noqa: E402
from qbs.trading_agent.runner import run_session  # noqa: E402

NB_DIR = os.path.join(ROOT, "notebooks", "agent_tests")
NOTEBOOKS = sorted(f for f in os.listdir(NB_DIR) if f.endswith(".ipynb"))


def _replay(fx, start=(9, 31), **feed_kw):
    ws = tk.workspace(tk.MOCK)
    day = fx.replay_date
    clock = tk.Clock(tk.et(day, 9, 0))
    ss.save_daily_prompt(ws.cfg, fx.prompt, clock())
    assert ss.authorize(ws.cfg, clock())[0]
    clock.set(tk.et(day, *start))
    feed = tk.FixtureBars(fx, clock, **feed_kw)
    llm = tk.make_provider(tk.MOCK, ws.cfg)
    sess = run_session(ws.cfg, llm, feed, clock=clock, sleep=clock.sleep,
                       install_signals=False)
    return ws, sess, feed, llm


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def test_six_notebooks_exist():
    assert NOTEBOOKS == ["01_mcp_market_context.ipynb", "02_intraday_data.ipynb",
                         "03_llm_decision.ipynb", "04_memory_state.ipynb",
                         "05_token_benchmark.ipynb", "06_end_to_end_dry_run.ipynb"]


def test_fixture_matches_its_manifest():
    fx = tk.load_fixture()
    assert fx.symbols == ["TEAM", "MRVL", "TSM"]
    assert len(fx.bars) == 9 and fx.manifest["source"] == "synthetic"
    for (sym, iv), b in fx.bars.items():
        assert str(b.index.tz) == "America/New_York", (sym, iv)
        assert (b.High >= b[["Open", "Close"]].max(axis=1)).all(), (sym, iv)
        assert (b.Low <= b[["Open", "Close"]].min(axis=1)).all(), (sym, iv)


def test_a_tampered_fixture_is_refused(tmp_path):
    shutil.copytree(os.path.join(tk.FIXTURE_DIR, "v1"), tmp_path / "v1")
    path = tmp_path / "v1" / "TEAM_15m.csv"
    path.write_text(path.read_text().replace(",", ";", 1))
    with pytest.raises(ValueError, match="does not match"):
        tk.load_fixture("v1", root=str(tmp_path))


def test_fixture_builder_is_deterministic(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_agent_fixtures.py"),
                    "--root", str(tmp_path)], check=True, capture_output=True)
    built = tk.load_fixture("v1", root=str(tmp_path)).manifest["files"]
    assert built == tk.load_fixture().manifest["files"]


# --------------------------------------------------------------------------
# No look-ahead
# --------------------------------------------------------------------------

def test_fixture_feed_reveals_only_completed_bars():
    fx = tk.load_fixture()
    day = fx.replay_date
    clock = tk.Clock(tk.et(day, 10, 22))
    feed = tk.FixtureBars(fx, clock)
    b15 = feed.bars("TEAM", "15m")
    assert (b15.index + pd.Timedelta(minutes=15) <= clock()).all()
    assert b15.index[-1] == pd.Timestamp(tk.et(day, 10, 0))
    b30 = feed.bars("TEAM", "30m")
    assert b30.index[-1] == pd.Timestamp(tk.et(day, 9, 30))
    assert (feed.bars("TEAM", "1d").index.date < day).all()


def test_full_replay_never_sees_the_future():
    fx = tk.load_fixture()
    ws, sess, feed, llm = _replay(fx)
    snaps = store.rows(ws.cfg.db_path, "SELECT candle_ts, snapshot_json FROM snapshots")
    import json
    assert snaps and all(json.loads(s["snapshot_json"])["latest_completed_15m_end"]
                         == s["candle_ts"] for s in snaps)
    assert feed.revealed_max <= tk.et(fx.replay_date, 16, 0)


# --------------------------------------------------------------------------
# Modes, budget, isolation
# --------------------------------------------------------------------------

def test_mock_is_the_default_and_paid_needs_opt_in():
    assert tk.test_mode({}) == tk.MOCK
    assert tk.test_mode({tk.MODE_VAR: "something"}) == tk.MOCK
    with pytest.raises(RuntimeError, match="paid"):
        tk.test_mode({tk.MODE_VAR: "REAL_LLM"})
    assert tk.test_mode({tk.MODE_VAR: "REAL_LLM", tk.PAID_VAR: "1"}) == tk.REAL_LLM
    assert tk.max_calls({}) == 20 and tk.max_calls({tk.MAX_CALLS_VAR: "3"}) == 3


def test_call_budget_stops_before_sending():
    from qbs.trading_agent.llm import LLMSettings, MockProvider

    inner = MockProvider(["x"])
    budget = tk.CallBudget(inner, limit=2)
    budget.complete("s", "u", LLMSettings("m"))
    budget.complete("s", "u", LLMSettings("m"))
    with pytest.raises(tk.BudgetExceeded):
        budget.complete("s", "u", LLMSettings("m"))
    assert len(inner.calls) == 2


def test_cost_preview_is_shown_before_paid_runs():
    ws = tk.workspace(tk.MOCK, provider="gemini", model="gemini-2.5-flash")
    assert "estimated ~$" in tk.cost_preview(ws.cfg, 10)
    assert "unknown" in tk.cost_preview(tk.workspace(tk.MOCK).cfg, 10)


def test_workspace_is_isolated_from_production():
    ws = tk.workspace(tk.MOCK)
    assert ws.cfg.db_path != DEFAULT_DB_PATH
    assert not ws.cfg.db_path.startswith(os.path.join(ROOT, "var"))
    assert ws.cfg.provider == "mock"
    mtime = os.path.getmtime(DEFAULT_CONFIG_PATH) if os.path.exists(DEFAULT_CONFIG_PATH) else None
    _replay(tk.load_fixture())
    assert (os.path.getmtime(DEFAULT_CONFIG_PATH) if os.path.exists(DEFAULT_CONFIG_PATH)
            else None) == mtime


# --------------------------------------------------------------------------
# Reproducible replay and lifecycle
# --------------------------------------------------------------------------

def _decisions(ws):
    cols = ["symbol", "candle_ts", "status", "action", "entry_price", "stop_loss",
            "suggested_quantity"]
    return [tuple(r[c] for c in cols) for r in
            store.rows(ws.cfg.db_path, "SELECT * FROM decisions ORDER BY candle_ts, symbol")]


def test_mock_replay_is_reproducible_and_exercises_the_pipeline():
    fx = tk.load_fixture()
    a, sa, *_ = _replay(fx)
    b, sb, *_ = _replay(fx)
    assert _decisions(a) == _decisions(b)
    acts = {(d[0], d[3]) for d in _decisions(a)}
    assert ("TEAM", "BUY") in acts and ("MRVL", "SELL") in acts and ("TSM", "BUY") in acts
    assert sa.state == ss.STOPPED and sb.state == ss.STOPPED


def test_replay_with_a_late_candle_skips_it():
    fx = tk.load_fixture()
    late = tk.et(fx.replay_date, 10, 30)
    ws, sess, feed, llm = _replay(fx, withhold={("TEAM", late.isoformat())})
    row = store.one(ws.cfg.db_path, "SELECT status FROM cycles WHERE symbol='TEAM' AND "
                                    "candle_ts=?", (late.isoformat(),))
    assert row["status"] == "SKIPPED_STALE"
    assert not any(late.isoformat() in c["user"] and '"symbol": "TEAM"' in c["user"]
                   for c in llm.calls)


def test_no_broker_module_is_loaded_by_a_replay():
    before = set(sys.modules)
    _replay(tk.load_fixture())
    assert tk.broker_modules_loaded(before) == []


# --------------------------------------------------------------------------
# The notebooks themselves (MOCK mode)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs_in_mock_mode(name, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    monkeypatch.setenv(tk.MODE_VAR, tk.MOCK)
    monkeypatch.delenv(tk.PAID_VAR, raising=False)
    monkeypatch.setenv("QBS_AGENT_TEST_RESULTS", str(tmp_path))
    nb = nbformat.read(os.path.join(NB_DIR, name), as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": NB_DIR}}).execute()
    assert list(tmp_path.iterdir()), "the notebook saved no results"
