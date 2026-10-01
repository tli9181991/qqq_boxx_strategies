"""Shared test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_finviz_cooldown(tmp_path, monkeypatch):
    """Point the Finviz cool-down stamp at a per-test temp file.

    A test that fakes a block ("rate limited", 403) correctly starts a
    cool-down. Written to the real data/universe/ stamp, that cool-down would
    outlive the test, block every later test that expects a pull, and stop the
    developer's own dashboard from asking Finviz for hours.
    """
    import qbs.finviz as finviz
    monkeypatch.setattr(finviz, "BLOCK_STAMP", str(tmp_path / "finviz_blocked_until.txt"))
