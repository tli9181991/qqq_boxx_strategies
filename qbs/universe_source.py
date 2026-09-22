"""Which provider supplies the US universe: Finviz or TradingView.

One seam, two implementations of the same contract. `fetch_universe` returns
exactly what either provider's `fetch_us_universe` returns -- `(DataFrame
[Ticker, Sector, Industry, Country], error)` -- so a caller names a source
and nothing else in the app changes.

The switch is `QBS_UNIVERSE_SOURCE`, and the default is `finviz` because that
is what every number in this repo has been measured against so far. A
different provider is a different universe: the two apply the same rules to
different listings databases and will not agree on the last hundred names,
so breadth counts move when you switch. That is a legitimate thing to do and
not a thing to do by accident, which is why it is opt-in rather than
whichever-answers-first.

Failover is deliberately NOT automatic here. A dashboard that quietly served
Tuesday from one source and Wednesday from another would make a breadth
series that steps for a reason nobody can see in the data. The caller is told
which source produced the frame and can offer a manual switch; silently
changing the measurement is worse than showing a stale one.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .finviz import UniverseFilters

SOURCE_VAR = "QBS_UNIVERSE_SOURCE"
DEFAULT_SOURCE = "finviz"
SOURCES = ("finviz", "tradingview")


def resolve_source(source: Optional[str] = None,
                   environ: Optional[Dict[str, str]] = None) -> str:
    """The provider to use: the argument, else the environment, else Finviz.

    An unrecognised name falls back to the default rather than raising. This
    is read on a dashboard's start-up path, and a typo in a `.env` should
    cost a line on screen, not a page that will not load -- `source_note`
    is what says the typo out loud.
    """
    env = os.environ if environ is None else environ
    name = (source or env.get(SOURCE_VAR) or DEFAULT_SOURCE).strip().lower()
    return name if name in SOURCES else DEFAULT_SOURCE


def source_note(source: Optional[str] = None,
                environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """A complaint about `QBS_UNIVERSE_SOURCE`, or None if it is fine."""
    env = os.environ if environ is None else environ
    raw = (source or env.get(SOURCE_VAR) or "").strip()
    if raw and raw.lower() not in SOURCES:
        return (f"{SOURCE_VAR}={raw!r} is not a source this package knows "
                f"({', '.join(SOURCES)}) — falling back to {DEFAULT_SOURCE}")
    return None


def _provider(name: str) -> Callable:
    if name == "tradingview":
        from .tradingview import fetch_us_universe
        return fetch_us_universe
    from .finviz import fetch_us_universe
    return fetch_us_universe


def fetch_universe(
    filters: Optional[UniverseFilters] = None,
    source: Optional[str] = None,
    **kwargs,
) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
    """`(universe, error, source)` from whichever provider is selected.

    The third element is the name of the provider that actually ran, so a UI
    can say where the numbers came from. Two universes that disagree by a
    hundred names look like a market event when the screen does not name the
    source.
    """
    name = resolve_source(source)
    uni, err = _provider(name)(filters, **kwargs)
    note = source_note(source)
    if note:
        err = "; ".join(x for x in (err, note) if x)
    return uni, err, name


def available_sources() -> Dict[str, str]:
    """`{source: status}` -- which providers this interpreter can actually run.

    Import-only, so it costs nothing and touches no network. Says "installed"
    or why not, which is the question someone has when a source they set is
    not the source they got.
    """
    out: Dict[str, str] = {}
    try:
        import finvizfinance  # noqa: F401
        out["finviz"] = "installed"
    except ImportError as exc:
        out["finviz"] = f"not installed ({exc})"
    try:
        import tradingview_screener  # noqa: F401
        out["tradingview"] = "installed"
    except ImportError as exc:
        out["tradingview"] = f"not installed ({exc})"
    return out
