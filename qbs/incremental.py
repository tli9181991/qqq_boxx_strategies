"""Download only what the cache is missing, without breaking the adjustment basis.

Why this is not simply "append the new rows"
--------------------------------------------
yfinance serves closes adjusted for splits and dividends, and adjustments are
applied BACKWARDS: the day a stock goes ex-dividend or splits, every close
before that day changes. A cache that only appends new rows keeps the old
basis for its history and the new basis for the rows it appended, and the seam
between them reads as a return that never happened -- a 10:1 split becomes a
-90% day. That is why every refresh used to re-download the whole history.

So a refresh here downloads a short RECENT window that overlaps the cache by a
few confirmed sessions, and checks the overlap name by name:

* the overlap agrees -> nothing was re-based; the new rows are appended.
* the overlap disagrees -> the provider re-based that name's history (a split
  or a dividend); that name alone is re-downloaded in full.
* a name the cache does not hold, or holds with no overlap to check -> full
  history for that name.

Provisional cells
-----------------
The newest close can also come from the screener (`qbs.quotes`), before
yfinance publishes it. Those cells are written to the cache so the next load
does not need the network at all, but they are RAW prices, not adjusted ones,
and they are not yfinance's. Each one is recorded in a sidecar file
(`<cache>.provisional.csv`) and:

* is excluded from the overlap check -- a raw print disagreeing with an
  adjusted one is not evidence of a re-basing;
* anchors the next download window, so the first fetch after it always covers
  it again;
* is overwritten by yfinance's value as soon as yfinance has one, and its mark
  is dropped then.

A full download rewrites the cache and clears every mark.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

# How many confirmed sessions of overlap a recent download carries. One would
# do to detect a re-basing; a few more cover a missed day or a holiday.
OVERLAP_SESSIONS = 5
# Relative disagreement above which a name counts as re-based. The smallest
# real adjustment -- a tiny dividend -- is several times this; float noise from
# a CSV round trip is many orders of magnitude below it.
REBASE_TOL = 1e-5

Mark = Tuple[pd.Timestamp, str]
# (tickers, start) -> (closes, volumes or None, failed tickers)
Downloader = Callable[[List[str], str],
                      Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]]


def marks_path(cache_path: str) -> str:
    root, _ = os.path.splitext(cache_path)
    return root + ".provisional.csv"


def read_marks(cache_path: str) -> Set[Mark]:
    """The (date, ticker) cells of `cache_path` that came from the screener."""
    path = marks_path(cache_path)
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
    except Exception:  # noqa: BLE001 -- a broken sidecar costs a re-check only
        return set()
    return {(pd.Timestamp(d).normalize(), str(t))
            for d, t in zip(df["Date"], df["Ticker"])}


def write_marks(cache_path: str, marks: Iterable[Mark]) -> None:
    path = marks_path(cache_path)
    marks = sorted(set(marks))
    if not marks:
        if os.path.exists(path):
            os.remove(path)
        return
    pd.DataFrame(marks, columns=["Date", "Ticker"]).to_csv(path, index=False)


def clear_marks(cache_path: str) -> None:
    write_marks(cache_path, [])


# How far back a thin session (most names missing) is still re-fetched. Older
# than this it is history the provider evidently does not have.
GAP_REPAIR_SESSIONS = 60


def window_start(cached: pd.DataFrame, marks: Iterable[Mark],
                 overlap: int = OVERLAP_SESSIONS) -> pd.Timestamp:
    """Where a recent download must start to re-check the cache's front edge.

    Anchored on the OLDEST of: a provisional cell, so every screener print is
    re-fetched until yfinance has replaced it; a thin session in the last
    `GAP_REPAIR_SESSIONS` rows, so a session a download returned for only a
    few hundred names is fetched again rather than kept for good; and
    otherwise the newest row. `overlap` business days before that anchor are
    the confirmed sessions the re-basing check compares.
    """
    from .data import thin_rows

    dates = [d for d, _ in marks]
    dates += list(thin_rows(cached.tail(GAP_REPAIR_SESSIONS + 20)))
    dates = [d for d in dates
             if d >= cached.index[-min(len(cached), GAP_REPAIR_SESSIONS)]]
    anchor = min(dates) if dates else cached.index.max()
    return (pd.Timestamp(anchor) - pd.offsets.BDay(overlap)).normalize()


def _norm(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "Date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.loc[:, ~frame.columns.duplicated()]


def merge_recent(cached: pd.DataFrame, recent: pd.DataFrame,
                 marks: Iterable[Mark] = (), tol: float = REBASE_TOL
                 ) -> Tuple[pd.DataFrame, List[str], Set[Mark]]:
    """Fold a recent download into the cache.

    Returns `(merged, rebased, marks_left)`:

    * `merged` -- the cache with every name that passed the overlap check
      updated from `recent`. A name that did not pass is left as cached.
    * `rebased` -- names that need their full history: the overlap disagrees,
      or there is no confirmed overlap to check.
    * `marks_left` -- the provisional cells yfinance has not replaced yet.

    Names only in the cache are kept untouched (a quiet day, or a name the
    download skipped); names only in `recent` are the caller's business, since
    a window is not a history.
    """
    marks = set(marks)
    recent = _norm(recent)
    merged = cached.reindex(cached.index.union(recent.index))
    rebased: List[str] = []
    replaced: Set[Mark] = set()
    for t in recent.columns:
        if t not in cached.columns:
            continue
        new = recent[t].dropna()
        if new.empty:
            continue
        old = cached[t].reindex(new.index)
        confirmed = [d for d in new.index
                     if pd.notna(old.get(d)) and (d, t) not in marks]
        if not confirmed:
            rebased.append(t)
            continue
        drift = (new.loc[confirmed] / old.loc[confirmed] - 1.0).abs().max()
        if not drift <= tol:
            rebased.append(t)
            continue
        merged.loc[new.index, t] = new.values
        replaced.update((d, t) for d in new.index)
    return merged, rebased, marks - replaced


def merge_volumes(cached: Optional[pd.DataFrame], recent: Optional[pd.DataFrame],
                  skip: Iterable[str] = ()) -> Optional[pd.DataFrame]:
    """Recent volumes over the cached ones, for every name not in `skip`.

    No re-basing check of its own: a split re-bases volumes too, but it
    re-bases the closes in the same response, so `merge_recent` has already
    sent that name to a full download.
    """
    if recent is None or recent.empty:
        return cached
    recent = _norm(recent)
    if cached is None or cached.empty:
        return recent.drop(columns=[c for c in skip if c in recent.columns])
    merged = cached.reindex(cached.index.union(recent.index))
    skip = set(skip)
    for t in recent.columns:
        if t in skip:
            continue
        new = recent[t].dropna()
        if t not in merged.columns:
            merged[t] = float("nan")
        merged.loc[new.index, t] = new.values
    return merged


def replace_columns(base: Optional[pd.DataFrame],
                    full: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """`base` with every column of `full` swapped in whole."""
    if full is None or full.empty:
        return base
    full = _norm(full)
    if base is None or base.empty:
        return full
    out = base.reindex(base.index.union(full.index))
    out = out.drop(columns=[c for c in full.columns if c in out.columns])
    out = out.join(full.reindex(out.index), how="left")
    return out.sort_index()


def refresh_incremental(
    cached_c: pd.DataFrame,
    cached_v: Optional[pd.DataFrame],
    marks: Set[Mark],
    tickers: List[str],
    start: str,
    download: Downloader,
    overlap: int = OVERLAP_SESSIONS,
    tol: float = REBASE_TOL,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Set[Mark], dict]:
    """Bring a cache up to date with the smallest download that is still safe.

    `download(tickers, start)` fetches `(closes, volumes, failed)`; it is
    called once for the recent window over the names the cache holds, and at
    most once more for the names that need full history. Raises RuntimeError
    when the recent download returns nothing at all, so the caller can fall
    back to the cache and say why -- the same contract as a full download.

    Returns `(closes, volumes, marks_left, info)`, where `info` has `recent`
    (names updated from the window), `full` (names re-downloaded in full) and
    `failed` (names a FULL download could not get -- the ones worth
    quarantining; a name missing from the window keeps its cached history).
    """
    held = [t for t in tickers if t in cached_c.columns]
    new_names = [t for t in tickers if t not in cached_c.columns]
    since = f"{window_start(cached_c, marks, overlap):%Y-%m-%d}"

    rc, rv, _ = download(held, since) if held else (None, None, [])
    if held and (rc is None or rc.empty):
        raise RuntimeError(f"no recent prices returned for any of {len(held)} "
                           f"cached tickers since {since}")
    closes, rebased, marks_left = (
        merge_recent(cached_c, rc, marks, tol) if rc is not None
        else (cached_c, [], set(marks)))
    volumes = merge_volumes(cached_v, rv, skip=rebased)

    full = rebased + new_names
    failed: List[str] = []
    if full:
        fc, fv, failed = download(full, start)
        if fc is not None and not fc.empty:
            fc = fc.drop(columns=[c for c in fc.columns
                                  if fc[c].notna().sum() == 0])
            closes = replace_columns(closes, fc)
            volumes = replace_columns(volumes, fv)
            got = set(fc.columns)
            marks_left = {(d, t) for d, t in marks_left if t not in got}
        else:
            failed = list(full)
    from .data import thin_rows

    # A thin session still thin after the update is one the provider does not
    # have either. Reported so the caller can say which day is missing.
    holes = {d: v for d, v in thin_rows(closes).items()
             if d >= pd.Timestamp(since)}
    info = dict(recent=len(held) - len(rebased), full=len(full) - len(failed),
                rebased=rebased, failed=failed, since=since, holes=holes)
    return closes, volumes, marks_left, info


def persist_fill(cache_path: str, session: pd.Timestamp, closes: pd.Series,
                 vol_path: Optional[str] = None,
                 volumes: Optional[pd.Series] = None) -> int:
    """Write screener closes for `session` into the cache, marked provisional.

    Only cells the cache does NOT already hold are written: a yfinance value
    on disk is the authority and is never overwritten by a screener print.
    Returns how many cells were written. Never raises -- failing to save a
    top-up costs one extra download, not the page.
    """
    try:
        if not os.path.exists(cache_path) or closes is None or closes.empty:
            return 0
        session = pd.Timestamp(session).normalize()
        disk = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        if session not in disk.index:
            disk.loc[session] = float("nan")
            disk = disk.sort_index()
        written = []
        for t, v in closes.dropna().items():
            if t in disk.columns and pd.notna(disk.at[session, t]):
                continue
            disk.loc[session, t] = float(v)
            written.append(t)
        if not written:
            return 0
        disk.to_csv(cache_path)
        write_marks(cache_path, read_marks(cache_path)
                    | {(session, t) for t in written})

        if vol_path and volumes is not None and os.path.exists(vol_path):
            vd = pd.read_csv(vol_path, parse_dates=["Date"], index_col="Date")
            if session not in vd.index:
                vd.loc[session] = float("nan")
                vd = vd.sort_index()
            for t in written:
                v = volumes.get(t)
                if v is not None and pd.notna(v):
                    vd.loc[session, t] = float(v)
            vd.to_csv(vol_path)
        return len(written)
    except Exception:  # noqa: BLE001
        return 0


def cache_report(cache_path: str, rows: int = 15) -> str:
    """How many names each recent session has, the thin ones flagged.

    `python -m qbs.incremental data/universe/us_closes.csv` -- the quickest
    way to see whether a breadth reading is built on a whole session or on the
    few hundred names a download happened to return for it.
    """
    from .data import thin_rows

    frame = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
    thin = thin_rows(frame)
    marks = read_marks(cache_path)
    per_day: Dict[pd.Timestamp, int] = {}
    for d, _ in marks:
        per_day[d] = per_day.get(d, 0) + 1
    lines = [f"{cache_path}: {frame.shape[1]} tickers, "
             f"{frame.index.min():%Y-%m-%d} .. {frame.index.max():%Y-%m-%d}"]
    for d, n in frame.notna().sum(axis=1).tail(rows).items():
        flag = ""
        if d in thin:
            flag = f"  <-- THIN: {thin[d][0]} of ~{thin[d][1]} names"
        if per_day.get(d):
            flag += f"  ({per_day[d]} provisional, from the screener)"
        lines.append(f"  {d:%Y-%m-%d}  {n:5d}{flag}")
    older = [d for d in thin if d < frame.index[-rows]] if len(frame) > rows else []
    if older:
        lines.append("older thin sessions: "
                     + ", ".join(f"{d:%Y-%m-%d}" for d in older))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:] or [os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "universe", "us_closes.csv")]:
        print(cache_report(p))
