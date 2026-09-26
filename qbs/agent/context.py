"""Pre-computed JSON context for the dashboard's analyst.

Two blocks, computed by this package and handed to the model up front
instead of through tool calls:

    Market overview (~2,400 US names)  ->  market_context()  -+
    NDX + watchlist closes             ->  stock_context()   -+->  Gemini
    news / fundamentals tools          ---------------------- +

Why up front
------------
Everything in these two blocks is already computed by the dashboard -- the
breadth table, the checklist, both books' rankings, the chart's EMAs and
levels. Making the model fetch it one tool call at a time costs a round trip
per number and lets it decide to skip one. Handed over as one aggregate, the
model can answer "why do the two momentum ranks differ?" with no call at
all, and only reaches for a tool when the question needs something the
dashboard does not hold: fundamentals, and news.

Only aggregates reach the model, never the ~2,400-name frame itself.

Scope
-----
`stock_context` covers exactly what the dashboard can chart: a Nasdaq-100
constituent, or a watchlist name. A watchlist name outside the index is
labelled `watchlist_outside_ndx`, and its momentum is a PLACEMENT rank --
where it would sit among the constituents -- with `currently_held` False by
construction, so the model cannot mistake it for a constituent or a holding.

Every value is JSON-safe: floats rounded, NaN turned into null.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..breadth import BreadthParams, atr_class, momentum_label, sector_breakdown
from ..breakout import atr, closes_to_bars, levels_in_view, sr_levels
from ..candles import volume_stats
from ..config import BreakoutParams, MomentumParams, ResidualMomentumParams
from ..data import sessions_behind
from .evidence import Book
from .market import CHECKLIST_QUESTIONS, MANUAL_ROWS, Market
from .stock import _ret, _usable_ohlc


def _num(v, nd: int = 4):
    """A float rounded for JSON, or None for NaN / missing."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return int(round(f)) if nd == 0 else round(f, nd)


def _today() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%d")


def to_json(ctx: Dict, compact: bool = False) -> str:
    """Pretty for a person, compact for the prompt -- same content."""
    if compact:
        return json.dumps(ctx, separators=(",", ":"), ensure_ascii=False)
    return json.dumps(ctx, indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------
# Context 1: the market
# --------------------------------------------------------------------------

def market_context(market: Market, top_sectors: int = 5,
                   p: Optional[BreadthParams] = None) -> Dict:
    """The Market overview tab as one aggregate: breadth, index stretch, the
    breadth trend, the checklist and sector leadership."""
    p = p or BreadthParams()
    res = market.get_breadth()
    tbl = res.table
    if tbl is None or tbl.empty:
        return {"error": "market breadth could not be computed"}
    last = tbl.iloc[-1]

    def ago(col: str, n: int):
        return tbl[col].iloc[-1 - n] if len(tbl) > n else np.nan

    share = 100.0 * tbl["mli_n"] / tbl["n_stocks"]
    ctx: Dict = {
        "analysis_date": _today(),
        "data_as_of": f"{tbl.index[-1]:%Y-%m-%d}",
        "sessions_behind": int(sessions_behind(tbl.index[-1])),
        "universe": ("Nasdaq-100 constituents -- FALLBACK, an index and not "
                     "the market" if market.index_fallback else
                     "US common stocks and ADRs"),
        "sample_size": int(last["n_stocks"]),
        "breadth": {
            "up_4pct": _num(last["up4"], 0),
            "down_4pct": _num(last["dn4"], 0),
            "up_down_ratio": _num(last["up4"] / last["dn4"]
                                  if last["dn4"] else np.nan, 2),
            "above_20d_pct": _num(last["pct_above_fast"], 1),
            "above_50d_pct": _num(last["pct_above_slow"], 1),
            "leader_count": int(last["mli_n"]),
            "leader_share_pct": _num(share.iloc[-1], 1),
            "leader_advance_pct": _num(last["mli_up_pct"], 1),
            "leader_avg_move_pct": _num(last["mli_pct"], 2),
        },
        "index_condition": {
            "spy_atr_from_ema50": _num(last.get("spy_atr"), 2),
            "qqq_atr_from_ema50": _num(last.get("qqq_atr"), 2),
            "spy_state": atr_class(last.get("spy_atr", np.nan), p),
            "qqq_state": atr_class(last.get("qqq_atr", np.nan), p),
            "stretched_beyond_atr": p.atr_stretched,
        },
        "breadth_trend_5d": {
            "above_20d_change": _num(last["pct_above_fast"]
                                     - ago("pct_above_fast", 5), 1),
            "above_50d_change": _num(last["pct_above_slow"]
                                     - ago("pct_above_slow", 5), 1),
            "leader_count_change": _num(last["mli_n"] - ago("mli_n", 5), 0),
            "up_4pct_avg": _num(tbl["up4"].tail(5).mean(), 1),
            "down_4pct_avg": _num(tbl["dn4"].tail(5).mean(), 1),
        },
        "breadth_trend_20d": {
            "above_50d_change": _num(last["pct_above_slow"]
                                     - ago("pct_above_slow", 20), 1),
            "leader_share_pct_20d_ago": _num(share.iloc[-21]
                                             if len(share) > 20 else np.nan, 1),
        },
    }

    checks = market.get_checklist()
    rows = [{"key": c["key"],
             "question": CHECKLIST_QUESTIONS.get(c["key"], c["key"]),
             "answer": c.get("answer"), "reading": c.get("reading", "")}
            for c in checks]
    ctx["checklist"] = {
        "automatic_score": sum(1 for c in checks if c.get("answer")),
        "automatic_rows": len(checks),
        "triggered": [c["key"] for c in checks if c.get("answer")],
        "rows": rows,
        "note": "Yes is the BEARISH reading on every row.",
    }

    if market.sectors:
        sect = sector_breakdown(market.closes, market.sectors,
                                asof=market.asof, volumes=market.volumes)
        ctx["sector_leadership"] = [
            {"sector": r.sector, "leaders": int(r.n),
             "leader_share_pct": _num(r.share_pct, 1),
             "universe_share_pct": _num(r.pool_pct, 1),
             "excess_pp": _num(r.excess_pp, 1)}
            for r in sect.head(top_sectors).itertuples()]
        ctx["top3_sector_concentration_pct"] = _num(
            sect["share_pct"].head(3).sum() if not sect.empty else np.nan, 1)
    else:
        ctx["sector_leadership"] = None

    limits = [f"Manual checklist questions excluded: {'; '.join(MANUAL_ROWS)}"]
    if not res.has_volume:
        limits.append("Leader volume leg (> 300k shares/day) not applied: "
                      "leader_count is an over-estimate")
    if market.index_fallback:
        limits.append("Measured over ~100 Nasdaq-100 names, not the US "
                      "market; 4% counts are not comparable to a 2,400-name "
                      "sample")
    if not market.sectors:
        limits.append("No sector map in this run, so no sector leadership")
    if market.spy is None and market.spy_ohlc is None:
        limits.append("No SPY series in this run")
    ctx["interpretation_limits"] = limits
    return ctx


# --------------------------------------------------------------------------
# Context 2: one stock
# --------------------------------------------------------------------------

def momentum_ranks(
    book: Book,
    ticker: str,
    momentum: Optional[MomentumParams] = None,
    residual: Optional[ResidualMomentumParams] = None,
) -> Dict[str, tuple]:
    """`{"normal": (rank, score), "residual": (rank, score)}` for one name.

    Through `qbs.shadow`, the dashboard's own watchlist placement: a
    constituent reports its standing rank, an outsider the rank it would
    take among the constituents -- nothing else moves.
    """
    from ..shadow import watchlist_residual_rows, watchlist_rows

    t = ticker.upper().strip()
    s = book.closes(t)
    if s is None:
        return {}
    frame = s.reindex(book.universe.index).to_frame(t)
    rows = watchlist_rows(book.universe, book.safe, frame,
                          momentum or MomentumParams(), asof=book.asof)
    norm = next(((r["rank"], r["score"]) for r in rows if r["symbol"] == t),
                (np.nan, np.nan))
    res = watchlist_residual_rows(book.universe, book.safe,
                                  book.prices["QQQ"], frame,
                                  residual or ResidualMomentumParams(),
                                  asof=book.asof)
    return {"normal": norm, "residual": res.get(t, (np.nan, np.nan))}


def current_holdings(
    book: Book,
    momentum: Optional[MomentumParams] = None,
    residual: Optional[ResidualMomentumParams] = None,
) -> Dict[str, List[str]]:
    """What each book holds on the last bar. The dashboard passes its own."""
    from ..strategies import cross_sectional_momentum, residual_momentum

    out = {}
    mom = cross_sectional_momentum(book.universe, book.safe,
                                   momentum or MomentumParams())
    res = residual_momentum(book.universe, book.safe, book.prices["QQQ"],
                            residual or ResidualMomentumParams())
    for key, sig in (("normal", mom), ("residual", res)):
        log = sig.holdings_log
        out[key] = list(log[max(log)]) if log else []
    return out


def _book_block(kind: str, outsider: bool, rank_score: tuple,
                held: Sequence[str], ticker: str, n_hold: int,
                exit_rank: int, label: str) -> Dict:
    rank, score = rank_score if rank_score else (np.nan, np.nan)
    blk: Dict = {"lookback": label}
    if outsider:
        blk["placement_rank_against_ndx"] = _num(rank, 0)
    else:
        blk["rank"] = _num(rank, 0)
    blk["score"] = _num(score, 4)
    blk["score_unit"] = ("trailing return (fraction)" if kind == "normal"
                         else "t-statistic of market-neutral drift")
    blk["currently_held"] = False if outsider else ticker in set(held)
    blk["book_slots"] = n_hold
    blk["sell_below_rank"] = exit_rank
    if rank != rank:
        blk["unranked_reason"] = ("filtered out: too little history, or it "
                                  "lost to the safe asset (BOXX)")
    return blk


def stock_context(
    book: Book,
    ticker: str,
    market: Optional[Market] = None,
    watchlist: Sequence[str] = (),
    ohlc: Optional[pd.DataFrame] = None,
    spy: Optional[pd.Series] = None,
    ranks: Optional[Dict[str, tuple]] = None,
    held: Optional[Dict[str, List[str]]] = None,
    momentum: Optional[MomentumParams] = None,
    residual: Optional[ResidualMomentumParams] = None,
) -> Dict:
    """One chartable name: membership, both books' momentum, trend,
    relative performance, levels and volume."""
    t = ticker.upper().strip()
    mp = momentum or MomentumParams()
    rp = residual or ResidualMomentumParams()
    close = book.closes(t)
    if close is None:
        return {"ticker": t, "error": "not a Nasdaq-100 constituent or a "
                "watchlist name with prices -- no context is built for it"}
    outsider = book.is_outsider(t)
    asof = close.index[-1]
    last = float(close.iloc[-1])
    watched = t in {w.upper() for w in watchlist}

    ranks = ranks if ranks is not None else momentum_ranks(book, t, mp, rp)
    held = held if held is not None else current_holdings(book, mp, rp)

    ctx: Dict = {
        "ticker": t,
        "analysis_date": _today(),
        "data_as_of": f"{asof:%Y-%m-%d}",
        "sessions_behind": int(sessions_behind(asof)),
        "membership": "watchlist_outside_ndx" if outsider else "Nasdaq-100",
        "watchlist": watched or outsider,
        "sector": (market.sectors.get(t) if market is not None
                   and market.sectors else None),
        "normal_momentum": _book_block(
            "normal", outsider, ranks.get("normal"), held.get("normal", []),
            t, mp.n_hold, mp.exit_rank, momentum_label(mp)),
        "residual_momentum": _book_block(
            "residual", outsider, ranks.get("residual"),
            held.get("residual", []), t, rp.n_hold, rp.exit_rank,
            momentum_label(rp)),
    }

    # ---- trend: the chart's lines ---------------------------------------
    trend: Dict = {"close": _num(last, 2)}
    for label, n in (("1d", 1), ("1w", 5), ("1m", 21), ("3m", 63),
                     ("6m", 126), ("12m", 252)):
        trend[f"return_{label}"] = _num(_ret(close, n))
    emas = {}
    for n in (10, 20, 50, 200):
        e = close.ewm(span=n, adjust=False, min_periods=n).mean()
        emas[n] = e.iloc[-1]
        trend[f"vs_ema{n}"] = _num(last / e.iloc[-1] - 1 if e.iloc[-1] == e.iloc[-1]
                                   else np.nan)
        trend[f"ema{n}_rising"] = (bool(e.iloc[-1] > e.iloc[-6])
                                   if len(e) > 5 and e.iloc[-1] == e.iloc[-1]
                                   else None)
    sma200 = close.rolling(200, min_periods=200).mean().iloc[-1]
    trend["vs_sma200"] = _num(last / sma200 - 1 if sma200 == sma200 else np.nan)
    vals = [emas[n] for n in (10, 20, 50, 200)]
    if all(v == v for v in vals):
        trend["ema_order"] = ("bullish" if all(a > b for a, b in zip(vals, vals[1:]))
                              else "bearish" if all(a < b for a, b in zip(vals, vals[1:]))
                              else "mixed")
    bars = _usable_ohlc(ohlc, asof)
    hi = bars["High"] if bars is not None else close
    trend["off_52w_high"] = _num(1 - last / float(hi.tail(252).max()))
    trend["above_52w_low"] = _num(last / float(
        (bars["Low"] if bars is not None else close).tail(252).min()) - 1)
    ab = bars if bars is not None else closes_to_bars(close.to_frame(t))[t]
    a = float(atr(ab, 14).iloc[-1])
    trend["atr_pct"] = _num(a / last)
    trend["atr_source"] = "real high/low" if bars is not None else "closes only (understates)"
    trend["realised_vol_20d"] = _num(close.pct_change().tail(20).std() * np.sqrt(252))
    ctx["trend"] = trend

    # ---- relative performance ------------------------------------------
    qqq = book.prices["QQQ"].loc[:asof].dropna() if "QQQ" in book.prices else None
    spy = spy.loc[:asof].dropna() if spy is not None else None
    rel: Dict = {}
    for label, n in (("1m", 21), ("3m", 63), ("6m", 126)):
        r, q = _ret(close, n), _ret(qqq, n)
        rel[f"qqq_return_{label}"] = _num(q)
        rel[f"spy_return_{label}"] = _num(_ret(spy, n))
        rel[f"excess_vs_qqq_{label}"] = _num(r - q)
    if qqq is not None:
        j = pd.concat([close.pct_change(), qqq.pct_change()], axis=1,
                      join="inner").dropna().tail(126)
        if len(j) >= 60 and j.iloc[:, 1].var() > 0:
            rel["beta_qqq_126d"] = _num(j.iloc[:, 0].cov(j.iloc[:, 1])
                                        / j.iloc[:, 1].var(), 2)
    if market is not None and t in market.closes.columns:
        mc = market.closes.loc[:asof]
        for label, n in (("1m", 21), ("3m", 63)):
            if len(mc) > n:
                rr = (mc.iloc[-1] / mc.iloc[-1 - n] - 1).dropna()
                if t in rr:
                    rel[f"market_percentile_{label}"] = _num(
                        (rr < rr[t]).mean() * 98 + 1, 0)
        lm = market.leaders()
        if asof in lm.index:
            rel["is_momentum_leader"] = bool(lm.loc[asof].get(t, False))
    ctx["relative"] = rel

    # ---- levels ------------------------------------------------------------
    levels = sr_levels(ab.tail(504), BreakoutParams())
    w = close.tail(252)
    _, _, overhead = levels_in_view(levels, last, float(w.min()), float(w.max()), 30)
    above = sorted(L for L in levels if L > last)
    below = sorted((L for L in levels if L <= last), reverse=True)
    ctx["levels"] = {
        "nearest_resistance": _num(above[0], 2) if above else None,
        "resistance_distance": _num(above[0] / last - 1) if above else None,
        "nearest_support": _num(below[0], 2) if below else None,
        "support_distance": _num(below[0] / last - 1) if below else None,
        "has_overhead_resistance": bool(overhead),
        "rule": "swing levels from history up to data_as_of only",
    }

    # ---- volume ------------------------------------------------------------
    if bars is not None and "Volume" in bars.columns and bars["Volume"].notna().any():
        vol = bars["Volume"].astype(float)
        vs = volume_stats(vol, bars["Close"], window=20)
        chg = bars["Close"].diff().tail(20)
        v20 = vol.reindex(chg.index)
        dn = float(v20[chg < 0].sum())
        ctx["volume"] = {
            "latest_ratio": _num(vs["ratio"], 2),
            "avg_dollar_volume_20d": _num(vs["avg_value"], 0),
            "up_down_volume_ratio_20d": _num(float(v20[chg > 0].sum()) / dn
                                             if dn > 0 else np.nan, 2),
        }
    else:
        ctx["volume"] = None

    ctx["last_5_sessions"] = [
        {"date": f"{d:%Y-%m-%d}", "close": _num(close.loc[d], 2),
         "change": _num(close.loc[d] / close.shift(1).loc[d] - 1)}
        for d in close.index[-5:][::-1]]

    notes = ["returns and distances are fractions (0.08 = 8%)"]
    if outsider:
        notes.append("not a Nasdaq-100 constituent: momentum ranks are "
                     "PLACEMENTS among the constituents, and no book can "
                     "hold it")
    if ctx["volume"] is None:
        notes.append("no volume in this run")
    ctx["notes"] = notes
    return ctx


def context_block(market_ctx: Optional[Dict], stock_ctx: Optional[Dict]) -> str:
    """Both contexts as the prompt carries them: labelled, compact JSON.

    A missing block is stated rather than omitted, so the model says the
    market (or the stock) is unavailable instead of assuming it.
    """
    parts = []
    for name, ctx in (("MARKET_CONTEXT", market_ctx), ("STOCK_CONTEXT", stock_ctx)):
        body = (to_json(ctx, compact=True) if ctx else
                '{"error":"not available in this run"}')
        parts.append(f"{name}\n```json\n{body}\n```")
    return "\n\n".join(parts)
