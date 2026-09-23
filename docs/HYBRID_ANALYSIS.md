# Can the Top-6 momentum book earn more at the same risk?

A measured answer, on the cached window (2025-01-21 → 2026-09-08, 410 trading
days, 1bp commission + 5bp slippage, one-day execution lag).

> ⚠️ **Every number here is against the 12-1 momentum book**, which was the
> shipped default when this study was run. The default has since moved to 6-1,
> so the specific cells are stale. What is not stale is the method and its
> conclusion — that this sample's parameter surface is noise-dominated, and
> that a candidate must be scored *paired across cells* rather than at the
> default one. That is the part worth keeping.

The question: **increase return while retaining a similar risk level to the
Top-6 Nasdaq-100 momentum selection.** The baseline to beat:

| | CAGR | Vol | Sharpe | Max DD | Calmar | Turnover |
|---|---|---|---|---|---|---|
| Top-6 NDX momentum (12-1, band 10) | 46.1% | 49.0% | 0.94 | −34.8% | 1.32 | 5.7× |

---

## The short answer

**Nothing tested here raises return at constant risk in a way that survives
scrutiny**, and the reason is worth more than any of the candidates: on a
20-month sample the differences between these parameterisations are not
distinguishable from noise.

What *is* robust is the opposite trade — volatility targeting lowers risk
reliably and lowers return reliably. And one change, a **correlation cap on
selection**, buys volatility reduction without a systematic return cost,
which is the only thing in this study that moves the frontier rather than
sliding along it.

---

## Why "more return at the same risk" cannot come from leverage

At a fixed risk level, return ≈ Sharpe × volatility. So the only way to earn
more at the same risk is to raise the book's Sharpe — re-levering a book
whose Sharpe is unchanged just walks up and down the same line.

That is exactly what the vol-target dial does, and the grid says so plainly.
Scaling the book and allowing gearing (`max_weight` above 1) leaves Sharpe
flat across the whole surface:

| Target vol | max_weight | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|
| — (baseline) | — | 46.1% | 49.0% | 0.94 | −34.8% |
| 45% | 1.0 | 41.3% | 41.1% | 0.95 | −31.6% |
| 50% | 1.5 | 44.4% | 50.7% | 0.90 | −34.4% |
| 55% | 1.5 | 48.8% | 54.7% | 0.93 | −37.1% |
| 55% | 2.0 | 44.8% | 56.8% | 0.87 | −37.1% |

Sharpe never leaves 0.85–0.95. Gearing back up to 50% vol returns 44.4%
against the baseline's 46.1% — slightly *worse*, because the leverage is paid
for in turnover (11.9× against 5.7×). Leverage is not a source of return here.

---

## The methodology that matters: pair the comparison

The trap on this sample is that a single cell of the parameter surface tells
you almost nothing. Measured across `(n_hold, exit_rank)` combinations, the
**baseline itself** ranges from 19% to 141% CAGR — and the shipped default,
`(6, 10)`, is a *below-median* cell of its own surface:

| n_hold, exit_rank | 4,8 | 6,6 | 6,8 | **6,10 (shipped)** | 6,12 | 6,25 |
|---|---|---|---|---|---|---|
| Baseline CAGR | 99.6% | 76.2% | 80.8% | **46.1%** | 63.9% | 35.5% |

So any variant measured only at `(6, 10)` is being compared against one of the
baseline's unluckiest draws, and will look brilliant for no reason at all.
Every candidate below is therefore scored **paired across 41 cells** of that
surface — same cell, both strategies, count the wins.

### What that test does to the candidates

| Candidate (vs plain 12-1, 41 paired cells) | Sharpe wins | CAGR wins | Max DD better | median ΔCAGR |
|---|---|---|---|---|
| + book vol-target 45% | **36/41** | **0/41** | **41/41** | −3.1% |
| + book vol-target 35% | 23/41 | **0/41** | **41/41** | −12.5% |
| + per-name SMA(200) gate | 24/41 | 24/41 | 12/41 | +0.9% |
| + horizon blend 6…12 | 18/41 | 23/41 | 4/41 | +0.6% |
| + risk-adjusted score (mom/vol) | 17/41 | 6/41 | 39/41 | −11.6% |

Three readings:

1. **Vol targeting is the only unambiguous effect in the study** — and it is
   unambiguous in *both* directions. It improves drawdown in 41 cells out of
   41, and improves return in **zero**. It is a risk dial, precisely as
   `BookVolTargetParams` already documents. It cannot answer this question.
2. **The horizon blend is dead.** It looked like +30 points of CAGR at the
   default cell (75.7% against 46.1%) and the sub-period split liked it too —
   but paired across the surface it wins 18/41 on Sharpe, a coin flip, with
   *worse* drawdown in 37 of 41 cells. That gain was the shipped cell being
   unlucky, not the blend being good.
3. **Nothing raises return.** Not one candidate wins CAGR in more than 24 of
   41 cells.

> The horizon blend is the cautionary tale of this document. It had everything
> a finding is supposed to have — a plateau on a 2-D map, both sub-periods
> positive, a clean look-ahead test, robustness to costs and to execution lag
> — and it was still noise. Only the paired test caught it.

---

## The one change that moves the frontier

The Top-6 book's real structural weakness is not its lookback. It is that
**six slots are not six bets**:

```
mean pairwise correlation of the held book   0.43
effective independent bets, 6/(1+5ρ)         1.91  of 6 slots
```

Momentum is a trend signal, so the names trending hardest at any moment tend
to be one sector. On the last cached bar the book held `LRCX, MU, AMAT, INTC,
AMD, MRVL` — six semiconductors. The README's own "concentration blow-up"
episode (−30% while VIX sat at 16–20) is that fact expressing itself.

`MomentumParams.max_corr` refuses a candidate whose trailing correlation with
an already-selected name exceeds a cap; the slot goes to the next name down,
or to cash. `sweep_corr_cap()` on the cached window:

| Max corr | CAGR | Vol | Sharpe | Max DD | Turnover | Book corr | Eff. bets |
|---|---|---|---|---|---|---|---|
| off | 46.1% | 49.0% | 0.94 | −34.8% | 5.7× | 0.43 | 1.91 |
| 0.85 | 52.7% | 49.1% | 1.03 | −36.0% | 6.4× | 0.40 | 2.01 |
| **0.75** | **62.4%** | **48.1%** | **1.17** | −35.7% | 7.4× | 0.37 | **2.11** |
| 0.65 | 49.4% | 44.9% | 1.03 | −35.7% | 13.5× | 0.32 | 2.31 |
| 0.55 | 49.9% | 44.6% | 1.04 | −35.7% | 14.1× | 0.30 | 2.39 |

**Read the last two columns, not the first.** Effective bets rise
monotonically and volatility falls monotonically — that part is close to
mechanical, and it is what the parameter is for. CAGR is not monotone, which
is what noise looks like. Paired across the surface the cap lowers volatility
in 12 of 16 cells (median −1.2%) while its return effect is a coin flip
(Sharpe 9/16, CAGR 8/16).

That combination — reliably less volatility, no systematic return cost — is
what distinguishes it from vol targeting, which buys the same risk reduction
and charges return for it in 41 cells out of 41.

**The honest expectation is "same risk, same-ish return, a genuinely more
diversified book", not the +16 points the default cell happens to show.**

---

## What to actually run

```python
from qbs.config import BookVolTargetParams, Config, MomentumParams
from qbs.pipeline import run, sweep_corr_cap

cfg = Config()
cfg.momentum = MomentumParams(max_corr=0.75)   # six bets, not one bet six times
cfg.book_vol = BookVolTargetParams(target_vol=0.45)   # the risk dial, set high

lab = run(cfg)
sweep_corr_cap(lab)      # look for the monotone columns, not the best CAGR
```

Measured at the shipped `(6, 10)` cell:

| | CAGR | Vol | Sharpe | Max DD | Calmar | Turnover |
|---|---|---|---|---|---|---|
| Baseline Top-6 | 46.1% | 49.0% | 0.94 | −34.8% | 1.32 | 5.7× |
| + corr cap 0.75 | 62.4% | 48.1% | 1.17 | −35.7% | 1.75 | 7.4× |
| + corr cap + VT 45% | 55.9% | 40.6% | 1.20 | **−30.8%** | **1.81** | 8.4× |
| + corr cap + VT 35% | 48.4% | 35.1% | 1.19 | −25.2% | 1.92 | 8.9× |

Set `target_vol` by the drawdown you are willing to sit through, not by the
CAGR column. The cap is the part with a mechanism behind it; the vol target is
the part with 41-out-of-41 evidence behind it. Neither is a return generator.

---

## What would actually settle this

Every number above inherits the README's two standing warnings, and on a
question about *selection* they bind harder than usual:

- **Survivorship bias.** The ranking universe is today's Nasdaq-100 applied to
  all history. A study of whether one selection rule picks better than another
  is exactly the study that bias corrupts most. Supply point-in-time
  membership (`load_pit_universe`) before believing any of it.
- **One regime, 410 days.** The sample contains a single dominant story — the
  2025–26 semiconductor run — and every candidate here is really being asked
  "did you hold the semis?" A 20-month window cannot separate a selection edge
  from that.

The parameter surface swinging 19%→141% on the *unmodified* strategy is the
measurement of how little this sample can resolve. Until the window is longer
and the membership is point-in-time, the defensible changes are the ones with
a mechanism — decorrelation, risk scaling — and not the ones with a backtest.
