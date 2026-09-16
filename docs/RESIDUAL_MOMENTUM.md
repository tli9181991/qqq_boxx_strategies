# Residual momentum

Rank on the part of a stock's return the market cannot explain.

Added because it is the only candidate tested in this lab that survived the
paired-across-the-surface test — the one that killed several better-looking
ideas in [`HYBRID_ANALYSIS.md`](HYBRID_ANALYSIS.md).

## The idea

Blitz, Huij & Martens (2009) regress each stock's returns on common factors,
keep the residual, and run the usual 12-1 momentum on that instead of on total
return. They report risk-adjusted profits about twice those of total-return
momentum, with greater consistency.

The construction here, on daily bars:

| | Rule |
|---|---|
| Market model | rolling single-factor regression against **QQQ**, `beta_window` (252) days |
| Residual | `r − β·r_mkt`, then net of its own rolling mean (the intercept) |
| Score | sum of residuals over the 12-1 window ÷ their standard deviation |
| Everything else | unchanged — six slots, band 10, absolute filter vs BOXX, cash leg |

Only the ranking changes. Held against `momentum` on the same universe and the
same engine, the difference can only be the score.

**One factor, not three.** The paper uses Fama-French three-factor over 36
months. The lab has no SMB/HML offline, and on a Nasdaq-100 universe the
market/tech factor is the one doing the damage. This is a real simplification
and it is the one the criticism below attacks.

## Why it belongs in *this* lab

The Top-6 book's weakness was never that it picks weak names. It is that it
picks the same bet six times:

```
held book mean pairwise correlation   0.43
effective independent bets            1.91  of 6 slots
```

Total-return momentum ranks a name highly partly for having a **large beta in a
rising market** — so it systematically selects the crowded trade, and the six
slots collapse into one. Removing the market component from the score attacks
that at the source. Effective bets go to **2.19**, better than the explicit
correlation cap achieves (2.11) and without any diversification constraint.

## What it measures on the cached window

2025-01-21 → 2026-09-08, 1bp + 5bp, one-day lag:

| | CAGR | Vol | Sharpe | Max DD | Calmar | Turnover |
|---|---|---|---|---|---|---|
| Top-6 NDX momentum (12-1) | 46.1% | 49.0% | 0.94 | −34.8% | 1.32 | 5.7× |
| **Top-6 residual momentum** | **68.4%** | **39.6%** | **1.41** | −34.9% | **1.96** | 14.5× |

More return at lower volatility. But the single-cell number is exactly what
this lab has learned not to trust, so:

### Paired across 41 `(n_hold, exit_rank)` cells

| | wins | median Δ |
|---|---|---|
| Sharpe | **36/41** | **+0.20** |
| Ann. vol (lower) | **40/41** | **−9.6%** |
| Calmar | 32/41 | +0.36 |
| Max drawdown (better) | 33/41 | +2.6% |
| CAGR | 23/41 | +4.2% |

Read the shape, not the headline: this is **primarily a risk reduction**
(vol lower in 40 of 41 cells) that holds return roughly flat-to-better, which
is precisely the claim in the source paper — "the risk reduction is significant
and comes with no detrimental effect on performance". The CAGR column at
23/41 is close to a coin flip and should not be leaned on.

For contrast, from the same test harness: the horizon blend managed 18/41 on
Sharpe, the correlation cap 9/16, and book vol targeting wins drawdown 41/41
while winning CAGR **0/41**.

### It has a plateau

The parameter the whole thing hangs on is the beta window, and it does not
matter much, which is what you want:

| `beta_window` | 126 | 189 | **252** | 378 | 504 |
|---|---|---|---|---|---|
| CAGR | 84.1% | 67.4% | **68.4%** | 59.9% | 58.0% |
| Sharpe | 1.70 | 1.38 | **1.41** | 1.31 | 1.34 |
| Vol | 37.6% | 40.4% | **39.6%** | 38.4% | 35.9% |

Every cell from half a year to two years beats the baseline's 0.94 Sharpe and
49.0% vol. The 12-1 horizon blend never had this.

### It survives costs, lag and both halves

| slippage | 5bp | 20bp | 50bp | 100bp |
|---|---|---|---|---|
| Residual CAGR | 68.4% | 64.7% | 57.7% | 46.7% |
| Residual Sharpe | 1.41 | 1.36 | 1.25 | 1.06 |
| *(baseline at 5bp)* | *46.1%* | | | *Sharpe 0.94* |

It trades 2.5× as much as the plain book, so this mattered — but at **100bp of
slippage, twenty times the default**, it still matches the baseline's return at
a higher Sharpe. Execution lag of 2 or 3 days changes little (Sharpe 1.38/1.50).

By sub-period:

| | 2025 H1 CAGR | H1 DD | 2025H2–26 CAGR | DD |
|---|---|---|---|---|
| Baseline 12-1 | 29.8% | −34.8% | 62.9% | −34.8% |
| Residual | 64.1% | **−17.3%** | 72.3% | −34.9% |

The first half is the 2025 selloff, and halving the drawdown there is the
clearest sign the market component was what hurt.

## Caveats

**The criticism is real and it lands harder here.** Ehsani & Linnainmaa (2022)
argue residual momentum may simply be harvesting factors *omitted* from the
regression that happen to be more autocorrelated than the ones included. A
one-factor model omits strictly more than the three-factor model they were
criticising. Treat this as *"momentum with the market bet removed"* — which is
measurable and is what the effective-bets number shows — rather than as a
distinct anomaly, which is contested.

**It does not compose with the correlation cap.** Stacking
`max_corr=0.75` on top gives 59.8% at Sharpe 1.33, worse than either alone.
They are substitutes: both decorrelate the book, and doing both over-constrains
a 99-name universe. Pick one.

**The lab's standing warnings still apply**, and a selection study is what
survivorship bias corrupts most: the ranking universe is today's Nasdaq-100
applied to all history, the sample is 410 trading days, and it contains one
dominant story. Point-in-time membership would move these numbers more than any
parameter here.

## Running it

```python
from qbs.config import Config, ResidualMomentumParams
from qbs.pipeline import run

cfg = Config()
cfg.resmom = ResidualMomentumParams(beta_window=252, standardise=True)
lab = run(cfg)
lab.summary_pretty.loc["Top-6 residual momentum"]
```

```bash
python run_backtest.py --offline --beta-window 126    # the plateau's other end
python run_backtest.py --offline --no-resmom          # skip it
```

`standardise=False` scores the raw cumulative residual instead of a
t-statistic on it. That is *not* the paper's construction and it is worse here
(Sharpe 1.10 against 1.41) — the division by residual vol is what makes this a
risk-adjusted score rather than a return.

## Sources

- Blitz, Huij & Martens (2009), *Residual Momentum* — [SSRN 2319861](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2319861)
- Ehsani & Linnainmaa (2022), on omitted-factor autocorrelation — discussed in [Enhanced Momentum Strategies, Hanauer & Windmüller](http://wp.lancs.ac.uk/mhf2019/files/2019/09/MHF-2019-076-Matthias-Hanauer.pdf)
- Barroso & Santa-Clara (2015), momentum risk management — the argument behind `BookVolTargetParams`
