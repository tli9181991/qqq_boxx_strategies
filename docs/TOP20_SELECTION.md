# Choosing six from the momentum top-20

**Question.** The Top-6 book holds the six highest 6-1 momentum names. What
if the same ranking only defined a pool — the top 20 — and a different rule
picked the six held from inside it? Which choice gives the best result and
the lowest turnover?

**Short answer.** Keep buying the top six, but stop selling them so early.
Holding a name until it leaves the top 20, rather than past rank 8, cuts
turnover by about two thirds (21× → 7×, 25× → 9× a year) with return and
drawdown in the same range — better in one window, worse in the other.
Picking *other* names from the pool (ranks 7–20, lowest vol, lowest beta,
nearest the high, best residual momentum) does not hold up: each loses to the
plain top six in most of the six pool-size × window cells, several by a lot.

| | Default window | | | | Extended window | | | |
|---|---|---|---|---|---|---|---|---|
| | CAGR | Vol | Max DD | Turnover | CAGR | Vol | Max DD | Turnover |
| Shipped Top-6 (sell past rank 8) | 48.1% | 49.2% | −40.4% | 21.1× | 34.0% | 40.1% | −40.4% | 25.4× |
| **Top-6, sell on leaving the top 20** | 31.3% | 49.1% | −39.7% | **7.4×** | 40.3% | 40.2% | −39.7% | **8.9×** |
| Ranks 7–20, sell on leaving the top 20 | 46.9% | 45.4% | −33.6% | 9.6× | 22.4% | 36.7% | −33.6% | 13.3× |
| Top-20 equal weight (20 slots) | 34.9% | 33.3% | −28.8% | 34.1× | 20.2% | 28.2% | −28.8% | 40.1× |
| Lowest 60-day vol in the top 20 | 9.6% | 26.3% | −19.2% | 30.3× | 2.4% | 23.6% | −23.9% | 33.0× |

## How it was tested

* **Same ranking, same machinery.** 6-1 momentum among Nasdaq-100 names that
  beat BOXX over the same window; the pool is its top N. The second rule is
  passed to `cross_sectional_momentum` as its `score`, so slot weighting, the
  cash leg, costs (1 bp + 5 bp slippage), the one-day lag and the engine are
  the shipped book's. Rebuilding the ranking this way reproduces the shipped
  book exactly (48.1% / −40.4% / 21.1×).
* **Exit.** "Exit 20" sells a name only when it leaves the top-20 pool;
  "exit 10" also sells it once it falls past 10th on the second rule.
* **Windows.** Default (2025-01-20 on) and extended (2022-01-03 on, a 0%
  cash proxy before BOXX listed), as in `docs/STOP_LOSS.md`.
* **Robustness.** Each rule at pools of 15, 20 and 30 in both windows — six
  cells — scored against "top 6 by momentum" in the same pool.

## Reading it

1. **The band is the turnover lever, not the pick.** "Top 6 by momentum,
   exit 20" buys exactly what the shipped book buys and simply holds longer.
   It has the lowest turnover of anything tested (median 8.2× across the six
   cells) and the highest median CAGR (41.7%). Its return relative to the
   shipped exit-8 book flips sign between windows — 31% vs 48%, then 40% vs
   34% — so the robust claim is the turnover cut, not a return gain.
   `sweep_band()` shows the same curve on the plain book.
2. **Ranks 7–20 are weaker names, and it shows.** Median CAGR 17% against
   42% for the top six of the same pool, beating it on Calmar in 1 of 6
   cells. The one good cell (default window, pool 20) is the outlier.
3. **Low vol and low beta pick the pool's weakest trends.** Within a pool of
   momentum leaders the calm, low-beta names are the ones the trend is
   leaving. Drawdown falls to around −20%, but CAGR collapses to 2–10%.
   Book vol targeting (`momentum_vt`) gets the risk cut without that.
4. **Residual momentum inside the pool** was strong in the default window
   (53–68% CAGR) and weak in the extended one (17–23%), beating the top six
   in 2 of 6 cells. The standalone residual book (`resmom`) is the better
   way to use that score.
5. **"Strongest last month"** tied with the top six (3 of 6 cells, similar
   turnover). No reason to prefer it.

The exit-10 row for "ranks 7+" is not a fair test: a held name that climbs
into the top 6 sorts last on that rule and is sold, which is where its 66–78×
turnover comes from. The exit-20 row keeps such a name.

## Exit at entry rank + 8

A follow-up test gave each held name its own exit line: the rank it was
bought at plus 8. A name bought at rank 2 is sold past 10; one bought at 6,
past 14. `MomentumParams.exit_drop` implements it (off by default).

| 6-stock book | Default: CAGR | Max DD | Turnover | Extended: CAGR | Max DD | Turnover |
|---|---|---|---|---|---|---|
| Fixed exit 8 (shipped) | 48.1% | −40.4% | 21.1× | 34.0% | −40.4% | 25.4× |
| Entry rank + 8 | 51.5% | −39.2% | 10.9× | 42.4% | −39.2% | 13.4× |
| Fixed exit 14 | 55.0% | −38.4% | 9.4× | 40.4% | −38.4% | 11.2× |

At 4, 6 and 8 names in both windows, entry rank + 8 beat the shipped exit on
CAGR in 6 of 6 cells at about half the turnover, and beat a fixed exit of
`n_hold + 8` on CAGR in 4 of 6; the fixed exit had the lower turnover in all
6. Most of the gain is the wider band. Drops of 4 and 12 were worse than 8.

It is logged, not traded: preflight scores it daily into
`var/strategy_comparison.csv` as `momentum6_exitdrop`, beside `momentum6`
(`QBS_COMPARE_EXIT_DROP`, see `deploy/docker/README.md`).

## BE and TSM

Neither is a Nasdaq-100 constituent — Bloom Energy and TSMC's ADR both list
on the NYSE — so neither is in this universe, and no rule here could have
held them. The dashboard's watchlist interpolates them into the ranking to
show where they *would* place, which is the right way to watch them.
Adding them to the backtest because they have done well would be hindsight:
the names that also sat outside the top six and did badly are not on the
list.

## Caveats

Survivorship bias (today's index applied to history), and a sample of 4.7
years at most. Six cells is a coarse robustness check.

## Reproduce

```bash
python scripts/top20_selection_study.py        # about 3 minutes, offline
```

---

# Results

## default window (2025-01-20 to 2026-09-08)

Pool = top 20 by 6-1 momentum. Exit 20 = sold only on leaving the pool; exit 10 = also sold past 10th on the rule.

| Rule                           | Pool   |   Exit | CAGR   | Vol   |   Sharpe | MaxDD   |   Calmar | Turnover   | Exposure   |
|:-------------------------------|:-------|-------:|:-------|:------|---------:|:--------|---------:|:-----------|:-----------|
| shipped Top-6 (exit rank 8)    | -      |      8 | 48.1%  | 49.2% |     0.96 | -40.4%  |     1.19 | 21.1x      | 100.0%     |
| top-20 equal weight (20 slots) | 20     |     20 | 34.9%  | 33.3% |     0.94 | -28.8%  |     1.21 | 34.1x      | 99.9%      |
| top 6 by momentum              | 20     |     20 | 31.3%  | 49.1% |     0.72 | -39.7%  |     0.79 | 7.4x       | 100.0%     |
| top 6 by momentum              | 20     |     10 | 50.8%  | 49.8% |     0.99 | -42.1%  |     1.21 | 14.5x      | 100.0%     |
| ranks 7+ by momentum           | 20     |     20 | 46.9%  | 45.4% |     0.98 | -33.6%  |     1.39 | 9.6x       | 100.0%     |
| ranks 7+ by momentum           | 20     |     10 | 37.3%  | 38.4% |     0.91 | -31.1%  |     1.2  | 65.6x      | 100.0%     |
| lowest 60d vol                 | 20     |     20 | 9.6%   | 26.3% |     0.32 | -19.2%  |     0.5  | 30.3x      | 100.0%     |
| lowest 60d vol                 | 20     |     10 | 3.8%   | 24.0% |     0.11 | -18.7%  |     0.2  | 37.5x      | 100.0%     |
| lowest beta to QQQ             | 20     |     20 | 8.9%   | 26.4% |     0.3  | -22.9%  |     0.39 | 27.5x      | 100.0%     |
| lowest beta to QQQ             | 20     |     10 | 8.6%   | 22.7% |     0.3  | -18.3%  |     0.47 | 34.4x      | 100.0%     |
| best residual momentum         | 20     |     20 | 53.1%  | 42.6% |     1.12 | -34.9%  |     1.52 | 11.3x      | 100.0%     |
| best residual momentum         | 20     |     10 | 68.2%  | 46.0% |     1.27 | -36.1%  |     1.89 | 17.8x      | 100.0%     |
| nearest 52w high               | 20     |     20 | 40.4%  | 44.6% |     0.89 | -37.6%  |     1.07 | 9.4x       | 100.0%     |
| nearest 52w high               | 20     |     10 | 27.4%  | 32.9% |     0.78 | -25.4%  |     1.08 | 50.6x      | 100.0%     |
| strongest last month           | 20     |     20 | 44.5%  | 49.8% |     0.91 | -41.1%  |     1.08 | 7.2x       | 100.0%     |
| strongest last month           | 20     |     10 | 57.2%  | 42.3% |     1.19 | -31.0%  |     1.84 | 45.3x      | 100.0%     |

## extended window (2022-01-03 to 2026-09-08)

Pool = top 20 by 6-1 momentum. Exit 20 = sold only on leaving the pool; exit 10 = also sold past 10th on the rule.

| Rule                           | Pool   |   Exit | CAGR   | Vol   |   Sharpe | MaxDD   |   Calmar | Turnover   | Exposure   |
|:-------------------------------|:-------|-------:|:-------|:------|---------:|:--------|---------:|:-----------|:-----------|
| shipped Top-6 (exit rank 8)    | -      |      8 | 34.0%  | 40.1% |     0.84 | -40.4%  |     0.84 | 25.4x      | 100.0%     |
| top-20 equal weight (20 slots) | 20     |     20 | 20.2%  | 28.2% |     0.66 | -28.8%  |     0.7  | 40.1x      | 98.4%      |
| top 6 by momentum              | 20     |     20 | 40.3%  | 40.2% |     0.95 | -39.7%  |     1.02 | 8.9x       | 100.0%     |
| top 6 by momentum              | 20     |     10 | 37.8%  | 40.4% |     0.91 | -42.1%  |     0.9  | 18.0x      | 100.0%     |
| ranks 7+ by momentum           | 20     |     20 | 22.4%  | 36.7% |     0.63 | -33.6%  |     0.67 | 13.3x      | 100.0%     |
| ranks 7+ by momentum           | 20     |     10 | 13.5%  | 31.9% |     0.44 | -35.5%  |     0.38 | 77.6x      | 100.0%     |
| lowest 60d vol                 | 20     |     20 | 2.4%   | 23.6% |     0.06 | -23.9%  |     0.1  | 33.0x      | 100.0%     |
| lowest 60d vol                 | 20     |     10 | 2.7%   | 21.6% |     0.06 | -23.3%  |     0.12 | 42.0x      | 100.0%     |
| lowest beta to QQQ             | 20     |     20 | 4.3%   | 24.3% |     0.15 | -23.5%  |     0.18 | 30.4x      | 100.0%     |
| lowest beta to QQQ             | 20     |     10 | 2.1%   | 21.9% |     0.04 | -21.5%  |     0.1  | 38.5x      | 100.0%     |
| best residual momentum         | 20     |     20 | 23.0%  | 35.3% |     0.66 | -34.9%  |     0.66 | 15.2x      | 100.0%     |
| best residual momentum         | 20     |     10 | 16.6%  | 35.7% |     0.51 | -41.0%  |     0.41 | 21.9x      | 100.0%     |
| nearest 52w high               | 20     |     20 | 29.4%  | 35.4% |     0.8  | -37.6%  |     0.78 | 12.9x      | 100.0%     |
| nearest 52w high               | 20     |     10 | 15.1%  | 27.7% |     0.52 | -25.4%  |     0.6  | 53.3x      | 100.0%     |
| strongest last month           | 20     |     20 | 41.9%  | 39.5% |     0.99 | -41.1%  |     1.02 | 9.8x       | 100.0%     |
| strongest last month           | 20     |     10 | 30.0%  | 34.6% |     0.83 | -34.0%  |     0.88 | 47.2x      | 100.0%     |

## Robustness: pools of 15 / 20 / 30, both windows (exit = pool)

| Rule                   |   Cells | Calmar > top-6 in pool   | Median CAGR   |   Median Calmar | Median MaxDD   | Median turnover   |
|:-----------------------|--------:|:-------------------------|:--------------|----------------:|:---------------|:------------------|
| top 6 by momentum      |       6 | 0/6                      | 41.7%         |            1.05 | -39.7%         | 8.2x              |
| ranks 7+ by momentum   |       6 | 1/6                      | 17.3%         |            0.54 | -32.1%         | 11.5x             |
| lowest 60d vol         |       6 | 1/6                      | 9.1%          |            0.52 | -21.5%         | 30.2x             |
| lowest beta to QQQ     |       6 | 0/6                      | 5.9%          |            0.26 | -22.6%         | 28.9x             |
| best residual momentum |       6 | 2/6                      | 31.7%         |            0.9  | -34.9%         | 13.3x             |
| nearest 52w high       |       6 | 1/6                      | 31.6%         |            0.88 | -35.8%         | 10.7x             |
| strongest last month   |       6 | 3/6                      | 40.9%         |            1.02 | -39.3%         | 8.5x              |

### every cell

| Window   |   Pool | Rule                   | CAGR   | Vol   |   Sharpe | MaxDD   |   Calmar | Turnover   | Exposure   |
|:---------|-------:|:-----------------------|:-------|:------|---------:|:--------|---------:|:-----------|:-----------|
| default  |     15 | top 6 by momentum      | 48.6%  | 50.2% |     0.96 | -39.3%  |     1.24 | 9.0x       | 100.0%     |
| default  |     15 | ranks 7+ by momentum   | 35.2%  | 40.6% |     0.85 | -30.6%  |     1.15 | 17.2x      | 100.0%     |
| default  |     15 | lowest 60d vol         | 37.5%  | 32.5% |     1.02 | -21.9%  |     1.71 | 30.1x      | 100.0%     |
| default  |     15 | lowest beta to QQQ     | 24.8%  | 31.3% |     0.73 | -22.2%  |     1.11 | 34.2x      | 100.0%     |
| default  |     15 | best residual momentum | 57.3%  | 47.6% |     1.1  | -39.2%  |     1.46 | 15.4x      | 100.0%     |
| default  |     15 | nearest 52w high       | 42.8%  | 44.4% |     0.93 | -35.8%  |     1.2  | 11.9x      | 100.0%     |
| default  |     15 | strongest last month   | 51.9%  | 49.4% |     1.01 | -39.3%  |     1.32 | 10.9x      | 100.0%     |
| default  |     20 | top 6 by momentum      | 31.3%  | 49.1% |     0.72 | -39.7%  |     0.79 | 7.4x       | 100.0%     |
| default  |     20 | ranks 7+ by momentum   | 46.9%  | 45.4% |     0.98 | -33.6%  |     1.39 | 9.6x       | 100.0%     |
| default  |     20 | lowest 60d vol         | 9.6%   | 26.3% |     0.32 | -19.2%  |     0.5  | 30.3x      | 100.0%     |
| default  |     20 | lowest beta to QQQ     | 8.9%   | 26.4% |     0.3  | -22.9%  |     0.39 | 27.5x      | 100.0%     |
| default  |     20 | best residual momentum | 53.1%  | 42.6% |     1.12 | -34.9%  |     1.52 | 11.3x      | 100.0%     |
| default  |     20 | nearest 52w high       | 40.4%  | 44.6% |     0.89 | -37.6%  |     1.07 | 9.4x       | 100.0%     |
| default  |     20 | strongest last month   | 44.5%  | 49.8% |     0.91 | -41.1%  |     1.08 | 7.2x       | 100.0%     |
| default  |     30 | top 6 by momentum      | 57.2%  | 47.5% |     1.1  | -40.2%  |     1.42 | 6.1x       | 100.0%     |
| default  |     30 | ranks 7+ by momentum   | 12.2%  | 35.8% |     0.39 | -30.1%  |     0.41 | 7.0x       | 100.0%     |
| default  |     30 | lowest 60d vol         | 8.7%   | 18.3% |     0.33 | -16.1%  |     0.54 | 22.9x      | 100.0%     |
| default  |     30 | lowest beta to QQQ     | 3.2%   | 20.0% |     0.06 | -21.9%  |     0.15 | 23.8x      | 100.0%     |
| default  |     30 | best residual momentum | 36.7%  | 40.6% |     0.87 | -32.8%  |     1.12 | 6.8x       | 100.0%     |
| default  |     30 | nearest 52w high       | 33.9%  | 36.9% |     0.87 | -34.7%  |     0.98 | 8.6x       | 100.0%     |
| default  |     30 | strongest last month   | 35.3%  | 45.2% |     0.8  | -35.6%  |     0.99 | 5.7x       | 100.0%     |
| extended |     15 | top 6 by momentum      | 41.8%  | 40.7% |     0.97 | -39.3%  |     1.07 | 10.7x      | 100.0%     |
| extended |     15 | ranks 7+ by momentum   | 12.1%  | 33.8% |     0.4  | -30.6%  |     0.4  | 20.2x      | 100.0%     |
| extended |     15 | lowest 60d vol         | 15.5%  | 27.8% |     0.53 | -21.9%  |     0.71 | 37.8x      | 100.0%     |
| extended |     15 | lowest beta to QQQ     | 6.1%   | 27.5% |     0.22 | -23.0%  |     0.27 | 37.0x      | 100.0%     |
| extended |     15 | best residual momentum | 26.7%  | 38.0% |     0.72 | -39.2%  |     0.68 | 21.5x      | 100.0%     |
| extended |     15 | nearest 52w high       | 23.2%  | 35.3% |     0.67 | -35.8%  |     0.65 | 17.4x      | 100.0%     |
| extended |     15 | strongest last month   | 39.8%  | 39.8% |     0.95 | -39.3%  |     1.01 | 12.4x      | 100.0%     |
| extended |     20 | top 6 by momentum      | 40.3%  | 40.2% |     0.95 | -39.7%  |     1.02 | 8.9x       | 100.0%     |
| extended |     20 | ranks 7+ by momentum   | 22.4%  | 36.7% |     0.63 | -33.6%  |     0.67 | 13.3x      | 100.0%     |
| extended |     20 | lowest 60d vol         | 2.4%   | 23.6% |     0.06 | -23.9%  |     0.1  | 33.0x      | 100.0%     |
| extended |     20 | lowest beta to QQQ     | 4.3%   | 24.3% |     0.15 | -23.5%  |     0.18 | 30.4x      | 100.0%     |
| extended |     20 | best residual momentum | 23.0%  | 35.3% |     0.66 | -34.9%  |     0.66 | 15.2x      | 100.0%     |
| extended |     20 | nearest 52w high       | 29.4%  | 35.4% |     0.8  | -37.6%  |     0.78 | 12.9x      | 100.0%     |
| extended |     20 | strongest last month   | 41.9%  | 39.5% |     0.99 | -41.1%  |     1.02 | 9.8x       | 100.0%     |
| extended |     30 | top 6 by momentum      | 41.6%  | 39.0% |     1    | -40.2%  |     1.04 | 6.9x       | 100.0%     |
| extended |     30 | ranks 7+ by momentum   | 9.3%   | 30.9% |     0.33 | -34.5%  |     0.27 | 9.3x       | 100.0%     |
| extended |     30 | lowest 60d vol         | 8.5%   | 18.1% |     0.34 | -21.0%  |     0.41 | 21.8x      | 100.0%     |
| extended |     30 | lowest beta to QQQ     | 5.7%   | 18.6% |     0.2  | -21.9%  |     0.26 | 23.3x      | 100.0%     |
| extended |     30 | best residual momentum | 19.4%  | 32.7% |     0.59 | -34.9%  |     0.55 | 11.4x      | 100.0%     |
| extended |     30 | nearest 52w high       | 15.2%  | 29.5% |     0.5  | -29.1%  |     0.52 | 9.5x       | 100.0%     |
| extended |     30 | strongest last month   | 30.4%  | 37.5% |     0.8  | -38.0%  |     0.8  | 7.1x       | 100.0%     |
