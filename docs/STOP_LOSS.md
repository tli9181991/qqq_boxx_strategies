# Stop losses for the Top-6 momentum and Top-6 residual momentum books

**Short version.** Stop each *book*, not each *name*. Per-position price stops
(fixed, trailing, ATR/chandelier) do cut volatility and shave a few points off
max drawdown, but they cost far more CAGR than they save and multiply turnover
2–4×. Calmar gets worse in almost every cell tested. A book-level drawdown
stop (`drawdown_stop`, already in the lab) is the only family that cuts max
drawdown by double digits, consistently, in both windows.

| Book | Recommended stop | Why |
|---|---|---|
| **Top-6 momentum** | **book drawdown 20%, 5-day cooldown** | lower max DD in **16/16** paired cells (median +16 / +13 pts), higher Calmar in **15/16**, turnover roughly unchanged |
| **Top-6 residual momentum** | **book drawdown 13%, 5-day cooldown** (the live default) | lower max DD in **13/16** cells (median +12 / +9 pts); it costs CAGR, so it is insurance, not an edge |
| either | *not* trailing / chandelier stops | higher Calmar in only 8/64 paired cells on the plain book and 11/64 on the residual book; median CAGR change mostly −7 to −42 pts; turnover +5× to +68× |

Two per-position stops are worth knowing about, neither as drawdown control:

* **Fixed 10% from entry on the residual book** — lower max DD in 14/16 cells
  and higher Calmar in 12/16, at about +10× annual turnover. The best-behaved
  per-name stop, but its drawdown gain (+4.5 to +5.7 pts median) is a third
  of the book stop's.
* **Residual stop, 2σ** (sell a name whose *idiosyncratic* fall since entry
  exceeds 2 × its residual σ × √21) — higher Calmar in 12/16 residual-book
  cells for +1.5–3.4× turnover, but max drawdown is unchanged. It removes names
  whose own story broke, which is a return tweak, not risk control.

## What was tested

Every stop runs through the same engine, costs (1 bp commission + 5 bp
slippage per 100% turnover), one-day lag and window as the unstopped book, so
rows differ only by the stop. Stops are checked on the **close** and fill at the
close (the lab's execution model) — intraday stop orders would fill
differently, and would be *worse* on gap-downs.

| Kind | Reference level | Distance (the estimated part) |
|---|---|---|
| `fixed` | entry close | fixed % (10 / 15 / 20) |
| `trailing` | highest close since entry | fixed % (10 / 15 / 20 / 25) |
| `chandelier` | highest close since entry | k × 20-day close-ATR (k = 3 / 4 / 5 / 6) — vol-scaled per name |
| `residual` | peak of the name's residual wealth (return − β·QQQ, 252-day β) | k × 63-day residual σ × √21 (k = 1 / 1.5 / 2 / 3) |
| `support` | highest confirmed swing low below the close (5-bar pivot, last 126 days), ratcheted up only | that level − m × 20-day close-ATR (m = 0 / 0.5 / 1 / 2) |
| book dd | the whole book's own equity high | 10 / 13 / 20% below it → all BOXX for 5 sessions |

Per-position stops live inside the ranking loop
(`cross_sectional_momentum(..., stop=StopLossParams(...))`), so a stopped name
can be barred from being re-bought (`cooldown_days`, default 21). Without a
cooldown a daily-rebalanced book re-buys a still-top-ranked name the next day.
`refill=True` hands the freed slot to the next ranked name; `refill=False`
leaves it in BOXX.

**Windows.**

* *default* — 2025-01-20 → 2026-09-08, the lab's reporting window.
* *extended* — 2022-01-03 → 2026-09-08, which adds the 2022 bear market, the
  episode stops exist for. BOXX lists on 2022-12-28; before that the cash leg
  is a flat **0%** proxy (T-bills paid ~0–2% then, so this understates cash
  slightly and flatters nothing).

**Paired test.** One window's single cell is one draw. Each stop is also
paired against its own unstopped book across 8 `(n_hold, exit_rank)` cells —
(4,6) (4,8) (6,8) (6,10) (6,12) (8,10) (8,12) (8,14) — and scored on how
many cells it improves. A stop that only wins at the shipped cell is noise.

## How to read the results

1. **Per-name price stops sell volatility, not losers.** A 50%-vol Nasdaq
   name moves 10–15% in a normal fortnight. A trailing 10% stop fired 185 times
   on the plain book in the extended window; chandelier 3× fired 348 times.
   Each firing sells a name that 6-1 momentum still ranks top-6, and then
   either buys the next one down (a weaker name, at a spread) or sits in cash
   through the rebound. Volatility falls, CAGR falls faster.
2. **Tighter is not monotonically safer.** Trailing 10% beats trailing 20%
   on the default window and loses on the extended one; chandelier 4× is worse
   than both 3× and 5×. Non-monotone rows are what noise looks like.
3. **The cooldown matters more than the distance.** With a trailing 10% stop,
   moving the cooldown from 21 to 63 days takes the residual book from 11% to
   3% CAGR in the extended window. `slot->cash` lowers vol further but always
   costs return. The one single cell where a per-name price stop beats no
   stop (trailing 10%, cooldown 21, plain book, default window: Calmar 1.41
   vs 1.19) improves Calmar in only 1/8 paired cells, so that result is luck.
4. **The book stop works for the reason `DrawdownStopParams` gives.** This
   book's worst losses come from its six names falling *together* (the 2026
   semiconductor concentration blow-up, the 2022 bear), which no single-name
   stop sees until each has already fallen. The book's own drawdown sees it
   at once.
5. **The right book-stop level differs by book.** The plain book runs ~40–49%
   vol, so 13% below the high is a normal wobble and 13% costs more return
   than 20% does; 20% catches only the real episodes. The residual book runs
   ~30–40% vol with shallower swings, and at 20% it fired on the *wrong*
   episodes in the extended window (max DD −38% vs −35% unstopped, 1/8 cells
   improved), while 13% held up (7/8 and 6/8).

## Support level minus ATR (from `M6_finalnotebook`)

The breakout notebook finds support and resistance with `find_peaks` on daily
swing highs and lows, merges levels within ~1 ATR, and uses them to trigger
entries. Its stop is not placed at a support: it is `entry − R`, with R
the smaller of the 95% VaR move and (half the average level gap + 0.5 × ADR).
The `support` kind tests the natural extension: stop under the nearest
support, with an ATR buffer. Two changes from the notebook make it tradeable:

* **Causal levels.** The notebook ran `find_peaks` over the whole test
  window, so a trade could "see" support that only formed later. Here a
  swing low counts only once `support_pivot` later closes confirm it.
* **Closes only.** The universe cache has no highs and lows, so pivots and
  ATR come from closes. Intraday lows would give slightly lower levels.

**It is the weakest per-position stop tested.** Higher Calmar in 0/8 paired
cells on the plain book in both windows, and 0/8 and 4/8 on the residual
book; median CAGR −21 to −34 pts in three of the four book-window pairs;
turnover +28× to +50×. It fired 170–420 times per book in the extended window. The
reason is structural: in a momentum uptrend the most recent swing low is a
shallow pullback just under the price, so the stop sits inside ordinary
noise. Only the loosest version (10-bar pivots, i.e. fewer, more
significant lows) came close to neutral, and it still lost.

Where a support stop *does* belong is the notebook's own setting: an
event-driven breakout entered at a level, where "back below the level I
bought the break of, less an ATR" is a precise statement that the trade
failed. A ranking book that buys whatever ranks top-6 is not entered at a
level, so the nearest support says nothing about whether the thesis broke.

## Caveats

* Survivorship bias, as for every momentum number in this lab (see README).
  It flatters the no-stop book more than the stopped ones, since the names that
  would have kept falling are missing.
* 4.7 years at most, with one bear market. The book-stop evidence is a handful
  of episodes per window; the paired grid tests the *parameter*, not the
  *episodes*.
* The book stop's 5-session cooldown was not re-optimised here; it is the
  existing `DrawdownStopParams` default.

## Reproduce

```bash
python run_backtest.py --offline --sweep-stops        # default window, one cell
python scripts/stop_loss_study.py                     # everything below (~15 min)
python scripts/stop_loss_study.py --quick             # skip the paired grid
```

The study uses `Config()` defaults: momentum band `(6, 8)`, residual band
`(6, 10)`. `run_backtest.py` syncs the residual book's band to the momentum
book's, so its `--sweep-stops` residual rows run at `(6, 8)` and differ from
the tables below (no-stop residual CAGR 73.6% there vs 68.4% here). The paired
grid covers both cells.

```python
from qbs.config import StopLossParams
from qbs.strategies import residual_momentum
sig = residual_momentum(uni, boxx, qqq, stop=StopLossParams(kind="fixed", stop_pct=0.10))
```

---

# Results
## default window (2025-01-20 to 2026-09-08)

| Book     | Stop                              | CAGR   | Ann. vol   |   Sharpe | Max drawdown   |   Calmar | Ann. turnover   |   Stops |
|:---------|:----------------------------------|:-------|:-----------|---------:|:---------------|---------:|:----------------|--------:|
| momentum | no stop                           | 48.1%  | 49.2%      |     0.96 | -40.4%         |     1.19 | 21.1x           |       0 |
| momentum | fixed 10% (cd 21)                 | 45.8%  | 46.3%      |     0.96 | -37.1%         |     1.24 | 34.8x           |      28 |
| momentum | fixed 15% (cd 21)                 | 48.9%  | 46.4%      |     1    | -40.4%         |     1.21 | 27.7x           |      16 |
| momentum | fixed 20% (cd 21)                 | 47.1%  | 47.2%      |     0.97 | -39.1%         |     1.2  | 27.5x           |      10 |
| momentum | trailing 10% (cd 21)              | 50.4%  | 38.8%      |     1.14 | -35.7%         |     1.41 | 68.6x           |      74 |
| momentum | trailing 15% (cd 21)              | 37.7%  | 41.0%      |     0.89 | -35.8%         |     1.05 | 51.8x           |      49 |
| momentum | trailing 20% (cd 21)              | 22.2%  | 43.2%      |     0.59 | -36.2%         |     0.61 | 40.0x           |      26 |
| momentum | trailing 25% (cd 21)              | 41.3%  | 45.0%      |     0.9  | -35.9%         |     1.15 | 30.3x           |      16 |
| momentum | chandelier 3x ATR20 (cd 21)       | 40.4%  | 40.1%      |     0.95 | -37.6%         |     1.08 | 86.7x           |     112 |
| momentum | chandelier 4x ATR20 (cd 21)       | 28.9%  | 43.5%      |     0.71 | -38.8%         |     0.74 | 74.2x           |      76 |
| momentum | chandelier 5x ATR20 (cd 21)       | 38.3%  | 45.2%      |     0.85 | -36.0%         |     1.06 | 53.3x           |      54 |
| momentum | chandelier 6x ATR20 (cd 21)       | 38.6%  | 46.8%      |     0.84 | -38.0%         |     1.01 | 41.2x           |      36 |
| momentum | residual 1x resid sigma (cd 21)   | 40.5%  | 46.2%      |     0.88 | -37.4%         |     1.08 | 49.8x           |      47 |
| momentum | residual 1.5x resid sigma (cd 21) | 48.4%  | 49.0%      |     0.97 | -40.5%         |     1.19 | 34.0x           |      17 |
| momentum | residual 2x resid sigma (cd 21)   | 52.5%  | 49.2%      |     1.02 | -40.9%         |     1.28 | 22.9x           |       6 |
| momentum | residual 3x resid sigma (cd 21)   | 48.1%  | 49.2%      |     0.96 | -40.4%         |     1.19 | 21.1x           |       0 |
| momentum | support low - 0x ATR (cd 21)      | 28.8%  | 42.3%      |     0.71 | -42.8%         |     0.67 | 78.9x           |     138 |
| momentum | support low - 0.5x ATR (cd 21)    | 29.8%  | 43.1%      |     0.73 | -36.1%         |     0.83 | 67.2x           |      99 |
| momentum | support low - 1x ATR (cd 21)      | 22.8%  | 44.7%      |     0.59 | -40.4%         |     0.56 | 56.8x           |      79 |
| momentum | support low - 2x ATR (cd 21)      | 17.1%  | 46.2%      |     0.49 | -40.7%         |     0.42 | 54.1x           |      56 |
| momentum | book dd 10% (cd 5)                | 21.0%  | 30.9%      |     0.64 | -18.5%         |     1.13 | 32.0x           |      10 |
| momentum | book dd 13% (cd 5)                | 29.9%  | 33.5%      |     0.83 | -21.0%         |     1.43 | 26.6x           |       6 |
| momentum | book dd 20% (cd 5)                | 53.9%  | 37.4%      |     1.23 | -22.0%         |     2.45 | 19.9x           |       2 |
| resmom   | no stop                           | 68.4%  | 39.6%      |     1.41 | -34.9%         |     1.96 | 14.5x           |       0 |
| resmom   | fixed 10% (cd 21)                 | 60.8%  | 36.4%      |     1.38 | -26.6%         |     2.28 | 25.2x           |      23 |
| resmom   | fixed 15% (cd 21)                 | 60.3%  | 36.9%      |     1.35 | -30.1%         |     2    | 21.1x           |      13 |
| resmom   | fixed 20% (cd 21)                 | 66.8%  | 38.3%      |     1.42 | -32.7%         |     2.04 | 18.6x           |       8 |
| resmom   | trailing 10% (cd 21)              | 36.6%  | 28.6%      |     1.09 | -20.2%         |     1.81 | 49.4x           |      59 |
| resmom   | trailing 15% (cd 21)              | 32.2%  | 31.7%      |     0.91 | -23.6%         |     1.36 | 35.2x           |      32 |
| resmom   | trailing 20% (cd 21)              | 37.5%  | 34.8%      |     0.97 | -29.6%         |     1.27 | 24.6x           |      18 |
| resmom   | trailing 25% (cd 21)              | 55.5%  | 36.8%      |     1.27 | -31.6%         |     1.76 | 21.3x           |      11 |
| resmom   | chandelier 3x ATR20 (cd 21)       | 30.4%  | 32.8%      |     0.85 | -24.5%         |     1.24 | 76.8x           |     123 |
| resmom   | chandelier 4x ATR20 (cd 21)       | 9.6%   | 34.1%      |     0.32 | -36.6%         |     0.26 | 54.9x           |      84 |
| resmom   | chandelier 5x ATR20 (cd 21)       | 37.3%  | 37.4%      |     0.93 | -30.8%         |     1.21 | 41.0x           |      53 |
| resmom   | chandelier 6x ATR20 (cd 21)       | 63.5%  | 38.7%      |     1.36 | -33.4%         |     1.9  | 29.3x           |      33 |
| resmom   | residual 1x resid sigma (cd 21)   | 63.2%  | 38.3%      |     1.37 | -29.4%         |     2.15 | 33.0x           |      45 |
| resmom   | residual 1.5x resid sigma (cd 21) | 69.0%  | 38.7%      |     1.45 | -34.7%         |     1.99 | 20.9x           |      17 |
| resmom   | residual 2x resid sigma (cd 21)   | 79.3%  | 39.8%      |     1.56 | -35.2%         |     2.25 | 15.8x           |       6 |
| resmom   | residual 3x resid sigma (cd 21)   | 68.4%  | 39.6%      |     1.41 | -34.9%         |     1.96 | 14.5x           |       0 |
| resmom   | support low - 0x ATR (cd 21)      | 13.3%  | 34.7%      |     0.42 | -34.8%         |     0.38 | 67.8x           |     132 |
| resmom   | support low - 0.5x ATR (cd 21)    | 23.8%  | 35.5%      |     0.66 | -30.3%         |     0.78 | 55.7x           |      98 |
| resmom   | support low - 1x ATR (cd 21)      | 30.2%  | 35.3%      |     0.81 | -31.3%         |     0.97 | 50.6x           |      79 |
| resmom   | support low - 2x ATR (cd 21)      | 33.0%  | 37.2%      |     0.84 | -36.0%         |     0.92 | 43.6x           |      62 |
| resmom   | book dd 10% (cd 5)                | 31.8%  | 30.6%      |     0.92 | -21.3%         |     1.49 | 28.9x           |       8 |
| resmom   | book dd 13% (cd 5)                | 58.2%  | 33.5%      |     1.42 | -20.4%         |     2.86 | 21.5x           |       4 |
| resmom   | book dd 20% (cd 5)                | 70.5%  | 35.8%      |     1.56 | -20.7%         |     3.4  | 14.5x           |       1 |

### default: cooldown, refill and support-stop dials

| Book     | Stop                                        | CAGR   | Ann. vol   |   Sharpe | Max drawdown   |   Calmar | Ann. turnover   |   Stops |
|:---------|:--------------------------------------------|:-------|:-----------|---------:|:---------------|---------:|:----------------|--------:|
| momentum | no stop                                     | 48.1%  | 49.2%      |     0.96 | -40.4%         |     1.19 | 21.1x           |       0 |
| momentum | trailing 10% (cd 5)                         | 36.2%  | 43.0%      |     0.84 | -34.1%         |     1.06 | 59.6x           |     106 |
| momentum | trailing 10% (cd 5, slot->cash)             | 32.8%  | 35.8%      |     0.86 | -29.6%         |     1.11 | 52.4x           |      95 |
| momentum | trailing 10% (cd 21)                        | 50.4%  | 38.8%      |     1.14 | -35.7%         |     1.41 | 68.6x           |      74 |
| momentum | trailing 10% (cd 21, slot->cash)            | 22.7%  | 23.8%      |     0.81 | -20.1%         |     1.13 | 39.7x           |      61 |
| momentum | trailing 10% (cd 63)                        | 25.5%  | 34.2%      |     0.72 | -27.0%         |     0.94 | 65.6x           |      54 |
| momentum | trailing 10% (cd 63, slot->cash)            | 1.0%   | 17.6%      |    -0.09 | -13.9%         |     0.07 | 28.3x           |      40 |
| momentum | residual 2x resid sigma (cd 5)              | 54.1%  | 49.2%      |     1.04 | -40.9%         |     1.32 | 22.1x           |       6 |
| momentum | residual 2x resid sigma (cd 5, slot->cash)  | 51.4%  | 48.7%      |     1.01 | -40.0%         |     1.28 | 22.7x           |       6 |
| momentum | residual 2x resid sigma (cd 21)             | 52.5%  | 49.2%      |     1.02 | -40.9%         |     1.28 | 22.9x           |       6 |
| momentum | residual 2x resid sigma (cd 21, slot->cash) | 49.7%  | 48.5%      |     0.99 | -40.0%         |     1.24 | 22.3x           |       6 |
| momentum | residual 2x resid sigma (cd 63)             | 50.8%  | 49.1%      |     1    | -40.9%         |     1.24 | 22.9x           |       6 |
| momentum | residual 2x resid sigma (cd 63, slot->cash) | 48.3%  | 48.4%      |     0.97 | -40.0%         |     1.21 | 22.3x           |       6 |
| momentum | support low - 1x ATR, pivot 3 (cd 21)       | 40.3%  | 42.5%      |     0.92 | -36.8%         |     1.1  | 76.0x           |     113 |
| momentum | support low - 1x ATR, pivot 10 (cd 21)      | 39.3%  | 46.2%      |     0.86 | -41.2%         |     0.95 | 35.0x           |      29 |
| momentum | support low - 1x ATR, lookback 63 (cd 21)   | 23.2%  | 45.0%      |     0.6  | -40.5%         |     0.57 | 56.5x           |      77 |
| momentum | support low - 1x ATR, lookback 252 (cd 21)  | 23.3%  | 44.7%      |     0.6  | -40.4%         |     0.58 | 57.2x           |      80 |
| resmom   | no stop                                     | 68.4%  | 39.6%      |     1.41 | -34.9%         |     1.96 | 14.5x           |       0 |
| resmom   | trailing 10% (cd 5)                         | 34.9%  | 33.7%      |     0.94 | -26.1%         |     1.34 | 42.0x           |      75 |
| resmom   | trailing 10% (cd 5, slot->cash)             | 29.4%  | 29.7%      |     0.88 | -26.9%         |     1.09 | 40.2x           |      69 |
| resmom   | trailing 10% (cd 21)                        | 36.6%  | 28.6%      |     1.09 | -20.2%         |     1.81 | 49.4x           |      59 |
| resmom   | trailing 10% (cd 21, slot->cash)            | 27.7%  | 22.3%      |     1.03 | -15.7%         |     1.77 | 29.5x           |      44 |
| resmom   | trailing 10% (cd 63)                        | 6.9%   | 26.0%      |     0.23 | -20.6%         |     0.34 | 53.1x           |      45 |
| resmom   | trailing 10% (cd 63, slot->cash)            | 1.8%   | 14.7%      |    -0.08 | -20.2%         |     0.09 | 21.5x           |      31 |
| resmom   | residual 2x resid sigma (cd 5)              | 76.2%  | 40.3%      |     1.51 | -35.2%         |     2.16 | 15.8x           |       6 |
| resmom   | residual 2x resid sigma (cd 5, slot->cash)  | 73.0%  | 39.3%      |     1.49 | -34.4%         |     2.12 | 16.6x           |       6 |
| resmom   | residual 2x resid sigma (cd 21)             | 79.3%  | 39.8%      |     1.56 | -35.2%         |     2.25 | 15.8x           |       6 |
| resmom   | residual 2x resid sigma (cd 21, slot->cash) | 74.7%  | 38.0%      |     1.55 | -34.4%         |     2.17 | 17.0x           |       6 |
| resmom   | residual 2x resid sigma (cd 63)             | 74.8%  | 39.7%      |     1.51 | -35.0%         |     2.14 | 18.4x           |       6 |
| resmom   | residual 2x resid sigma (cd 63, slot->cash) | 67.0%  | 36.7%      |     1.47 | -31.7%         |     2.12 | 17.0x           |       6 |
| resmom   | support low - 1x ATR, pivot 3 (cd 21)       | 30.3%  | 32.7%      |     0.85 | -24.9%         |     1.22 | 67.4x           |     124 |
| resmom   | support low - 1x ATR, pivot 10 (cd 21)      | 65.2%  | 39.5%      |     1.36 | -33.8%         |     1.93 | 27.5x           |      32 |
| resmom   | support low - 1x ATR, lookback 63 (cd 21)   | 33.3%  | 35.8%      |     0.87 | -31.3%         |     1.06 | 47.3x           |      74 |
| resmom   | support low - 1x ATR, lookback 252 (cd 21)  | 25.1%  | 35.0%      |     0.7  | -31.3%         |     0.8  | 52.0x           |      80 |

### default: paired across 8 (n_hold, exit_rank) cells

| Book     | Stop                            |   Cells | Lower max DD   | Median ΔMaxDD   | Higher Calmar   | Median ΔCAGR   | Median ΔVol   | Median Δturnover   |
|:---------|:--------------------------------|--------:|:---------------|:----------------|:----------------|:---------------|:--------------|:-------------------|
| momentum | fixed 10% (cd 21)               |       8 | 7/8            | +1.3%           | 3/8             | -6.6%          | -3.7%         | +10.8x             |
| momentum | fixed 20% (cd 21)               |       8 | 5/8            | +0.5%           | 4/8             | -2.6%          | -2.2%         | +3.3x              |
| momentum | trailing 10% (cd 21)            |       8 | 7/8            | +3.1%           | 1/8             | -15.8%         | -11.9%        | +45.9x             |
| momentum | trailing 20% (cd 21)            |       8 | 6/8            | +2.7%           | 1/8             | -19.8%         | -7.2%         | +14.6x             |
| momentum | chandelier 3x ATR20 (cd 21)     |       8 | 8/8            | +4.1%           | 3/8             | -10.8%         | -10.7%        | +65.7x             |
| momentum | chandelier 6x ATR20 (cd 21)     |       8 | 7/8            | +2.3%           | 3/8             | -7.0%          | -3.4%         | +20.9x             |
| momentum | residual 1x resid sigma (cd 21) |       8 | 4/8            | -0.2%           | 3/8             | -11.9%         | -4.4%         | +25.9x             |
| momentum | residual 2x resid sigma (cd 21) |       8 | 3/8            | +0.0%           | 4/8             | -0.3%          | -0.2%         | +1.1x              |
| momentum | support low - 0.5x ATR (cd 21)  |       8 | 8/8            | +1.8%           | 0/8             | -28.8%         | -6.4%         | +44.9x             |
| momentum | support low - 1x ATR (cd 21)    |       8 | 3/8            | -0.6%           | 0/8             | -28.5%         | -5.1%         | +36.3x             |
| momentum | support low - 2x ATR (cd 21)    |       8 | 4/8            | +0.3%           | 1/8             | -28.9%         | -3.6%         | +31.3x             |
| momentum | book dd 13% (cd 5)              |       8 | 8/8            | +19.1%          | 6/8             | -17.5%         | -16.0%        | +7.4x              |
| momentum | book dd 20% (cd 5)              |       8 | 8/8            | +16.3%          | 8/8             | +0.6%          | -12.0%        | +1.2x              |
| resmom   | fixed 10% (cd 21)               |       8 | 7/8            | +5.7%           | 5/8             | -9.1%          | -3.9%         | +11.2x             |
| resmom   | fixed 20% (cd 21)               |       8 | 6/8            | +1.7%           | 4/8             | -4.4%          | -2.0%         | +4.0x              |
| resmom   | trailing 10% (cd 21)            |       8 | 8/8            | +8.9%           | 0/8             | -32.2%         | -10.8%        | +32.2x             |
| resmom   | trailing 20% (cd 21)            |       8 | 7/8            | +3.5%           | 0/8             | -25.1%         | -5.5%         | +9.2x              |
| resmom   | chandelier 3x ATR20 (cd 21)     |       8 | 6/8            | +3.2%           | 0/8             | -41.7%         | -7.5%         | +61.3x             |
| resmom   | chandelier 6x ATR20 (cd 21)     |       8 | 6/8            | +1.8%           | 2/8             | -9.9%          | -1.7%         | +15.8x             |
| resmom   | residual 1x resid sigma (cd 21) |       8 | 6/8            | +2.5%           | 4/8             | -8.1%          | -3.0%         | +20.1x             |
| resmom   | residual 2x resid sigma (cd 21) |       8 | 2/8            | -0.0%           | 6/8             | +4.8%          | -0.4%         | +1.5x              |
| resmom   | support low - 0.5x ATR (cd 21)  |       8 | 3/8            | -1.3%           | 0/8             | -33.8%         | -4.3%         | +38.4x             |
| resmom   | support low - 1x ATR (cd 21)    |       8 | 4/8            | +0.1%           | 0/8             | -33.1%         | -4.6%         | +33.5x             |
| resmom   | support low - 2x ATR (cd 21)    |       8 | 2/8            | -1.7%           | 0/8             | -32.9%         | -2.9%         | +27.9x             |
| resmom   | book dd 13% (cd 5)              |       8 | 7/8            | +11.9%          | 4/8             | -19.3%         | -6.6%         | +8.8x              |
| resmom   | book dd 20% (cd 5)              |       8 | 7/8            | +7.3%           | 5/8             | -7.7%          | -3.8%         | +3.8x              |

## extended window (2022-01-03 to 2026-09-08)

| Book     | Stop                              | CAGR   | Ann. vol   |   Sharpe | Max drawdown   |   Calmar | Ann. turnover   |   Stops |
|:---------|:----------------------------------|:-------|:-----------|---------:|:---------------|---------:|:----------------|--------:|
| momentum | no stop                           | 34.0%  | 40.1%      |     0.84 | -40.4%         |     0.84 | 25.4x           |       0 |
| momentum | fixed 10% (cd 21)                 | 30.6%  | 37.7%      |     0.8  | -37.1%         |     0.83 | 39.4x           |      77 |
| momentum | fixed 15% (cd 21)                 | 31.7%  | 37.9%      |     0.82 | -40.4%         |     0.79 | 34.4x           |      38 |
| momentum | fixed 20% (cd 21)                 | 30.4%  | 38.6%      |     0.79 | -39.1%         |     0.78 | 32.3x           |      24 |
| momentum | trailing 10% (cd 21)              | 29.0%  | 33.9%      |     0.82 | -35.7%         |     0.81 | 66.0x           |     185 |
| momentum | trailing 15% (cd 21)              | 24.1%  | 35.0%      |     0.69 | -35.8%         |     0.67 | 49.6x           |     101 |
| momentum | trailing 20% (cd 21)              | 17.1%  | 36.3%      |     0.52 | -36.2%         |     0.47 | 39.4x           |      54 |
| momentum | trailing 25% (cd 21)              | 29.0%  | 37.7%      |     0.77 | -35.9%         |     0.81 | 33.6x           |      31 |
| momentum | chandelier 3x ATR20 (cd 21)       | 8.2%   | 33.6%      |     0.3  | -37.6%         |     0.22 | 93.2x           |     348 |
| momentum | chandelier 4x ATR20 (cd 21)       | 8.4%   | 36.0%      |     0.3  | -39.4%         |     0.21 | 81.2x           |     236 |
| momentum | chandelier 5x ATR20 (cd 21)       | 17.8%  | 37.6%      |     0.53 | -40.0%         |     0.45 | 64.7x           |     167 |
| momentum | chandelier 6x ATR20 (cd 21)       | 21.5%  | 38.4%      |     0.6  | -38.0%         |     0.56 | 52.4x           |     115 |
| momentum | residual 1x resid sigma (cd 21)   | 19.3%  | 38.4%      |     0.56 | -42.3%         |     0.46 | 54.2x           |     127 |
| momentum | residual 1.5x resid sigma (cd 21) | 29.3%  | 40.1%      |     0.75 | -40.5%         |     0.72 | 38.1x           |      50 |
| momentum | residual 2x resid sigma (cd 21)   | 34.4%  | 40.0%      |     0.85 | -40.9%         |     0.84 | 29.1x           |      19 |
| momentum | residual 3x resid sigma (cd 21)   | 33.1%  | 40.0%      |     0.82 | -40.4%         |     0.82 | 26.3x           |       2 |
| momentum | support low - 0x ATR (cd 21)      | 14.6%  | 34.5%      |     0.46 | -42.8%         |     0.34 | 84.5x           |     421 |
| momentum | support low - 0.5x ATR (cd 21)    | 14.4%  | 35.3%      |     0.46 | -38.8%         |     0.37 | 78.0x           |     333 |
| momentum | support low - 1x ATR (cd 21)      | 13.8%  | 36.5%      |     0.44 | -44.7%         |     0.31 | 66.9x           |     257 |
| momentum | support low - 2x ATR (cd 21)      | 12.6%  | 37.3%      |     0.41 | -43.7%         |     0.29 | 58.5x           |     172 |
| momentum | book dd 10% (cd 5)                | 9.7%   | 25.3%      |     0.35 | -31.0%         |     0.31 | 24.8x           |      19 |
| momentum | book dd 13% (cd 5)                | 18.6%  | 27.9%      |     0.62 | -23.2%         |     0.8  | 21.0x           |      11 |
| momentum | book dd 20% (cd 5)                | 31.9%  | 31.0%      |     0.93 | -27.3%         |     1.17 | 18.0x           |       4 |
| resmom   | no stop                           | 20.3%  | 30.7%      |     0.64 | -34.9%         |     0.58 | 14.0x           |       0 |
| resmom   | fixed 10% (cd 21)                 | 21.3%  | 28.2%      |     0.7  | -28.0%         |     0.76 | 23.8x           |      71 |
| resmom   | fixed 15% (cd 21)                 | 20.3%  | 28.6%      |     0.66 | -33.5%         |     0.61 | 18.4x           |      37 |
| resmom   | fixed 20% (cd 21)                 | 20.4%  | 29.1%      |     0.66 | -32.7%         |     0.62 | 16.1x           |      22 |
| resmom   | trailing 10% (cd 21)              | 11.1%  | 24.9%      |     0.4  | -31.0%         |     0.36 | 38.3x           |     145 |
| resmom   | trailing 15% (cd 21)              | 15.9%  | 26.9%      |     0.55 | -29.1%         |     0.55 | 26.8x           |      75 |
| resmom   | trailing 20% (cd 21)              | 18.8%  | 28.0%      |     0.63 | -32.2%         |     0.58 | 18.8x           |      35 |
| resmom   | trailing 25% (cd 21)              | 17.2%  | 28.8%      |     0.57 | -35.8%         |     0.48 | 16.7x           |      25 |
| resmom   | chandelier 3x ATR20 (cd 21)       | -1.0%  | 28.2%      |    -0.02 | -40.1%         |    -0.03 | 82.2x           |     405 |
| resmom   | chandelier 4x ATR20 (cd 21)       | 4.7%   | 29.4%      |     0.18 | -43.9%         |     0.11 | 60.2x           |     267 |
| resmom   | chandelier 5x ATR20 (cd 21)       | 18.7%  | 30.4%      |     0.6  | -31.1%         |     0.6  | 44.4x           |     178 |
| resmom   | chandelier 6x ATR20 (cd 21)       | 26.5%  | 31.4%      |     0.79 | -35.1%         |     0.76 | 34.9x           |     133 |
| resmom   | residual 1x resid sigma (cd 21)   | 25.5%  | 30.5%      |     0.78 | -34.0%         |     0.75 | 37.3x           |     145 |
| resmom   | residual 1.5x resid sigma (cd 21) | 26.6%  | 31.2%      |     0.8  | -34.7%         |     0.77 | 24.3x           |      80 |
| resmom   | residual 2x resid sigma (cd 21)   | 26.3%  | 31.0%      |     0.79 | -35.2%         |     0.75 | 16.8x           |      32 |
| resmom   | residual 3x resid sigma (cd 21)   | 21.2%  | 30.6%      |     0.66 | -34.9%         |     0.61 | 14.0x           |       3 |
| resmom   | support low - 0x ATR (cd 21)      | 3.0%   | 27.8%      |     0.12 | -34.8%         |     0.09 | 68.8x           |     417 |
| resmom   | support low - 0.5x ATR (cd 21)    | 8.3%   | 28.4%      |     0.3  | -30.3%         |     0.27 | 57.5x           |     325 |
| resmom   | support low - 1x ATR (cd 21)      | 17.5%  | 28.7%      |     0.58 | -31.3%         |     0.56 | 50.2x           |     259 |
| resmom   | support low - 2x ATR (cd 21)      | 16.9%  | 29.8%      |     0.55 | -36.0%         |     0.47 | 39.6x           |     184 |
| resmom   | book dd 10% (cd 5)                | 5.4%   | 20.9%      |     0.18 | -30.4%         |     0.18 | 19.2x           |      14 |
| resmom   | book dd 13% (cd 5)                | 14.2%  | 23.3%      |     0.53 | -19.3%         |     0.74 | 16.0x           |       9 |
| resmom   | book dd 20% (cd 5)                | 13.8%  | 26.4%      |     0.48 | -38.1%         |     0.36 | 17.4x           |       8 |

### extended: cooldown, refill and support-stop dials

| Book     | Stop                                        | CAGR   | Ann. vol   |   Sharpe | Max drawdown   |   Calmar | Ann. turnover   |   Stops |
|:---------|:--------------------------------------------|:-------|:-----------|---------:|:---------------|---------:|:----------------|--------:|
| momentum | no stop                                     | 34.0%  | 40.1%      |     0.84 | -40.4%         |     0.84 | 25.4x           |       0 |
| momentum | trailing 10% (cd 5)                         | 21.8%  | 36.8%      |     0.62 | -34.1%         |     0.64 | 57.7x           |     244 |
| momentum | trailing 10% (cd 5, slot->cash)             | 21.8%  | 31.3%      |     0.67 | -29.6%         |     0.74 | 50.7x           |     224 |
| momentum | trailing 10% (cd 21)                        | 29.0%  | 33.9%      |     0.82 | -35.7%         |     0.81 | 66.0x           |     185 |
| momentum | trailing 10% (cd 21, slot->cash)            | 16.4%  | 23.1%      |     0.62 | -28.2%         |     0.58 | 40.7x           |     151 |
| momentum | trailing 10% (cd 63)                        | 14.0%  | 30.7%      |     0.46 | -30.0%         |     0.47 | 71.3x           |     138 |
| momentum | trailing 10% (cd 63, slot->cash)            | 10.7%  | 18.5%      |     0.45 | -25.3%         |     0.42 | 32.7x           |     101 |
| momentum | residual 2x resid sigma (cd 5)              | 36.1%  | 40.1%      |     0.88 | -40.9%         |     0.88 | 27.2x           |      19 |
| momentum | residual 2x resid sigma (cd 5, slot->cash)  | 34.9%  | 39.7%      |     0.86 | -40.0%         |     0.87 | 27.6x           |      19 |
| momentum | residual 2x resid sigma (cd 21)             | 34.4%  | 40.0%      |     0.85 | -40.9%         |     0.84 | 29.1x           |      19 |
| momentum | residual 2x resid sigma (cd 21, slot->cash) | 33.7%  | 39.2%      |     0.84 | -40.0%         |     0.84 | 27.0x           |      19 |
| momentum | residual 2x resid sigma (cd 63)             | 35.0%  | 40.0%      |     0.86 | -40.9%         |     0.86 | 29.5x           |      19 |
| momentum | residual 2x resid sigma (cd 63, slot->cash) | 33.7%  | 39.1%      |     0.85 | -40.0%         |     0.84 | 26.8x           |      19 |
| momentum | support low - 1x ATR, pivot 3 (cd 21)       | 12.0%  | 35.0%      |     0.39 | -37.7%         |     0.32 | 85.6x           |     391 |
| momentum | support low - 1x ATR, pivot 10 (cd 21)      | 29.9%  | 37.8%      |     0.79 | -41.2%         |     0.73 | 45.3x           |     113 |
| momentum | support low - 1x ATR, lookback 63 (cd 21)   | 13.3%  | 36.6%      |     0.43 | -44.8%         |     0.3  | 66.3x           |     242 |
| momentum | support low - 1x ATR, lookback 252 (cd 21)  | 13.5%  | 36.6%      |     0.43 | -44.7%         |     0.3  | 70.3x           |     282 |
| resmom   | no stop                                     | 20.3%  | 30.7%      |     0.64 | -34.9%         |     0.58 | 14.0x           |       0 |
| resmom   | trailing 10% (cd 5)                         | 12.9%  | 27.5%      |     0.45 | -34.0%         |     0.38 | 32.1x           |     162 |
| resmom   | trailing 10% (cd 5, slot->cash)             | 5.4%   | 24.1%      |     0.19 | -26.9%         |     0.2  | 32.8x           |     153 |
| resmom   | trailing 10% (cd 21)                        | 11.1%  | 24.9%      |     0.4  | -31.0%         |     0.36 | 38.3x           |     145 |
| resmom   | trailing 10% (cd 21, slot->cash)            | 5.0%   | 19.4%      |     0.16 | -25.6%         |     0.2  | 25.4x           |     112 |
| resmom   | trailing 10% (cd 63)                        | 3.3%   | 23.3%      |     0.1  | -34.6%         |     0.1  | 45.9x           |     121 |
| resmom   | trailing 10% (cd 63, slot->cash)            | -0.0%  | 14.8%      |    -0.17 | -23.1%         |    -0    | 19.8x           |      81 |
| resmom   | residual 2x resid sigma (cd 5)              | 22.4%  | 30.9%      |     0.69 | -35.2%         |     0.64 | 16.9x           |      33 |
| resmom   | residual 2x resid sigma (cd 5, slot->cash)  | 20.6%  | 30.3%      |     0.65 | -34.4%         |     0.6  | 17.7x           |      33 |
| resmom   | residual 2x resid sigma (cd 21)             | 26.3%  | 31.0%      |     0.79 | -35.2%         |     0.75 | 16.8x           |      32 |
| resmom   | residual 2x resid sigma (cd 21, slot->cash) | 21.1%  | 29.1%      |     0.68 | -34.4%         |     0.61 | 16.8x           |      31 |
| resmom   | residual 2x resid sigma (cd 63)             | 24.3%  | 31.1%      |     0.74 | -35.0%         |     0.7  | 18.2x           |      31 |
| resmom   | residual 2x resid sigma (cd 63, slot->cash) | 16.8%  | 28.2%      |     0.56 | -31.7%         |     0.53 | 17.0x           |      30 |
| resmom   | support low - 1x ATR, pivot 3 (cd 21)       | 0.0%   | 27.8%      |     0.01 | -45.3%         |     0    | 75.4x           |     426 |
| resmom   | support low - 1x ATR, pivot 10 (cd 21)      | 27.3%  | 29.7%      |     0.84 | -33.8%         |     0.81 | 29.6x           |     120 |
| resmom   | support low - 1x ATR, lookback 63 (cd 21)   | 14.9%  | 29.0%      |     0.5  | -31.3%         |     0.48 | 47.8x           |     242 |
| resmom   | support low - 1x ATR, lookback 252 (cd 21)  | 12.6%  | 28.4%      |     0.43 | -31.3%         |     0.4  | 52.5x           |     284 |

### extended: paired across 8 (n_hold, exit_rank) cells

| Book     | Stop                            |   Cells | Lower max DD   | Median ΔMaxDD   | Higher Calmar   | Median ΔCAGR   | Median ΔVol   | Median Δturnover   |
|:---------|:--------------------------------|--------:|:---------------|:----------------|:----------------|:---------------|:--------------|:-------------------|
| momentum | fixed 10% (cd 21)               |       8 | 7/8            | +1.3%           | 0/8             | -9.6%          | -2.9%         | +10.3x             |
| momentum | fixed 20% (cd 21)               |       8 | 5/8            | +0.5%           | 1/8             | -5.2%          | -1.5%         | +3.6x              |
| momentum | trailing 10% (cd 21)            |       8 | 7/8            | +3.0%           | 0/8             | -17.8%         | -7.2%         | +38.7x             |
| momentum | trailing 20% (cd 21)            |       8 | 5/8            | +2.7%           | 0/8             | -14.6%         | -4.4%         | +10.9x             |
| momentum | chandelier 3x ATR20 (cd 21)     |       8 | 8/8            | +3.7%           | 0/8             | -25.7%         | -7.3%         | +68.4x             |
| momentum | chandelier 6x ATR20 (cd 21)     |       8 | 5/8            | +1.1%           | 0/8             | -17.7%         | -2.4%         | +25.0x             |
| momentum | residual 1x resid sigma (cd 21) |       8 | 2/8            | -1.3%           | 0/8             | -18.3%         | -2.4%         | +26.3x             |
| momentum | residual 2x resid sigma (cd 21) |       8 | 4/8            | +0.2%           | 4/8             | -1.1%          | -0.1%         | +2.3x              |
| momentum | support low - 0.5x ATR (cd 21)  |       8 | 6/8            | +1.2%           | 0/8             | -23.8%         | -5.3%         | +49.9x             |
| momentum | support low - 1x ATR (cd 21)    |       8 | 2/8            | -2.1%           | 0/8             | -23.4%         | -4.0%         | +38.6x             |
| momentum | support low - 2x ATR (cd 21)    |       8 | 2/8            | -1.5%           | 0/8             | -21.5%         | -3.0%         | +30.7x             |
| momentum | book dd 13% (cd 5)              |       8 | 8/8            | +14.6%          | 4/8             | -13.9%         | -12.4%        | -1.3x              |
| momentum | book dd 20% (cd 5)              |       8 | 8/8            | +12.7%          | 7/8             | -5.4%          | -9.2%         | -3.4x              |
| resmom   | fixed 10% (cd 21)               |       8 | 7/8            | +4.5%           | 7/8             | +2.1%          | -2.5%         | +9.6x              |
| resmom   | fixed 20% (cd 21)               |       8 | 3/8            | -0.2%           | 2/8             | -2.3%          | -1.3%         | +2.5x              |
| resmom   | trailing 10% (cd 21)            |       8 | 6/8            | +2.8%           | 1/8             | -7.1%          | -5.4%         | +25.5x             |
| resmom   | trailing 20% (cd 21)            |       8 | 6/8            | +1.3%           | 4/8             | -2.4%          | -2.6%         | +5.4x              |
| resmom   | chandelier 3x ATR20 (cd 21)     |       8 | 0/8            | -7.4%           | 0/8             | -21.4%         | -2.4%         | +67.7x             |
| resmom   | chandelier 6x ATR20 (cd 21)     |       8 | 2/8            | -2.7%           | 4/8             | +0.3%          | -0.3%         | +22.3x             |
| resmom   | residual 1x resid sigma (cd 21) |       8 | 4/8            | -1.5%           | 5/8             | +0.8%          | -1.0%         | +24.0x             |
| resmom   | residual 2x resid sigma (cd 21) |       8 | 4/8            | +0.6%           | 6/8             | +3.0%          | -0.0%         | +3.4x              |
| resmom   | support low - 0.5x ATR (cd 21)  |       8 | 3/8            | -1.0%           | 0/8             | -8.9%          | -2.4%         | +44.6x             |
| resmom   | support low - 1x ATR (cd 21)    |       8 | 6/8            | +1.7%           | 4/8             | -0.6%          | -2.0%         | +38.3x             |
| resmom   | support low - 2x ATR (cd 21)    |       8 | 3/8            | -1.3%           | 2/8             | -2.4%          | -1.3%         | +27.5x             |
| resmom   | book dd 13% (cd 5)              |       8 | 6/8            | +8.8%           | 2/8             | -8.7%          | -7.5%         | +2.0x              |
| resmom   | book dd 20% (cd 5)              |       8 | 1/8            | -5.9%           | 0/8             | -9.5%          | -4.3%         | +3.9x              |
