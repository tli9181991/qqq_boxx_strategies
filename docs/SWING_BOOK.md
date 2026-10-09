# An auto-selecting swing book: 6 slots × $1,000

**Question.** Can the six swing screens of the Swing trades tab (`qbs/swing.py`)
pick ~6 stocks automatically, $1,000 each, and swing-trade them — like the
Top-6 momentum book, but entering on setups and exiting on stops and targets?

**Short answer: not profitably, as screened.** Ranked by reward/risk, the
swing book loses money after costs in both windows, and has no edge over QQQ
even with costs set to zero. Two things sink it:

1. **The screens do not pick winners by themselves.** Two thirds of trades
   end at their stop. With zero costs the book makes 15.9% a year in the
   default window (QQQ 21.9%) and −1.8% in the extended one (QQQ 14.0%).
2. **$1,000 trades are expensive.** IB's $1 minimum commission is 0.1% each
   way on $1,000; with slippage and ~240 trades a year that is **8–12% of
   capital a year** in costs.

Ranking the same candidates by **6-1 momentum** instead of reward/risk is
what helps — the edge is momentum, not the setup. The best swing variant
(momentum first, stops 2× wider) makes 32.7% / 23.2% a year, still below
the plain **Top-6 momentum book traded at the same $1,000 slots** (56.7% /
27.9%), which trades a quarter as often.

| | Default: CAGR | Max DD | Trades/yr | Costs/yr | Extended: CAGR | Max DD |
|---|---|---|---|---|---|---|
| Swing, best R:R first | 0.2% | −30.0% | 245 | 11.7% | −19.4% | −64.1% |
| Swing, best R:R first, **no costs** | 15.9% | −29.7% | 241 | 0% | −1.8% | −40.2% |
| Swing, momentum first | 32.6% | −35.4% | 241 | 11.6% | 15.1% | −34.0% |
| **Swing, momentum first, stops 2× wider** | 32.7% | −36.9% | 160 | 7.6% | 23.2% | −31.7% |
| **Top-6 momentum, $1k slots** | **56.7%** | −32.6% | 60 | 3.0% | **27.9%** | −37.2% |
| QQQ buy & hold | 21.9% | −22.8% | — | — | 14.0% | −34.8% |

## How it trades (`qbs/swing_book.py`)

On each session's close:

* **Exits first** — at the screen's stop, at its target, or after the setup's
  longest typical hold (trend pullback 15 sessions, breakout retest 20, range
  bounce 10, mean reversion 7, volatility contraction 15, relative strength 20).
* **Then entries** — free slots are filled from that day's swing scan, one
  trade per ticker, best first (by reward/risk, or by 6-1 momentum). A name
  stopped out today is not re-bought the same day.
* **Sizing and costs** — $1,000 a slot in whole shares (a stock above $1,000
  cannot be bought at all; a $505 stock fills one share), IB fixed commission
  ($0.005/share, $1 minimum, 1% maximum), 5 bp slippage, idle cash in BOXX.

The scans are re-run as of every session on prices up to that session only,
so no entry sees the future. Fills are at the decision close, and stops are
checked on closes (the cache has no intraday highs and lows).

## What the variants say

* **Every setup on its own loses or barely breaks even** after costs: the best
  single setup was relative strength in the default window (8.6%) and range
  bounce / mean reversion in the extended one (1.6–2.5%).
* **Wider stops** cut the number of trades and costs but do not fix the
  reward/risk-ranked book (−11.1% / −14.4%); they help the momentum-ranked one.
* **Bigger slots** ($3,000) halve the cost drag (6.6% / 5.1% a year) — the
  commission minimum is the problem at $1,000 — but the book is still below
  QQQ.
* **Fractional shares** help a little (no idle cash in expensive names).

## If you want a small automated book anyway

Trade the **Top-6 momentum book at $1,000 a slot**: same ranking as the live
book, ~60–75 trades a year, ~3% a year in costs. Expect a large share of
idle cash in names above $1,000 (one share or none). Keep the Swing trades
tab as a discretionary screen — its stops, targets and Fibonacci levels are
useful reference prices for a trade you choose, but the screens are not a
signal to automate.

## Two years on real daily bars (IB)

The tables above use the lab's cached closes: a day's range is the
close-to-close envelope and a stop is seen only at the close. The same book
was re-run on **real daily Open/High/Low/Close from IB** for 44 of the
Nasdaq-100 names (the ones with four years of IB history under one contract;
ARM, AZN and LRCX had too little and were left out), over the two years
**2024-09-09 → 2026-09-08**:

* the screens size ATR, stops and level merges from true daily ranges, as the
  tab does with Source → Online;
* stops and targets fill **intraday** — at the stop when the low reaches it,
  at the open on a gap through it; a day touching both counts as a stop.

Every arm uses the same 44 names and identical closes, so the arms differ
only in the data. (`scripts/swing_ohlc_study.py`; IB data not committed.)

| 44 names, 2 years | Closes only | **Real bars** | Real scans, close exits |
|---|---|---|---|
| All setups, best R:R first | 16.9% / −18.7% | **19.9% / −17.8%** | 11.9% / −21.5% |
| All setups, best momentum first | 48.3% / −26.6% | **47.1% / −24.6%** | 52.9% / −19.7% |
| Momentum first, stops 2× wider | 55.4% / −21.7% | **62.1% / −21.5%** | 53.0% / −21.2% |
| *Top-6 momentum, $1k slots (same 44)* | | **83.6% / −32.5%** | |
| *QQQ buy & hold* | | **26.6% / −22.8%** | |

(CAGR / max drawdown. Swing books trade 140–235 times a year at 7–11% of
capital a year in costs; Top-6 momentum trades 41 times at 2%.)

**What real bars change — and do not.**

* The verdict on the setups stands. Ranked by reward/risk the book makes
  ~20% a year, **below QQQ (26.6%)**. Of the six setups on their own, none
  beats QQQ on real bars; mean reversion (16.5%) and range bounce (14.8%,
  −8% drawdown) come closest, and trend pullback (−4.5%) and breakout retest
  (0.0%) are the weakest once stops fill intraday.
* Real bars help the **momentum-ranked** book: 62.1% a year with a −21.5%
  drawdown and the best Calmar of the full books (2.89, against 2.58 for
  Top-6 momentum on the same names; only the volatility-contraction-only
  book scored higher, and it sits in cash most of the time at 32 trades a
  year). That is the one swing variant worth a
  second look: the setups used as **entry timing for strong names**, with
  stops twice the screens' width. It still earns about 20 points a year less
  than simply holding the Top-6 momentum book, and trades four times as often.
* Three quarters of R:R-ranked trades still end at the stop (−4.2% average)
  or the time limit; the 10% winners at target are too few to pay for them.

## Caveats

Same as every study here: today's Nasdaq-100 applied to history
(survivorship bias), split-adjusted closes, at most 4.7 years with one bear
market, and close-only stops.

## Reproduce

```bash
python scripts/swing_ohlc_study.py --ib path/to/ib_daily   # real bars, 2 years
python scripts/swing_book_study.py --build-scans   # first run: ~10 min on 4 cores
python scripts/swing_book_study.py                 # later runs reuse data/swing_scans.pkl
```

---

# Results

## default window (2025-01-20 to 2026-09-08)

| Book                                   | CAGR   | Vol   | MaxDD   |   Calmar |   Trades/yr | Win rate   | Avg trade   | Avg hold   | Costs/yr   | Turnover   |
|:---------------------------------------|:-------|:------|:--------|---------:|------------:|:-----------|:------------|:-----------|:-----------|:-----------|
| swing: all setups, best R:R first      | 0.2%   | 22.0% | -30.0%  |     0.01 |         245 | 31.7%      | -0.1%       | 6          | 11.7%      | 71.7x      |
| swing: all setups, best momentum first | 32.6%  | 32.2% | -35.4%  |     0.92 |         241 | 44.2%      | 1.0%        | 6          | 11.6%      | 70.4x      |
| swing: R:R >= 2 only                   | 0.0%   | 22.1% | -30.0%  |     0    |         244 | 31.3%      | -0.1%       | 6          | 11.7%      | 71.6x      |
| swing: fractional shares               | 1.7%   | 20.8% | -27.9%  |     0.06 |         210 | 33.1%      | 0.0%        | 6          | 10.6%      | 71.2x      |
| swing: stops 2x wider                  | -11.1% | 21.2% | -35.1%  |    -0.31 |         187 | 39.1%      | -0.3%       | 8          | 8.7%       | 47.7x      |
| swing: $3k slots                       | 13.0%  | 22.8% | -30.5%  |     0.43 |         245 | 33.8%      | 0.3%        | 6          | 6.6%       | 76.9x      |
| swing: NO costs (signal only)          | 15.9%  | 22.1% | -29.7%  |     0.54 |         241 | 33.5%      | 0.4%        | 6          | 0.0%       | 78.8x      |
| swing, momentum first: stops 2x wider  | 32.7%  | 32.0% | -36.9%  |     0.89 |         160 | 52.7%      | 1.4%        | 9          | 7.6%       | 45.1x      |
| swing, momentum first: NO costs        | 49.5%  | 32.7% | -34.3%  |     1.44 |         229 | 45.7%      | 1.4%        | 7          | 0.0%       | 76.2x      |
| swing: trend_pullback only             | 3.8%   | 20.1% | -28.8%  |     0.13 |         238 | 34.2%      | 0.1%        | 6          | 11.3%      | 66.9x      |
| swing: breakout_retest only            | 1.4%   | 20.6% | -22.7%  |     0.06 |         221 | 38.0%      | -0.0%       | 6          | 10.7%      | 66.6x      |
| swing: range_bounce only               | -3.7%  | 18.5% | -24.9%  |    -0.15 |         205 | 38.6%      | -0.2%       | 6          | 9.9%       | 60.4x      |
| swing: mean_reversion only             | 3.6%   | 21.1% | -23.9%  |     0.15 |         314 | 54.6%      | 0.0%        | 4          | 15.2%      | 94.3x      |
| swing: volatility_contraction only     | -8.0%  | 14.5% | -18.7%  |    -0.43 |         128 | 29.0%      | -0.6%       | 7          | 6.2%       | 38.8x      |
| swing: relative_strength only          | 8.6%   | 19.7% | -15.6%  |     0.55 |         213 | 49.0%      | 0.3%        | 7          | 10.4%      | 66.1x      |
| Top-6 momentum, $1k slots              | 56.7%  | 41.0% | -32.6%  |     1.74 |          60 | 51.5%      | 2.1%        | 21         | 3.0%       | 18.6x      |
| QQQ buy & hold                         | 21.9%  | 22.6% | -22.8%  |     0.96 |           0 | —          | —           | —          | 0.0%       | 0.0x       |

### default: how trades end (all setups, best R:R first)

          n    avg hold
reason                 
stop    263  -3.7%  4.0
target   58 +10.9%  7.7
time     76  +3.8% 12.1

## extended window (2022-01-03 to 2026-09-08)

| Book                                   | CAGR   | Vol   | MaxDD   |   Calmar |   Trades/yr | Win rate   | Avg trade   | Avg hold   | Costs/yr   | Turnover   |
|:---------------------------------------|:-------|:------|:--------|---------:|------------:|:-----------|:------------|:-----------|:-----------|:-----------|
| swing: all setups, best R:R first      | -19.4% | 22.7% | -64.1%  |    -0.3  |         230 | 31.1%      | -0.7%       | 6          | 9.5%       | 45.0x      |
| swing: all setups, best momentum first | 15.1%  | 28.4% | -34.0%  |     0.44 |         241 | 41.6%      | 0.5%        | 6          | 11.3%      | 66.4x      |
| swing: R:R >= 2 only                   | -12.9% | 23.2% | -59.2%  |    -0.22 |         230 | 30.2%      | -0.6%       | 6          | 9.7%       | 47.5x      |
| swing: fractional shares               | -4.9%  | 21.8% | -53.8%  |    -0.09 |         139 | 30.3%      | -0.2%       | 6          | 7.0%       | 46.6x      |
| swing: stops 2x wider                  | -14.4% | 22.2% | -58.6%  |    -0.25 |         180 | 37.6%      | -0.6%       | 8          | 7.8%       | 39.8x      |
| swing: $3k slots                       | -5.8%  | 23.3% | -55.0%  |    -0.1  |         230 | 31.6%      | -0.2%       | 6          | 5.1%       | 52.5x      |
| swing: NO costs (signal only)          | -1.8%  | 22.2% | -40.2%  |    -0.04 |         229 | 31.5%      | -0.0%       | 6          | 0.0%       | 64.3x      |
| swing, momentum first: stops 2x wider  | 23.2%  | 25.3% | -31.7%  |     0.73 |         166 | 51.5%      | 1.3%        | 9          | 7.9%       | 47.0x      |
| swing, momentum first: NO costs        | 23.3%  | 25.4% | -23.8%  |     0.98 |         242 | 43.2%      | 0.8%        | 6          | 0.0%       | 76.2x      |
| swing: trend_pullback only             | -10.1% | 21.6% | -57.8%  |    -0.18 |         214 | 31.6%      | -0.5%       | 6          | 9.2%       | 46.6x      |
| swing: breakout_retest only            | -17.9% | 23.9% | -60.1%  |    -0.3  |         208 | 37.7%      | -0.6%       | 6          | 9.1%       | 46.0x      |
| swing: range_bounce only               | 1.6%   | 18.6% | -26.0%  |     0.06 |         197 | 37.7%      | 0.0%        | 6          | 9.5%       | 58.3x      |
| swing: mean_reversion only             | 2.5%   | 18.3% | -25.9%  |     0.1  |         280 | 52.5%      | 0.0%        | 4          | 13.4%      | 82.0x      |
| swing: volatility_contraction only     | -6.5%  | 15.8% | -31.5%  |    -0.21 |         129 | 31.7%      | -0.5%       | 8          | 6.1%       | 35.7x      |
| swing: relative_strength only          | 0.7%   | 23.3% | -41.9%  |     0.02 |         210 | 47.6%      | 0.1%        | 7          | 10.0%      | 60.3x      |
| Top-6 momentum, $1k slots              | 27.9%  | 30.4% | -37.2%  |     0.75 |          73 | 47.6%      | 2.5%        | 19         | 3.5%       | 20.5x      |
| QQQ buy & hold                         | 14.0%  | 23.3% | -34.8%  |     0.4  |           0 | —          | —           | —          | 0.0%       | 0.0x       |

### extended: how trades end (all setups, best R:R first)

          n   avg hold
reason                
stop    697 -3.9%  4.2
target  164 +8.4%  8.3
time    211 +2.9% 12.2
