# Intraday add-on strategies beside the momentum book

**Question.** The live momentum book keeps a VM up through market hours.
Could a 15-minute monitor on the same VM run a profitable swing / intraday
sleeve in parallel?

**Short answer: no rule tested has an edge.** The ones that make money only
earn the market's drift — what simply holding the stock over the same hours
earns — and the momentum book already collects that with far more exposure.
The intraday-specific rules (opening-range breakout, last-hour momentum, gap
fades, buying first-hour drops) lose after costs.

## Data

Collected through the IB connector (regular-hours OHLCV, as traded, checked
against the lab's cached closes to the cent):

| Set | Range | Used for |
|---|---|---|
| QQQ daily | 2022-10-14 → 2026-10-08 (4.0 years) | overnight vs intraday, gaps |
| Hourly: QQQ + AAPL, AMAT, AMD, AMZN, ARM, AVGO, GOOGL, INTC, META, MRVL, MSFT, MU, NFLX, NVDA, PLTR, TSLA | 2026-03-02 → 2026-10-08 (154 sessions) | intraday rules |
| QQQ 15-minute | 2026-08-14 → 2026-10-08 (39 sessions) | cross-check |

IB serves at most 1,000 bars a request and no end date, so 15-minute history
reaches back only ~38 sessions; hourly is the finest bar with a usable
sample. The data is not committed (IB market data is licensed); re-collect it
with the connector to reproduce.

**Execution.** Every rule decides on a bar's close and fills at the next
bar's open (or the session close), long only, with 2 bp per side for
commission and spread.

## Results

| Rule | Where | Trades | Net per trade | t | Verdict |
|---|---|---|---|---|---|
| Hold overnight (buy close, sell next open) | QQQ, 4 yrs | 997 | +2.9 bp | 1.1 | real but small: +6.6%/yr vs +30.4% buy & hold. The momentum book holds overnight already |
| After a down day, hold to the next close | QQQ, 4 yrs | 461 | +9.7 bp | 1.4 | not significant |
| Gap up > 0.5%, buy open → close | QQQ, 4 yrs | 233 | −11.1 bp | −1.5 | loses |
| Hourly RSI(2) < 10 dip in an uptrend | 16 stocks | 280 | +21.6 bp | 1.2 | **only drift** — see below |
| Opening-range breakout, exit at close | 16 stocks | 952 | −3.5 bp | −0.8 | loses |
| Last-hour momentum (first 30 min up) | 16 stocks | 1,237 | −0.9 bp | −0.4 | loses |
| First hour down > 1%, buy → close | 16 stocks | 629 | −0.1 bp | 0.0 | worse than drift (below) |
| RSI(2) dip, momentum-book names only | held names | 61 | +10.8 bp | 0.2 | worse than drift |

**Against drift.** In this sample QQQ rose ~42% a year, so any long trade
looks good. Measured against what holding the same stock for the same hours
earned:

| Rule | Excess over drift per trade | t |
|---|---|---|
| Hourly RSI(2) dip in an uptrend | −3.0 bp | −0.16 |
| First hour down > 1%, buy → close | −16.6 bp | −2.24 |
| RSI(2) dip in a momentum-book name | −48.0 bp | −0.79 |

The dips earn exactly the drift and nothing more. Buying a first-hour drop
does *significantly worse* than holding: early weakness tends to continue
into the close, not reverse.

## What the monitor is worth instead

A process watching 15-minute prices does not earn its keep as a trading
sleeve here. If it runs anyway, its value is as a risk monitor for the book
it sits beside: the book drawdown stop (13%/20%, `docs/STOP_LOSS.md`)
currently reads the close only; an intraday reading of the same number
would see a crash a session earlier. That would need its own test before
it changed anything.

## Caveats

Seven months of hourly bars in a strong up-market, 39 days of 15-minute
bars, and 4 years of daily QQQ. Long only; shorting the losing rules is not a
strategy either (they lose by less than costs would take back).

## Reproduce

```bash
# Put the IB JSON files (SYMBOL_1h.json, QQQ_15m.json, QQQ_1d.json) in a folder:
python scripts/intraday_study.py --data path/to/ib
```

---

# Results

## QQQ daily bars, 2022-10-14 to 2026-10-08 (4.0 years; buy & hold +30.4% a year)

| Strategy                                    |   Trades |   Trades/yr | Win   |   Avg gross (bp) |   Avg net (bp) |   t(net) | Sleeve ann.   |
|:--------------------------------------------|---------:|------------:|:------|-----------------:|---------------:|---------:|:--------------|
| overnight: buy close, sell next open        |      997 |         251 | 53%   |              6.9 |            2.9 |     1.11 | +6.6%         |
| intraday: buy open, sell close              |      997 |         251 | 52%   |              4.4 |            0.4 |     0.13 | -0.3%         |
| gap down > 0.5%: buy open, sell close       |      184 |          46 | 52%   |              3.5 |           -0.5 |    -0.06 | -0.5%         |
| gap up > 0.5%: buy open, sell close         |      233 |          59 | 49%   |             -7.1 |          -11.1 |    -1.54 | -6.7%         |
| after a down day: buy close, sell next open |      461 |         116 | 54%   |              7.9 |            3.9 |     0.93 | +4.1%         |
| after a down day: buy close, hold 1 day     |      461 |         116 | 53%   |             13.7 |            9.7 |     1.43 | +10.6%        |

## Hourly bars, 2026-03-02 to 2026-10-08 (0.61 years; QQQ buy & hold +41.7% a year)

### QQQ alone (one slot)

| Strategy                                       |   Trades |   Trades/yr | Win   |   Avg gross (bp) |   Avg net (bp) |   t(net) | Sleeve ann.   |
|:-----------------------------------------------|---------:|------------:|:------|-----------------:|---------------:|---------:|:--------------|
| RSI2 < 10 in uptrend, exit RSI2 > 70 or 2 days |       19 |          31 | 68%   |             24.9 |           20.9 |     1.1  | +6.6%         |
| opening-range breakout, exit at close          |       78 |         128 | 49%   |             -0.5 |           -4.5 |    -0.67 | -5.8%         |
| first hour down > 1%: buy, exit at close       |        6 |          10 | 67%   |             -3.4 |           -7.4 |    -0.32 | -0.7%         |
| last-hour momentum (first 30 min up)           |       85 |         139 | 51%   |              1.2 |           -2.8 |    -1.02 | -3.8%         |

### 16 stocks (six slots)

| Strategy                                       |   Trades |   Trades/yr | Win   |   Avg gross (bp) |   Avg net (bp) |   t(net) | Sleeve ann.   |
|:-----------------------------------------------|---------:|------------:|:------|-----------------:|---------------:|---------:|:--------------|
| RSI2 < 10 in uptrend, exit RSI2 > 70 or 2 days |      280 |         458 | 58%   |             25.6 |           21.6 |     1.2  | +12.1%        |
| opening-range breakout, exit at close          |      952 |        1558 | 46%   |              0.5 |           -3.5 |    -0.78 | -5.8%         |
| first hour down > 1%: buy, exit at close       |      629 |        1029 | 51%   |              3.9 |           -0.1 |    -0.01 | +7.7%         |
| last-hour momentum (first 30 min up)           |     1237 |        2024 | 49%   |              3.1 |           -0.9 |    -0.39 | -3.0%         |
| RSI2 < 10 dip in a momentum-book name          |       61 |         100 | 56%   |             14.8 |           10.8 |     0.19 | +1.4%         |

Momentum-book holdings come from the cached closes, which end 2026-09-08; that rule only trades up to then.

## Cross-check: QQQ 15-minute bars, 2026-08-14 to 2026-10-08 (39 days)

| Strategy                                       |   Trades |   Trades/yr | Win   |   Avg gross (bp) |   Avg net (bp) | t(net)   | Sleeve ann.   |
|:-----------------------------------------------|---------:|------------:|:------|-----------------:|---------------:|:---------|:--------------|
| RSI2 < 10 in uptrend, exit RSI2 > 70 or 2 days |       20 |         129 | 75%   |             12   |            8   | +1.75    | +10.9%        |
| opening-range breakout, exit at close          |       22 |         142 | 50%   |             -8.4 |          -12.4 | -1.10    | -16.3%        |
| first hour down > 1%: buy, exit at close       |        1 |           6 | 100%  |             28.3 |           24.3 | —        | +1.6%         |
| last-hour momentum (first 30 min up)           |       23 |         149 | 48%   |             -1.1 |           -5.1 | -1.42    | -7.4%         |
