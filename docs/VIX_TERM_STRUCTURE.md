# The VIX term structure breaker

Risk off while the VIX curve is **inverted**, on while it is in contango.
The signal is `VIX / VIX3M`; above 1.0 the market is paying more for a month
of protection than for three.

> ## ⚠️ This strategy is implemented and tested, but **not measured**
>
> `^VIX3M` is not in the bundled cache and could not be downloaded in the
> environment this was written in — both Yahoo and CBOE were refused by the
> network policy. Every number below therefore comes from `synthetic_vix3m`,
> a fixture built to exercise the code.
>
> The rules are real and the tests are real. **The edge is unknown.** Fetch
> `^VIX3M` and re-run before you believe any figure on this page.
>
> The repo has precedent for shipping in this state — the weekly breakout book
> is validated on synthetic hourly bars for the same kind of reason — and the
> same rule applies: read this as a description of behaviour, never of edge.

## Why the slope rather than the level

The README already concedes the problem with the level breaker, and the
numbers are stark. `VixBreakerParams(exit_level=17)` sits near the **median**
of VIX's own distribution, so the breaker is engaged about half the time: on
the cached window it is invested 35% of days and takes 29 trips to cash. That
is not a crash filter, it is a different strategy — one that mostly isn't in
the market.

There is a deeper objection than calibration. A high VIX is historically
followed by *high* returns: that is the volatility risk premium, paid to
whoever holds through the fear. Selling because implied vol is high is close
to selling because you are being offered a premium to stay.

Inversion is a different claim. It is rare — `^VIX` has closed above `^VIX3M`
on roughly **8% of trading days since 2010** — and it says something the level
does not: the curve itself has stopped treating the stress as transitory.

On the fixture, that difference shows up exactly where you would expect:

| | Time invested | Trips to cash |
|---|---|---|
| VIX level breaker (17/16) | 35% | 29 |
| **VIX term structure (1.00/0.95)** | **90%** | **9** |

It behaves like a filter rather than a replacement. That much is structural
and does not depend on the fixture.

## The rules

Identical to the level breaker's, deliberately — so a comparison between them
is a comparison of *signals* and nothing else. Both now call one shared state
machine (`_stress_breaker`), which is what guarantees it.

| State | Transition | Holds |
|---|---|---|
| `INVESTED` | ratio > `exit_ratio` → `CASH` | the base strategy's weights |
| `CASH` | `park_after_days` elapsed → `PARKED`<br>ratio < `entry_ratio` **and** `min_cash_days` met → `INVESTED` | nothing — true 0% cash |
| `PARKED` | ratio < `entry_ratio` → `INVESTED` | BOXX |

A missing or zero long leg produces a NaN ratio, which leaves the machine
where it is. "The data did not print today" is not the same as "the curve is
calm", and the code must not confuse them.

## What it does on the cached window

Real VIX, **synthetic VIX3M**, Top-6 momentum as the base:

| | CAGR | Vol | Sharpe | Max DD | Turnover |
|---|---|---|---|---|---|
| Top-6 NDX momentum | 48.1% | 49.2% | 0.96 | −40.4% | 21.1× |
| Top-6 + VIX level breaker | 3.8% | 24.7% | 0.11 | −26.5% | 32.4× |
| Top-6 + VIX term structure | 39.4% | 43.8% | 0.88 | **−40.4%** | 32.8× |

**It costs return and removes no drawdown at all.** That second part is not a
rounding coincidence — the drawdown is identical at every trigger level in the
sweep, and the reason is worth more than the strategy:

```
worst drawdown -40.4%:  2026-06-22 -> 2026-07-29
  days inverted during it: 0 of 27      (ratio never exceeded 0.86)
  VIX over the window:     15.0 - 20.7  (at or below its median)
  QQQ over the window:     -10.3%       (the book fell four times as far)
```

The book's defining drawdown is the **concentration blow-up**, not a market
event. Six correlated semiconductors fell together while the index barely
moved and the curve stayed in contango throughout. No term-structure signal
can see that, for the same reason no VIX level and no `QQQ > SMA(200)` filter
can: the market was not what went wrong.

So this is a third independent confirmation of the argument the README already
makes for `BookVolTargetParams` — **measure the thing you actually hold.**
A curve signal is a market signal, and this book's risk is not primarily
market risk.

Where it should help, and cannot be shown to here, is a genuine market-wide
selloff. The 2025 episode in this sample is the only candidate and one episode
is not evidence.

## Running it

```python
from qbs.config import Config, VixTermStructureParams
from qbs.pipeline import run, sweep_term_structure

cfg = Config()
cfg.vix_ts = VixTermStructureParams(exit_ratio=1.00, entry_ratio=0.95)

lab = run(cfg, vix3m=my_real_vix3m_series)   # <- supply the real thing
sweep_term_structure(lab)
```

```bash
python run_backtest.py --sweep-term-structure
python run_backtest.py --ts-exit 1.05 --ts-entry 1.00
python run_backtest.py --no-vix-ts            # skip it
```

`sweep_term_structure()` reports **Days inverted** next to **Time invested**
for one reason: if the first column is far above ~8%, the trigger has been set
inside the ordinary distribution and has become the thing it was meant to fix.

## The fixture, and one bug it caused

`synthetic_vix3m` damps VIX's deviation from its own median and lags it, so
the long leg barely moves when the short leg spikes. It then **scales the
level so the curve inverts on about 8% of days** — the same device as
`synthetic_vix`'s `target_median`, and for the same reason: a threshold
strategy's behaviour is decided entirely by where the trigger sits in the
signal's distribution.

That calibration is not decoration. Before it existed, a premium tuned against
the *synthetic* VIX was being applied to the *real* cached VIX and produced
inversion on **24% of days instead of 8%** — tripping three times too often
and making the rules look far worse than they are (CAGR 8.8%, drawdown −42.4%,
against 39.4% and −40.4% once calibrated). A fixture that is not calibrated to
the series it is handed does not test the strategy, it replaces it.

What it still does not reproduce is the **depth** of a real inversion. A real
crisis reaches 1.3–1.4; feeding it the real VIX this reaches 1.93 but with a
median ratio of 0.74 against the real ~0.92, so the distribution's shape is
wrong even where its tail frequency is right. One more reason the table above
is a description of code, not of markets.

## Sources

- [Macrosynergy, *VIX term structure as a trading signal*](https://macrosynergy.com/research/vix-term-structure-as-a-trading-signal/)
- [SystemTrader, VIX/VIX3M backwardation & contango tracker](https://www.systemtrader.co/tools/vix) — the ~8% figure
