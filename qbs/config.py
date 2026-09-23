"""Central configuration: universe, backtest window, strategy parameters, palette.

Everything a user is likely to tweak lives here so the strategy modules stay
free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------

RISK_ASSET = "QQQ"      # Nasdaq-100 -- the growth sleeve
INTL_ASSET = "VEU"      # Vanguard FTSE All-World ex-US -- GEM's relative-momentum rival
SAFE_ASSET = "BOXX"     # Alpha Architect 1-3 Month Box ETF -- T-bill-like cash proxy
VOL_INDEX = "^VIX"      # CBOE Volatility Index -- the circuit breaker's trigger
VOL_INDEX_3M = "^VIX3M"  # 3-month VIX, for the term-structure variant

TICKERS: List[str] = [RISK_ASSET, INTL_ASSET, SAFE_ASSET]

# --------------------------------------------------------------------------
# Backtest window
# --------------------------------------------------------------------------
# The reported backtest runs from inauguration day 2025-01-20 forward.
# Data is *downloaded* from an earlier date because GEM needs 12 months of
# history and the Connors filter needs 200 trading days before the first
# tradeable bar. Never shorten DOWNLOAD_START without checking WARMUP_DAYS.

BACKTEST_START = "2025-01-20"   # Trump inauguration
BACKTEST_END = None             # None -> today

WARMUP_DAYS = 300               # trading days of history required before BACKTEST_START
DOWNLOAD_START = "2023-06-01"   # comfortably covers WARMUP_DAYS + GEM's 12m lookback

# --------------------------------------------------------------------------
# Execution assumptions
# --------------------------------------------------------------------------

EXECUTION_LAG = 1        # signal computed on close of day t -> position held from t+1
COST_BPS = 1.0           # round-trip cost per 100% of turnover, in basis points
TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Strategy parameters
# --------------------------------------------------------------------------

@dataclass
class RSI2Params:
    """Larry Connors' RSI(2) mean reversion, long-only.

    The canonical rule set: only buy dips while the long-term trend is up,
    and get out fast. `entry_threshold` of 5 is Connors' aggressive variant;
    10 trades more often with a lower per-trade edge.
    """
    rsi_period: int = 2
    entry_threshold: float = 5.0     # buy when RSI(2) closes below this
    exit_rsi: float = 70.0           # sell when RSI(2) closes above this
    trend_sma: int = 200             # only trade long above this SMA
    exit_sma: int = 5                # sell when close crosses back above SMA(5)
    use_sma_exit: bool = True        # SMA(5) exit (Connors' default) vs RSI-only exit
    max_hold_days: int = 10          # hard time stop -- prevents an open trade running forever
    park_in_safe: bool = True        # hold BOXX when flat instead of 0% cash


@dataclass
class GEMParams:
    """Antonacci Global Equities Momentum, monthly rebalanced.

    Relative momentum picks the stronger equity sleeve; absolute momentum
    then checks it against the safe asset before allowing any equity risk.
    """
    lookback_months: int = 12
    risk_assets: List[str] = field(default_factory=lambda: [RISK_ASSET, INTL_ASSET])
    safe_asset: str = SAFE_ASSET
    rebalance: str = "ME"            # pandas month-end frequency alias


@dataclass
class VolTargetParams:
    """Volatility-targeting overlay.

    Scales exposure so that *forecast* portfolio volatility sits near
    `target_vol`. Vol is forecast from an EWMA of daily returns, which reacts
    faster than a plain rolling window without the cliff-edge of one dropping out.
    """
    target_vol: float = 0.15         # annualised
    halflife: int = 20               # EWMA halflife in trading days
    min_periods: int = 20
    max_weight: float = 1.0          # 1.0 = never lever; raise to allow leverage
    min_weight: float = 0.0
    rebalance_band: float = 0.05     # only trade when target weight moves this much
    vol_floor: float = 0.03          # guards against a divide-by-almost-zero blow-up


@dataclass
class MomentumParams:
    """Top-N cross-sectional momentum on the Nasdaq-100, with a hysteresis band.

    The band is the important parameter. Without it, a name that slips from
    rank 6 to rank 7 is sold and bought back a week later -- pure churn, paid
    for in spread and short-term capital gains. `exit_rank` says: buy into the
    top `n_hold`, but do not sell until the name has fallen past `exit_rank`.
    Setting exit_rank == n_hold disables the band and gives the naive version.
    """
    lookback_months: int = 6
    skip_months: int = 1          # 6-1 momentum: skip the most recent month
    n_hold: int = 6
    exit_rank: int = 8            # hysteresis band; == n_hold means no band
    rebalance: str = "daily"      # "daily" | "ME" (month end) | "W-FRI"
    weighting: str = "equal"      # "equal" | "inv_vol"
    absolute_filter: bool = True  # a name must also beat the safe asset
    safe_asset: str = SAFE_ASSET
    vol_lookback: int = 60        # for inv_vol weighting
    min_history: int = 260        # trading days a name needs before it is rankable

    # ---- correlation cap: how many DIFFERENT bets the six slots hold -----
    # The rank alone has no opinion about whether the six names it picks are
    # six bets or one bet held six times. On the Nasdaq-100 they are usually
    # closer to the latter: momentum is a trend-following signal, and the
    # names trending hardest at any moment are typically the same sector.
    # Measured on the cached window the book's held names ran a mean pairwise
    # correlation of 0.43, i.e. roughly 1.9 independent bets across 6 slots.
    #
    # `max_corr` refuses a candidate whose trailing correlation with a name
    # already selected exceeds it; the slot then goes to the next name down,
    # or to cash if nothing qualifies. None is OFF and reproduces the plain
    # ranker exactly, which is the default so nothing changes silently.
    #
    # READ THE SWEEP BEFORE SETTING IT. `sweep_corr_cap()` exists because on
    # the 20-month sample this lifts return at the shipped (6, 10) cell but
    # is close to a coin flip across the wider (n_hold, exit_rank) surface.
    # What it does do consistently is lower volatility.
    max_corr: float | None = None   # None = off. 0.75 is the sweep's mid-range
    corr_window: int = 60           # trading days of returns behind the estimate
    corr_pool: int = 30             # how far down the ranking a slot may reach

    def __post_init__(self):
        if self.exit_rank < self.n_hold:
            raise ValueError("exit_rank must be >= n_hold (the band cannot be negative)")
        if self.max_corr is not None and not -1.0 <= self.max_corr <= 1.0:
            raise ValueError("max_corr must be a correlation in [-1, 1], or None for off")
        if self.corr_pool < self.n_hold:
            raise ValueError("corr_pool must be >= n_hold (the book could never fill)")


@dataclass
class ResidualMomentumParams:
    """Residual (idiosyncratic) momentum -- Blitz, Huij & Martens (2009).

    Rank on the part of a stock's return the market cannot explain, rather
    than on its total return. Fit a market model over a trailing window, keep
    the residual return stream, and score the 12-1 window of residuals divided
    by their own standard deviation.

    Why it belongs in THIS lab specifically. The Top-6 book's problem is not
    that it picks weak names, it is that it picks the same bet six times: the
    held names run a mean pairwise correlation of 0.43, about 1.9 independent
    positions across six slots, because whatever is trending hardest in the
    Nasdaq-100 is usually one sector. Total-return momentum ranks a name highly
    for having a large beta in a rising market, so it systematically selects
    the crowded trade. Stripping the market component removes exactly that.

    The academic construction regresses on Fama-French three factors over 36
    months. This uses a SINGLE factor -- QQQ -- because the lab has no SMB/HML
    offline, and because on a Nasdaq-100 universe the market/tech factor is the
    one doing the damage. That is a real simplification and it cuts in a
    specific direction: see the warning below.

    `standardise` is what makes this a risk-adjusted score rather than a
    residual return. It is the difference between Sharpe 1.41 and 1.10 on the
    cached window, and it is in the source paper, so it defaults on.

    ⚠️ Ehsani & Linnainmaa (2022) argue residual momentum may simply be
    harvesting factors OMITTED from the regression that happen to be more
    autocorrelated than the ones included. A one-factor model omits more than
    a three-factor model does, so that critique applies here with more force,
    not less. Treat this as "momentum with the market bet removed", which is
    measurable, rather than as a distinct anomaly, which is contested.
    """
    beta_window: int = 252        # trailing days behind the market-model beta
    lookback_months: int = 12
    skip_months: int = 1          # the same 12-1 convention as the total-return book
    standardise: bool = True      # divide by the residual's own vol -- the paper's score
    # Below this daily residual standard deviation the score is a 0/0 and the
    # name is simply not rankable. Guards the degenerate case of a name the
    # market explains exactly; see `residual_momentum_score`.
    resid_vol_floor: float = 1e-6
    market_asset: str = RISK_ASSET
    n_hold: int = 6
    exit_rank: int = 10
    rebalance: str = "daily"
    absolute_filter: bool = True  # still judged on 12-1 vs the safe asset
    safe_asset: str = SAFE_ASSET
    min_history: int = 260
    max_corr: float | None = None   # the correlation cap composes with this too
    corr_window: int = 60
    corr_pool: int = 30

    def __post_init__(self):
        if self.exit_rank < self.n_hold:
            raise ValueError("exit_rank must be >= n_hold (the band cannot be negative)")
        if self.lookback_months <= self.skip_months:
            raise ValueError("lookback_months must exceed skip_months")
        if self.beta_window < 20:
            raise ValueError("beta_window is too short to estimate a beta from")


@dataclass
class FinvizScreenParams:
    """The high-momentum screen: filter on the definition, rank, take the top N.

    Filter, then a relative-strength ranking of whatever survived, then the
    strongest `n_hold`. The filter is the same high-momentum definition the
    market-overview tab's leader group uses -- price over $5, over 300k shares
    a day, up more than 28% on the quarter -- so a name shown as a "momentum
    leader" there and a name held here are selected on the same rule.

    It started as a roll-forward of `finviz_filter_with_daily_summary.ipynb`
    and kept that notebook's Finviz filters. Those are now OFF by default
    (`above_sma`, `within_52w_high_pct`) and their fields say what turning
    them back on costs. The ranking stage is unchanged and still the
    notebook's: RS Rank on a one-year return, tie-broken by distance below the
    52-week high.

    Three things worth knowing:

    * `rs_lookback` is a FIXED 252 bars. The notebook downloads `period="1y"`
      and takes `(last - first) / first`, so a name with 210 bars of history
      contributes a 210-day return to a column compared against other names'
      252-day returns. Ranking two different horizons against each other is
      not a like-for-like comparison, so the lookback is pinned.
    * `min_volume` NEEDS `volumes=`. Unlike the old `min_avg_volume`, which was
      skipped silently when volume was missing, this one raises: it is a leg of
      the definition, and dropping a leg on the floor overstates the screen.
      Set it to None to opt out deliberately -- the caller then knows, and can
      say so on screen.
    * Market cap over $300m, one of the notebook's filters, needs fundamentals
      and is not applied. On a Nasdaq-100 ranking universe it is non-binding;
      on the broad US universe it is not, and its absence is permissive.

    **`n_hold` is 20 and the universe matters.** On the ~99-name Nasdaq-100
    cache the filter passes a median of 11 names, so a top-20 is usually a
    top-whatever-qualified. Run it over the broad US universe
    (`qbs.finviz.fetch_us_universe`) for the ranking to be a real selection
    rather than a list of everyone who cleared the bar.
    """
    # ---- Finviz stage-1 filters -----------------------------------------
    # The three legs of the high-momentum definition (see BreadthParams, which
    # holds the same thresholds for the market-overview tab's leader group).
    # These are the whole filter now.
    min_price: float = 5.0              # > $5
    min_volume: float | None = 300_000.0    # > 300k shares/day -- needs volumes=
    min_quarter_return: float | None = 0.28  # > 28% over `quarter_lookback`
    quarter_lookback: int = 63          # ~one quarter

    # Subsumed by `min_quarter_return`: a name up 28% is up. Left here because
    # setting min_quarter_return to None should still leave a usable screen.
    require_quarter_up: bool = False

    # The two legs the original Finviz notebook applied and the high-momentum
    # definition does not. OFF by default; both still work if set.
    #
    # `above_sma` was near-redundant anyway -- a name up 28% on the quarter is
    # essentially always above its 200-day average, and on the cached universe
    # adding it back changes the passing count by zero. The proximity filter
    # is the one that bites: it roughly halves the qualifying set, and with it
    # on, a top-20 fills on 3% of sessions instead of 15%.
    above_sma: int = 0                  # 0 = off; 200 = "price above SMA200"
    within_52w_high_pct: float | None = None   # None = off; 0.10 = the old rule
    # The notebook's filter is a ceiling only: 0-10% below the high. Setting a
    # FLOOR turns it into a band, which is what a breakout entry needs -- a
    # name already at its high has nothing overhead left to break through.
    # 0.0 keeps the notebook's behaviour.
    min_off_high_pct: float = 0.0
    high_window: int = 252
    # Superseded by `min_volume`, which is the definition's own leg (same-day
    # shares, not a 50-day average). Off by default so the screen applies one
    # volume test rather than two.
    min_avg_volume: float | None = None  # e.g. 200_000 -- needs volumes=
    avg_volume_window: int = 50          # the notebook's Avg_Vol_50D

    # ---- other optional gates -------------------------------------------
    # Dollar volume (close x shares) -- NOT the portfolio turnover of `Ann.
    # turnover`, which is what the old name here was confusable with, and no
    # longer the same test as the leader rule's, which counts shares.
    min_dollar_volume: float | None = None    # e.g. 5e6 -- needs volumes=

    # ---- stage-3 ranking ------------------------------------------------
    rs_lookback: int = 252              # the notebook's Perf_1Y
    rs_buckets: int = 100               # its qcut(..., q=min(100, n)) RS Rank
    min_history: int = 252              # bars before a name is rankable

    # ---- turning a watchlist into a book --------------------------------
    n_hold: int = 20                    # 0 -> hold every name that passes
    exit_rank: int = 0                  # 0 = no band, which is the notebook's rule
    rebalance: str = "daily"            # "daily" | "ME" | "W-FRI"
    safe_asset: str = SAFE_ASSET
    equal_weight_slots: bool = True     # size per slot, matching the momentum book

    def __post_init__(self):
        if self.n_hold < 0:
            raise ValueError("n_hold must be >= 0 (0 means hold every passing name)")
        if self.exit_rank and self.n_hold and self.exit_rank < self.n_hold:
            raise ValueError("exit_rank must be 0 (no band) or >= n_hold")


@dataclass
class VixBreakerParams:
    """A VIX circuit breaker laid over any base strategy.

    This is a *binary* risk switch, not a vol-targeting overlay: exposure is
    on or off, not scaled. (The continuous version is `VolTargetParams`, and
    the two can be stacked.) The state machine:

        INVESTED  VIX close > exit_level          -> CASH
        CASH      held at 0% return. Then either:
                    days_in_cash >= park_after    -> PARKED  (buy the safe asset)
                    VIX < entry_level AND
                      days_in_cash >= min_days    -> INVESTED
        PARKED    VIX < entry_level               -> INVESTED

    `exit_level` above `entry_level` is a hysteresis band on volatility --
    the same trick as the momentum rank band, and for the same reason: a
    single threshold makes the strategy thrash every time VIX oscillates
    across it.

    READ THE DEFAULTS AS A WARNING, NOT A RECOMMENDATION. VIX has a long-run
    median near 17-18, so `exit_level=17` sits at roughly the middle of the
    distribution: the breaker will be tripped about half the time, which is a
    "mostly in cash" strategy rather than a crash filter. Run `sweep_vix()`
    before trusting any single pair of levels.
    """
    exit_level: float = 17.0      # VIX close above this -> go to cash
    entry_level: float = 16.0     # VIX below this -> resume the base strategy
    park_after_days: int = 3      # trading days in cash before buying the safe asset
    min_cash_days: int = 2        # minimum dwell in cash, even if VIX recovers at once
    safe_asset: str = SAFE_ASSET
    sell_safe_too: bool = True    # True: trigger sells the safe sleeve as well


@dataclass
class DrawdownStopParams:
    """A circuit breaker on the book's own drawdown.

    The overlay of last resort: when the book is far enough below its own high,
    hold nothing but the safe asset for a few sessions, then resume. It sits on
    top of a finished strategy and changes only the exposure, never the picks.

    Why the *book's* drawdown and not the market's or a position's. A market
    trigger misses the failure this book is most prone to -- in June 2026 QQQ
    fell 11% while the book fell 16%, because six correlated semiconductors
    moved together in an index that did not. A position trigger fires on
    ordinary noise: a 13% fall in a stock running at 50% vol is a routine week,
    and stopping it swaps one volatile name for another mid-decline. Measured
    over 2020-2026 both made drawdown *worse*; only the book-level measure
    improved it, from -33.9% to -17.6% with Calmar 0.81 -> 1.31.

    Why a fixed cooldown rather than a recovery signal. Every "confirm the
    recovery" rule tested was worse, and the conditions that flicker were much
    worse: re-entering when VIX returned to its average produced 87 episodes
    against this rule's 5, and turned Calmar 1.31 into 0.34. The red flag is
    already a lagging condition, so a lagging green flag stacks delay on delay
    and sells the decline while missing the rebound.

    `qqq_drawdown` is a second, redundant trigger. Over 2020-2026 the OR fires
    on exactly the same five episodes as the book leg alone and for the same
    27% of sessions, because by the time QQQ is 15% below its high this book is
    already past 13% (Calmar 1.30 against 1.31 -- inside the noise). It is kept
    because a market-wide guard is worth having on the day the two stop
    agreeing, and because it costs nothing when it never binds. Set it to 0.0
    to drop it and with it the live dependency on a benchmark series.

    Enabled. The evidence is five episodes over six years, one of them outside
    the window the live book has ever seen, so this is a deliberate choice made
    on thin data rather than a conclusion the sample forced.
    """
    enabled: bool = True
    exit_drawdown: float = 0.13   # book this far below its own high -> cash
    qqq_drawdown: float = 0.15    # same, on the benchmark. 0.0 = off
    cooldown_days: int = 5        # sessions to stay out after the flag clears
    safe_asset: str = SAFE_ASSET

    def __post_init__(self):
        if not 0.0 < self.exit_drawdown < 1.0:
            raise ValueError("exit_drawdown must be in (0, 1)")
        if not 0.0 <= self.qqq_drawdown < 1.0:
            raise ValueError("qqq_drawdown must be in [0, 1); 0 disables it")
        if self.cooldown_days < 0:
            raise ValueError("cooldown_days must not be negative")


@dataclass
class VixTermStructureParams:
    """A risk switch driven by the SLOPE of the VIX curve, not its level.

    `VixBreakerParams` asks "is implied volatility high?". This asks "is the
    near-term contract priced above the three-month one?" -- i.e. is the curve
    inverted. They are different questions and the second is the one the
    literature prefers, for a reason the README already concedes about the
    level version: a high VIX tends to precede *high* future returns, because
    you are being paid the volatility risk premium to hold through it. Level
    is a poor timing signal almost by construction.

    Backwardation is rarer and less ambiguous. VIX has closed above VIX3M on
    roughly 8% of trading days since 2010, and those episodes cluster in
    genuine stress rather than in ordinary chop. That rarity is the point: the
    default `exit_ratio` of 1.0 sits at about the 92nd percentile of the
    signal, where `VixBreakerParams(exit_level=17)` sits near the MEDIAN of
    its own and is consequently engaged half the time.

    The state machine is identical to the level breaker's, because the useful
    parts of it -- cash before the safe asset, a minimum dwell, a hysteresis
    band -- are properties of the switch, not of the signal:

        INVESTED  ratio > exit_ratio            -> CASH
        CASH      park_after_days elapsed       -> PARKED (buy the safe asset)
                  ratio < entry_ratio AND
                    min_cash_days met           -> INVESTED
        PARKED    ratio < entry_ratio           -> INVESTED

    ⚠️ NOT VALIDATED ON REAL DATA IN THIS REPO. ^VIX3M is not in the bundled
    cache and could not be downloaded in the environment this was written in,
    so every number the lab reports for this strategy comes from
    `synthetic_vix3m` -- a fixture built to exercise the code, whose inversions
    are shallower than real ones. The rules are implemented and tested; the
    edge is unmeasured. Fetch ^VIX3M and re-run before believing anything.
    """
    exit_ratio: float = 1.00      # VIX/VIX3M above this (inverted) -> risk off
    entry_ratio: float = 0.95     # back below this (contango) -> risk on
    park_after_days: int = 3      # trading days in cash before buying the safe asset
    min_cash_days: int = 2        # minimum dwell, even if the curve snaps back
    safe_asset: str = SAFE_ASSET
    sell_safe_too: bool = True    # True: the trigger sells the safe sleeve as well

    def __post_init__(self):
        if self.entry_ratio > self.exit_ratio:
            raise ValueError(
                "entry_ratio must be <= exit_ratio (the band cannot be inverted)")
        if self.exit_ratio <= 0 or self.entry_ratio <= 0:
            raise ValueError("ratios are VIX/VIX3M and must be positive")


@dataclass
class BookVolTargetParams:
    """Volatility targeting applied to a WHOLE book, on its own realised vol.

    `VolTargetParams` scales a single risk asset by *that asset's* volatility.
    This scales an entire multi-name portfolio by *the portfolio's own*
    realised volatility, which is a different and, for a concentrated book,
    far more useful thing.

    Why it matters for the Top-6 momentum strategy. That book runs at roughly
    49% annualised volatility with a beta near 1.7 to QQQ: most of its
    drawdown is leverage, not bad stock selection. Its two worst episodes had
    completely different causes --

      * a market-wide selloff (VIX 28-45), and
      * a concentration blow-up with VIX at 16-20, i.e. a calm index,

    -- and no index-level signal can see the second one. A VIX breaker and a
    `QQQ > SMA(200)` filter both leave that drawdown untouched, because the
    market was not the thing going wrong. The book's own realised vol rises in
    BOTH cases, which is why targeting it works on both.

    What this does NOT do is add return: it rescales risk. Sharpe is roughly
    unchanged; what improves is Calmar, because de-levering cuts drawdown
    faster than it cuts return. It also cannot help with an overnight gap in a
    single name -- it manages sustained volatility, not jumps.
    """
    target_vol: float = 0.25     # annualised. 0.15 for a genuinely conservative book
    halflife: int = 20           # EWMA halflife in trading days
    min_periods: int = 20
    max_weight: float = 1.0      # 1.0 = de-risk only, never lever the book up
    rebalance_band: float = 0.05  # only retrade when the scalar moves this far
    vol_floor: float = 0.05      # guards the divide when the book goes quiet
    safe_asset: str = SAFE_ASSET  # unallocated weight parks here


@dataclass
class BreakoutParams:
    """Hourly resistance-breakout trading, from M6_finalnotebook.ipynb.

    The notebook's rules, as written there:

        Entry   an hourly close crosses above the nearest resistance level,
                and price is STILL above that level `confirm_hours` later
        Exit    below entry - R                              -> stop
                between entry +/- R, held 3 weeks or closing
                  below the fast EMA minus one ADR           -> time/trend exit
                above entry + R, closing below EMA - ADR     -> take profit

    Three departures from the notebook's code, all removing look-ahead. Each
    is switchable so you can measure what it was worth:

    * `causal_levels` -- the notebook runs `find_peaks` over the WHOLE daily
      history once, then replays trades across that same history, so a trade
      is placed at a level defined by pivots that had not happened yet. On a
      2.5-year sample a trade a quarter of the way in draws three quarters of
      its levels from its own future. Set False to reproduce the notebook.
    * `causal_risk` -- likewise for Var95, which the notebook takes over the
      full return history and then uses to size R on every trade.
    * `lag_daily_indicators` -- the notebook resamples hourly to daily and
      forward-fills onto the hourly index, so the 09:30 bar of day D already
      carries the EMA, ATR and ADR computed from day D's CLOSE. Shifting one
      day makes the regime gate and the EMA exit use only completed days.

    Two things in the notebook are inert and are not reproduced: `rr_takeprofit`
    (a 2R target that `generate_trades` computes and never reads) and
    `require_retest`. The markdown also says "10-day sma" where the code uses
    the fast EMA minus one ADR; the code is what is implemented here.
    """
    # ---- level extraction (on daily bars) -------------------------------
    swing_lookback: int = 1           # find_peaks `distance`
    prominence_frac: float = 0.015    # prominence as a fraction of last close
    atr_window: int = 14
    atr_merge_mult: float = 1.0       # first merge pass, within 1 ATR
    second_merge_mult: float = 1.3    # the notebook's repeated merge, 1.3 ATR
    max_levels: int = 20

    # ---- entry ----------------------------------------------------------
    confirm_hours: int = 2            # bars after the cross that must hold
    ema_fast_days: int = 10
    ema_slow_days: int = 20
    adr_window: int = 14

    # ---- risk unit R ----------------------------------------------------
    # R is `min(|entry x Var95|, avg_level_gap/2 + ADR/2)`, then scaled by
    # `r_mult`. The Var95 cap is usually the binding one, and Var95 on a
    # volatile name is a single bad session -- which a breakout routinely
    # gives back before it works. That is the first thing to sweep when the
    # exit mix comes back dominated by `stop_R`.
    var_confidence: float = 0.95      # Var95 of daily returns caps R
    use_var_cap: bool = True          # False -> structural distance alone sets R
    r_mult: float = 1.0               # widen (>1) or tighten (<1) the whole stop
    hold_weeks: float = 3.0           # the time stop on a trade going nowhere

    # ---- honesty switches -----------------------------------------------
    causal_levels: bool = True
    causal_risk: bool = True
    lag_daily_indicators: bool = True


@dataclass
class WeeklyBookParams:
    """The portfolio wrapper: a weekend watchlist traded by breakout, 6 slots.

    Selection happens once a week on the last session of the week. The names
    that pass go onto a ranked watchlist; during the following week a slot is
    taken when one of them triggers a confirmed breakout.

    `refill_within_week = False` is the rule as described: a slot freed by a
    stop-out or a failed breakout holds cash until the next weekend rather
    than being handed to the next name down. That is a real constraint, not a
    detail -- it caps how often the book can be wrong in a week, and it is why
    average exposure sits well below 100%.
    """
    n_slots: int = 6
    watchlist_size: int = 20          # ranked names carried into the week
    refill_within_week: bool = False  # a freed slot waits for the weekend
    selection_day: str = "W-FRI"      # when the watchlist is rebuilt
    safe_asset: str = SAFE_ASSET      # where an unused slot sits
    equal_weight_slots: bool = True   # 1/n_slots per slot, matching the others

    def __post_init__(self):
        if self.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if self.watchlist_size < self.n_slots:
            raise ValueError("watchlist_size must be >= n_slots")


@dataclass
class Config:
    tickers: List[str] = field(default_factory=lambda: list(TICKERS))
    backtest_start: str = BACKTEST_START
    backtest_end: str | None = BACKTEST_END
    download_start: str = DOWNLOAD_START
    execution_lag: int = EXECUTION_LAG
    cost_bps: float = COST_BPS
    slippage_bps: float = 5.0     # decide at 15:30, fill at the 16:00 close
    rsi2: RSI2Params = field(default_factory=RSI2Params)
    gem: GEMParams = field(default_factory=GEMParams)
    vol: VolTargetParams = field(default_factory=VolTargetParams)
    momentum: MomentumParams = field(default_factory=MomentumParams)
    resmom: ResidualMomentumParams = field(default_factory=ResidualMomentumParams)
    finviz: FinvizScreenParams = field(default_factory=FinvizScreenParams)
    breakout: BreakoutParams = field(default_factory=BreakoutParams)
    weekly_book: WeeklyBookParams = field(default_factory=WeeklyBookParams)
    vix: VixBreakerParams = field(default_factory=VixBreakerParams)
    vix_ts: VixTermStructureParams = field(default_factory=VixTermStructureParams)
    book_vol: BookVolTargetParams = field(default_factory=BookVolTargetParams)
    dd_stop: DrawdownStopParams = field(default_factory=DrawdownStopParams)
    dd_stop_benchmark: str = RISK_ASSET   # the series dd_stop.qqq_drawdown reads
    # Restrict the ranker to the dashboard's momentum-leader set before it
    # picks the top N -- close over $5, over 300k shares a day, up more than
    # 28% on the quarter (the thresholds live in `breadth.BreadthParams`, so
    # there is one definition and the dashboard and the ranker cannot drift).
    #
    # Off, and measured harmful: on 2020-2026 it takes CAGR from 22.9% to 8.1%
    # and max drawdown from -17.6% to -30.4%, at 26x annual turnover against
    # 11x. The leader set is not stable enough to hold a book -- a name crosses
    # the quarterly threshold in and out constantly, and since the mask forces
    # an exit the whole risk sleeve churns. Every threshold from 0% to 28% was
    # worse than no filter, in every window tested.
    use_leader_filter: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Chart palette
# --------------------------------------------------------------------------
# Three categorical slots, validated for colour-vision deficiency in
# all-pairs mode. Benchmarks deliberately sit outside the categorical set so
# they never compete with a strategy for identity.

PALETTE = {
    "rsi2":      "#2a78d6",   # slot 1 -- blue
    "gem":       "#eb6834",   # slot 2 -- orange
    "voltarget": "#1baf7a",   # slot 3 -- aqua
    "momentum":  "#eda100",   # slot 4 -- yellow (low contrast: always direct-labelled)
    "momentum_vix": "#e87ba4",  # slot 5 -- magenta
    "momentum_vt": "#8a63d2",   # slot 6 -- violet
    # Slot 7 sits OUTSIDE the validated six-hue set above: the categorical
    # hues are used up, so this one separates by lightness instead.
    "finviz":    "#3d4f5c",   # slot 7 -- dark slate
    "breakout":  "#a8572c",   # slot 8 -- burnt umber, also outside the set
    # Slot 9 is also outside the validated six-hue set: it separates from the
    # momentum yellow by lightness, which is what the eye uses when the
    # categorical hues are spent.
    "resmom":    "#00696e",   # slot 9 -- deep teal
    # Slot 10 sits next to the level breaker's magenta on purpose: the two are
    # the same switch on different signals, and the chart should say so.
    "momentum_ts": "#9c3d6b",  # slot 10 -- deep magenta
    "bh_qqq":    "#898781",   # benchmark -- muted
    "bh_boxx":   "#c3c2b7",   # benchmark -- fainter still
    "buy":       "#0ca30c",   # status: good
    "sell":      "#d03b3b",   # status: critical
    "ink":       "#0b0b0b",
    "ink_2":     "#52514e",
    "muted":     "#898781",
    "grid":      "#e1e0d9",
    "axis":      "#c3c2b7",
    "surface":   "#fcfcfb",
    "shade_risk": "#cde2fb",  # regime shading -- risk-on
    "shade_safe": "#f0efec",  # regime shading -- risk-off
}

def notebook_screen_params(**kw) -> "FinvizScreenParams":
    """The ORIGINAL Finviz-notebook filter set, which is no longer the default.

    `FinvizScreenParams()` now applies the high-momentum definition -- over $5,
    over 300k shares, up more than 28% on the quarter -- and leaves the
    notebook's 200-day and 52-week-high filters off. That is the right default
    for the screen, and the wrong one for anything built against the old rules.

    The breakout research is exactly that: its watchlist is this screen's
    output, and its entry logic assumes names selected NEAR their highs, with
    `min_off_high_pct` turning that ceiling into a band. Handing it the new
    defaults would change a documented result without changing a line of its
    own code, so it asks for this instead.
    """
    for key, value in (("min_price", 10.0), ("above_sma", 200),
                       ("within_52w_high_pct", 0.10),
                       ("require_quarter_up", True),
                       ("min_quarter_return", None),
                       ("min_volume", None),
                       ("min_avg_volume", 200_000.0)):
        kw.setdefault(key, value)
    return FinvizScreenParams(**kw)


STRATEGY_LABELS = {
    "rsi2": "Connors RSI(2)",
    "gem": "GEM dual momentum",
    "voltarget": "Vol-targeted QQQ",
    "momentum": "Top-6 NDX momentum",
    "momentum_vix": "Top-6 + VIX breaker",
    "momentum_vt": "Top-6 vol-targeted",
    "finviz": "Top-20 high momentum screen",
    "breakout": "Weekly breakout, 6 slots",
    "resmom": "Top-6 residual momentum",
    "momentum_ts": "Top-6 + VIX term structure",
    "bh_qqq": "Buy & hold QQQ",
    "bh_boxx": "Buy & hold BOXX",
}
