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
    lookback_months: int = 12
    skip_months: int = 1          # 12-1 momentum: skip the most recent month
    n_hold: int = 6
    exit_rank: int = 10           # hysteresis band; == n_hold means no band
    rebalance: str = "daily"      # "daily" | "ME" (month end) | "W-FRI"
    weighting: str = "equal"      # "equal" | "inv_vol"
    absolute_filter: bool = True  # a name must also beat the safe asset
    safe_asset: str = SAFE_ASSET
    vol_lookback: int = 60        # for inv_vol weighting
    min_history: int = 260        # trading days a name needs before it is rankable

    def __post_init__(self):
        if self.exit_rank < self.n_hold:
            raise ValueError("exit_rank must be >= n_hold (the band cannot be negative)")


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
    vix: VixBreakerParams = field(default_factory=VixBreakerParams)
    book_vol: BookVolTargetParams = field(default_factory=BookVolTargetParams)

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

STRATEGY_LABELS = {
    "rsi2": "Connors RSI(2)",
    "gem": "GEM dual momentum",
    "voltarget": "Vol-targeted QQQ",
    "momentum": "Top-6 NDX momentum",
    "momentum_vix": "Top-6 + VIX breaker",
    "momentum_vt": "Top-6 vol-targeted",
    "bh_qqq": "Buy & hold QQQ",
    "bh_boxx": "Buy & hold BOXX",
}
