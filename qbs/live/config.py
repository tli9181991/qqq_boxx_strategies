"""Live trading configuration.

Separate from `qbs.config` on purpose. That file holds *strategy* parameters,
which must stay identical between the backtest and the live book or the two
stop being comparable. This file holds *deployment* parameters -- account,
connection, sizing, safety limits -- which have no meaning in a backtest.

Everything here can be overridden by an environment variable so the VM can be
configured without editing tracked files. Secrets never belong in this file:
IB Gateway credentials live in the Gateway's own config, not here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from ..config import SAFE_ASSET

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STATE_DIR = os.environ.get("QBS_STATE_DIR", os.path.join(REPO_ROOT, "var"))


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v in (None, ""):
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LiveConfig:
    """Deployment settings for the Top-6 vol-targeted book.

    Sizing
    ------
    `notional` is a fixed dollar figure, not the account's net liquidation
    value. That is a deliberate choice: it makes the position sizes
    reproducible run to run, so a difference between two days' order lists is
    always a *signal* change and never an account-value change. The trade-off
    is that it does not compound -- as the paper account gains or loses, the
    book stays the same size until you edit this number.

    Safety
    ------
    The guards exist because this runs unattended. Every one of them halts the
    run rather than trading through a condition it does not understand:
    a half-downloaded universe, a stale price file, or an order list that
    implies churning the whole book in a single session are all far more
    likely to be a data bug than a real signal.
    """

    # ---- connection ------------------------------------------------------
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002          # 4002 = Gateway paper, 4001 = Gateway live,
                                 # 7497 = TWS paper, 7496 = TWS live
    ib_client_id: int = 17
    ib_account: str = ""         # blank -> whichever account the Gateway serves
    connect_timeout: float = 30.0
    order_timeout: float = 120.0  # how long to wait for MOC acknowledgement

    # ---- sizing ----------------------------------------------------------
    notional: float = 100_000.0
    hold_safe_asset: bool = True   # False -> leave the cash leg as cash, not BOXX
    safe_asset: str = SAFE_ASSET

    # ---- what counts as worth trading ------------------------------------
    min_order_shares: int = 1
    min_order_notional: float = 250.0   # skip dust rebalances; they cost more than they fix

    # ---- safety guards ---------------------------------------------------
    max_order_notional: float = 40_000.0   # per single order
    max_gross_turnover: float = 1.60       # abort if one session would trade >160% of notional
    max_positions: int = 12                # sanity bound on the book's width
    moc_cutoff_hhmm: str = "15:45"         # refuse to submit MOC after this exchange-local time
    max_price_staleness_days: int = 5      # last close must be this recent (covers a long weekend)
    min_universe_coverage: float = 0.85    # fraction of NDX tickers that must have downloaded
    allow_live_account: bool = False       # refuses to run against a non-paper port unless set

    # ---- operational -----------------------------------------------------
    state_dir: str = DEFAULT_STATE_DIR
    kill_switch: str = ""        # if this path exists, every phase refuses to trade
    dry_run: bool = False        # log the orders, send nothing
    market_tz: str = "America/New_York"
    extra_tickers: List[str] = field(default_factory=list)  # always download these too

    def __post_init__(self):
        if not self.kill_switch:
            self.kill_switch = os.path.join(self.state_dir, "HALT")
        if self.notional <= 0:
            raise ValueError("notional must be positive")
        if self.max_gross_turnover <= 0:
            raise ValueError("max_gross_turnover must be positive")
        if not 0.0 < self.min_universe_coverage <= 1.0:
            raise ValueError("min_universe_coverage must be in (0, 1]")

    # ---- derived ---------------------------------------------------------
    @property
    def is_paper_port(self) -> bool:
        """IB's paper ports. 4001/7496 are the live ones and are refused by default."""
        return self.ib_port in (4002, 7497)

    @property
    def state_path(self) -> str:
        return os.path.join(self.state_dir, "state.json")

    @property
    def orders_log_path(self) -> str:
        return os.path.join(self.state_dir, "orders.csv")

    @property
    def fills_log_path(self) -> str:
        return os.path.join(self.state_dir, "fills.csv")

    # ---- loading ---------------------------------------------------------
    @classmethod
    def from_env(cls, path: Optional[str] = None) -> "LiveConfig":
        """Build from an optional JSON file, then let QBS_* env vars win.

        The JSON file is for settings you want in version control on the VM;
        the environment is for anything host-specific. Env always wins so a
        systemd drop-in can override one value without rewriting the file.
        """
        data = {}
        path = path or os.environ.get("QBS_LIVE_CONFIG")
        if path and os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")

        cfg = cls(**data)
        cfg.ib_host = os.environ.get("QBS_IB_HOST", cfg.ib_host)
        cfg.ib_port = _env_int("QBS_IB_PORT", cfg.ib_port)
        cfg.ib_client_id = _env_int("QBS_IB_CLIENT_ID", cfg.ib_client_id)
        cfg.ib_account = os.environ.get("QBS_IB_ACCOUNT", cfg.ib_account)
        cfg.notional = _env_float("QBS_NOTIONAL", cfg.notional)
        cfg.state_dir = os.environ.get("QBS_STATE_DIR", cfg.state_dir)
        cfg.dry_run = _env_bool("QBS_DRY_RUN", cfg.dry_run)
        cfg.allow_live_account = _env_bool("QBS_ALLOW_LIVE", cfg.allow_live_account)
        cfg.hold_safe_asset = _env_bool("QBS_HOLD_SAFE_ASSET", cfg.hold_safe_asset)
        cfg.__post_init__()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def redacted(self) -> dict:
        """Safe to write to a log: no account number."""
        d = self.to_dict()
        if d.get("ib_account"):
            d["ib_account"] = d["ib_account"][:4] + "***"
        return d
