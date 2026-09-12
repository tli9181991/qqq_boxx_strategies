"""Turning target weights into orders, and refusing to when something looks wrong.

Deliberately pure: no IB, no network, no clock. Everything here is a function
from (targets, positions, prices, config) to an order list or an exception,
which is what makes the risky part of an unattended trading loop testable.

The order of operations matters:

    target weights -> target shares -> deltas vs actual -> filter dust -> guards

Guards run *last*, on the final order list, because that is the only point
where "how much is this session about to trade" is actually known.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Order:
    symbol: str
    action: str          # "BUY" | "SELL"
    quantity: int        # always positive
    price_hint: float    # the price the sizing was based on -- for logging, not for the order
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.quantity * self.price_hint

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.action == "BUY" else -self.quantity

    def __str__(self) -> str:
        return (f"{self.action:4} {self.quantity:>6} {self.symbol:<6} "
                f"~${self.notional:>10,.0f}  {self.reason}")


class GuardTripped(RuntimeError):
    """A safety limit refused the order list. Nothing has been sent."""


def target_shares(
    weights: Dict[str, float],
    prices: Dict[str, float],
    notional: float,
) -> Dict[str, int]:
    """Weights -> whole share counts, rounded down.

    Rounding down rather than to nearest keeps the book from creeping above
    `notional` through accumulated rounding across six positions plus the cash
    leg. The dropped fraction lands in cash, which is the safe direction.
    """
    if notional <= 0:
        raise ValueError("notional must be positive")

    out: Dict[str, int] = {}
    for sym, w in weights.items():
        px = prices.get(sym)
        if px is None or px <= 0:
            raise ValueError(f"no usable price for {sym} (got {px!r})")
        out[sym] = int(math.floor(w * notional / px))
    return out


def diff_positions(
    target: Dict[str, int],
    actual: Dict[str, int],
    prices: Dict[str, float],
    min_shares: int = 1,
    min_notional: float = 0.0,
) -> List[Order]:
    """Deltas between what we want and what we hold, as an order list.

    Symbols held but no longer wanted are closed in full. Symbols whose delta
    is too small to be worth a spread are skipped -- a two-share rebalance on
    a $200 stock costs more in commission and slippage than the tracking error
    it removes.

    Sells are emitted before buys so that, in a fully-invested book, the cash
    from an exit is available for the entry that replaces it.
    """
    orders: List[Order] = []
    for sym in sorted(set(target) | set(actual)):
        want = int(target.get(sym, 0))
        have = int(actual.get(sym, 0))
        delta = want - have
        if delta == 0:
            continue

        px = prices.get(sym)
        if px is None or px <= 0:
            if want == 0:
                # Closing a position we have no price for is still safe to do;
                # size the notional as unknown rather than blocking the exit.
                px = 0.0
            else:
                raise ValueError(f"no usable price for {sym} (got {px!r})")

        if abs(delta) < min_shares:
            log.info("skip %s: delta %+d below min_shares %d", sym, delta, min_shares)
            continue
        if px > 0 and abs(delta) * px < min_notional and want != 0:
            log.info("skip %s: delta %+d (~$%.0f) below min_notional $%.0f",
                     sym, delta, abs(delta) * px, min_notional)
            continue

        if want == 0:
            reason = "exit"
        elif have == 0:
            reason = "entry"
        else:
            reason = f"rebalance {have} -> {want}"

        orders.append(Order(
            symbol=sym,
            action="BUY" if delta > 0 else "SELL",
            quantity=abs(delta),
            price_hint=px,
            reason=reason,
        ))

    orders.sort(key=lambda o: (o.action != "SELL", o.symbol))
    return orders


def apply_guards(
    orders: List[Order],
    notional: float,
    target: Dict[str, int],
    max_order_notional: float,
    max_gross_turnover: float,
    max_positions: int,
    unknown_positions: Optional[Dict[str, int]] = None,
    safe_asset: str = "BOXX",
) -> List[Order]:
    """Refuse the whole list if any limit is breached. Returns it unchanged if not.

    All-or-nothing on purpose. Sending the orders that happen to pass while
    dropping the one that tripped a limit leaves the book in a state no part
    of the system intended -- half-rebalanced, with the strategy believing it
    is fully positioned. Better to send nothing and raise loudly.

    The safe asset is exempt from `max_order_notional` and capped at the book
    size instead. That cap exists to catch a runaway order in a single *risk*
    name; the cash leg is a different thing entirely, and legitimately runs to
    the whole book -- when the vol scalar cuts equity exposure to 36%, BOXX is
    64% of notional by construction. Applying the per-name cap to it would trip
    the guard on a completely normal session.
    """
    if not orders:
        return orders

    too_big = [o for o in orders
               if o.symbol != safe_asset and o.notional > max_order_notional]
    if too_big:
        raise GuardTripped(
            "single order exceeds max_order_notional "
            f"${max_order_notional:,.0f}: " +
            ", ".join(f"{o.symbol} ~${o.notional:,.0f}" for o in too_big))

    safe_leg = [o for o in orders
                if o.symbol == safe_asset and o.notional > notional * 1.01]
    if safe_leg:
        raise GuardTripped(
            f"{safe_asset} order of ~${safe_leg[0].notional:,.0f} exceeds the "
            f"${notional:,.0f} book size -- the cash leg cannot be larger than "
            "the book, so this is a sizing bug")

    gross = sum(o.notional for o in orders)
    if gross > max_gross_turnover * notional:
        raise GuardTripped(
            f"session would trade ${gross:,.0f} against a ${notional:,.0f} book "
            f"({gross / notional:.0%}), above the {max_gross_turnover:.0%} limit. "
            "This is usually a data bug, not a signal -- check the universe download.")

    n_target = sum(1 for q in target.values() if q > 0)
    if n_target > max_positions:
        raise GuardTripped(
            f"target book has {n_target} positions, above the {max_positions} limit")

    if unknown_positions:
        log.warning("positions held outside the strategy universe, left untouched: %s",
                    ", ".join(f"{k}={v}" for k, v in sorted(unknown_positions.items())))

    return orders


def build_orders(
    weights: Dict[str, float],
    prices: Dict[str, float],
    actual: Dict[str, int],
    notional: float,
    max_order_notional: float,
    max_gross_turnover: float,
    max_positions: int,
    min_shares: int = 1,
    min_notional: float = 0.0,
    hold_safe_asset: bool = True,
    safe_asset: str = "BOXX",
    universe: Optional[Iterable[str]] = None,
) -> tuple[List[Order], Dict[str, int]]:
    """The whole path: weights -> orders, guards applied. Returns (orders, targets).

    With `hold_safe_asset=False` the cash leg is simply left as cash: the safe
    asset is dropped from the target and any existing position in it is closed.

    `universe` is every name the strategy may trade. It is what separates a
    position the strategy has just exited from one that was never its business,
    and both look identical from `weights` alone -- an exited name has weight
    zero, and zero weights are not carried. Without it, a held name missing
    from today's targets is read as somebody else's holding and left alone, so
    the book buys its replacements while never selling its exits. Pass it.
    """
    weights = dict(weights)
    if not hold_safe_asset:
        weights.pop(safe_asset, None)

    target = target_shares(weights, prices, notional)
    if not hold_safe_asset and safe_asset in actual:
        target[safe_asset] = 0

    strategy_syms = set(target) | set(weights) | set(universe or ())
    unknown = {k: v for k, v in actual.items() if k not in strategy_syms and v != 0}

    orders = diff_positions(target, {k: v for k, v in actual.items()
                                     if k in strategy_syms},
                            prices, min_shares=min_shares, min_notional=min_notional)
    orders = apply_guards(orders, notional, target, max_order_notional,
                          max_gross_turnover, max_positions,
                          unknown_positions=unknown, safe_asset=safe_asset)
    return orders, target


def format_order_table(orders: List[Order], target: Dict[str, int],
                       actual: Dict[str, int]) -> str:
    """A block suitable for the run log -- what changed and what the book becomes."""
    if not orders:
        return "no orders: the live book already matches the target"

    lines = [f"{'ACTION':<6} {'QTY':>7} {'SYMBOL':<7} {'~NOTIONAL':>12}  REASON",
             "-" * 62]
    for o in orders:
        lines.append(f"{o.action:<6} {o.quantity:>7} {o.symbol:<7} "
                     f"{o.notional:>12,.0f}  {o.reason}")
    gross = sum(o.notional for o in orders)
    lines.append("-" * 62)
    lines.append(f"{len(orders)} orders, gross ~${gross:,.0f}")
    lines.append("resulting book: " + ", ".join(
        f"{k}={v}" for k, v in sorted(target.items()) if v > 0) or "(all cash)")
    return "\n".join(lines)
