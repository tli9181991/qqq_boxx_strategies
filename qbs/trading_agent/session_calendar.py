"""US regular sessions: holidays, early closes, DST, and the analysis slots.

Why rules and not a calendar package
------------------------------------
`qbs.live.runner.us_early_close` already makes the argument for this repo:
the NYSE schedule is a handful of rules derivable from the date alone, so
rules need neither a dependency on a small box nor a list that silently
expires on December 31st. The full-day holidays are the same kind of rule.
Unscheduled closures (a national day of mourning, a hurricane) are not
derivable, so `extra_holidays` / `extra_early_closes` take them as dates --
and on any day the market did not trade the data check fails anyway, so the
worst case is a skipped cycle logged as stale, never an invented decision.

Daylight saving is not handled here at all, on purpose: every time is built
in America/New_York with zoneinfo, so 09:30 ET is 13:30 UTC in summer and
14:30 UTC in winter without a line of code knowing which.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

MARKET_TZ = "America/New_York"
ET = ZoneInfo(MARKET_TZ)
OPEN = time(9, 30)
CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
CANDLE = timedelta(minutes=15)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (the anonymous algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> Optional[date]:
    """NYSE observance: Saturday -> Friday, Sunday -> Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> List[date]:
    """Full-day NYSE closures for `year`, by the standing rules."""
    out = []
    ny = date(year, 1, 1)
    # A Saturday New Year is NOT observed on the Friday before (NYSE rule 7.2):
    # that Friday is the last session of the previous year.
    if ny.weekday() != 5:
        out.append(_observed(ny))
    out += [
        _nth_weekday(year, 1, 0, 3),               # Martin Luther King Jr.
        _nth_weekday(year, 2, 0, 3),               # Washington's Birthday
        _easter(year) - timedelta(days=2),         # Good Friday
        _last_weekday(year, 5, 0),                 # Memorial Day
    ]
    if year >= 2022:
        out.append(_observed(date(year, 6, 19)))   # Juneteenth
    out += [
        _observed(date(year, 7, 4)),               # Independence Day
        _nth_weekday(year, 9, 0, 1),               # Labor Day
        _nth_weekday(year, 11, 3, 4),              # Thanksgiving
        _observed(date(year, 12, 25)),             # Christmas
    ]
    return sorted(out)


def _parse_dates(values: Iterable[str]) -> set:
    out = set()
    for v in values or ():
        try:
            out.add(date.fromisoformat(str(v).strip()))
        except ValueError:
            continue
    return out


def is_trading_day(d: date, extra_holidays: Iterable[str] = ()) -> bool:
    if d.weekday() >= 5:
        return False
    if d in nyse_holidays(d.year) or d in _parse_dates(extra_holidays):
        return False
    return True


def is_early_close(d: date, extra_early_closes: Iterable[str] = ()) -> bool:
    """13:00 ET close: July 3rd and Christmas Eve when Mon-Thu, and the day
    after Thanksgiving -- the rules `qbs.live.runner.us_early_close` uses."""
    if d in _parse_dates(extra_early_closes):
        return True
    if d.month == 7 and d.day == 3 and d.weekday() <= 3:
        return True
    if d.month == 12 and d.day == 24 and d.weekday() <= 3:
        return True
    return d == _nth_weekday(d.year, 11, 3, 4) + timedelta(days=1)


def session_bounds(d: date, extra_holidays: Iterable[str] = (),
                   extra_early_closes: Iterable[str] = ()
                   ) -> Optional[Tuple[datetime, datetime]]:
    """`(open, close)` tz-aware in ET, or None when the market is shut."""
    if not is_trading_day(d, extra_holidays):
        return None
    close = EARLY_CLOSE if is_early_close(d, extra_early_closes) else CLOSE
    return (datetime.combine(d, OPEN, tzinfo=ET),
            datetime.combine(d, close, tzinfo=ET))


def to_et(when: datetime) -> datetime:
    if when.tzinfo is None:
        raise ValueError("naive datetime: every time here must be tz-aware")
    return when.astimezone(ET)


def analysis_slots(d: date, interval_minutes: int = 15,
                   include_close: bool = False,
                   extra_holidays: Iterable[str] = (),
                   extra_early_closes: Iterable[str] = ()) -> List[datetime]:
    """End times of the completed candles to analyse on `d`, in ET.

    A slot IS a candle end: the 10:15 slot analyses the 15-minute candle that
    ran 10:00-10:15, and runs only once that candle is complete.
    """
    bounds = session_bounds(d, extra_holidays, extra_early_closes)
    if bounds is None:
        return []
    open_, close = bounds
    step = timedelta(minutes=interval_minutes)
    out, t = [], open_ + step
    while t < close or (include_close and t == close):
        out.append(t)
        t += step
    return out


def due_slot(now: datetime, interval_minutes: int = 15,
             delay_seconds: int = 0, include_close: bool = False,
             extra_holidays: Iterable[str] = (),
             extra_early_closes: Iterable[str] = ()) -> Optional[datetime]:
    """The newest slot whose candle is complete and whose delay has passed."""
    now = to_et(now)
    lag = timedelta(seconds=delay_seconds)
    due = [s for s in analysis_slots(now.date(), interval_minutes, include_close,
                                     extra_holidays, extra_early_closes)
           if s + lag <= now]
    return due[-1] if due else None


def next_slot(now: datetime, interval_minutes: int = 15,
              delay_seconds: int = 0, include_close: bool = False,
              extra_holidays: Iterable[str] = (),
              extra_early_closes: Iterable[str] = ()) -> Optional[datetime]:
    """The next slot today whose run time (slot + delay) is after `now`."""
    now = to_et(now)
    lag = timedelta(seconds=delay_seconds)
    for s in analysis_slots(now.date(), interval_minutes, include_close,
                            extra_holidays, extra_early_closes):
        if s + lag > now:
            return s
    return None


def session_date(now: Optional[datetime] = None) -> date:
    """Today's date on the exchange's calendar, wherever this runs."""
    now = now or datetime.now(ET)
    return to_et(now).date()
