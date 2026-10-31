"""Cron expressions: when a schedule fires next.

Five fields - minute hour day-of-month month day-of-week - each `*`, a number, a range `a-b`, a step `*/n` or
`a-b/n`, or a list of those; months and days by name too (JAN-DEC, SUN-SAT; 0 and 7 are Sunday). As in cron,
when both day fields are restricted a day matches either. Shorthands: @yearly @monthly @weekly @daily @hourly.
Times are in the schedule's time zone (UTC by default), so "0 7 * * MON-FRI" is 07:00 there, summer or not.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

SHORTHANDS = {"@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *", "@weekly": "0 0 * * 0",
              "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@hourly": "0 * * * *"}
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                      "DEC"], start=1)}
DAYS = {d: i for i, d in enumerate(["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"])}
FIELDS = (("minute", 0, 59, {}), ("hour", 0, 23, {}), ("day", 1, 31, {}), ("month", 1, 12, MONTHS),
          ("weekday", 0, 7, DAYS))


class CronError(ValueError):
    pass


def _field(text: str, name: str, low: int, high: int, names: dict[str, int]) -> tuple[set[int], bool]:
    """The values a field allows, and whether it restricts anything."""
    values: set[int] = set()
    for item in text.split(","):
        base, _, step_text = item.partition("/")
        step = int(step_text) if step_text else 1
        if step < 1:
            raise CronError(f"{name}: step must be at least 1")
        if base == "*":
            start, end = low, high
        else:
            start_text, _, end_text = base.partition("-")
            start = _value(start_text, name, names)
            end = _value(end_text, name, names) if end_text else (high if step_text else start)
        if not (low <= start <= high and low <= end <= high) or start > end:
            raise CronError(f"{name}: {item!r} is outside {low}-{high}")
        values.update(range(start, end + 1, step))
    if name == "weekday" and 7 in values:  # 7 is Sunday too
        values.discard(7)
        values.add(0)
    return values, text != "*"


def _value(text: str, name: str, names: dict[str, int]) -> int:
    if text.upper() in names:
        return names[text.upper()]
    try:
        return int(text)
    except ValueError:
        raise CronError(f"{name}: {text!r} is not a number{' or a name' if names else ''}") from None


class Cron:
    def __init__(self, expression: str, tz: str = "UTC"):
        text = SHORTHANDS.get(expression.strip().lower(), expression.strip())
        parts = text.split()
        if len(parts) != 5:
            raise CronError(f"{expression!r}: five fields (minute hour day month weekday), or a @shorthand")
        (self.minutes, _), (self.hours, _), (self.days, self.day_set), (self.months, _), (self.weekdays, self.wd_set) = (
            _field(p, name, low, high, names) for p, (name, low, high, names) in zip(parts, FIELDS))
        self.expression = expression
        try:
            self.tz = ZoneInfo(tz)
        except Exception as e:  # noqa: BLE001
            raise CronError(f"unknown time zone {tz!r}") from e

    def _day_ok(self, day: datetime) -> bool:
        dom, dow = day.day in self.days, (day.isoweekday() % 7) in self.weekdays
        if self.day_set and self.wd_set:
            return dom or dow
        return dom and dow

    def next(self, after: datetime) -> datetime:
        """The first time after `after` (aware, any zone) that matches: in UTC."""
        t = after.astimezone(self.tz).replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(366 * 5):  # days to look at: a schedule that never fires is an error
            if t.month in self.months and self._day_ok(t):
                for hour in sorted(h for h in self.hours if h >= t.hour):
                    minutes = [m for m in sorted(self.minutes) if hour > t.hour or m >= t.minute]
                    if minutes:
                        found = t.replace(hour=hour, minute=minutes[0])
                        # a wall-clock time that doesn't exist (a spring-forward gap) is taken as it lands in UTC
                        return found.astimezone(timezone.utc)
            t = (t + timedelta(days=1)).replace(hour=0, minute=0)
        raise CronError(f"{self.expression!r} never fires")


def parse_every(text: str) -> timedelta:
    """`30s`, `15m`, `2h`, `1d` (or a number of seconds)."""
    text = str(text).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        seconds = float(text[:-1]) * units[text[-1]] if text[-1] in units else float(text)
    except (ValueError, IndexError):
        raise CronError(f"every: {text!r} - like 30s, 15m, 2h, 1d") from None
    if seconds <= 0:
        raise CronError("every: must be more than 0")
    return timedelta(seconds=seconds)
