from datetime import datetime, timezone

import pytest

from llamphouse.runtime.triggers.cron import Cron, CronError, parse_every

UTC = timezone.utc


def at(*a):
    return datetime(*a, tzinfo=UTC)


@pytest.mark.parametrize("expr, after, expected", [
    ("*/15 * * * *", at(2026, 10, 31, 10, 7), at(2026, 10, 31, 10, 15)),
    ("0 7 * * MON-FRI", at(2026, 10, 31, 10, 0), at(2026, 11, 2, 7, 0)),  # a Saturday: Monday 07:00
    ("0 0 1 * *", at(2026, 12, 15), at(2027, 1, 1)),
    ("30 9 * * 7", at(2026, 10, 31, 12, 0), at(2026, 11, 1, 9, 30)),  # 7 is Sunday
    ("@hourly", at(2026, 10, 31, 10, 59, 30), at(2026, 10, 31, 11, 0)),
    ("0 12 13 * FRI", at(2026, 11, 1), at(2026, 11, 6, 12, 0)),  # both day fields: either matches
    ("0 0 29 FEB *", at(2026, 3, 1), at(2028, 2, 29)),
])
def test_next(expr, after, expected):
    assert Cron(expr).next(after) == expected


def test_time_zones_follow_the_wall_clock():
    winter = Cron("0 7 * * *", "Europe/Brussels").next(at(2026, 1, 10, 12, 0))
    summer = Cron("0 7 * * *", "Europe/Brussels").next(at(2026, 7, 10, 12, 0))
    assert winter.hour == 6 and summer.hour == 5  # 07:00 there, in UTC


@pytest.mark.parametrize("bad", ["* * * *", "61 * * * *", "* * * * MOX", "*/0 * * * *", "0 0 31 2 *"])
def test_mistakes(bad):
    with pytest.raises(CronError):
        Cron(bad).next(at(2026, 1, 1))


def test_every():
    assert parse_every("15m").total_seconds() == 900 and parse_every("2h").total_seconds() == 7200
    with pytest.raises(CronError):
        parse_every("soon")
