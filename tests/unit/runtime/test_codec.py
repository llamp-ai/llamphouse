"""The codec: what the engine keeps, as JSON and back - the same types."""

import dataclasses
import json
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from llamphouse.runtime.codec import decode, encode
from llamphouse.runtime.engine import RunRecord, SessionRecord
from llamphouse.sdk import (ArtifactPart, DataPart, EventKind, Filter, InputRequest, Message, Origin, RunRef,
                            RunResult, RunStatus, Sender, SessionView, StateConflict, TextPart, ThreadEvent,
                            ThreadRef, ToolCallPart)


class Price(BaseModel):
    city: str
    per_day: int


def round_trip(value):
    return decode(json.loads(json.dumps(encode(value))))


def test_plain_values_tuples_sets_and_odd_dicts():
    value = {"a": [1, 2.5, None, True], "t": (1, "x"), "s": {1, 2}, "b": b"\x00\x01",
             "at": datetime(2026, 1, 1, tzinfo=timezone.utc), "d": timedelta(seconds=90),
             "keys": {(1, 2): "tuple key", "$x": "dollar"}}
    assert round_trip(value) == value


def test_sdk_types_come_back_as_themselves():
    message = Message.create(Sender.person("an"), "hi", DataPart(data={"k": 1}),
                             ArtifactPart(name="a", parts=[TextPart(text="x")]), ToolCallPart(call_id="c", name="n", arguments={}))
    event = ThreadEvent(position=3, kind=EventKind.MESSAGE, run=RunRef("r1"), message=message, thread=ThreadRef("t1"))
    f = Filter(kinds=[EventKind.MESSAGE], origin=Origin.INSIDE, thread=ThreadRef("t2"))
    result = RunResult(id="r2", agent="a", status=RunStatus.COMPLETED, output=Price(city="rome", per_day=1))
    for value in (message, event, f, result, InputRequest(reason="month?"), RunStatus.WAITING,
                  SessionView([message], id="s1", version=2, revising=True)):
        back = round_trip(value)
        assert type(back) is type(value)
        assert back == value or (isinstance(value, SessionView) and list(back) == list(value) and back.id == "s1")
    assert {f: 1}[round_trip(f)] == 1  # still hashable: a subscription key


def test_exceptions_and_the_engines_records():
    conflict = round_trip(StateConflict("k", 1, 2))
    assert isinstance(conflict, StateConflict) and (conflict.key, conflict.expected, conflict.actual) == ("k", 1, 2)
    rec = RunRecord(id="r1", thread_id="t1", agent="a", input=Message.create(Sender.person("an"), "go"))
    rec.steps["search#1"] = Price(city="rome", per_day=160)
    rec.subs[(Filter(kinds=[EventKind.MESSAGE]),)] = None
    back = round_trip(rec)
    assert back.steps == rec.steps and back.input == rec.input and list(back.subs) == list(rec.subs)
    assert round_trip(SessionRecord(id="s1", thread_id="t1", agent="a")).id == "s1"


@dataclasses.dataclass
class Grown:
    a: int
    b: list = dataclasses.field(default_factory=list)
    c: str = "new"


def test_a_field_added_since_a_value_was_stored_reads_back_as_its_default():
    stored = {"$dc": f"{__name__}:Grown", "fields": {"a": 1, "gone": 2}}
    assert decode(stored) == Grown(a=1, b=[], c="new")
