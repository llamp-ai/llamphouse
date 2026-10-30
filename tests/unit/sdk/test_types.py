"""The contract's types: open kinds, typed references, the wire format."""

import pytest

from llamphouse.sdk import (
    ArtifactPart,
    Custom,
    ThreadCreated,
    DataPart,
    EventKind,
    FilePart,
    Filter,
    Message,
    MessageCompleted,
    Progress,
    Origin,
    Sender,
    SenderKind,
    Viewpoint,
    RunRef,
    TextPart,
    ThreadEvent,
    ToolCallPart,
    ToolResultPart,
    UnknownEvent,
    UnknownPart,
)
from llamphouse.sdk.wire import event_from_dict, message_from_dict, part_from_dict, to_dict


def test_a_message_from_strings_and_parts():
    m = Message.create(Sender.person("u1"), "Hello ", TextPart(text="world"), DataPart(data={"a": 1}))
    assert m.id.startswith("msg_")
    assert m.text == "Hello world"
    assert m.data == {"a": 1}


def test_sender_kinds_are_open():
    sender = Sender(kind="device", id="sensor-7")  # a newer kind passes through
    assert Message.create(sender, "hi").sender.kind == "device"
    assert SenderKind.PERSON == "person"  # and known ones compare to their wire value


def test_the_viewpoint_depends_on_the_reader():
    """One message on a shared thread reads differently for each agent: the role is derived."""
    from llamphouse.sdk import viewpoint

    by_planner = Message.create(Sender.agent("planner", run=RunRef("r1")), "plan")
    assert viewpoint(by_planner, own={RunRef("r1")}) is Viewpoint.OWN  # assistant, for the planner
    assert viewpoint(by_planner, own={RunRef("r2")}) is Viewpoint.OTHER  # user, for anyone else
    assert viewpoint(Message.create(Sender.person("u1"), "hi"), own={RunRef("r1")}) is Viewpoint.OTHER
    assert viewpoint(Message.create(Sender.system(), "be brief"), own={RunRef("r1")}) is Viewpoint.SYSTEM


def test_the_viewpoint_follows_the_run_not_the_agent_name():
    """A parent and its child of the same agent: each owns its own messages, the other is OTHER."""
    from llamphouse.sdk import viewpoint

    by_parent = Message.create(Sender.agent("planner", run=RunRef("parent")), "split the work")
    by_child = Message.create(Sender.agent("planner", run=RunRef("child")), "part 1 done")
    assert viewpoint(by_parent, own={RunRef("child")}) is Viewpoint.OTHER
    assert viewpoint(by_child, own={RunRef("parent")}) is Viewpoint.OTHER
    assert viewpoint(by_child, own={RunRef("child")}) is Viewpoint.OWN


def test_run_references_are_equal_whatever_their_class():
    class Handle(RunRef):
        pass

    assert Handle("r1") == RunRef("r1")
    assert {Handle("r1"), RunRef("r1")} == {RunRef("r1")}
    assert RunRef("r1") != RunRef("r2")
    assert RunRef("r1") != "r1"  # a reference is not its id string


PERSON, AGENT = Sender.person("u1"), Sender.agent("researcher")


def _message_event(sender, *, to=None, run=None, agent=None):
    return ThreadEvent(position=1, kind=EventKind.MESSAGE, run=run, agent=agent, to=to,
                       message=Message.create(sender, "x"))


@pytest.mark.parametrize("filter_, event, expected", [
    (Filter(senders=[SenderKind.PERSON]), _message_event(PERSON), True),
    (Filter(senders=[SenderKind.PERSON]), _message_event(AGENT), False),
    (Filter(kinds=["run.*"]), ThreadEvent(position=1, kind=EventKind.RUN_COMPLETED), True),
    (Filter(kinds=["x-*"]), ThreadEvent(position=1, kind="x-trip.plan_ready"), True),
    (Filter(kinds=["x-*"]), ThreadEvent(position=1, kind=EventKind.MESSAGE), False),
    (Filter(runs=[RunRef("c1")]), _message_event(AGENT, run=RunRef("c1")), True),
    (Filter(runs=[RunRef("c1")]), _message_event(AGENT, run=RunRef("c2")), False),
    (Filter(agents=["researcher"]), _message_event(AGENT, agent="researcher"), True),
    # an addressed event only matches a filter with that `to`
    (Filter(senders=[SenderKind.PERSON]), _message_event(PERSON, to=RunRef("r1")), False),
    (Filter(to=RunRef("r1")), _message_event(PERSON, to=RunRef("r1")), True),
    (Filter(to=RunRef("r1")), _message_event(PERSON, to=RunRef("r2")), False),
    (Filter(to=RunRef("r1")), _message_event(PERSON), False),
])
def test_filters(filter_, event, expected):
    assert filter_.matches(event) is expected


@pytest.mark.parametrize("sender, agent, expected", [
    (Sender.person("u1"), None, True),  # a person
    (Sender.agent("partner-planner"), None, True),  # a caller: another agent
    (Sender.agent("researcher", run=RunRef("r1")), "researcher", False),  # this agent: any of its runs
])
def test_origin_outside_is_not_this_agent(sender, agent, expected):
    event = ThreadEvent(position=1, kind=EventKind.MESSAGE, agent=agent, message=Message.create(sender, "x"))
    assert Filter(origin=Origin.OUTSIDE).matches(event, agent="researcher") is expected
    assert Filter().matches(event, agent="researcher") is True  # Origin.ANY


def test_filters_are_hashable():
    assert len({Filter(kinds=["message"]), Filter(kinds=("message",))}) == 1


def test_parts_and_messages_round_trip():
    message = Message.create(
        Sender.agent("reporter", run=RunRef("r9"), name="Reporter"),
        "text",
        FilePart(mime_type="application/pdf", name="r.pdf", data=b"%PDF\x00\x01"),
        DataPart(data={"k": [1, 2]}, schema={"type": "object"}),
        ToolCallPart(call_id="c1", name="search", arguments={"q": "x"}),
        ToolResultPart(call_id="c1", output=[TextPart(text="found")], is_error=False),
        ArtifactPart(name="report", description="The report",
                     parts=[TextPart(text="summary"), FilePart(mime_type="application/pdf", uri="mem://files/1")]),
        references=[RunRef("r0")],
    )
    again = message_from_dict(to_dict(message))
    assert again == message
    assert again.sender.kind is SenderKind.AGENT and again.sender.run == RunRef("r9")
    assert again.references == (RunRef("r0"),) and to_dict(message)["references"] == ["r0"]


def test_an_unknown_part_kind_passes_through_unchanged():
    raw = {"kind": "hologram", "frames": 24, "metadata": {"v": 2}}
    part = part_from_dict(raw)
    assert isinstance(part, UnknownPart) and part.kind == "hologram" and part.metadata == {"v": 2}
    assert to_dict(part) == raw


def test_a_known_kind_with_a_newer_field_loses_only_that_field():
    part = part_from_dict({"kind": "text", "text": "hi", "language": "nl"})
    assert part == TextPart(text="hi")


def test_events_round_trip_and_unknown_kinds_pass_through():
    completed = MessageCompleted(message=Message.create(AGENT, "done"))
    assert event_from_dict(to_dict(completed)) == completed
    created = ThreadCreated(thread_id="t2", agent="desk", forked_from="t1", run_id="r2")
    assert event_from_dict(to_dict(created)) == created
    assert event_from_dict(to_dict(Progress(label="Searching", fraction=0.5))) == Progress(label="Searching", fraction=0.5)
    custom = Custom(kind="x-trip.plan_ready", data={"plan": [1]})  # an app's own event, with its data
    assert event_from_dict(to_dict(custom)) == custom
    raw = {"kind": "session.something_newer", "metadata": {}, "plan": [1]}  # a newer kind: kept as it came
    ev = event_from_dict(raw)
    assert isinstance(ev, UnknownEvent) and to_dict(ev) == raw


# ── custom parts ──

from dataclasses import dataclass  # noqa: E402
from typing import Any, Mapping  # noqa: E402

from llamphouse.sdk import Part, part  # noqa: E402


@part
@dataclass(frozen=True, kw_only=True)
class ChartPart(Part):
    kind: str = "x-chart"
    spec: Mapping[str, Any]
    title: str = ""
    thumbnail: bytes | None = None


def test_a_registered_custom_part_survives_the_wire():
    chart = ChartPart(spec={"type": "bar", "series": [1, 2]}, title="Sales", thumbnail=b"\x89PNG",
                      metadata={"v": 1})
    message = Message.create(Sender.agent("charts"), "here", chart)
    again = message_from_dict(to_dict(message))
    assert again == message and isinstance(again.parts[1], ChartPart)


def test_an_unregistered_custom_part_stays_unknown_and_unchanged():
    raw = {"kind": "x-map", "center": [50.8, 4.4], "metadata": {}}
    assert isinstance(part_from_dict(raw), UnknownPart) and to_dict(part_from_dict(raw)) == raw


def test_registering_twice_is_fine_but_custom_kinds_must_be_x_and_unique():
    assert part(ChartPart) is ChartPart

    @dataclass(frozen=True, kw_only=True)
    class Plain(Part):
        kind: str = "chart"

    with pytest.raises(ValueError, match="'x-'"):
        part(Plain)

    @dataclass(frozen=True, kw_only=True)
    class OtherChart(Part):
        kind: str = "x-chart"

    with pytest.raises(ValueError, match="already registered"):
        part(OtherChart)
    with pytest.raises(TypeError):
        part(dict)  # type: ignore[arg-type]
