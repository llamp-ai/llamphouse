"""The session providers in the box, on their own."""

from llamphouse.runtime.sessions import RollingWindow, Revise, SessionInfo, SessionProvider, Summarizing
from llamphouse.sdk import Message, RunResult, Sender, SenderKind, ThreadEvent

AN = Sender.person("an")


def session(n: int) -> SessionInfo:
    messages = tuple(Message.create(AN, str(i)) for i in range(n))
    return SessionInfo(id="s1", thread_id="t1", agent="a", messages=messages, version=1, revisions=0, revising=False)


def test_the_base_provider_shows_everything_and_never_revises():
    base, s = SessionProvider(), session(3)
    assert list(base.view(s)) == list(s.messages)
    event = ThreadEvent(position=1, kind="message", message=s.messages[0])
    assert base.on_event(s, event) is None and base.on_entry(s, s.messages[0]) is None
    assert base.after_run(s, RunResult(id="r1", agent="a", status="completed")) is None


async def test_the_base_provider_declines_a_revision():
    assert await SessionProvider().revise(session(3), {"reason": "requested"}) is None


def test_rolling_window_shows_the_latest_and_has_no_policy():
    window, s = RollingWindow(2), session(5)
    assert [m.text for m in window.view(s)] == ["3", "4"]
    assert window.on_entry(s, s.messages[0]) is None


async def test_summarizing_asks_past_its_limit_and_keeps_the_latest():
    seen = []

    async def summarize(older):  # a model call, in real life
        seen.append(len(older))
        return f"summary of {len(older)}"

    provider = Summarizing(summarize, max_messages=4, keep=2)
    assert provider.on_entry(session(4), session(4).messages[0]) is None
    assert provider.on_entry(session(5), session(5).messages[0]) == Revise({"reason": "policy"})
    revised = await provider.revise(session(5), {"reason": "policy"})
    assert [m.text for m in revised] == ["summary of 3", "3", "4"]
    assert revised[0].sender.kind == SenderKind.SYSTEM
    revised = await provider.revise(session(5), {"keep": 4})  # a request may keep more
    assert [m.text for m in revised] == ["summary of 1", "1", "2", "3", "4"]
    assert await provider.revise(session(2), {}) is None  # nothing older than what it keeps: declined
    assert seen == [3, 1]
