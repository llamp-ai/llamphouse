"""Scenarios: real agents on the in-memory Scenario runtime - one agent per thread, parallel runs of
an agent, other agents as callers and callees, and a shared conversation (a room) built on top.

Three groups: agents together, runtime guarantees, single-agent behaviour.
"""

import asyncio
from collections import Counter

import pytest
from pydantic import BaseModel, ValidationError

from llamphouse.sdk import (
    requires,
    Agent,
    AgentError,
    Agents,
    AuthRequest,
    Config,
    EventKind,
    Filter,
    Message,
    RunRef,
    RunStatus,
    Sender,
    SenderKind,
    Origin,
    StateConflict,
    ThreadClosed,
    ThreadBusy,
    NoActiveRun,
    AgentType,
    ThreadRef,
    Viewpoint,
    Workflow,
)
from llamphouse.sdk import MessageCompleted, TextPart, ToolCallPart, ToolResultPart
from llamphouse.sdk.testing import Scenario
from llamphouse.runtime.sessions import RollingWindow, Revise, SessionProvider, Summarizing

AN = Sender.person("an", name="An")


# ═════════════════════════════════════════════════════════════════════════════
# Agents together
# ═════════════════════════════════════════════════════════════════════════════

class Moderator(Agent):
    async def run(self, ctx):
        said = ctx.input
        if said.sender.kind == SenderKind.PERSON:
            await ctx.send("Who takes hotels?")
        elif said.text == "I do":
            await ctx.send("Great, thanks.")


class Hotels(Agent):
    async def run(self, ctx):
        said = ctx.input
        if "hotels" in said.text.lower() and not ctx.session.is_own(said):
            await ctx.send("I do")


class Room:
    """A shared conversation, built on top of llamphouse: several agents and people. Each agent keeps
    its own thread in the room; the room relays every message to the agents that should see it and
    publishes their replies. Who sees what and who may speak is the room's policy, not llamphouse's."""

    def __init__(self, sc: Scenario, *agents: str, rounds: int = 10):
        self.sc, self.agents, self.rounds = sc, agents, rounds
        self.threads: dict[str, object] = {}
        self.transcript: list[tuple[str, str]] = []
        self._seen: dict[str, int] = {}

    async def open(self) -> "Room":
        for a in self.agents:
            self.threads[a] = await self.sc.thread(a)
        return self

    async def post(self, text: str, by: Sender) -> None:
        self.transcript.append((by.id, text))
        for a in self.agents:
            if a != by.id:  # an agent doesn't get its own message back
                await self.sc.say(self.threads[a], text, by=by)

    async def settle(self) -> None:
        for _ in range(self.rounds):
            await self.sc.settle()
            replies = []
            for a, thread in self.threads.items():
                mine = [e.message for e in self.sc.timeline(thread) if e.message and e.message.sender.id == a]
                replies += [(a, m) for m in mine[self._seen.get(a, 0):]]
                self._seen[a] = len(mine)
            if not replies:
                return
            for a, m in replies:
                await self.post(m.text, by=Sender.agent(a))
        raise AssertionError("the room didn't quiet down")


async def test_a_room_built_on_top_lets_agents_and_people_talk_together():
    async with Scenario({"moderator": Moderator, "hotels": Hotels}) as sc:
        room = await Room(sc, "moderator", "hotels").open()
        await room.post("Plan a trip to Bangkok", by=AN)
        await room.settle()
        assert room.transcript == [
            ("an", "Plan a trip to Bangkok"),
            ("moderator", "Who takes hotels?"),
            ("hotels", "I do"),  # hotels heard the moderator's question on its own thread
            ("moderator", "Great, thanks."),
        ]
        # one agent per thread: each thread holds one agent's conversation with the others
        assert {who for who, _ in sc.messages(room.threads["hotels"])} == {"an", "moderator", "hotels"}
        assert all(r.thread_id == room.threads[r.agent].id for r in sc.runs())


class Recurses(Agent):
    async def run(self, ctx):
        await ctx.fork("again")  # a fork of itself, which forks again


async def test_an_agent_that_keeps_forking_itself_hits_the_runs_per_root_limit():
    async with Scenario({"r": Recurses}, max_runs_per_root=10) as sc:
        thread = await sc.thread("r")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        runs = sc.runs()  # each on a fork of the one before
        assert len(runs) == 10 and any(r.status is RunStatus.FAILED and "a loop" in r.error for r in runs)
        assert len({r.thread_id for r in runs}) == 10


async def test_runs_that_start_runs_too_deep_are_stopped():
    async with Scenario({"r": Recurses}, max_depth=3) as sc:
        thread = await sc.thread("r")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        runs = sc.runs()
        assert len(runs) == 4 and "deep" in next(r.error for r in runs if r.status is RunStatus.FAILED)


class Researcher(Agent):
    """Another agent, on its own thread: it talks there; its caller listens."""

    async def run(self, ctx):
        await ctx.send(f"started on {ctx.input.text}")
        steer = await ctx.next()  # the caller's next message on this thread
        await ctx.send(f"{ctx.input.text}, {steer.message.text}")


class Lead(Agent):
    """Supervises another agent: one run that waits on the user and on the other agent's thread at once,
    wakes on each event, and restarts per iteration."""

    async def run(self, ctx) -> list[tuple[str, str | None]]:
        mine = ctx.run.state  # kept across restarts
        if (researcher := await mine.get("researcher")) is None:
            researcher = await ctx.call("researcher", "hotels")  # a new thread of it
            await mine.set("researcher", researcher)  # kept as a reference: comes back as its handle
        ev = await ctx.subscribe(
            *ctx.inbox,  # the user, on this thread
            Filter(thread=researcher, agents=["researcher"], kinds=[EventKind.MESSAGE, EventKind.RUN_COMPLETED]),
        ).next()
        log = await mine.update("log", lambda log: log + [(str(ev.kind), ev.message.text if ev.message else None)], [])
        if ev.message is not None and ev.message.sender.kind == SenderKind.PERSON:
            await researcher.send(ev.message.text)  # the user steers it: a message on its thread
        if ev.kind == EventKind.RUN_COMPLETED:
            await ctx.send(f"done: {(await researcher.messages())[-1].text}")
            return log
        await ctx.restart()


async def test_a_supervisor_waits_on_the_user_and_another_agents_thread_at_once():
    async with Scenario({"lead": Lead, "researcher": Researcher}) as sc:
        thread = await sc.thread("lead")
        await sc.say(thread, "Plan my trip", by=AN)
        await sc.settle()
        (lead,) = sc.runs(thread, agent="lead")
        assert lead.status is RunStatus.WAITING  # holds no worker

        await sc.say(thread, "cheaper", by=AN)  # delivered to the waiting run, not a new one
        await sc.settle()
        assert len(sc.runs(thread, agent="lead")) == 1
        assert lead.status is RunStatus.COMPLETED
        assert lead.output == [("message", "started on hotels"), ("message", "cheaper"),
                               ("message", "hotels, cheaper"), ("run.completed", None)]
        (researcher,) = sc.runs(agent="researcher")
        assert researcher.thread_id != thread.id  # another agent works on its own thread
        assert lead.text == "done: hotels, cheaper"
        assert max(lead.replays) <= 4  # restart keeps every wake-up short: a few steps, never the history


class Splitter(Agent):
    """Forks itself: the fork carries on from the parent's history (its own), and from the fork on, each
    owns its own messages and the other is someone else."""

    async def run(self, ctx) -> list[tuple[str, bool]]:
        if ctx.run.parent is None:
            await ctx.send("I'll split this")
            fork = await ctx.fork("part 1")
            await fork.wait()
            seen = [*await ctx.session.messages(), *await fork.messages()]  # its own, and the fork's: read
        else:
            await ctx.send("part 1 done")  # on its own thread: the parent listens
            seen = list(await ctx.session.messages())
        return [(m.text, ctx.session.is_own(m)) for m in seen]


async def test_a_fork_carries_on_from_the_parents_history_and_then_is_someone_else():
    async with Scenario({"splitter": Splitter}) as sc:
        thread = await sc.thread("splitter")
        await sc.say(thread, "Write a report", by=AN)
        await sc.settle()
        parent, child = sc.runs(agent="splitter")
        assert parent.output == [("Write a report", False), ("I'll split this", True),
                                 ("part 1", True), ("part 1 done", False)]  # the fork's: read
        assert child.output == [("Write a report", False),
                                ("I'll split this", True),  # before the fork: its own history
                                ("part 1", False),  # its input, sent by its parent
                                ("part 1 done", True)]
        assert sc.forks(thread) == [(ThreadRef(child.thread_id), 3)]  # forked after the parent's reply
        assert [who for who, _ in sc.messages(ThreadRef(child.thread_id))] == [
            "an", "splitter", "splitter", "splitter"]  # to itself, a thread like any other
        assert [who for who, _ in sc.messages(thread)] == ["an", "splitter"]  # nothing written back


class Echo(Agent):
    async def run(self, ctx):
        said = ctx.input
        own = "own" if ctx.session.is_own(said) else "other"
        await ctx.send(f"{own} from {said.sender.kind}:{ctx.run.caller.subject}")


async def test_an_agent_calling_over_a2a_is_just_another_sender():
    async with Scenario({"echo": Echo}) as sc:
        thread = await sc.thread("echo")
        await sc.say(thread, "hi", by=Sender.agent("remote-planner"))
        await sc.settle()
        assert sc.messages(thread)[-1] == ("echo", "other from agent:remote-planner")


class Specialist(Agent):
    async def run(self, ctx):
        await ctx.send(f"seen {[m.text for m in await ctx.session.messages()]}")


class Delegates(Agent):
    """Asks another agent twice: on two new threads, or twice on one."""

    same_thread = False

    async def run(self, ctx):
        first = await ctx.call("specialist", "q1")
        await first.wait()
        if self.same_thread:
            await first.start("q2")  # the same thread: a second turn, and it remembers q1
            await first.wait()
        else:
            await (await ctx.call("specialist", "q2")).wait()  # a new thread: a new conversation
        return [m.text for m in await ctx.session.messages()]


class AsksTwiceOnOneThread(Delegates):
    same_thread = True


@pytest.mark.parametrize("parent, second", [
    (Delegates, "seen ['q2']"),
    (AsksTwiceOnOneThread, "seen ['q1', \"seen ['q1']\", 'q2']"),
])
async def test_talking_to_another_agent_on_one_thread_or_two(parent, second):
    async with Scenario({"p": parent, "specialist": Specialist}) as sc:
        thread = await sc.thread("p")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        (p,) = sc.runs(thread, agent="p")
        assert sc.runs(agent="specialist")[1].text == second
        assert p.output == ["go"]  # the specialist's conversation stays on its own thread


class AsksEcho(Agent):
    async def run(self, ctx):
        echo = await ctx.call("echo", "x")  # another agent: its own thread
        await echo.wait()
        await ctx.send((await echo.messages())[-1].text)  # what it said there: copied over


class TriesToReadIt(Agent):
    async def run(self, ctx):
        echo = await ctx.call("echo", "x")
        await echo.thread_events()  # another agent's thread: talk to it, don't read it


async def test_another_agents_thread_is_talked_to_not_read():
    async with Scenario({"p": TriesToReadIt, "echo": Echo}) as sc:
        thread = await sc.thread("p")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.FAILED and run.error == "another_agents_thread"


async def test_another_agent_keeps_its_messages_on_its_own_thread():
    async with Scenario({"p": AsksEcho, "echo": Echo}) as sc:
        thread = await sc.thread("p")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        assert [who for who, _ in sc.messages(thread)] == ["an", "p"]  # its reply isn't here: p copied it
        (echo,) = sc.created(thread)
        assert [who for who, _ in sc.messages(echo)] == ["p", "echo"]
        assert any(e.kind == EventKind.RUN_COMPLETED for e in sc.timeline(echo))  # its end: on its thread


# ═════════════════════════════════════════════════════════════════════════════
# Runtime guarantees
# ═════════════════════════════════════════════════════════════════════════════

class FanOut(Agent):
    async def run(self, ctx):
        threads = [await ctx.call("echo", str(i)) for i in range(3)]
        for t in threads:
            await t.wait()
        await ctx.send(", ".join([(await t.messages())[-1].text for t in threads]))


async def test_a_waiting_run_holds_no_worker():
    async with Scenario({"p": FanOut, "echo": Echo}, workers=1) as sc:  # one worker: holding it would deadlock
        thread = await sc.thread("p")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        assert sc.runs(thread, agent="p")[0].status is RunStatus.COMPLETED


EFFECTS: Counter = Counter()


async def charge(invoice):
    EFFECTS[f"charge {invoice}"] += 1
    return "ref-1"


async def notify():
    EFFECTS["notify"] += 1


class Pays(Agent):
    async def run(self, ctx):
        ref = await ctx.step("charge", charge, "inv-1")
        await ctx.step("notify", notify)
        await ctx.send(f"paid {ref}")


@pytest.mark.parametrize("crash, charged", [
    ({"after_step": "charge"}, 1),  # recorded before the crash: replayed, not run again
    ({"before_record": "charge"}, 2),  # ran, but not recorded: runs again (steps run at least once)
])
async def test_a_crash_and_recovery(crash, charged):
    EFFECTS.clear()
    async with Scenario({"pays": Pays}) as sc:
        sc.crash(agent="pays", **crash)
        thread = await sc.thread("pays")
        await sc.say(thread, "pay", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.RUNNING  # the worker died; the lease still holds it
        assert await sc.recover() == 1
        await sc.settle()
        assert run.status is RunStatus.COMPLETED and run.text == "paid ref-1"
        assert EFFECTS == {"charge inv-1": charged, "notify": 1}
        assert sc.messages(thread).count(("pays", "paid ref-1")) == 1


class Ticks(Agent):
    restarts = True

    async def run(self, ctx):
        ev = await ctx.subscribe(Filter(kinds=["x-tick"])).next()
        await ctx.step("handle", lambda: ev.data)
        n = await ctx.run.state.update("n", lambda n: n + 1, 0)
        if ev.data == 20:
            return n
        if self.restarts:
            await ctx.restart()
        return await self.loop(ctx, n)

    async def loop(self, ctx, n):
        async for ev in ctx.subscribe(Filter(kinds=["x-tick"])):
            await ctx.step("handle", lambda: ev.data)
            n += 1
            if ev.data == 20:
                return n


class TicksWithoutRestart(Ticks):
    restarts = False


@pytest.mark.parametrize("agent, flat", [(Ticks, True), (TicksWithoutRestart, False)])
async def test_restart_keeps_replay_flat(agent, flat):
    async with Scenario({"t": agent}) as sc:
        thread = await sc.thread("t")
        await sc.say(thread, "go", by=AN)
        for i in range(1, 21):
            await sc.post(thread, "x-tick", i)
            await sc.settle()
        (run,) = sc.runs(thread)
        assert run.output == 20
        assert (max(run.replays) <= 2) is flat


class Waits(Agent):
    async def run(self, ctx):
        await ctx.run.wait_for_signal("finish")
        await ctx.send("finished")


async def test_a_message_sent_to_a_run_that_never_reads_it_stays_unread():
    async with Scenario({"w": Waits}) as sc:
        thread = await sc.thread("w")
        await sc.say(thread, "first", by=AN)
        await sc.settle()
        (first,) = sc.runs(thread)
        await sc.say(thread, "second", by=AN)  # sent to the active run: on the thread, in its inbox
        await sc.signal(first.ref, "finish")
        await sc.settle()
        assert [r.input.text for r in sc.runs(thread)] == ["first"]  # nothing starts by itself
        assert [m.text for m in sc.unread(first.ref)] == ["second"]  # the app decides: start a run with it?
        assert ("an", "second") in sc.messages(thread)  # recorded when it was sent
        assert "second" not in [m.text for m in sc.session(first.ref).messages]  # never read: never in its session
        await sc.start("w", sc.unread(first.ref)[0], thread=thread)  # e.g. like this
        await sc.settle()
        assert [r.input.text for r in sc.runs(thread)] == ["first", "second"]


class Tree(Agent):
    async def run(self, ctx):
        await ctx.call("w", "attached")
        await ctx.call("w", "detached", detach=True)
        await ctx.run.wait_for_signal("never")


async def test_cancelling_cascades_except_to_detached_children():
    async with Scenario({"tree": Tree, "w": Waits}) as sc:
        thread = await sc.thread("tree")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        (root,) = sc.runs(thread, agent="tree")
        await sc.cancel(root.ref)
        await sc.settle()
        assert {r.input.text: r.status for r in sc.runs(agent="w")} == {
            "attached": RunStatus.CANCELLED, "detached": RunStatus.WAITING}


class Asks(Agent):
    async def run(self, ctx):
        await ctx.send((await ctx.ask("Approve?")).text)


class Approves(Agent):
    async def run(self, ctx):
        asks = await ctx.call("asks", "x")
        await asks.events(EventKind.RUN_INPUT_REQUIRED).next()  # it asks something, on its thread
        await ctx.run.wait_for_signal("go")
        won = await asks.respond("parent says yes")
        await asks.wait()
        return {"won": won, "child": (await asks.messages())[-1].text}


@pytest.mark.parametrize("client_first", [False, True])
async def test_a_childs_question_the_first_answer_wins(client_first):
    async with Scenario({"p": Approves, "asks": Asks}) as sc:
        thread = await sc.thread("p")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        child = sc.runs(agent="asks")[0]
        assert child.status is RunStatus.INPUT_REQUIRED and child.pending_request.reason == "Approve?"
        if client_first:
            assert await sc.answer(child.ref, "client says yes", by=AN)
        await sc.signal(sc.runs(thread, agent="p")[0].ref, "go")
        await sc.settle()
        winner = "client says yes" if client_first else "parent says yes"
        assert sc.runs(thread, agent="p")[0].output == {"won": not client_first, "child": winner}


class Sleeps(Agent):
    async def run(self, ctx):
        await ctx.sleep(3600)
        await ctx.send("awake")


async def test_a_sleeping_run_wakes_when_the_clock_says_so():
    async with Scenario({"s": Sleeps}) as sc:
        thread = await sc.thread("s")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        sc.advance(3599)
        await sc.settle()
        assert sc.runs(thread)[0].status is RunStatus.WAITING
        sc.advance(1)
        await sc.settle()
        assert sc.runs(thread)[0].text == "awake"


# ═════════════════════════════════════════════════════════════════════════════
# Single-agent behaviour
# ═════════════════════════════════════════════════════════════════════════════

class HrConfig(Config):
    index: str


class Hr(Agent[HrConfig]):
    async def run(self, ctx):
        year = (await ctx.ask("Which year?")).text
        async with ctx.stream() as m:
            await m.write(f"{ctx.config.index} policy ")
            await m.write(year)


async def test_an_agent_asks_and_streams_its_answer():
    async with Scenario({"hr": (Hr, HrConfig(index="hr-be"))}) as sc:
        thread = await sc.thread("hr")
        await sc.say(thread, "Leave policy?", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.INPUT_REQUIRED
        await sc.answer(run.ref, "2026", by=AN)
        await sc.settle()
        assert run.sent[-1].text == "hr-be policy 2026"
        deltas = [e.delta for _, e in run.stream if e.kind == EventKind.MESSAGE_DELTA]
        assert deltas == ["hr-be policy ", "2026"]  # streamed once, not again on the replay after the answer


class InvoiceIn(BaseModel):
    invoice_id: str


class InvoiceOut(BaseModel):
    paid: bool


class PayInvoice(Workflow[Config, InvoiceIn, InvoiceOut]):
    async def run(self, ctx):
        await ctx.step("pay", lambda: ctx.input.data.invoice_id)
        return InvoiceOut(paid=True)


async def test_a_workflow_starts_with_typed_input_and_ends_with_its_output():
    async with Scenario({"pay": PayInvoice}) as sc:
        with pytest.raises(ValidationError):
            await sc.start("pay", {"wrong": 1})  # a bad input never becomes a run
        ref = await sc.start("pay", {"invoice_id": "i1"})
        await sc.settle()
        run = sc.run(ref)
        assert run.output == InvoiceOut(paid=True)
        (output,) = run.sent[-1].artifacts
        assert output.name == "output" and output.data == {"paid": True}


class NeedsCalendar(Agent):
    async def run(self, ctx):
        if "weekend" in ctx.input.text:
            await ctx.reject("I don't book weekends")
        grant = await ctx.run.request_auth(AuthRequest(reason="calendar", scheme="oauth2", scopes=["calendar.write"]))
        await ctx.send(f"booked with {grant.data['token']}")


async def test_auth_required_and_rejected():
    async with Scenario({"cal": NeedsCalendar}) as sc:
        thread = await sc.thread("cal")
        await sc.say(thread, "book monday", by=AN)
        await sc.settle()
        first = sc.runs(thread)[0]
        assert first.status is RunStatus.AUTH_REQUIRED and first.pending_request.scopes == ["calendar.write"]
        with pytest.raises(ThreadBusy):
            await sc.start("cal", "book weekend", thread=thread)  # one run at a time
        await sc.authorize(first.ref, {"token": "t-1"}, by=AN)
        await sc.settle()
        await sc.say(thread, "book weekend", by=AN)
        await sc.settle()
        first, second = sc.runs(thread)
        assert first.text == "booked with t-1"
        assert second.status is RunStatus.REJECTED and second.error == "I don't book weekends"


# ── shared state ──

class Slow(Agent):
    async def run(self, ctx):
        await ctx.sleep(60)


async def test_a_thread_has_at_most_one_active_run():
    async with Scenario({"s": Slow}) as sc:
        thread = await sc.thread("s")
        first = await sc.start("s", "one", thread=thread)
        await sc.settle()
        with pytest.raises(ThreadBusy) as busy:
            await sc.start("s", "two", thread=thread)  # the API refuses a second run...
        assert busy.value.run_id == first.id
        await sc.send(thread, "hurry")  # ...but a message can be sent to the active one
        sc.advance(60)
        await sc.settle()
        with pytest.raises(NoActiveRun):
            await sc.send(thread, "anyone?")  # and nothing to send to once it has ended
        await sc.start("s", "two", thread=thread)  # now a new one may start


class CountsOnce(Agent):
    async def run(self, ctx):
        await ctx.state.update("done", lambda n: n + 1, 0)
        await ctx.step("after", lambda: None)


async def test_a_replay_after_a_crash_does_not_update_state_twice():
    async with Scenario({"counter": CountsOnce}) as sc:
        sc.crash(agent="counter", after_step="after")
        thread = await sc.thread("counter")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        await sc.recover()
        await sc.settle()
        assert sc.runs(thread)[0].status is RunStatus.COMPLETED
        assert sc.state_of(thread)["done"] == (1, 1)
        assert len([e for e in sc.timeline(thread) if e.kind == EventKind.STATE_CHANGED]) == 1


# ── how a run started: always one message ──

class OnOrder(Agent):
    async def run(self, ctx):
        started = ctx.input
        if started.sender.kind == SenderKind.TRIGGER:
            await ctx.send(f"order {started.data['order']} via {started.sender.id}")
        else:
            await ctx.send(f"chat: {started.text}")


async def test_a_trigger_is_the_sender_of_the_message_that_starts_the_run():
    from llamphouse.sdk import DataPart

    async with Scenario({"orders": OnOrder}) as sc:
        thread = await sc.thread("orders")
        await sc.say(thread, DataPart(data={"order": 42}), by=Sender.trigger("orders-webhook"))
        await sc.settle()
        await sc.say(thread, "where is my order?", by=AN)
        await sc.settle()
        assert [text for who, text in sc.messages(thread) if who == "orders"] == [
            "order 42 via orders-webhook", "chat: where is my order?"]


class ReadsLater(Agent):
    async def run(self, ctx) -> tuple[list[str], list[str]]:
        await ctx.sleep(60)
        before = [m.text for m in await ctx.session.messages()]
        ev = await ctx.next(timeout=0)  # takes it in: from now on, it is in the session
        after = [m.text for m in await ctx.session.messages()]
        return before, after


async def test_a_message_sent_to_a_run_is_on_the_thread_when_sent_and_in_its_session_when_read():
    async with Scenario({"r": ReadsLater}) as sc:
        thread = await sc.thread("r")
        await sc.say(thread, "start", by=AN)
        await sc.settle()
        run = sc.runs(thread)[0]
        await sc.send(thread, "focus on cost", by=AN)
        with pytest.raises(ThreadBusy):
            await sc.append(thread, "just a note", by=AN)  # while a run works: only through the run
        sc.advance(60)
        await sc.settle()
        assert run.output == (["start"], ["start", "focus on cost"])
        read = [e for _, e in run.stream if e.kind == EventKind.MESSAGE_READ]
        sent = next(e for e in sc.timeline(thread) if e.message is not None and e.message.text == "focus on cost")
        assert len(read) == 1 and read[0].position == sent.position and sent.to == run.ref


class PaysThroughWorkflow(Agent):
    async def run(self, ctx):
        pay = await ctx.call("pay", {"invoice_id": "i9"})  # data: the Input
        await ctx.send(f"paid: {(await pay.wait()).output.paid}")  # the run, finished: its typed Output


async def test_a_child_workflow_gets_its_input_as_data():
    async with Scenario({"p": PaysThroughWorkflow, "pay": PayInvoice}) as sc:
        thread = await sc.thread("p")
        await sc.say(thread, "pay i9", by=AN)
        await sc.settle()
        assert sc.runs(thread, agent="p")[0].text == "paid: True"
        child = sc.runs(agent="pay")[0]
        assert child.input.data == InvoiceIn(invoice_id="i9")  # validated before the run started


# ── files ──

from llamphouse.sdk import ArtifactPart, DataPart, FilePart, Files  # noqa: E402

PDF = b"%PDF-1.7 a contract"


@requires(Files)
class ReadsUpload(Agent):
    async def run(self, ctx):
        upload = next(p for p in ctx.input.parts if isinstance(p, FilePart))
        data = await ctx.capability(Files).get(upload)
        await ctx.send(f"{upload.name}: {len(data)} bytes",
                        ArtifactPart(name="notes", parts=[FilePart(mime_type="text/plain", name="notes.txt",
                                                                   data=b"annotated")]))


async def test_an_upload_in_a_message_is_stored_and_the_message_keeps_a_reference():
    async with Scenario({"r": ReadsUpload}) as sc:
        thread = await sc.thread("r")
        await sc.say(thread, "Summarise this", FilePart(mime_type="application/pdf", name="contract.pdf", data=PDF),
                     by=AN)
        await sc.settle()
        uploaded, reply = [e.message for e in sc.timeline(thread) if e.message]
        part = uploaded.parts[1]
        assert part.data is None and part.uri.startswith("files://") and part.size == len(PDF)
        record = await sc.file(part)
        assert record.thread_id == thread.id and record.expires_at is None  # owned by the thread
        notes = reply.artifacts[0].parts[0]  # the agent's file, inside an artifact: stored too
        assert reply.text == f"contract.pdf: {len(PDF)} bytes" and notes.data is None
        assert await sc.file_store.read_all((await sc.file(notes)).id) == b"annotated"


async def test_upload_first_then_reference_and_abandoned_uploads_are_cleaned_up():
    async with Scenario({"r": ReadsUpload}) as sc:
        thread = await sc.thread("r")
        part = await sc.upload(PDF, mime_type="application/pdf", name="contract.pdf", by=AN)
        orphan = await sc.upload(b"never sent", mime_type="text/plain", by=AN)
        before = await sc.file(part)
        assert before.thread_id is None and before.expires_at is not None
        await sc.say(thread, "Summarise this", part, by=AN)  # the message claims the file
        await sc.settle()
        after = await sc.file(part)
        assert after.thread_id == thread.id and after.expires_at is None
        sc.advance(3601)
        assert await sc.cleanup() == 1  # only the abandoned upload
        assert [str(e.kind) for e in sc.system_events] == [
            "file.stored", "file.stored",  # the two uploads
            "file.claimed",  # the message referencing the first
            "file.stored",  # the agent's notes (owned by the thread at once: no claim)
            "file.deleted"]  # the abandoned upload
        assert await sc.file_store.stat(orphan.uri.removeprefix("files://")) is None
        assert await sc.file_store.read_all(after.id) == PDF


async def test_the_same_bytes_are_stored_once():
    async with Scenario({"r": ReadsUpload}) as sc:
        a = await sc.upload(PDF, mime_type="application/pdf")
        b = await sc.upload(PDF, mime_type="application/pdf")
        assert a.uri != b.uri and a.sha256 == b.sha256 and len(sc.file_store.blobs) == 1


async def test_a_message_cannot_reference_a_file_of_another_thread():
    async with Scenario({"r": ReadsUpload}) as sc:
        mine, other = await sc.thread("r"), await sc.thread("r")
        await sc.say(other, "x", FilePart(mime_type="application/pdf", name="secret.pdf", data=PDF), by=AN)
        await sc.settle()
        secret = next(e.message.parts[1] for e in sc.timeline(other) if e.message)
        with pytest.raises(PermissionError):  # refused before the message is written
            await sc.say(mine, "Summarise this", secret, by=AN)
        assert sc.runs(mine) == []


@requires(Files)
class PutsThenCrashes(Agent):
    async def run(self, ctx):
        part = await ctx.capability(Files).put(b"report", mime_type="text/plain", name="r.txt")
        await ctx.step("after_put", lambda: None)  # the worker dies after this step
        await ctx.send(ArtifactPart(name="report", parts=[part]))


async def test_a_replay_gets_the_same_stored_file_back():
    async with Scenario({"p": PutsThenCrashes}) as sc:
        sc.crash(agent="p", after_step="after_put")
        thread = await sc.thread("p")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        await sc.recover()
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.COMPLETED
        assert len(sc.file_store.records) == 1  # put once, not again on the replay
        assert (await sc.file(run.sent[-1].artifacts[0].parts[0])).thread_id == thread.id


class SendsAcrossRestarts(Agent):
    async def run(self, ctx):
        if await ctx.run.state.get("iteration", 1) == 1:
            await ctx.send("note to self, iteration 1")
            await ctx.run.state.set("iteration", 2)
            await ctx.restart()
        await ctx.send("done in iteration 2")  # the same call key (send#1) as iteration 1's message


async def test_messages_after_a_restart_are_not_swallowed_as_duplicates():
    async with Scenario({"s": SendsAcrossRestarts}) as sc:
        thread = await sc.thread("s")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        assert [text for _, text in sc.messages(thread)] == ["go", "note to self, iteration 1", "done in iteration 2"]


class KnowsItself(Agent):
    async def run(self, ctx) -> tuple[str, AgentType]:
        return ctx.me.name, ctx.me.type


async def test_a_workflow_and_an_agent_both_know_themselves_as_me():
    async with Scenario({"helper": KnowsItself, "pay": PayInvoice}) as sc:
        thread = await sc.thread("helper")
        await sc.say(thread, "hi", by=AN)
        await sc.settle()
        helper = sc.runs(thread)[0]
        assert helper.output == ("helper", "agent")
        ref = await sc.start("pay", {"invoice_id": "i1"})
        await sc.settle()
        assert sc.run(ref).agent_uid.startswith("agt_") and sc.run(ref).agent_uid != helper.agent_uid


class ShowsSession(Agent):
    """Forks itself; the fork's session starts from a copy of this one (the default), nothing, or a
    brief this run wrote."""

    seed = "copy"

    async def run(self, ctx):
        if ctx.run.parent is None and ctx.input.text == "delegate":
            seeds = {"none": None, "brief": [Message.create(Sender.system(), "a brief")]}
            fork = await (ctx.fork("sub") if self.seed == "copy" else ctx.fork("sub", session=seeds[self.seed]))
            await fork.wait()
        view = await ctx.session.messages()
        if ctx.run.parent is not None:
            await ctx.send("sub done")
        return view.id, [m.text for m in view]


class ShowsNoSession(ShowsSession):
    seed = "none"


class ShowsBrief(ShowsSession):
    seed = "brief"


@pytest.mark.parametrize("agent, fork_sees", [
    (ShowsSession, ["first", "delegate", "sub"]),  # a copy of the parent's working memory, then its input
    (ShowsNoSession, ["sub"]),  # a clean start
    (ShowsBrief, ["a brief", "sub"]),  # what the parent computed for it
])
async def test_the_session_is_the_agents_on_the_thread_and_a_fork_starts_from_what_it_is_given(agent, fork_sees):
    async with Scenario({"s": agent}) as sc:
        thread = await sc.thread("s")
        await sc.say(thread, "first", by=AN)
        await sc.settle()
        await sc.say(thread, "delegate", by=AN)
        await sc.settle()
        first, second, fork = sc.runs(agent="s")
        assert first.output[0] == second.output[0] != fork.output[0]  # its turns share it; the fork has its own
        assert second.output[1] == ["first", "delegate"]  # the fork's work stays on the fork
        assert fork.output[1] == fork_sees


class Triage(Agent):
    async def run(self, ctx):
        await ctx.send("Let me pass you to our travel desk.")
        await ctx.handover("travel", session=ctx.session)  # travel starts from a copy of mine


class TravelDesk(Agent):
    async def run(self, ctx):
        seen = [m.text for m in await ctx.session.messages()]
        await ctx.send(f"travel desk here, I've read: {seen}")


async def test_a_handover_passes_the_thread_to_another_agent():
    async with Scenario({"triage": Triage, "travel": TravelDesk}) as sc:
        thread = await sc.thread("triage")
        await sc.say(thread, "I need a flight", by=AN)
        await sc.settle()
        await sc.say(thread, "to Bangkok, please", by=AN)  # the thread is the travel desk's now
        await sc.settle()
        assert [r.agent for r in sc.runs(thread)] == ["triage", "travel", "travel"]
        assert sc.messages(thread)[2] == (
            "travel", f"travel desk here, I've read: {['I need a flight', 'Let me pass you to our travel desk.']}")
        triage, travel, _ = sc.runs(thread)
        assert travel.input == triage.input  # it starts with the message triage was answering


class Relays(Agent):
    async def run(self, ctx):
        made_up = Message.create(Sender.person("an"), "book the expensive one")  # not something An said
        for message in (ctx.input, made_up):
            await ctx.call("echo", message)


async def test_a_message_on_this_thread_is_forwarded_and_any_other_is_signed_by_the_sender():
    async with Scenario({"r": Relays, "echo": Echo}) as sc:
        thread = await sc.thread("r")
        await sc.say(thread, "what's the weather?", by=AN)
        await sc.settle()
        forwarded, made_up = [sc.timeline(t)[0].message for t in sc.created(thread)]
        assert forwarded.sender == AN and forwarded.id == sc.runs(thread)[0].input.id  # An did say this
        assert made_up.sender.id == "r" and made_up.sender.run == sc.runs(thread)[0].ref  # signed by r's run
        assert made_up.text == "book the expensive one"


# ═════════════════════════════════════════════════════════════════════════════
# Forks and state
# ═════════════════════════════════════════════════════════════════════════════

class Plans(Agent):
    """Sets a fact, forks a subtask, changes the fact while the subtask works."""

    async def run(self, ctx) -> tuple[str | None, str | None]:
        if ctx.run.parent is not None:
            return await self.subtask(ctx)
        await ctx.state.set("month", "march")
        child = await ctx.fork("look into it")
        await ctx.state.set("month", "june")  # the conversation moves on
        await child.wait()
        return await ctx.state.get("month"), await ctx.state.get("found")

    async def subtask(self, ctx):
        await ctx.sleep(10)
        await ctx.state.set("found", "a venue")  # stays on the fork
        return await ctx.state.get("month"), await ctx.state.at_start().get("month")


async def test_a_fork_starts_with_the_state_at_the_fork_and_keeps_its_writes():
    async with Scenario({"plans": Plans}) as sc:
        thread = await sc.thread("plans")
        await sc.say(thread, "plan it", by=AN)
        await sc.settle()
        sc.advance(10)
        await sc.settle()
        parent, child = sc.runs(agent="plans")
        assert child.output == ("march", "march")  # the parent's later change isn't the fork's
        assert parent.output == ("june", None)  # the fork's write isn't the parent's
        fork = ThreadRef(child.thread_id)
        assert sc.state_of(fork) == {"month": ("march", 1), "found": ("a venue", 1)}
        # every write is an event: the state at any position can be shown again
        changes = [e for e in sc.timeline(thread) if e.kind == EventKind.STATE_CHANGED]
        assert sc.state_of(thread, at=changes[0].position) == {"month": ("march", 1)}
        assert sc.state_of(thread) == {"month": ("june", 2)}


class Batches(Agent):
    async def run(self, ctx) -> tuple[list[str], str]:
        async with ctx.state.batch() as s:
            s.set("cities", ["lisbon"])
            s.set("month", "march")
        _, version = await ctx.state.get_versioned("cities")
        try:
            async with ctx.state.batch() as s:
                s.set("month", "june")
                s.set("cities", ["rome"], if_version=version + 1)  # stale: the whole batch fails
        except StateConflict:
            pass
        return await ctx.state.get("cities"), await ctx.state.get("month")


async def test_a_batch_is_all_or_nothing_and_one_event():
    async with Scenario({"b": Batches}) as sc:
        thread = await sc.thread("b")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        assert sc.runs(thread)[0].output == (["lisbon"], "march")
        (change,) = [e for e in sc.timeline(thread) if e.kind == EventKind.STATE_CHANGED]
        assert [c["key"] for c in change.data["changes"]] == ["cities", "month"]


class Coder(Agent):
    """A main run that forks subtasks in the background and listens to them; a subtask forks again, and
    may be steered while it works."""

    async def run(self, ctx):
        task = ctx.input.data
        if task is None:  # a request from the user: split it up
            forks = [await ctx.fork(DataPart(data={"task": part})) for part in ("styles", "tests")]
            results = []
            async for ev in ctx.subscribe(*[Filter(thread=f, kinds=[EventKind.MESSAGE], origin=Origin.INSIDE)
                                            for f in forks]):  # what the forks say, on their threads
                results.append(ev.message.text)
                if len(results) == 2:
                    break
            await ctx.send(" + ".join(sorted(results)))
            return
        if task["task"] == "tests":  # a subtask with subtasks of its own
            parts = [await ctx.fork(DataPart(data={"task": t})) for t in ("unit", "e2e")]
            for p in parts:
                await p.wait()
            await ctx.send("tests: " + ", ".join([(await p.messages())[-1].text for p in parts]))
            return
        await ctx.sleep(60)  # a long build
        await ctx.step(f"edit {task['task']}", lambda: None)
        steer = await ctx.next(timeout=0)
        await ctx.send(f"{task['task']} done" + (f" ({steer.message.text})" if steer else ""))


async def test_forks_of_forks_each_durable_and_the_parent_listens():
    async with Scenario({"coder": Coder}, workers=3) as sc:
        thread = await sc.thread("coder")
        await sc.say(thread, "Add dark mode", by=AN)
        await sc.settle()
        main = sc.runs(thread)[0]
        styles = next(r for r in sc.runs() if r.input.data == {"task": "styles"})
        await sc.say(ThreadRef(styles.thread_id), "use CSS variables", by=AN)  # steer it, on its own thread
        sc.crash(agent="coder", after_step="edit styles")
        sc.advance(60)
        await sc.settle()
        assert styles.status is RunStatus.RUNNING  # the worker died in the fork
        await sc.recover()
        await sc.settle()
        assert main.status is RunStatus.COMPLETED
        assert main.text == "styles done (use CSS variables) + tests: unit done, e2e done"
        assert styles.activations == 3 and max(styles.replays) >= 1  # recovered like any main run
        # a tree of threads: two forks of the main thread, two forks of the tests' fork
        tests = next(r for r in sc.runs() if r.input.data == {"task": "tests"})
        assert len(sc.forks(thread)) == 2 and len(sc.forks(ThreadRef(tests.thread_id))) == 2
        # the work stays on the forks: nothing is written back on the main thread
        assert [who for who, _ in sc.messages(thread)] == ["an", "coder"]


def summary_of(older):
    """What a model would write: here, just how many messages it summarised."""
    return f"(summary of {len(older)} messages)"


class Remembers(Agent):
    async def run(self, ctx) -> list[str]:
        if ctx.run.parent is None and ctx.input.text == "fork":
            now = await ctx.session.messages()  # as it is: the revision runs in the background
            settled = await ctx.session.settled()  # wait for it, holding no worker
            await (await ctx.fork("sub")).wait()  # the fork starts from the revised session
            return [str(now.revising), *[m.text for m in settled]]
        seen = [m.text for m in await ctx.session.messages()]
        if ctx.run.parent is not None:
            await ctx.send("ok")
        return seen


async def slow_summary_of(older):
    """A model call takes a while: long enough for the run to read the session while it is revised."""
    await asyncio.sleep(0.2)
    return summary_of(older)


async def test_a_session_is_revised_in_the_background_and_a_run_can_wait_for_it():
    async with Scenario({"r": Remembers}, sessions=Summarizing(slow_summary_of, max_messages=4, keep=2)) as sc:
        thread = await sc.thread("r")
        for text in ("one", "two", "three", "four", "fork"):  # the fifth message: past the window
            await sc.say(thread, text, by=AN)
            await sc.settle()
        parent, child = sc.runs(agent="r")[-2:]
        assert parent.output == ["True", "(summary of 3 messages)", "four", "fork"]  # was revising; then revised
        assert child.output == ["(summary of 3 messages)", "four", "fork", "sub"]  # the fork: from the revised one
        session = sc.session(parent.ref)
        assert sc.session(child.ref).forked_from == session.id
        assert [e["kind"] for e in session.log] == ["session.revision.started", "session.revision.completed"]
        assert not any(str(e.kind).startswith("session.") for e in sc.timeline(thread))  # on the session's log


class Works(Agent):
    """Keeps its tool calls in its session, not on the thread; asks for a revision with its own options."""

    async def run(self, ctx) -> list[tuple[str, bool]]:
        await ctx.session.add(ToolCallPart(call_id="c1", name="search", arguments={"q": "Rome"}))
        await ctx.session.add(ToolResultPart(call_id="c1", output=[TextPart(text="3 hotels")]))
        await ctx.send("Three hotels in Rome.")
        await ctx.session.revise(keep="tool results")  # returns at once
        view = await ctx.session.settled()
        return [(type(m.parts[0]).__name__, ctx.session.is_own(m)) for m in view]


class DropsToolCalls(SessionProvider):
    """An app's own provider: shows everything; a revision drops tool calls - and tool results too,
    unless the request asks to keep them."""

    async def revise(self, session, options):
        keep = options.get("keep")
        return [m for m in session.messages if not any(isinstance(p, ToolCallPart) for p in m.parts)
                and (keep == "tool results" or not any(isinstance(p, ToolResultPart) for p in m.parts))]


async def test_session_entries_are_not_on_the_thread_and_the_provider_decides_how_they_show():
    async with Scenario({"w": Works}, sessions={"w": DropsToolCalls()}) as sc:
        thread = await sc.thread("w")
        await sc.say(thread, "find me a hotel in Rome", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.output == [("TextPart", False), ("ToolResultPart", True), ("TextPart", True)]  # the call: revised away
        assert [text for _, text in sc.messages(thread)] == ["find me a hotel in Rome", "Three hotels in Rome."]
        log = sc.session(run.ref).log
        assert [e["kind"] for e in log] == ["session.entry.added", "session.entry.added",
                                            "session.revision.started", "session.revision.completed"]
        assert log[2]["options"] == {"reason": "requested", "keep": "tool results"}


class KeepsHandles(Agent):
    async def run(self, ctx):
        if ctx.run.parent is not None:
            return await ctx.run.wait_for_signal("go")
        mine = ctx.run.state
        if await mine.get("forks") is None:
            await mine.set("forks", {"a": await ctx.fork("a")})
            await ctx.step("after_fork", lambda: None)  # the worker dies here
        forks = await mine.get("forks")
        await forks["a"].signal("go", "went")  # a handle again: after a crash, too
        await forks["a"].wait()
        return type(forks["a"]).__name__


async def test_thread_handles_in_state_come_back_as_handles():
    async with Scenario({"k": KeepsHandles}) as sc:
        sc.crash(agent="k", after_step="after_fork")
        thread = await sc.thread("k")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        await sc.recover()
        await sc.settle()
        parent, fork = sc.runs(agent="k")
        assert parent.output == "_ThreadHandle" and fork.output == "went"
        stored = parent.run_state["forks"][0]["a"]
        assert type(stored) is ThreadRef and stored.id == fork.thread_id  # stored as a plain reference


class Watches(Agent):
    """Forks two subtasks; one fails without a word. The parent sees it from the run's end, and copies
    what it wants from the other's thread and state."""

    async def run(self, ctx) -> tuple[list[str], list[str], str, str | None] | None:
        if ctx.input.text == "breaks":
            raise AgentError("broken", "the build broke")
        if ctx.input.text == "works":
            await ctx.send("tried a, b, c")  # on its own thread
            await ctx.state.set("best", "c")
            return
        works, breaks = [await ctx.fork(t) for t in ("works", "breaks")]
        ended = sorted([str((await works.wait()).status), str((await breaks.wait()).status)])
        notes = [e.message.text for e in await works.thread_events(EventKind.MESSAGE)]
        return ended, notes, await works.state.get("best"), await ctx.state.get("best")


async def test_a_parent_sees_a_failed_fork_and_reads_another_forks_thread_and_state():
    async with Scenario({"w": Watches}) as sc:
        thread = await sc.thread("w")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        parent = sc.runs(thread)[0]
        assert parent.output == (["completed", "failed"],
                                 ["works", "tried a, b, c"],  # its thread after the fork: its input, its reply
                                 "c", None)  # the fork's state is the fork's: the parent copies what it wants


class LeavesSomethingRunning(Agent):
    async def run(self, ctx) -> str | None:
        if ctx.run.parent is None:
            await ctx.fork("background", detach=True)  # may outlive this run
            await ctx.send("started it, done for now")
            return None
        await ctx.sleep(60)
        await ctx.send("finished")  # on its own thread: the only one it writes on
        return "done"


async def test_a_detached_fork_outlives_its_parent_and_writes_only_on_its_own_thread():
    async with Scenario({"l": LeavesSomethingRunning}) as sc:
        thread = await sc.thread("l")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        before = len(sc.timeline(thread))
        parent, fork = sc.runs(agent="l")
        assert parent.status is RunStatus.COMPLETED and fork.status is RunStatus.WAITING
        sc.advance(60)
        await sc.settle()
        assert fork.output == "done"
        assert len(sc.timeline(thread)) == before  # nothing on the parent's thread, not even the fork's end


class Breaks(Agent):
    async def run(self, ctx) -> tuple[RunStatus, str | None, RunStatus, InvoiceOut] | InvoiceOut:
        if ctx.input.text == "break":
            raise AgentError("broken", "it broke")
        child = await (await ctx.fork("break")).wait()  # never raises
        pay = await (await ctx.call("pay", {"invoice_id": "i1"})).wait()
        return child.status, child.error, pay.status, pay.output


async def test_wait_gives_the_run_back_finished():
    async with Scenario({"b": Breaks, "pay": PayInvoice}) as sc:
        thread = await sc.thread("b")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        assert sc.runs(thread)[0].output == (RunStatus.FAILED, "broken", RunStatus.COMPLETED, InvoiceOut(paid=True))


# ── outputs: declared by run()'s return annotation ──

class Price(BaseModel):
    city: str
    per_day: int


class Quotes(Agent):
    async def run(self, ctx) -> list[Price]:  # checked at load, validated at the end, typed from wait()
        if ctx.run.parent is not None:
            return [{"city": "rome", "per_day": "160"}]  # validated into its type
        quote = await (await ctx.fork("x")).wait()
        return quote.output


class Silent(Agent):
    async def run(self, ctx) -> None:
        return "oops"  # declared: no output


class Untyped(Agent):
    async def run(self, ctx):  # no annotation: any storable value
        return {"open": object()}


class Client:
    pass


class ReturnsAClient(Agent):
    async def run(self, ctx) -> Client:
        return Client()


async def test_a_declared_output_is_validated_stored_as_json_and_typed_from_wait():
    async with Scenario({"q": Quotes}) as sc:
        thread = await sc.thread("q")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        parent, child = sc.runs(agent="q")
        assert parent.output == [Price(city="rome", per_day=160)]  # typed, through wait()
        completed = next(e for e in sc.timeline(ThreadRef(child.thread_id)) if e.kind == EventKind.RUN_COMPLETED)
        assert completed.data == {"output": [{"city": "rome", "per_day": 160}]}  # stored as JSON
        assert Quotes.describe("q").output_schema["items"]["$ref"].endswith("Price")


@pytest.mark.parametrize("agent, error", [(Silent, "output_undeclared"), (Untyped, "output_not_storable")])
async def test_an_output_that_cannot_be_stored_fails_the_run(agent, error):
    async with Scenario({"a": agent}) as sc:
        thread = await sc.thread("a")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.FAILED and run.error == error


async def test_an_output_type_that_cannot_be_stored_is_refused_before_any_run():
    with pytest.raises(ValueError, match="can't be stored as JSON"):
        Scenario({"c": ReturnsAClient})

    class Mismatch(PayInvoice):
        async def run(self, ctx) -> Price:
            ...

    with pytest.raises(ValueError, match="its Output is InvoiceOut"):
        Scenario({"m": Mismatch})


# ── talking to threads ──

class Briefed(Agent):
    async def run(self, ctx) -> list[tuple[str, bool]]:
        return [(m.text, ctx.session.is_own(m)) for m in await ctx.session.messages()]


class Briefs(Agent):
    async def run(self, ctx):
        await ctx.send("noted")
        briefed = await ctx.call("briefed", "your turn", session=await ctx.session.messages())
        events = [e async for e in briefed.watch()]  # live in the runtime; here once it has ended
        return (await briefed.wait()).output, [str(e.kind) for e in events]


async def test_another_agent_can_start_from_a_session_given_to_it_and_everything_in_it_is_someone_elses():
    async with Scenario({"b": Briefs, "briefed": Briefed}) as sc:
        thread = await sc.thread("b")
        await sc.say(thread, "hello", by=AN)
        await sc.settle()
        output, streamed = sc.runs(thread)[0].output
        assert output == [["hello", False], ["noted", False], ["your turn", False]]  # another agent's: none its own
        assert streamed == []  # it emitted nothing


class Snoops(Agent):
    async def run(self, ctx):
        await ctx.subscribe(Filter(thread=ThreadRef("thread-elsewhere"))).next()


async def test_a_run_listens_only_to_its_own_thread_and_threads_it_created():
    async with Scenario({"s": Snoops}) as sc:
        thread = await sc.thread("s")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        assert sc.runs(thread)[0].error == "not_your_thread"


class ListsForks(Agent):
    async def run(self, ctx) -> list[str] | int:
        if ctx.run.parent is not None:
            return len(ctx.forks)  # its own forks: none - its siblings are out of reach
        if not ctx.forks:
            for part in ("a", "b"):
                await ctx.fork(part)
            await ctx.step("forked", lambda: None)  # the worker dies here
            await ctx.restart()
        forks = ctx.forks  # after a crash and a restart: the runtime remembers them
        return [str((await f.wait()).output) for f in forks]


async def test_a_run_gets_its_forks_back_after_a_crash_and_a_restart():
    async with Scenario({"l": ListsForks}) as sc:
        sc.crash(agent="l", after_step="forked")
        thread = await sc.thread("l")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        await sc.recover()
        await sc.settle()
        parent = sc.runs(thread)[0]
        assert parent.output == ["0", "0"] and len(sc.forks(thread)) == 2  # two forks, not four


# ── the everyday surface ──

class Waits_(Agent):
    """ctx.next(*forks): whatever comes first - a message for me, or news from my forks."""

    async def run(self, ctx) -> list[str] | None:
        if ctx.run.parent is not None:
            await ctx.sleep(10 if ctx.input.text == "slow" else 1)
            await ctx.send(f"{ctx.input.text} done")
            return None
        if not ctx.forks:
            for task in ("quick", "slow"):
                await ctx.fork(task)
        seen = await ctx.run.state.get("seen", [])
        while len(seen) < 5:
            ev = await ctx.next(*ctx.forks)
            what = ev.message.text if ev.message else str(ev.kind)
            seen = await ctx.run.state.update("seen", lambda s: s + [what], [])
        return seen


async def test_next_gives_whatever_comes_first_from_this_thread_and_the_forks():
    async with Scenario({"w": Waits_}) as sc:
        thread = await sc.thread("w")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        sc.advance(1)
        await sc.settle()
        await sc.say(thread, "how is it going?", by=AN)  # while the slow fork still works
        await sc.settle()
        sc.advance(10)
        await sc.settle()
        assert sc.runs(thread)[0].output == ["quick done", "run.completed", "how is it going?",
                                              "slow done", "run.completed"]


class Closes(Agent):
    async def run(self, ctx) -> str:
        if ctx.run.parent is not None:
            await ctx.run.wait_for_signal("never")
        fork = await ctx.fork("wait forever")
        await fork.close()
        try:
            await fork.send("still there?")
        except ThreadClosed:
            return str((await fork.wait()).status)


async def test_closing_a_thread_stops_what_runs_there_and_takes_no_more_messages():
    async with Scenario({"c": Closes}) as sc:
        thread = await sc.thread("c")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        assert sc.runs(thread)[0].output == "cancelled"


class Reads(Agent):
    async def run(self, ctx) -> list[str]:
        await ctx.send("noted")
        return [m.text for m in await ctx.thread.messages()]  # the record of my thread


class EmitsAMessage(Agent):
    async def run(self, ctx):
        await ctx.emit(MessageCompleted(message=Message.create(Sender.agent(ctx.me.name), "hi")))  # not the way to send


async def test_a_run_reads_its_own_thread_and_emits_only_its_own_kinds_of_event():
    async with Scenario({"r": Reads, "e": EmitsAMessage}) as sc:
        thread = await sc.thread("r")
        await sc.say(thread, "hello", by=AN)
        await sc.settle()
        assert sc.runs(thread)[0].output == ["hello", "noted"]
        other = await sc.thread("e")
        await sc.say(other, "go", by=AN)
        await sc.settle()
        assert "is the runtime's to emit" in sc.runs(other)[0].error


class Counts(SessionProvider):
    """A provider with a policy of its own: revise after every second run; show only the agent's own."""

    def __init__(self):
        self.runs = 0

    def view(self, session):
        return [m for m in session.messages if m.sender.kind == SenderKind.AGENT]

    def after_run(self, session, run):
        self.runs += 1
        return Revise({"reason": "every second run"}) if self.runs % 2 == 0 else None

    async def revise(self, session, options):
        return session.messages[-1:]


class Answers(Agent):
    async def run(self, ctx) -> list[str]:
        await ctx.send(f"re: {ctx.input.text}")
        return [m.text for m in await ctx.session.messages()]


async def test_the_provider_decides_the_view_and_when_to_revise():
    async with Scenario({"a": Answers}, sessions={"a": Counts()}) as sc:
        thread = await sc.thread("a")
        for text in ("one", "two", "three"):
            await sc.say(thread, text, by=AN)
            await sc.settle()
        first, second, third = sc.runs(thread)
        assert second.output == ["re: one", "re: two"]  # its view: only the agent's own
        assert third.output == ["re: two", "re: three"]  # revised after the second run: kept the last one
        log = sc.session(third.ref).log
        assert [e["kind"] for e in log] == ["session.revision.started", "session.revision.completed"]
        assert log[0]["options"] == {"reason": "every second run"}


async def test_rolling_window_is_the_default_and_declines_a_revision():
    class Asks(Agent):
        async def run(self, ctx) -> int:
            await ctx.session.revise(reason="please")
            return len(await ctx.session.settled())

    async with Scenario({"a": Asks}, window=1) as sc:  # RollingWindow(1)
        thread = await sc.thread("a")
        await sc.say(thread, "hi", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.output == 1
        assert [e["kind"] for e in sc.session(run.ref).log] == ["session.revision.started", "session.revision.declined"]
