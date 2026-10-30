"""The 2.0 design's hypotheses, one test (or a few) each. See README.md."""

from __future__ import annotations

import asyncio
from collections import Counter

import pytest

from rig import (Agent, AgentCall, ChildSession, ChildThread, Crash, EventKind, Filter, Role, RunRef,
                 Runtime)

pytestmark = pytest.mark.asyncio


async def started(rt: Runtime, agent: str, text: str = "go") -> tuple[str, str]:
    """A thread with a top-level run of `agent`; returns (thread id, run id)."""
    thread = await rt.new_thread(agent)
    await rt.user_message(thread, text)
    await rt.settle()
    return thread, rt.runs_of(thread)[0].id


class Echo(Agent):
    async def run(self, ctx):
        await ctx.reply(f"done:{ctx.input[0]}")


class WaitsForMessage(Agent):
    """A child that waits for a message addressed to it, then answers with it."""
    async def run(self, ctx):
        msg = await ctx.next_input(wait=True)
        await ctx.reply(f"{ctx.input[0]}:{msg}")


# ── H1 Order: thread positions are gapless and increasing ───────────────────

async def test_h1_concurrent_writers_get_gapless_positions():
    async with Runtime({"echo": Echo}) as rt:
        thread = await rt.new_thread("echo")
        await asyncio.gather(*[rt.post(thread, "x-item", i) for i in range(200)])
        assert [e.pos for e in rt.read(thread)] == list(range(1, 201))
    # the Postgres side (bigserial vs per-thread position) is in test_pg_ordering.py


# ── H2 Idempotency: a retried write is a no-op ──────────────────────────────

class ReplyThenCrash(Agent):
    async def run(self, ctx):
        await ctx.reply("hello")
        await ctx.hook("crash")


async def test_h2_events_with_the_same_key_are_written_once():
    async with Runtime({"echo": Echo}) as rt:
        thread = await rt.new_thread("echo")
        await rt.post(thread, "x-item", 1, key="k")
        await rt.post(thread, "x-item", 1, key="k")
        assert len([e for e in rt.read(thread) if e.kind == "x-item"]) == 1


async def test_h2_a_reply_is_not_written_twice_after_a_crash():
    async with Runtime({"r": ReplyThenCrash}) as rt:
        async def crash():
            raise Crash()
        rt.hook("crash", crash)
        thread, run_id = await started(rt, "r")
        assert rt.runs[run_id].status == "running"  # the worker died mid-run
        rt.advance(31)
        assert await rt.recover() == 1
        await rt.settle()
        assert rt.runs[run_id].status == "completed"
        assert [e.data for e in rt.read(thread) if e.kind == "message" and e.role == "agent"] == ["hello"]


# ── H3 Chunks: no thread position, so streaming runs never wait on each other ──

class Streamer(Agent):
    async def run(self, ctx):
        for i in range(200):
            await ctx.emit_delta(str(i))
        await ctx.reply("final")


async def test_h3_chunks_are_run_events_and_take_no_thread_position():
    async with Runtime({"lead": Streamer, "s": Streamer}, workers=4) as rt:
        thread = await rt.new_thread("lead")
        await rt.user_message(thread, "a")
        await rt.user_message(thread, "b")  # unread when the first run completes: a second run (H12)
        await rt.settle()
        runs = rt.runs_of(thread)
        assert len(runs) == 2 and rt.metrics["run_events"] == 400
        assert rt.metrics["thread_appends"] == len(rt.read(thread)) < 10  # positions only for thread events
        assert all([seq for seq, _ in r.stream] == list(range(1, 201)) for r in runs)


# ── H4 A waiting run holds no worker ────────────────────────────────────────

class Lead(Agent):
    async def run(self, ctx):
        children = [await ctx.agents.start(AgentCall("echo", [i])) for i in range(3)]
        await ctx.reply([await c.result() for c in children])


async def test_h4_a_supervisor_and_its_children_finish_on_one_worker():
    async with Runtime({"lead": Lead, "echo": Echo}, workers=1) as rt:
        thread, run_id = await started(rt, "lead")
        assert rt.runs[run_id].status == "completed"  # a supervisor holding the worker would deadlock
        assert rt.runs[run_id].output == ["done:0", "done:1", "done:2"]


# ── H5 Same run, woken by a user message, a child's post, a child finishing ──

class Researcher(Agent):
    async def run(self, ctx):
        await ctx.post("x-finding", {"topic": ctx.input[0]})
        steer = await ctx.next_input(wait=True)  # waits for a message addressed to it
        await ctx.reply(f"{ctx.input[0]}:{steer}")


class Supervisor(Agent):
    async def run(self, ctx):
        state = ctx.restart_state
        if state is None:
            child = await ctx.agents.start(AgentCall("researcher", ["hotels"]))
            state = {"child": child.id, "log": []}
        child = ctx.agents.get(state["child"])
        ev = await ctx.subscribe(
            Filter(kinds=[EventKind.MESSAGE], roles=[Role.USER]),
            Filter(runs=[child], kinds=["x-*", EventKind.RUN_COMPLETED]),
        ).next()
        log = state["log"] + [ev.kind]
        if ev.kind == EventKind.MESSAGE:
            await child.send(ev.data)  # the user steers the child
        if ev.kind == EventKind.RUN_COMPLETED:
            await ctx.reply({"log": log, "result": ev.data})
            return
        await ctx.restart({"child": state["child"], "log": log})


async def test_h5_one_run_wakes_on_a_post_a_user_message_and_a_child_finishing():
    async with Runtime({"supervisor": Supervisor, "researcher": Researcher}) as rt:
        thread, sup = await started(rt, "supervisor", "plan my trip")
        assert rt.runs[sup].status == "waiting"
        await rt.user_message(thread, "cheaper")
        await rt.settle()
        assert rt.runs[sup].status == "completed"
        assert rt.runs[sup].output == {"log": ["x-finding", "message", "run.completed"], "result": "hotels:cheaper"}
        # no run per event, and the steering message started no new run
        assert Counter(r.agent for r in rt.runs_of(thread)) == {"supervisor": 1, "researcher": 1}


# ── H6 No missed wake-up: check and park in one transaction ─────────────────

class WaitsForPing(Agent):
    async def run(self, ctx):
        ev = await ctx.subscribe(Filter(kinds=["x-ping"])).next()
        await ctx.reply(ev.kind)


@pytest.mark.parametrize("atomic, expected", [(True, "completed"), (False, "waiting")])
async def test_h6_an_event_during_parking_wakes_the_run_only_when_parking_is_atomic(atomic, expected):
    async with Runtime({"w": WaitsForPing}, atomic_park=atomic) as rt:
        thread = await rt.new_thread("w")
        rt.hook("park_gap", lambda: rt.post(thread, "x-ping"))  # arrives while the run parks
        await rt.user_message(thread, "go")
        await rt.settle()
        assert rt.runs_of(thread)[0].status == expected  # "waiting" = the wake-up is lost forever


# ── H7 No double wake-up ────────────────────────────────────────────────────

class Collects(Agent):
    async def run(self, ctx):
        seen = []
        async for ev in ctx.subscribe(Filter(kinds=["x-ping"])):
            seen.append(ev.data)
            if len(seen) == 10:
                await ctx.reply(seen)
                return


async def test_h7_a_burst_of_events_activates_the_run_one_at_a_time():
    async with Runtime({"c": Collects}, workers=4) as rt:
        thread, run_id = await started(rt, "c")
        await asyncio.gather(*[rt.post(thread, "x-ping", i) for i in range(10)])
        await rt.settle()  # raises if a run was ever active on two workers
        run = rt.runs[run_id]
        assert run.output == list(range(10))
        assert run.activations <= 11


# ── H8 Recorded deliveries: a replay sees the same events in the same order ──

EFFECTS: Counter = Counter()


async def effect(item):
    EFFECTS[item] += 1


class Consumer(Agent):
    async def run(self, ctx):
        sub = ctx.subscribe(Filter(kinds=["x-item"]))
        got = []
        for _ in range(3):
            ev = await sub.next()
            await ctx.step("effect", effect, ev.data)
            got.append(ev.data)
            await ctx.hook(f"after_{ev.data}")
        await ctx.reply(got)


async def test_h8_a_crash_replays_the_recorded_deliveries_and_effects_run_once():
    EFFECTS.clear()
    async with Runtime({"c": Consumer}) as rt:
        thread = await rt.new_thread("c")

        async def more_then_crash():
            await rt.post(thread, "x-item", 2)
            await rt.post(thread, "x-item", 3)
            raise Crash()

        rt.hook("after_1", more_then_crash)
        await rt.user_message(thread, "go")
        await rt.post(thread, "x-item", 1)
        await rt.settle()
        rt.advance(31)
        await rt.recover()
        await rt.settle()
        run = rt.runs_of(thread)[0]
        assert run.output == [1, 2, 3]
        assert EFFECTS == {1: 1, 2: 1, 3: 1}


# ── H9 restart(state) bounds the replay ─────────────────────────────────────

N = 30


async def noop(_):
    return None


class LoopWithoutRestart(Agent):
    async def run(self, ctx):
        n = 0
        async for ev in ctx.subscribe(Filter(kinds=["x-tick"])):
            await ctx.step("handle", noop, ev.data)
            n += 1
            if ev.data == N:
                await ctx.reply(n)
                return


class LoopWithRestart(Agent):
    async def run(self, ctx):
        n = ctx.restart_state or 0
        ev = await ctx.subscribe(Filter(kinds=["x-tick"])).next()
        await ctx.step("handle", noop, ev.data)
        n += 1
        if ev.data == N:
            await ctx.reply(n)
            return
        await ctx.restart(n)


@pytest.mark.parametrize("agent", [LoopWithoutRestart, LoopWithRestart])
async def test_h9_replay_per_wake_up_grows_without_restart_and_stays_flat_with_it(agent):
    async with Runtime({"a": agent}) as rt:
        thread, run_id = await started(rt, "a")
        for i in range(1, N + 1):
            await rt.post(thread, "x-tick", i)
            await rt.settle()
        assert rt.runs[run_id].output == N
        replays = [n for rid, n in rt.activation_log if rid == run_id]
        if agent is LoopWithRestart:
            assert max(replays) == 0
        else:
            assert max(replays) >= 2 * (N - 1)  # every wake-up replays all before it
            assert sum(replays) > N * N / 2


# ── H10 Structured lifetime ─────────────────────────────────────────────────

class StartsAndReturns(Agent):
    async def run(self, ctx):
        for name in ("a", "b"):
            await ctx.agents.start(AgentCall("child", [name]))
        await ctx.reply("started")


async def test_h10_a_run_that_returns_waits_for_its_children():
    async with Runtime({"p": StartsAndReturns, "child": WaitsForMessage}) as rt:
        thread, parent = await started(rt, "p")
        assert rt.runs[parent].status == "waiting"
        children = rt.runs[parent].children
        await rt.user_message(thread, "x", to=RunRef(children[0]))
        await rt.settle()
        assert rt.runs[parent].status == "waiting"  # one child still active
        await rt.user_message(thread, "y", to=RunRef(children[1]))
        await rt.settle()
        assert rt.runs[parent].status == "completed"
        assert [rt.runs[c].output for c in children] == ["a:x", "b:y"]
        assert len([e for e in rt.read(thread) if e.kind == "message" and e.run_id == parent]) == 1  # replayed, not re-sent


class Tree(Agent):
    async def run(self, ctx):
        await ctx.agents.start(AgentCall("middle", ["m"]))
        await ctx.agents.start(AgentCall("child", ["detached"], detach=True))
        await ctx.subscribe(Filter(kinds=["x-never"])).next()


class Middle(Agent):
    async def run(self, ctx):
        await ctx.agents.start(AgentCall("child", ["leaf"]))
        await ctx.subscribe(Filter(kinds=["x-never"])).next()


async def test_h10_cancelling_a_run_cancels_its_tree_but_not_a_detached_child():
    async with Runtime({"tree": Tree, "middle": Middle, "child": WaitsForMessage}) as rt:
        thread, root = await started(rt, "tree")
        await rt.cancel(root)
        await rt.settle()
        status = {r.input[0] if r.parent_id else "root": r.status for r in rt.runs_of(thread)}
        assert status == {"root": "cancelled", "m": "cancelled", "leaf": "cancelled", "detached": "waiting"}
        detached = next(r for r in rt.runs_of(thread) if r.detached)
        activations = rt.runs[root].activations
        await rt.user_message(thread, "z", to=RunRef(detached.id))  # the child ends after its parent was cancelled
        await rt.settle()
        assert detached.status == "completed"
        assert rt.runs[root].status == "cancelled" and rt.runs[root].activations == activations  # woke nothing


# ── H11 start() is atomic with its step: a crash never duplicates a child ────

class StartThenCrash(Agent):
    async def run(self, ctx):
        child = await ctx.agents.start(AgentCall("echo", ["x"]))
        await ctx.hook("crash")
        await ctx.reply(await child.result())


async def test_h11_a_crash_after_start_does_not_start_the_child_twice():
    async with Runtime({"p": StartThenCrash, "echo": Echo}) as rt:
        async def crash():
            raise Crash()
        rt.hook("crash", crash)
        thread, parent = await started(rt, "p")
        rt.advance(31)
        await rt.recover()
        await rt.settle()
        assert rt.runs[parent].output == "done:x"
        assert Counter(r.agent for r in rt.runs_of(thread)) == {"p": 1, "echo": 1}


# ── H12 User messages: no subscriber starts a run; no gap at completion ──────

class Answers(Agent):
    async def run(self, ctx):
        await ctx.hook("before_return")
        await ctx.reply(f"answer to {ctx.input}")


async def test_h12_a_message_unread_at_completion_starts_a_new_run():
    async with Runtime({"a": Answers}) as rt:
        thread = await rt.new_thread("a")
        rt.hook("before_return", lambda: rt.user_message(thread, "second"))  # arrives while the run is active
        await rt.user_message(thread, "first")
        await rt.settle()
        await rt.user_message(thread, "third")  # nobody listens any more
        await rt.settle()
        assert [r.input for r in rt.runs_of(thread)] == [["first"], ["second"], ["third"]]
        assert all(r.status == "completed" for r in rt.runs_of(thread))


# ── H13 A child's question: parent or client, the first answer wins ─────────

class Asks(Agent):
    async def run(self, ctx):
        await ctx.reply(await ctx.ask("Approve?"))


class Approver(Agent):
    async def run(self, ctx):
        child = await ctx.agents.start(AgentCall("asks", ["x"]))
        await ctx.subscribe(Filter(runs=[child], kinds=[EventKind.RUN_INPUT_REQUIRED])).next()
        await ctx.hook("before_respond")
        ok = await child.respond("parent")
        await ctx.reply({"won": ok, "child": await child.result()})


@pytest.mark.parametrize("client_first", [False, True])
async def test_h13_the_first_answer_wins(client_first):
    async with Runtime({"p": Approver, "asks": Asks}) as rt:
        thread = await rt.new_thread("p")

        async def client_answers():
            child = next(r for r in rt.runs_of(thread) if r.agent == "asks")
            assert await rt.answer(child.id, child.pending, "client")

        if client_first:
            rt.hook("before_respond", client_answers)
        await rt.user_message(thread, "go")
        await rt.settle()
        parent = next(r for r in rt.runs_of(thread) if r.agent == "p")
        winner = "client" if client_first else "parent"
        assert parent.output == {"won": not client_first, "child": winner}


# ── H14 Sessions of child runs ──────────────────────────────────────────────

class Talks(Agent):
    async def run(self, ctx):
        await ctx.reply("thinking out loud")
        await ctx.reply(await ctx.session.prepare())


class Delegates(Agent):
    mode = ChildSession.NEW

    async def run(self, ctx):
        first = await ctx.agents.invoke(AgentCall("talks", ["q1"], session=self.mode))
        second = await ctx.agents.invoke(AgentCall("talks", ["q2"], session=self.mode))
        await ctx.reply({"first": first, "second": second, "mine": await ctx.session.prepare()})


class DelegatesToSpecialist(Delegates):
    mode = ChildSession.AGENT


class DelegatesWithFork(Delegates):
    mode = ChildSession.FORK


async def test_h14_a_new_session_holds_only_the_input_and_the_parent_sees_what_children_send():
    async with Runtime({"p": Delegates, "talks": Talks}) as rt:
        thread, parent = await started(rt, "p", "help me")
        out = rt.runs[parent].output
        assert out["first"] == [("input", "q1"), ("self", "thinking out loud")]  # not the user's message
        assert out["mine"] == [("user", "help me"),
                               ("child", "thinking out loud"), ("child", out["first"]),
                               ("child", "thinking out loud"), ("child", out["second"])]


async def test_h14_an_agent_session_remembers_across_delegations():
    async with Runtime({"p": DelegatesToSpecialist, "talks": Talks}) as rt:
        thread, parent = await started(rt, "p", "help me")
        out = rt.runs[parent].output
        assert out["second"][:3] == [("input", "q1"), ("self", "thinking out loud"), ("self", out["first"])]
        assert ("input", "q2") in out["second"]


async def test_h14_a_forked_session_starts_from_the_parent_and_never_changes_it():
    async with Runtime({"p": DelegatesWithFork, "talks": Talks}) as rt:
        thread, parent = await started(rt, "p", "help me")
        out = rt.runs[parent].output
        assert out["first"][0] == ("user", "help me")  # the parent's working memory, copied
        assert ("self", "thinking out loud") not in out["mine"]  # the child wrote only to its copy


# ── H15 Threads of child runs ───────────────────────────────────────────────

class Spawns(Agent):
    thread = ChildThread.SAME

    async def run(self, ctx):
        await ctx.reply(await ctx.agents.invoke(AgentCall("echo", ["x"], thread=self.thread)))


class SpawnsOnNewThread(Spawns):
    thread = ChildThread.NEW


class SpawnsOnFork(Spawns):
    thread = ChildThread.FORK


@pytest.mark.parametrize("agent", [Spawns, SpawnsOnNewThread, SpawnsOnFork])
async def test_h15_child_threads(agent):
    async with Runtime({"p": agent, "echo": Echo}) as rt:
        thread, parent = await started(rt, "p", "hello")
        assert rt.runs[parent].output == "done:x"  # the parent gets the result on every option
        child = next(r for r in rt.runs.values() if r.agent == "echo")
        mine = [e for e in rt.read(thread) if e.run_id == child.id]
        if agent.thread is ChildThread.SAME:
            assert child.thread_id == thread and any(e.kind == "message" for e in mine)
        else:
            assert child.thread_id != thread
            assert [e.kind for e in mine] == ["run.completed"]  # only its end is recorded here
            seen = [e.data for e in rt.read(child.thread_id) if e.kind == "message"]
            assert seen == (["hello", "done:x"] if agent.thread is ChildThread.FORK else ["done:x"])


# ── H16 send: on the run's own thread by default; `to` narrows it to one run ──

class Reports(Agent):
    async def run(self, ctx):
        await ctx.send("halfway")                      # on its thread: the parent sees it
        await ctx.send("note for you", to=ctx.parent)  # addressed: on the parent's thread
        await ctx.reply("done")


class HearsReports(Agent):
    thread = ChildThread.SAME
    expect = 3

    async def run(self, ctx):
        child = await ctx.agents.start(AgentCall("reports", ["x"], thread=self.thread))
        sub = ctx.subscribe(Filter(runs=[child], kinds=[EventKind.MESSAGE]),   # what the child says on this thread
                            Filter(kinds=[EventKind.MESSAGE], to=ctx.me))      # what it addresses to me
        heard = [(await sub.next()).data for _ in range(self.expect)]          # wakes on each, mid-run
        await ctx.reply({"heard": heard, "session": await ctx.session.prepare()})


class HearsReportsFromNewThread(HearsReports):
    thread = ChildThread.NEW
    expect = 1


@pytest.mark.parametrize("agent", [HearsReports, HearsReportsFromNewThread])
async def test_h16_a_child_reports_mid_run_and_the_parent_wakes_on_it(agent):
    async with Runtime({"p": agent, "reports": Reports}) as rt:
        thread, parent = await started(rt, "p", "go")
        out = rt.runs[parent].output
        if agent.thread is ChildThread.SAME:
            assert out["heard"] == ["halfway", "note for you", "done"]
            assert ("child", "halfway") in out["session"]  # mid-run updates are in the parent's session
        else:  # a child on its own thread: only what it addresses to the parent reaches the parent's thread
            assert out["heard"] == ["note for you"]
            assert [e.data for e in rt.read(thread) if e.kind == EventKind.MESSAGE and e.role == Role.AGENT
                    and e.run_id != parent] == ["note for you"]
