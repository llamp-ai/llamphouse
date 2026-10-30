"""Agents and workflows on FakeRun: the contract as agent code uses it."""

import pytest
from pydantic import BaseModel, ValidationError

from llamphouse.sdk import (
    ThreadRef,
    Identity,
    requirements_of,
    Scheduler,
    CapabilityMissing,
    requires,
    Agent,
    ArtifactPart,
    AuthRequest,
    Rejected,
    Request,
    RunStatus,
    StateConflict,
    AgentError,
    AgentType,
    Agents,
    Config,
    DataPart,
    EventKind,
    FilePart,
    Files,
    Filter,
    InputRequest,
    Message,
    MessageCompleted,
    Progress,
    Origin,
    Sender,
    SenderKind,
    Viewpoint,
    RunContext,
    RunRef,
    ThreadEvent,
    ToolCallPart,
    Usage,
    Workflow,
    job,
    step,
)
from llamphouse.sdk.testing import FakeAgents, FakeFiles, FakeRun, NotScripted


# ── an agent like the HR example ──

class HrConfig(Config):
    index: str
    top_k: int = 8


async def search(question: str, index: str, top_k: int) -> list[str]:
    return [] if index == "empty" else [f"{index}:{question}"]


class HrAssistant(Agent[HrConfig]):
    """Answers HR questions"""

    async def run(self, ctx: RunContext[HrConfig]) -> None:
        question = ctx.input.text
        docs = await ctx.step("search", search, question, ctx.config.index, ctx.config.top_k)
        if not docs:
            year = (await ctx.ask("Which year's policy do you mean?")).text
            docs = await ctx.step("search", search, f"{question} {year}", "archive", ctx.config.top_k)
        async with ctx.stream() as reply:
            for word in ("Here ", "you ", "go."):
                await reply.write(word)
        await ctx.emit(Usage(model="m", input_tokens=10, output_tokens=3))
        await ctx.send("Sources:", ArtifactPart(name="sources", parts=[DataPart(data=docs)]))


def test_the_config_schema_comes_from_the_generic():
    assert HrAssistant.config_class() is HrConfig
    assert set(HrAssistant.config_schema()["properties"]) == {"index", "top_k"}
    assert Agent.config_class() is Config
    assert HrAssistant.type is AgentType.AGENT


def test_config_is_strict():
    with pytest.raises(ValidationError):
        HrConfig(index="hr", unknown=1)


async def test_an_agent_that_asks_streams_and_returns_an_artifact():
    run = FakeRun("What is the leave policy?", config=HrConfig(index="empty"), answers=["2026"])
    await run.start(HrAssistant())

    assert [r.reason for r in run.input_requests] == ["Which year's policy do you mean?"]
    assert run.steps == {"search": 2}
    assert [str(e.kind) for e in run.emitted][:5] == ["message.started", "message.delta", "message.delta",
                                                      "message.delta", "message.completed"]
    assert run.replies[0].text == "Here you go."
    (sources,) = run.replies[1].artifacts  # prose and an artifact in one message
    assert run.replies[1].text == "Sources:" and sources.name == "sources"
    assert sources.data == ["archive:What is the leave policy? 2026"]
    assert run.events_of(EventKind.USAGE)[0].output_tokens == 3


async def test_an_unscripted_wait_fails_the_test_at_once():
    run = FakeRun("What is the leave policy?", config=HrConfig(index="empty"))
    with pytest.raises(NotScripted, match="no answer is scripted"):
        await run.start(HrAssistant())


# ── an async generator agent ──

class Yields(Agent):
    async def run(self, ctx):
        yield Progress(label="Thinking")
        yield MessageCompleted(message=Message.create(ctx.sender, "done"))


async def test_an_agent_may_yield_its_events():
    run = FakeRun("go")
    await run.start(Yields())
    assert [str(e.kind) for e in run.emitted] == ["progress", "message.completed"]


# ── @step ──

class UsesStepDecorator(Agent):
    @step
    async def fetch(self, ctx, query: str) -> str:
        return f"result for {query}"

    @step(name="store")
    async def save(self, ctx, value: str) -> None:
        await ctx.state.set("saved", value)

    async def run(self, ctx):
        value = await self.fetch(ctx, "x")
        await self.save(ctx, value)
        await ctx.send(value)


async def test_step_decorated_methods_go_through_ctx_step():
    run = FakeRun("go")
    await run.start(UsesStepDecorator())
    assert run.steps == {"fetch": 1, "store": 1}
    assert await run.state.get("saved") == "result for x"
    assert run.text == "result for x"


def test_job_marks_a_function_for_the_scheduler():
    @job("reindex")
    async def reindex(ctx, index: str) -> None: ...

    assert reindex.__llamphouse_job__ == "reindex"


# ── a workflow ──

class InvoiceIn(BaseModel):
    invoice_id: str
    amount: float


class InvoiceOut(BaseModel):
    paid: bool
    reference: str | None = None


class PayInvoice(Workflow[Config, InvoiceIn, InvoiceOut]):
    async def run(self, ctx) -> InvoiceOut:
        invoice = ctx.input.data
        if invoice.amount > 10_000:
            approval = await ctx.run.request_input(InputRequest(reason="approve_payment",
                                                            input_schema={"type": "object"}))
            if not approval.data["approved"]:
                return InvoiceOut(paid=False)
        ref = await ctx.step("pay", lambda: f"ref-{invoice.invoice_id}")
        await ctx.sleep(3600)
        await ctx.step("confirm", lambda: None)
        return InvoiceOut(paid=True, reference=ref)


def test_a_workflow_is_an_agent_with_a_type_and_schemas():
    assert issubclass(PayInvoice, Agent)
    assert PayInvoice.type is AgentType.WORKFLOW
    assert set(PayInvoice.input_schema()["properties"]) == {"invoice_id", "amount"}
    assert set(PayInvoice.output_schema()["properties"]) == {"paid", "reference"}


async def test_a_workflow_validates_its_input_and_returns_its_output():
    run = FakeRun({"invoice_id": "i1", "amount": 50_000}, answers=[{"approved": True}])
    out = await run.start(PayInvoice())
    assert out == InvoiceOut(paid=True, reference="ref-i1")
    assert run.input_requests[0].reason == "approve_payment"
    assert run.steps == {"pay": 1, "confirm": 1} and run.sleeps == [3600]


async def test_a_workflow_takes_its_input_from_a_data_part():
    run = FakeRun(Message.create(Sender.service("erp"), DataPart(data={"invoice_id": "i2", "amount": 10})))
    assert await run.start(PayInvoice()) == InvoiceOut(paid=True, reference="ref-i2")


async def test_bad_workflow_input_never_starts_the_run():
    run = FakeRun({"invoice_id": "i1"})  # no amount
    with pytest.raises(ValidationError):
        await run.start(PayInvoice())
    assert run.steps == {}


class ReturnsWrongOutput(PayInvoice):
    async def run(self, ctx):
        return {"paid": "not a bool"}


async def test_a_workflow_output_is_validated():
    with pytest.raises(AgentError) as e:
        await FakeRun({"invoice_id": "i", "amount": 1}).start(ReturnsWrongOutput())
    assert e.value.code == "output_invalid"


# ── other agents and forks ──

class Supervisor(Agent):

    async def run(self, ctx):
        hotels = await ctx.call("hotels", ctx.input)  # a new thread of another agent
        await hotels.send("cheaper please")  # the same thread
        flights = await ctx.call("flights", ctx.input)
        check = await ctx.fork("check the visa rules")  # a fork of this thread
        for t in (hotels, flights, check):
            await t.wait()
        said = [(await t.messages())[-1].text for t in (hotels, flights, check)]
        await ctx.send(" / ".join(said))


async def test_a_supervisor_talks_to_other_agents_and_forks_and_listens():
    agents = FakeAgents({"hotels": "Hotel A", "flights": lambda first: f"Flight for {first.text}"})
    run = FakeRun("Bangkok", agents=agents, forks=lambda first: f"done: {first.text}")
    await run.start(Supervisor())
    assert run.text == "Hotel A / Flight for Bangkok / done: check the visa rules"
    hotels, flights = agents.threads
    assert (hotels.agent, flights.agent) == ("hotels", "flights")
    assert [m.text for m in hotels.sent] == ["Bangkok", "cheaper please"]  # what it sent there
    (fork,) = run.forked
    assert fork.forked_from[0] == run.thread and fork.agent == "agent"


async def test_a_missing_required_capability_refuses_the_start():
    with pytest.raises(AssertionError, match="missing \\['Files'\\]"):
        await FakeRun("x").start(StoresAFile())


# ── subscriptions, inputs, restart ──

class Counts(Agent):
    """A long-lived loop: one event per iteration, restarting each time."""

    async def run(self, ctx):
        seen = await ctx.run.state.get("seen", [])
        ev = await ctx.subscribe(Filter(kinds=["x-tick"]), *ctx.inbox).next(timeout=0)
        if ev is None:
            await ctx.send(f"saw {seen}")
            return
        await ctx.run.state.set("seen", seen + [ev.data if ev.message is None else ev.message.text])
        await ctx.restart()


async def test_run_state_survives_a_restart_and_subscriptions_deliver_in_order():
    run = FakeRun("go", events=[
        ThreadEvent(position=1, kind="x-tick", data=1),
        ThreadEvent(position=2, kind=EventKind.MESSAGE, to=RunRef("run-1"),  # sent to the run: its inbox
                    message=Message.create(Sender.person("u1"), "hi")),
        ThreadEvent(position=3, kind=EventKind.MESSAGE, run=RunRef("run-1"),  # its own message: not in the inbox
                    message=Message.create(Sender.agent("agent", run=RunRef("run-1")), "mine")),
        ThreadEvent(position=4, kind="x-other", data=0),
        ThreadEvent(position=5, kind="x-tick", data=2),
    ])
    await run.start(Counts())
    assert run.text == "saw [1, 'hi', 2]"
    assert run.restarts == 3


class ReadsSteering(Agent):
    async def run(self, ctx):
        notes = []
        while (ev := await ctx.next(timeout=0)) is not None:
            notes.append(ev.message.text)
        await ctx.send(", ".join(notes))


async def test_next_reads_the_messages_for_the_run():
    run = FakeRun("go", inputs=["first", "second"])
    await run.start(ReadsSteering())
    assert run.text == "first, second"


class CallsClientTool(Agent):
    async def run(self, ctx):
        result = await ctx.client.call_tool(ToolCallPart(call_id="c1", name="pick_date", arguments={}))
        await ctx.send(result.output[0].text)


async def test_client_tools_are_scripted():
    run = FakeRun("go", client_results=["2026-11-01"])
    await run.start(CallsClientTool())
    assert run.client_calls[0].name == "pick_date" and run.text == "2026-11-01"


@requires(Files)
class StoresAFile(Agent):
    async def run(self, ctx):
        part = await ctx.capability(Files).put(b"report", mime_type="text/plain", name="r.txt")
        await ctx.send(ArtifactPart(name="report", parts=[part]))


async def test_files_capability_and_artifacts():
    files = FakeFiles()
    run = FakeRun("go", capabilities=[files])
    await run.start(StoresAFile())
    (report,) = run.replies[0].artifacts
    assert report.name == "report" and await files.get(report.parts[0]) == b"report"


class Fails(Agent):
    async def run(self, ctx):
        raise AgentError("no_policy", "No policy found", retryable=False)


async def test_agent_errors_carry_a_code():
    with pytest.raises(AgentError) as err:
        await FakeRun("x").start(Fails())
    assert err.value.code == "no_policy" and not err.value.retryable


# ── A2A v1.0 parity: rejected, auth_required, request info, skills, extensions, artifacts ──

class Picky(Agent):
    async def run(self, ctx):
        if "image/png" not in ctx.run.request.accepts and ctx.run.request.accepts:
            await ctx.reject("I only produce charts as PNG")
        await ctx.send("chart")


async def test_an_agent_can_reject_the_work():
    run = FakeRun("chart please", request=Request(accepts=["text/plain"]))
    await run.start(Picky())
    assert run.rejected == "I only produce charts as PNG" and run.sent == []


async def test_raising_rejected_is_the_same():
    class Declines(Agent):
        async def run(self, ctx):
            raise Rejected("out of scope")

    run = FakeRun("x")
    await run.start(Declines())
    assert run.rejected == "out of scope"


class NeedsCalendar(Agent):
    async def run(self, ctx):
        grant = await ctx.run.request_auth(AuthRequest(reason="read your calendar", scheme="oauth2",
                                                   scopes=["calendar.read"]))
        await ctx.send(f"token {grant.data['token']}")


async def test_an_agent_can_wait_for_credentials():
    run = FakeRun("book a meeting", auth=[{"token": "t-1"}])
    await run.start(NeedsCalendar())
    assert run.auth_requests[0].scopes == ["calendar.read"] and run.text == "token t-1"
    with pytest.raises(NotScripted, match="credentials"):
        await FakeRun("book a meeting").start(NeedsCalendar())


class Charts(Agent[HrConfig]):
    """Draws charts.

        Bar charts only, for now.
    """
    version = "1.2.0"

    async def run(self, ctx):
        async with ctx.stream() as m:
            await m.write("part 1, ")
            await m.write("part 2")
        await ctx.send("see the chart", ArtifactPart(name="chart", description="Sales per month",
                                                      parts=[DataPart(data={"bars": [3, 5]})]),
                        references=[RunRef("earlier-run")])


def test_an_agent_describes_what_its_code_knows():
    info = Charts.describe("charts")
    assert (info.name, info.version, info.type, info.description) == ("charts", "1.2.0", AgentType.AGENT,
                                                                       "Draws charts.\n\nBar charts only, for now.")
    assert set(info.config_schema["properties"]) == {"index", "top_k"}
    workflow = PayInvoice.describe("pay")
    assert workflow.type is AgentType.WORKFLOW and set(workflow.input_schema["properties"]) == {"invoice_id", "amount"}


def test_the_description_is_the_docstring_unless_set_and_never_inherited():
    class Explicit(Charts):
        """Developer notes, not published."""
        description = "Draws charts for reports"

    class Undocumented(Charts):
        pass

    assert Explicit.describe("x").description == "Draws charts for reports"
    assert Undocumented.describe("x").description == ""  # not the docstring of Charts, nor of Agent


async def test_a_streamed_message_then_an_artifact_that_references_earlier_runs():
    run = FakeRun("x", config=HrConfig(index="i"))
    await run.start(Charts())
    streamed, note = run.replies
    assert streamed.text == "part 1, part 2"
    assert note.text == "see the chart" and note.artifacts[0].description == "Sales per month"
    assert [str(e.kind) for e in run.emitted][:3] == ["message.started", "message.delta", "message.delta"]
    assert note.references == (RunRef("earlier-run"),)


def test_new_run_states_are_known():
    assert RunStatus.REJECTED == "rejected" and RunStatus.AUTH_REQUIRED == "auth_required"
    assert EventKind.RUN_REJECTED == "run.rejected" and EventKind.RUN_AUTH_REQUIRED == "run.auth_required"


class LogsItsVersion(Agent):
    version = "2.0.1"

    async def run(self, ctx):
        await ctx.send(f"{ctx.me.version} r{ctx.me.revision}")


async def test_the_run_knows_its_agent_version_and_revision():
    run = FakeRun("x", agent_revision=3)
    await run.start(LogsItsVersion())
    assert run.text == "2.0.1 r3"
    assert Agent.version is None  # unversioned by default


# ── senders, viewpoints, the inbox ──

class Moderates(Agent):
    """In a group chat: builds a model prompt from the thread, with roles derived per reader."""

    async def run(self, ctx):
        prompt = []
        for m in await ctx.session.messages():
            role = ("assistant" if ctx.session.is_own(m) else
                    "system" if m.sender.kind == SenderKind.SYSTEM else "user")
            prompt.append((role, m.sender.name or m.sender.id, m.text))
        await ctx.send(DataPart(data=prompt))


async def test_roles_are_derived_for_the_reader():
    history = [
        Message.create(Sender.system(), "Keep it short"),
        Message.create(Sender.person("u1", name="An"), "Plan a trip"),
        Message.create(Sender.agent("moderator", run=RunRef("turn-1")), "Who takes hotels?"),  # its earlier turn
        Message.create(Sender.agent("hotels-bot", run=RunRef("bot-1")), "I do"),  # another agent: a user, for it
        Message.create(Sender.agent("moderator", run=RunRef("sub-1")), "a child of the same agent"),  # OTHER
    ]
    run = FakeRun(Message.create(Sender.person("u1", name="An"), "And a hotel"), conversation=history,
                  agent="moderator", earlier_runs=[RunRef("turn-1")])
    await run.start(Moderates())
    assert run.replies[0].data == [("system", "system", "Keep it short"), ("user", "An", "Plan a trip"),
                                   ("assistant", "moderator", "Who takes hotels?"), ("user", "hotels-bot", "I do"),
                                   ("user", "moderator", "a child of the same agent"), ("user", "An", "And a hotel")]
    assert run.replies[0].sender == Sender.agent("moderator", run=RunRef("run-1"))


# ── requires, the inbox, versioned state ──

def test_requires_is_published():
    assert StoresAFile.describe("files-agent").requires == ("Files",)
    assert Agent.describe("plain").requires == ()


async def test_a_runs_inbox_is_what_is_sent_to_it():
    class Shows(Agent):
        async def run(self, ctx) -> tuple[Filter, ...]:  # dataclasses are storable too
            return ctx.inbox

    run = FakeRun("x")
    (sent,) = await run.start(Shows())
    assert sent == Filter(kinds=[EventKind.MESSAGE], to=run.run)


async def test_state_has_versions_and_conflicts():
    class Counts(Agent):
        async def run(self, ctx) -> tuple[str, str, int, int]:
            v1 = await ctx.state.set("plan", "draft")
            value, version = await ctx.state.get_versioned("plan")
            await ctx.state.set("plan", "final", if_version=version)
            try:
                await ctx.state.set("plan", "stale", if_version=v1)
            except StateConflict as conflict:
                return value, await ctx.state.get("plan"), conflict.actual, await ctx.state.update("n", lambda n: n + 1, 0)

    assert await FakeRun("x").start(Counts()) == ("draft", "final", 2, 1)


async def test_run_state_batches_and_the_state_at_start():
    class Plans(Agent):
        async def run(self, ctx) -> tuple[str, str, int, None]:
            async with ctx.state.batch() as s:
                s.set("month", "june")
                s.set("cities", ["rome"])
            await ctx.run.state.set("step", 2)
            return (await ctx.state.get("month"), await ctx.state.at_start().get("month"),
                    await ctx.run.state.get("step"), await ctx.state.get("step"))

    run = FakeRun("x", state={"month": "march"})
    assert await run.start(Plans()) == ("june", "march", 2, None)  # run state is the run's alone
    assert await run.run.state.get("step") == 2 and await run.state.get("cities") == ["rome"]


async def test_messages_is_the_conversation_in_order():
    class Talks(Agent):
        async def run(self, ctx):
            await ctx.send("first answer")
            await ctx.next()
            await ctx.send("second answer")
            return [m.text for m in await ctx.session.messages()]

    run = FakeRun("question", conversation=["earlier"], inputs=["follow-up"])
    assert await run.start(Talks()) == ["earlier", "question", "first answer", "follow-up", "second answer"]
    assert run.input.text == "question" and run.input.sender.kind == SenderKind.PERSON


async def test_file_bytes_are_stored_before_a_message_is_recorded():
    class Exports(Agent):
        async def run(self, ctx):
            await ctx.send(ArtifactPart(name="out", parts=[FilePart(mime_type="text/csv", name="out.csv",
                                                                     data=b"a,b\n1,2\n")]))

    files = FakeFiles()
    run = FakeRun("export", capabilities=[files])
    await run.start(Exports())
    stored = run.replies[0].artifacts[0].parts[0]
    assert stored.data is None and stored.uri.startswith("files://") and stored.size == 8 and len(stored.sha256) == 64
    assert await files.get(stored) == b"a,b\n1,2\n"


# ── @requires and ctx.require ──

@requires(Files)
class Reporter(Agent):
    async def run(self, ctx):
        files = ctx.require(Files)  # never None for a declared requirement
        await ctx.send(type(files).__name__)


class DailyReporter(Reporter):  # inherits Files
    pass


@requires(Scheduler)
class ScheduledReporter(Reporter):  # adds Scheduler to the inherited Files
    pass


def test_requires_is_inherited_and_added_to():
    assert requirements_of(Reporter) == {Files}
    assert requirements_of(DailyReporter) == {Files}
    assert requirements_of(ScheduledReporter) == {Files, Scheduler}
    assert requirements_of(Reporter()) == {Files}  # an instance too
    assert requirements_of(Agent) == frozenset() and not hasattr(Agent, "requires")  # nothing on Agent itself
    assert ScheduledReporter.describe("r").requires == ("Files", "Scheduler")


def test_requires_refuses_what_is_not_a_capability():
    with pytest.raises(TypeError, match="always available"):
        requires(Agents)
    with pytest.raises(TypeError, match="Agent class"):
        requires(Files)(dict)


async def test_require_gives_the_capability_or_a_clear_error():
    run = FakeRun("x", capabilities=[FakeFiles()])
    await run.start(Reporter())
    assert run.text == "FakeFiles"

    class Optional_(Agent):
        async def run(self, ctx):
            return ctx.capability(Scheduler), ctx.require(Scheduler)

    with pytest.raises(CapabilityMissing):
        await FakeRun("x").start(Optional_())


async def test_me_run_thread_and_caller():
    class Introduces(Agent):
        version = "3.1.0"

        async def run(self, ctx) -> tuple[str, AgentType, str, str, RunRef, ThreadRef, str]:
            return (str(ctx.me), ctx.me.type, ctx.me.display_name, ctx.me.version, ctx.run, ctx.thread,
                    ctx.run.caller.subject)

    run = FakeRun("hi", agent="hr-thailand", display_name="HR Thailand", run_id="r7", thread_id="t3",
                  caller=Identity(subject="an"))
    assert await run.start(Introduces()) == ("hr-thailand", AgentType.AGENT, "HR Thailand", "3.1.0",
                                             RunRef("r7"), ThreadRef("t3"), "an")


async def test_fakerun_forwards_a_message_of_the_conversation_and_signs_any_other():
    class Relays(Agent):
        async def run(self, ctx):
            chat = await ctx.call("echo", ctx.input)  # forwarded: the person did say it
            await chat.send(Message.create(Sender.person("test-user"), "made up"))  # signed by this run

    agents = FakeAgents({"echo": "ok"})
    run = FakeRun("hello", agents=agents)
    await run.start(Relays())
    forwarded, made_up = agents.threads[0].sent
    assert forwarded.sender.kind == SenderKind.PERSON and forwarded.id == run.input.id
    assert made_up.sender == run.sender and made_up.text == "made up"


async def test_fakerun_keeps_session_entries_off_the_thread_and_records_revisions():
    class Works(Agent):
        async def run(self, ctx) -> int:
            await ctx.session.add(ToolCallPart(call_id="c1", name="search", arguments={}))
            await ctx.send("found it")
            await ctx.session.revise(reason="tidy")
            return len(await ctx.session.settled())

    run = FakeRun("find it")
    assert await run.start(Works()) == 3  # the input, the entry, the reply
    assert [m.text for m in run.replies] == ["found it"]  # the entry isn't a reply on the thread
    assert len(run.session_entries) == 1 and run.revisions == [{"reason": "tidy"}]
