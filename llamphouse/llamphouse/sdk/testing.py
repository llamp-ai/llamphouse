"""Test agents without the runtime.

Two tools: `FakeRun` for one agent against a scripted world, and `Scenario`
(see `llamphouse.sdk.scenario`) for several real agents on shared threads.

`FakeRun` is an in-memory `RunContext`: you script what the outside world
does (answers to input requests, messages for the run, thread events,
signals, client tool results, what other agents and forks reply) and it
records everything the agent does (events, messages, posts, steps, requests,
sleeps, forks and the threads it talked to).

    run = FakeRun("What is the leave policy?", config=HrConfig(...), answers=["2026"])
    await run.start(HrAssistant())
    assert run.replies[-1].text == "..."

Nothing waits: a request with no scripted answer fails the test at once
(`NotScripted`), so a test shows exactly what the agent asked for.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Mapping, NoReturn, Sequence, TypeVar, Union

from pydantic import BaseModel

from .agent import Agent, Config, Workflow, requirements_of
from .capabilities import Agents, CapabilityMissing, AgentInfo, Files, RunResult, Scheduler, SessionSeed, Thread
from .content import (ArtifactPart, Content, DataPart, FilePart, Message, Metadata, Part, Sender, TextPart, ToolCallPart,
                      ToolResultPart, as_input, map_file_parts, to_parts, viewpoint)
from .context import ThisRun, ThisThread
from .events import AGENT_EVENTS, Event, MessageCompleted, MessageStarted, PartDelta, Rejected
from .thread import (AuthRequest, Filter, Identity, InputRequest, Request, SessionView, StateConflict, ThreadEvent,
                     ToolSpec)
from .card import AgentIdentity
from .vocab import AgentType, EventKind, RunRef, RunStatus, ThreadRef, Viewpoint

T = TypeVar("T")

from .scenario import Crash, RunRecord, Scenario, SessionRecord  # noqa: E402  (several agents, in memory)

__all__ = ["FakeRun", "FakeAgents", "FakeThread", "FakeFiles", "FakeScheduler", "NotScripted", "Sent",
           "Scenario", "Crash", "RunRecord", "SessionRecord"]


class NotScripted(AssertionError):
    """The agent waited for something the test didn't script."""


class _Restart(BaseException):
    pass


class _Handover(BaseException):
    def __init__(self, agent: str, session: SessionSeed):
        self.agent, self.session = agent, session


@dataclass(frozen=True)
class Sent:
    """A message the run wrote on its thread."""

    message: Message


Script = Union[str, Message, Mapping[str, Any], Any]


TEST_USER = Sender.person("test-user", name="Test user")


def _as_message(value: Script, sender: Sender = TEST_USER) -> Message:
    """Scripted values: a string is text, a Message is used as is, anything else is data.
    Scripted messages come from outside the run (a person, by default)."""
    if isinstance(value, Message):
        return value
    if isinstance(value, (str, Part)):
        return Message.create(sender, value)
    return Message.create(sender, DataPart(data=value))


class _Subscription:
    def __init__(self, run: "FakeRun", filters: tuple[Filter, ...]):
        self.run, self.filters = run, filters

    async def next(self, timeout: float | None = None) -> ThreadEvent | None:
        for i, ev in enumerate(self.run._events):
            if i not in self.run._delivered and any(f.matches(ev, agent=self.run._agent)
                                                     for f in self.filters):
                self.run._delivered.add(i)
                return ev
        if timeout is not None:
            return None
        raise NotScripted(f"the run waits for a thread event matching {self.filters}, and none is scripted")

    def __aiter__(self) -> AsyncIterator[ThreadEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ThreadEvent]:
        while (ev := await self.next(timeout=0)) is not None:  # ends when the script does
            yield ev

    async def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, run: "FakeRun"):
        self.run = run

    @property
    def id(self) -> str:
        return f"session-{self.run.run.id}"

    async def messages(self) -> SessionView:
        """The scripted conversation, then the input, then what arrived, what the run said and added."""
        return SessionView(self.run._conversation, id=self.id, version=len(self.run._conversation))

    async def add(self, *content: Content) -> Message:
        entry = Message.create(self.run.sender, *content)
        self.run._conversation.append(entry)
        self.run.session_entries.append(entry)
        return entry

    async def revise(self, **options: Any) -> None:
        self.run.revisions.append(options)

    async def settled(self) -> SessionView:
        return await self.messages()

    def is_own(self, message: Message) -> bool:
        return viewpoint(message, own={self.run.run, *self.run._earlier_runs}) is Viewpoint.OWN


class _View:
    """State to read, in memory."""

    def __init__(self, values: Mapping[str, Any] | None):
        self.values: dict[str, Any] = dict(values or {})
        self.versions: dict[str, int] = {k: 1 for k in self.values}

    async def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    async def get_versioned(self, key: str) -> tuple[Any, int]:
        return self.values.get(key), self.versions.get(key, 0)


class _State(_View):
    """State with versions, in memory: the thread's (`ctx.state`) or the run's (`ctx.run.state`)."""

    def __init__(self, values: Mapping[str, Any] | None):
        super().__init__(values)
        self._start = _View(values)  # the state as the run found it

    def _check(self, key: str, if_version: int | None) -> None:
        if if_version is not None and if_version != self.versions.get(key, 0):
            raise StateConflict(key, if_version, self.versions.get(key, 0))

    async def set(self, key: str, value: Any, *, if_version: int | None = None) -> int:
        self._check(key, if_version)
        self.values[key] = value
        self.versions[key] = self.versions.get(key, 0) + 1
        return self.versions[key]

    async def update(self, key: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        value = fn(self.values.get(key, default))
        await self.set(key, value)
        return value

    async def delete(self, key: str, *, if_version: int | None = None) -> None:
        self._check(key, if_version)
        self.values.pop(key, None)
        self.versions.pop(key, None)

    def batch(self) -> "_Batch":
        return _Batch(self)

    def at_start(self) -> _View:
        return self._start


class _Batch:
    def __init__(self, state: _State):
        self.state, self.writes = state, []

    def set(self, key: str, value: Any, *, if_version: int | None = None) -> None:
        self.writes.append((key, value, if_version, False))

    def delete(self, key: str, *, if_version: int | None = None) -> None:
        self.writes.append((key, None, if_version, True))

    async def __aenter__(self) -> "_Batch":
        return self

    async def __aexit__(self, exc_type: Any, *exc: Any) -> None:
        if exc_type is not None:
            return
        for key, _, if_version, _ in self.writes:  # all or nothing
            self.state._check(key, if_version)
        for key, value, _, delete in self.writes:
            await (self.state.delete(key) if delete else self.state.set(key, value))


class _FakeThisRun(ThisRun):
    def __init__(self, run: "FakeRun"):
        super().__init__(id=run._run_id)
        object.__setattr__(self, "_run", run)

    parent = property(lambda self: self._run._parent)
    depth = property(lambda self: self._run._depth)
    caller = property(lambda self: self._run._caller)
    request = property(lambda self: self._run._request)
    cancelled = property(lambda self: self._run._cancelled)
    state = property(lambda self: self._run._run_state)

    async def request_input(self, request: InputRequest) -> Message:
        return await self._run._request_input(request)

    async def request_auth(self, request: AuthRequest) -> Message:
        run = self._run
        run.auth_requests.append(request)
        if not run._auth:
            raise NotScripted(f"the run asks for credentials ({request.reason}), and none are scripted")
        return _as_message(run._auth.popleft())

    async def wait_for_signal(self, key: str, *, timeout: float | None = None) -> Any:
        run = self._run
        run.signals_waited.append(key)
        if key not in run._signals:
            if timeout is not None:
                return None
            raise NotScripted(f"the run waits for signal {key!r}, and none is scripted")
        return run._signals[key]


class _FakeThisThread(ThisThread):
    def __init__(self, run: "FakeRun"):
        super().__init__(id=run._thread_id)
        object.__setattr__(self, "_run", run)

    agent = property(lambda self: self._run._agent)
    forked_from = property(lambda self: None)
    metadata = property(lambda self: dict(self._run._thread_metadata))

    async def messages(self) -> Sequence[Message]:
        return list(self._run._conversation)


class _FakeClient:
    def __init__(self, run: "FakeRun"):
        self.run = run

    tools = property(lambda self: self.run._client_tools)

    async def call_tool(self, call: ToolCallPart) -> ToolResultPart:
        run = self.run
        run.client_calls.append(call)
        if not run._client_results:
            raise NotScripted(f"the run calls client tool {call.name!r}, and no result is scripted")
        result = run._client_results.popleft()
        if isinstance(result, ToolResultPart):
            return result
        return ToolResultPart(call_id=call.call_id, output=to_parts([result]) if isinstance(result, (str, Part))
                              else [DataPart(data=result)])


class _Writer:
    def __init__(self, run: "FakeRun"):
        self.run = run
        self.message_id = f"msg_{next(run._ids)}"
        self.parts: list[Part] = []
        self._text: list[str] = []

    async def __aenter__(self) -> "_Writer":
        self.run.emitted.append(MessageStarted(message_id=self.message_id))
        return self

    async def write(self, text: str) -> None:
        if text:
            self._text.append(text)
            self.run.emitted.append(PartDelta(message_id=self.message_id, index=0, delta=text))

    async def add(self, part: Part) -> None:
        self.parts.append(part)

    async def __aexit__(self, exc_type: Any, *exc: Any) -> None:
        if exc_type is None:
            parts = ([TextPart(text="".join(self._text))] if self._text else []) + self.parts
            message = Message(id=self.message_id, sender=self.run.sender, parts=parts)
            await self.run._deliver_message(message)


class FakeRun:
    """An in-memory RunContext. See the module docstring."""

    def __init__(
        self,
        input: Script = None,
        *,
        config: Config | None = None,
        answers: Iterable[Script] = (),
        auth: Iterable[Script] = (),
        inputs: Iterable[Script] = (),
        events: Iterable[ThreadEvent] = (),
        signals: Mapping[str, Any] | None = None,
        client_results: Iterable[ToolResultPart | Any] = (),
        conversation: Iterable[Script] = (),  # the conversation before this run, as ctx.session.messages() shows it
        state: Mapping[str, Any] | None = None,  # the thread's state as the run finds it
        run_state: Mapping[str, Any] | None = None,  # its own (as after a restart)
        agents: "FakeAgents | None" = None,
        forks: "Reply | None" = None,  # what this run's forks reply (a value, or a function of the first message)
        capabilities: Iterable[Any] = (),
        caller: Identity | None = None,
        display_name: str | None = None,
        client_tools: Sequence[ToolSpec] = (),
        request: Request | None = None,
        agent: str = "agent",
        agent_version: str | None = None,  # default: the agent's own `version`
        agent_revision: int | None = None,
        earlier_runs: Iterable[RunRef] = (),  # earlier runs sharing this run's session: their messages are OWN
        run_id: str = "run-1",
        thread_id: str = "thread-1",
        thread_metadata: Mapping[str, Any] | None = None,
        parent: RunRef | None = None,
        depth: int = 0,
        max_restarts: int = 100,
    ):
        self._input = as_input(input, TEST_USER)
        self._config = config if config is not None else Config()
        self._answers = deque(answers)
        self._auth = deque(auth)
        self._request = request or Request()
        self._inputs = deque(_as_message(i) for i in inputs)
        self._events = list(events)
        self._delivered: set[int] = set()
        self._signals = dict(signals or {})
        self._client_results = deque(client_results)
        self._conversation = [_as_message(m) for m in conversation]
        self._state = _State(state)
        self._run_state = _State(run_state)
        self._capabilities = {c.capability: c for c in capabilities}
        self._agents_fake = agents if agents is not None else FakeAgents()
        self._fork_reply = forks
        self._files = self._capabilities.get(Files) or FakeFiles()  # where sent file bytes are stored
        self._inbox: tuple[Filter, ...] = ()
        self._caller = caller or Identity(subject="test")
        self._display_name, self._agent_type = display_name, AgentType.AGENT
        self._client_tools = list(client_tools)
        self._agent, self._agent_version, self._run_id, self._thread_id = agent, agent_version, run_id, thread_id
        self._agent_revision = agent_revision
        self._thread_metadata = dict(thread_metadata or {})
        self._earlier_runs = set(earlier_runs)
        for fake in {id(f): f for f in [self._agents_fake, self._files, *self._capabilities.values()]}.values():
            if hasattr(fake, "bind"):
                fake.bind(self)
        self._parent, self._depth = parent, depth
        self._max_restarts = max_restarts
        self._cancelled = False
        self._ids = itertools.count(1)
        self._positions = itertools.count(1)
        # what the agent did
        self.emitted: list[Event] = []
        self.sent: list[Sent] = []
        self.posts: list[ThreadEvent] = []
        self.steps: Counter = Counter()
        self.input_requests: list[InputRequest] = []
        self.auth_requests: list[AuthRequest] = []
        self.rejected: str | None = None  # the reason, when the agent declined
        self.client_calls: list[ToolCallPart] = []
        self.sleeps: list[float | datetime | None] = []
        self.signals_waited: list[str] = []
        self.restarts = 0
        self.handed_over_to: str | None = None
        self.handover_session: Any = None  # what the handover's session started with
        self.forked: list[FakeThread] = []  # the forks it made, with what was sent to them
        self.session_entries: list[Message] = []  # what it added to its session (not on the thread)
        self.revisions: list[dict[str, Any]] = []  # the revisions it asked for, with their options
        self._reported: set[str] = set()  # threads whose end ctx.next delivered
        self.output: Any = None

    # ── driving it ──

    async def start(self, agent: Agent) -> Any:
        """Run the agent to its end, as the runtime would (restarts included). Returns its output."""
        if self._agent_version is None:
            self._agent_version = agent.version
        self._agent_type = agent.type
        self._inbox = (Filter(kinds=[EventKind.MESSAGE], to=self.run),)  # what is sent to it
        missing = [c.__name__ for c in requirements_of(agent) if c not in self._capabilities]
        if missing:
            raise AssertionError(f"the runtime refuses to start {type(agent).__name__}: missing {missing}")
        type(agent).check_output()  # as the runtime does when it loads the agent
        if isinstance(agent, Workflow):  # the Input, validated: a bad one never becomes a run
            validated = agent.parse_input(self._input.data)
            self._input = dataclasses.replace(self._input, parts=[DataPart(data=validated)])
        self._conversation.append(self._input)
        for _ in range(self._max_restarts + 1):
            try:
                result = agent.run(self)
                if inspect.isasyncgen(result):
                    async for event in result:  # what it yields: as if emitted, and a completed message sent
                        if isinstance(event, MessageCompleted):
                            await self._deliver_message(event.message)
                        elif isinstance(event, (MessageStarted, PartDelta)):
                            self.emitted.append(event)
                        else:
                            await self.emit(event)
                    output = None
                else:
                    output = await result
            except _Restart:
                self.restarts += 1
                continue
            except _Handover as h:
                self.handed_over_to, self.handover_session = h.agent, h.session
                return None
            except Rejected as r:
                self.rejected = r.reason
                return None
            break
        else:
            raise AssertionError(f"more than {self._max_restarts} restarts")
        output = type(agent).output_from_json(type(agent).output_to_json(output))  # as the run record stores it
        self.output = output
        return output

    def deliver(self, *events: ThreadEvent) -> None:
        """More thread events for the run's subscriptions."""
        self._events.extend(events)

    def cancel(self) -> None:
        self._cancelled = True

    # ── what the agent did, for assertions ──

    @property
    def replies(self) -> list[Message]:
        """The messages it wrote on its thread."""
        return [s.message for s in self.sent]

    @property
    def text(self) -> str:
        """The text of all replies, joined."""
        return "".join(m.text for m in self.replies)

    def events_of(self, kind: EventKind | str) -> list[Event]:
        return [e for e in self.emitted if str(e.kind) == str(kind)]

    # ── RunContext ──

    run = property(lambda self: _FakeThisRun(self))
    thread = property(lambda self: _FakeThisThread(self))
    client = property(lambda self: _FakeClient(self))
    input = property(lambda self: self._input)
    config = property(lambda self: self._config)
    state = property(lambda self: self._state)
    agents = property(lambda self: self._agents_fake)
    inbox = property(lambda self: self._inbox)

    @property
    def me(self) -> AgentIdentity:
        return AgentIdentity(name=self._agent, type=self._agent_type, display_name=self._display_name,
                             version=self._agent_version, revision=self._agent_revision)

    @property
    def sender(self) -> Sender:
        """How this run signs its messages."""
        return Sender.agent(self._agent, run=self.run)

    @property
    def session(self) -> "_FakeSession":
        return _FakeSession(self)

    async def emit(self, event: Event) -> None:
        if not isinstance(event, AGENT_EVENTS):
            raise TypeError(f"{type(event).__name__} is the runtime's to emit: an agent emits Progress, Usage, "
                            "StateSnapshot, StateDelta or Custom (a message goes on the thread with ctx.send)")
        self.emitted.append(event)

    async def _deliver_message(self, message: Message) -> Message:
        # as the runtime does: file bytes go to the file store, the message keeps a reference
        message = dataclasses.replace(message, parts=await map_file_parts(message.parts, self._files.save))
        self.sent.append(Sent(message))
        self._conversation.append(message)
        self.emitted.append(MessageCompleted(message=message))
        return message

    async def send(self, *content: Content, references: Sequence[RunRef] = ()) -> Message:
        return await self._deliver_message(Message.create(self.sender, *content, references=references))

    def stream(self) -> _Writer:
        return _Writer(self)

    async def ask(self, text: str, *, schema: Mapping[str, Any] | None = None) -> Message:
        return await self._request_input(InputRequest(reason=text, prompt=[TextPart(text=text)], input_schema=schema))

    async def _request_input(self, request: InputRequest) -> Message:
        self.input_requests.append(request)
        if not self._answers:
            raise NotScripted(f"the run asks for input ({request.reason}), and no answer is scripted")
        return _as_message(self._answers.popleft())

    async def step(self, name: str, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        self.steps[name] += 1
        result = fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    async def sleep(self, seconds: float | None = None, *, until: datetime | None = None) -> None:
        self.sleeps.append(until if until is not None else seconds)

    async def fork(self, *content: Content, session: Any = None, detach: bool = False) -> "FakeThread":
        """A fork, answering with the scripted `forks` reply; recorded in `forked`."""
        if self._fork_reply is None:
            raise NotScripted("the run forks, and no `forks` reply is scripted")
        thread = FakeThread(self, f"fork-{len(self.forked) + 1}", self._agent, self._fork_reply,
                            forked_from=(self.thread, len(self._conversation)), session=session, detach=detach)
        self.forked.append(thread)
        await thread.send(*content)
        return thread

    @property
    def forks(self) -> list["FakeThread"]:
        return list(self.forked)

    async def call(self, agent: str, *content: Content, session: SessionSeed = None,
                   detach: bool = False) -> "FakeThread":
        return await self._agents_fake._call(self, agent, content, session, detach)

    async def handover(self, agent: str, *, session: SessionSeed = None) -> NoReturn:
        self._agents_fake.handovers.append(agent)
        raise _Handover(agent, session)

    async def next(self, *threads: "FakeThread", timeout: float | None = None) -> ThreadEvent | None:
        """A scripted input first; then, of the given threads, the next one's end - once each."""
        if self._inputs:
            message = self._inputs.popleft()
            self._conversation.append(message)
            return ThreadEvent(position=next(self._positions), kind=EventKind.MESSAGE, message=message,
                               thread=self.thread)
        for t in threads:
            if t.id not in self._reported:
                self._reported.add(t.id)
                result = await t.wait()
                return ThreadEvent(position=next(self._positions), kind=f"run.{result.status}", run=result,
                                   agent=t.agent, thread=ThreadRef(t.id))
        if timeout is not None:
            return None
        raise NotScripted("the run waits for what comes next (ctx.next), and nothing is scripted")

    async def restart(self) -> NoReturn:
        raise _Restart()

    async def reject(self, reason: str) -> NoReturn:
        raise Rejected(reason)

    def subscribe(self, *filters: Filter) -> _Subscription:
        return _Subscription(self, filters)

    async def post(self, kind: str, data: Any = None) -> ThreadEvent:
        event = ThreadEvent(position=next(self._positions), kind=kind, run=self.run, agent=self._agent, data=data)
        self.posts.append(event)
        return event

    def capability(self, kind: type[T]) -> T | None:
        return self._capabilities.get(kind)

    def require(self, kind: type[T]) -> T:
        found = self.capability(kind)
        if found is None:
            raise CapabilityMissing(f"{kind.__name__} is not available here")
        return found


# ── Fake capabilities ───────────────────────────────────────────────────────

Reply = Union[str, Message, Sequence[Message], BaseModel, Callable[[Message], Any]]


class _Empty:
    """A subscription with nothing scripted: it ends at once."""

    async def next(self, timeout: float | None = None) -> ThreadEvent | None:
        return None

    def __aiter__(self) -> AsyncIterator[ThreadEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ThreadEvent]:
        return
        yield

    async def close(self) -> None:
        return None


class FakeThread(Thread):
    """A thread the run created - a fork, or another agent's - answering with a scripted reply: a
    string, a Message, a list of Messages, a pydantic model (its output), or a function of the first
    message sent to it (sync or async) returning one of those. Records what was sent to it."""

    def __init__(self, run: FakeRun, thread_id: str, agent: str, reply: Reply, *,
                 forked_from: tuple[ThreadRef, int] | None = None, session: Any = None, detach: bool = False):
        super().__init__(id=thread_id, agent=agent, forked_from=forked_from)
        for name, value in {"_run": run, "_reply": reply, "session": session, "detach": detach, "sent": [],
                            "answers": [], "signals": {}, "cancelled": False, "_result": None,
                            "_messages": [], "thread_state": _State(None), "events_on_thread": []}.items():
            object.__setattr__(self, name, value)

    async def send(self, *content: Content) -> None:
        """As the runtime does: a Message already in the run's conversation is forwarded as is; any
        other is signed by the run."""
        run = self._run
        message = Message.create(run.sender, *content) if len(content) != 1 else as_input(content[0], run.sender)
        if message.sender != run.sender and message.id not in {m.id for m in run._conversation}:
            message = dataclasses.replace(message, id=f"msg_{next(run._ids)}", sender=run.sender)
        self.sent.append(message)

    async def start(self, *content: Content) -> None:
        """A follow-up turn: recorded with what was sent; `wait()` answers it with the scripted reply."""
        await self.send(*content)
        object.__setattr__(self, "_result", None)

    async def _answer(self) -> tuple[list[Message], Any]:
        value = self._reply
        if callable(value) and not isinstance(value, (Message, BaseModel)):
            value = value(self.sent[0])
            if inspect.isawaitable(value):
                value = await value
        if isinstance(value, BaseModel):  # its output
            return [], value
        if isinstance(value, (str, Message, Part)) or not isinstance(value, Sequence):
            value = [value]
        return [_as_message(v, Sender.agent(self.agent)) for v in value], None

    async def wait(self) -> RunResult:
        if self._result is None:
            if self.cancelled:
                result = RunResult(id=f"run-{self.id}", agent=self.agent, status=RunStatus.CANCELLED)
            else:
                messages, output = await self._answer()
                object.__setattr__(self, "_messages", messages)
                result = RunResult(id=f"run-{self.id}", agent=self.agent, status=RunStatus.COMPLETED, output=output)
            object.__setattr__(self, "_result", result)
        return self._result

    async def messages(self) -> Sequence[Message]:
        await self.wait()
        return [*self.sent, *self._messages]

    def events(self, *kinds: str) -> _Empty:
        return _Empty()

    async def watch(self) -> AsyncIterator[Event]:
        return
        yield

    async def respond(self, answer: Message | Content) -> bool:
        self.answers.append(answer)
        return len(self.answers) == 1

    async def signal(self, key: str, value: Any = None) -> None:
        self.signals[key] = value

    async def close(self) -> None:
        object.__setattr__(self, "cancelled", True)

    async def thread_events(self, *kinds: str) -> Sequence[ThreadEvent]:
        """A fork's: what the test put in `events_on_thread`."""
        return [e for e in self.events_on_thread if not kinds or str(e.kind) in {str(k) for k in kinds}]

    @property
    def state(self) -> _State:
        """A fork's: what the test put in `thread_state`."""
        return self.thread_state


class FakeAgents:
    """The other agents, with a scripted reply per agent (see `FakeThread`)."""

    capability = Agents

    def __init__(self, replies: Mapping[str, Reply] | None = None, *, infos: Sequence[AgentInfo] = ()):
        self.replies = dict(replies or {})
        self.infos = list(infos) or [AgentInfo(name=n) for n in self.replies]
        self.threads: list[FakeThread] = []  # the threads the run started, with what it sent them
        self.handovers: list[str] = []

    def list(self) -> Sequence[AgentInfo]:
        return self.infos

    async def _call(self, run: FakeRun, agent: str, content: Sequence[Content], session: SessionSeed,
                    detach: bool) -> FakeThread:
        """`ctx.call`: a new thread of `agent`, answering with its scripted reply."""
        if agent not in self.replies:
            raise NotScripted(f"no reply scripted for agent {agent!r}")
        thread = FakeThread(run, f"thread-{agent}-{len(self.threads) + 1}", agent, self.replies[agent],
                            session=session, detach=detach)
        self.threads.append(thread)
        await thread.send(*content)
        return thread

    def as_tools(self) -> Sequence[ToolSpec]:
        return [ToolSpec(name=i.name, description=i.description,
                         input_schema=dict(i.input_schema or {"type": "object", "properties": {"input": {"type": "string"}}}))
                for i in self.infos]


class FakeFiles:
    """The Files capability on an in-memory file store (llamphouse.runtime.file_store)."""

    capability = Files

    def __init__(self, store: Any = None) -> None:
        from llamphouse.runtime.file_store import InMemoryFileStore  # the runtime's own in-memory store

        self.store = store if store is not None else InMemoryFileStore()
        self.run_id, self.thread_id = "run-1", "thread-1"

    def bind(self, run: FakeRun) -> None:
        self.run_id, self.thread_id = run.run.id, run.thread.id

    async def put(self, data: bytes, *, mime_type: str, name: str | None = None) -> FilePart:
        return await self.store.put(data, mime_type=mime_type, name=name, created_by=self.run_id)

    async def get(self, part: FilePart) -> bytes:
        from llamphouse.runtime.file_store import file_id

        if part.data is not None:
            return part.data
        return await self.store.read_all(file_id(part))

    async def save(self, part: FilePart) -> FilePart:
        """What the runtime does before writing a message: bytes handed in are stored (owned by the
        thread), references to the file store are claimed by the thread."""
        from llamphouse.runtime.file_store import file_id

        if part.data is not None:
            stored = await self.store.put(part.data, mime_type=part.mime_type, name=part.name,
                                          created_by=self.run_id, thread_id=self.thread_id)
            return dataclasses.replace(stored, metadata=part.metadata)
        if file_id(part):
            await self.store.claim([file_id(part)], self.thread_id)
        return part


@dataclass
class _Scheduled:
    job: str
    at: datetime | None
    delay: float | None
    key: str | None
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeScheduler:
    """The Scheduler capability: records what was scheduled."""

    capability = Scheduler

    def __init__(self) -> None:
        self.scheduled: list[_Scheduled] = []
        self.cancelled: list[str] = []

    async def schedule(self, job: str, *, at: datetime | None = None, delay: float | None = None,
                       key: str | None = None, **kwargs: Any) -> str:
        self.scheduled.append(_Scheduled(job, at, delay, key, kwargs))
        return key or f"job-{len(self.scheduled)}"

    async def cancel(self, key: str) -> None:
        self.cancelled.append(key)
