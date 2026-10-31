"""LangGraph on llamphouse: a graph's checkpoints in the run's state, and a graph as an agent.

    LlamphouseCheckpointer(ctx)     a LangGraph checkpointer keeping its checkpoints in `ctx.run.state`
                                    (scope="thread": the graph's state carries over to the thread's next run)
    await run_graph(ctx, graph, x)  runs a graph durably inside any agent: resumed from its checkpoint when
                                    the run executes again, `interrupt()` asked with `ctx.ask`, streamed
    LangGraphAgent                  a graph as the whole agent: `graph = builder.compile()` on a subclass,
                                    or `Planner = LangGraphAgent.of(graph)` for an agent file's entrypoint

Needs `pip install llamphouse[langgraph]`; a node's `interrupt()` needs Python 3.11+ (LangGraph, in async runs). Compile graphs without a checkpointer: `run_graph` gives each run
its own (a graph compiled with one of yours keeps it).

Durability: a run that waits (an interrupt asked) or is taken over after a crash executes again from the
top. The graph then carries on from its last checkpoint - nodes that finished don't run again - instead of
starting over. The checkpointer reads and writes the run's state outside the replay (`ctx.run.state.live`),
so it always sees the latest checkpoint. It keeps the latest checkpoint, not the history (a graph with
delta channels keeps their chain until they snapshot).
"""

from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, ClassVar, Literal, Mapping, Sequence

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError as e:  # pragma: no cover - the extra
    raise ImportError("llamphouse.helpers.langgraph needs LangGraph: pip install llamphouse[langgraph]") from e

from llamphouse.sdk import Agent

Scope = Literal["run", "thread"]
_RESUME = "__resume__"  # LangGraph's channel for the answers a task's interrupts got so far


def _enc(typed: tuple[str, bytes]) -> list[str]:
    return [typed[0], base64.b64encode(typed[1]).decode("ascii")]


def _dec(value: Sequence[str]) -> tuple[str, bytes]:
    return value[0], base64.b64decode(value[1])


class LlamphouseCheckpointer(InMemorySaver):
    """LangGraph's in-memory checkpointer, kept in the run's state under `key` after every write.

    scope="run" (default): one graph thread per run - durable while the run lasts.
    scope="thread": one per llamphouse thread - the graph's state carries over from run to run. During a
    run it is kept in the run's state; `await keep()` writes it to the thread's state (`run_graph` does,
    when the graph ends or asks), one `state.changed` event per run rather than one per graph step."""

    def __init__(self, ctx: Any, *, scope: Scope = "run", key: str = "langgraph", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if scope not in ("run", "thread"):
            raise ValueError("scope must be 'run' or 'thread'")
        self.ctx, self.scope, self.key = ctx, scope, key
        self._loaded: set[str] = set()

    @property
    def thread_id(self) -> str:
        """The graph thread this checkpointer keeps: the run's id, or the thread's."""
        return self.ctx.thread.id if self.scope == "thread" else self.ctx.run.id

    def config(self, **configurable: Any) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id, **configurable}}

    # ── to and from the state ────────────────────────────────────────────

    def _dump(self, thread_id: str) -> dict[str, Any]:
        keep = self._needed(thread_id)
        return {
            "checkpoints": [[ns, cid, _enc(c), _enc(m), parent]
                            for ns, saved in self.storage.get(thread_id, {}).items()
                            for cid, (c, m, parent) in saved.items() if (ns, cid) in keep],
            "blobs": [[ns, channel, version, _enc(value)] for (t, ns, channel, version), value in self.blobs.items()
                      if t == thread_id and (ns, channel, version) in keep],
            "writes": [[ns, cid, tid, idx, task, channel, _enc(value), path]
                       for (t, ns, cid), writes in self.writes.items() if t == thread_id and (ns, cid) in keep
                       for (tid, idx), (task, channel, value, path) in writes.items()],
        }

    def _needed(self, thread_id: str) -> set[tuple]:
        """The latest checkpoint of each namespace, the blobs it reads and its pending writes. A graph
        with delta channels (LangGraph's DeltaChannel, rebuilt from its ancestors' writes) keeps the
        whole chain until they snapshot - LangGraph records that in the checkpoint's metadata."""
        keep: set[tuple] = set()
        for ns, saved in self.storage.get(thread_id, {}).items():
            if not saved:
                continue
            latest = max(saved)
            if self.serde.loads_typed(saved[latest][1]).get("counters_since_delta_snapshot"):
                chain = list(saved)  # rebuilt from its ancestors: keep them
            else:
                chain = [latest]
            for cid in chain:
                keep.add((ns, cid))
                for channel, version in self.serde.loads_typed(saved[cid][0]).get("channel_versions", {}).items():
                    keep.add((ns, channel, version))
        return keep

    def _load(self, thread_id: str, data: Mapping[str, Any] | None) -> None:
        for ns, cid, c, m, parent in (data or {}).get("checkpoints", []):
            self.storage[thread_id][ns][cid] = (_dec(c), _dec(m), parent)
        for ns, channel, version, value in (data or {}).get("blobs", []):
            self.blobs[(thread_id, ns, channel, version)] = _dec(value)
        for ns, cid, tid, idx, task, channel, value, path in (data or {}).get("writes", []):
            self.writes.setdefault((thread_id, ns, cid), {})[(tid, idx)] = (task, channel, _dec(value), path)

    async def _ensure(self, thread_id: str) -> None:
        if thread_id in self._loaded:
            return
        data = await self.ctx.run.state.live.get(self.key)
        if data is None and self.scope == "thread":
            data = await self.ctx.state.live.get(self.key)  # where the thread's last run left it
        self._load(thread_id, data)
        self._loaded.add(thread_id)

    async def _save(self, thread_id: str) -> None:
        await self.ctx.run.state.live.set(self.key, self._dump(thread_id))

    async def keep(self) -> None:
        """scope="thread": the graph's state onto the thread, for its next run."""
        if self.scope == "thread" and self.thread_id in self._loaded:
            await self.ctx.state.live.set(self.key, self._dump(self.thread_id))

    # ── LangGraph's async interface ──────────────────────────────────────

    async def aput(self, config, checkpoint, metadata, new_versions):  # type: ignore[no-untyped-def]
        thread_id = config["configurable"]["thread_id"]
        await self._ensure(thread_id)
        saved = self.put(config, checkpoint, metadata, new_versions)
        await self._save(thread_id)
        return saved

    async def aput_writes(self, config, writes, task_id, task_path: str = "") -> None:  # type: ignore[no-untyped-def]
        thread_id = config["configurable"]["thread_id"]
        await self._ensure(thread_id)
        self.put_writes(config, writes, task_id, task_path)
        await self._save(thread_id)

    async def aget_tuple(self, config):  # type: ignore[no-untyped-def]
        await self._ensure(config["configurable"]["thread_id"])
        return self.get_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None) -> AsyncIterator:  # type: ignore[no-untyped-def]
        if config is not None:
            await self._ensure(config["configurable"]["thread_id"])
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aget_delta_channel_history(self, *, config, channels):  # type: ignore[no-untyped-def]
        await self._ensure(config["configurable"]["thread_id"])
        return self.get_delta_channel_history(config=config, channels=channels)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)
        await self.ctx.run.state.live.delete(self.key)
        if self.scope == "thread":
            await self.ctx.state.live.delete(self.key)


# ── running a graph inside an agent ─────────────────────────────────────────


def _interrupts(snapshot: Any) -> list[tuple[Any, Any]]:
    return [(task, intr) for task in (snapshot.tasks or ()) for intr in (getattr(task, "interrupts", ()) or ())]


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _question(value: Any) -> tuple[str, Mapping[str, Any] | None]:
    """What an `interrupt(value)` asks: `{"prompt" | "question": ..., "schema": ...}`, a string, or
    anything else as JSON."""
    if isinstance(value, Mapping):
        text = value.get("prompt") or value.get("question")
        return (str(text) if text else json.dumps(_jsonable(value))), value.get("schema")
    return (value if isinstance(value, str) else json.dumps(_jsonable(value))), None


def _answer(message: Any) -> Any:
    """What resumes the interrupt: the answer's data if it has any, else its text."""
    data = getattr(message, "data", None)
    return data if data is not None else message.text


def _values(values: Any) -> Any:
    return {k: v for k, v in values.items() if k != "__interrupt__"} if isinstance(values, dict) else values


async def _answers(ctx: Any, graph: Any, config: dict[str, Any], snapshot: Any, key: str) -> Any:
    """Ask for each interrupt the graph stopped at - the run waits at the first one unanswered - and
    the command resuming it. Keyed by the interrupt (and how many answers its task got), not by order."""
    from langgraph.types import Command

    resumed: dict[str, int] = {}
    saved = await graph.checkpointer.aget_tuple(config)
    for task_id, channel, value in (saved.pending_writes if saved else ()):
        if channel == _RESUME and isinstance(value, list):
            resumed[task_id] = len(value)
    answers = {}
    for task, intr in _interrupts(snapshot):
        text, schema = _question(intr.value)
        n = resumed.get(task.id, 0) + 1
        answers[intr.id] = _answer(await ctx.ask(text, schema=schema, key=f"{key}:{intr.id}#{n}"))
    return Command(resume=answers)


def _text(chunk: Any) -> str:
    """The text of a model's message (chunk) from `stream_mode="messages"`, if it is the model's."""
    if getattr(chunk, "type", None) not in ("ai", "AIMessageChunk"):
        return ""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, Mapping) and p.get("type") == "text")


async def _invoke(ctx: Any, graph: Any, value: Any, config: dict[str, Any], *, stream: bool, key: str) -> None:
    if not stream:
        await graph.ainvoke(value, config)
        return
    writer = None
    try:
        async for chunk, _ in graph.astream(value, config, stream_mode="messages"):
            text = _text(chunk)
            if not text:
                continue
            if writer is None:  # one message per stretch of the graph, keyed by where it started
                writer = ctx.stream(key=key)
                await writer.__aenter__()
            await writer.write(text)
    except BaseException as e:
        if writer is not None:
            await writer.__aexit__(type(e), e, e.__traceback__)
        raise
    if writer is not None:
        await writer.__aexit__(None, None, None)


async def run_graph(ctx: Any, graph: Any, input: Any, *, scope: Scope = "run", stream: bool = True,
                    key: str = "langgraph", configurable: Mapping[str, Any] | None = None) -> Any:
    """Run a compiled LangGraph graph as part of this run, durably; returns its final values.

    - The first time: the graph starts with `input`. When the run executes again (it waited, or another
      worker took it over), the graph carries on from its last checkpoint - with the same input it never
      sees twice.
    - A node's `interrupt(value)` is asked with `ctx.ask` (a value `{"prompt": ..., "schema": ...}`, or
      a string); the run waits without holding a worker, and the graph resumes in that node with the
      answer (its data, else its text).
    - stream=True: what the graph's models say is streamed into this run's messages as it comes.
    - scope="thread": the graph's state carries over to the thread's next run (LangGraph memory).
    """
    saver = getattr(graph, "checkpointer", None)
    if saver in (None, False):
        saver = LlamphouseCheckpointer(ctx, scope=scope, key=key)
        graph = graph.copy(update={"checkpointer": saver})
    thread_id = ctx.thread.id if scope == "thread" else ctx.run.id
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id, **(configurable or {})}}

    snapshot = await graph.aget_state(config)
    at = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id") or ""
    started = await ctx.run.state.live.get(f"{key}.start")
    if started is None:  # this run's first execution: where the graph was when it started
        await ctx.run.state.live.set(f"{key}.start", at)
        started = at
    if at == started:
        value: Any = input  # nothing of this run's work checkpointed yet
    elif _interrupts(snapshot):
        value = await _answers(ctx, graph, config, snapshot, key)
    elif snapshot.next:
        value = None  # carry on from the last checkpoint
    else:
        value = ...  # it ran to its end in an earlier execution
    while value is not ...:
        await _invoke(ctx, graph, value, config, stream=stream, key=f"{key}:{at or 'start'}")
        snapshot = await graph.aget_state(config)
        at = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id") or at
        if not _interrupts(snapshot):
            break
        if isinstance(saver, LlamphouseCheckpointer):
            await saver.keep()  # the thread sees where it stopped
        value = await _answers(ctx, graph, config, snapshot, key)
    if isinstance(saver, LlamphouseCheckpointer):
        await saver.keep()
    return _values(snapshot.values)


# ── a graph as an agent ─────────────────────────────────────────────────────


class LangGraphAgent(Agent):
    """A LangGraph graph as the whole agent.

        class Planner(LangGraphAgent):
            \"\"\"Plans trips.\"\"\"
            graph = builder.compile()
            scope = "thread"

    or `Planner = LangGraphAgent.of(builder.compile(), scope="thread")`. Override `graph_input` to map
    the run to the graph's input, and `reply` for what it says at the end."""

    graph: ClassVar[Any] = None
    scope: ClassVar[Scope] = "run"
    stream: ClassVar[bool] = True

    @classmethod
    def of(cls, graph: Any, *, scope: Scope = "run", stream: bool = True, name: str = "GraphAgent",
           description: str | None = None) -> type["LangGraphAgent"]:
        attrs: dict[str, Any] = {"graph": graph, "scope": scope, "stream": stream, "__doc__": description}
        return type(name, (cls,), attrs)

    async def graph_input(self, ctx: Any) -> Any:
        """The graph's input: `{"messages": [...]}` - the run's session (scope="run"), or only what
        started this run (scope="thread": the graph keeps the conversation itself)."""
        if self.scope == "thread":
            return {"messages": [{"role": "user", "content": ctx.input.text}]}
        messages = await ctx.session.messages()
        return {"messages": [{"role": "assistant" if ctx.session.is_own(m) else "user", "content": m.text}
                             for m in messages]}

    def reply(self, values: Any) -> str | None:
        """What it says at the end when nothing was streamed: the graph's `output` if a string, else
        the content of its last message."""
        if isinstance(values, Mapping):
            if isinstance(values.get("output"), str):
                return values["output"]
            messages = values.get("messages") or []
            if messages:
                last = messages[-1]
                content = getattr(last, "content", None) if not isinstance(last, Mapping) else last.get("content")
                if isinstance(content, str) and content:
                    return content
        return None

    async def run(self, ctx: Any) -> None:
        if self.graph is None:
            raise TypeError(f"{type(self).__name__}: set `graph` (a compiled LangGraph graph)")
        values = await run_graph(ctx, self.graph, await self.graph_input(ctx), scope=self.scope, stream=self.stream)
        if not self.stream and (text := self.reply(values)):
            await ctx.send(text)


__all__ = ["LlamphouseCheckpointer", "run_graph", "LangGraphAgent"]
