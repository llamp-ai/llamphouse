from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .base import BaseAgentWrapper
from ..context import Context
from ..data_stores.base_data_store import BaseDataStore

logger = logging.getLogger("llamphouse.wrappers.langgraph")

# LangGraph's channel for the answers a task's interrupts got so far.
_RESUME = "__resume__"
# The graph already ran to its end in an earlier attempt of the run.
_FINISHED = object()


class LangGraphAgent(BaseAgentWrapper):
    """LLAMPHouse Agent wrapper for LangGraph flows.

    This wrapper keeps LLAMPHouse as the external contract (A2A, Assistant API,
    Compass) while delegating internal execution to a LangGraph graph object.

    Expected graph capabilities:
    - async `ainvoke(state)` for final response
    - optional async `astream(state)` for incremental chunks

    Durable runs: a LangGraph graph runs with a checkpointer - LangGraph's
    own persistence - keyed by the run, so a run that runs again (resumed,
    retried, taken over after its worker died) carries on from the graph's
    last checkpoint instead of from ``START``. A node that calls LangGraph's
    ``interrupt(value)`` parks the run until a human answers - exactly like
    ``context.wait_for_input``: an ``input-required`` task over A2A, a
    signal over ``POST .../signals`` - and the graph then resumes inside
    that node with the answer (``Command(resume=...)``). An interrupt value
    ``{"prompt": ..., "schema": ...}`` is shown as such; any other value is
    shown as the prompt, and passed on as ``request["interrupt"]``.

    :param checkpointer: ``"llamphouse"`` (default): checkpoints in the
        LLAMPHouse data store (``LLAMPHouseCheckpointSaver``); a LangGraph
        checkpointer of your own (e.g. ``PostgresSaver``); or ``None``: no
        checkpoints, the graph runs as a whole each time (and ``interrupt``
        cannot be used). A graph compiled with a checkpointer keeps it.
    :param checkpoint_scope: ``"run"`` (default): one LangGraph thread per
        run. ``"thread"``: one per LLAMPHouse thread, so the graph's state
        carries over from run to run (LangGraph memory); each new run then
        passes the mapped state as new input.
    :param interrupt_signal: the signal name an interrupt waits for
        (``"human_input"``, as ``context.wait_for_input``).
    """

    def __init__(
        self,
        *,
        graph: Any,
        stream: bool = True,
        map_nodes_to_steps: bool = True,
        persist_step_state: bool = True,
        checkpointer: Any = "llamphouse",
        checkpoint_scope: str = "run",
        interrupt_signal: str = "human_input",
        durable: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if checkpoint_scope not in ("run", "thread"):
            raise ValueError("checkpoint_scope must be 'run' or 'thread'")
        self.graph = graph
        self.stream = stream
        self.map_nodes_to_steps = map_nodes_to_steps
        self.persist_step_state = persist_step_state
        self.checkpointer = checkpointer
        self.checkpoint_scope = checkpoint_scope
        self.interrupt_signal = interrupt_signal
        # A run whose worker died resumes from its last checkpoint.
        self.durable = (checkpointer is not None) if durable is None else durable
        self._graphs: dict[int, Any] = {}  # data store -> graph with its checkpointer

    # ── Durable runs ─────────────────────────────────────────────────────

    def _durable_graph(self, context: Context) -> Any:
        """The graph to run with checkpoints, or None to run it as before."""
        graph = self.graph
        if not (hasattr(graph, "aget_state") and hasattr(graph, "copy")):
            return None  # not a compiled LangGraph graph
        if not getattr(context, "durable", False):
            return None
        if getattr(graph, "checkpointer", None) not in (None, False):
            return graph  # compiled with a checkpointer of its own
        if self.checkpointer is None:
            return None
        data_store = context.data_store
        cached = self._graphs.get(id(data_store))
        if cached is not None:
            return cached
        saver = self.checkpointer
        if saver == "llamphouse":
            if type(data_store).kv_put is BaseDataStore.kv_put:
                logger.warning(
                    "%s keeps no key-value records: LangGraph agent %s runs without checkpoints.",
                    type(data_store).__name__, self.id,
                )
                return None
            from .langgraph_checkpoint import LLAMPHouseCheckpointSaver

            saver = LLAMPHouseCheckpointSaver(data_store)
        cached = graph.copy(update={"checkpointer": saver})
        self._graphs[id(data_store)] = cached
        return cached

    def _graph_thread_id(self, context: Context) -> str:
        return context.thread_id if self.checkpoint_scope == "thread" else context.run_id

    async def run(self, context: Context):
        graph = self._durable_graph(context)
        if graph is None:
            return await super().run(context)

        config = {"configurable": {"thread_id": self._graph_thread_id(context)}}
        saver = graph.checkpointer
        forget = getattr(saver, "forget", None)
        if forget is not None:
            forget(config["configurable"]["thread_id"])  # read what the store holds now
            saver.bind(config["configurable"]["thread_id"], context.thread_id)
        try:
            output = await self._run_durable(context, graph, config)
        finally:
            if forget is not None:
                forget(config["configurable"]["thread_id"])
        reply = self.output_mapper(output)
        if reply:
            await context.reply(reply)

    async def _run_durable(self, context: Context, graph: Any, config: dict) -> Any:
        start_key = self._call_key_started(context)
        started = context._replayed(start_key)
        if started is None:
            step = await context.start_step("langgraph.start", call_key=start_key)
            await context.complete_step(step.id)
            graph_input: Any = self.state_mapper(context)
        else:
            graph_input = await self._resume_input(context, graph, config)
        # The graph carries on from its checkpoint: what it does from here is
        # new, nothing of it a replay (see Context.replaying).
        context._ahead.clear()

        output = None
        while True:
            if graph_input is not _FINISHED:
                output = await self.invoke_framework(context, graph_input, config=config)
            snapshot = await graph.aget_state(config)
            if not self._interrupts(snapshot):
                break
            graph_input = await self._answers(context, graph, config, snapshot)
        if graph_input is _FINISHED or not self._has_text(output):
            output = self._values(snapshot.values)
        return output

    def _call_key_started(self, context: Context) -> str:
        return context._call_key("langgraph.start")

    async def _resume_input(self, context: Context, graph: Any, config: dict) -> Any:
        """Where an earlier attempt of this run left the graph."""
        snapshot = await graph.aget_state(config)
        if snapshot.created_at is None:
            return self.state_mapper(context)  # stopped before its first checkpoint
        if self._interrupts(snapshot):
            return await self._answers(context, graph, config, snapshot)
        if snapshot.next:
            return None  # carry on from the last checkpoint
        return _FINISHED

    @staticmethod
    def _interrupts(snapshot: Any) -> list:
        return [(task, intr) for task in (snapshot.tasks or ()) for intr in (getattr(task, "interrupts", ()) or ())]

    async def _answers(self, context: Context, graph: Any, config: dict, snapshot: Any) -> Any:
        """Wait for the answer to each interrupt the graph stopped at (the
        run parks at the first one unanswered); the input resuming it."""
        from langgraph.types import Command

        resumed: dict[str, list] = {}
        saved = await graph.checkpointer.aget_tuple(config)
        for task_id, channel, value in (saved.pending_writes if saved else ()):
            if channel == _RESUME and isinstance(value, list):
                resumed[task_id] = value
        answers = {}
        for task, intr in self._interrupts(snapshot):
            # The n-th interrupt of a task: the answers it got so far, plus one.
            n = len(resumed.get(task.id, [])) + 1
            answers[intr.id] = await context.wait_for_signal(
                self.interrupt_signal,
                request=self._interrupt_request(intr.value),
                key=f"langgraph:{intr.id}#{n}",
            )
        return Command(resume=answers)

    def _interrupt_request(self, value: Any) -> Dict[str, Any]:
        safe = self._to_json_safe(value)
        if isinstance(value, dict):
            prompt = value.get("prompt") or value.get("question") or json.dumps(safe, default=str)
            return {"prompt": prompt, "schema": value.get("schema"), "interrupt": safe}
        return {"prompt": value if isinstance(value, str) else json.dumps(safe, default=str), "interrupt": safe}

    def _values(self, values: Any) -> Any:
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if k != "__interrupt__"}
        return values

    @staticmethod
    def _has_text(output: Any) -> bool:
        return isinstance(output, dict) and isinstance(output.get("output"), str) and set(output) == {"output"}

    # ── Running the graph ────────────────────────────────────────────────

    async def invoke_framework(self, context: Context, state: Dict[str, Any], config: Optional[dict] = None) -> Any:
        step_ids: dict[str, str] = {}
        extra = (config,) if config is not None else ()

        if self.map_nodes_to_steps and hasattr(self.graph, "astream_events"):
            stream_output = await self._stream_via_events(context, state, step_ids, config)
            if stream_output is not None or (config is not None and self.stream):
                # With checkpoints the graph runs once: no text streamed, the
                # caller reads the result from the checkpoint.
                return stream_output

        graph = self._graphs_for(context, config)
        # Optional streaming path for better UX over A2A/SSE clients.
        if self.stream and hasattr(graph, "astream"):
            chunks = []
            async for evt in graph.astream(state, *extra):
                if self.map_nodes_to_steps:
                    await self._handle_stream_update_steps(context, evt, step_ids)
                text = self._extract_text_delta(evt)
                if text:
                    chunks.append(text)
                    context.send_chunk(text)
            if chunks or config is not None:
                return {"output": "".join(chunks)} if chunks else None

        if not hasattr(graph, "ainvoke"):
            raise RuntimeError("LangGraph wrapper requires an object with async `ainvoke(state)`")

        return await graph.ainvoke(state, *extra)

    def _graphs_for(self, context: Context, config: Optional[dict]) -> Any:
        """The graph a call runs: with its checkpointer when it has a config."""
        if config is None:
            return self.graph
        return self._durable_graph(context) or self.graph

    async def _stream_via_events(
        self,
        context: Context,
        state: Dict[str, Any],
        step_ids: dict[str, str],
        config: Optional[dict] = None,
    ) -> Dict[str, Any] | None:
        if not self.stream:
            return None

        graph = self._graphs_for(context, config)
        extra = (config,) if config is not None else ()
        chunks: list[str] = []
        # LangGraph/LangChain event stream contains node-level start/end hooks.
        async for event in graph.astream_events(state, *extra, version="v2"):
            await self._handle_astream_event_steps(context, event, step_ids)
            text = self._extract_text_delta(event)
            if text:
                chunks.append(text)
                context.send_chunk(text)

        # Ensure started steps are closed even if event stream is sparse.
        for key, step_id in list(step_ids.items()):
            await context.complete_step(step_id, status="completed")
            step_ids.pop(key, None)

        if chunks:
            return {"output": "".join(chunks)}
        return None

    async def _handle_stream_update_steps(
        self,
        context: Context,
        event: Any,
        step_ids: dict[str, str],
    ) -> None:
        if not isinstance(event, dict):
            return

        for node_name, node_output in event.items():
            if not isinstance(node_name, str):
                continue
            if node_name not in step_ids:
                step = await context.start_step(
                    name=node_name,
                    input=self._build_step_input({"phase": "start", "event": event}),
                    metadata=self._build_step_metadata(
                        node_name=node_name,
                        event_name="stream_update",
                        event_source="astream",
                        state_snapshot=event,
                    ),
                )
                step_ids[node_name] = step.id
            step_id = step_ids.get(node_name)
            if step_id:
                await context.complete_step(
                    step_id,
                    output=self._build_step_output(
                        node_output=node_output,
                        state_snapshot={node_name: node_output},
                    ),
                    status="completed",
                )
                step_ids.pop(node_name, None)

    async def _handle_astream_event_steps(
        self,
        context: Context,
        event: Any,
        step_ids: dict[str, str],
    ) -> None:
        if not isinstance(event, dict):
            return

        event_name = event.get("event")
        node_name = event.get("name")
        if not isinstance(node_name, str):
            return

        if event_name == "on_chain_start":
            if node_name not in step_ids:
                payload = event.get("data", {}).get("input")
                step = await context.start_step(
                    name=node_name,
                    input=self._build_step_input(payload),
                    metadata=self._build_step_metadata(
                        node_name=node_name,
                        event_name=event_name,
                        event_source="astream_events",
                        state_snapshot=payload,
                    ),
                )
                step_ids[node_name] = step.id
            return

        if event_name == "on_chain_end":
            step_id = step_ids.get(node_name)
            if step_id:
                payload = event.get("data", {}).get("output")
                await context.complete_step(
                    step_id,
                    output=self._build_step_output(
                        node_output=payload,
                        state_snapshot=payload,
                    ),
                    status="completed",
                )
                step_ids.pop(node_name, None)
            return

        if event_name == "on_chain_error":
            step_id = step_ids.get(node_name)
            if step_id:
                err = event.get("data", {}).get("error") or "LangGraph node failed"
                await context.complete_step(step_id, error=str(err), status="failed")
                step_ids.pop(node_name, None)

    def _build_step_metadata(
        self,
        *,
        node_name: str,
        event_name: str,
        event_source: str,
        state_snapshot: Any,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "framework": "langgraph",
            "step_type": "langgraph_node",
            "node_name": node_name,
            "event_name": event_name,
            "event_source": event_source,
        }
        if self.persist_step_state:
            metadata["state"] = self._to_json_safe(state_snapshot)
        return metadata

    def _build_step_input(self, payload: Any) -> Any:
        if not self.persist_step_state:
            return payload
        return {
            "state": self._to_json_safe(payload),
        }

    def _build_step_output(self, *, node_output: Any, state_snapshot: Any) -> Any:
        if not self.persist_step_state:
            return node_output
        return {
            "node_output": self._to_json_safe(node_output),
            "state": self._to_json_safe(state_snapshot),
        }

    def _to_json_safe(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass
        try:
            json.dumps(value)
            return value
        except Exception:
            try:
                return json.loads(json.dumps(value, default=str))
            except Exception:
                return str(value)

    def _extract_text_delta(self, event: Any) -> str:
        """Best-effort extraction of text from streamed LangGraph events."""
        if event is None:
            return ""
        if isinstance(event, str):
            return event
        if isinstance(event, dict):
            if isinstance(event.get("text"), str):
                return event["text"]
            if isinstance(event.get("delta"), str):
                return event["delta"]
            # Common nested shape: {"messages": [...]} with last assistant delta
            msgs = event.get("messages")
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, str):
                    return last
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, str):
                        return content
            if isinstance(event.get("output"), str):
                return event["output"]
            data = event.get("data")
            if isinstance(data, dict):
                chunk = data.get("chunk")
                if isinstance(chunk, dict):
                    if isinstance(chunk.get("output"), str):
                        return chunk["output"]
                    if isinstance(chunk.get("text"), str):
                        return chunk["text"]
                    if isinstance(chunk.get("delta"), str):
                        return chunk["delta"]
            # Common astream update shape: {"node_name": {"output": "..."}}
            for key, val in event.items():
                if key in {"event", "name", "run_id", "tags", "metadata", "data"}:
                    continue
                if isinstance(val, dict):
                    if isinstance(val.get("output"), str):
                        return val["output"]
                    if isinstance(val.get("text"), str):
                        return val["text"]
                    if isinstance(val.get("delta"), str):
                        return val["delta"]
        return ""
