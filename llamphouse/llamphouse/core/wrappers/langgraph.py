from __future__ import annotations

import json
from typing import Any, Dict

from .base import BaseAgentWrapper
from ..context import Context


class LangGraphAgent(BaseAgentWrapper):
    """LLAMPHouse Agent wrapper for LangGraph flows.

    This wrapper keeps LLAMPHouse as the external contract (A2A, Assistant API,
    Compass) while delegating internal execution to a LangGraph graph object.

    Expected graph capabilities:
    - async `ainvoke(state)` for final response
    - optional async `astream(state)` for incremental chunks
    """

    def __init__(
        self,
        *,
        graph: Any,
        stream: bool = True,
        map_nodes_to_steps: bool = True,
        persist_step_state: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.stream = stream
        self.map_nodes_to_steps = map_nodes_to_steps
        self.persist_step_state = persist_step_state

    async def invoke_framework(self, context: Context, state: Dict[str, Any]) -> Any:
        step_ids: dict[str, str] = {}

        if self.map_nodes_to_steps and hasattr(self.graph, "astream_events"):
            stream_output = await self._stream_via_events(context, state, step_ids)
            if stream_output is not None:
                return stream_output

        # Optional streaming path for better UX over A2A/SSE clients.
        if self.stream and hasattr(self.graph, "astream"):
            chunks = []
            async for evt in self.graph.astream(state):
                if self.map_nodes_to_steps:
                    await self._handle_stream_update_steps(context, evt, step_ids)
                text = self._extract_text_delta(evt)
                if text:
                    chunks.append(text)
                    context.send_chunk(text)
            if chunks:
                return {"output": "".join(chunks)}

        if not hasattr(self.graph, "ainvoke"):
            raise RuntimeError("LangGraph wrapper requires an object with async `ainvoke(state)`")

        return await self.graph.ainvoke(state)

    async def _stream_via_events(
        self,
        context: Context,
        state: Dict[str, Any],
        step_ids: dict[str, str],
    ) -> Dict[str, Any] | None:
        if not self.stream:
            return None

        chunks: list[str] = []
        # LangGraph/LangChain event stream contains node-level start/end hooks.
        async for event in self.graph.astream_events(state, version="v2"):
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
