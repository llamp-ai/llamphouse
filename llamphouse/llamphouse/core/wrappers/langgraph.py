from __future__ import annotations

from typing import Any, Dict, Optional

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

    def __init__(self, *, graph: Any, stream: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.stream = stream

    async def invoke_framework(self, context: Context, state: Dict[str, Any]) -> Any:
        # Optional streaming path for better UX over A2A/SSE clients.
        if self.stream and hasattr(self.graph, "astream"):
            chunks = []
            async for evt in self.graph.astream(state):
                text = self._extract_text_delta(evt)
                if text:
                    chunks.append(text)
                    context.send_chunk(text)
            if chunks:
                return {"output": "".join(chunks)}

        if not hasattr(self.graph, "ainvoke"):
            raise RuntimeError("LangGraph wrapper requires an object with async `ainvoke(state)`")

        return await self.graph.ainvoke(state)

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
        return ""
