from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from .base_event_handler import BaseEventHandler
from .stream_events import (
    CanonicalStreamEvent,
    StreamError,
    StreamFinished,
    StreamStarted,
    TextDelta,
    TextSnapshot,
    ToolCallDelta,
)

@dataclass
class _ToolState:
    run_step_id: str
    tool_call_id: str
    name: str = ""
    arguments: str = ""
    completed: bool = False


class StreamingEmitter(BaseEventHandler):
    def __init__(self, send_event, assistant_id: str, thread_id: str, run_id: str):
        super().__init__(send_event)
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.run_id = run_id

        self.message_id: Optional[str] = None
        self.message_started: bool = False
        self.message_text: str = ""

        self.tools_by_id: Dict[str, _ToolState] = {}
        self.tool_id_by_index: Dict[int, str] = {}
        self.last_tool_call_id: Optional[str] = None

        self.done_sent: bool = False

    @property
    def content(self) -> str:
        return self.message_text

    def handle(self, event: CanonicalStreamEvent) -> None:
        if isinstance(event, StreamStarted):
            self._on_started()
            return
        if isinstance(event, TextDelta):
            self._on_text_delta(event)
            return
        if isinstance(event, TextSnapshot):
            self._on_text_snapshot(event)
            return
        if isinstance(event, ToolCallDelta):
            self._on_tool_call_delta(event)
            return
        if isinstance(event, StreamError):
            self._on_error(event)
            return
        if isinstance(event, StreamFinished):
            self._on_finished(event)
            return
        raise TypeError(f"Unsupported canonical event: {type(event)!r}")

    def _on_started(self) -> None:
        return
    
    def _ensure_message_started(self, message_id: Optional[str]) -> str:
        if self.message_started:
            return self.message_id or message_id or str(uuid.uuid4())

        self.message_started = True
        self.message_id = message_id or str(uuid.uuid4())
        self.send_event_message_created(self.assistant_id, self.thread_id, self.run_id, self.message_id)
        self.send_event_message_in_progress(self.assistant_id, self.thread_id, self.run_id, self.message_id)
        return self.message_id
    
    def _ensure_tool_started(self, tool_call_id: str, name: Optional[str]) -> _ToolState:
        tool = self.tools_by_id.get(tool_call_id)
        if tool:
            if name and not tool.name:
                tool.name = name
            return tool
        
        tool = _ToolState(run_step_id=str(uuid.uuid4()), tool_call_id=tool_call_id, name=name or "")
        self.tools_by_id[tool_call_id] = tool
        self.send_event_run_step_tool_created(
            self.assistant_id,
            self.thread_id,
            self.run_id,
            tool.run_step_id,
            tool.tool_call_id,
            tool.name,
        )

        return tool
    
    def _on_text_delta(self, event: TextDelta) -> None:
        if not event.text:
            return
        
        msg_id = self._ensure_message_started(event.message_id)

        content_block_index = event.index if isinstance(event.index, int) and event.index >= 0 else 0
        self.index = content_block_index
        self.message_text += event.text
        self.send_event_message_delta(msg_id, event.text)

    def _on_text_snapshot(self, event: TextSnapshot) -> None:
        msg_id = self._ensure_message_started(event.message_id)

        full_text = event.full_text or ""
        if not full_text:
            return
        
        if full_text.startswith(self.message_text):
            delta = full_text[len(self.message_text) :]
            if delta:
                self.message_text = full_text
                self.send_event_message_delta(msg_id, delta)
            return
        
        self.message_text = full_text

    def _resolve_tool_call_id(self, event: ToolCallDelta) -> str:
        if event.tool_call_id:
            self.tool_id_by_index[event.index] = event.tool_call_id
            return event.tool_call_id
        
        mapped = self.tool_id_by_index.get(event.index)
        if mapped:
            return mapped
        
        fallback = self.last_tool_call_id
        if fallback:
            return fallback
        
        generated = str(uuid.uuid4())
        self.tool_id_by_index[event.index] = generated
        return generated

    def _on_tool_call_delta(self, event: ToolCallDelta) -> None:
        tool_call_id = self._resolve_tool_call_id(event)
        tool = self._ensure_tool_started(tool_call_id, event.name)
        self.last_tool_call_id = tool.tool_call_id

        if event.arguments_delta:
            tool.arguments += event.arguments_delta
            self.send_event_run_step_tool_delta(
                tool.run_step_id,
                tool.tool_call_id,
                tool.name,
                event.arguments_delta,
            )

    def _complete_tool(self, tool: _ToolState) -> None:
        if tool.completed:
            return
        tool.completed = True
        self.send_event_run_step_completed(
            self.assistant_id,
            self.thread_id,
            self.run_id,
            tool.run_step_id,
            tool.tool_call_id,
            tool.name,
            tool.arguments,
        )
    
    def _complete_message(self) -> None:
        msg_id = self.message_id or str(uuid.uuid4())
        self.send_event_message_completed(
            self.assistant_id,
            self.thread_id,
            self.run_id,
            msg_id,
            self.message_text,
        )

    def _send_done_once(self) -> None:
        if self.done_sent:
            return
        self.done_sent = True
        self.send_event_done(self.assistant_id, self.thread_id, self.run_id)

    def _on_error(self, event: StreamError) -> None:
        payload = {
            "error": "StreamingError",
            "message": event.message,
            "code": event.code,
        }
        if event.raw is not None:
            payload["raw"] = str(event.raw)

        self._emit("error", payload)
        self._send_done_once()

    def _on_finished(self, event: StreamFinished) -> None:
        if event.reason == "tool_calls":
            if self.last_tool_call_id and self.last_tool_call_id in self.tools_by_id:
                self._complete_tool(self.tools_by_id[self.last_tool_call_id])
            else:
                for tool in list(self.tools_by_id.values()):
                    self._complete_tool(tool)
            self._send_done_once()
            return
        
        if event.reason == "stop":
            if not self.message_started:
                self._ensure_message_started(self.message_id)
            self._complete_message()
            self._send_done_once()
            return
        
        print(f"StreamingEmitter: finish_reason={event.reason}", flush=True)
        self._send_done_once()

