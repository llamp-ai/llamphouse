from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from ..assistant import Agent
from ..context import Context
from ..types.message import MessageObject


class BaseAgentWrapper(Agent, ABC):
    """Base wrapper for integrating external agent frameworks into LLAMPHouse.

    A wrapper adapts an external runtime (LangGraph, LangChain, etc.) to
    LLAMPHouse's Agent contract while preserving LLAMPHouse adapters (A2A,
    Assistant API), data-store lifecycle, and observability.
    """

    def __init__(
        self,
        *,
        id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        skills: Optional[list] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[str]] = None,
        state_mapper: Optional[Callable[[Context], Dict[str, Any]]] = None,
        output_mapper: Optional[Callable[[Any], str]] = None,
    ) -> None:
        super().__init__(
            id=id,
            name=name,
            description=description,
            version=version,
            skills=skills,
            model=model,
            temperature=temperature,
            top_p=top_p,
            instructions=instructions,
            tools=tools,
        )
        self.state_mapper = state_mapper or self.default_state_mapper
        self.output_mapper = output_mapper or self.default_output_mapper

    def default_state_mapper(self, context: Context) -> Dict[str, Any]:
        """Map LLAMPHouse context -> generic framework state.

        Default shape follows common chat-style state with a `messages` list.
        """
        return {
            "thread_id": context.thread_id,
            "run_id": context.run_id,
            "messages": [self._message_to_dict(m) for m in context.messages],
        }

    def default_output_mapper(self, framework_output: Any) -> str:
        """Map framework output -> assistant reply text."""
        if framework_output is None:
            return ""
        if isinstance(framework_output, str):
            return framework_output
        if isinstance(framework_output, dict):
            # Common shapes across graph/agent frameworks.
            if isinstance(framework_output.get("output"), str):
                return framework_output["output"]
            if isinstance(framework_output.get("text"), str):
                return framework_output["text"]
            msgs = framework_output.get("messages")
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, str):
                    return last
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, str):
                        return content
        return str(framework_output)

    def _message_to_dict(self, message: MessageObject) -> Dict[str, Any]:
        return {
            "id": message.id,
            "role": message.role,
            "content": [p.model_dump() for p in message.parts],
            "run_id": message.run_id,
            "assistant_id": message.assistant_id,
            "metadata": message.metadata or {},
        }

    @abstractmethod
    async def invoke_framework(self, context: Context, state: Dict[str, Any]) -> Any:
        """Execute the external framework and return its final output."""

    async def run(self, context: Context):
        state = self.state_mapper(context)
        output = await self.invoke_framework(context, state)
        reply = self.output_mapper(output)
        if reply:
            await context.reply(reply)
