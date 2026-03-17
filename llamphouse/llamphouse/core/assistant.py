from typing import List, Optional
from abc import ABC, abstractmethod
import uuid
from datetime import datetime, timezone
from .context import Context
from .types.config import BaseParam

class Agent(ABC):
    """Base class for all LLAMPHouse agents.

    Subclass this and implement ``run(context)`` to define your agent's
    behavior.  Previously named ``Assistant`` — that name is kept as a
    backward-compatible alias.

    Core parameters (id, name, description, version, skills) are used by
    every adapter.  Assistant-API parameters (model, temperature, top_p,
    instructions, tools) are optional and only relevant when the
    ``AssistantAPIAdapter`` is mounted.
    """
    config: List[BaseParam] = []
    def __init__(
        self,
        id: str,
        name: Optional[str] = None, 
        description: Optional[str] = None, 
        version: Optional[str] = None,
        skills: Optional[list] = None,
        # ── Assistant-API parameters (only needed with AssistantAPIAdapter) ──
        model: Optional[str] = None,
        temperature: Optional[float] = None, 
        top_p: Optional[float] = None, 
        instructions: Optional[str] = None,
        tools: Optional[List[str]] = None,
    ):
        # ── Core identity ───────────────────────────────────────────────
        self.id = id
        self.name = name
        self.description = description
        self.version = version
        self.skills = skills
        self.created_at = datetime.now(timezone.utc)

        # ── Assistant-API fields ────────────────────────────────────────
        self.model = model or ""
        self.object = "assistant"
        self.temperature = temperature
        self.top_p = top_p
        self.instructions = instructions or description or ""
        self.tools = tools or []

        # ── Name fallback ───────────────────────────────────────────
        if not name:
            self.name = id

    async def on_startup(self) -> None:
        """Called once when the server starts.

        Override this to initialise expensive resources such as HTTP
        clients, database connections, or model handles.
        """

    async def on_shutdown(self) -> None:
        """Called once when the server shuts down.

        Override this to close HTTP clients, database connections, or
        any other resources that were opened in ``on_startup``.
        """

    @abstractmethod
    async def run(self, context: Context):
        """
        Define the agent's behavior.

        Override this method to process messages within a thread/run
        and produce responses via ``context.reply()``.

        :param context: The execution context with messages, thread, and helpers.
        """
        pass


# Backward-compatible alias
Assistant = Agent