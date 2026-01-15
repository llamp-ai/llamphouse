from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional, List, TYPE_CHECKING
from .retention import RetentionPolicy, PurgeStats
from ..types.run import ModifyRunRequest, RunCreateRequest, RunObject, ToolOutput
from ..types.thread import CreateThreadRequest, ModifyThreadRequest, ThreadObject
from ..types.assistant import AssistantObject
from ..types.message import CreateMessageRequest, MessageObject, ModifyMessageRequest
from ..types.enum.message_status import COMPLETED as MESSAGE_COMPLETED
from ..types.list import ListResponse
from ..types.run_step import CreateRunStepRequest, RunStepObject
from ..streaming.event_queue.base_event_queue import BaseEventQueue

if TYPE_CHECKING:
    from ..assistant import Assistant

class BaseDataStore(ABC):

    def __init__(self):
        self.assistants: list["Assistant"] = []
        pass

    def init(self, assistants: list["Assistant"]) -> None:
        """Set the list of assistants."""
        self.assistants = assistants

    @abstractmethod
    async def listen(self) -> AsyncIterator[Any]:
        """Yield new items as they arrive."""
        pass

    @abstractmethod
    async def ack(self, item: Any) -> None:
        """Mark item as processed (if applicable)."""
        pass

    @abstractmethod
    async def push(self, item: Any) -> None:
        """Add a new item to the storage."""
        pass

    @abstractmethod
    async def insert_message(self, thread_id: str, message: CreateMessageRequest, status: str = MESSAGE_COMPLETED, event_queue: BaseEventQueue = None) -> MessageObject | None:
        """Insert a new message into a thread."""
        pass

    @abstractmethod
    async def list_messages(self, thread_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        """List messages for a specific thread with pagination and ordering."""
        pass

    @abstractmethod
    async def get_message_by_id(self, thread_id: str, message_id: str) -> MessageObject | None:
        """Retrieve a message by its ID within a specific thread."""
        pass

    @abstractmethod
    async def update_message(self, thread_id: str, message_id: str, modifications: ModifyMessageRequest) -> MessageObject | None:
        """Update an existing message."""
        pass

    @abstractmethod
    async def delete_message(self, thread_id: str, message_id: str) -> str | None:
        """Delete a message by its ID within a specific thread."""
        pass

    @abstractmethod
    async def get_thread_by_id(self, thread_id: str) -> ThreadObject | None:
        """Retrieve a thread by its ID."""
        pass

    @abstractmethod
    async def update_thread(self, thread_id: str, modifications: ModifyThreadRequest) -> ThreadObject | None:
        """Update thread."""
        pass

    @abstractmethod
    async def delete_thread(self, thread_id: str) -> str | None:
        """Delete a thread by its ID."""
        pass

    @abstractmethod
    async def insert_thread(self, thread: CreateThreadRequest, event_queue: BaseEventQueue = None) -> ThreadObject | None:
        """Insert a new thread."""
        pass

    @abstractmethod
    async def get_run_by_id(self, thread_id: str, run_id: str) -> RunObject | None:
        """Retrieve a run by its ID."""
        pass

    @abstractmethod
    async def insert_run(self, thread_id: str, run: RunCreateRequest, assistant: AssistantObject, event_queue: BaseEventQueue = None) -> RunObject | None:
        """Insert a new run associated with a thread."""
        pass

    @abstractmethod
    async def list_runs(self, thread_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        """List runs for a specific thread with pagination and ordering."""
        pass

    @abstractmethod
    async def update_run(self, thread_id: str, run_id: str, modifications: ModifyRunRequest) -> RunObject | None:
        """Update an existing run."""
        pass

    @abstractmethod
    async def submit_tool_outputs_to_run(self, thread_id: str, run_id: str, tool_outputs: List[ToolOutput]) -> RunObject | None:
        """Submit tool outputs to a specific run."""
        pass

    @abstractmethod
    async def insert_run_step(self, thread_id: str, run_id: str, step: CreateRunStepRequest) -> RunStepObject | None:
        """Insert a new step for a specific run."""
        pass

    @abstractmethod
    def list_run_steps(self, thread_id: str, run_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        """List steps for a specific run with pagination and ordering."""
        pass

    @abstractmethod
    def get_run_step_by_id(self, thread_id: str, run_id: str, step_id: str) -> RunStepObject | None:
        """Retrieve a run step by its ID within a specific thread and run."""
        pass

    @abstractmethod
    async def get_latest_run_step_by_run_id(self, run_id: str) -> RunStepObject | None:
        """Retrieve the most recent run step for a run."""
        pass

    @abstractmethod
    async def update_run_status(self, thread_id: str, run_id: str, status: str, error: dict | None = None) -> RunObject | None:
        """Update status of a run."""
        pass

    @abstractmethod
    async def update_run_step_status(self, run_step_id: str, status: str, output=None, error: str | None = None) -> RunStepObject | None:
        """Update status/output/error of a run step."""
        pass

    @abstractmethod
    async def purge_expired(self, policy: RetentionPolicy) -> PurgeStats:
        """Purge records older than policy cutoff (respects dry_run/batch_size)."""
        pass

    def close(self) -> None:
        """Close any underlying resources (default: no-op)."""
        return None