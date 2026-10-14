import contextvars
import functools
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional, List, TYPE_CHECKING
from .retention import RetentionPolicy, PurgeStats
from ..types.run import ModifyRunRequest, RunCreateRequest, RunObject, ToolOutput
from ..types.thread import CreateThreadRequest, ModifyThreadRequest, ThreadObject
from ..types.assistant import AgentObject, AssistantObject
from ..types.message import CreateMessageRequest, MessageObject, ModifyMessageRequest
from ..types.enum.message_status import COMPLETED as MESSAGE_COMPLETED
from ..types.list import ListResponse
from ..types.run_step import CreateRunStepRequest, RunStepObject
from ..types.webhook import WebhookCommand, WebhookCommandResult
from ..streaming.event_queue.base_event_queue import BaseEventQueue

if TYPE_CHECKING:
    from ..assistant import Agent

logger = logging.getLogger("llamphouse.data_store")

# Told of every change of a run's status: (the run as it is now, the status
# it had - None for a new run).
RunStatusListener = Callable[[RunObject, Optional[str]], Awaitable[None]]


# Set while a wrapped write runs: a store built on another store (calling
# super()) is wrapped twice, and only the outermost write notifies.
_NOTIFYING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "llamphouse_notifying", default=False
)


def _notifying(method: Callable, *, created: bool) -> Callable:
    """Wrap a store's status write so its listeners hear about the change."""

    @functools.wraps(method)
    async def wrapper(self: "BaseDataStore", *args: Any, **kwargs: Any) -> Any:
        if _NOTIFYING.get():  # inside another store's wrapped write
            return await method(self, *args, **kwargs)
        listeners = list(getattr(self, "_run_status_listeners", None) or [])
        previous = None
        if listeners and not created:
            thread_id, run_id = _ids(args, kwargs)
            before = await self.get_run_by_id(thread_id, run_id)
            previous = before.status if before is not None else None
        token = _NOTIFYING.set(True)
        try:
            result = await method(self, *args, **kwargs)
        finally:
            _NOTIFYING.reset(token)
        # Outside the guard: a status a listener writes is heard too.
        if listeners and result is not None and (created or result.status != previous):
            for listener in listeners:
                try:
                    await listener(result, previous)
                except Exception:  # a listener never breaks the store
                    logger.exception("Run status listener failed for %s", result.id)
        return result

    return wrapper


def _ids(args: tuple, kwargs: dict) -> tuple[str, str]:
    thread_id = kwargs.get("thread_id", args[0] if args else None)
    run_id = kwargs.get("run_id", args[1] if len(args) > 1 else None)
    return thread_id, run_id


class BaseDataStore(ABC):

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Every store's status writes notify, without each store having to.
        if "update_run_status" in cls.__dict__:
            cls.update_run_status = _notifying(cls.__dict__["update_run_status"], created=False)
        if "insert_run" in cls.__dict__:
            cls.insert_run = _notifying(cls.__dict__["insert_run"], created=True)

    def __init__(self):
        self.agents: list["Agent"] = []
        # Backward-compatible alias
        self.assistants = self.agents
        pass

    def init(self, agents: list["Agent"]) -> None:
        """Set the list of agents."""
        self.agents = agents
        self.assistants = agents  # backward-compat alias

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
    async def list_threads(
        self,
        limit: int = 50,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[List[dict]] = None,
        include_total: bool = True,
    ) -> ListResponse | None:
        """List threads with pagination, ordering, and optional filters.

        ``filters`` is a list of ``{"field", "operator", "value", "value2"?}``
        dicts.  Implementations should silently ignore filters referencing
        unsupported fields.

        Set ``include_total=False`` to skip the matching ``COUNT(*)`` query —
        useful for views that only need a page of rows and treat the total as
        a nice-to-have.
        """
        pass

    @abstractmethod
    async def get_run_by_id(self, thread_id: str, run_id: str) -> RunObject | None:
        """Retrieve a run by its ID."""
        pass

    @abstractmethod
    async def get_run_by_run_id(self, run_id: str) -> RunObject | None:
        """Retrieve a run by its ID across all threads."""
        pass

    @abstractmethod
    async def insert_run(self, thread_id: str, run: RunCreateRequest, assistant: AgentObject, event_queue: BaseEventQueue = None) -> RunObject | None:
        """Insert a new run associated with a thread."""
        pass

    async def execute_webhook_command(self, command: WebhookCommand) -> WebhookCommandResult:
        """Atomically execute an inbound webhook command.

        Stores that support webhook idempotency should override this method
        with a single transaction or critical section covering idempotency
        claim, thread/message/run creation, and response persistence.
        """
        raise NotImplementedError("execute_webhook_command is not implemented by this data store.")

    @abstractmethod
    async def list_runs(self, thread_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        """List runs for a specific thread with pagination and ordering."""
        pass

    @abstractmethod
    async def list_all_runs(
        self,
        limit: int = 50,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[List[dict]] = None,
        include_total: bool = True,
    ) -> ListResponse | None:
        """List runs across all threads with pagination, ordering, and optional filters.

        Set ``include_total=False`` to skip the matching ``COUNT(*)`` query.
        """
        pass

    @abstractmethod
    async def get_run_any_thread(self, run_id: str) -> RunObject | None:
        """Fetch a single run by id without knowing its ``thread_id``.

        Used by graph-walking flows (e.g. the Compass agent flow view) that
        need to traverse parent_run_id pointers without scanning all runs.
        """
        pass

    @abstractmethod
    async def list_runs_by_parent_ids(self, parent_ids: List[str]) -> List[RunObject]:
        """Return every run whose ``metadata.parent_run_id`` is in
        ``parent_ids``.  One bulk query — used to BFS down a run tree."""
        pass

    @abstractmethod
    async def count_threads(self) -> int:
        """Total number of threads in the store."""
        pass

    @abstractmethod
    async def count_runs(self) -> int:
        """Total number of runs in the store."""
        pass

    @abstractmethod
    async def count_messages(self) -> int:
        """Total number of messages in the store."""
        pass

    async def get_first_run_assistant_ids(self, thread_ids: List[str]) -> dict[str, str]:
        """Return a ``{thread_id: assistant_id}`` mapping for the first
        (earliest) run in each given thread.  Threads with no runs are
        omitted from the result.

        The default implementation loops over ``list_runs`` per thread.
        Concrete stores should override with a single bulk query.
        """
        out: dict[str, str] = {}
        for tid in thread_ids:
            try:
                result = await self.list_runs(tid, limit=1, order="asc", after=None, before=None)
            except Exception:
                continue
            if result and result.data:
                aid = getattr(result.data[0], "assistant_id", None)
                if aid:
                    out[tid] = aid
        return out

    @abstractmethod
    async def update_run(self, thread_id: str, run_id: str, modifications: ModifyRunRequest) -> RunObject | None:
        """Update an existing run."""
        pass

    @abstractmethod
    async def submit_tool_outputs_to_run(self, thread_id: str, run_id: str, tool_outputs: List[ToolOutput]) -> RunObject | None:
        """Submit tool outputs to a specific run."""
        pass

    @abstractmethod
    async def insert_run_step(self, thread_id: str, run_id: str, step: CreateRunStepRequest, status: str = "completed", event_queue: BaseEventQueue = None) -> RunStepObject | None:
        """Insert a new step for a specific run."""
        pass

    @abstractmethod
    async def list_run_steps(self, thread_id: str, run_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        """List steps for a specific run with pagination and ordering."""
        pass

    @abstractmethod
    async def get_run_step_by_id(self, thread_id: str, run_id: str, step_id: str) -> RunStepObject | None:
        """Retrieve a run step by its ID within a specific thread and run."""
        pass

    @abstractmethod
    async def get_latest_run_step_by_run_id(self, run_id: str) -> RunStepObject | None:
        """Retrieve the most recent run step for a run."""
        pass

    @abstractmethod
    async def update_run_status(self, thread_id: str, run_id: str, status: str, error: dict | None = None, usage: dict | None = None) -> RunObject | None:
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

    async def close(self) -> None:
        """Close any underlying resources (default: no-op)."""
        return None

    # ── Run lifecycle ────────────────────────────────────────────────────

    def add_run_status_listener(self, listener: RunStatusListener) -> None:
        """Be told of every change of a run's status made through this store
        (in this process): ``await listener(run, previous_status)``."""
        if not hasattr(self, "_run_status_listeners"):
            self._run_status_listeners: list[RunStatusListener] = []
        self._run_status_listeners.append(listener)

    def remove_run_status_listener(self, listener: RunStatusListener) -> None:
        if listener in getattr(self, "_run_status_listeners", []):
            self._run_status_listeners.remove(listener)

    async def heartbeat_run(self, thread_id: str, run_id: str) -> str | None:
        """Record that a running run is alive; returns its status, so the
        worker sees a cancel requested meanwhile. Stores that keep a
        heartbeat override this; by default it only reads the status."""
        run = await self.get_run_by_id(thread_id, run_id)
        return run.status if run is not None else None

    async def request_cancel(self, thread_id: str, run_id: str) -> RunObject | None:
        """Ask for a run to stop: a queued one is cancelled at once (its
        worker skips it), a running one becomes 'cancelling' and its worker
        stops it at the next heartbeat. A finished run is left as it is."""
        from ..types.enum import run_status

        run = await self.get_run_by_id(thread_id, run_id)
        if run is None:
            return None
        if run.status == run_status.QUEUED:
            return await self.update_run_status(thread_id, run_id, run_status.CANCELLED)
        if run.status in (run_status.IN_PROGRESS, run_status.AWAITING_TOOLS):
            return await self.update_run_status(thread_id, run_id, run_status.CANCELLING)
        return run
