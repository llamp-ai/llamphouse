"""Sessions: the view of a thread an agent works with on a run.

A thread keeps every message. An agent rarely wants them all: past a few
dozen, older messages cost tokens and attention for little. A **session** is
the agent's view of the thread for one run - a summary of the older
messages, and the recent ones as they are - and a **session manager** keeps
that view: when the thread grows past its limits, it folds the oldest
messages into the summary (compaction).

    class SupportAgent(Agent):
        session_manager = SessionManager(
            max_messages=40,          # compact past 40 messages ...
            keep_messages=20,         # ... down to the latest 20
            summarizer=summarize,     # async (previous_summary, messages) -> str
        )

        async def run(self, context):
            history = context.session.to_messages()   # summary + recent messages
            ...

The summary is kept per thread and agent (in the data store's key-value
records), so each run only folds what is new since the last compaction. The
compaction of a run is recorded as one of its steps: a run that runs again
(durable runs) gets the same session, without calling the summarizer twice.

The summarizer is yours - any LLM, any prompt; ``SessionManager`` calls it
with the summary so far (None at first) and the messages to fold into it.
Without one, folded messages are dropped from the view (a sliding window).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

from .types.message import MessageObject

if TYPE_CHECKING:
    from .context import Context

logger = logging.getLogger("llamphouse.sessions")

Summarizer = Callable[[Optional[str], List[MessageObject]], Awaitable[str]]
TokenCounter = Callable[[str], int]

NAMESPACE_PREFIX = "session:"
_STEP = "session.compact"


def estimate_tokens(text: str) -> int:
    """About 4 characters a token: a rough count, no tokenizer needed."""
    return (len(text) + 3) // 4


@dataclass
class Session:
    """An agent's view of a thread for a run.

    :ivar summary: what the messages before ``messages`` said, or None.
    :ivar messages: the recent messages, as they are, oldest first.
    :ivar folded: how many messages of the thread the summary stands for.
    """
    summary: Optional[str] = None
    messages: List[MessageObject] = field(default_factory=list)
    folded: int = 0

    def to_messages(self, summary_role: str = "system") -> List[Dict[str, str]]:
        """``[{"role", "content"}]`` for a chat-completion API: the summary
        first (as ``summary_role``), then the recent messages."""
        out: List[Dict[str, str]] = []
        if self.summary:
            out.append({"role": summary_role, "content": f"Summary of the earlier conversation:\n{self.summary}"})
        out.extend({"role": m.role, "content": m.text} for m in self.messages)
        return out

    def text(self) -> str:
        """The session as one text: the summary, then ``role: text`` lines."""
        lines = [f"[summary] {self.summary}"] if self.summary else []
        lines.extend(f"{m.role}: {m.text}" for m in self.messages)
        return "\n".join(lines)


class SessionManager:
    """Keeps an agent's session of each thread within limits.

    A run's session holds every message since the last compaction. When
    that is more than ``max_messages`` messages, or more than ``max_tokens``
    tokens, the oldest are folded into the summary until at most
    ``keep_messages`` and ``keep_tokens`` remain - so the summarizer runs
    once in a while, not on every run.

    :param summarizer: ``async (previous_summary, messages) -> summary``.
        None: folded messages are dropped (a sliding window).
    :param max_messages: compact past this many messages (None: no limit).
    :param keep_messages: messages left after compacting (default: half
        of ``max_messages``).
    :param max_tokens: compact past this many tokens (None: no limit).
    :param keep_tokens: tokens left after compacting (default: half of
        ``max_tokens``).
    :param count_tokens: ``(text) -> tokens`` (default: about 4 characters a
        token); pass your model's tokenizer for exact counts.
    """

    def __init__(
        self,
        *,
        summarizer: Optional[Summarizer] = None,
        max_messages: Optional[int] = 50,
        keep_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        keep_tokens: Optional[int] = None,
        count_tokens: TokenCounter = estimate_tokens,
    ) -> None:
        if max_messages is None and max_tokens is None:
            raise ValueError("SessionManager needs max_messages or max_tokens")
        self.summarizer = summarizer
        self.max_messages = max_messages
        self.keep_messages = keep_messages if keep_messages is not None else (
            max(1, max_messages // 2) if max_messages is not None else None
        )
        self.max_tokens = max_tokens
        self.keep_tokens = keep_tokens if keep_tokens is not None else (
            max_tokens // 2 if max_tokens is not None else None
        )
        self.count_tokens = count_tokens

    # ── What the agent sees ──────────────────────────────────────────────

    async def load(self, context: "Context") -> Session:
        """The session of ``context``'s thread for its run, compacted if it
        grew past the limits."""
        key = context._call_key(_STEP)
        done = context._replayed(key)
        if done is not None and done.status == "completed" and (done.step_details.output or {}).get("folded_until"):
            # This run compacted already, in an earlier attempt: the same view.
            result = done.step_details.output
            state = await self._state(context)
            if state.get("folded_until") != result["folded_until"]:
                await self._save(context, result)  # it stopped before saving
            messages = await self._messages_after(context, result["folded_until"])
            return Session(summary=result.get("summary"), messages=messages, folded=result["folded"])

        state = await self._state(context)
        messages = await self._messages_after(context, state.get("folded_until"))
        session = Session(summary=state.get("summary"), messages=messages, folded=state.get("folded", 0))
        if not self._over(session.messages):
            return session

        step = await context.start_step(_STEP, input={"messages": len(session.messages)}, call_key=key)
        result = await self._compact(session)
        await context.complete_step(step.id, output=result)
        if result.get("folded_until"):
            await self._save(context, result)
        return self._apply(session, result)

    async def compact(self, session: Session) -> Session:
        """Fold ``session``'s oldest messages now, whatever the limits say
        (down to ``keep_messages`` / ``keep_tokens``). Not persisted: for
        tools and tests; runs compact through ``load``."""
        return self._apply(session, await self._compact(session))

    # ── Compaction ───────────────────────────────────────────────────────

    def _tokens(self, messages: List[MessageObject]) -> int:
        return sum(self.count_tokens(m.text) for m in messages)

    def _over(self, messages: List[MessageObject]) -> bool:
        if self.max_messages is not None and len(messages) > self.max_messages:
            return True
        return self.max_tokens is not None and self._tokens(messages) > self.max_tokens

    def _fold_count(self, messages: List[MessageObject]) -> int:
        """How many of the oldest messages to fold."""
        n = 0
        if self.keep_messages is not None:
            n = max(n, len(messages) - self.keep_messages)
        if self.keep_tokens is not None:
            tokens = self._tokens(messages[n:])
            while n < len(messages) - 1 and tokens > self.keep_tokens:
                tokens -= self.count_tokens(messages[n].text)
                n += 1
        return min(n, len(messages) - 1)  # the latest message always stays

    async def _compact(self, session: Session) -> Dict[str, Any]:
        n = self._fold_count(session.messages)
        if n <= 0:
            return {}
        folding = session.messages[:n]
        summary = session.summary
        if self.summarizer is not None:
            summary = await self.summarizer(session.summary, folding)
        return {
            "summary": summary,
            "folded_until": folding[-1].id,
            "folded": session.folded + n,
            "count": n,
        }

    @staticmethod
    def _apply(session: Session, result: Dict[str, Any]) -> Session:
        if not result.get("folded_until"):
            return session
        return Session(
            summary=result.get("summary"),
            messages=session.messages[result["count"]:],
            folded=result["folded"],
        )

    # ── Persistence ──────────────────────────────────────────────────────

    @staticmethod
    def _namespace(context: "Context") -> str:
        return f"{NAMESPACE_PREFIX}{context.assistant_id}:{context.thread_id}"

    @staticmethod
    def _persists(context: "Context") -> bool:
        from .data_stores.base_data_store import BaseDataStore

        return type(context.data_store).kv_get_all is not BaseDataStore.kv_get_all

    async def _state(self, context: "Context") -> Dict[str, Any]:
        if not self._persists(context):
            return {}
        records = await context.data_store.kv_get_all(self._namespace(context))
        return records.get("state") or {}

    async def _save(self, context: "Context", result: Dict[str, Any]) -> None:
        if not self._persists(context):
            logger.warning(
                "%s keeps no key-value records: the session summary of thread %s is not kept.",
                type(context.data_store).__name__, context.thread_id,
            )
            return
        state = {k: result.get(k) for k in ("summary", "folded_until", "folded")}
        await context.data_store.kv_put(self._namespace(context), {"state": state}, thread_id=context.thread_id)

    @staticmethod
    async def _messages_after(context: "Context", message_id: Optional[str]) -> List[MessageObject]:
        """The thread's messages after ``message_id`` (all if None), oldest
        first - less what this run said already, if it runs again."""
        own = context._said_ids
        out: List[MessageObject] = []
        after = message_id
        while True:
            page = await context.data_store.list_messages(
                thread_id=context.thread_id, limit=100, order="asc", after=after, before=None,
            )
            if not page or not page.data:
                break
            out.extend(m for m in page.data if m.id not in own)
            if not page.has_more:
                break
            after = page.data[-1].id
        return out


__all__ = ["Session", "SessionManager", "Summarizer", "estimate_tokens"]
