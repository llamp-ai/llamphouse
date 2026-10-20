"""Sessions: the view of a thread an agent works with on a run.

A thread keeps every message. An agent rarely wants them all: past a few
dozen, older messages cost tokens and attention for little. A **session** is
the agent's view of the thread for one run - the messages it gets, and a
summary of older ones if it keeps one - and a **session manager** decides
that view. Three come with LLAMPHouse:

- ``FullHistory()`` - the whole thread.
- ``SlidingWindow(max_messages=20)`` - the latest messages (or tokens),
  optionally with the thread's first few kept in view.
- ``Summarizing(summarizer, max_messages=40, keep_messages=20)`` - past the
  limit, the oldest messages are folded into a running summary
  (compaction); ``llm_summarizer`` and ``openai_summarizer`` make the
  summarizer from an LLM call.

::

    class SupportAgent(Agent):
        session_manager = Summarizing(openai_summarizer(), max_messages=40, keep_messages=20)

        async def run(self, context):
            reply = await llm(messages=context.session.to_messages())
            ...

A manager of your own subclasses ``SessionManager`` and implements
``build(messages, memory)``: what the agent sees, from the thread's messages
and what the manager kept from earlier runs. Everything else - reading the
thread, keeping the memory per thread and agent, recording a compaction so
a run that runs again (durable runs) gets the same session - is done here.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

from .types.message import MessageObject, TextPart

if TYPE_CHECKING:
    from .context import Context

logger = logging.getLogger("llamphouse.sessions")

Summarizer = Callable[[Optional[str], List[MessageObject]], Awaitable[str]]
TokenCounter = Callable[[str], int]

NAMESPACE_PREFIX = "session:"
_STEP = "session.compact"
_PAGE = 100


def estimate_tokens(text: str) -> int:
    """About 4 characters a token: a rough count, no tokenizer needed."""
    return (len(text) + 3) // 4


@dataclass
class Session:
    """An agent's view of a thread for a run.

    :ivar summary: what older messages said, or None.
    :ivar messages: the messages in view, oldest first.
    :ivar folded: how many of the thread's messages the summary stands for.
    :ivar memory: set by ``SessionManager.build`` to what the manager keeps
        for the next runs (it replaces what was kept); None keeps it as it is.
    """
    summary: Optional[str] = None
    messages: List[MessageObject] = field(default_factory=list)
    folded: int = 0
    memory: Optional[Dict[str, Any]] = None

    def to_messages(self, summary_role: str = "system") -> List[Dict[str, str]]:
        """``[{"role", "content"}]`` for a chat-completion API: the summary
        first (as ``summary_role``), then the messages."""
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


# ── The base: subclass it for a manager of your own ──────────────────────


class SessionManager(ABC):
    """Decides an agent's view of a thread, run after run.

    Subclass it and implement ``build``; override ``messages`` to read less
    of the thread. The memory (``build``'s second argument, and
    ``Session.memory``) is a JSON dict the manager keeps per thread and
    agent; a ``"folded_until"`` message id in it means "the session no
    longer needs this message and those before it": ``messages`` starts
    after it.

    A session whose ``memory`` is set (the manager changed its memory - it
    compacted) is recorded as a ``session.compact`` step of the run: a run
    that runs again gets that same session, and ``build`` is not called
    twice for it. Any other ``build`` must give the same view for the same
    messages and memory.

    :param count_tokens: ``(text) -> tokens`` (default: about 4 characters
        a token); pass your model's tokenizer for exact counts.
    :param max_message_tokens: shorten any single message in view past this
        many tokens (a pasted document, a long tool output) - in the view
        only, the thread keeps it whole.
    """

    #: Whether the manager keeps a memory (read and written each run).
    uses_memory: bool = True

    def __init__(self, *, count_tokens: TokenCounter = estimate_tokens, max_message_tokens: Optional[int] = None) -> None:
        self.count_tokens = count_tokens
        self.max_message_tokens = max_message_tokens

    @abstractmethod
    async def build(self, messages: List[MessageObject], memory: Dict[str, Any]) -> Session:
        """The session of a run: from the thread's ``messages`` (what
        ``messages()`` read, oldest first) and the manager's ``memory``
        ({} at first). Set ``Session.memory`` to change the memory."""

    async def messages(self, context: "Context", memory: Dict[str, Any]) -> List[MessageObject]:
        """The thread's messages ``build`` gets: by default every one after
        ``memory["folded_until"]`` (all of them, at first)."""
        return await thread_messages(context, after=memory.get("folded_until"))

    # Helpers for subclasses.

    def tokens(self, messages: List[MessageObject]) -> int:
        return sum(self.count_tokens(m.text) for m in messages)

    # ── What the framework calls ─────────────────────────────────────────

    async def load(self, context: "Context") -> Session:
        """The session of ``context``'s run."""
        key = context._call_key(_STEP)
        done = context._replayed(key)
        if done is not None and done.status == "completed" and isinstance(done.step_details.output, dict):
            session = await self._replay(context, done.step_details.output)
        else:
            memory = await self._memory(context) if self.uses_memory else {}
            session = await self.build(await self.messages(context, memory), memory)
            if session.memory is not None:
                step = await context.start_step(_STEP, input={"messages": len(session.messages)}, call_key=key)
                await context.complete_step(step.id, output=_record(session))
                await self._save(context, session.memory)
        return self._trim(session)

    async def _replay(self, context: "Context", recorded: Dict[str, Any]) -> Session:
        """The session this run had in an earlier attempt."""
        memory = recorded.get("memory") or {}
        if self.uses_memory and await self._memory(context) != memory:
            await self._save(context, memory)  # that attempt stopped before saving
        messages = []
        for message_id in recorded.get("message_ids") or []:
            message = await context.data_store.get_message_by_id(context.thread_id, message_id)
            if message is not None:
                messages.append(message)
        return Session(summary=recorded.get("summary"), messages=messages, folded=recorded.get("folded", 0))

    def _trim(self, session: Session) -> Session:
        if self.max_message_tokens is None:
            return session
        limit = self.max_message_tokens
        trimmed = []
        for m in session.messages:
            if self.count_tokens(m.text) <= limit:
                trimmed.append(m)
                continue
            # About `limit` tokens of it, by the same ~4 characters a token.
            text = m.text[: max(1, limit * 4)].rstrip() + " … [shortened]"
            trimmed.append(m.model_copy(update={"parts": [TextPart(text=text)]}))
        return Session(summary=session.summary, messages=trimmed, folded=session.folded, memory=session.memory)

    @staticmethod
    def _namespace(context: "Context") -> str:
        return f"{NAMESPACE_PREFIX}{context.assistant_id}:{context.thread_id}"

    @staticmethod
    def _persists(context: "Context") -> bool:
        from .data_stores.base_data_store import BaseDataStore

        return type(context.data_store).kv_get_all is not BaseDataStore.kv_get_all

    async def _memory(self, context: "Context") -> Dict[str, Any]:
        if not self._persists(context):
            return {}
        records = await context.data_store.kv_get_all(self._namespace(context))
        return records.get("state") or {}

    async def _save(self, context: "Context", memory: Dict[str, Any]) -> None:
        if not self._persists(context):
            logger.warning(
                "%s keeps no key-value records: the session memory of thread %s is not kept.",
                type(context.data_store).__name__, context.thread_id,
            )
            return
        await context.data_store.kv_put(self._namespace(context), {"state": memory}, thread_id=context.thread_id)


def _record(session: Session) -> Dict[str, Any]:
    return {
        "summary": session.summary,
        "folded": session.folded,
        "message_ids": [m.id for m in session.messages],
        "memory": session.memory,
    }


async def thread_messages(
    context: "Context",
    *,
    after: Optional[str] = None,
    latest: Optional[int] = None,
    first: Optional[int] = None,
) -> List[MessageObject]:
    """Messages of ``context``'s thread, oldest first, less what this run
    said already if it runs again: every one ``after`` a message id (all if
    None), or only the ``latest`` n, or the ``first`` n."""
    own = context._said_ids
    store, thread_id = context.data_store, context.thread_id
    if latest is not None or first is not None:
        n = latest if latest is not None else first
        order = "desc" if latest is not None else "asc"
        out: List[MessageObject] = []
        cursor = None
        while len(out) < n:
            page = await store.list_messages(thread_id=thread_id, limit=_PAGE, order=order, after=cursor, before=None)
            if not page or not page.data:
                break
            out.extend(m for m in page.data if m.id not in own)
            if not page.has_more:
                break
            cursor = page.data[-1].id
        out = out[:n]
        return list(reversed(out)) if latest is not None else out
    out = []
    cursor = after
    while True:
        page = await store.list_messages(thread_id=thread_id, limit=_PAGE, order="asc", after=cursor, before=None)
        if not page or not page.data:
            break
        out.extend(m for m in page.data if m.id not in own)
        if not page.has_more:
            break
        cursor = page.data[-1].id
    return out


# ── Built-in managers ────────────────────────────────────────────────────


class FullHistory(SessionManager):
    """The whole thread, as it is. For short conversations - or with
    ``max_message_tokens``, to keep a few long messages in check."""

    uses_memory = False

    async def build(self, messages: List[MessageObject], memory: Dict[str, Any]) -> Session:
        return Session(messages=list(messages))


class SlidingWindow(SessionManager):
    """The latest messages: at most ``max_messages`` of them, and at most
    ``max_tokens`` tokens. Nothing is summarized; older messages are just
    out of view.

    :param keep_first: also keep the thread's first n messages in view (the
        task the conversation started with, say).
    """

    uses_memory = False

    def __init__(
        self,
        max_messages: Optional[int] = 20,
        *,
        max_tokens: Optional[int] = None,
        keep_first: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if max_messages is None and max_tokens is None:
            raise ValueError("SlidingWindow needs max_messages or max_tokens")
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.keep_first = keep_first

    async def messages(self, context: "Context", memory: Dict[str, Any]) -> List[MessageObject]:
        if self.max_tokens is None:
            recent = await thread_messages(context, latest=self.max_messages + self.keep_first)
        else:
            recent = await thread_messages(context)  # a token budget: read it all
        if not self.keep_first:
            return recent
        first = await thread_messages(context, first=self.keep_first)
        seen = {m.id for m in recent}
        return [m for m in first if m.id not in seen] + recent

    async def build(self, messages: List[MessageObject], memory: Dict[str, Any]) -> Session:
        pinned = messages[: self.keep_first]
        rest = messages[self.keep_first:]
        if self.max_messages is not None:
            rest = rest[-self.max_messages:] if self.max_messages else []
        if self.max_tokens is not None:
            budget = self.max_tokens - self.tokens(pinned)
            kept: List[MessageObject] = []
            for m in reversed(rest):
                cost = self.count_tokens(m.text)
                if kept and cost > budget:
                    break
                kept.append(m)
                budget -= cost
            rest = list(reversed(kept))
        return Session(messages=pinned + rest)


class Summarizing(SessionManager):
    """Past its limits, folds the oldest messages into a running summary.

    The session holds every message since the last compaction. When that is
    more than ``max_messages`` messages, or more than ``max_tokens`` tokens,
    the oldest are folded into the summary until at most ``keep_messages``
    and ``keep_tokens`` remain - so the summarizer runs once in a while, not
    on every run. The summary is kept per thread and agent: each compaction
    folds only what is new, on top of the summary so far.

    :param summarizer: ``async (previous_summary, messages) -> summary``
        (see ``llm_summarizer``, ``openai_summarizer``).
    :param max_messages: compact past this many messages (None: no limit).
    :param keep_messages: messages left after compacting (default: half of
        ``max_messages``).
    :param max_tokens: compact past this many tokens (None: no limit).
    :param keep_tokens: tokens left after compacting (default: half of
        ``max_tokens``).
    """

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        max_messages: Optional[int] = 50,
        keep_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        keep_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if max_messages is None and max_tokens is None:
            raise ValueError("Summarizing needs max_messages or max_tokens")
        self.summarizer = summarizer
        self.max_messages = max_messages
        self.keep_messages = keep_messages if keep_messages is not None else (
            max(1, max_messages // 2) if max_messages is not None else None
        )
        self.max_tokens = max_tokens
        self.keep_tokens = keep_tokens if keep_tokens is not None else (
            max_tokens // 2 if max_tokens is not None else None
        )

    def over(self, messages: List[MessageObject]) -> bool:
        if self.max_messages is not None and len(messages) > self.max_messages:
            return True
        return self.max_tokens is not None and self.tokens(messages) > self.max_tokens

    def fold_count(self, messages: List[MessageObject]) -> int:
        """How many of the oldest messages to fold (the latest always stays)."""
        n = 0
        if self.keep_messages is not None:
            n = max(n, len(messages) - self.keep_messages)
        if self.keep_tokens is not None:
            tokens = self.tokens(messages[n:])
            while n < len(messages) - 1 and tokens > self.keep_tokens:
                tokens -= self.count_tokens(messages[n].text)
                n += 1
        return min(n, len(messages) - 1)

    async def build(self, messages: List[MessageObject], memory: Dict[str, Any]) -> Session:
        session = Session(summary=memory.get("summary"), messages=list(messages), folded=memory.get("folded", 0))
        if not self.over(messages):
            return session
        return await self.compact(session)

    async def compact(self, session: Session) -> Session:
        """Fold ``session``'s oldest messages into its summary now, down to
        ``keep_messages`` / ``keep_tokens``."""
        n = self.fold_count(session.messages)
        if n <= 0:
            return session
        folding = session.messages[:n]
        summary = await self.summarizer(session.summary, folding)
        folded = session.folded + n
        return Session(
            summary=summary,
            messages=session.messages[n:],
            folded=folded,
            memory={"summary": summary, "folded_until": folding[-1].id, "folded": folded},
        )


# ── Summarizers ──────────────────────────────────────────────────────────

SUMMARY_INSTRUCTIONS = (
    "You maintain the running summary of a conversation between a user and an assistant. "
    "Update the summary with the new messages. Keep what matters for continuing the "
    "conversation: the user's goals, facts and preferences they gave, decisions made, "
    "open questions and commitments. Drop small talk. Write plain prose, at most {max_words} "
    "words. Reply with the summary only."
)


def summary_prompt(previous: Optional[str], messages: List[MessageObject], instructions: str) -> str:
    transcript = "\n".join(f"{m.role}: {m.text}" for m in messages)
    return (
        f"{instructions}\n\n"
        f"Summary so far:\n{previous or '(none yet)'}\n\n"
        f"New messages:\n{transcript}"
    )


def llm_summarizer(
    complete: Callable[[str], Union[str, Awaitable[str]]],
    *,
    instructions: str = SUMMARY_INSTRUCTIONS,
    max_words: int = 200,
) -> Summarizer:
    """A summarizer from any LLM call: ``complete(prompt) -> text`` (sync or
    async). The prompt holds ``instructions``, the summary so far and the
    messages to fold in.

        Summarizing(llm_summarizer(lambda prompt: my_llm(prompt)), max_messages=40)
    """
    text = instructions.format(max_words=max_words) if "{max_words}" in instructions else instructions

    async def summarize(previous: Optional[str], messages: List[MessageObject]) -> str:
        result = complete(summary_prompt(previous, messages, text))
        if inspect.isawaitable(result):
            result = await result
        return str(result).strip()

    return summarize


def openai_summarizer(
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
    instructions: str = SUMMARY_INSTRUCTIONS,
    max_words: int = 200,
    **create_kwargs: Any,
) -> Summarizer:
    """A summarizer on an OpenAI-compatible chat-completions API. ``client``:
    an ``openai.AsyncOpenAI`` (default: one from the environment, made on
    first use); ``create_kwargs`` go to ``chat.completions.create``."""
    state: Dict[str, Any] = {"client": client}

    async def complete(prompt: str) -> str:
        if state["client"] is None:
            from openai import AsyncOpenAI

            state["client"] = AsyncOpenAI()
        response = await state["client"].chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **create_kwargs,
        )
        return response.choices[0].message.content or ""

    return llm_summarizer(complete, instructions=instructions, max_words=max_words)


__all__ = [
    "Session",
    "SessionManager",
    "FullHistory",
    "SlidingWindow",
    "Summarizing",
    "Summarizer",
    "llm_summarizer",
    "openai_summarizer",
    "thread_messages",
    "estimate_tokens",
]
