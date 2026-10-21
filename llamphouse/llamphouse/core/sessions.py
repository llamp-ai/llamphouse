"""Sessions: the view of a thread an agent works with on a run.

A thread keeps every message. An agent rarely wants them all: past a few
dozen, older messages cost tokens and attention for little. A **session** is
the agent's view of the thread for one run - the messages it gets, and a
summary of older ones if it keeps one - and a **session manager** decides
that view. Three come with LLAMPHouse:

- ``FullHistory()`` - the whole thread.
- ``SlidingWindow(max_messages=20)`` - the latest messages (or tokens),
  optionally with the thread's first few kept in view.
- ``Summarizing(summarizer, max_messages=40, keep_messages=20)`` - the latest
  messages, and a running summary of the older ones.
- ``Compacting(summarizer, max_tokens=100_000)`` - a token budget that fills
  up; at 90% it is compacted to 25% (a new summary and the latest
  messages), and fills up again.

``llm_summarizer`` and ``openai_summarizer`` make the summarizer from an
LLM call.

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

# ``async (previous_summary, messages) -> summary``; one that also takes a
# ``max_tokens`` keyword is told the summary's target length.
Summarizer = Callable[..., Awaitable[str]]
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
        count_tokens: TokenCounter = estimate_tokens,
        max_message_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(count_tokens=count_tokens, max_message_tokens=max_message_tokens)
        if max_messages is None and max_tokens is None:
            raise ValueError("SlidingWindow needs max_messages or max_tokens")
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.keep_first = keep_first

    async def messages(self, context: "Context", memory: Dict[str, Any]) -> List[MessageObject]:
        if self.max_tokens is None and not self.keep_first and self.max_messages <= 100:
            # The context holds the thread's latest 100 messages already.
            return list(context.messages[-self.max_messages:]) if self.max_messages else []
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


class _RunningSummary(SessionManager):
    """What Summarizing and Compacting share: fold the oldest messages of a
    session into its summary, with the summarizer told how long the summary
    may be, and the session left as it is when summarizing fails."""

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        summary_tokens: Optional[int],
        count_tokens: TokenCounter,
        max_message_tokens: Optional[int],
    ) -> None:
        super().__init__(count_tokens=count_tokens, max_message_tokens=max_message_tokens)
        self.summarizer = summarizer
        self.summary_tokens = summary_tokens

    def summary_size(self, session: Session) -> int:
        return self.count_tokens(session.summary) if session.summary else 0

    async def fold(self, session: Session, n: int) -> Session:
        """``session`` with its ``n`` oldest messages folded into its summary
        (the latest message always stays)."""
        n = min(n, len(session.messages) - 1)
        if n <= 0:
            return session
        folding = session.messages[:n]
        try:
            summary = await summarize(self.summarizer, session.summary, folding, max_tokens=self.summary_tokens)
        except Exception:
            # The run goes on with the session as it is; a later run compacts.
            logger.warning("Summarizing %d messages failed; the session is not compacted this run.", n, exc_info=True)
            return session
        if self.summary_tokens is not None and self.count_tokens(summary) > 2 * self.summary_tokens:
            logger.warning(
                "The summary is %d tokens, over twice the %d asked for.", self.count_tokens(summary), self.summary_tokens,
            )
        folded = session.folded + n
        return Session(
            summary=summary,
            messages=session.messages[n:],
            folded=folded,
            memory={"summary": summary, "folded_until": folding[-1].id, "folded": folded},
        )

    @staticmethod
    def resumed(messages: List[MessageObject], memory: Dict[str, Any]) -> Session:
        return Session(summary=memory.get("summary"), messages=list(messages), folded=memory.get("folded", 0))


class Summarizing(_RunningSummary):
    """Keeps the latest messages as they are, and a running summary of the
    rest: once the session holds more than ``max_messages`` messages, the
    oldest are folded into the summary until ``keep_messages`` remain - so
    the summarizer runs once in a while, not on every run. Each compaction
    folds only what is new, on top of the summary so far.

    For a budget in tokens, see ``Compacting``.

    :param summarizer: ``async (previous_summary, messages) -> summary``
        (see ``llm_summarizer``, ``openai_summarizer``).
    :param max_messages: compact past this many messages.
    :param keep_messages: messages left after compacting (default: half of
        ``max_messages``).
    :param summary_tokens: how long the summary may be; told to the
        summarizer (the built-in ones aim for it).
    """

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        max_messages: int = 50,
        keep_messages: Optional[int] = None,
        summary_tokens: Optional[int] = 400,
        count_tokens: TokenCounter = estimate_tokens,
        max_message_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(
            summarizer, summary_tokens=summary_tokens, count_tokens=count_tokens, max_message_tokens=max_message_tokens,
        )
        if max_messages < 1:
            raise ValueError("Summarizing needs max_messages >= 1")
        self.max_messages = max_messages
        self.keep_messages = keep_messages if keep_messages is not None else max(1, max_messages // 2)

    async def build(self, messages: List[MessageObject], memory: Dict[str, Any]) -> Session:
        session = self.resumed(messages, memory)
        if len(session.messages) <= self.max_messages:
            return session
        return await self.compact(session)

    async def compact(self, session: Session) -> Session:
        """Fold ``session``'s oldest messages into its summary now, down to
        ``keep_messages``."""
        return await self.fold(session, len(session.messages) - self.keep_messages)


class Compacting(_RunningSummary):
    """A token budget that fills up, is compacted, and fills up again.

    The session - its summary and its messages, plus ``reserve_tokens`` for
    what the agent adds (instructions, tools, the answer) - may take
    ``max_tokens``. While it stays under ``compact_at`` of that, nothing
    happens: the messages are all there, as they are. Once it reaches it,
    the session is compacted down to ``compact_to``: everything but the
    latest messages is folded into a new summary (of at most
    ``summary_tokens``, the previous summary included), and the session
    starts filling up again.

    :param summarizer: ``async (previous_summary, messages) -> summary``
        (see ``llm_summarizer``, ``openai_summarizer``).
    :param max_tokens: the session's budget - typically a share of the
        model's context window.
    :param compact_at: compact when the session reaches this share of the
        budget (default 0.9).
    :param compact_to: ... down to this share (default 0.25): the new summary
        and the latest messages that fit.
    :param summary_tokens: how long the summary may be (default: a tenth of
        the budget, and at most ``compact_to`` of it); told to the
        summarizer.
    :param reserve_tokens: taken off the budget for what is not in the
        session (system prompt, tools, the answer).
    """

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        max_tokens: int,
        compact_at: float = 0.9,
        compact_to: float = 0.25,
        summary_tokens: Optional[int] = None,
        reserve_tokens: int = 0,
        count_tokens: TokenCounter = estimate_tokens,
        max_message_tokens: Optional[int] = None,
    ) -> None:
        if not 0 < compact_to < compact_at <= 1:
            raise ValueError("Compacting needs 0 < compact_to < compact_at <= 1")
        budget = max_tokens - reserve_tokens
        if budget <= 0:
            raise ValueError("Compacting needs max_tokens above reserve_tokens")
        if summary_tokens is None:
            summary_tokens = max(1, min(int(budget * 0.1), int(budget * compact_to)))
        super().__init__(
            summarizer, summary_tokens=summary_tokens, count_tokens=count_tokens, max_message_tokens=max_message_tokens,
        )
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.compact_at = compact_at
        self.compact_to = compact_to

    @property
    def budget(self) -> int:
        """The tokens the session may take: ``max_tokens`` less the reserve."""
        return self.max_tokens - self.reserve_tokens

    def size(self, session: Session) -> int:
        """The tokens ``session`` takes: its summary and its messages."""
        return self.summary_size(session) + self.tokens(session.messages)

    def full(self, session: Session) -> bool:
        return self.size(session) >= self.compact_at * self.budget

    async def build(self, messages: List[MessageObject], memory: Dict[str, Any]) -> Session:
        session = self.resumed(messages, memory)
        if not self.full(session):
            return session
        return await self.compact(session)

    async def compact(self, session: Session) -> Session:
        """Compact ``session`` now: the latest messages that fit in
        ``compact_to`` of the budget (next to a summary of
        ``summary_tokens``) stay, the rest is folded into the summary."""
        room = int(self.compact_to * self.budget) - self.summary_tokens
        keep, used = 0, 0
        for m in reversed(session.messages):
            cost = self.count_tokens(m.text)
            if keep and used + cost > room:
                break
            keep += 1
            used += cost
        return await self.fold(session, len(session.messages) - keep)


async def summarize(
    summarizer: Summarizer,
    previous: Optional[str],
    messages: List[MessageObject],
    *,
    max_tokens: Optional[int] = None,
) -> str:
    """Call ``summarizer``, telling it ``max_tokens`` - the summary's target
    length - if it takes that keyword (the built-in ones do)."""
    takes = False
    try:
        params = inspect.signature(summarizer).parameters.values()
        takes = any(p.name == "max_tokens" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
    except (TypeError, ValueError):
        pass
    if takes and max_tokens is not None:
        return await summarizer(previous, messages, max_tokens=max_tokens)
    return await summarizer(previous, messages)


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
    messages to fold in; the summary is asked to be at most ``max_words``
    words, or the target length a session manager gives
    (``summary_tokens``).

        Summarizing(llm_summarizer(lambda prompt: my_llm(prompt)), max_messages=40)
    """
    def text(words: int) -> str:
        return instructions.format(max_words=words) if "{max_words}" in instructions else instructions

    async def summarize(previous: Optional[str], messages: List[MessageObject], max_tokens: Optional[int] = None) -> str:
        # A target length in tokens (from the session manager) wins: about
        # 3 words in 4 tokens.
        words = max(20, int(max_tokens * 0.75)) if max_tokens else max_words
        result = complete(summary_prompt(previous, messages, text(words)))
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


# ── The default ──────────────────────────────────────────────────────────

#: The session manager of an agent that sets none, on a server that sets none.
DEFAULT_SESSION_MANAGER: SessionManager = SlidingWindow(max_messages=20)


def session_manager_of(agent: Any) -> SessionManager:
    """The manager of ``agent``'s sessions: its own, else the default."""
    return getattr(agent, "session_manager", None) or DEFAULT_SESSION_MANAGER


def apply_default_session_manager(agents: Any, manager: Optional[SessionManager]) -> None:
    """Give ``manager`` to every agent of ``agents`` that has none of its own
    (a server's default); None leaves them to ``DEFAULT_SESSION_MANAGER``."""
    if manager is None:
        return
    for agent in agents or []:
        if getattr(agent, "session_manager", None) is None:
            agent.session_manager = manager


__all__ = [
    "DEFAULT_SESSION_MANAGER",
    "Session",
    "SessionManager",
    "FullHistory",
    "SlidingWindow",
    "Summarizing",
    "Compacting",
    "Summarizer",
    "summarize",
    "llm_summarizer",
    "openai_summarizer",
    "thread_messages",
    "estimate_tokens",
]
