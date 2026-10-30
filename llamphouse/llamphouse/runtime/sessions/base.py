"""The session provider: who shapes an agent's working memory on a thread.

llamphouse keeps the session - its messages and entries, its versions, its
log - and runs the provider's hooks at the right moments. The provider
decides: how the session shows (`view`), when to revise it (the policy hooks,
cheap and inline), and how (`revise`, run as a job in the background, never
on a read). Mechanism here, policy there: a summary is the provider's own
code, often a model call of the agent's - llamphouse calls no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from llamphouse.sdk import Message, RunResult, ThreadEvent


@dataclass(frozen=True)
class SessionInfo:
    """A session as a provider sees it: read-only."""

    id: str
    thread_id: str
    agent: str
    messages: tuple[Message, ...]  # the thread's messages and the agent's entries, in order
    version: int
    revisions: int  # so far
    revising: bool  # a revision is in flight


@dataclass(frozen=True)
class Revise:
    """A policy hook's answer: revise now, with these options (passed to `revise`)."""

    options: Mapping[str, Any] = field(default_factory=dict)


class SessionProvider:
    """Override what you need. The defaults: show everything, never revise, decline requests.

    The policy hooks are called inline, as things happen - keep them cheap: they only decide. A
    revision they ask for runs later, as a job: one at a time per session, on a snapshot; what is
    added meanwhile survives. A request from agent code (`ctx.session.revise(**options)`) goes
    straight to `revise`, with `reason="requested"` added to its options."""

    def view(self, session: SessionInfo) -> Sequence[Message]:
        """What `ctx.session.messages()` shows. Cheap: no model calls."""
        return session.messages

    def on_event(self, session: SessionInfo, event: ThreadEvent) -> Revise | None:
        """A message on the thread came into the session."""
        return None

    def on_entry(self, session: SessionInfo, entry: Message) -> Revise | None:
        """The agent added an entry (`ctx.session.add`)."""
        return None

    def after_run(self, session: SessionInfo, run: RunResult) -> Revise | None:
        """A run of the agent on this thread ended."""
        return None

    async def revise(self, session: SessionInfo, options: Mapping[str, Any]) -> Sequence[Message] | None:
        """The revision itself, as a job: the session's new messages, or None to decline (nothing to
        do). Raising leaves the session as it was."""
        return None
