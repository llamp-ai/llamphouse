"""Providers in the box: a rolling window (the default) and a reference summarizer."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

from llamphouse.sdk import Message, Sender, ThreadEvent

from .base import Revise, SessionInfo, SessionProvider


class RollingWindow(SessionProvider):
    """The default: the most recent `size` messages; no revisions."""

    def __init__(self, size: int = 50):
        self.size = size

    def view(self, session: SessionInfo) -> Sequence[Message]:
        return session.messages[-self.size:]


Summarize = Callable[[Sequence[Message]], Union[str, Awaitable[str]]]


class Summarizing(SessionProvider):
    """A reference provider: when the session holds more than `max_messages`, everything but the last
    `keep` becomes one summary. The summary comes from `summarize` - your function, usually a model
    call: llamphouse calls no model. A request may pass `keep=` to keep more or fewer."""

    def __init__(self, summarize: Summarize, *, max_messages: int = 50, keep: int = 10):
        self.summarize, self.max_messages, self.keep = summarize, max_messages, keep

    def _policy(self, session: SessionInfo) -> Revise | None:
        return Revise({"reason": "policy"}) if len(session.messages) > self.max_messages else None

    def on_event(self, session: SessionInfo, event: ThreadEvent) -> Revise | None:
        return self._policy(session)

    def on_entry(self, session: SessionInfo, entry: Message) -> Revise | None:
        return self._policy(session)

    async def revise(self, session: SessionInfo, options: Mapping[str, Any]) -> Sequence[Message] | None:
        split = max(len(session.messages) - int(options.get("keep", self.keep)), 0)
        older, recent = session.messages[:split], session.messages[split:]
        if not older:
            return None  # nothing to summarize
        text = self.summarize(older)
        if inspect.isawaitable(text):
            text = await text
        return [Message.create(Sender.system("session"), text), *recent]
