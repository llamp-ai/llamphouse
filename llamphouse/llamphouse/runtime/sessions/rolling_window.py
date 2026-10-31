"""The default session provider: the most recent messages, never revised."""

from __future__ import annotations

from typing import Sequence

from llamphouse.sdk import Message

from .base import SessionInfo, SessionProvider


class RollingWindow(SessionProvider):
    """The default: the most recent `size` messages; no revisions."""

    def __init__(self, size: int = 50):
        self.size = size

    def view(self, session: SessionInfo) -> Sequence[Message]:
        return session.messages[-self.size:]
