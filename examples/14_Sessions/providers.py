"""The session side: a summarizer for the built-in `Summarizing` provider, and a provider of our own.

Session providers run in the runtime, next to the agents: they decide what an agent sees of its thread
(`view`), and when and how to revise it (in the background, never on a read). llamphouse calls no model:
a summary is your function's.
"""

from __future__ import annotations

import os
import re
from typing import Sequence

from llamphouse.runtime.sessions import SessionInfo, SessionProvider
from llamphouse.sdk import Message, Sender, SenderKind


async def summarize(older: Sequence[Message]) -> str:
    """For `Summarizing`: the older messages as one text. With OPENAI_API_KEY set a model writes it;
    without, a stand-in keeps the first words of what the person said."""
    said = [m.text for m in older if m.sender.kind == SenderKind.PERSON and m.text]
    if os.environ.get("OPENAI_API_KEY"):
        from openai import AsyncOpenAI

        response = await AsyncOpenAI().responses.create(
            model=os.environ.get("SUMMARY_MODEL", "gpt-5-mini"),  # set in agents.yaml (`env:`)
            instructions="Summarize what the user told you, in one or two sentences. Keep every number.",
            input="\n".join(said))
        return f"Summary: {response.output_text}"
    return "Summary: " + "; ".join(" ".join(text.split()[:5]) for text in said)


class Facts(SessionProvider):
    """A provider of our own: the numbers the person mentioned, as one note, then the latest `keep`
    messages. Only `view` - cheap, no revisions. Override `on_event`, `on_entry` or `after_run` to
    return `Revise(...)`, and `revise` to rewrite the session in the background."""

    def __init__(self, keep: int = 2):
        self.keep = keep

    def view(self, session: SessionInfo) -> Sequence[Message]:
        facts = [n for m in session.messages if m.sender.kind == SenderKind.PERSON
                 for n in re.findall(r"\d[\d,.]*", m.text or "")]
        note = Message.create(Sender.system("facts"), "Numbers mentioned: " + (", ".join(facts) or "none"))
        return [note, *session.messages[-self.keep:]]
