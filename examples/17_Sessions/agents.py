"""Sessions example: agents that show the session they work with.

Each run, ``context.session`` is the agent's view of the thread. Three
agents, three session managers, on the same kind of conversation:

- ``summarizing-agent`` - ``Summarizing``: past 6 messages the oldest are
  folded into a running summary, down to the latest 3.
- ``window-agent`` - ``SlidingWindow``: the latest 4 messages, plus the
  first one (the request the conversation started with).
- ``compacting-agent`` - ``Compacting``: a budget of 120 tokens that fills
  up; at 90% it is compacted to 25% (a new summary and the latest
  messages), and fills up again.
- ``facts-agent`` - a manager of its own (``KeyFacts``): the latest 2
  messages, and every number the user mentioned, remembered across runs.
"""

import os
import re

from llamphouse.core import Agent, Compacting, Context, Session, SessionManager, SlidingWindow, Summarizing, openai_summarizer
from llamphouse.core.sessions import thread_messages


async def first_words(previous: str | None, messages, max_tokens: int | None = None) -> str:
    """A stand-in summarizer without an LLM: the first words of each user
    message - enough to see compaction happen - cut to ``max_tokens``
    (about 4 characters a token), the length the session manager asks for."""
    said = "; ".join(" ".join(m.text.split()[:4]) for m in messages if m.role == "user")
    summary = f"{previous}; {said}" if previous else said
    if max_tokens and len(summary) > max_tokens * 4:  # keep the latest, from a word on
        summary = summary[-max_tokens * 4:].split(" ", 1)[-1]
    return summary


# With OPENAI_API_KEY set, a real LLM summary.
summarizer = openai_summarizer(os.getenv("OPENAI_MODEL", "gpt-4o-mini")) if os.getenv("OPENAI_API_KEY") else first_words


class KeyFacts(SessionManager):
    """A session manager of your own: implement ``build``.

    It keeps the latest 2 messages in view, and remembers every number the
    user mentioned (a budget, a date, ...) in its memory, so they stay in
    the summary long after the messages left the view.
    """

    async def messages(self, context, memory):
        # Only what came after what was read last time.
        return await thread_messages(context, after=memory.get("folded_until"))

    async def build(self, messages, memory):
        facts = list(memory.get("facts", []))
        facts += [n for m in messages if m.role == "user" for n in re.findall(r"\d[\d.,]*", m.text)]
        summary = f"Numbers the user mentioned: {', '.join(facts)}" if facts else None
        if not messages:
            return Session(summary=summary)
        # Remember the facts, and continue after the last message read.
        return Session(
            summary=summary,
            messages=messages[-2:],
            memory={"facts": facts, "folded_until": messages[-1].id},
        )


def session_tokens(session: Session) -> int:
    text = (session.summary or "") + "".join(m.text for m in session.messages)
    return (len(text) + 3) // 4


class SessionAgent(Agent):
    """Replies with the session it sees. In a real agent:
    ``llm(messages=context.session.to_messages())``."""

    def __init__(self, id: str, session_manager: SessionManager):
        super().__init__(id=id)
        self.session_manager = session_manager

    async def run(self, context: Context):
        session = context.session
        # In a real agent: llm(messages=session.to_messages()).
        tokens = session_tokens(session)
        lines = [f"I see {len(session.messages)} messages, ~{tokens} tokens ({session.folded} folded into the summary)."]
        if session.summary:
            lines.append(f"Summary: {session.summary}")
        await context.reply("\n".join(lines))


def summarizing_agent(_cfg=None):
    return SessionAgent("summarizing-agent", Summarizing(summarizer, max_messages=6, keep_messages=3))


def window_agent(_cfg=None):
    return SessionAgent("window-agent", SlidingWindow(max_messages=4, keep_first=1))


def compacting_agent(_cfg=None):
    return SessionAgent("compacting-agent", Compacting(summarizer, max_tokens=120))


def facts_agent(_cfg=None):
    return SessionAgent("facts-agent", KeyFacts())
