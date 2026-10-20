"""Sessions example: an agent that shows the session it works with.

Each run, ``context.session`` is the agent's view of the thread: a summary
of the older messages and the recent ones as they are. Past 6 messages the
session manager folds the oldest into the summary, down to the latest 3.
"""

import os

from llamphouse.core import Agent, Context, SessionManager


async def summarize(previous: str | None, messages) -> str:
    """Fold ``messages`` into the summary so far.

    With OPENAI_API_KEY set this asks an LLM; without one it keeps the first
    words of each message - enough to see compaction happen.
    """
    if os.getenv("OPENAI_API_KEY"):
        from openai import AsyncOpenAI

        transcript = "\n".join(f"{m.role}: {m.text}" for m in messages)
        response = await AsyncOpenAI().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Update the conversation summary with the new messages. Be brief."},
                {"role": "user", "content": f"Summary so far: {previous or '(none)'}\n\nNew messages:\n{transcript}"},
            ],
        )
        return response.choices[0].message.content
    said = "; ".join(" ".join(m.text.split()[:4]) for m in messages if m.role == "user")
    return f"{previous}; {said}" if previous else said


class SessionAgent(Agent):
    session_manager = SessionManager(summarizer=summarize, max_messages=6, keep_messages=3)

    async def run(self, context: Context):
        session = context.session
        # In a real agent: llm(messages=session.to_messages()).
        lines = [f"I see {len(session.messages)} recent messages ({session.folded} folded into the summary)."]
        if session.summary:
            lines.append(f"Summary: {session.summary}")
        await context.reply("\n".join(lines))
