"""One agent class, three instances, three session providers (see agents.yaml).

The agent hands no model anything: it says what its session holds - the working memory it would give a
model this turn. The thread keeps every message either way; the session provider decides what the agent
sees of it.
"""

from llamphouse.sdk import Agent, SenderKind


def short(text: str | None, width: int = 48) -> str:
    line = (text or "").splitlines()[0] if text else ""
    return line if len(line) <= width else line[:width - 1] + "…"


class Mirror(Agent):
    """Says what its session holds this turn."""

    async def run(self, ctx) -> None:
        session = await ctx.session.messages()  # what a model would get: the provider's view
        lines = [f"I see {len(session)} messages:"]
        for m in session:
            if m.sender.kind == SenderKind.SYSTEM:
                who = m.sender.id  # a provider's own message: a summary, notes
            elif ctx.session.is_own(m):
                who = "me"  # the model's assistant role
            else:
                who = m.sender.name or m.sender.id
            lines.append(f"  {who}: {short(m.text, 100 if m.sender.kind == SenderKind.SYSTEM else 48)}")
        await ctx.send("\n".join(lines))
