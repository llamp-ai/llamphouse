"""A notes agent that tells who called it - the caller of the run, as the deployment's auth found it."""

from __future__ import annotations

from llamphouse.sdk import Agent


class NotesAgent(Agent):
    """Keeps notes for a team; says who it is talking to."""

    async def run(self, ctx) -> str:
        caller = ctx.run.caller  # who started the run: subject, tenant, metadata
        notes = await ctx.state.get("notes", [])
        notes = [*notes, ctx.input.text]
        await ctx.state.set("notes", notes)  # the thread's state: only callers of its tenant can read it
        reply = f"Noted for {caller.subject} ({caller.tenant or 'no tenant'}): {len(notes)} note(s) on this thread."
        await ctx.send(reply)
        return reply
