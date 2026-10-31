"""The simplest llamphouse agent: no model, no keys. It greets whoever talks to it and says back what they said."""

from llamphouse.sdk import Agent


class Hello(Agent):
    """Greets you and echoes what you said."""

    async def run(self, ctx) -> None:
        # ctx.input: the message that started this run. ctx.send: a message on this run's thread.
        name = ctx.input.sender.name or ctx.input.sender.id
        await ctx.send(f"Hello {name}! You said: {ctx.input.text}")
