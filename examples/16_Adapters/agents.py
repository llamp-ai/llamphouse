"""One agent, served over every protocol (see llamphouse.yaml). It knows nothing about them: it asks, keeps
thread state, calls the client's tool when there is one, and streams its reply. No model.
"""

from llamphouse.sdk import Agent, ToolCallPart


class Concierge(Agent):
    """Asks your name once per conversation, then answers - with the local time when your client offers it."""

    async def run(self, ctx) -> None:
        name = await ctx.state.get("name")
        if name is None:
            answer = await ctx.ask("Hello! What's your name?")  # a question, in each protocol's own form
            name = answer.text.strip()
            await ctx.state.set("name", name)  # thread state: the next turns know it
        turn = await ctx.state.update("turns", lambda n: n + 1, 0)

        time = ""
        if any(t.name == "local_time" for t in ctx.client.tools):  # tools the client declared
            result = await ctx.client.call_tool(ToolCallPart(call_id="local_time", name="local_time", arguments={}))
            time = f" It's {result.output[0].text} where you are."

        async with ctx.stream() as out:  # streamed: deltas over SSE, wherever the protocol has them
            await out.write(f"{name}, you said: {ctx.input.text!r}. ")
            await out.write(f"Turn {turn} of this conversation.{time}")
