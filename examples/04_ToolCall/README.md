# Tool calls

A model tool loop. The agent calls OpenAI's Responses API with two function tools (`current_time`,
`calculate`); when the model asks for tools the agent runs them and calls the model again with the results,
until the model answers. Every model call and every tool call is a **step** (`@step`): done once and
recorded. If the worker dies mid-loop, the run is executed again and replays the recorded results - the model
isn't asked again and no tool runs twice - then carries on. The tool calls and results go into the agent's
session (`ctx.session.add`), not on the thread: the model sees them next turn, the conversation stays clean.

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
llamphouse up --memory --agents agents.yaml     # http://127.0.0.1:8000, Compass at /compass
python client.py                                # in a second terminal: chat; an empty line stops
```

```
you: What time is it in Tokyo, and how many minutes until midnight there?
  [current_time({"timezone": "Asia/Tokyo"}) -> Saturday 31 October 2026, 19:58:50 JST]
  [calculate({"expression": "(24*60) - (19*60 + 58)"}) -> 242]
assistant: It's 19:58 in Tokyo - 242 minutes until midnight.
```

The client prints the tools a run used from the agent's session (`GET /v2/runs/{id}/session`).

## Try

- *"What's 2**64 divided by 3?"* - one tool call. *"And in Lisbon?"* after a time question: the model reads
  its earlier tool calls from the session.
- A bad expression (*"calculate 2 + banana"*): the tool's error goes back to the model as its result.
- `config: {max_rounds: 1}` in `agents.yaml`: the run fails with `too_many_rounds` when the model needs a tool.

## In Compass

Open a run: its **Steps** tab lists `think#1`, `calculate#1`, `current_time#1`, `think#2`, … with their
recorded results - what a replay gives back instead of doing again - between the session reads and adds
(`session.messages#1`, `session.add#1`, …). Its **Session** tab shows the tool calls and results as the
agent's own entries; the thread shows only the conversation.

## The code

```python
class Assistant(Agent[ToolConfig]):
    async def run(self, ctx) -> None:
        for _ in range(ctx.config.max_rounds):
            said = await self.think(ctx, to_input(ctx, await ctx.session.messages()))
            if not said["calls"]:
                await ctx.send(said["text"])
                return
            for call in said["calls"]:
                part = ToolCallPart(call_id=call["call_id"], name=call["name"], arguments=call["arguments"])
                await ctx.session.add(part)
                output = await self.use(ctx, part)            # runs the tool: a step
                await ctx.session.add(ToolResultPart(call_id=part.call_id, output=[TextPart(text=output)]))
        raise AgentError("too_many_rounds", ...)

    @step
    async def think(self, ctx, items: list[dict]) -> dict:  # one model call, recorded
        response = await openai().responses.create(model=ctx.config.model, input=items, tools=TOOLS, ...)
        ...

    @step
    async def calculate(self, ctx, expression: str) -> str:
        ...
```

A step's result must be JSON: `think` returns the text and the calls as plain data, not the SDK's response
object. Replay goes by the order of the calls, so the model call is a step too: a model asked again could
answer differently, and the recorded tool steps would no longer match.
