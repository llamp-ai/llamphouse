# LangGraph

`llamphouse.helpers.langgraph` runs [LangGraph](https://langchain-ai.github.io/langgraph/) graphs on llamphouse:
durably, across waits and crashes, with a node's `interrupt()` asked through the run, and the graph's messages
streamed into the run.

```bash
pip install 'llamphouse[langgraph]'
```

It has three parts:

| | |
|---|---|
| `LangGraphAgent` | a graph as the whole agent |
| `await run_graph(ctx, graph, input)` | a graph as part of any agent's run |
| `LlamphouseCheckpointer(ctx)` | a LangGraph checkpointer that keeps checkpoints in the run's state |

A node's `interrupt()` needs Python 3.11 or later (a LangGraph limitation for async runs).

## A graph as the agent

```python
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from llamphouse.helpers.langgraph import LangGraphAgent


class Trip(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    city: str
    month: str


def research(state: Trip) -> Trip:
    return {"city": state["messages"][-1].content.split()[-1].strip("?.!")}


def ask_month(state: Trip) -> Trip:
    return {"month": interrupt({"prompt": f"Which month for {state['city']}?"})}


def book(state: Trip) -> Trip:
    return {"messages": [AIMessage(content=f"Booked {state['city']} in {state['month']}.")]}


builder = StateGraph(Trip)
builder.add_node("research", research)
builder.add_node("ask_month", ask_month)
builder.add_node("book", book)
builder.add_edge(START, "research")
builder.add_edge("research", "ask_month")
builder.add_edge("ask_month", "book")
builder.add_edge("book", END)
graph = builder.compile()  # no checkpointer: each run gets its own


class Planner(LangGraphAgent):
    """Plans trips."""

    graph = graph
```

```yaml
# agents.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: planner}
spec:
  entrypoint: agents.py:Planner
```

Or without a class statement: `Planner = LangGraphAgent.of(graph, name="Planner", description="Plans trips.")`,
still at module level so the agent file can import it.

What happens on "A trip to Rome":

1. The graph starts with the run's session as `{"messages": [...]}` (the agent's messages as `assistant`, the
   rest as `user`).
2. `ask_month` calls `interrupt(...)`. The run asks "Which month for Rome?" with `ctx.ask` and waits
   (`input_required`), holding no worker.
3. The answer arrives (`POST /v2/runs/{id}/answer`, an adapter, Compass). The run executes again; the graph
   resumes in `ask_month` from its checkpoint, with the answer as the value of `interrupt()`. `research` does
   not run again.
4. What the graph says (its model's tokens, the messages its nodes return) is streamed into the run's message:
   "Booked Rome in June."

### Options

| Class attribute | Default | |
|---|---|---|
| `graph` | | the compiled graph (required) |
| `scope` | `"run"` | `"thread"` keeps the graph's state from one run to the next on the same thread |
| `stream` | `True` | stream what the graph says into the run's messages; with `False`, the agent sends `reply(values)` at the end |

Override `graph_input(self, ctx)` to build the graph's input differently, and `reply(self, values)` for what it
says at the end when it doesn't stream (by default the graph's `output` if it is a string, else the content of
its last message).

## A graph inside an agent

`run_graph` runs a graph as part of any run, with work of your own around it. It returns the graph's final
values.

```python
from llamphouse.helpers.langgraph import run_graph
from llamphouse.sdk import Agent


class Desk(Agent):
    """Plans a trip with a graph, then does more work of its own."""

    async def run(self, ctx) -> dict:
        values = await run_graph(ctx, graph, {"messages": [("user", ctx.input.text)]}, stream=False)
        await ctx.send(f"Planned: {values['city']} in {values['month']}.")
        return {"city": values["city"], "month": values["month"]}
```

`run_graph(ctx, graph, input, *, scope="run", stream=True, key="langgraph", configurable=None)`:

- **The first time**, the graph starts with `input`. When the run executes again (it waited for an answer, or
  another worker took it over after a crash), the graph carries on from its last checkpoint and never gets its
  input twice. Nodes that finished don't run again; a node that was running when the worker died runs again.
- **`interrupt(value)`** is asked with `ctx.ask`. `value` can be a string, or `{"prompt": ..., "schema": ...}`
  (a JSON Schema for the answer). The answer's data resumes the node if it has any, else its text. Several
  interrupts at once are asked one after another.
- **`stream=True`** streams the graph's messages (`stream_mode="messages"`) into the run's messages as they come.
- **`scope="thread"`** keeps the graph's state on the thread, so the next run on the thread continues from it
  (LangGraph memory). It is written once per run, as one `state.changed` event.
- **`configurable`** adds values to the graph's `configurable` config, next to `thread_id`.

A graph compiled with a checkpointer of your own keeps it; compile without one and `run_graph` gives each run a
`LlamphouseCheckpointer`.

## The checkpointer

`LlamphouseCheckpointer(ctx, scope="run", key="langgraph")` is LangGraph's in-memory saver, kept in the run's
state (`ctx.run.state`) after every write. It reads and writes the state as it is now, outside the run's replay
(`ctx.run.state.live`), so it always sees the latest checkpoint.

```python
from llamphouse.helpers.langgraph import LlamphouseCheckpointer

saver = LlamphouseCheckpointer(ctx, scope="thread")
compiled = builder.compile(checkpointer=saver)
result = await compiled.ainvoke(state, saver.config())   # config(): {"configurable": {"thread_id": ...}}
await saver.keep()                                         # scope="thread": onto the thread, for its next run
```

It keeps the latest checkpoint of each namespace, not the whole history. A graph with delta channels keeps their
chain until the channels snapshot. The graph's thread id is the run's id (`scope="run"`) or the llamphouse
thread's (`scope="thread"`).

Using the checkpointer directly gives you the storage only; `run_graph` adds resuming, interrupts and
streaming on top.

## Testing

`run_graph` works under `llamphouse.sdk.testing.FakeRun`, with the interrupts' answers scripted:

```python
from llamphouse.sdk.testing import FakeRun


async def test_desk():
    run = FakeRun("a trip to Rome", answers=["June"])
    output = await run.start(Desk())
    assert output == {"city": "Rome", "month": "June"}
    assert [r.reason for r in run.input_requests] == ["Which month for Rome?"]
```

## See also

- [Example 15_LangGraph](../../examples/15_LangGraph)
- [Durable runs](../concepts/durable-runs.md): why a run executes again, and what replays
