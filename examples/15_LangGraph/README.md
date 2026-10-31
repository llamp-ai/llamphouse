# LangGraph

A LangGraph graph served by llamphouse, with `llamphouse.helpers.langgraph`. The graph is a small support desk:
it routes a message (order status, refund, anything else), refunds up to 100 EUR at once, and asks a person to
approve larger refunds with LangGraph's `interrupt()`. It runs twice: as the whole agent (`LangGraphAgent`,
`support`), and inside an agent of its own (`run_graph`, `refunds`) that does work around it. Plain nodes, no
model, no keys.

## Run it

Needs Python 3.11+ (LangGraph's `interrupt()` in async runs).

```bash
pip install -r requirements.txt                 # llamphouse[langgraph]
llamphouse up --memory --agents agents.yaml     # http://127.0.0.1:8000/compass
python client.py                                # approves the large refund; `python client.py no` declines it
```

```
== support: the graph is the agent; its state carries over the thread's runs
ada: Where is my order?
support: Your order shipped yesterday; it arrives on Friday.
ada: I'd like a refund of 40 EUR
support: Refunded 40.00 EUR. 1 refund(s) so far, 40.00 EUR in all.
ada: And a refund of 250 EUR for the chair
support asks: Approve a refund of 250.00 EUR? (yes/no)
ada: yes
support: Refunded 250.00 EUR. 2 refund(s) so far, 290.00 EUR in all.

== refunds: the graph runs inside an agent of its own
ada: Please refund 300 EUR for order A-17
refunds asks: Approve a refund of 300.00 EUR? (yes/no)
ada: yes
refunds: A-17 (Oak chair): Refunded 300.00 EUR. 1 refund(s) so far, 300.00 EUR in all.
[run completed, output {'order': 'A-17', 'item': 'Oak chair', 'amount': 300.0, 'refunded': True}]
```

## What happens

- **`interrupt()` is `ctx.ask`.** The run waits (`input_required`) without holding a worker; the answer
  (`POST /v2/runs/{id}/answer`, or Compass's composer) resumes the graph in that node. Any protocol's way of
  answering works: an A2A `input-required` task, a Responses follow-up, an AG-UI `resume`.
- **Checkpoints in the run's state.** When the run executes again - it waited, or a worker died and another
  took it over - the graph carries on from its last checkpoint: nodes that finished don't run again, and its
  input isn't given twice. The latest checkpoint is kept, not the history.
- **Thread scope is memory.** `support` has `scope = "thread"`: the graph's state (its messages, the
  `refunds` list) carries over to the thread's next run, so the third message knows about the first refund.
  It is written to the thread's state (`langgraph`) when the graph ends or asks, not on every step.
- **Streaming.** Messages the graph's nodes produce - a model's tokens, or an `AIMessage` a node returns - are
  streamed into the run's messages. `refunds` passes `stream=False` and says what it wants itself.

In Compass: a `support` thread's **State** shows the `langgraph` key, written when the graph ends or asks; a run that asked
shows the question and the answer in its story; `refunds` runs show the recorded `lookup` step (replayed, not
done again, when the run resumes after the answer) and their typed output.

## The code

The graph as the whole agent:

```python
class SupportDesk(LangGraphAgent):
    """Order status and refunds; a person approves refunds above 100 EUR."""
    graph = GRAPH            # build().compile(), without a checkpointer: llamphouse gives each run its own
    scope = "thread"
```

or `SupportDesk = LangGraphAgent.of(GRAPH, scope="thread")`. Override `graph_input(ctx)` to map the run to
the graph's input, and `reply(values)` for what it says at the end.

The graph inside an agent:

```python
class Refunds(Agent):
    async def run(self, ctx) -> RefundResult:
        item = await self.lookup(ctx, order)                     # a @step: recorded once
        values = await run_graph(ctx, GRAPH, {"messages": [("user", ctx.input.text)]}, stream=False)
        await ctx.send(f"{order} ({item}): {values['messages'][-1].content}")
        return RefundResult(...)
```

The node that asks:

```python
def approve(state: Support) -> Support:
    answer = interrupt({"prompt": f"Approve a refund of {state['amount']:.2f} EUR? (yes/no)"})
    return {"approved": str(answer).strip().lower() in ("yes", "y", "approve")}
```

`interrupt(value)` takes a string, or `{"prompt": ..., "schema": ...}` (the schema goes to the client with the
question); the answer's data, else its text, is what `interrupt()` returns. For your own checkpointer setup,
`LlamphouseCheckpointer(ctx, scope="run" | "thread")` is a LangGraph checkpointer on its own.
