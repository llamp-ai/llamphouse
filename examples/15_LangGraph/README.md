# LangGraph Branching Wrapper Example

This example shows a non-trivial LangGraph workflow running behind the LLAMPHouse runtime contract.

## What it demonstrates

- `LangGraphAgent` wrapper usage
- Intent routing with conditional edges
- Branch fan-out/fan-in (parallel branches that join)
- LLAMPHouse step mapping for LangGraph nodes (visible in Compass)
- Tracing enabled for this example (run-level traces visible in Compass)
- A2A serving via `A2AAdapter`
- Human in the loop with LangGraph's `interrupt()`: the run parks without holding a worker, the A2A task is `input-required`, and the answer resumes the node
- Durable runs: checkpoints in the LLAMPHouse data store, so a resumed or retried run carries on from the last checkpoint

## Graph shape

1. `ingest_user_input` extracts latest user text from message parts.
2. `route_intent` classifies request into `math`, `research`, `refund`, or `general`.
3. Conditional branch:
	- `math` -> `solve_math` -> `format_math_output`
	- `research` -> `make_research_plan` -> (`collect_facts` + `collect_risks`) -> `synthesize_research`
	- `general` -> `general_reply`
	- `refund` -> `draft_refund` -> `human_approval` (`interrupt()`) -> `finalize_refund`
4. Terminal nodes produce final `output`.

Because the wrapper runs with `map_nodes_to_steps=True`, node execution is mapped into LLAMPHouse run steps.

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the server:

```bash
llamphouse up
```

3. Run the client:

```bash
python client.py
```

The client sends four prompts to exercise each branch. The refund prompt
stops at `human_approval`: the task comes back `input-required` with the
question, and the client answers it with a message carrying the task id
(`{"approved": true}` as a data part), which resumes the graph.

## Durable runs and `interrupt()`

`LangGraphAgent` runs the graph with a checkpointer (LangGraph's own
persistence), keyed by the run and stored in the LLAMPHouse data store:

```python
def human_approval(state):
    answer = interrupt({"prompt": "Approve this refund?", "schema": {...}})
    return {"refund_approved": answer["approved"]}
```

- `interrupt(value)` parks the run (`requires_action`), holding no worker.
  `{"prompt", "schema"}` is shown as the question; over A2A the task is
  `input-required`, over the Assistants API you answer with
  `POST /threads/{thread_id}/runs/{run_id}/signals` (`name: "human_input"`).
- The answer resumes the graph *inside* that node (`Command(resume=...)`);
  the nodes before it do not run again. LangGraph runs the interrupted node
  again from its top, so keep side effects after `interrupt()` or in a
  later node.
- A run that fails, or whose worker dies, carries on from the last
  checkpoint (the agent is `durable` by default).
- `LangGraphAgent(checkpointer=None)` turns this off; pass your own
  LangGraph checkpointer (e.g. `PostgresSaver`) to use it instead, and
  `checkpoint_scope="thread"` to keep the graph state across runs of a thread.

## Inspect in Compass

The server mounts `CompassAdapter`, so after running requests you can inspect node-level steps in Compass.

Tracing is force-enabled by the server code for this example and uses an in-memory tracing store.
Open Compass and inspect:

- Traces list for run-level span trees
- Run detail -> Trace tab for run-specific traces

## YAML-based startup

This example now includes `llamphouse.yaml` and runs config-first.

- agent definition entrypoint: `server.py:create_agent`
- deployment: `langgraph-agent`
- adapters: `a2a`, `compass`
- stores: `in_memory` data + `in_memory` tracing
- tracing/env metadata: configured in `globals.env`

If you still want the original direct Python startup path, `python server.py` remains supported.
