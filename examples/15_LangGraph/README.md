# LangGraph Branching Wrapper Example

This example shows a non-trivial LangGraph workflow running behind the LLAMPHouse runtime contract.

## What it demonstrates

- `LangGraphAgent` wrapper usage
- Intent routing with conditional edges
- Branch fan-out/fan-in (parallel branches that join)
- LLAMPHouse step mapping for LangGraph nodes (visible in Compass)
- Tracing enabled for this example (run-level traces visible in Compass)
- A2A serving via `A2AAdapter`

## Graph shape

1. `ingest_user_input` extracts latest user text from message parts.
2. `route_intent` classifies request into `math`, `research`, or `general`.
3. Conditional branch:
	- `math` -> `solve_math` -> `format_math_output`
	- `research` -> `make_research_plan` -> (`collect_facts` + `collect_risks`) -> `synthesize_research`
	- `general` -> `general_reply`
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

The client sends three prompts to exercise each branch.

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
