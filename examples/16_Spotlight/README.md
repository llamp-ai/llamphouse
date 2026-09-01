# Spotlight custom-route example

This example keeps Llamphouse core unchanged. It mounts a small read-only
Spotlight-compatible API directly on the example's FastAPI application and
reuses the existing `data_store` and `tracing_store` objects.

## Routes

The server exposes:

```text
GET /spotlight/agents
GET /spotlight/agents/{external_agent_ref}/conversations
GET /spotlight/agents/{external_agent_ref}/conversations/{conversation_id}/messages
GET /spotlight/agents/{external_agent_ref}/executions
GET /spotlight/agents/{external_agent_ref}/runs
GET /spotlight/agents/{external_agent_ref}/runs/{run_id}
GET /spotlight/agents/{external_agent_ref}/runs/{run_id}/steps
GET /spotlight/agents/{external_agent_ref}/traces
GET /spotlight/agents/{external_agent_ref}/traces/{trace_id}
```

List responses use `{ "data": [...], "next_cursor": null }`. The trace
detail response uses `{ "data": {...} }`. All responses contain bounded
projections only. Message content, metadata, raw OTel attributes/events and
backend objects are not exposed.

Spotlight routes do not require an additional credential. The
`external_agent_ref` path value is the exact registered `Agent.id` exposed by
the example's agent configuration/Agent Card.

## Run

Install the example dependencies and set `OPENAI_API_KEY`, then run:

```bash
uv run server.py
```

In another terminal, send an A2A request:

```bash
uv run client.py
```

To exercise the Spotlight error responses:

```bash
uv run client_error.py
```

The example connects directly to the routes above; there is no separate
Spotlight client process. The A2A card and A2A JSON-RPC routes remain
provided by the existing `A2AAdapter`; observability is only the additional
custom route mounted by `server.py`.
