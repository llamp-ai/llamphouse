# Spotlight custom-route example

This example keeps Llamphouse core unchanged. It mounts a small read-only
Spotlight-compatible API directly on the example's FastAPI application and
reuses the existing `data_store` and `tracing_store` objects.

## Routes

The server exposes:

```text
GET /spotlight/agents
GET /spotlight/agents/{runtime_agent_id}/runs
GET /spotlight/agents/{runtime_agent_id}/runs/{run_id}
GET /spotlight/agents/{runtime_agent_id}/threads
GET /spotlight/agents/{runtime_agent_id}/threads/{thread_id}/messages
GET /spotlight/agents/{runtime_agent_id}/traces
GET /spotlight/agents/{runtime_agent_id}/traces/{run_id}
```

The responses contain bounded projections only. Message content, metadata,
usage, raw OTel attributes/events and backend objects are not exposed.

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

Spotlight connects directly to the routes above; there is no separate
Spotlight client process. The A2A card and A2A JSON-RPC routes remain provided by the existing
`A2AAdapter`; Spotlight is only the additional custom route mounted by
`server.py`.
