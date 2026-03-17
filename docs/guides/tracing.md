# Tracing

LLAMPHouse includes **automatic OpenTelemetry (OTel) tracing** for all operations — runs, agent dispatches, data store calls, and streaming. Traces give you visibility into what your agents are doing, how long operations take, and where issues occur.

## Enabling tracing

Tracing is enabled by default. To send traces to a collector, set the environment variables:

```bash
LLAMPHOUSE_TRACING_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
OTEL_SERVICE_NAME=llamphouse
```

## What gets traced

LLAMPHouse automatically creates spans for:

| Span | Description |
|---|---|
| `llamphouse.run` | The full lifecycle of an agent run |
| `llamphouse.call_agent` | A `call_agent()` dispatch to another agent |
| `llamphouse.handover` | A `handover_to_agent()` dispatch |
| `llamphouse.stream` | Streaming operations |
| `llamphouse.data_store.*` | Data store read/write operations |

### Span attributes

Spans include rich attributes for debugging:

- `assistant.id` / `assistant.name` — the agent handling the run
- `dispatch.type` — `call_agent` or `handover`
- `dispatch.target_agent` / `dispatch.source_agent` — for inter-agent calls
- `dispatch.child_run` / `dispatch.child_thread` — child run tracking
- `gen_ai.system` — always `llamphouse`

## Viewing traces in Compass

The [Compass dashboard](compass.md) includes a built-in **trace viewer** powered by ClickHouse. To enable it:

1. Run ClickHouse (included in the Docker Compose setup)
2. Set the `CLICKHOUSE_URL` environment variable
3. Open Compass at `/compass` → navigate to the Traces tab

The trace viewer shows:

- **Span tree** — hierarchical view of all spans in a trace
- **Agent badges** — which agent produced each span
- **Timing** — duration and timeline for each operation
- **Attributes** — all span attributes for debugging

## Excluding spans

To reduce noise, you can exclude spans by pattern:

```python
app = LLAMPHouse(
    agents=[...],
    exclude_spans=["llamphouse.data_store.list_messages", "pattern.*"],
)
```

## Trace propagation

LLAMPHouse propagates trace context (`traceparent`) across agent boundaries. When one agent calls another via `call_agent()` or `handover_to_agent()`, the child run's spans are linked to the parent trace — giving you a single end-to-end view.

## Docker setup

The included Docker Compose configuration sets up the full tracing pipeline:

```yaml
# docker/docker-compose.yml includes:
# - OTel Collector (port 4318)
# - ClickHouse (port 8123)
```

```bash
cd docker
docker compose up -d
```

See [Deployment](../deployment.md) for the full Docker setup.

## Next steps

- [Compass Dashboard](compass.md) — view traces in the built-in UI
- [Deployment](../deployment.md) — Docker setup with OTel and ClickHouse
- [Configuration](../configuration.md) — tracing-related environment variables
