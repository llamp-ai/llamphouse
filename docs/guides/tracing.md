# Tracing

llamphouse traces runs with [OpenTelemetry](https://opentelemetry.io/). Each execution of a run is a span, each
step it executes is a span under it, and spans your own code makes nest under the run's span. Spans go to a
**tracing store**, from which the API and Compass read a run's trace back, and optionally to any OpenTelemetry
collector.

```bash
pip install 'llamphouse[tracing]'
```

Without the extra, or without a tracing store and a tracer provider, nothing is imported and no span is made.
Tracing never fails a run: an error in tracing is logged and the run carries on; spans are exported in the
background.

## What is traced

| Span | When | Attributes |
|---|---|---|
| `llamphouse.run` | each execution of a run, as a root span, current while `run(ctx)` executes | `run.id`, `llamphouse.run_id`, `thread.id`, `llamphouse.thread_id`, `llamphouse.agent`, `llamphouse.parent_run`, `llamphouse.run.activation`, `llamphouse.engine`; at the end `llamphouse.run.status`, `llamphouse.run.error`, `llamphouse.run.replayed_steps` |
| `llamphouse.step` | each `ctx.step` (or `@step`) executed, under the span current at the time | `llamphouse.step.name`, `llamphouse.step.key`, `run.id` |
| `llamphouse.session.revise` | each session revision (a session provider's summary, say), as a root span | `llamphouse.session.id`, `thread.id`, `llamphouse.agent`, `llamphouse.session.version`, `llamphouse.session.reason`, `llamphouse.session.messages`, `llamphouse.session.revision` |

A run executes more than once when it waits (an answer, the clock, a signal, another thread) or is taken over
after its worker died. Each execution is a new `llamphouse.run` span, in a new trace, with the same `run.id`.
A step that is replayed from the record makes no span: it was traced when it executed.

## Your own spans

Use any OpenTelemetry tracer in agent code. While the agent runs, the run's span is the current span, so yours
nest under it:

```python
from opentelemetry import trace

from llamphouse.sdk import Agent

tracer = trace.get_tracer("my_agents")


class Researcher(Agent):
    """Looks something up, traced."""

    async def run(self, ctx) -> None:
        with tracer.start_as_current_span("search") as span:
            span.set_attribute("input.value", ctx.input.text)
            found = await ctx.step("fetch", fetch, ctx.input.text)   # a llamphouse.step span, under "search"
            span.set_attribute("output.value", found)
        await ctx.send(found)


async def fetch(query: str) -> str:
    return f"3 results for {query}"
```

Instrumentation libraries work the same way: spans from an instrumented OpenAI or HTTP client called inside the
run end up in its trace. Compass shows `gen_ai.input.messages`, `gen_ai.output.messages`, `input.value` and
`output.value` readable and foldable.

## Tracing stores

A tracing store receives spans (through an exporter added to the tracer provider) and reads them back by run.

| Name | Class | Options | |
|---|---|---|---|
| `in_memory` | `InMemoryTracingStore` | none | this process's spans; gone on restart |
| `postgres` | `PostgresTracingStore` | `database_url` | the `llamphouse_traces` table (psycopg2, in the `postgres` extra) |
| `clickhouse` | `ClickHouseTracingStore` | `clickhouse_url` | reads what an OpenTelemetry Collector writes to ClickHouse (`otel.otel_traces`); llamphouse exports nothing to it itself |
| `module:Class` | your own `BaseTracingStore` | yours | |

In the runtime file:

```yaml
tracing: {postgres: {database_url: "${LLAMPHOUSE_DATABASE_URL}"}}
```

Without `tracing` in the runtime file, `llamphouse up` takes the store from the environment, as 1.x did:

| Variable | |
|---|---|
| `TRACING_STORE` | `memory`, `postgres` (uses `DATABASE_URL`) or `clickhouse` (uses `CLICKHOUSE_URL`). Not set: ClickHouse when `CLICKHOUSE_URL` is set, else in memory. |
| `LLAMPHOUSE_TRACING_ENABLED` | `false` turns tracing off |

In code: `Engine(agents, tracing_store=PostgresTracingStore(url))` sets tracing up when the engine opens and
flushes it when it closes. `llamphouse.runtime.tracing.stores` has the classes.

With several processes, give them all the same Postgres or ClickHouse store: a run executed by a worker is read
back by whichever API process serves the request.

## Sending spans elsewhere too

The standard OpenTelemetry environment variables add exporters next to the store's:

| Variable | |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | send spans to an OTLP/HTTP collector too (`/v1/traces` is added when missing) |
| `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TRACES_HEADERS` | headers for it (an API key, say) |
| `OTEL_TRACES_EXPORTER=console` | print spans instead |
| `OTEL_SERVICE_NAME` | the service name (`llamphouse` by default) |
| `OTEL_SERVICE_VERSION` / `SERVICE_VERSION` | `service.version` |
| `OTEL_DEPLOYMENT_ENVIRONMENT` / `DEPLOYMENT_ENVIRONMENT` / `ENVIRONMENT` | `deployment.environment` |

If your application has set up its own OpenTelemetry tracer provider before llamphouse starts, llamphouse adds
the store's exporter to it instead of making its own, and never shuts it down.

## Reading traces

- **Compass**: the Trace panel in a run's view (the spans as a tree, with durations, status and attributes), and
  the Traces page (recent traces, by agent). See [Compass](compass.md).
- **HTTP**: `GET /v2/runs/{id}/trace` returns `{"run", "tracing", "spans"}`: the spans of every trace that has
  the run in it (one per execution), oldest first. Each span is a dict with `TraceId`, `SpanId`,
  `ParentSpanId`, `SpanName`, `Timestamp`, `Duration` (ns), `StatusCode`, `StatusMessage`, `SpanAttributes` and
  its events. Without a store, `"tracing": false`.
- **In code**: `await store.get_trace(run_id)` and `await store.list_traces(limit=50, agent=None)`.

## Usage telemetry

llamphouse 2.0 sends no usage data anywhere. (1.x had an anonymous usage client controlled by
`LLAMPHOUSE_TELEMETRY`; 2.0 does not have it.)

## See also

- [Example 07_Tracing](../../examples/07_Tracing)
- [Compass](compass.md)
