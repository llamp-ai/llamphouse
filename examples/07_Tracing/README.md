# Tracing

An agent that answers questions from a small handbook, traced with OpenTelemetry. llamphouse makes a
`llamphouse.run` span around each execution of a run and a `llamphouse.step` span for each step it executes;
the spans the agent makes itself, with any OpenTelemetry tracer, nest under them. Where the spans go is the
runtime file's `tracing:` - here a store in memory. You read a run's spans back over the API
(`GET /v2/runs/{id}/trace`) and in Compass's Trace tab. No model and no key: a stand-in answers.

## Run

```bash
pip install -r requirements.txt          # llamphouse[tracing]: OpenTelemetry
llamphouse up --memory --config llamphouse.yaml --agents agents.yaml     # http://127.0.0.1:8000/compass
python client.py                         # in a second terminal
python client.py "Can I work remotely?"
```

```
> How many days of leave do I get?
You get 25 days of paid leave a year; ask your manager two weeks ahead.

trace of run_480f96b5fc064bc2 (5 spans):
  llamphouse.run                      75.4 ms  {'llamphouse.run.status': 'completed'}
    llamphouse.step                   22.4 ms  {'llamphouse.step.name': 'search'}
      handbook.search                 22.1 ms  {'search.hits': 1}
    rerank                             0.0 ms  {'rerank.kept': 1}
    chat stand-in                     52.1 ms  {'gen_ai.request.model': 'stand-in'}
```

`LLAMPHOUSE_URL` points the client elsewhere (default `http://127.0.0.1:8000`). Without a `tracing:` entry
the store comes from the environment, `TRACING_STORE` (in memory by default); `LLAMPHOUSE_TRACING_ENABLED=false`
turns tracing off.

## In Compass

- **Runs** -> a run -> **Trace**: the spans as a tree with their durations and status; click one for its
  attributes. `gen_ai.input.messages` / `gen_ai.output.messages` (and `input.value` / `output.value`) are
  shown as readable, foldable messages.
- **Traces**: every run's trace, newest first, filterable by agent.

## The code

Make spans as you would anywhere; while `run()` executes, the run's span is the current one:

```python
from opentelemetry import trace

tracer = trace.get_tracer("handbook")

class HandbookAgent(Agent):
    @step
    async def search(self, ctx, question):           # a llamphouse.step span...
        with tracer.start_as_current_span("handbook.search") as span:   # ...with this one under it
            span.set_attribute("search.hits", ...)

    async def run(self, ctx) -> str:
        passages = await self.search(ctx, ctx.input.text)
        with tracer.start_as_current_span("chat stand-in") as span:     # under llamphouse.run
            span.set_attribute("gen_ai.input.messages", json.dumps([...]))
```

```yaml
# llamphouse.yaml
tracing: {in_memory: {}}                                             # this process's spans
# tracing: {postgres: {database_url: "${LLAMPHOUSE_DATABASE_URL}"}}   # the llamphouse_traces table
# tracing: {clickhouse: {clickhouse_url: "${CLICKHOUSE_URL}"}}       # what an OpenTelemetry Collector writes there
```

- `llamphouse.run` carries `run.id`, `thread.id`, `llamphouse.agent`, `llamphouse.parent_run` and the run's
  status and error. A run that waits (an `ask`, a `sleep`) or is taken over after a crash is executed again:
  a new span, the same `run.id` - the trace endpoint returns them all, oldest first.
- A step replayed from the record makes no span: it wasn't done again.
- `OTEL_EXPORTER_OTLP_ENDPOINT` also sends the spans to an OpenTelemetry collector. If the app set up a
  tracer provider of its own, the store's exporter joins it.
- Tracing never fails a run: without a store or a tracer provider no span is made.
