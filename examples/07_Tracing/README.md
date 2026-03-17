# Tracing Example (A2A)

End-to-end OpenTelemetry tracing with an A2A client and server. The client creates a root span and injects W3C `traceparent` headers into every httpx request, so all server-side spans appear nested under a single trace.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`
- Docker (for ClickHouse + OTel Collector)

## Setup

1. Navigate to this example:

   ```bash
   cd llamphouse/examples/08_Tracing
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` from the sample:

   ```bash
   cp .env.sample .env
   # Add your OPENAI_API_KEY
   ```

4. Start ClickHouse and the OTel Collector (from the repo root):

   ```bash
   cd ../../docker
   docker compose up -d clickhouse otel-collector
   ```

## How It Works

| Component | What happens |
|-----------|-------------|
| **Client** | Creates a root span, injects `traceparent` into httpx headers, then uses the `a2a-sdk` to stream a question via `message/stream`. |
| **Server** | Mounts `A2AAdapter`, extracts incoming trace context, and creates child spans for every operation (queue, worker, streaming, data-store). |
| **Collector** | Receives OTLP spans on port 4318 and writes them to ClickHouse. |
| **Compass** | Reads from ClickHouse and displays traces at `http://127.0.0.1:8000/compass/`. |

## Exclude Spans (Optional)

Suppress noisy spans by prefix in `server.py`:

```python
llamphouse = LLAMPHouse(
    agents=[my_agent],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
    exclude_spans=[
        "llamphouse.data_store",
        "llamphouse.queue",
    ],
)
```

## Running

1. Start the server:

   ```bash
   python server.py
   ```

2. In a second terminal, run the client:

   ```bash
   python client.py
   ```

3. Open Compass to view the trace:

   ```
   http://127.0.0.1:8000/compass/
   ```

## Alternative: Export to Langfuse

To send traces to Langfuse instead of the local Collector, update your `.env`:

```dotenv
OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<BASE64(pk:sk)>
```
