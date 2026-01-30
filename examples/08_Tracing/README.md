# Tracing Example (Langfuse)

This example demonstrates how to enable end-to-end tracing in LLAMPHouse and export spans to Langfuse via OpenTelemetry (OTLP). It includes a server and client that share trace context.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`
- Langfuse project keys (public + secret)
- (Optional) PostgreSQL database (only if you want persistence)

## Setup

1. Clone the repository and go to this example:

   ```bash
   git clone https://github.com/llamp-ai/llamphouse.git
   cd llamphouse/examples/08_Tracing
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Create your `.env` from `.env.sample`:

   - `OPENAI_API_KEY=...` (required)
   - `DATABASE_URL=...` (optional; only for Postgres)

   ```bash
   cp .env.sample .env
   ```

   Required for Langfuse (OTLP):

   - `TRACING_ENABLED=True`
   - `OTEL_TRACES_EXPORTER=otlp`
   - `OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel/v1/traces`
   - `OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20[BASE64(pk:sk)]`
   - `OTEL_SERVICE_NAME=llamphouse-server`

   PowerShell base64 example:

   ```powershell
   [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("pk-...:sk-..."))
   ```

## How it works

- Client creates a root span and injects trace headers.
- Server extracts the trace context and creates child spans for:
  - threads/messages/runs routes
  - queue operations
  - worker execution
  - data store CRUD

## Exclude spans (optional)

You can suppress noisy spans by prefix. Any span name that starts with a value in `exclude_spans` will be skipped.

```py
llamphouse = LLAMPHouse(
    assistants=[my_assistant],
    data_store=data_store,
    event_queue_class=event_queue_class,
    exclude_spans=[
        "llamphouse.data_store",
        "llamphouse.queue",
    ],
)
```

Example: excluding `llamphouse.data_store` also hides `llamphouse.data_store.insert_message`.

## Choose `data_store`

### Option A: In-memory (default, no DB required)

This example supports multiple data stores (pluggable `data_store`).

```py
data_store = InMemoryDataStore()
```

Notes:

- No migrations needed
- Data resets when the server restarts

### Option B: Postgres (optional)

1. Ensure Postgres is running and set `DATABASE_URL` in `.env` (see `.env.sample`)

   ```bash
   docker run --rm -d --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 postgres
   docker exec -it postgres psql -U postgres -c 'CREATE DATABASE llamphouse;'
   ```
2. Switch in [server.py](server.py#L76) :

   ```python
   data_store = PostgresDataStore()
   ```

   * Run migrations (from the `llamphouse/` folder that contains `migrations/`)

     ```bash
     cd ../..
     alembic upgrade head
     cd examples/08_Tracing
     ```

## Choose `event_queue`

This example can use different event queue implementations:

- Default: `InMemoryEventQueue`
- Optional: `JanusEventQueue`

Switch in [server.py](server.py#L56):

```py
event_queue_class = InMemoryEventQueue # or JanusEventQueue
```

## Running the Server

1. Navigate to the example directory:
   ```sh
   cd llamphouse/examples/08_Tracing
   ```
2. Start the server `http://127.0.0.1:8000`:
   ```sh
   python server.py
   ```

## Running the Client

1. Open a new terminal and navigate to the example directory:

   ```sh
   cd llamphouse/examples/08_Tracing
   ```
2. Run the client:

   ```sh
   python client.py
   ```
   After running, check Langfuse to see a single trace with server spans nested under the client root.
