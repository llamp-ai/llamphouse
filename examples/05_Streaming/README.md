# Streaming Example

This example demonstrates how to use LLAMPHouse as a local server that supports streaming responses (similar to OpenAI streaming). It allows a client to send a message to an assistant and receive incremental text deltas in real time.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`
- (Optional) PostgreSQL database (only if you want persistence)

## Setup

1. Clone the repository and go to this example:

   ```bash
   git clone https://github.com/llamp-ai/llamphouse.git
   cd llamphouse/examples/05_Streaming
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

## Streaming (How it works)

- Server starts an OpenAI streaming completion [server.py](server.py#L25) with `stream=True`.
- Server converts stream chunks into events using `context.handle_completion_stream(...)` + `get_adapter("openai")`.
- `on_event(evt)` is a user-defined server-side callback (hook). LLAMPHouse will call it every time a new stream event is produced, so you can “intercept” the stream in real time and decide what to do with it.

  - Example usage patterns: print to console, log/trace, push deltas to a UI (WebSocket/SSE), or collect metrics.
  - In this example:
    - `TextDelta` prints incremental text as it arrives.
    - `ToolCallDelta` prints tool-call name
- After stream finishes, the server stores the final assistant text via `context.insert_message(...)`.
- Client uses `client.beta.threads.runs.stream(...)` and prints deltas in `on_text_delta`.

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
2. Switch in [server.py](server.py#L53) :

   ```python
   data_store = PostgresDataStore()
   ```

   * Run migrations (from the `llamphouse/` folder that contains `migrations/`)

     ```bash
     cd ../..
     alembic upgrade head
     cd examples/05_Streaming
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
   cd llamphouse/examples/05_Streaming
   ```
2. Start the server `http://127.0.0.1:8000`:
   ```sh
   python server.py
   ```

## Running the Client

1. Open a new terminal and navigate to the example directory:
   ```sh
   cd llamphouse/examples/05_Streaming
   ```
2. Run the client:
   ```sh
   python client.py
   ```
