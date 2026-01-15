# Gemini Streaming Example

This example demonstrates using LLAMPHouse as an OpenAI-compatible local server that streams responses, while the upstream model is Gemini (via `google-genai`).

## Prerequisites

- Python 3.10+
- `GEMINI_API_KEY` (required)
- (Optional) PostgreSQL database (only if you want persistence)

## Setup

1. Clone the repository and go to this example:

   ```bash
   git clone https://github.com/llamp-ai/llamphouse.git
   cd llamphouse/examples/06_GeminiStreaming
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Create your `.env` from `.env.sample`:

   - `GEMINI_API_KEY=...` (required)
   - `DATABASE_URL=...` (optional; only for Postgres)

   ```bash
   cp .env.sample .env
   ```

## Streaming (How it works)

- Server creates a Gemini streaming generator via `client.models.generate_content_stream(...)` in [server.py](server.py#L28).
- Stream chunks are normalized into LLAMPHouse “canonical” streaming events using `get_adapter("gemini")` + `context.handle_completion_stream(...)`.
- `on_event(evt)` is a user-defined server-side callback (hook). LLAMPHouse will call it every time a new stream event is produced, so you can “intercept” the stream in real time and decide what to do with it.

  - Example usage patterns: print to console, log/trace, push deltas to a UI (WebSocket/SSE), or collect metrics.
  - In this example:
    - `TextDelta` logs a preview of each text delta (e.g., delta_len and the first few characters) as it arrives.
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

1. Ensure Postgres is running and set `DATABASE_URL` in `.env` (see `.env.sample`).
2. Switch in [server.py](server.py#L56) :

```py
data_store = PostgresDataStore()
```

3. Run migrations (from the `llamphouse/` folder that contains `migrations/`):

```sh
cd ../..
alembic upgrade head
cd examples/06_GeminiStreaming
```

## Choose `event_queue`

This example can use different event queue implementations:

- Default: `InMemoryEventQueue`
- Optional: `JanusEventQueue` for async support

Switch in [server.py](server.py#L59):

```py
event_queue_class = InMemoryEventQueue # or JanusEventQueue
```

## Running the Server

1. Navigate to the example directory:
   ```sh
   cd llamphouse/examples/06_GeminiStreaming
   ```
2. Start the server `http://127.0.0.1:8000`:
   ```sh
   python server.py
   ```

## Running the Client

1. Open a new terminal and navigate to the example directory:
   ```sh
   cd llamphouse/examples/06_GeminiStreaming
   ```
2. Run the client:
   ```sh
   python client.py
   ```
