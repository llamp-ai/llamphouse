# Tool Call Example

This example demonstrates a minimal tool-calling loop inside a custom `Assistant` running on a LLAMPHouse server.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`
- (Optional) PostgreSQL database (only if you want persistence)

## Setup

1. Clone the repository and go to this example:

   ```bash
   git clone https://github.com/llamp-ai/llamphouse.git
   cd llamphouse/examples/04_ToolCall
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

## Tool Call (How it works)

- Tool function: [get_current_time](server.py#L15)
- Tool schema exposed to the model: [TOOL_SCHEMAS](server.py#L19)
- Tool registry used to execute tool locally: [TOOL_REGISTRY](server.py#L30)
- The assistant calls OpenAI with `tools=...` and `tool_choice="auto"`
- If the model returns `tool_calls`, the server executes the tool and appends a `"tool"` message with the result
- Loop is limited to 3 iterations

Client sends the [question](client.py#L3) defined in `client.py`.

## Choose `data_store`

### Option A: In-memory (default, no DB required)

`server.py` already uses:

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
2. Switch in [server.py](server.py#L82) :

   ```python
   data_store = PostgresDataStore()
   ```

   * Run migrations (from the `llamphouse/` folder that contains `migrations/`)

     ```bash
     cd ../..
     alembic upgrade head
     cd examples/04_ToolCall
     ```

## Running the Server

1. Navigate to the example directory:
   ```sh
   cd llamphouse/examples/04_ToolCall
   ```
2. Start the server `http://127.0.0.1:8000`:
   ```sh
   python server.py
   ```

## Running the Client

1. Open a new terminal and navigate to the example directory:
   ```sh
   cd llamphouse/examples/04_ToolCall
   ```
2. Run the client:
   ```sh
   python client.py
   ```
