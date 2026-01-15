# Chat Example

This example demonstrates a simple interactive chat client talking to a LLAMPHouse server.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`
- (Optional) PostgreSQL database (only if you want persistence)

## Setup

1. Clone the repository and go to this example.

   ```sh
   git clone https://github.com/llamp-ai/llamphouse.git
   cd llamphouse/examples/02_Chat
   ```
2. Install dependencies from `requirements.txt`.

   ```sh
   pip install -r requirements.txt
   ```
3. Create your `.env` from `.env.sample`:

   - `DATABASE_URL=...` (optional; only for Postgres)
   - `OPENAI_API_KEY=...` (required)

   ```bash
   cp .env.sample .env
   ```

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
2. Switch in [server.py](server.py#L36) :

   ```python
   data_store = PostgresDataStore()
   ```

   * Run migrations (from the `llamphouse/` folder that contains `migrations/`)

     ```bash
     cd ../..
     alembic upgrade head
     cd examples/02_Chat
     ```

## Running the Server

1. Navigate to the example directory:

   ```sh
   cd llamphouse/examples/02_Chat
   ```
2. Start the server `http://127.0.0.1:8000`:

   ```sh
   python server.py
   ```

## Running the Client

1. Open a new terminal and navigate to the example directory:

   ```sh
   cd llamphouse/examples/02_Chat
   ```
2. Run the client:

   ```sh
   python client.py
   ```
