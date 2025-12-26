# Custom Authenticator Example

This example demonstrates how to implement a custom authenticator on your LLAMPHouse server.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`
- (Optional) PostgreSQL database (only if you want persistence)

## Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/llamp-ai/llamphouse.git
   cd llamphouse/examples/03_CustomAuth
   ```
2. Install any required packages:

   ```sh
   pip install -r requirements.txt
   ```
3. Create your `.env` from `.env.sample`:

   - `DATABASE_URL=...` (optional; only for Postgres)
   - `OPENAI_API_KEY=...` (required)

   ```sh
   cp .env.sample .env
   ```

## Authentication (Custom Auth)

This example uses a custom authenticator by extending `BaseAuth`:

- Server: `CustomAuth` in [server.py](server.py#L13) only accepts `api_key == "secret_key"`.
- Client: must use the same key in [client.py](client.py#L5) via `OpenAI(api_key="secret_key", base_url=...)`.

If you change the key, update it in both files.

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
2. Switch in [server.py](server.py#L42) :

   ```python
   data_store = PostgresDataStore()
   ```

   * Run migrations (from the `llamphouse/` folder that contains `migrations/`)

     ```bash
     cd ../..
     alembic upgrade head
     cd examples/03_CustomAuth
     ```

## Running the Server

1. Navigate to the example directory:

   ```sh
   cd llamphouse/examples/03_CustomAuth
   ```
2. Start the server `http://127.0.0.1:8000`:

   ```sh
   python server.py
   ```

## Running the Client

1. Open a new terminal and navigate to the example directory:

   ```sh
   cd llamphouse/examples/03_CustomAuth
   ```
2. Run the client:

   ```sh
   python client.py
   ```
