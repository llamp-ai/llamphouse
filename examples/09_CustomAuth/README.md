# Custom Authenticator Example

Demonstrates how to implement a **custom authenticator** on your LLAMPHouse server with **A2A streaming**.

## What it shows

| Feature | How it's used |
|---|---|
| **Custom auth** | `CustomAuth(BaseAuth)` — only accepts `api_key == "secret_key"` |
| **A2A adapter** | Agent discovered via `/.well-known/agent.json` |
| **Streaming** | Real-time token streaming via `process_stream()` |
| **Auth headers** | Client passes `Authorization: Bearer secret_key` on every request |

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY` in a `.env` file

## Setup

```sh
cd examples/09_CustomAuth
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Authentication

This example uses a custom authenticator by extending `BaseAuth`:

```python
class CustomAuth(BaseAuth):
    def authenticate(self, api_key: str):
        if api_key == "secret_key":
            return True
        return False
```

- **Server:** `CustomAuth` only accepts `api_key == "secret_key"`
- **Client:** passes the key via `Authorization: Bearer secret_key` header

If you change the key, update it in both files.

## Running

**Terminal 1 — start the server:**

```sh
python server.py
```

**Terminal 2 — run the client:**

```sh
python client.py
```

The client discovers the agent via A2A, sends a message, and streams the response token-by-token.

## Compass Dashboard

Open [http://127.0.0.1:8000/compass](http://127.0.0.1:8000/compass) to view threads, messages, and runs.
