# Installation

## Requirements

- Python 3.10 or later (3.11 or later if you use LangGraph's `interrupt()`).
- For production: a Postgres and a Redis. For development you need neither: everything runs in
  memory, in one process.

## Install

llamphouse 2.0 is an alpha, so pip only picks it when you ask for it:

```bash
pip install "llamphouse==2.0.0a0"
```

Pin the exact version: the API may change between alphas.

The base install is small (pydantic, FastAPI, uvicorn, PyYAML, httpx). It runs `llamphouse up --memory`,
the HTTP API, Compass and the in-memory backends.

## Extras

Backends and integrations are extras. Install the ones you use:

| extra | what it adds | install |
|---|---|---|
| `postgres` | the Postgres data store, file store, config store and tracing store; migrations (Alembic) | `pip install "llamphouse[postgres]==2.0.0a0"` |
| `redis` | the Redis run queue and event queue, for several processes | `pip install "llamphouse[redis]==2.0.0a0"` |
| `tracing` | OpenTelemetry tracing, and the ClickHouse tracing store | `pip install "llamphouse[tracing]==2.0.0a0"` |
| `oidc` | the `oidc` authenticator (JWT bearer tokens checked against the issuer's keys) | `pip install "llamphouse[oidc]==2.0.0a0"` |
| `ag-ui` | the AG-UI adapter | `pip install "llamphouse[ag-ui]==2.0.0a0"` |
| `langgraph` | `llamphouse.helpers.langgraph`: LangGraph graphs as durable agents | `pip install "llamphouse[langgraph]==2.0.0a0"` |
| `all` | `postgres`, `redis`, `tracing` and `oidc` | `pip install "llamphouse[all]==2.0.0a0"` |

Combine them as usual: `pip install "llamphouse[postgres,redis]==2.0.0a0"`. When a runtime file names a
backend whose extra is missing, llamphouse says which extra to install.

Model SDKs are not extras. Your agent calls the model, so install the SDK yourself (for example
`pip install openai`).

## From source

```bash
git clone https://github.com/llamp-ai/llamphouse.git
cd llamphouse
pip install -e ".[all]"
```

## Check it works

```bash
llamphouse --help
```

You should see the `up`, `check`, `db` and `migrate` commands. Then go to the
[Quickstart](quickstart.md).
