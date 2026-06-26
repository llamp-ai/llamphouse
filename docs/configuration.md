# Configuration

## LLAMPHouse constructor

The `LLAMPHouse` class accepts the following parameters:

```python
from llamphouse.core import LLAMPHouse
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.adapters.assistant_api import AssistantAPIAdapter
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.data_stores.postgres_store import PostgresDataStore

app = LLAMPHouse(
    agents=[...],                        # List of Agent instances
    adapters=[A2AAdapter()],             # Protocol adapters
    data_store=InMemoryDataStore(),      # Storage backend
    authenticator=None,                  # Optional authentication
    worker=None,                         # Optional custom worker
    event_queue_class=None,              # Event queue implementation
    run_queue=None,                      # Run queue implementation
    config_store=None,                   # Runtime config store
    retention_policy=None,               # Data retention policy
    exclude_spans=None,                  # Tracing span exclusions
    compass=True,                        # Enable Compass dashboard
)
```

### Parameter reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agents` | `list[Agent]` | `[]` | List of agent instances to register |
| `adapters` | `list[BaseAPIAdapter]` | `[AssistantAPIAdapter()]` | Protocol adapters. `None` → default; `[]` → none |
| `data_store` | `BaseDataStore` | `InMemoryDataStore()` | Async storage backend for threads, messages, runs, and run steps |
| `authenticator` | `BaseAuth` | `None` | Authentication handler |
| `worker` | `BaseWorker` | `None` | Custom worker implementation |
| `event_queue_class` | `BaseEventQueue` | `InMemoryEventQueue` | Event queue class for streaming |
| `run_queue` | `BaseQueue` | `InMemoryQueue()` | Queue for pending runs |
| `config_store` | `BaseConfigStore` | `InMemoryConfigStore()` | Runtime config parameter store |
| `retention_policy` | `RetentionPolicy` | Default policy | Data retention/purge configuration |
| `exclude_spans` | `list[str]` | `[]` | Glob patterns for spans to exclude from tracing |
| `compass` | `bool` | `True` | Auto-mount the Compass dashboard adapter |

### `ignite()` method

```python
app.ignite(
    host="0.0.0.0",   # Bind address
    port=80,           # Port number
    reload=False,      # Enable auto-reload (development)
)
```

## Data stores

| Store | Class | When to use |
|---|---|---|
| **In-memory** | `InMemoryDataStore` | Development, testing, stateless deployments |
| **Postgres** | `PostgresDataStore` | Production, persistent data |

```python
# In-memory (default)
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
data_store = InMemoryDataStore()

# Postgres
from llamphouse.core.data_stores.postgres_store import PostgresDataStore
data_store = PostgresDataStore(database_url="postgresql+asyncpg://...")
```

## YAML configuration

`llamphouse up` can build the runtime from `llamphouse.yaml`.

```yaml
version: "0.1"

definitions:
  - name: report-agent
    entrypoint: agents.py:ReportAgent

agents:
  - name: report-worker
    definition: report-agent
    triggers:
      - webhook:
          path: /triggers/report
          secret_env: WEBHOOK_SECRET
          idempotency:
            key: id
          thread_metadata:
            tenant_id: tenant.id
          run_metadata:
            event_type: type
            event_id: id

data_store:
  postgres:
    database_url: ${DATABASE_URL}
    pool_size: 5

tracing:
  in_memory:
```

String values in `llamphouse.yaml` can reference environment variables with
`${ENV_VAR}`. Missing environment variables fail config loading before the
server starts.

Validate a config without starting the server:

```bash
llamphouse check --config llamphouse.yaml
llamphouse check --config llamphouse.yaml --format json
```

`llamphouse check` loads `.env` from the config directory before falling back
to the current working directory. It validates schema, entrypoints, framework
component configuration, route conflicts, and lightweight external dependency
pings. It does not run agents, start workers, create tracing tables, run
migrations, or audit database structure.

Webhook trigger metadata mappings copy values from the JSON payload into
thread or run metadata. Missing payload paths are ignored and the webhook still
returns `202 Accepted`; the full payload remains available on
`run.metadata["__trigger__"]["data"]`.

Webhook idempotency is opt-in. `idempotency.key` is a JSON body dot-path; when
the same Agent Deployment receives the same key on the same webhook path,
LLAMPHouse returns the original `run_id` and `thread_id` with `deduped: true`
instead of creating another run.

## Queue backends

| Queue | Use case |
|---|---|
| `InMemoryQueue` | Single-process deployments (default) |
| `RedisQueue` | Multi-process / distributed deployments |

| Event Queue | Use case |
|---|---|
| `InMemoryEventQueue` | Single-process (default) |
| `RedisEventQueue` | Multi-process / distributed |

## Authentication

Implement `BaseAuth` for custom authentication:

```python
from llamphouse.core.auth.key_auth import KeyAuth

app = LLAMPHouse(
    agents=[...],
    authenticator=KeyAuth("my-secret-key"),
)
```

Clients must include the key in the `Authorization` header:

```
Authorization: Bearer my-secret-key
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | Postgres connection string passed to `PostgresDataStore` | _(none)_ |
| `REDIS_URL` | Redis URL for queues | _(in-memory if unset)_ |
| `LLAMPHOUSE_TRACING_ENABLED` | Enable OpenTelemetry tracing | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | _(none)_ |
| `OTEL_SERVICE_NAME` | Service name for traces | `llamphouse` |
| `CLICKHOUSE_URL` | ClickHouse URL for Compass traces view | _(none)_ |

## Next steps

- [Deployment](deployment.md) — Docker setup with Postgres, Redis, and tracing
- [Adapters](concepts/adapters.md) — protocol adapter configuration
- [Config Store](guides/config-store.md) — runtime-tunable parameters
