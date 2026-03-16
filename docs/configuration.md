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
| `data_store` | `BaseDataStore` | `InMemoryDataStore()` | Storage backend for threads, messages, runs |
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
data_store = PostgresDataStore()  # uses DATABASE_URL env var
```

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
| `DATABASE_URL` | Postgres connection string | _(in-memory if unset)_ |
| `REDIS_URL` | Redis URL for queues | _(in-memory if unset)_ |
| `TRACING_ENABLED` | Enable OpenTelemetry tracing | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | _(none)_ |
| `OTEL_SERVICE_NAME` | Service name for traces | `llamphouse` |
| `CLICKHOUSE_URL` | ClickHouse URL for Compass traces view | _(none)_ |

## Next steps

- [Deployment](deployment.md) — Docker setup with Postgres, Redis, and tracing
- [Adapters](concepts/adapters.md) — protocol adapter configuration
- [Config Store](guides/config-store.md) — runtime-tunable parameters
