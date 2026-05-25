# Deployment

## Docker Compose

LLAMPHouse includes a Docker Compose setup for production deployments with all supporting services.

### Quick start

```bash
cd docker
docker compose up -d
```

This starts:

| Service | Port | Purpose |
|---|---|---|
| **Runtime** | `8080` | Your agent server |
| **Postgres** | `5432` | Persistent data store |
| **Redis** | `6379` | Run queue and event queue |
| **OTel Collector** | `4318` | Trace collection |
| **ClickHouse** | `8123` | Trace storage for Compass |

### Configuration

The Docker setup uses environment variables. Key variables in the compose file:

```yaml
services:
  runtime:
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:password@postgres:5432/llamphouse
      REDIS_URL: redis://redis:6379
      LLAMPHOUSE_TRACING_ENABLED: "true"
      OTEL_EXPORTER_OTLP_ENDPOINT: http://otel-collector:4318
      OTEL_SERVICE_NAME: llamphouse
      CLICKHOUSE_URL: http://clickhouse:8123
```

## Postgres

For production, use Postgres instead of the in-memory store:

```python
import os
from llamphouse.core.data_stores.postgres_store import PostgresDataStore

app = LLAMPHouse(
    agents=[...],
    data_store=PostgresDataStore(os.environ["DATABASE_URL"]),
)
```

### Database migrations

LLAMPHouse uses [Alembic](https://alembic.sqlalchemy.org/) for schema migrations:

```bash
# Start a local Postgres
docker run --rm -d --name postgres \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=llamphouse \
  -p 5432:5432 postgres

# Set the connection string
export DATABASE_URL=postgresql://postgres:password@localhost:5432/llamphouse

# If port 5432 is already used locally, map another host port instead:
docker run --rm -d --name postgres-5433 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=llamphouse \
  -p 5433:5432 postgres
export DATABASE_URL=postgresql://postgres:password@localhost:5433/llamphouse

# Apply all migrations
uv run alembic upgrade head

# Create a new migration (after model changes)
uv run alembic revision --autogenerate -m "description of change"

# Roll back all migrations
uv run alembic downgrade base
```

`PostgresDataStore` accepts both `postgresql://` and
`postgresql+asyncpg://` URLs. Sync-style Postgres URLs are converted to
`asyncpg` internally, while Alembic uses the same `DATABASE_URL` when applying
schema migrations.

## Redis

For multi-process deployments, use Redis for the run queue and event queues:

```python
import os
from llamphouse.core.queue.redis_queue import RedisQueue
from llamphouse.core.streaming.event_queue.redis_event_queue import RedisEventQueueFactory

app = LLAMPHouse(
    agents=[...],
    run_queue=RedisQueue(os.environ["REDIS_URL"]),
    event_queue_class=RedisEventQueueFactory(os.environ["REDIS_URL"]),
)
```

## Distributed workers

For high-throughput deployments, separate the API server from worker processes. The API server handles HTTP requests, while workers pull runs from the shared queue and execute agent logic. Use Postgres for shared run state and Redis for run/event queues. Persisted run fields include `stream`, `provider_config`, `config_values`, lifecycle timestamps, and `usage`, so workers can execute runs created by another process with the same behavior as the in-memory store.

```python
# api.py - API server only
import os
from llamphouse.core import LLAMPHouse
from llamphouse.core.queue.redis_queue import RedisQueue
from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.streaming.event_queue.redis_event_queue import RedisEventQueueFactory

app = LLAMPHouse(
    agents=[...],
    data_store=PostgresDataStore(os.environ["DATABASE_URL"]),
    run_queue=RedisQueue(os.environ["REDIS_URL"]),
    event_queue_class=RedisEventQueueFactory(os.environ["REDIS_URL"]),
)
# Start with: llamphouse serve api:app --no-workers

# worker.py - Worker process
import asyncio
import os
from llamphouse.core.queue.redis_queue import RedisQueue
from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.workers.distributed_worker import DistributedWorker

worker = DistributedWorker(
    redis_url=os.environ["REDIS_URL"],
    agents=[...],
    data_store=PostgresDataStore(os.environ["DATABASE_URL"]),
    run_queue=RedisQueue(os.environ["REDIS_URL"]),
)
asyncio.run(worker.run_forever())
```

Scale by running multiple worker processes:

```bash
# Terminal 1: API server
python api.py

# Terminal 2-N: Workers
python worker.py
python worker.py
python worker.py
```

See the `docker/docker-compose.prod.yml` for a production split-mode setup and [example 10_DistributedWorker](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_DistributedWorker) for a complete implementation.

## Production checklist

- [ ] Use `PostgresDataStore` for persistent storage
- [ ] Use `RedisQueue` and `RedisEventQueue` for scalability
- [ ] Run Alembic migrations before deploying
- [ ] Enable tracing with an OTel collector
- [ ] Set up ClickHouse for Compass trace viewing
- [ ] Configure authentication (`KeyAuth` or custom `BaseAuth`)
- [ ] Set appropriate `retention_policy` for data cleanup
- [ ] Consider split-mode (API + workers) for high-throughput

## Next steps

- [Configuration](configuration.md) — full parameter reference
- [Tracing](guides/tracing.md) — OpenTelemetry setup
- [Compass Dashboard](guides/compass.md) — built-in monitoring UI
