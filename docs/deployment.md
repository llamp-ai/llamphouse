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
from llamphouse.core.data_stores.postgres_store import PostgresDataStore

app = LLAMPHouse(
    agents=[...],
    data_store=PostgresDataStore(),  # reads DATABASE_URL
)
```

### Database migrations

LLAMPHouse uses [Alembic](https://alembic.sqlalchemy.org/) for schema migrations:

```bash
# Start a local Postgres
docker run --rm -d --name postgres \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres
docker exec -it postgres psql -U postgres -c 'CREATE DATABASE llamphouse;'

# Set the connection string
export DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/llamphouse

# Apply all migrations
alembic upgrade head

# Create a new migration (after model changes)
alembic revision --autogenerate -m "description of change"

# Roll back all migrations
alembic downgrade base
```

## Redis

For multi-process deployments, use Redis for the run queue and event queues:

```python
from llamphouse.core.queues.redis_queue import RedisQueue
from llamphouse.core.streaming.event_queue.redis_event_queue import RedisEventQueue

app = LLAMPHouse(
    agents=[...],
    run_queue=RedisQueue(),               # reads REDIS_URL
    event_queue_class=RedisEventQueue,    # reads REDIS_URL
)
```

## Distributed workers

For high-throughput deployments, separate the API server from worker processes. The API server handles HTTP requests, while workers pull runs from the shared queue and execute agent logic.

```python
# api.py — API server only (no worker)
app = LLAMPHouse(
    agents=[...],
    data_store=PostgresDataStore(),
    run_queue=RedisQueue(),
    event_queue_class=RedisEventQueue,
)
# Start with: python -m llamphouse --no-workers api.py

# worker.py — Worker process
from llamphouse.core.worker import Worker

worker = Worker(
    agents=[...],
    data_store=PostgresDataStore(),
    run_queue=RedisQueue(),
    event_queue_class=RedisEventQueue,
)
worker.start()
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

See the `docker/docker-compose.prod.yml` for a production split-mode setup and [example 14_DistributedWorker](https://github.com/llamp-ai/llamphouse/tree/main/examples/14_DistributedWorker) for a complete implementation.

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
