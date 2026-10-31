# Deployment

llamphouse runs as two kinds of process on one shared Postgres and one shared Redis:

- **API processes** serve the HTTP API (`/v2`), the adapters, webhooks and Compass. They hold no state: every
  request reads the store or writes one transaction. Run as many as you like behind a load balancer.
- **Worker processes** execute runs. They also sweep for due work: sleeps that end, runs whose lease expired,
  schedules that are due, subscription deliveries. Run as many as your load needs.

Postgres is the truth. Redis is the doorbell (the job queue only says "run X can make progress") and the
loudspeaker (live events for streams, between processes). A lost or repeated Redis message is harmless.

For development, one process does everything in memory:

```bash
llamphouse up --memory --agents agents.yaml          # http://127.0.0.1:8000, Compass at /compass
```

## Roles

```bash
llamphouse up --role all    --config llamphouse.yaml --agents agents.yaml   # API + worker in one process
llamphouse up --role api    --config llamphouse.yaml --agents agents.yaml   # API and Compass, executes nothing
llamphouse up --role worker --config llamphouse.yaml --agents agents.yaml   # executes runs, serves no HTTP
```

| Role | Serves HTTP | Executes runs | Sweeps (schedules, deliveries, leases) |
|---|---|---|---|
| `all` | yes | yes | yes |
| `api` (or `compass`) | yes | no | no |
| `worker` | no | yes | yes |

Every process gets the same agent files: an API process needs them to validate input and describe agents, a
worker to run them. Any role other than `all` shares work with other processes, so it refuses to start on
in-memory components: it needs a shared `data_store`, `run_queue` and `event_queue`.

### `llamphouse up` options

| Option | Default | |
|---|---|---|
| `--role` | `all` | `all`, `api`, `compass` (same as `api`), `worker` |
| `--config` | `LLAMPHOUSE_CONFIG`, else `./llamphouse.yaml` | the runtime file |
| `--agents` | the runtime file's agent documents | agent files, directories, or `module:NAME`; repeatable |
| `--host` | `HOST`, else `127.0.0.1` | use `0.0.0.0` in a container |
| `--port` | `PORT`, else `8000` | |
| `--workers` | `engine.workers`, else `LLAMPHOUSE_WORKERS`, else 4 | runs this process executes at once |
| `--database-url` | | a Postgres URL: the data store and file store |
| `--redis-url` | | a Redis URL: the run queue and event queue |
| `--api-key` | `LLAMPHOUSE_API_KEY` | one more API key the API accepts |
| `--memory` | off | everything in memory, one process: ignores the runtime file's components and the `LLAMPHOUSE_*` URLs |
| `--no-compass` | off | don't mount Compass |
| `--log-level` | `info` | |

`python -m llamphouse.runtime app:AGENTS --role api` still works and means
`llamphouse up --agents app:AGENTS --role api`.

### Where components come from

For each component, the first of:

1. a command-line option or an argument in code (`--database-url`, `--redis-url`, `store=...`);
2. the runtime file (see [Configuration](configuration.md));
3. the environment: `LLAMPHOUSE_DATABASE_URL` gives a Postgres data store and file store,
   `LLAMPHOUSE_REDIS_URL` a Redis run queue and event queue;
4. in memory.

With `LLAMPHOUSE_DATABASE_URL`, the **config store** (agent config changed at runtime) is the Postgres one too,
unless the runtime file names one. **Tracing** comes from the runtime file's `tracing`, else from
`TRACING_STORE` (see [Tracing](guides/tracing.md)).

## Postgres

`pip install 'llamphouse[postgres]'`. The schema is managed with Alembic; its revisions ship in the package
(`llamphouse.runtime.migrations`). The 2.0 tables are named `llamphouse_*` and sit on their own Alembic branch
(`v2`), next to the 1.x tables if the database has them.

By default every process brings the schema up to date when it starts, under an advisory lock so processes that
start together take turns. To make migrations a deploy step instead, turn that off and run the upgrade
yourself:

```yaml
data_store:   {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}", migrate: false}}
file_store:   {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}", migrate: false}}
config_store: {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}", migrate: false}}
```

```bash
llamphouse db upgrade --config llamphouse.yaml          # bring the schema up to date
llamphouse db upgrade --config llamphouse.yaml --sql    # print the SQL instead, to review or run yourself
llamphouse db current --config llamphouse.yaml          # where the database is
```

`llamphouse db` finds the database from `--database-url`, else the runtime file's `data_store.postgres.dsn`,
else `LLAMPHOUSE_DATABASE_URL` (or `DATABASE_URL`). `--target` picks what to upgrade to: `v2@head` (the
default: the 2.0 tables), `v1@head` (the 1.x tables) or `heads` (both).

## Redis

`pip install 'llamphouse[redis]'`. One Redis serves both the run queue (`run_queue: {redis: {url: ...}}`, a list
under the key `llamphouse:jobs`) and the event queue (`event_queue: {redis: {url: ...}}`, pub/sub channels
prefixed `llamphouse:`). Give each deployment sharing one Redis its own `key` and `prefix`. Redis holds nothing
that can't be lost: queued runs are rung again from Postgres after `redeliver` seconds.

## Scaling

- **API processes** scale horizontally without coordination. A live stream (SSE) is served by whichever process
  the client is connected to; events from workers reach it over Redis.
- **Workers** scale horizontally too. Each takes a run only if the store says it is queued, and holds a lease
  on it while it executes (`engine.lease_ttl`, 30 s by default, renewed while it runs). `--workers` sets how
  many runs one process executes at once; a run that waits (an answer, the clock, a signal, another thread)
  holds no worker.
- **The unit of work** is a conversation: a thread people talk to and every thread its runs create. Everything
  that happens in it is one transaction under a lock on it, so different conversations run in parallel across
  all processes. Each process keeps copies of the conversations it works on and reads only what changed since.

## Stopping

- **A graceful stop** (Ctrl-C, `SIGTERM`): a worker stops taking runs and hands back the runs it is executing.
  They are queued again at once and another worker carries on. An API process finishes its requests and closes;
  clients reconnect and resume streams with `?after=`.
- **A hard stop** (`kill -9`, a lost node): the dead worker's runs keep their lease until it expires, then the
  sweeper queues them again. The worker that takes one over executes it from the top and replays its recorded
  steps: what was recorded does not happen twice. See [Durable runs](concepts/durable-runs.md).

## Health

`GET /health` answers `{"ok": true, "engine": "<id>", "workers": <n>}` without authentication. Use it for
readiness and liveness checks on API processes. Worker processes serve no HTTP; check that the process is
running.

## A Postgres and a Redis for development

[Example 10_Server](../examples/10_Server) includes a `docker-compose.yml` that describes a Postgres 16 on port
5433 and a Redis 7 on port 6380, and a runtime file that points at them (with `${LLAMPHOUSE_DATABASE_URL:-...}`
defaults). Any Postgres and Redis work: a local install, or managed services. Point the runtime file (or
`LLAMPHOUSE_DATABASE_URL` / `LLAMPHOUSE_REDIS_URL`) at them.

## On Kubernetes or another orchestrator

The same image runs every role; only the command changes.

- **Migrations:** a Job (or an init step of the release) runs `llamphouse db upgrade`, with `migrate: false`
  in the runtime file so processes don't race to do it.
- **API:** a Deployment running `llamphouse up --role api --host 0.0.0.0 --config /etc/llamphouse/llamphouse.yaml
  --agents /app/agents.yaml`, behind a Service, with `/health` as its readiness probe. Turn off response
  buffering for `text/event-stream` in proxies in front of it, and allow long-lived connections.
- **Workers:** a Deployment running `llamphouse up --role worker ...`. Set `terminationGracePeriodSeconds`
  long enough for the current step of a run to finish; the worker then hands its runs back. Scale it on CPU or
  on the run queue's length.
- **Configuration:** the runtime file and agent files from a ConfigMap; secrets as environment variables (from
  a Secret), referenced with `${VAR}` or through `secrets` / `secrets_store`.
- **Tracing:** a tracing store in the runtime file, and `OTEL_EXPORTER_OTLP_ENDPOINT` to send spans to a
  collector too.

## In code

```python
from llamphouse.runtime import server

server.serve(["agents.yaml"], role="all", config="llamphouse.yaml", port=8000)   # what `llamphouse up` does
app = server.app(["agents.yaml"], role="api", config="llamphouse.yaml")          # an ASGI app for any server
```

`server.app(...)` opens the engine when the ASGI app starts and closes it when it stops. `await
server.work(...)` runs a worker until `SIGINT` / `SIGTERM`.

## Environment variables

| Variable | Used for |
|---|---|
| `LLAMPHOUSE_CONFIG` | the runtime file, when `--config` isn't given |
| `LLAMPHOUSE_DATABASE_URL` | Postgres for the data store and file store, when the runtime file has none |
| `LLAMPHOUSE_REDIS_URL` | Redis for the run queue and event queue, when the runtime file has none |
| `LLAMPHOUSE_API_KEY` | an API key the API accepts (with any authenticators in the runtime file) |
| `LLAMPHOUSE_WORKERS` | runs a process executes at once, when neither `--workers` nor `engine.workers` says |
| `LLAMPHOUSE_AGENT_VERSION` | the version reported for agents whose class sets none |
| `HOST`, `PORT` | defaults for `--host` and `--port` |
| `LLAMPHOUSE_TRACING_ENABLED`, `TRACING_STORE`, `OTEL_*` | see [Tracing](guides/tracing.md) |

## See also

- [Configuration](configuration.md): every key of the agent and runtime files
- [Example 10_Server](../examples/10_Server): API and workers on Postgres and Redis
- [Auth](guides/auth.md): who may call the API
