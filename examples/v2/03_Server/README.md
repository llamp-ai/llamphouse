# Served (llamphouse 2.0)

The research desk from `01_ResearchDesk`, served for real: API processes and worker processes on one
Postgres (the truth) and one Redis (the job queue and live events), with live streams (SSE) and a basic
Compass.

## One process, nothing to install

```bash
../../../.venv/bin/python -m llamphouse.runtime app:AGENTS --role all --memory   # http://127.0.0.1:8000/compass
../../../.venv/bin/python chat.py                                                # talk to the desk over HTTP
```

Everything is in memory: nothing survives a restart.

## For real: Postgres + Redis, API and workers

You need a Postgres and a Redis (`docker-compose.yml` describes them; any install works). `app.py`
points at `localhost:5433` and `localhost:6380`; `LLAMPHOUSE_DATABASE_URL` and `LLAMPHOUSE_REDIS_URL`
point elsewhere.

```bash
python -m llamphouse.runtime app:AGENTS --role api --port 8000   # the API and Compass: executes nothing
python -m llamphouse.runtime app:AGENTS --role worker            # executes runs; start as many as you like
python -m llamphouse.runtime app:AGENTS --role worker
python chat.py
```

The schema is created on first start (and migrated on later ones). Try: *"Compare Bangkok and Lisbon"* →
it asks for the month → *"March"* → while the forks work (30 s each, on the clock), *"focus on cost
please"*. Then stop processes and start them again:

- **Ctrl-C a worker**: it hands back what it was running; another worker carries on at once.
- **Kill a worker hard** (`kill -9`): its runs are taken over when their lease expires (30 s), and
  replay their recorded steps - nothing recorded happens twice.
- **Restart the API**: it holds nothing; clients reconnect and carry on.

`LLAMPHOUSE_API_KEY` requires `Authorization: Bearer <key>` (Compass: `/compass?key=<key>`).

## Your own agents

```python
# app.py
from my_agents import Assistant, Researcher

AGENTS = {"assistant": Assistant, "researcher": Researcher}
```

```bash
python -m llamphouse.runtime app:AGENTS --role api
python -m llamphouse.runtime app:AGENTS --role worker --workers 8
```

or in code: `llamphouse.runtime.serve(AGENTS, role=..., database_url=..., redis_url=...)`, and
`llamphouse.runtime.server.app(AGENTS, role="api", ...)` is the ASGI app for any ASGI server. Agent
classes must be importable by module path (not defined inside a function): what a run recorded is read
back by them, in any process.

## How it works

- **Postgres is the truth.** A conversation - a thread people talk to, and every thread its runs create
  (forks, threads of other agents) - is one *tree*. Everything that happens is one transaction on one
  tree, under a lock on that tree: different conversations run in parallel across all processes. Each
  process keeps copies of the trees it works on and brings them up to date at the start of every
  transaction (only the rows written since).
- **Runs are leased.** A worker takes a run only if the store says it's queued, and holds a lease while
  it runs (renewed every 10 s). A sweeper in every worker queues runs again whose lease expired, wakes
  runs whose sleep or timeout ended, and rings again for queued runs nobody took.
- **Redis is the doorbell and the loudspeaker.** The job queue only says "run X can make progress": a
  lost or repeated job is harmless. Live events (new thread events, a run's stream with its text deltas)
  go over pub/sub to whichever API process serves the stream.

## The API

| | | |
|---|---|---|
| `POST /v2/threads` | `{"agent": "desk"}` | a new thread |
| `GET /v2/threads[?agent=&top=true]` | | threads, newest first |
| `GET /v2/threads/{id}` | | the thread: agent, active run, runs, threads it made |
| `POST /v2/threads/{id}/messages` | a message | append, for the record - **409 `thread_busy`** while a run is active |
| `POST /v2/threads/{id}/runs` | a message | start a run - **409 `thread_busy`** while one is active |
| `POST /v2/threads/{id}/close` | | no more messages; its active run is cancelled |
| `GET /v2/threads/{id}/events[/stream]?after=N` | | the thread's record; `/stream` live (SSE) |
| `GET /v2/threads/{id}/state[?at=N]` | | its state, now or at a position |
| `POST /v2/runs/{id}/messages` | a message | to the run's inbox - **409 `run_ended`** once it isn't the thread's active run |
| `POST /v2/runs/{id}/answer` | a message | answer its `ask()` (first answer wins) |
| `POST /v2/runs/{id}/signal` / `cancel` | `{"key", "value"}` / - | |
| `GET /v2/runs/{id}` · `/session` | | the run; its session |
| `GET /v2/runs/{id}/events[/stream]` | | its stream as stored; `/stream` live with text deltas, ends with the run |
| `POST /v2/files?name=x.pdf` | the bytes + Content-Type | upload first, reference in a message later |
| `GET /v2/files/{id}` · `GET /v2/agents` · `GET /health` | | |

A message: `{"text": "..."}`, `{"data": {...}}` or `{"parts": [...]}`, with an optional
`"sender": {"kind": "person", "id": "an", "name": "An"}`.

What a message *means* when a run is active - start a new run, steer the active one, answer its question,
queue it, refuse it - is the app's decision, not llamphouse's. `chat.py` makes one such decision (idle →
start, asked → answer, otherwise → the run's inbox); `client.py` is a thin client that makes none.
