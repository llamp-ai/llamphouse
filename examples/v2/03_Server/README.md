# Served (llamphouse 2.0)

The research desk from `01_ResearchDesk`, served for real: API processes and worker processes on one
Postgres (the truth) and one Redis (the job queue and live events), with live streams (SSE) and a basic
Compass.

Two files describe the deployment: `agents.yaml` (what the agents are - it travels with them) and
`llamphouse.yaml` (where and how they run - one per environment). `llamphouse check` loads both and says what
it found, without connecting to anything.

## One process, nothing to install

```bash
llamphouse up --role all --memory --config llamphouse.yaml --agents agents.yaml   # http://127.0.0.1:8000/compass
python chat.py                                                                    # talk to the desk over HTTP
```

`--memory` ignores the runtime file's components: everything in memory, nothing survives a restart.

## For real: Postgres + Redis, API and workers

You need a Postgres and a Redis (`docker-compose.yml` describes them; any install works). `llamphouse.yaml`
points at `localhost:5433` and `localhost:6380`; LLAMPHOUSE_DATABASE_URL and LLAMPHOUSE_REDIS_URL point
elsewhere.

```bash
llamphouse db upgrade --config llamphouse.yaml                                   # the schema (also done on start)
llamphouse up --role api    --config llamphouse.yaml --agents agents.yaml --port 8000
llamphouse up --role worker --config llamphouse.yaml --agents agents.yaml        # as many as you like
python chat.py
```

Try: *"Compare Bangkok and Lisbon"* → it asks for the month → *"March"* → while the forks work (30 s each, on
the clock), *"focus on cost please"*. Then stop processes and start them again:

- **Ctrl-C a worker**: it hands back what it was running; another worker carries on at once.
- **Kill a worker hard** (`kill -9`): its runs are taken over when their lease expires (30 s), and
  replay their recorded steps - nothing recorded happens twice.
- **Restart the API**: it holds nothing; clients reconnect and carry on.

`LLAMPHOUSE_API_KEY` requires `Authorization: Bearer <key>` (Compass: `/compass?key=<key>`).

## Your own agents

```yaml
# agents.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: assistant}
spec:
  entrypoint: my_agents/assistant.py:Assistant     # or a module path: my_agents.assistant:Assistant
  config: {model: gpt-5}                            # the agent's Config
```

Every component in the runtime file is swappable: by name (`data_store: {postgres: {...}}`,
`tracing: {clickhouse: {...}}`), or your own class (`file_store: {my_pkg.stores:S3Store: {...}}`). In code:
`llamphouse.runtime.serve(...)`, or `llamphouse.runtime.server.app(...)` for any ASGI server; a Python mapping
of agents works too (`--agents my_module:AGENTS`). Agent classes must be importable by module path (not
defined inside a function): what a run recorded is read back by them, in any process.

## Other protocols

`llamphouse.yaml` also mounts the OpenAI Responses + Conversations and Assistants APIs and A2A, on the same
agents and threads:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
first = client.responses.create(model="desk", input="Compare Bangkok and Lisbon")
print(first.output_text)                       # "For which month?"
done = client.responses.create(model="desk", input="March", previous_response_id=first.id)
```

A2A clients find the desk at `/a2a/desk/.well-known/agent-card.json`; AG-UI frontends can be added with
`- ag_ui: {prefix: /ag-ui}` (`pip install 'llamphouse[ag-ui]'`).

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
| `POST /v2/threads` | `{"agent": "desk"}`, optional `metadata`, `state`, `config`, `session` | a new thread: the app's labels, its initial state, the agent's config for it, what its session starts from |
| `GET /v2/threads[?agent=&top=true&metadata.org=acme]` | | threads, newest first |
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
| `GET /v2/stream/runs[?agent=&metadata.org=acme]` | | every run's status changes, live (SSE) |
| `GET /v2/files/{id}` · `GET /v2/agents` · `GET /health` | | |

A message: `{"text": "..."}`, `{"data": {...}}` or `{"parts": [...]}`, with an optional
`"sender": {"kind": "person", "id": "an", "name": "An"}`.

What a message *means* when a run is active - start a new run, steer the active one, answer its question,
queue it, refuse it - is the app's decision, not llamphouse's. `chat.py` makes one such decision (idle →
start, asked → answer, otherwise → the run's inbox); `client.py` is a thin client that makes none.
