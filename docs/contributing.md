# Contributing

Contributions are welcome: bug fixes, features, documentation and examples.

## Setting up

```bash
git clone https://github.com/<your-username>/llamphouse.git
cd llamphouse
uv sync                      # the package (editable) with every extra and the test tools
```

Without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e '.[all,ag-ui,langgraph]'`
plus `pytest pytest-asyncio httpx openai`.

## Where things are

```
llamphouse/llamphouse/
├── sdk/                    # the agent contract: what agent code imports (pydantic and the standard library only)
│   ├── agent.py            #   Agent, Workflow, Config, @step, @requires
│   ├── context.py          #   RunContext and what hangs off it (ctx.run, ctx.thread, ctx.client)
│   ├── thread.py           #   thread events, state, sessions, identities
│   ├── content.py          #   messages and parts
│   ├── events.py, vocab.py #   run events; enums and references
│   ├── capabilities.py     #   Files, Scheduler, Thread handles, Agents
│   ├── wire.py             #   to and from JSON
│   ├── testing.py          #   FakeRun
│   └── scenario.py         #   Scenario: the real engine, in memory, on a fake clock
├── runtime/                # serving agents
│   ├── engine.py           #   the engine: threads, runs, leases, replay, sessions, schedules, the outbox
│   ├── api.py              #   the /v2 HTTP API
│   ├── gateway.py          #   what every protocol shares: auth, access, messages as the caller
│   ├── server.py, cli.py   #   processes and roles; the `llamphouse` command
│   ├── config.py           #   agent files, the runtime file, the component registry
│   ├── data_store/  run_queue/  event_queue/  file_store/  config_store/
│   ├── auth/  sessions/  triggers/  notifications/  adapters/  tracing/
│   ├── compass/            #   Compass: the API it reads, and the app (frontend/ sources, static/ built)
│   └── migrations/         #   Alembic revisions (branch v1: 1.x tables, branch v2: 2.0 tables)
└── helpers/                # optional modules on top of the contract: streams, langgraph
examples/                   # one folder per example, each with a README
docs/                       # this site (mkdocs)
tests/
```

Each component folder under `runtime/` is named after its key in the runtime file and has a `base.py` (the
interface) and one module per implementation. A new backend is a new module there and, if it should be
available by name, an entry in `REGISTRY` in `runtime/config.py`.

Two rules the code follows:

- **The SDK stays small and stable.** `llamphouse.sdk` imports nothing but pydantic and the standard library (a
  test guards that). Within a major version, fields, event kinds and part kinds are only added, never renamed or
  removed.
- **Mechanisms, not policies.** llamphouse provides runtime mechanisms. Which model or provider an agent uses,
  what a message means while a run is busy, who belongs to which tenant: those belong to the app or the agent,
  not to llamphouse.

## Running the tests

```bash
uv run pytest tests/unit/sdk tests/unit/runtime tests/contract/file_store     # the 2.0 tests, in memory
uv run pytest tests/unit/runtime/test_api.py -k stream                        # one file, some tests
```

The 2.0 tests run in memory by default and take seconds. Three environment variables run the same tests in
other modes:

| Variable | What it changes |
|---|---|
| `LLAMPHOUSE_TEST_POSTGRES=postgresql://user:pass@host:port/db` | tests that need Postgres run against this database (they are skipped without it): the store, the file store, the config store, migrations, `llamphouse migrate`, restarts, tracing. Use a database you can empty. |
| `LLAMPHOUSE_SCENARIO_STORE=postgres` | every `Scenario` uses a Postgres store (on `LLAMPHOUSE_TEST_POSTGRES`), emptied when it opens, instead of the in-memory one |
| `LLAMPHOUSE_SCENARIO_SPLIT=1` | every `Scenario` executes its runs in a second engine on the same store, queue and bus, as an API process and a worker process would |

```bash
export LLAMPHOUSE_TEST_POSTGRES=postgresql://llamphouse:llamphouse@localhost:5432/llamphouse_test
uv run pytest tests/unit/sdk tests/unit/runtime tests/contract/file_store
LLAMPHOUSE_SCENARIO_STORE=postgres uv run pytest tests/unit/sdk tests/unit/runtime
LLAMPHOUSE_SCENARIO_SPLIT=1 uv run pytest tests/unit/sdk tests/unit/runtime
LLAMPHOUSE_SCENARIO_STORE=postgres LLAMPHOUSE_SCENARIO_SPLIT=1 uv run pytest tests/unit/sdk tests/unit/runtime
```

`tests/unit/runtime/test_processes.py` starts real API and worker processes and also needs
`LLAMPHOUSE_TEST_REDIS` (a Redis URL). Any Postgres 14+ and Redis 7 work, installed locally or elsewhere.

New behaviour comes with a test. Most engine behaviour is easiest to test with `Scenario` (real agents on the
real engine, a fake clock, crashes you inject with `sc.crash(...)`); agent-level behaviour with `FakeRun`; HTTP
with `create_app(engine)` and `httpx.ASGITransport`; the OpenAI adapters with the official `openai` client on
that transport.

## Compass

The app's sources are in `llamphouse/llamphouse/runtime/compass/frontend` (Vue 3, Vite); the built app in
`runtime/compass/static` ships in the package.

```bash
cd llamphouse/llamphouse/runtime/compass/frontend
npm install
npm run dev        # http://localhost:5173, proxying /v2 to a `llamphouse up` on port 8000
npm run build      # type-check and build into ../static
```

Commit the built `static/` with your change to the sources.

## Migrations

Schema changes are Alembic revisions in `llamphouse/llamphouse/runtime/migrations/versions`, on branch `v2`
for 2.0 tables. Write them as plain SQL, idempotent (`if not exists`), and test them with
`LLAMPHOUSE_TEST_POSTGRES` (`tests/unit/runtime/test_migrations.py`). `llamphouse db upgrade --sql` prints what
a revision does.

## Examples and docs

- An example is a folder `examples/NN_Name` with a `README.md`, `agents.py`, `agents.yaml`, and where useful a
  `llamphouse.yaml` and a `client.py`. It must pass `llamphouse check --agents examples/NN_Name/agents.yaml`.
  Examples that need a model call it in the example's own code.
- `docs/examples.md` is generated from the examples' READMEs when the site builds (`hooks/sync_examples.py`).
- Code in the docs must match the real API: run it before you commit it.

## Pull requests

1. Branch from `main`.
2. Make the change, with tests, and update `CHANGELOG.md` under the unreleased version.
3. Run the 2.0 tests (in memory at least; with Postgres if you touched the store, migrations or files).
4. Open a pull request that says what changed and why.

## Reporting issues

Open an issue on GitHub with what you did, what you expected, what happened, and your Python and llamphouse
versions.
