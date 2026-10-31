# Migrating from 1.x

llamphouse 2.0 is a new agent contract (`llamphouse.sdk`) and a new runtime (`llamphouse.runtime`). Moving an
agent over is a rewrite to the new contract, not a flag: usually a small one, because the model calls and tools in
your agent stay the same. Conversations stored by 1.x can be brought over with `llamphouse migrate`, and 1.x
clients can keep talking to 2.0 agents through the Assistants adapter.

2.0 is an alpha (`2.0.0a0`): pin it by version.

## What changed

| 1.x | 2.0 |
|---|---|
| `llamphouse.core` | `llamphouse.sdk` (agent code) and `llamphouse.runtime` (serving) |
| an `Agent` instance with `id`, `name`, `description`, `version`, `tools` | an `Agent` subclass; its name comes from the agent file, its description from the docstring, its version from a `version` attribute |
| `LLAMPHouse(agents=[...], adapters=[...], data_store=...)` and `ignite()` | agent files, a runtime file, and `llamphouse up` |
| one process (or a worker class) | API and worker processes on Postgres + Redis, roles `api` / `worker` / `all` |
| an assistant's thread, runs started per message | everything is a thread with one agent and at most one active run; the app decides what a message means: append, start, or send to the active run |
| `/threads/{id}/runs` (Assistants-style API) | `/v2` (threads, messages, runs, streams); the Assistants API stays available as an adapter |
| `config_store` and config parameters | a typed `Config` per agent, values in the agent file, changes at runtime in the config store, per-thread config |
| `on_run_status` listeners | `engine.follow_runs`, `GET /v2/stream/runs`, and [subscriptions](guides/subscriptions.md) with an outbox |
| durable runs with `@step` in 1.x workers | every run is durable: leased, taken over after a crash, replaying its recorded steps |
| anonymous usage telemetry (`LLAMPHOUSE_TELEMETRY`) | none: 2.0 sends no usage data |
| tables `threads`, `messages`, `runs`, `run_steps` | tables `llamphouse_*`, on their own Alembic branch; the 1.x tables are left alone |

## Agent code

A 1.x agent:

```python
from llamphouse.core import Agent
from llamphouse.core.context import Context


class ChatAgent(Agent):
    async def run(self, context: Context):
        messages = [{"role": m.role, "content": m.text} for m in context.messages]
        result = await openai_client.chat.completions.create(messages=messages, model="gpt-5-mini")
        await context.insert_message(result.choices[0].message.content)


agent = ChatAgent(id="chat-agent", name="Chat Agent", description="A conversational assistant.")
```

The same agent in 2.0, streaming through the Responses API:

```python
from openai import AsyncOpenAI

from llamphouse.helpers.streams import openai, pipe
from llamphouse.sdk import Agent, Config

client = AsyncOpenAI()


class ChatConfig(Config):
    model: str = "gpt-5-mini"


class Chat(Agent[ChatConfig]):
    """A conversational assistant."""

    async def run(self, ctx) -> None:
        history = [{"role": "assistant" if ctx.session.is_own(m) else "user", "content": m.text}
                   for m in await ctx.session.messages()]
        stream = await client.responses.create(model=ctx.config.model, input=history, stream=True)
        await pipe(ctx, openai(stream))
```

```yaml
# agents.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: chat-agent}
spec:
  entrypoint: agents.py:Chat
  display_name: Chat Agent
```

What maps to what:

| 1.x (`context.`) | 2.0 (`ctx.`) |
|---|---|
| `messages` (with `role`) | `await ctx.session.messages()`; `ctx.session.is_own(m)` tells the agent's own messages (the model's `assistant` role) from the rest. A message has a `sender` (person, agent, service, trigger, system), not a role. |
| `insert_message(text)`, `reply(text)` | `await ctx.send(text, *parts)` |
| `send_chunk` / `asend_chunk`, `process_stream(stream, adapter)` | `async with ctx.stream() as s: await s.write(...)`, or `await pipe(ctx, openai(stream))` (also `anthropic`, `gemini`); see [Streaming](guides/streaming.md) |
| `pending_tool_calls` | `said.tool_calls` from `pipe` |
| `insert_tool_calls_step`, `submit_tool_outputs` | tools are your code; run them as steps (`ctx.step`); client tools with `ctx.client.call_tool`; see [Tool calls](guides/tool-calls.md) |
| `call_agent(agent_id, message)` | `thread = await ctx.call(agent, message)` then `await thread.wait()` / `ctx.next(thread)`; for your own work in parallel, `ctx.fork(...)` |
| `handover_to_agent(agent_id)` | `await ctx.handover(agent)` |
| `wait_for_input(prompt, schema=)` | `await ctx.ask(prompt, schema=)` |
| `wait_for_signal(name)` | `await ctx.run.wait_for_signal(key)` |
| `sleep(seconds)` | `await ctx.sleep(seconds)` (holds no worker) |
| `@step` / durable steps | `await ctx.step(name, fn, ...)` or `@step` from `llamphouse.sdk` |
| `get_config()` | `ctx.config` (typed) |
| `emit(name, data)`, `emit_state(...)`, `emit_state_delta(...)` | `await ctx.emit(Custom(kind="x-...", data=...))`, `Progress(...)`, `StateSnapshot(...)`, `StateDelta(...)` |
| `update_thread_details(...)` | thread labels are the app's (`metadata` when creating a thread); the agent keeps facts in `ctx.state` |
| `compact(...)`, session managers | session providers (`RollingWindow`, `Summarizing`, your own) in the agent file |

New in 2.0, with no 1.x counterpart: thread state (`ctx.state`), the inbox (`ctx.next()`: messages sent to a
running run), forks, typed outputs and workflows, per-thread config, triggers, the Scheduler capability. See the
[concepts](concepts/threads-and-runs.md).

## The server

1.x:

```python
llamphouse = LLAMPHouse(agents=[agent], data_store=PostgresDataStore(), adapters=[A2AAdapter()],
                        authenticator=KeyAuth("secret"))
llamphouse.ignite(host="0.0.0.0", port=8000)
```

2.0:

```yaml
# llamphouse.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Runtime
data_store:   {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}
file_store:   {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}
config_store: {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}
run_queue:    {redis: {url: "${LLAMPHOUSE_REDIS_URL}"}}
event_queue:  {redis: {url: "${LLAMPHOUSE_REDIS_URL}"}}
auth: [{api_key: {keys: ["${LLAMPHOUSE_API_KEY}"]}}]
adapters:
  - a2a: {prefix: /a2a}
  - assistants: {prefix: /v1}       # for clients of the 1.x Assistants-style API
```

```bash
llamphouse up --role api    --host 0.0.0.0 --config llamphouse.yaml --agents agents.yaml
llamphouse up --role worker --config llamphouse.yaml --agents agents.yaml
```

Or `llamphouse up --memory --agents agents.yaml` for one process in memory. See
[Configuration](configuration.md) and [Deployment](deployment.md).

Environment variables: 1.x's `DATABASE_URL` becomes `LLAMPHOUSE_DATABASE_URL` (`llamphouse db` and
`llamphouse migrate` still fall back to `DATABASE_URL`). Tracing reads `TRACING_STORE`, `CLICKHOUSE_URL` and the
`OTEL_*` variables as 1.x did; a runtime file's `tracing:` takes precedence.

## Clients

Two options:

- **Keep the client.** Mount the Assistants adapter (`adapters: [{assistants: {prefix: /v1}}]`). Clients built
  on the Assistants-style API (threads, messages, runs, streaming, `submit_tool_outputs`) keep working against
  2.0 agents. An assistant id is now an agent's name. A run's question shows as `requires_action` with
  `wait_for_input`, answered with `POST .../runs/{id}/signals` and `{"name": "input", "payload": ...}`. See
  [Adapters](guides/adapters.md#openai-assistants).
- **Move to `/v2`** (or to the OpenAI Responses API through the `responses` adapter). Create a thread, then
  append, start a run, or send to the active run. The API answers `409 thread_busy` instead of queueing a second
  run on a busy thread: what a message means then is your app's decision. See [HTTP API](http-api.md).

## Data: `llamphouse migrate`

`llamphouse migrate` writes a 1.x database's conversations as 2.0 threads, with the same thread and message ids
and times, so clients that kept ids carry on and agents see what was said before. The 1.x tables are only read,
and they stay where they are.

```bash
llamphouse migrate --config llamphouse.yaml --dry-run                        # what would be written
llamphouse migrate --config llamphouse.yaml --rename chat-agent=chat --agent chat
```

| Flag | |
|---|---|
| `--config` | the runtime file: its `data_store` is the 2.0 database written to |
| `--database-url` | the 2.0 database, instead (default as `llamphouse db`: the runtime file's, else `LLAMPHOUSE_DATABASE_URL` or `DATABASE_URL`) |
| `--from URL` | the 1.x database (default: the same database) |
| `--agent NAME` | the agent of threads that have no runs (default: those threads are skipped) |
| `--rename OLD=NEW` | a 1.x assistant id to a 2.0 agent name; repeatable |
| `--user-id ID` | who wrote the 1.x user messages (default `user`) |
| `--thread ID` | only these threads; repeatable |
| `--dashboards FILE` | 1.x Compass dashboards (`compass_dashboards.json`) |
| `--charts FILE` | 1.x Compass charts (`compass_charts.json`) |
| `--dry-run` | read and report, write nothing |

Database URLs may be SQLAlchemy-style (`postgresql+asyncpg://...`, as 1.x used) or plain `postgresql://`.

What it writes:

- **Threads** keep their id. Their 1.x metadata becomes the thread's labels; a name or `tool_resources` go in a
  first `x-llamphouse.v1.thread` event.
- **Messages** keep their id and time. `user` messages come from a person (`--user-id`), `assistant` messages
  from the agent. Text stays text; other content is kept as it was. The 1.x fields (`run_id`, `assistant_id`,
  `attachments`, `status`) go in the message's metadata under `llamphouse.v1`.
- **Runs** become one `x-llamphouse.v1.run` event each, at the time the run ended, holding its status, model,
  times, error, usage and steps. It is a record only: nothing runs again.
- **A thread's agent** is the assistant of its last run (mapped with `--rename`), else `--agent`; a thread with
  neither is skipped and listed. Threads whose runs were of several assistants are listed too.
- **Compass dashboards and charts** (`--dashboards`, `--charts`) become 2.0 Compass dashboards. Their SQL still
  reads the 1.x tables, which stay.

It is re-runnable. Stop the 1.x workers first, or migrate again later: a thread that was migrated gains only what
it doesn't have yet (new messages, a run's later status), and nothing is written twice. `--dry-run` also lists
runs still active in 1.x (queued, in progress, requiring action).

The schema is brought up to date first (unless `--dry-run`). In code, the same is `Engine.import_thread(id,
agent, entries, metadata=, created_at=)`, which writes a thread kept elsewhere with its own ids, times and history.

## A checklist

1. Rewrite each agent as a 2.0 `Agent` (or `Workflow`) class and test it with `llamphouse.sdk.testing.FakeRun`.
2. Write agent files (one `kind: Agent` per agent name your clients use) and a runtime file.
3. `llamphouse check --config llamphouse.yaml --agents agents.yaml`.
4. `llamphouse db upgrade` on the database (the 2.0 tables are added next to the 1.x ones).
5. Stop the 1.x workers, then `llamphouse migrate --dry-run`, then `llamphouse migrate`.
6. Start the 2.0 API and workers, with the `assistants` adapter if old clients remain.
7. Move clients to `/v2` or the Responses API when convenient.
