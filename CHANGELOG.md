# Changelog

## [2.0.0a0] - unreleased

A new agent contract (`llamphouse.sdk`) and a new runtime (`llamphouse.runtime`), next to the 1.x core
(`llamphouse.core`), which stays as it is until each piece is ported. An alpha: the API may still change.
Pin it by version (`llamphouse==2.0.0a0`) or commit.

### Added

- **The 2.0 SDK** (`llamphouse.sdk`) - everything is a thread with one agent, and at most one active run
  per thread; nothing starts a run by itself.
  - Primitives: `append` (a message on the thread), `start` (a run), `send` (to the active run's inbox).
    What a message means while a run is busy is the app's decision.
  - `RunContext`: `input`, `session` (`messages`, `add`, `revise`, `settled`), thread state as events
    (`ctx.state`: versions, `batch`, snapshots at any position), `send` / `stream`, `ask`, `@step`,
    `sleep`, forks (`ctx.fork`, `ctx.forks`), other agents (`ctx.call`), `handover`, `ctx.next`.
  - Typed outputs from `run()`'s return annotation, checked when the agent loads; `Workflow` with typed
    Input / Output; the description from the class docstring.
  - Session providers (`RollingWindow`, `Summarizing`) with revisions in the background.
  - Test kits: `FakeRun` (one run, scripted) and `Scenario` (agents on the real engine, in memory, on a
    fake clock, with crashes you inject).
- **The 2.0 runtime** (`llamphouse.runtime`):
  - Processes: `python -m llamphouse.runtime app:AGENTS --role api|worker|all` (or `runtime.serve`) -
    any number of API and worker processes on one Postgres and one Redis; `--memory` for one process.
  - Postgres is the truth: a conversation (a thread and every thread its runs create) is the unit of a
    transaction, under a lock on it; processes cache what they work on. Runs are leased by workers and
    taken over when a worker dies, replaying their recorded steps; a graceful stop hands them back.
  - `PostgresStore` (migrations on start), `PostgresFileStore`, Redis job queue and pub/sub; in-memory
    backends for tests and development.
  - HTTP API `/v2` with live streams (SSE), and a basic Compass at `/compass`.
- **Threads made for an app**: `engine.thread(agent, metadata=..., state=..., config=..., session=...)`
  and `POST /v2/threads` with the same fields:
  - `metadata` - the app's labels (who a thread is for); threads its runs create inherit them; listed
    and filtered by (`list_threads(metadata=...)`, `GET /v2/threads?metadata.org=acme`); agent code
    reads them as `ctx.thread.metadata`.
  - `state` - its initial `ctx.state`.
  - `config` - the agent's Config for runs on this thread, over the engine's, validated when the thread
    is made: one agent class, many instances. A fork inherits it.
  - `session` - what its first session starts from, e.g. the summary of a thread it continues.
- **Hearing what happens**:
  - `engine.follow(thread, after=position, kinds=...)` - a thread's events, stored first then live, from
    whichever process writes them; resume by position and nothing is missed.
  - `engine.follow_runs(agent=..., metadata=...)` - every run's status changes in the deployment, live
    (`RunChange`: run, thread, agent, status, error, output, parent, the thread's labels); over HTTP:
    `GET /v2/stream/runs` (SSE).
- **Long-lived threads**: any number of runs over a thread's life; ended runs are never encoded or
  compared again, so the cost of a new message doesn't grow with history.
- **Tracing in the 2.0 runtime** (`llamphouse.runtime.tracing`, on `llamphouse.core.tracing` and its tracing stores):
  - `Engine(..., tracing_store=...)` sets tracing up on `open()` (`setup_tracing`) and flushes on `close()`; `runtime.server` takes the store from the environment like 1.x (`TRACING_STORE`; off with `LLAMPHOUSE_TRACING_ENABLED=false`, or `tracing_store=None`) and shuts tracing down when the process ends.
  - Spans: `llamphouse.run` per execution of a run (`run.id`, `llamphouse.run_id`, `thread.id`, `llamphouse.agent`, `llamphouse.parent_run`, `llamphouse.run.status`, `llamphouse.run.error`), current while the agent runs, so spans agent code makes nest under it; a run executed again is a new span with the same `run.id`. `llamphouse.step` per `ctx.step` executed (`llamphouse.step.name`; none for a replayed step). `llamphouse.session.revise` per session revision (`llamphouse.session.id`, `thread.id`, `llamphouse.session.reason`).
  - `GET /v2/runs/{id}/trace` — the run's spans from the tracing store (`{"run", "tracing", "spans"}`).
  - Compass: a Trace panel in the run view - the spans as a tree (name, duration, status), a span's attributes, and `gen_ai.input.messages` / `gen_ai.output.messages` / `input.value` / `output.value` readable and foldable.
  - Without a tracing store or a tracer provider, nothing is imported and no span is made; tracing never fails a run.

- **Deployment files and the `llamphouse` command** (`llamphouse.runtime.config`, `llamphouse.runtime.cli`):
  - Agent files (`kind: AgentDefinition`, `kind: Agent`: entrypoint, config, session provider, env,
    secrets) and a runtime file (`kind: Runtime`: `data_store`, `run_queue`, `event_queue`, `file_store`,
    `tracing`, `sessions`, `engine`, `env`, `secrets`, `secrets_store`), `apiVersion: llamphouse.io/v1alpha1`.
  - Every component is swappable: `{name: {options}}` from the registry, or your own class as
    `{module:Class: {options}}`; options are checked against the constructor before anything starts;
    `${VAR}` / `${VAR:-default}`, and `{import: module:attr}` for Python objects.
  - `llamphouse up --role all|api|compass|worker`, `llamphouse check`, `llamphouse db upgrade|current`.
    `python -m llamphouse.runtime app:AGENTS` still works.
- **Tracing stores in the 2.0 runtime** (`llamphouse.runtime.tracing.stores`: in-memory, Postgres,
  ClickHouse), listing traces by 2.0's root spans and agent; with a tracer provider the app already set up,
  the store's exporter joins it instead of replacing it.
- **Packaging**: a small base install (pydantic, FastAPI, uvicorn, PyYAML) that runs `llamphouse up` in
  memory; extras `postgres`, `redis`, `tracing`, `all` (and `v1` for `llamphouse.core` while it remains); a
  missing backend says which extra to install.

- **Who calls, and what they may touch** (`llamphouse.runtime.auth`):
  - Authenticators, tried in order (runtime file `auth: [...]`): `api_key` (several keys, each with its own
    identity: subject, tenant, metadata) and `oidc` (JWT bearer tokens against the issuer's JWKS, with
    tenant and scope checks; `pip install 'llamphouse[oidc]'`), or your own `Auth`. None configured:
    everyone is `anonymous`.
  - The caller is the sender of its messages and the caller of its runs (`ctx.run.caller`); it names
    another sender only when its identity may act for others (`metadata.act_as`).
  - Access (`access:`): open by default; `tenant` keeps tenants apart - threads are stamped with their
    creator's tenant (`llamphouse.tenant`, inherited by threads runs create), and a caller sees and acts
    only on its own; a thread it may not touch answers 404. Files follow their thread; uploads their
    uploader. Your own policy as an `Access`.
- **Agent permissions** (an agent file's `permissions`): `agents` (whom it may call or hand over to; what
  `ctx.agents.list()` shows) and `files` (`none`, `read`, `read-write`), enforced by the runtime.
- **The config store** (`llamphouse.runtime.config_store`: in-memory, Postgres): an agent's config changed
  at runtime for every process, from their next run on (`engine.set_agent_config`, `GET` / `PUT
  /v2/agents/{name}/config` with `if_version`); a run's config is the file's, then the store's, then its
  thread's. Checked against the agent's Config before it is kept.

- **Adapters** (`llamphouse.runtime.adapters`, the runtime file's `adapters:`), each mapping its protocol
  onto threads, runs and messages - several instances, each with its `prefix` and `agents`:
  - `responses`: the OpenAI Responses and Conversations APIs (and `/models`). A conversation is a thread; a
    response a stretch of a run: when the run calls a client tool (function `tools`) or asks something, the
    response completes with `function_call` items or the question, and the next response answers it and
    continues the same run.
  - `assistants`: the OpenAI Assistants API (threads, messages, runs, streaming, `submit_tool_outputs`;
    a run's question as `requires_action` `wait_for_input`, answered on `/signals`).
  - `a2a`: A2A v1.0 over JSON-RPC (`SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`,
    `CancelTask`, `SubscribeToTask`), an agent card per agent; a context is a thread, a task a run.
  - `ag_ui`: AG-UI (`RunAgentInput` in, events out): frontend tools as client tools, questions and signal
    waits as interrupts answered by `resume`.
  - All behind the deployment's auth and access (`llamphouse.runtime.gateway`), tested with the official
    `openai` client for the OpenAI APIs.
- `ctx.client.tools`: the tools the run's client declared (adapters pass them in the run's request).
- `engine.bind(thread, agent)` (a thread that has had no run goes to another agent) and
  `engine.set_thread_metadata(thread, ...)`.

- **Triggers** (an agent file's `triggers:`, `llamphouse.runtime.triggers`):
  - `webhook`: POST to a path starts a run - a bearer secret, an HMAC signature of the body, or `public:
    true`; the payload mapped to the input's text, the thread (`thread: {label: $.order.id}`: one thread per
    order) and its labels; `idempotency: {key: $.event_id}` (the same event again: the first answer, 200);
    `busy: send | skip | reject` when the thread's run is busy.
  - `schedule`: `cron` (with `timezone`), `every`, or once `at`; on a labelled thread or a new one; kept by
    the store and fired once by whichever worker sees it due first; a fire missed while nothing ran happens
    once when one runs again.
  - `engine.fire(agent, message, label=, busy=)`: the primitive both use; a label's thread is made once,
    whichever process comes first.
- **The Scheduler capability**: `ctx.require(Scheduler).schedule(agent, at= | delay=, key=, **data)` - a
  run of that agent later, kept by the store; `cancel(key)`; recorded like a step.
- **External subscribers** (`llamphouse.runtime.notifications`): `POST /v2/subscriptions` (a URL, a filter -
  agent, thread, run, statuses, labels - an HMAC secret, headers); every matching run status change is
  written to an outbox in the same transaction and POSTed by the workers, again with backoff until 2xx,
  dead after `delivery_attempts` (retry over the API); scoped to what the subscriber may see. A2A push
  notification configs are subscriptions in the A2A format (the agent card now says push notifications).
- **Store primitives**: claims (a key taken once), schedules, subscriptions and the outbox - in memory and
  in Postgres (Alembic revisions).
- **One folder per component**, named after the runtime file's keys, each with `base.py` and one module
  per implementation: `data_store/`, `run_queue/`, `event_queue/`, `file_store/`, `config_store/`,
  `auth/`, `sessions/`, `tracing/stores/`, `triggers/`, `notifications/`, `adapters/`. The earlier module
  paths (`runtime.store`, `runtime.queue`, `runtime.bus`, `runtime.files`) still import.
- httpx is a base dependency (deliveries).
- **Compass 2.0** (`llamphouse.runtime.compass`, at `/compass`, its API at `/v2/compass`) - to see, debug
  and learn what agents do, behind the deployment's auth and access (a tenant sees its own):
  - **Overview**: threads, runs per hour by status, who is waiting for someone (asked a question, needs
    auth), what failed recently, per-agent runs and p50/p95 durations, this deployment's components.
  - **Why**: for any run, in plain words, why it is where it is - queued (the next ring), running (whose
    lease, until when), waiting (on which threads, a signal, its inbox, a time), asking (the question or the
    client tool), ended (with a handover, a cancel, the error) - and how many times it was executed.
  - **A run's story**: its input, what it said and asked, what was answered, state it changed, threads and
    runs it started, messages sent to it and read; its steps (recorded, replayed), what it waits on, the
    tree of runs from the root (forks, calls), the session it saw (revisions), its trace, the raw record.
  - **Threads**: the conversation live, the timeline, the state at any position, the tree of threads, its
    runs; a composer offering exactly the primitives the thread's state allows (append / start, or send /
    answer / signal / cancel).
  - **Agents**: try one (a form from a workflow's input schema), its config (the effective config, what
    differs from the file, save or reset), its triggers (a webhook's curl, schedules with "fire now"), its
    schemas, permissions and session provider.
  - Runs (filter and compare two), traces, triggers, subscriptions (deliveries, retry), dashboards (charts
    on read-only SQL, Postgres only, admins only), a jump-to-id box (thread, run, response, session), dark
    theme, and **Learn** - each idea of 2.0 with code and a link to it in your own data.
  - Runs record `created_at` / `ended_at`; dashboards and charts are kept as store documents
    (`llamphouse_documents`, Alembic revision `c2e5a6b7d8f9`).
  - The app is built into the package (`runtime/compass/static`; sources in `frontend/`): no CDN, no
    build at runtime.
- The wheel no longer carries app sources, `node_modules` or bytecode (50 MB -> 0.7 MB).

### Changed (before 2.0.0a0 was released)

- A thread's `config` keeps only the values given (checked whole), so the file's and the store's still
  apply under it.
- Against loops, `Engine(max_runs_per_root=1000, max_depth=50)`: the runs one run started by the app may
  lead to, and how deep. There is no limit on runs per thread any more (`max_runs_per_thread`: a cap
  only if set); `max_runs_per_tree` still works, as the old name of `max_runs_per_root`.
- Stored records read back with defaults for fields added later: data written by an earlier 2.0 commit
  keeps loading.
- The Postgres schema is under **Alembic** (`llamphouse.runtime.migrations`, shipped in the package), in
  the same history as 1.x: the 1.x revisions are branch `v1`, the 2.0 tables branch `v2`. A new database
  gets only the 2.0 tables; a 1.x database keeps its tables and gains them; one set up by an earlier 2.0
  commit is adopted. Processes upgrade on start (`PostgresStore(migrate=False)` to make it a deploy step:
  `llamphouse db upgrade`, `--sql` to review). The repository's `migrations/` moved there.

### Migrating from 1.x

1.x keeps working: `llamphouse.core` is unchanged in this release, and 2.0 uses its own tables
(`llamphouse_*`). Moving an agent over is a rewrite to the new contract, not a flag:

- An assistant becomes an `Agent` (or `Workflow`) with `async def run(self, ctx)`; `context.reply` /
  `insert_message` become `ctx.send`; `call_agent` becomes `ctx.call` (another agent's thread) or
  `ctx.fork` (your own work in parallel); `handover_to_agent` becomes `ctx.handover`.
- The Assistants-style API (`/assistants`, `/threads/{id}/runs`) becomes `/v2`: create a thread,
  then append, start a run, or send to the active run - and decide in the app what a message means
  while a run is busy.
- Data is not migrated yet: 1.x threads and runs stay in their tables; a `llamphouse migrate` follows.
- See `docs/llamphouse 2.0 — design notes.md` and `examples/v2`.

## [1.4.0] - TBD

### Planned

- **Durable runs, what remains** — releasing leases on graceful shutdown, Postgres `NOTIFY` wake-ups, fencing step writes by `lease_epoch`, replaying `call_agent`. See `docs/DURABLE_RUNS_PLAN.md` §0 (P5).
- **Lifecycle events & subscribers** — transactional outbox, `subscribers = [...]` / `@on_event`, retry and dead-letter; also notifies humans when a run waits for input. See `docs/PLAN_LIFECYCLE_EVENTS.md`.
- **One active run per thread** — a rule (reject, queue or interrupt) for a new run on a thread that already has one, so retries, resumed runs and sessions never write to a thread at the same time.
- **Run policies** — retries and backoff, step and run time outs, and token budgets, configurable per agent in `llamphouse.yaml`.
- **AG-UI frontend tools** — calling the tools an AG-UI client declares (`tools`), and taking their results from the next input's tool messages.
- **Compass** — durable run timeline (steps, retries, replayed vs re-executed), a human-in-the-loop inbox to approve or reject waiting runs, and a session view (what the agent saw next to the full thread).
- **Upgrade guide** for the 1.4.0 migrations.

### Added

- **Durable runs** (`docs/concepts/durable-runs.md`) — a run survives a worker crash or a server restart, waits for hours without holding a worker, and never does twice what it already did:
  - **Replay** — a run that runs again (resumed, retried, taken over) runs `Agent.run` from the top, and every `@step`, `reply` / `insert_message`, `sleep` and wait it already completed returns what it returned then. Calls are recorded under a call key (`name#n`) in the run step's `metadata.call_key`.
  - **Leases** — a worker claims a run before running it (`BaseDataStore.claim_run`, atomic) and its heartbeat renews the lease (`renew_lease`); a worker that lost its lease stops the run (`LeaseLost`). `AsyncWorker` / `DistributedWorker` take `worker_id`, `lease_ttl` and `recover_interval`.
  - **Crash recovery** — a `RunRecovery` loop in every worker finds runs whose lease expired: a durable agent's run (`Agent.durable = True`) is queued again and resumes; any other agent's run fails. It also wakes waiting runs that are due.
  - **`context.sleep(seconds, until=)`** — parks the run (`requires_action`, `wait_until`) holding no worker; the run queue's `schedule_at` wakes it.
  - **`context.wait_for_signal(name, correlation_id=, timeout=, request=)`** and **`context.wait_for_input(prompt, schema=)`** (human in the loop) — park the run until a signal is delivered; a signal sent early is kept; `timeout` raises `TimeoutError` in the agent.
  - **Signals** — `POST /threads/{thread_id}/runs/{run_id}/signals` (Assistants API), `signal_run(...)`, and over **A2A**: a waiting task is `input-required` with the question (and schema) in its status message, and a reply carrying its `taskId` answers it.
  - **Retries** — `@step(retries=, backoff=)` retries a step in place; a run retried by the queue's `RetryPolicy` goes back to `queued` and carries on from the step that failed; `POST /threads/{thread_id}/runs/{run_id}/retry` (`retry_run(...)`) does that for a failed, expired or cancelled run.
  - `BaseDataStore` durable-runs methods (`claim_run`, `renew_lease`, `release_lease`, `suspend_run`, `list_runs_to_recover`, `list_steps_for_replay`, `deliver_signal`, `consume_signal`, `supports_durable_runs`) — implemented by `InMemoryDataStore` and `PostgresDataStore`, with contract tests; a custom store keeps working as before.
  - Migration `c3d8e1f2a4b5`: `runs.worker_id`, `lease_expires_at`, `lease_epoch`, `wake_at`, and a `signals` table.
- **Durable LangGraph runs** — `LangGraphAgent` runs the graph with a checkpointer keyed by the run (`checkpointer="llamphouse"` by default: `LLAMPHouseCheckpointSaver` in the data store; or your own LangGraph checkpointer; or `None`). A node's `interrupt()` parks the run like `wait_for_input` (A2A `input-required`, `/signals`) and the answer resumes the graph inside that node; a retried or taken-over run carries on from the last checkpoint. `checkpoint_scope="thread"` keeps the graph state across runs. Example 15 gains a refund branch with human approval.
- **Key-value records** — `BaseDataStore.kv_put` / `kv_get_all` / `kv_delete`: small JSON records by namespace, deleted with their thread (migration `d4e9f2a3b5c6`, `kv_entries`). Used for LangGraph checkpoints.
- `context.wait_for_signal(..., key=)` — a stable identity for a wait whose position may differ between attempts.
- **Sessions** (`docs/concepts/sessions.md`) — `context.session` is an agent's view of a thread for a run: the messages in view and a summary of older ones (`to_messages()` for a chat-completion API), decided by the agent's `session_manager`:
  - **Built in:** `FullHistory()` (the whole thread); `SlidingWindow(max_messages=, max_tokens=, keep_first=)` (the latest messages, optionally the first ones too); `Summarizing(summarizer, max_messages=, keep_messages=, summary_tokens=)` (the latest messages, and a running summary of the older ones, kept per thread and agent so each compaction folds only what is new); `Compacting(summarizer, max_tokens=, compact_at=0.9, compact_to=0.25, summary_tokens=, reserve_tokens=)` (a token budget - summary and messages, less a reserve for the prompt and answer - that fills up; at 90% everything but the latest messages is folded into a new summary, down to 25%, and it fills up again). All take `count_tokens=` and `max_message_tokens=` (shorten a long message in the view only).
  - **Summarizers:** `llm_summarizer(complete)` from any LLM call (`prompt -> text`, sync or async), `openai_summarizer(model, client=)` on an OpenAI-compatible API; any `async (previous_summary, messages) -> str` works, and one that takes `max_tokens` is told the summary's target length (`summary_tokens`). A failing summarizer leaves the session uncompacted for the run (a warning), instead of failing it.
  - **Your own:** subclass `SessionManager` and implement `build(messages, memory) -> Session`; the memory (a JSON dict per thread and agent) changes when `Session.memory` is set; override `messages()` to read less of the thread (`thread_messages(context, after=, latest=, first=)`).
  - **Default: a rolling window.** Every agent has a session: an agent that sets no manager gets the server's default — `LLAMPHouse(session_manager=...)` / `DistributedWorker(session_manager=...)`, or `session_manager:` at the top of `llamphouse.yaml` — else `SlidingWindow(max_messages=20)`. An agent's entry in `llamphouse.yaml` (`agents[].session_manager`) overrides its own; YAML names `full_history`, `sliding_window`, `summarizing`, `compacting`, with `summarizer: openai` / `{openai: {model: ...}}` / `file.py:name`. `llamphouse check` validates them.
  - **Mid-run compaction:** `context.compact(messages)` keeps an agent's own working transcript (a coding agent's tool loop) within its `Compacting` budget: leading system messages kept, the latest that fit kept, tool calls never split from their results (OpenAI and Anthropic shapes), the rest one summary message. Recorded and replayed as a `session.compact_messages` step.
  - **Compaction events:** `session.compaction.started` / `completed` / `failed` (scope `thread` or `run`, tokens before and after) on the run's stream - A2A status messages, AG-UI `CUSTOM` events.
  - A run's session loads under its heartbeat and time out (`Context.create(load_session=False)` + `load_session()` in the workers), so a slow summarizer does not lose the run its lease.
  - A session whose memory changed is a `session.compact` step, replayed by a run that runs again (no second summarizer call). `LangGraphAgent` passes the session's `messages` and `summary` to the graph.
- **Sessions example** (`examples/17_Sessions`) — four agents, four managers: `Summarizing`, `SlidingWindow`, `Compacting`, and a manager of its own.
- **AG-UI adapter** (`AGUIAdapter`, YAML `ag_ui`, `pip install llamphouse[ag-ui]`) — serves the agents over AG-UI 1.0 (`POST /ag-ui`, `POST /ag-ui/agents/{agent_id}`): replies and streamed text as `TEXT_MESSAGE_*`, `@step`s as `STEP_*`, tool calls as `TOOL_CALL_*`, `context.emit_state()` / `emit_state_delta()` as `STATE_SNAPSHOT` / `STATE_DELTA`, `context.emit()` as `CUSTOM`. A run waiting for input (`wait_for_input`, `wait_for_signal`, LangGraph `interrupt()`) finishes with an `interrupt` outcome carrying the prompt and response schema; the next input's `resume` entries answer it and the same run carries on (or `cancelled` cancels it). The AG-UI thread and message ids are the thread's; the input's state, context, tools and forwarded props reach the agent through `ag_ui_input(context)`.
- **`context.emit_state(state)` / `context.emit_state_delta(patch)`** — share the agent's state with the client.
- **`context.replaying`** — while a run redoes what an earlier attempt did, the events it sends (`send_chunk`, `emit`, `emit_state`) are not sent again.
- **AG-UI example** (`examples/18_AGUI`) — a trip planner with a web front-end (Vue 3 + Vite on `@ag-ui/client`): streamed chat, an approval form built from the interrupt's schema, the shared state, the run's steps; and a terminal client. The example clients read the server address from `LLAMPHOUSE_URL`.
- **DurableRuns example** (`examples/16_DurableRuns`) — a refund agent that asks a human over A2A, sleeps through a cool-off and retries a flaky payment API, surviving a restart.
- **Run lifecycle, the same for every worker** (`workers/run_lifecycle.py`, used by `AsyncWorker` and `DistributedWorker`):
  - a run cancelled while queued is not started: it ends `cancelled`;
  - a running run records a heartbeat (`BaseDataStore.heartbeat_run`) every few seconds — a run whose heartbeat stopped had its worker die;
  - the same beat reads the status back: a cancel requested meanwhile, from any process, stops the run and it ends `cancelled`.
- **Cancel running runs** — `BaseDataStore.request_cancel` moves `queued` → `cancelled` and running → `cancelling`. The cancel route now accepts running runs too (as OpenAI's Assistants API does) and refuses only finished ones.
- **Run status listeners** — `BaseDataStore.add_run_status_listener` and `LLAMPHouse(on_run_status=...)`, notified on every status write of every store with the status before; a failing listener never breaks a write.
- **`context.runtime`** — the data store, run queue, agents and event queues a run needs to start other runs, wherever it executes, instead of reaching for private attributes.
- `runs.heartbeat_at` column (migration `b7e2c9a4d1f3`, which also joins the two migration heads into one).

### Changed

- `LangGraphAgent`'s default state mapper passes the session's messages - by default the latest 20 - instead of `context.messages` (the latest 100); `session_manager=FullHistory()` passes the whole thread.

- A run waiting for a signal is cancelled at once by `request_cancel` (it has no worker to stop it); `submit_tool_outputs` refuses it.
- A2A `tasks/get` and `tasks/cancel` find a task's run after a restart too; `tasks/cancel` now stops running tasks as well (`request_cancel`).
- Running runs left `in_progress` by a worker that is gone (from before 1.4.0 too) are failed by the recovery loop once a 1.4.0 worker starts.

### Fixed

- `context.messages` holds a thread's **latest** 100 messages (oldest first); it held the first 100, so an agent on a long thread never saw what was said last. It also no longer prints "No messages found" for an empty thread.
- `LangGraphAgent` with checkpoints runs the graph once per attempt (without them, a stream with no text still falls back to `astream` and `ainvoke`, running the graph again).
- `InMemoryQueue.dequeue(timeout=...)` no longer returns empty when a scheduled message comes due while it waits.
- `PostgresDataStore.update_run_status` stores `expires_at` for expired runs (it set a non-existent `expired_at`), and no longer turns an error dict that has a `code` into a string.
- `AsyncWorker` keeps references to its run tasks, so none is garbage-collected mid-run.
- The ignite banner no longer raises `AttributeError` when `event_queue_class` is a factory instance (e.g. `RedisEventQueueFactory`), so distributed mode starts again.

### Known issues

- Streamed text (`send_chunk`, `asend_chunk`, `process_stream`) is not stored as a message of the thread unless the agent also stores it (`reply` / `insert_message`): a client that reloads the thread does not see it, and an AG-UI client's next input carries it as an assistant message the thread does not have.

- A signal delivered while its run is about to park (between `wait_for_signal` finding none and the run being parked) wakes the run only at the next recovery look, up to `recover_interval` (10 s) later.
- `call_agent` / `handover_to_agent` are not replayed: a resumed run calls its sub-agent again.
- Graceful shutdown does not release the leases of running runs: another worker takes them over only once they expire (`lease_ttl`, 30 s).
- Step writes are not fenced by `lease_epoch`: a worker that lost its lease mid-write (network partition) can still write until its next heartbeat stops it.
- `LangGraphAgent` without checkpoints (`checkpointer=None`, or a store without key-value records) still runs the graph again through `astream` and `ainvoke` when a stream yields no text.
- A2A `message/stream` forwards streamed text (`send_chunk`) as artifacts, but not a reply inserted at once (`context.reply`): the stream ends `completed` with an empty artifact; `message/send` and `tasks/get` do return it.

## [1.3.0] - 20/08/2026

### Added

- **Pluggable tracing stores** — new `BaseTracingStore` interface with `InMemoryTracingStore`, `PostgresTracingStore`, and `ClickHouseTracingStore` implementations for persisting and querying span data.
- **Trigger handling** — `BaseTrigger` and `WebhookTrigger` for starting an agent run from an external system (e.g. an HTTP POST). Available on `context.trigger`.
- **`llamphouse.yaml` config** — declarative server config loaded by `llamphouse.cli.config.loader` and validated by Pydantic models in `llamphouse.cli.config.schema`. CLI now reads it on startup.
- **CLI reorganisation** — moved entrypoints into the `llamphouse.cli` package.
- **Internal span exporter from tracing store** — tracing setup can wire the configured tracing store as a span exporter without external OTLP plumbing.
- **Compass dashboards** — new dashboards UI with `DashboardsView`, `DashboardView`, and `DashboardPresentView`, backed by `DashboardStore` and `ChartStore`. Includes a `ChartWidget` for visualising tracing/run data and a `FilterBuilder` for composing queries.
- **Runs view** — dedicated `RunsView` in Compass for browsing and inspecting runs.
- **Table preferences** — `useTablePrefs` composable for persisting column/sort/filter state in Compass data tables.
- **WebhookTrigger example** (`examples/11_WebhookTrigger`) — end-to-end demonstration of triggering an agent run from an external HTTP POST.
- **PlannerAgent example** (`examples/12_PlannerAgent`) — multi-node planner/executor/synthesizer agent with tracing wired in.
- **`llamphouse.yaml` config** — declarative server config loaded by `llamphouse.cli.config.loader` and validated by Pydantic models in `llamphouse.cli.config.schema`. CLI now reads it on startup.
- **Strict `llamphouse.yaml` validation** — unknown fields now fail fast, `${ENV_VAR}` references are resolved before validation, and missing environment variables produce user-friendly errors.
- **YAML-configured data stores** — `data_store` can now be configured from `llamphouse.yaml`; `data_store.postgres` reads `DATABASE_URL` from the environment.
- **YAML-configured webhook triggers** — deployment-level `triggers` can now define `WebhookTrigger` routes from `llamphouse.yaml`.
- **`llamphouse check` Health Check** — new preflight command for validating `llamphouse.yaml` without starting the server. It checks schema, component imports, entrypoints, route conflicts, data-store connectivity, tracing-store connectivity, and exits non-zero on failure for CI.
- **Health check output modes** — `llamphouse check` supports `--format text|json`, `--verbose`, and `--timeout` for external dependency checks.
- **CLI reorganisation** — moved entrypoints into the `llamphouse.cli` package.
- **Internal span exporter from tracing store** — tracing setup can wire the configured tracing store as a span exporter without external OTLP plumbing.
- **LLM usage metadata on streaming spans** — streaming adapters now forward provider response ids, actual response models, and provider-supplied token breakdowns to span attributes for observability and downstream cost analysis without calculating cost in LLAMPHouse.
- **MkDocs landing page hook** — `hooks/landing_page.py` generates a custom `site/index.html` after each build.
- **Examples sync hook** — `hooks/sync_examples.py` keeps `docs/examples.md` in sync with the `examples/` directory, including a structured table and progression guide.
- **WebhookTrigger example** (`examples/11_WebhookTrigger`) — end-to-end demonstration of triggering an agent run from an external HTTP POST.
- **PlannerAgent example** (`examples/12_PlannerAgent`) — multi-node planner/executor/synthesizer agent with tracing wired in.
- **LLAMPHouseYAML example** (`examples/13_LLAMPHouseYAML`) — declarative server setup driven by `llamphouse.yaml`.
- **Workflow `@step` decorator** — new `llamphouse.core.workflow.step` decorator that records step input/output and step lifecycle within a run.
- **`step` run-step type** — `run_step_type_enum` now includes `step` with matching Alembic migration support.
- **WorkflowSteps example** (`examples/14_WorkflowSteps`) — end-to-end demonstration of `@step` usage.
- **WebhookTrigger idempotency / dedupe** — webhook triggers can opt into atomic inbound command idempotency by mapping a request payload field as the idempotency key. Duplicate requests with the same semantic fingerprint replay the original response without inserting another user message or enqueueing another run; the same key with a different fingerprint returns `409 Conflict`.
- **WebhookTrigger inbound messages and thread selection** — webhook triggers can map a JSON payload field into an inbound user message and can continue an existing thread from a mapped `thread_id`, falling back to a new thread when no thread id is provided.
- **WebhookTrigger metadata mapping** — webhook request payload fields can be mapped into thread metadata and run metadata. Internal webhook metadata is stored under reserved `__webhook_*` run metadata keys.
- Database migration for runs now combines `stream`, `provider_config`, and float timestamp columns into the 1.3.0 migration chain.
- **Data-store list / count contract extended** — `BaseDataStore` now declares `list_threads`, `list_all_runs`, `get_run_any_thread`, `list_runs_by_parent_ids`, `get_first_run_assistant_ids`, `count_threads`, `count_runs`, and `count_messages`. Both `InMemoryDataStore` and `PostgresDataStore` implement them. These were previously missing or in-memory-only, which silently returned empty results on Postgres in Compass for Threads, Runs, the flow tree, and the home-page stats.
- **Server-side pagination on Compass list endpoints** — `/api/threads` and `/api/runs` now accept `limit` / `order` / `after` / `before` cursor params and return `{ data, first_id, last_id, has_more, total }`. Threads and Runs views render Prev / Next page controls; the cursor stack and active filters live in the URL so Back from a detail page restores the exact same state.
- **Server-side filtering** — shared `_filters` helper translates a Compass filter condition into either a SQLAlchemy clause or a Python predicate. Each list endpoint declares a filterable-field allowlist (`Thread`: `id`, `agent_id`, `created_at`, `metadata`; `Run`: `id`, `agent_id`, `thread_id`, `status`, `created_at`). `agent_id` on threads is resolved via a Postgres `EXISTS` subquery.
- **Reusable Compass `FilterBuilder`** — draft / applied state with explicit Apply, Reset, and Clear buttons; "N active" / "unsaved changes" indicators; quick-add chips per field. Views are unaffected until Apply is pressed.
- **Hard server-side cap** on list endpoints (`_MAX_PAGE_SIZE = 200`) so stale clients can't request 10 000-row pages.
- **`include_total=false` opt-out** on `list_threads` / `list_all_runs` (and the matching Compass routes), so views that only need a top-N can skip the `COUNT(*)` query. Compass Overview uses it.
- **Compass Overview rewrite** — stats / threads / runs render progressively with per-section spinners as their requests resolve; the old serial N+1 (`for thread in threads: listRuns(thread.id)`) is replaced by a single `listAllRuns` call. Three home-page counts run concurrently via `asyncio.gather`.
- **Progressive loading in Compass Run Detail view** — per-section loading flags (`run`, `steps`, `config`, `spans`, `flow`, `messages`); each tab renders its data when it lands, with inline tab-label spinners.
- **Flow tab always visible** with an explicit "No agent flow for this run" empty state instead of being hidden.
- **Bounded flow-tree walk** — flow route now walks up from a run via `get_run_any_thread` (depth-capped) and BFS-down via `list_runs_by_parent_ids` (node-capped at 5000). Replaces the unbounded "load all runs and scan in Python" implementation that silently dropped ancestors when the parent was older than the recent-runs window.
- **Synced Compass home-page example** (`examples/00_sync/server.py`) — minimal `HelloAgent` backed by `PostgresDataStore` with both A2A and Compass adapters mounted, loading `DATABASE_URL` from `.env`.
- **Plan: lifecycle events & subscribers** — `docs/PLAN_LIFECYCLE_EVENTS.md` for the upcoming Trigger / Event / Subscriber model.
- **Plan: Compass dev focus** — `docs/PLAN_COMPASS_DEV_FOCUS.md` for Playground, Replay, Scores, Datasets, SQL editor, editable Overview, and webhook actions.
- **Compass trace inspector overhaul** — Span inspector redesigned with richer chip metadata, workflow topology hints, dedicated task input/output cards, and improved readability for large span payloads.
- **Structured LLM message rendering in traces** — span attributes like `gen_ai.input.messages`, `gen_ai.output.messages`, and `gen_ai.system_instructions` are now parsed and displayed as conversation-style message blocks.
- **Reusable content-format viewer in Compass** — trace payload sections now support Plain / Markdown / JSON display toggles via a shared `ContentBlock` component.
- **Trace detail run/thread deep-links** — Trace Detail now surfaces contextual "Open Run" and "Open Thread" actions when linked IDs are available in span attributes, while still handling standalone traces gracefully.
- **Compass sidebar footer info endpoint** — added `/api/info` to expose package version + website (`https://llamp.ai`) and wired sidebar footer to display `LLAMPHouse vX.Y.Z` dynamically.

### Changed

- Orchestrator and Planner example agents now use `InMemoryTracingStore` by default.
- `SpanTree` truncation logic improved for cleaner display of long spans.
- Expanded `DataTable`, `MessageBubble`, `RunDetailView`, `ThreadDetailView`, and `ThreadsView` with richer rendering and interaction.
- `ConfigStore` example now ships with sample `compass_charts.json` and `compass_dashboards.json`.
- `in_memory_store` and `postgres_store` updated to align with the new tracing/trigger flows and model changes.
- Logging during startup now surfaces more detailed information about the application state.
- `RunDetailView` in Compass expanded to render workflow step information alongside existing run details.
- `context.py`, `in_memory_store`, `postgres_store`, and `types/run_step` updated to support `@step` integration and the new `step` run-step type.
- **Ignite banner reorganised** into `Adapters` / `Triggers` / `Agents` / `Infrastructure` / `Optional features` sections; webhook trigger routes are listed inline (`▸ WebhookTrigger    /triggers/report → report-agent`).
- **Route-conflict warnings on boot** — duplicate webhook trigger paths, or trigger paths that fall under a non-root adapter prefix, now log a warning at startup.
- **Compass flow rendering optimised** — edge geometry (`path`, `midX`, `midY`, colour, dash, marker) is pre-computed once inside `flowLayout` and bound directly in the template instead of being recomputed per-edge per-render. Roughly `O(E·N) → O(E + N)` per re-render.
- **Run-detail I/O resolver tolerates missing `run_id`** — assistant messages without a stamped `run_id` now match the run's `started_at..completed_at` window. Messages route also re-introduces `run_id` / `assistant_id` as explicit `null` (the prior `exclude_none=True` serialiser was stripping them entirely).
- **`PostgresDataStore.close()` is bounded** — `engine.dispose()` is wrapped in `asyncio.wait_for(..., timeout=5.0)` so a hung asyncpg socket can't block server shutdown for the OS TCP timeout.
- **Compass adapter no longer relies on missing methods.** All `hasattr(db, "…")` branches that masked data-store API gaps (`list_threads`, `list_all_runs`, `count_threads/runs/messages`, `get_run_any_thread`) are gone, replaced by abstract methods on `BaseDataStore`. The only `hasattr` left is the legitimate backend dispatch in the dashboard SQL endpoint.
- **Compass sidebar navigation updated** — default navigation now orders key views as Agents → Runs → Threads → Traces → Dashboards (after Overview), and removes Compare from the sidebar menu while keeping the API route available.
- **Trace status derivation improved** — span status UI now falls back to `gen_ai.task.status` (e.g. `success`, `failed`) when OTel `StatusCode` is unset, improving status accuracy for instrumentations that only emit GenAI task status.
- **Resource metadata surfaced in trace UI** — service version and deployment environment from OTel resource attributes are now merged into span attributes and rendered as inspector chips (e.g. Env, Version, Service).

### Fixed

- Compass home page no longer shows `0` for Agents / Threads / Runs against `PostgresDataStore` (missing count methods).
- Compass Threads tab no longer renders empty against `PostgresDataStore` (missing `list_threads`).
- Compass Runs tab no longer renders empty against `PostgresDataStore` (missing `list_all_runs`).
- Compass agent-flow view no longer silently truncates the tree when a parent run is older than the recent-runs window (the BFS used to bail out at the first missing ancestor).
- Compass Run Detail Input/Output panel no longer appears empty for messages produced by `context.insert_message(...)` that weren't stamped with a `run_id`.
- `SAWarning: garbage collector is trying to clean up non-checked-in connection` on Compass thread listings, caused by an N+1 burst of sessions for per-thread agent enrichment (replaced with a single `SELECT DISTINCT ON (thread_id) thread_id, assistant_id` query via the new `get_first_run_assistant_ids`).
- `SyntaxWarning: invalid escape sequence '\s'` in `PostgresDataStore` docstring.

### Deprecated

- `AssistantAPIAdapter` — superseded by `A2AAdapter`. Using it emits a `DeprecationWarning`.

### Removed

- `DashboardAdapter` and its API routes / static files. Dashboard functionality is now served exclusively via the Compass adapter.

## [1.2.4] - 3/07/2026

### Added

- Added core data-store contract methods for run lookup, operational listing, and aggregate counts.
- Added data-store contract coverage for run `stream`, `provider_config`, lifecycle timestamps, `usage`, and operational APIs.
- Added Compass compare integration coverage for loading runs by run id across data-store backends.
- Added Postgres connection pool starvation contract tests covering pool exhaustion, concurrent request handling, and connection cleanup.
- Added unit tests for `AsyncWorker` and `DistributedWorker` covering happy path (sync/async agents), streaming event sequences, assistant/run not found, timeout, generic exceptions, queue errors, and concurrency/semaphore behaviour (52 tests).
- Added unit tests for the streaming layer: `StreamingEmitter` wire format and full pipeline, A2A SSE route event shapes, and the `_INTERNAL_TO_OPENAI_EVENT` mapping (83 tests across 3 files).

### Changed

- Standardized run storage behavior across `InMemoryDataStore` and `PostgresDataStore`.
- Updated Compass and Dashboard routes to use public data-store APIs instead of in-memory private structures.
- Squashed `provider_config` into the existing `runs.stream` migration for a clean migration chain.

### Fixed

- Fixed Postgres-backed runs missing persisted `stream` data.
- Fixed run status updates so lifecycle timestamps and `usage` are persisted consistently.
- Fixed `LLAMPHouse` initialization with callable event queue factories such as `RedisEventQueueFactory`.
- Fixed `StreamingEmitter._on_text_snapshot` to guard against empty `full_text` before calling `_ensure_message_started()`, preventing spurious empty message delta events during streaming.

### Removed

- Removed dead `core/_utils` module (`get_max_db_connections`); the function was no longer called anywhere after the pool-size validation was dropped in 1.2.1.



## [1.2.3] - 11/05/2026

### Added

- **Anonymous usage telemetry** — opt-out singleton client (`llamphouse.core.telemetry`) batches lifecycle, usage, and runtime events to `https://api.llamp.ai/telemetry` over a daemon worker thread. Three tiers: `usage` (default), `lifecycle`, `off`. Configure via `LLAMPHOUSE_TELEMETRY` or programmatically.

### Privacy

- No payloads, prompts, or tool arguments are ever transmitted — only event names, counts, and durations.
- Disable entirely with `LLAMPHOUSE_TELEMETRY=off`.

## [1.2.2] - 19/03/2026

- Fix double logging LLAMPHouse events.
- Fix missing metadata property in messages within the in_memory_store.

## [1.2.1] - 19/03/2026

### Changed

- **Async Postgres data store** — rewrote `PostgresDataStore` to use SQLAlchemy's async engine (`create_async_engine`, `AsyncSession`, `async_sessionmaker`). All database I/O is now non-blocking, eliminating event-loop stalls in async environments.
- **Explicit configuration** — `PostgresDataStore` now takes `database_url` and `pool_size` as constructor arguments instead of reading `DATABASE_URL` / `POOL_SIZE` from environment variables at module level. No more hidden `load_dotenv()` side-effect on import.
- **Removed pool-size validation** — dropped the startup `SHOW max_connections` check and the `get_max_db_connections` utility. Pool sizing is now the caller's responsibility.
- **Bumped SQLAlchemy minimum** to `>=2.0.0` (required for `async_sessionmaker`).
- **Added `asyncpg`** driver dependency (`>=0.29.0,<1`).
- `list_run_steps` and `get_run_step_by_id` are now properly `async`.
- Removed duplicate `purge_expired` method.
- `close()` is now an async method (`await store.close()`) that disposes the engine cleanly.
- Add missing database migrations version.
- Updated logging in llamphouse to allow propagation to root logger for unified output, and set uvicorn access logger to not propagate to avoid duplicate logs.

## [1.2.0] - 16/03/2026

### Added

- **Pluggable adapter architecture** — the API layer is now built on `BaseAPIAdapter`. The OpenAI-compatible routes are wrapped in `AssistantAPIAdapter`, making it easy to mount additional protocols alongside each other.
- **A2A (Agent-to-Agent) adapter** — new `A2AAdapter` exposes agents over the Google A2A protocol, supporting task lifecycle, streaming via SSE, and push-notification callbacks.
- **Compass developer dashboard** — built-in Vue SPA served at `/compass` for inspecting agents, threads, runs, traces, and config in real time. Mountable as an adapter or run standalone via `llamphouse compass`.
- **Dashboard adapter** — lightweight `DashboardAdapter` at `/_dashboard` for minimal operational endpoints.
- **CLI (`llamphouse`)** — new command-line interface with `serve`, `worker`, and `compass` sub-commands. Supports `--host`, `--port`, `--no-workers`, and `--ws` flags.
- **Config store** — `BaseConfigStore` / `InMemoryConfigStore` for runtime-tunable agent parameters (`NumberParam`, `StringParam`, `PromptParam`, `BooleanParam`, `SelectParam`).
- **Distributed worker mode** — `DistributedWorker` consumes runs from a Redis-backed queue and publishes SSE events via Redis Pub/Sub, enabling horizontal scaling across multiple processes.
- **Redis queue** — `RedisQueue` implementation using Redis Streams with consumer groups for reliable, distributed run dispatch.
- **Redis event queue** — `RedisEventQueue` for cross-process SSE event delivery between workers and API pods.
- **Rich message parts** — `TextPart`, `ImagePart`, `FilePart`, and `DataPart` types for structured multi-modal message content.
- **WebSocket protocol flag** — `ignite()` and the CLI now accept a `--ws` parameter (forwarded to uvicorn) to select the WebSocket implementation (e.g. `websockets-sansio`).
- **Docker support** — added `Dockerfile`, `docker-compose.yml`, and `docker-compose.prod.yml` for containerised deployments with OpenTelemetry Collector sidecar.
- **MkDocs documentation site** — full docs covering installation, quickstart, concepts (agents, adapters, context, multi-agent), guides (streaming, tool calls, tracing, config store, compass), deployment, and API compatibility.
- **New examples** — reorganised and expanded to 10 examples: HelloWorld, Chat, Streaming, ToolCall, OrchestratorAgent, AgentHandover, Tracing, ConfigStore, CustomAuth, and DistributedWorker.

### Changed

- **Refactored package layout** — moved from flat `llamphouse/core/` to `llamphouse/llamphouse/core/` with proper namespacing.
- **Adapter initialisation** — passing an explicit `adapters` list to `LLAMPHouse()` now means "use exactly these"; Compass is only auto-mounted when `adapters` is omitted.
- **Bumped dependency bounds** — `uvicorn >=0.35.0,<1.0` (was `<0.41`), `fastapi >=0.100.0,<1.0` (was `<0.130`), `opentelemetry-instrumentation-fastapi >=0.60b0,<1.0` (was `<0.61`).
- **Stable Compass build output** — Vite now produces hash-free filenames for clean git diffs.
- **Auth system expanded** — `BaseAuth` now returns an `AuthResult` with richer context; added `KeyAuth` convenience implementation.
- **Context API extended** — additional helpers for tool-call steps, message insertion, and run/thread metadata updates.

### Fixed

- Compass adapter no longer silently injects itself when a custom `adapters` list is provided.

## [1.1.0] - 02/02/2026

- Added end-to-end tracing across the LLAMPHouse system.
- Consistent span naming and GenAI attributes for observability.
- Input/output payload attributes to surface request/response data in traces.
- Environment-based tracing configuration (enable/disable + exporter setup).
- Example configuration for tracing with OTEL/Langfuse.

## [1.0.1] - 29/1/2026

* Fixed the initial migration version chain (base revision alignment).
* Standardized timezone handling: run_steps now migrate to tz‑aware timestamps consistently.

## [1.0.0] - 9/1/2026

### Added

- Introduced a pluggable data_store architecture with in-memory and Postgres backends.
- Added event queues with in-memory and Janus implementations.
- Added streaming adapters for OpenAI, Gemini, and Anthropic.
- Added **Data Retention Policy** support with automated purge functionality for both In-Memory and Postgres backends.
- Added a **Tox compatibility matrix** to ensure stable performance across multiple dependency versions (FastAPI 0.100.0 to latest).
- Added a comprehensive **Purge Example** demonstrating data lifecycle management.
- Added **GeminiStreaming example** demonstrating real-time output integration with the new pluggable streaming architecture.

### Changed

- **Refactored API lifecycle management**: Migrated from deprecated `startup`/`shutdown` events to the modern FastAPI `lifespan` context manager for improved resource handling.
- Updated examples to align with the new pluggable backend/streaming flow.

### Testing

- Expanded testing across unit, contract, integration, and streaming layers.

## [0.0.8] - 14/08/2025

- Fix messages being limited to 20 items

## [0.0.7] - 19/02/2025

- Change order messages in context (from desc to asc)
- Add worker as object in server init
- Add authenticator as object in server init
- Add ability to create custom authentication verification
- Make Assistant init with only a unique name
- Change context create_message to insert_message
- Update and add examples

## [0.0.6] - 03/02/2025

- Implement Graceful shutdown for both worker types.
- Implement pool size customization by env variable and check with maximum available from database connection
- Enhanced `Context` class with the ability to create new tool call step.
- Add remaining runs endpoints: Submit tool output, Cancel run.
- Add remaining run_steps endpoints: List run_step, Retrieve run_step.

## [0.0.5] - 29/01/2025

- Fixed issue with created_at field causing incorrect ordering by using a separate session for each FastAPI request.
- Ensured sessions are properly closed after each request.
- Moved session maker to the class initialization part to manage sessions more effectively.

## [0.0.4] - 23/01/2025

- Enhanced `Context` class with the ability to update thread, messages, and run details.
- Create DatabaseManagement class to handle database interact function
- Enable thread worker count customization
- Enable worker's task timeout customization

## [0.0.3] - 14/01/2025

- Generate new thread, message or run id based on metadata

## [0.0.2] - 13/01/2025

- Add api key authentication
- Fix metadata type declaration

## [0.0.1] - 30/12/2024

- Add initial API server
- Create threads and messages

## [0.0.0] - 16/12/2024

- Start of the project: December 16
