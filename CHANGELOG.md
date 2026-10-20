# Changelog

## [1.4.0] - TBD

### Planned

- **Durable runs, what remains** — releasing leases on graceful shutdown, Postgres `NOTIFY` wake-ups, fencing step writes by `lease_epoch`, replaying `call_agent`. See `docs/DURABLE_RUNS_PLAN.md` §0 (P5).
- **Lifecycle events & subscribers** — transactional outbox, `subscribers = [...]` / `@on_event`, retry and dead-letter; also notifies humans when a run waits for input. See `docs/PLAN_LIFECYCLE_EVENTS.md`.
- **One active run per thread** — a rule (reject, queue or interrupt) for a new run on a thread that already has one, so retries, resumed runs and sessions never write to a thread at the same time.
- **Run policies** — retries and backoff, step and run time outs, and token budgets, configurable per agent in `llamphouse.yaml`.
- **AG-UI integration** — an AG-UI adapter, with human-in-the-loop mapped onto AG-UI interrupts and session state onto state snapshots / deltas, plus an example.
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
- **Sessions** (`docs/concepts/sessions.md`) — `context.session` is an agent's view of a thread for a run: a summary of the older messages and the recent ones as they are (`to_messages()` for a chat-completion API). `Agent.session_manager = SessionManager(summarizer=..., max_messages=..., keep_messages=..., max_tokens=..., count_tokens=...)` folds the oldest messages into the summary past its limits; the summary is kept per thread and agent (key-value records), so each run folds only what is new. The summarizer is any `async (previous_summary, messages) -> str`; without one the session is a sliding window. A compaction is a `session.compact` step, replayed by a run that runs again. `LangGraphAgent(session_manager=...)` passes the session's `messages` and `summary` to the graph.
- **Sessions example** (`examples/17_Sessions`).
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
