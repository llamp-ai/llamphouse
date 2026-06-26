# Changelog

## [1.3.0] - TBD

### Added

- **Pluggable tracing stores** — new `BaseTracingStore` interface with `InMemoryTracingStore`, `PostgresTracingStore`, and `ClickHouseTracingStore` implementations for persisting and querying span data.
- **Trigger handling** — `BaseTrigger` and `WebhookTrigger` for starting an agent run from an external system (e.g. an HTTP POST). Available on `context.trigger`.
- **Compass dashboards** — new dashboards UI with `DashboardsView`, `DashboardView`, and `DashboardPresentView`, backed by `DashboardStore` and `ChartStore`. Includes a `ChartWidget` for visualising tracing/run data and a `FilterBuilder` for composing queries.
- **Runs view** — dedicated `RunsView` in Compass for browsing and inspecting runs.
- **Table preferences** — `useTablePrefs` composable for persisting column/sort/filter state in Compass data tables.
- **WebhookTrigger example** (`examples/11_WebhookTrigger`) — end-to-end demonstration of triggering an agent run from an external HTTP POST.
- **PlannerAgent example** (`examples/12_PlannerAgent`) — multi-node planner/executor/synthesizer agent with tracing wired in.
- **`llamphouse.yaml` config** — declarative server config loaded by `llamphouse.cli.config.loader` and validated by Pydantic models in `llamphouse.cli.config.schema`. CLI now reads it on startup.
- **Strict `llamphouse.yaml` validation** — unknown fields now fail fast, `${ENV_VAR}` references are resolved before validation, and missing environment variables produce user-friendly errors.
- **YAML-configured data stores** — `data_store` can now be configured from `llamphouse.yaml`, including `postgres.database_url`.
- **YAML-configured webhook triggers** — deployment-level `triggers` can now define `WebhookTrigger` routes from `llamphouse.yaml`.
- **`llamphouse check` Health Check** — new preflight command for validating `llamphouse.yaml` without starting the server. It checks schema, component imports, entrypoints, route conflicts, data-store connectivity, tracing-store connectivity, and exits non-zero on failure for CI.
- **Health check output modes** — `llamphouse check` supports `--format text|json`, `--verbose`, and `--timeout` for external dependency checks.
- **CLI reorganisation** — moved entrypoints into the `llamphouse.cli` package.
- **Internal span exporter from tracing store** — tracing setup can wire the configured tracing store as a span exporter without external OTLP plumbing.
- **MkDocs landing page hook** — `hooks/landing_page.py` generates a custom `site/index.html` after each build.
- **Examples sync hook** — `hooks/sync_examples.py` keeps `docs/examples.md` in sync with the `examples/` directory, including a structured table and progression guide.
- **LLAMPHouseYAML example** (`examples/13_LLAMPHouseYAML`) — declarative server setup driven by `llamphouse.yaml`.
- **WebhookTrigger idempotency / dedupe** — webhook triggers can opt into idempotency by mapping a request payload field as the idempotency key. Duplicate requests return the original run instead of enqueueing a second run.
- **WebhookTrigger metadata mapping** — webhook request payload fields can be mapped into thread metadata and run metadata. Internal webhook metadata is stored under reserved `__webhook_*` run metadata keys.
- Database migration for runs now combines `stream`, `provider_config`, and float timestamp columns into the 1.3.0 migration chain.
- **Data-store list / count contract extended** — `BaseDataStore` now declares `list_threads`, `list_all_runs`, `get_run_any_thread`, `list_runs_by_parent_ids`, `get_first_run_assistant_ids`, `count_threads`, `count_runs`, and `count_messages`. Both `InMemoryDataStore` and `PostgresDataStore` implement them. These were previously missing or in-memory-only, which silently returned empty results on Postgres in Compass for Threads, Runs, the flow tree, and the home-page stats.
- **Run persistence contract coverage** — data-store contract tests now cover run stream values, `provider_config`, lifecycle timestamps, usage, and operational listing APIs across supported backends.
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

### Changed

- Orchestrator and Planner example agents now use `InMemoryTracingStore` by default.
- `SpanTree` truncation logic improved for cleaner display of long spans.
- Expanded `DataTable`, `MessageBubble`, `RunDetailView`, `ThreadDetailView`, and `ThreadsView` with richer rendering and interaction.
- `ConfigStore` example now ships with sample `compass_charts.json` and `compass_dashboards.json`.
- `in_memory_store` and `postgres_store` updated to align with new tracing/trigger flows and model changes.
- Logging during startup now surfaces more detailed information about the application state.
- **Ignite banner reorganised** into `Adapters` / `Triggers` / `Agents` / `Infrastructure` / `Optional features` sections; webhook trigger routes are listed inline (`▸ WebhookTrigger    /triggers/report → report-agent`).
- **Route-conflict warnings on boot** — duplicate webhook trigger paths, or trigger paths that fall under a non-root adapter prefix, now log a warning at startup.
- **Health checks use the runtime environment path** — `llamphouse check` loads `.env` from the config directory first, then falls back to the current working directory.
- **Compass flow rendering optimised** — edge geometry (`path`, `midX`, `midY`, colour, dash, marker) is pre-computed once inside `flowLayout` and bound directly in the template instead of being recomputed per-edge per-render. Roughly `O(E·N) → O(E + N)` per re-render.
- **Run-detail I/O resolver tolerates missing `run_id`** — assistant messages without a stamped `run_id` now match the run's `started_at..completed_at` window. Messages route also re-introduces `run_id` / `assistant_id` as explicit `null` (the prior `exclude_none=True` serialiser was stripping them entirely).
- **`PostgresDataStore.close()` is bounded** — `engine.dispose()` is wrapped in `asyncio.wait_for(..., timeout=5.0)` so a hung asyncpg socket can't block server shutdown for the OS TCP timeout.
- **Compass adapter no longer relies on missing methods.** All `hasattr(db, "…")` branches that masked data-store API gaps (`list_threads`, `list_all_runs`, `count_threads/runs/messages`, `get_run_any_thread`, `list_runs_all`) are gone, replaced by abstract methods on `BaseDataStore`. The only `hasattr` left is the legitimate backend dispatch in the dashboard SQL endpoint.

### Fixed

- Compass home page no longer shows `0` for Agents / Threads / Runs against `PostgresDataStore` (missing count methods).
- Compass Threads tab no longer renders empty against `PostgresDataStore` (missing `list_threads`).
- Compass Runs tab no longer renders empty against `PostgresDataStore` (missing `list_all_runs`).
- Compass agent-flow view no longer silently truncates the tree when a parent run is older than the recent-runs window (the BFS used to bail out at the first missing ancestor).
- Compass Run Detail Input/Output panel no longer appears empty for messages produced by `context.insert_message(...)` that weren't stamped with a `run_id`.
- Postgres-backed runs now persist and round-trip `stream`, `provider_config`, lifecycle timestamps, and usage consistently.
- LLAMPHouse initialization now works with callable event queue factories such as `RedisEventQueueFactory`.
- `llamphouse check` now returns a non-zero process exit code when any Health Check fails.
- `SAWarning: garbage collector is trying to clean up non-checked-in connection` on Compass thread listings, caused by an N+1 burst of sessions for per-thread agent enrichment (replaced with a single `SELECT DISTINCT ON (thread_id) thread_id, assistant_id` query via the new `get_first_run_assistant_ids`).
- `SyntaxWarning: invalid escape sequence '\s'` in `PostgresDataStore` docstring.

### Deprecated

- `AssistantAPIAdapter` — superseded by `A2AAdapter`. Using it emits a `DeprecationWarning`.

### Removed

- `DashboardAdapter` and its API routes / static files. Dashboard functionality is now served exclusively via the Compass adapter.

## [1.2.4] - 02/06/2026

### Added

- Added core data-store contract methods for run lookup, operational listing, and aggregate counts.
- Added data-store contract coverage for run `stream`, `provider_config`, lifecycle timestamps, `usage`, and operational APIs.
- Added Compass compare integration coverage for loading runs by run id across data-store backends.

### Changed

- Standardized run storage behavior across `InMemoryDataStore` and `PostgresDataStore`.
- Updated Compass and Dashboard routes to use public data-store APIs instead of in-memory private structures.
- Squashed `provider_config` into the existing `runs.stream` migration for a clean migration chain.

### Fixed

- Fixed Postgres-backed runs missing persisted `stream` data.
- Fixed run status updates so lifecycle timestamps and `usage` are persisted consistently.
- Fixed `LLAMPHouse` initialization with callable event queue factories such as `RedisEventQueueFactory`.

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
