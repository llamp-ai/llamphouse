# Changelog

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
