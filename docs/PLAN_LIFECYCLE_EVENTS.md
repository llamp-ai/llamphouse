# Plan — Lifecycle Events, Subscribers & Naming Cleanup

Status: **draft / not implemented**
Target release: TBD (post 1.3.0)
Scope: replace the ambiguous `Signal` abstraction with a clear three-concept model (Trigger / Event / Subscriber), add a durable event bus, and wire it into the existing worker.

---

## 1. Motivation

The 1.3.0 branch introduced `BaseSignal` / `WebhookSignal`, but the name conflates two genuinely different patterns:

- **Inbound**: external system → POST → LLAMPHouse starts a run. This is what `WebhookSignal` actually does.
- **Outbound**: agent finishes (or hits some state) → notify a registered destination. This does not exist yet.

`SignalInfo` also hints at a third pattern (agent-to-agent events via `source_agent_id` / `source_run_id`), further muddying the vocabulary.

Today the run state machine in [async_worker.py](../llamphouse/llamphouse/core/workers/async_worker.py) drives `queued → in_progress → completed | failed | expired | cancelled` and persists via `data_store.update_run_status(...)`, but there are no user-facing hooks. Tracing spans observe the transitions; nothing else can.

This plan formalises the missing pieces and renames the existing ones so each concept has one job.

---

## 2. Final vocabulary

| Concept | Direction | Today | Proposed |
|---|---|---|---|
| External thing → starts a run | inbound | `WebhookSignal` | **`Trigger`** (`WebhookTrigger`, future `ScheduleTrigger`, ...) |
| Framework emits when something happens during a run | outbound | *missing* | **`Event`** (plain data — `RunEvent`) |
| Thing that reacts to an event | outbound | *missing* | **`Subscriber`** (`WebhookSubscriber`, `AgentSubscriber`, plus `@on_event`) |

**Rename**: `BaseSignal` → `BaseTrigger`, `WebhookSignal` → `WebhookTrigger`, `SignalInfo` → `TriggerInfo`, `context.signal` → `context.trigger`.
Provide a deprecation shim re-exporting the old names for one minor version.

### 2.1 1.3.0 inbound webhook scope

`WebhookTrigger` is the only webhook concept intended for the 1.3.0 release. It is an inbound trigger: an external system sends `POST /triggers/...`, LLAMPHouse records the inbound payload, selects or creates a thread, optionally inserts an inbound user message, creates a run for the owning Agent Deployment, and enqueues that run.

`WebhookSubscriber` is a later outbound concept. It is not required for 1.3.0. A subscriber sends a webhook after LLAMPHouse emits a lifecycle event such as `run.completed`; it belongs to the post-1.3.0 event bus work in this plan.

Inbound webhook idempotency is optional for `WebhookTrigger`, but when configured in 1.3.0 it is atomic command idempotency for inbound webhook calls only. LLAMPHouse claims `(scope, key)` before thread creation, user message insertion, or run creation. A duplicate request with the same semantic fingerprint returns the stored run and thread identifiers without inserting another user message. A duplicate request with the same key but a different semantic fingerprint returns `409 Conflict`.

The inbound webhook semantic fingerprint is computed only from resolved fields used by the command: resolved thread id, resolved inbound user message text, mapped thread metadata, and mapped run metadata. It does not include the raw payload, idempotency key, Agent Deployment id, or trigger path; the deployment and path are part of `scope`, and the configured idempotency value is the `key`.

Mapped thread and run metadata values may be any JSON-serializable value, including objects and arrays. Fingerprinting uses a canonical JSON representation with sorted object keys, compact separators, UTF-8 encoding, and non-finite numbers rejected. Object key order is ignored, array order is preserved, explicit `null` is preserved, and missing mapped fields are omitted unless the mapping configuration supplies an explicit default. Non-JSON-serializable values and non-finite numbers are rejected.

The 1.3.0 implementation should expose one high-level store/service contract for executing an inbound webhook command atomically, rather than exposing separate idempotency claim/update methods for `WebhookTrigger` to compose. The contract represents the full command, for example `execute_webhook_command(command) -> WebhookCommandResult`.

That contract owns the transaction boundary. It atomically validates or claims `(scope, key)` when idempotency is configured, returns the stored result for a duplicate with the same fingerprint, rejects a duplicate with a different fingerprint, resolves or creates the thread, inserts the inbound user message when configured, creates the run, persists the idempotency result, and commits those effects together. `PostgresDataStore` should implement this with a transaction and `UNIQUE(scope, key)`; `InMemoryDataStore` should protect the same critical section with an in-process lock.

`WebhookTrigger` should remain responsible for HTTP concerns and command preparation only: verifying the secret, resolving mappings from the payload, validating required fields, computing the semantic fingerprint, building the `WebhookCommand`, calling `execute_webhook_command(...)`, and enqueueing the returned run through the existing run queue after the command commits.

The 1.3.0 `WebhookCommand` should only include fields with agreed behavior: `scope`, `idempotency_key`, `fingerprint`, `agent_id`, `trigger_path`, `thread_id`, `thread_metadata`, `message_text`, `run_metadata`, and `run_config_values`. It should not include `message_metadata`. The inbound user message inserted by a webhook is represented by role and content only in 1.3.0; conversation metadata belongs in `thread_metadata`, and event/run metadata belongs in `run_metadata`. If provenance is needed for 1.3.0, store it in run metadata, such as webhook trigger path, external event id, or idempotency key. A separate message metadata surface can be added later without breaking the 1.3.0 command contract.

The idempotency `scope` should be a deterministic SHA-256 hash of canonical structured data, not string concatenation:

```json
{
  "type": "webhook",
  "agent_id": "<agent_id>",
  "trigger_path": "<normalized_trigger_path>"
}
```

This avoids delimiter ambiguity, keeps the `UNIQUE(scope, key)` index short and predictable, remains deterministic across retries and processes, and leaves room for future scope types. Store the raw components separately for debugging and query.

The 1.3.0 table shape should be:

```text
webhook_idempotency_claims
  scope TEXT NOT NULL
  key TEXT NOT NULL
  fingerprint TEXT NOT NULL
  agent_id TEXT NOT NULL
  trigger_path TEXT NOT NULL
  thread_id TEXT NOT NULL
  message_id TEXT NULL
  run_id TEXT NOT NULL
  response_json JSON NOT NULL
  created_at TIMESTAMP NOT NULL
  updated_at TIMESTAMP NOT NULL
  expires_at TIMESTAMP NULL

UNIQUE(scope, key)
```

Do not store the raw payload by default. Store replayable response data in `response_json`; keep `expires_at` for later cleanup without requiring cleanup in 1.3.0. This design is scale-aware but not scale-complete for 1.3.0: it covers deterministic hashed scope, `UNIQUE(scope, key)`, canonical fingerprinting, response replay, the future cleanup hook, and the high-level `execute_webhook_command(...)` contract. It still excludes durable run-dispatch outbox, lifecycle event bus, outbound subscribers, and distributed rate limiting.

`response_json` stores the public response shape that the webhook endpoint can replay directly, not an internal-only payload. The original successful command stores fields such as `run_id`, `thread_id`, `message_id`, `deduped: false`, and `thread_created`. On a duplicate request with the same fingerprint, load `response_json`, override only `deduped` to `true` and `thread_created` to `false`, and return it. Do not recompute identifiers from related tables unless the stored response is unavailable.

The webhook endpoint status code matrix for 1.3.0 is:

- New webhook command accepted and enqueued through the existing run queue behavior: `202 Accepted`.
- Duplicate with the same fingerprint: `200 OK` with replayed `response_json`, overriding `deduped: true` and `thread_created: false`.
- Duplicate with the same `(scope, key)` but a different fingerprint: `409 Conflict`.
- Invalid payload or invalid config-derived required input: `400 Bad Request`.
- Well-formed thread id that does not exist: `404 Not Found`.
- Server-side webhook secret misconfiguration: `503 Service Unavailable`.
- Missing or invalid auth/signature: keep the existing `401 Unauthorized` / `403 Forbidden` behavior.

Run dispatch remains the existing behavior in 1.3.0. This release does not guarantee durable `run_queue` enqueue through a transactional outbox, and it does not include the lifecycle event bus, outbound webhook delivery, or `WebhookSubscriber`. Outbound subscriber idempotency is separate future work: event delivery will be at-least-once, and subscribers will receive a stable `event_id` / `Idempotency-Key` so receivers can dedupe delivery retries.

---

## 3. User-facing surface

Only **per-agent** registration. No server-level subscribers, no inline-in-`run()` decorators. Two syntactic forms feeding **one** internal registry:

### 3.1 Declarative — `subscribers = [...]`

For config-heavy things (URL, retry, secret, event filter). Reads as data.

```python
class ReportAgent(Agent):
    triggers = [
        WebhookTrigger(path="/triggers/report", secret_env="WEBHOOK_SECRET"),
    ]

    subscribers = [
        WebhookSubscriber(
            url="https://ops.example.com/hooks/agent-done",
            events=["run.completed", "run.failed"],
            secret_env="OUTBOUND_SECRET",   # HMAC-signs the body
            retry=RetryPolicy(max=8, backoff="exponential"),
        ),
        AgentSubscriber(
            agent_id="follow-up-agent",
            events=["run.failed"],
            payload_template=lambda evt: {"original_run": evt.run_id},
        ),
    ]
```

### 3.2 Decorator — `@on_event`

For inline custom code that needs `self`. Avoids the class-attribute binding problem because the framework collects decorated methods at agent boot and binds `self` per instance.

```python
class ReportAgent(Agent):
    @on_event("run.failed")
    async def on_failure(self, event: RunEvent):
        await self.notify_team(event.run_id, event.error)

    @on_event("run.completed", "run.failed")
    async def always_log(self, event: RunEvent):
        logger.info("run %s -> %s", event.run_id, event.name)
```

### 3.3 Both feed the same registry

At agent construction the framework walks `dir(cls)`, finds `@on_event` methods, wraps them as synthetic subscribers, and appends them to the agent's effective `subscribers` list. After that point the dispatcher cannot tell the two forms apart — same retry rules, same tracing, same Compass display.

### 3.4 Rule for users

> Writing **config**? Add to `subscribers`. Writing **code**? Use `@on_event`.

### 3.5 Out of scope (for v1)

- `LLAMPHouse(subscribers=[...])` global subscribers — defer until a real use case appears.
- `CallableSubscriber` taking a string method name or free function — `@on_event` covers this without the stringly-typed lookup.
- Inline-in-`run()` decorators — use `try/finally` for self-scoped cleanup.

---

## 4. Event types

### 4.1 Shape

```python
@dataclass(frozen=True)
class RunEvent:
    event_id: str          # UUID v4, stable across retries — used for idempotency
    name: str              # e.g. "run.completed"
    run_id: str
    thread_id: str
    agent_id: str
    occurred_at: datetime  # tz-aware UTC
    status: str            # snapshot of run.status at emit time
    error: Optional[dict]  # populated on failed/expired
    trigger: Optional[dict] # TriggerInfo.to_dict() if the run was trigger-initiated
    metadata: dict
```

### 4.2 v1 event catalogue

Mirrors the existing `run_status` enum, one event per transition emitted by the worker:

- `run.queued`
- `run.started`
- `run.completed`
- `run.failed`
- `run.expired`
- `run.cancelled`

### 4.3 Future events (not v1, reserve names)

- `message.created`
- `tool_call.started`, `tool_call.completed`
- `run_step.created`

### 4.4 Filter syntax

Subscribers declare `events=[...]` as exact names or glob (`run.*`). Glob compiles to a prefix match — keep it dumb.

---

## 5. Messaging bus — transactional outbox

### 5.1 Why an outbox

The only way to guarantee an event is never lost is to write it to durable storage **in the same transaction as the state change that produced it**. Any other pattern (asyncio task, pub/sub publish) has a crash window where the state changed but the event was never recorded.

### 5.2 Worker-side write

Every call site in [async_worker.py](../llamphouse/llamphouse/core/workers/async_worker.py) that runs `data_store.update_run_status(...)` becomes:

```
BEGIN
  UPDATE runs SET status = '<new>' ...
  INSERT INTO events (...) VALUES (...)
  INSERT INTO event_deliveries (...) VALUES (..., 'pending', ...) [one row per subscriber]
COMMIT
```

If the worker crashes mid-transaction, neither write happens — the next worker that picks up the run will retry from a consistent prior state.

### 5.3 Schema

```
events:
  event_id           UUID PRIMARY KEY
  name               TEXT NOT NULL
  run_id             UUID NOT NULL
  thread_id          UUID NOT NULL
  agent_id           TEXT NOT NULL
  payload            JSONB NOT NULL    -- serialised RunEvent
  created_at         TIMESTAMPTZ NOT NULL

event_deliveries:
  event_id           UUID NOT NULL REFERENCES events
  subscriber_id      TEXT NOT NULL     -- stable id per subscriber declaration
  status             TEXT NOT NULL     -- pending | in_flight | delivered | dead_letter
  attempts           INT NOT NULL DEFAULT 0
  next_attempt_at    TIMESTAMPTZ NOT NULL
  last_error         TEXT
  delivered_at       TIMESTAMPTZ
  PRIMARY KEY (event_id, subscriber_id)

  INDEX (status, next_attempt_at)   -- the dispatcher's hot read path
```

### 5.4 Subscriber identity

`subscriber_id` is a stable string derived from `(agent_id, type, declaration_index)` so a subscriber declared at agent boot keeps the same id across restarts. Required so partially-delivered events can resume after deploy.

### 5.5 Dispatcher

Separate component. Two regimes mirroring the existing worker split:

**Single-process (`AsyncWorker`)**: dispatcher is an asyncio task in the same process. Polls `event_deliveries WHERE status = 'pending' AND next_attempt_at <= now()`, dispatches in parallel.

**Distributed (`DistributedWorker`)**: N dispatcher processes coordinating via `SELECT ... FOR UPDATE SKIP LOCKED` (Postgres) or Redis Streams consumer groups (already used by `RedisQueue`). Each dispatcher claims a batch, processes, releases. No coordinator.

### 5.6 Per-subscriber isolation

One row per `(event_id, subscriber_id)` so a slow / failing subscriber backs off on its **own row** without blocking others. Optional per-subscriber concurrency cap (defer to v2).

### 5.7 Retry + dead-letter

- Exponential backoff with jitter: `next_attempt_at = now + min(2^attempts, max_backoff) * (0.5 + rand())`
- `max_attempts` default 8 (≈ 4 minutes total with cap of 1 min)
- After `max_attempts`: status → `dead_letter`
- Compass surfaces DLQ count + per-event inspection + manual replay button

### 5.8 Idempotency

Delivery is at-least-once — receivers will occasionally see the same event twice. The framework makes dedupe easy:

- `RunEvent.event_id` is stable across retries.
- `WebhookSubscriber` sends `Idempotency-Key: <event_id>` header.
- `AgentSubscriber` stamps the created run's metadata with `source_event_id` and checks for an existing run with that key before creating — natural dedupe.
- `@on_event` handlers are passed the event; users can guard themselves if needed.

**Be explicit in the docs**: we deliver at-least-once, not exactly-once. Subscribers must tolerate duplicates.

### 5.9 In-process handlers — `durable=True` by default

`@on_event` handlers go through the outbox by default. Crash mid-handler → re-dispatched on restart. Cost: one DB write per event per handler.

Opt-out per subscriber for explicitly best-effort handlers (debug logging, metrics):

```python
@on_event("run.completed", durable=False)
async def metrics(self, event):
    statsd.incr("runs.completed")
```

`durable=False` dispatches as a fire-and-forget asyncio task with no persistence.

### 5.10 In-memory data store

`InMemoryDataStore` keeps the outbox in memory — lost on restart. Document that `InMemoryDataStore` is dev-only when using subscribers; production needs `PostgresDataStore`.

### 5.11 Retention / GC

`events` and `event_deliveries` grow forever without GC. Extend the existing data retention policy (1.0.0) to cover them. Default: drop `delivered` rows older than 7 days; keep `dead_letter` rows for 30 days for operator visibility.

---

## 6. Ordering & delivery contract

Document precisely:

- **Per run**: events are emitted in the order they occur (worker is single-threaded per run).
- **Across subscribers**: no order guarantee — `subscriber A` may see `run.completed` for run X before `subscriber B` sees `run.started` for run X.
- **Per subscriber**: events for the same run are dispatched in occur-order *unless* the subscriber is configured for parallel delivery (v2).

---

## 7. Scaling characteristics

| Dimension | Substrate | Limit |
|---|---|---|
| Events/sec write | Postgres `INSERT` in worker tx | bounded by worker throughput; not a new bottleneck |
| Dispatcher throughput | Horizontal: N processes × `SKIP LOCKED` batches | scales linearly until DB write contention |
| Per-subscriber isolation | Per-row delivery state | a single bad subscriber cannot stall others |
| Storage growth | `events` + `event_deliveries` rows | bounded by retention policy |
| Latency (event emit → delivery) | dispatcher poll interval + network | ~tens of ms baseline, fine for any non-realtime use |

---

## 8. Failure-mode matrix

| Failure | What happens | Recovery |
|---|---|---|
| Worker crash before status update tx commits | Run still in old status, no event written | Run gets re-picked up; idempotent state machine |
| Worker crash after tx commits, before reply to queue | Event written; run done | Dispatcher picks up the event normally |
| Dispatcher crash mid-delivery | Delivery row stuck in `in_flight` | Stale-lease reaper resets `in_flight` rows older than N seconds back to `pending` |
| Subscriber returns 5xx | Delivery row stays `pending` with incremented attempts | Backoff + retry |
| Subscriber returns 4xx (other than 429) | Treated as terminal — straight to `dead_letter` | Operator inspects via Compass |
| Network partition between dispatcher and subscriber | Same as 5xx | Backoff + retry |
| `@on_event` handler raises | Caught by dispatcher, logged, retried per policy | Same as webhook failure |
| Postgres unreachable | Worker tx fails, run not marked done | Worker retries; event never lost because state never advanced |

---

## 9. Public API summary

New classes / functions to expose:

```python
from llamphouse import (
    Agent,
    # triggers (renamed from signals)
    BaseTrigger, WebhookTrigger,
    # events
    RunEvent,
    # subscribers
    BaseSubscriber, WebhookSubscriber, AgentSubscriber,
    on_event,
    # config
    RetryPolicy,
)
```

Removed / deprecated (kept as shims for one release):

- `BaseSignal`, `WebhookSignal`, `SignalInfo` → emit `DeprecationWarning`, re-export as alias.
- `context.signal` → alias for `context.trigger`.

---

## 10. Compass surface

Out of scope to fully spec here, but required for the feature to be operable in production:

- **Events view**: timeline of recent events with filter by name / agent / run.
- **Subscribers view**: per-agent list of subscribers, declared form (declarative vs `@on_event`), recent delivery success rate.
- **Dead-letter queue view**: list of DLQ rows with last error + manual replay button.
- **Per-run detail**: events emitted by this run + their delivery state per subscriber.

---

## 11. Stepped development plan

Each step is intended to be roughly one PR. Steps within a phase are ordered by dependency. Phases can ship independently (each ends at a usable state). Every step ends with a **Done when** criterion so it's clear when to stop.

### Phase 1 — Rename Signal → Trigger (foundational, no behaviour change)

> Goal: clean up vocabulary before adding new concepts. Ship before 1.3.0 to avoid two breaking renames.

> **Note:** the `Signal` naming was only ever on the unreleased 1.3.0 feature branch, so this rename is a straight replacement with no deprecation shims or backward-compat code.

- [x] **1. Rename `BaseSignal` → `BaseTrigger`** in `core/signals/base.py`; move file to `core/triggers/base.py`. Update all imports. *Done when:* package imports work, tests pass.
- [x] **2. Rename `WebhookSignal` → `WebhookTrigger`** in `core/signals/webhook_signal.py`; move to `core/triggers/webhook_trigger.py`. *Done when:* import path updated everywhere; example still runs.
- [x] **3. Rename `SignalInfo` → `TriggerInfo`** and `context.signal` → `context.trigger`. *Done when:* `grep -r "signal" llamphouse/` shows no references.
- [x] **4. Delete the old `core/signals/` package** — no shims needed (not yet released). *Done when:* directory gone; no imports of `llamphouse.core.signals` remain.
- [x] **5. Rename `Agent.signals` → `Agent.triggers`** attribute. *Done when:* worker / route registration loops only read `triggers`.
- [x] **6. Rename example `examples/11_WebhookSignal/` → `examples/11_WebhookTrigger/`**. Update README and client. *Done when:* example runs end-to-end.
- [x] **7. Update CHANGELOG** under `[1.3.0] - TBD`: describe "Trigger handling" (no rename note needed, since `Signal` never shipped). *Done when:* CHANGELOG accurate.
- [ ] **8. Update mkdocs concepts/guides** that mention signals. *Done when:* `mkdocs build` passes; no stale "signal" references in user-facing docs.

### Phase 2 — Core event types and registry (no dispatch yet)

> Goal: data model + agent-side declaration. Events can be created and collected but nothing fires.

- [ ] **9. Add `RunEvent` dataclass** in `core/events/types.py` per §4.1. *Done when:* unit tests cover serialisation round-trip.
- [ ] **10. Add `BaseSubscriber` abstract class** in `core/subscribers/base.py` — abstract `subscriber_id` property, `events: list[str]`, `durable: bool`, `retry: RetryPolicy`. *Done when:* a trivial concrete subclass is dispatchable in tests.
- [ ] **11. Add `RetryPolicy` dataclass** — `max_attempts`, `backoff`, `max_backoff`, `jitter`. Static method `next_attempt_at(attempts) -> datetime`. *Done when:* unit tests cover exponential + jitter bounds.
- [ ] **12. Add `@on_event(*names, durable=True)` decorator** in `core/subscribers/on_event.py`. Marks the method with `__llamphouse_event_subscriber__ = (names, durable)`. *Done when:* decorated method retains the attribute; calling the method still works normally.
- [ ] **13. Add agent-boot introspection** — `Agent.__init_subclass__` (or a helper called from `Agent.__init__`) walks methods, collects `@on_event` markers, computes stable subscriber ids per §5.4, builds the effective subscribers list. *Done when:* `agent.effective_subscribers` returns both declared and decorated, with deterministic ids across restarts.
- [ ] **14. Add glob → predicate compilation** for `events=["run.*"]` filters. *Done when:* unit tests cover exact, glob, and mixed lists.

### Phase 3 — Persistence + worker emit (single-process durable)

> Goal: events are written transactionally with run status changes. Nothing dispatches yet — outbox accumulates.

- [ ] **15. Add Alembic migration: `events` table** per §5.3. *Done when:* `alembic upgrade head` applies cleanly on a fresh DB.
- [ ] **16. Add Alembic migration: `event_deliveries` table** per §5.3 with composite PK and `(status, next_attempt_at)` index. *Done when:* migration applies; `EXPLAIN` on the dispatcher query uses the index.
- [ ] **17. Add data store methods** to [BaseDataStore](../llamphouse/llamphouse/core/data_stores/base_data_store.py): `insert_event_with_deliveries(event, subscribers)`, `claim_pending_deliveries(limit, now)`, `mark_delivered(event_id, subscriber_id)`, `mark_failed(event_id, subscriber_id, error, next_attempt_at)`, `mark_dead_letter(...)`. *Done when:* abstract methods defined with type stubs.
- [ ] **18. Implement methods in `PostgresDataStore`** — single transaction for emit, `SELECT FOR UPDATE SKIP LOCKED` for claim. *Done when:* concurrent claim test (10 workers) shows no double-claim and no missed rows.
- [ ] **19. Implement methods in `InMemoryDataStore`** — dict-backed, log a warning on `insert_event_with_deliveries` that subscribers in this store are dev-only. *Done when:* in-memory unit tests pass; warning fires.
- [ ] **20. Wire emit into [async_worker.py](../llamphouse/llamphouse/core/workers/async_worker.py)** — every `update_run_status(...)` call site becomes a combined `update_run_status_and_emit_event(...)` running both writes in one tx. *Done when:* every status transition (queued, in_progress, completed, failed, expired, cancelled) writes a matching event row.
- [ ] **21. Add integration test**: run an agent, assert the `events` table contains the expected sequence; assert `event_deliveries` rows exist (one per subscriber declared). *Done when:* test green.

### Phase 4 — In-process dispatcher

> Goal: events actually fire. Subscribers run. Retries and DLQ work. Single-process only.

- [ ] **22. Implement `WebhookSubscriber`** — HMAC-SHA256 body signing, `Idempotency-Key` header, configurable timeout. *Done when:* contract test against a stub HTTP server verifies headers and retry behaviour.
- [ ] **23. Implement `AgentSubscriber`** — creates run on target agent, stamps `metadata.source_event_id`, dedupes by checking for existing run with that key. *Done when:* duplicate delivery of the same event creates only one downstream run.
- [ ] **24. Implement `CallableSubscriber`** (internal — wraps `@on_event` methods). Not part of public API. *Done when:* `@on_event` handlers fire and exceptions propagate to dispatcher.
- [ ] **25. Add `AsyncDispatcher`** — asyncio task started in `LLAMPHouse.ignite()` lifespan. Polls `claim_pending_deliveries` every N ms, dispatches in parallel, marks done. *Done when:* end-to-end test: trigger agent → run completes → webhook receives POST with correct body and idempotency key.
- [ ] **26. Implement retry path** — on failure, compute `next_attempt_at` via `RetryPolicy`, mark row `pending` again. After `max_attempts`, mark `dead_letter`. *Done when:* test with always-500 webhook ends in DLQ after exactly `max_attempts` POSTs.
- [ ] **27. Implement `durable=False` fast path** — bypass outbox, `asyncio.create_task` directly. Used by best-effort `@on_event` handlers. *Done when:* `durable=False` subscribers fire without writing to `event_deliveries`.
- [ ] **28. Stale-lease reaper** — periodic task that resets `in_flight` rows older than `lease_ttl` back to `pending`. *Done when:* test simulates dispatcher crash mid-delivery; reaper recovers the row.

### Phase 5 — Distributed dispatcher

> Goal: dispatch scales horizontally for `DistributedWorker` users. Postgres + Redis substrates both supported.

- [ ] **29. Postgres `SKIP LOCKED` dispatcher process** — standalone entrypoint `llamphouse dispatcher` CLI sub-command. *Done when:* N=3 dispatcher processes deliver a 1000-event burst with no duplicates and no losses.
- [ ] **30. Redis Streams dispatcher variant** — consumer group per subscriber type, PEL handling, claim-old-pending for crashed consumers. *Done when:* equivalent burst test passes against Redis substrate.
- [ ] **31. Add dispatcher selection to `DistributedWorker`** — auto-pick based on configured data store. *Done when:* docker-compose distributed example fires events end-to-end.

### Phase 6 — Compass UI

> Goal: events are observable and the DLQ is actionable.

- [ ] **32. Backend: events list endpoint** — `GET /compass/events` with filter by name / agent / run / time range. *Done when:* paginated response works against Postgres.
- [ ] **33. Backend: subscribers list endpoint** — `GET /compass/agents/{id}/subscribers`. *Done when:* response includes declaration source (declarative vs `@on_event`) and recent success rate.
- [ ] **34. Backend: DLQ + replay endpoint** — `GET /compass/dlq`, `POST /compass/dlq/{event_id}/{subscriber_id}/replay`. *Done when:* replay resets `attempts=0`, sets status=`pending`, dispatcher picks it up.
- [ ] **35. Frontend: Events view** in Compass — timeline + filter panel. *Done when:* renders against live data; filters work.
- [ ] **36. Frontend: Subscribers view** — per-agent table with success rate sparkline. *Done when:* matches backend output.
- [ ] **37. Frontend: DLQ view** — list with last error, payload preview, replay button. *Done when:* button triggers replay and row disappears.
- [ ] **38. Frontend: per-run event panel** — extend [RunDetailView.vue](../llamphouse/llamphouse/core/adapters/compass/frontend/src/views/RunDetailView.vue) with an "Events" tab showing what this run emitted + delivery state per subscriber. *Done when:* visible alongside existing run detail.

### Phase 7 — Retention, metrics, docs

> Goal: production-ready. Operators have what they need.

- [ ] **39. Extend retention policy** — config knob for `events_ttl_days` (default 7 for delivered, 30 for dead_letter). Add to purge worker. *Done when:* purge test removes old delivered rows and keeps dead_letter beyond the delivered cutoff.
- [ ] **40. Emit metrics** — delivery success rate (counter), dispatch latency (histogram), DLQ size (gauge). Expose via existing Prometheus/OTel endpoint. *Done when:* metrics visible in `/metrics` and a Grafana panel example shipped in docker examples.
- [ ] **41. Operational runbook** — `docs/guides/event-bus-operations.md`: "what to do when DLQ grows", "tuning retry policy", "scaling dispatchers". *Done when:* runbook reviewed; included in `mkdocs.yml`.
- [ ] **42. Delivery semantics doc** — `docs/concepts/events.md`: at-least-once contract, idempotency rules, ordering guarantees. *Done when:* doc reviewed; referenced from the subscribers guide.
- [ ] **43. New example: `examples/13_AgentEvents`** — agent with one `WebhookSubscriber`, one `AgentSubscriber`, one `@on_event` handler. Shows the three patterns side-by-side. *Done when:* example runs against `docker-compose.yml`.
- [ ] **44. CHANGELOG entry** for the release that ships phases 2–7. *Done when:* CHANGELOG updated with Added/Changed sections.

### Cross-cutting (apply to every step)

- [ ] Unit tests at each step touching code; integration tests at phase boundaries (steps 21, 28, 31, 38).
- [ ] No `# TODO` left behind — each step ends fully done.
- [ ] Migrations are forward-only and reviewed before merge (`alembic downgrade` is not part of the contract).
- [ ] Public API additions go through `llamphouse/__init__.py` so the surface stays explicit.

---

## 12. Open questions

1. **Subscriber identity stability** — derive from `(agent_id, type, index)` or require users to give each subscriber an explicit `id=`? Explicit is safer for reordering but more boilerplate.
2. **Dispatch concurrency model** — global pool, per-subscriber pool, or per-subscriber semaphore? Affects how isolated a slow subscriber really is.
3. **Webhook signing format** — HMAC-SHA256 of body with shared secret (Stripe style) vs full request signing (AWS SigV4 style)? Stripe style is simpler and good enough.
4. **AgentSubscriber thread scope** — does the chained agent run in a new thread, the source thread, or the user's choice? Probably new thread by default with an option to inherit.
5. **Replay semantics from DLQ** — does replay reset `attempts` to 0 or count toward the original cap? Probably reset (operator is making an explicit decision).
6. **Should `triggers` and `subscribers` share infra** for declaration / discovery? They're symmetric in feel; might justify a common base.

---

## 13. Non-goals

- Exactly-once delivery (impossible; we deliver at-least-once + idempotency).
- Cross-agent event routing beyond `AgentSubscriber` (no event broker semantics — no topics, no consumer groups exposed to users).
- Event sourcing / replay-from-zero as a programming model (events are notifications, not the system of record).
- Sub-millisecond latency (the outbox adds tens of ms; users wanting tighter need a different mechanism).
