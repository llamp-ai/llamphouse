# Durable Runs — Implementation Plan

This document defines the path from the current `@step` checkpointing layer to
**durable runs**: workflows that survive worker crashes, can sleep for arbitrary
durations, and resume on external signals — without holding a process open.

The plan assumes a **no-long-blocking-sync** policy for user code: anything
that takes meaningful wall-clock time must be async (or wrapped in
`asyncio.to_thread`). Heartbeats and the dispatcher loop run on the same event
loop as user code.

---

## 0. Progress checklist

Tick items as they land. Phases map to §12.

### P1 — Leasing
- [ ] Migration: add `worker_id`, `lease_expires_at`, `lease_epoch`, `wake_at` to `runs` (§3.1)
- [ ] Indexes: `idx_runs_pickup_inprogress`, `idx_runs_pickup_wake` (§3.1)
- [ ] Data store API: `claim_run`, `renew_lease`, `release_lease`, `list_runs_to_pickup` (§4) — Postgres
- [ ] Same API for `InMemoryStore` (§4)
- [ ] `LeaseHolder` async context manager with heartbeat task (§5.1)
- [ ] Lease-lost cancellation wired into run task (§5.1)
- [ ] `AsyncWorker` dispatcher loop with claim + jitter (§5.2)
- [ ] Worker ID config (`WORKER_ID`, defaults) (§9)
- [ ] Graceful shutdown releases leases (§5.3)
- [ ] Unit tests: claim race, heartbeat, lease loss (§11)
- [ ] Integration test: kill worker mid-run → run goes to `failed` (no auto-resume yet)

### P2 — Step memoization
- [ ] Migration: add `call_key` to `run_steps` + unique index (§3.2)
- [ ] Data store API: `find_step_by_call_key`, `next_step_ordinal` (§4)
- [ ] `Context._next_ordinal` per-run counter, rebuilt on resume
- [ ] `@step` decorator: lookup by `call_key`, replay completed, reset stale `in_progress` (§6.1)
- [ ] `start_step` accepts and persists `call_key`
- [ ] Reaper auto-resumes after crash (re-dispatches via `agent.run`)
- [ ] Unit tests: replay returns stored output without re-invoking user code
- [ ] Integration test: kill worker mid-run → second worker resumes, no double execution

### P3 — Durable sleep
- [ ] `wake_at` populated when suspending; dispatcher predicate covers `wait_until` (§4 SQL)
- [ ] `WorkflowSuspended(BaseException)` sentinel (§6.4)
- [ ] `@step` lets `WorkflowSuspended` pass through without marking failure (§6.4)
- [ ] `context.sleep(seconds=..., until=...)` (§6.2)
- [ ] Worker `_execute` catches `WorkflowSuspended`, releases lease cleanly
- [ ] Resume flow: `Context.__init__` finalizes due `wait_until` step (§6.5)
- [ ] Unit tests: sleep with mocked clock; replay after wake
- [ ] Example update / new example demonstrating sleep

### P4 — Wait-for-signal
- [ ] Migration: `signals` table + lookup index (§3.3)
- [ ] Data store API: `deliver_signal`, `consume_signal`, `suspend_run`, `resume_run` (§4)
- [ ] Dispatcher predicate covers `wait_for_signal` with delivered signal (§4 SQL)
- [ ] `context.wait_for_signal(name, correlation_id, timeout)` (§6.3)
- [ ] HTTP endpoint `POST .../signals` (§7.1)
- [ ] Optional Python client convenience `client.runs.signal(...)` (§7.3)
- [ ] Resume flow: consume matching signal, mark step completed (§6.5)
- [ ] Timeout path: signal never arrives → step `expired`, raise `TimeoutError`
- [ ] Unit tests: signal-before-wait, signal-during-wait, FIFO ordering, idempotency
- [ ] Integration test: end-to-end refund-style example (§13)

### P5 — Polish & production readiness
- [ ] `NOTIFY`-based wakeup on Postgres (latency optimization)
- [ ] Configurable `@step(on_resume="retry"|"fail"|"skip")` policy (§8)
- [ ] Cancel-while-suspended path verified
- [ ] `run.expires_at` honored by reaper for suspended runs
- [ ] Fencing-token enforcement on all step writes (`lease_epoch` check) (§8)
- [ ] Property-based crash-injection tests (§11)
- [ ] Docs: concept page (`docs/concepts/durable-runs.md`)
- [ ] Docs: migration / upgrade guide for existing users
- [ ] CHANGELOG entry
- [ ] Feature flag `DURABLE_RUNS_ENABLED` removed (or stays as opt-out)

### Open RFCs (not blocking)
- [ ] Per-step retry policy design
- [ ] Cross-tenant fairness in dispatcher
- [ ] Signal payload size cap + blob-reference convention
- [ ] Idempotent signal delivery (`signal_id` from caller)
- [ ] Step versioning (`@step(version=N)`, `patched()` equivalent)

---

## 1. Goals & Non-Goals

### Goals

- **Crash recovery**: a worker dying mid-run leaves no orphaned `in_progress`
  rows; another worker takes over within ≤ `lease_ttl`.
- **At-most-one active execution** per run across the cluster.
- **Step memoization**: re-entering `agent.run()` skips already-completed
  `@step` calls and replays their stored outputs.
- **Durable sleep**: `await context.sleep(hours=24)` releases the worker; some
  worker resumes the run when the timer fires.
- **Durable wait**: `await context.wait_for_signal("name")` suspends until
  an external event delivers the payload.
- **No new tables for v1**: reuse `runs.required_action` JSON column and the
  existing `RunStep` journal.

### Non-goals (deferred)

- Cross-version replay safety (`patched()` style versioning).
- Strict determinism enforcement (lint, sandboxing).
- Distributed tracing of suspend/resume edges.
- Parallel step fan-out with deterministic merge.

---

## 2. State Model

| Run status | Worker active? | Lease + heartbeat? | Pickup trigger |
|---|---|---|---|
| `queued` | ❌ | ❌ | Any worker claim |
| `in_progress` | ✅ | ✅ | Lease expiry → another worker claims |
| `requires_action` (`wait_until`) | ❌ | ❌ | Timer expiry |
| `requires_action` (`wait_for_signal`) | ❌ | ❌ | Signal delivery |
| `requires_action` (`submit_tool_outputs`) | ❌ | ❌ | Tool-output POST (existing) |
| `completed` / `failed` / `cancelled` / `expired` | ❌ terminal | ❌ | — |

**Invariant**: only `in_progress` runs hold a lease. Suspended runs cost
nothing; only the row exists.

### `required_action` payloads (extends existing discriminated union)

```jsonc
// Existing
{ "type": "submit_tool_outputs", "submit_tool_outputs": { "tool_calls": [...] } }

// New
{ "type": "wait_until",
  "wait_until": { "timestamp": 1747000000.0, "step_id": "step_xyz" } }

{ "type": "wait_for_signal",
  "wait_for_signal": {
    "signal": "stripe.payment_received",
    "correlation_id": "ord_123",
    "step_id": "step_abc",
    "timeout_at": 1747086400.0   // optional
  } }
```

`step_id` ties the wait to a memoizable `@step`-style row so the resumed run
can find the persisted output and continue.

---

## 3. Schema Changes

### 3.1 `runs` table — leasing columns

```sql
ALTER TABLE runs
  ADD COLUMN worker_id        TEXT,
  ADD COLUMN lease_expires_at TIMESTAMPTZ,
  ADD COLUMN lease_epoch      INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN wake_at          DOUBLE PRECISION;  -- mirror of required_action.wait_until.timestamp for cheap indexing

CREATE INDEX idx_runs_pickup_inprogress
  ON runs (status, lease_expires_at)
  WHERE status = 'in_progress';

CREATE INDEX idx_runs_pickup_wake
  ON runs (status, wake_at)
  WHERE status = 'requires_action' AND wake_at IS NOT NULL;
```

- `worker_id` — opaque string (e.g. `hostname:pid:uuid4`).
- `lease_expires_at` — UTC; renewed by heartbeat.
- `lease_epoch` — fencing token; bumped on every claim. Step writes carry
  the epoch and stale writes are rejected.
- `wake_at` — denormalized from `required_action` so the reaper can scan
  with an indexable predicate.

### 3.2 `run_steps` — memoization key

```sql
ALTER TABLE run_steps
  ADD COLUMN call_key TEXT;

CREATE UNIQUE INDEX idx_run_steps_call_key
  ON run_steps (run_id, call_key)
  WHERE call_key IS NOT NULL;
```

`call_key` format: `<step_name>#<ordinal>` where ordinal is a per-run,
per-step-name monotonic counter assigned by the decorator on first call.
Unique index prevents duplicate writes during racy claim transitions.

### 3.3 New table — `signals` (single row per delivery)

Avoids losing a signal that arrives before the run reaches its `wait_for`.

```sql
CREATE TABLE signals (
  id              TEXT PRIMARY KEY,         -- signal_xxx
  run_id          TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  correlation_id  TEXT,
  payload         JSONB,
  delivered_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  consumed_at     TIMESTAMPTZ
);

CREATE INDEX idx_signals_lookup
  ON signals (run_id, name, correlation_id)
  WHERE consumed_at IS NULL;
```

Migrations live under [migrations/versions/](migrations/versions) following
the existing alembic convention.

---

## 4. Data Store API additions

Add to both `InMemoryStore` and `PostgresStore` (interface in
[llamphouse/llamphouse/core/data_stores/store.py](llamphouse/llamphouse/core/data_stores/store.py)):

```python
# Leasing
async def claim_run(run_id: str, worker_id: str, ttl_seconds: float) -> Optional[RunLease]
async def renew_lease(run_id: str, worker_id: str, ttl_seconds: float) -> bool
async def release_lease(run_id: str, worker_id: str) -> None

# Pickup
async def list_runs_to_pickup(now: float, limit: int = 32) -> list[Run]

# Memoization
async def find_step_by_call_key(run_id: str, call_key: str) -> Optional[RunStep]
async def next_step_ordinal(run_id: str, step_name: str) -> int

# Suspension
async def suspend_run(run_id: str, lease_epoch: int, required_action: dict, wake_at: Optional[float]) -> bool
async def resume_run(run_id: str) -> bool   # requires_action → in_progress (atomic)

# Signals
async def deliver_signal(run_id: str, name: str, correlation_id: Optional[str], payload: Any) -> str
async def consume_signal(run_id: str, name: str, correlation_id: Optional[str]) -> Optional[Signal]
```

`RunLease` is a small dataclass: `{ run_id, worker_id, lease_epoch, lease_expires_at }`.

### Atomic claim SQL

```sql
UPDATE runs
   SET worker_id = :me,
       lease_expires_at = now() + (:ttl || ' seconds')::interval,
       lease_epoch = lease_epoch + 1,
       status = CASE WHEN status = 'queued' THEN 'in_progress' ELSE status END,
       started_at = COALESCE(started_at, extract(epoch from now()))
 WHERE id = :run_id
   AND (
     status = 'queued'
     OR (status = 'in_progress' AND (worker_id IS NULL OR lease_expires_at < now()))
     OR (status = 'requires_action' AND (
            (required_action->>'type' = 'wait_until'
             AND (required_action->'wait_until'->>'timestamp')::float <= extract(epoch from now()))
         OR (required_action->>'type' = 'wait_for_signal'
             AND EXISTS (
               SELECT 1 FROM signals s
                WHERE s.run_id = runs.id
                  AND s.name = required_action->'wait_for_signal'->>'signal'
                  AND (s.correlation_id IS NOT DISTINCT FROM required_action->'wait_for_signal'->>'correlation_id')
                  AND s.consumed_at IS NULL
             ))
     ))
RETURNING id, lease_epoch, lease_expires_at;
```

Empty result → another worker won the race; move on.

---

## 5. Worker-side runtime

### 5.1 New module: `llamphouse/llamphouse/core/runtime/lease.py`

```python
class LeaseHolder:
    def __init__(self, store, run_id, worker_id, ttl): ...
    async def __aenter__(self):
        # start heartbeat task
    async def __aexit__(self, exc_type, exc, tb):
        # cancel heartbeat, release lease (if no exception or recoverable)
    async def _heartbeat_loop(self):
        while not self._stop.is_set():
            await asyncio.sleep(self.ttl / 3)
            ok = await self.store.renew_lease(self.run_id, self.worker_id, self.ttl)
            if not ok:
                # We lost the lease — another worker took over.
                # Cancel the run task to avoid double-execution side effects.
                self._lease_lost.set()
```

The run task awaits `asyncio.wait({run_task, lease_lost_event}, FIRST_COMPLETED)`.
If the lease is lost mid-run, the task is cancelled (which propagates to
`@step`'s existing `CancelledError` handler).

### 5.2 Dispatcher loop (extends `AsyncWorker`)

```python
async def _dispatcher(self):
    while not self._shutdown.is_set():
        candidates = await self.store.list_runs_to_pickup(now(), limit=self.batch)
        for run in candidates:
            lease = await self.store.claim_run(run.id, self.worker_id, self.ttl)
            if lease is None:
                continue   # someone else got it
            asyncio.create_task(self._execute(run, lease))
        await asyncio.sleep(self.poll_interval + jitter())
```

- Poll interval: `lease_ttl / 3` (default 10 s with 30 s TTL).
- Jitter: `random.uniform(0, poll_interval / 2)` to avoid thundering herd
  after mass restart.
- Batch size: bounded by worker concurrency limit.

`_execute` wraps `agent.run(context)` in the `LeaseHolder` and the
`WorkflowSuspended` handler (§6.3).

### 5.3 Graceful shutdown

On `SIGTERM`:

1. Stop dispatcher polling.
2. For each in-flight run: `release_lease()` (sets `lease_expires_at = now()`).
   Another worker can pick it up immediately rather than waiting for TTL.
3. Cancel the run tasks (existing `CancelledError` path persists `cancelled`
   step rows, but the run itself stays `in_progress` so the next worker
   resumes it).

---

## 6. Decorator & context primitives

### 6.1 `@step` — add memoization

In [llamphouse/llamphouse/core/workflow.py](llamphouse/llamphouse/core/workflow.py):

```python
async def async_wrapper(*args, **kwargs):
    ctx = _find_context(args, kwargs)
    if ctx is None:
        return await func(*args, **kwargs)

    ordinal = await ctx._next_ordinal(step_name)
    call_key = f"{step_name}#{ordinal}"

    existing = await ctx.data_store.find_step_by_call_key(ctx.run.id, call_key)
    if existing and existing.status == "completed":
        return existing.step_details.output            # replay hit
    if existing and existing.status == "in_progress":
        # Crash mid-step. Policy: mark failed, then re-execute.
        # (Alternative: configurable resume policy.)
        await ctx.complete_step(existing.id, output=None,
                                error="Resumed after crash", status="failed")

    # ... existing start_step / execute / complete_step path,
    #     but pass call_key to start_step.
```

`ctx._next_ordinal` is an in-memory counter on `Context` (rebuilt on resume
by scanning existing steps once at run start).

### 6.2 `context.sleep`

```python
async def sleep(self, seconds: float = 0, *, until: Optional[float] = None):
    wake_ts = until if until is not None else time.time() + seconds
    call_key = f"__sleep__#{await self._next_ordinal('__sleep__')}"

    existing = await self.data_store.find_step_by_call_key(self.run.id, call_key)
    if existing and existing.status == "completed":
        return  # already slept on a prior incarnation

    step = await self.start_step(name="__sleep__", input={"wake_at": wake_ts},
                                 call_key=call_key)
    required_action = {
        "type": "wait_until",
        "wait_until": {"timestamp": wake_ts, "step_id": step.id},
    }
    await self.data_store.suspend_run(self.run.id, self._lease_epoch,
                                      required_action, wake_at=wake_ts)
    raise WorkflowSuspended(reason="sleep", required_action=required_action)
```

On resume, the decorator's memoization sees the sleep step still
`in_progress`; the resumption logic in `_execute` finalizes it
(`complete_step(status="completed")`) before user code re-runs.

### 6.3 `context.wait_for_signal`

```python
async def wait_for_signal(
    self, name: str, *,
    correlation_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Any:
    call_key = f"__wait__#{name}#{await self._next_ordinal('__wait__')}"
    existing = await self.data_store.find_step_by_call_key(self.run.id, call_key)
    if existing and existing.status == "completed":
        return existing.step_details.output

    # Was a signal already delivered before we got here?
    sig = await self.data_store.consume_signal(self.run.id, name, correlation_id)
    if sig is not None:
        step = await self.start_step(name="__wait__", input={...}, call_key=call_key)
        await self.complete_step(step.id, output=sig.payload)
        return sig.payload

    timeout_at = (time.time() + timeout) if timeout else None
    step = await self.start_step(name="__wait__", input={...}, call_key=call_key)
    required_action = {
        "type": "wait_for_signal",
        "wait_for_signal": {
            "signal": name, "correlation_id": correlation_id,
            "step_id": step.id, "timeout_at": timeout_at,
        },
    }
    await self.data_store.suspend_run(self.run.id, self._lease_epoch,
                                      required_action, wake_at=timeout_at)
    raise WorkflowSuspended(reason="signal", required_action=required_action)
```

### 6.4 `WorkflowSuspended` sentinel

```python
class WorkflowSuspended(BaseException):
    """Raised by suspend primitives. Caught by the worker; not a failure."""
    def __init__(self, reason: str, required_action: dict):
        self.reason = reason
        self.required_action = required_action
```

Inherits from `BaseException` so `except Exception` in user code doesn't
swallow it. The `@step` decorator must also let it pass through:

```python
except WorkflowSuspended:
    raise   # do NOT mark step failed
except asyncio.CancelledError:
    ...
except Exception as exc:
    ...
```

### 6.5 Resume flow

When `_execute` claims a `requires_action` run and re-enters
`agent.run(context)`:

1. `Context.__init__` loads existing steps, builds the ordinal counter map,
   and resolves any pending `requires_action`:
   - `wait_until`: if `wake_at <= now`, mark the linked sleep step
     `completed` and clear `required_action`.
   - `wait_for_signal`: pop the matching signal row, store payload as
     the step's `output`, mark `completed`, clear `required_action`.
2. User code re-runs from the top of `agent.run`.
3. Memoization fast-forwards through completed steps.
4. The previously-suspending `await context.sleep/wait_for_signal` now finds
   the step `completed` and returns its output instantly.
5. Execution continues past the suspend point.

---

## 7. HTTP/API surface

### 7.1 Signal delivery endpoint

```
POST /threads/{thread_id}/runs/{run_id}/signals
{ "name": "stripe.payment_received", "correlation_id": "ord_123",
  "payload": { ... } }
```

Implementation:

```python
await store.deliver_signal(run_id, name, correlation_id, payload)
# Best-effort wakeup (NOTIFY on PG, in-process queue elsewhere)
await dispatcher.notify_wakeup(run_id)
```

The dispatcher's pickup query already finds runs whose signal has arrived,
so wakeup notification is a latency optimization, not a correctness
requirement.

### 7.2 Existing endpoints unchanged

`POST .../submit_tool_outputs` continues to work. New `required_action`
types simply have different submit semantics.

### 7.3 Optional convenience client

```python
# Python client
client.runs.signal(thread_id, run_id, name="payment", payload={...})
```

---

## 8. Edge cases & policies

| Case | Handling |
|---|---|
| Worker crash mid-step | Step row stays `in_progress`. On replay, decorator marks it `failed` and re-executes. (Configurable: `@step(on_resume="retry"|"fail"|"skip")`.) |
| Network partition (worker alive, DB unreachable) | Heartbeat fails → `lease_lost` event → run task cancelled. Other worker takes over. Fencing epoch prevents stale writes. |
| Signal delivered before wait | `consume_signal` finds it on entry; no suspension. |
| Signal delivered to terminal run | `deliver_signal` returns 410 / no-op. |
| Multiple signals match | `consume_signal` returns the oldest unconsumed; document FIFO. |
| Wait timeout fires | Reaper picks up by `wake_at`; on resume, `consume_signal` returns None → step completes with `error=TimeoutError`, status `expired`. User sees `TimeoutError` from `await context.wait_for_signal`. |
| `run.expires_at` passes while suspended | Reaper transitions to `expired`; cancels any pending wait. |
| Cancel while suspended | API sets status `cancelling`; reaper transitions to `cancelled`, fires the `WorkflowSuspended` cleanup if any. |
| User mistakenly does long sync work | Heartbeat starves → lease lost → run is killed and resumed elsewhere. **Documented as user error**; no runtime guard for v1. |
| Replay determinism | Document: all non-determinism (UUIDs, timestamps, network) must live inside `@step`. No enforcement v1. |
| Schema changes mid-run | Document: in-flight runs replay against new code. Add `@step(version=N)` later. |

---

## 9. Configuration

New env / `LLamphouseConfig` fields:

```python
DURABLE_RUNS_ENABLED: bool = False         # opt-in for v1
LEASE_TTL_SECONDS: float = 30.0
HEARTBEAT_INTERVAL_SECONDS: float = 10.0   # default ttl/3
DISPATCHER_POLL_INTERVAL_SECONDS: float = 5.0
DISPATCHER_BATCH_SIZE: int = 32
WORKER_ID: str = "${HOSTNAME}:${PID}:${UUID}"
```

`DURABLE_RUNS_ENABLED=False` keeps current behavior: no leasing, no
memoization (sleep/wait_for raise `NotImplementedError`).

---

## 10. Backwards compatibility

- Existing `@step` users: zero behavioral change when feature flag off; with
  flag on, get free crash recovery via memoization.
- Existing `submit_tool_outputs` flow: unchanged. New `required_action.type`
  values are additive.
- OpenAI API compatibility: clients that look only for `submit_tool_outputs`
  will see other run states they don't understand but documented as
  vendor-specific.
- SQLite: leasing works (it has timestamp arithmetic). `NOTIFY`-style wakeup
  falls back to polling. Document slight latency cost.

---

## 11. Testing strategy

### Unit
- `claim_run` race: 50 workers vs 1 expired run → exactly 1 wins.
- Heartbeat renew → `lease_expires_at` advances; `lease_epoch` stable.
- Heartbeat fails → `LeaseHolder` cancels the wrapped task.
- Memoization: re-entering `run()` with completed steps returns stored
  outputs without invoking user code (assertable via mock).
- `consume_signal` FIFO ordering and idempotency.
- `WorkflowSuspended` propagates through `@step` without marking failure.

### Integration (Postgres)
- Kill worker process during a multi-step run → second worker resumes,
  no step executes twice.
- 24-hour sleep simulated with mocked clock → run resumes when wake_at
  passes.
- Signal delivered before `wait_for` → run does not suspend.
- Signal delivered during `wait_for` → run resumes within poll interval.
- Network partition simulation (block heartbeat queries) → lease lost,
  takeover within `lease_ttl + poll_interval`.

### Property-based
- For any sequence of crash points within a run, the final state of
  completed steps is identical to a crash-free execution.

---

## 12. Rollout phases

| Phase | Scope | Outcome |
|---|---|---|
| **P1 — Leasing only** | §3.1 schema, §4 leasing API, §5.1 `LeaseHolder`, §5.2 dispatcher claim/heartbeat. No memoization. | Crash detection. Crashed runs go to `failed` (not auto-resumed). |
| **P2 — Memoization** | §3.2 `call_key`, §6.1 decorator changes. Reaper auto-resumes. | True durable replay. |
| **P3 — Sleep** | §6.2, `wake_at`, dispatcher `wait_until` predicate. | Long-running workflows. |
| **P4 — Wait-for-signal** | §3.3 signals table, §6.3, §7.1 endpoint. | Event-driven workflows. |
| **P5 — Polish** | Graceful shutdown, NOTIFY-based wakeup, configurable on-resume policies, docs/examples. | Production-ready. |

Each phase is independently shippable and gated behind `DURABLE_RUNS_ENABLED`
or per-feature flags during the rollout.

---

## 13. Example (post-P4)

```python
class RefundAgent(Agent):
    @step
    async def validate(self, context, order_id: str) -> dict:
        return await db.fetch_order(order_id)

    @step
    async def request_approval(self, context, order: dict) -> str:
        await notify_slack(f"Approve refund for {order['id']}?")
        return "requested"

    async def run(self, context):
        order = await self.validate(context, context.input["order_id"])
        await self.request_approval(context, order)

        # Suspend until human clicks "approve" in Slack — could be hours.
        decision = await context.wait_for_signal(
            "refund.approved",
            correlation_id=order["id"],
            timeout=24 * 3600,
        )

        if decision["approved"]:
            await context.sleep(seconds=300)        # rate-limit window
            await self.issue_refund(context, order)

        await context.reply("done")
```

The same agent process can be killed and restarted at any point and the run
resumes correctly.

---

## 14. Open questions

- **Per-step retry policy** vs run-level: deferred to a later RFC.
- **Cross-tenant fairness**: dispatcher pulls oldest first; revisit if
  noisy-tenant problem appears.
- **Signal payload size**: cap at e.g. 256 KB; large payloads should be
  references to blob storage.
- **Idempotent signal delivery**: optional `signal_id` from caller for
  dedupe. Defer.
