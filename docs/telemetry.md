# Telemetry

LLAMPHouse ships with a tiny, **anonymous, non-blocking** telemetry
client that helps the maintainers understand which features are used in
the wild and where to focus development effort. It is **enabled by
default** but trivial to opt out of, and is designed so it can never
slow down or break your application.

This page documents exactly what is collected, how it is transmitted,
where it is stored, and how to disable, redirect, or self-host the
collector.

---

## TL;DR — Opt out

```bash
export LLAMPHOUSE_TELEMETRY=0    # disable (also accepts false / no / off)
# or
export NO_TRACKING=1              # alias (also accepts true / yes / on)
```

Either env var disables the client completely — no thread is started,
no network request is ever made.

## Tiers

Telemetry has two on tiers; the default sends lifecycle events plus a
single aggregated counter event every 5 minutes.

| Tier | Env var | What it sends |
|---|---|---|
| **off** | `LLAMPHOUSE_TELEMETRY=0` or `NO_TRACKING=1` | Nothing. No background thread is started. |
| **lifecycle** | `LLAMPHOUSE_TELEMETRY=lifecycle` (also `minimal`/`basic`) | `llamphouse_init`, `llamphouse_ignite`, `llamphouse_shutdown` only |
| **usage** _(default)_ | _(unset)_, `LLAMPHOUSE_TELEMETRY=1`, or `=usage`/`full`/`detailed`/`all` | Lifecycle **plus** a single aggregated `llamphouse_usage` event flushed every 5 minutes with thread/run counters and a run-duration histogram |

The usage tier is on by default because it produces only one extra
heartbeat event per active install per 5 minutes. It contains **only
counters** — no thread IDs, run IDs, agent names, model names, prompts,
or content of any kind. Set `LLAMPHOUSE_TELEMETRY=lifecycle` if you
want to drop even those counters.

---

## What is collected

For each lifecycle event the client emits a JSON object like this:

```json
{
  "event": "llamphouse_init",
  "ts": 1715300000.0,
  "install_id": "9f2c1b4e7a3d4f10b7d2c9a1e8f6b0d3",
  "session_id": "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
  "llamphouse_version": "1.2.3",
  "python_version": "3.11.6",
  "os": "Darwin",
  "arch": "arm64",
  "props": { "agents": 2, "auth": false }
}
```

| Field | Description |
|---|---|
| `event` | One of `llamphouse_init`, `llamphouse_ignite`, `llamphouse_shutdown` |
| `ts` | Unix timestamp (seconds) when the event was recorded |
| `install_id` | Random UUID stored once at `~/.llamphouse/telemetry_id` — stable across runs on the same machine |
| `session_id` | Random UUID generated per process — changes every time you start the app |
| `llamphouse_version` | The installed `llamphouse` package version |
| `python_version` | `platform.python_version()` |
| `os` | `platform.system()` (`Darwin`, `Linux`, `Windows`) |
| `arch` | `platform.machine()` (`arm64`, `x86_64`, …) |
| `props` | Small dict of event-specific counts/flags (e.g. number of agents, whether auth is enabled) |
| `tracking_id` | _Optional_ — only present if `LLAMPHOUSE_TRACKING_ID` is set to a valid UUID |

### What is **never** collected

- Agent class names, source code, prompts, or instructions
- Message content, conversation history, or tool inputs/outputs
- File paths, file contents, or workspace layout
- Environment variables or secrets
- API keys, model identifiers, or LLM provider URLs
- End-user data of any kind

Server-side, the source IP is truncated to a **/24 (IPv4)** or **/48
(IPv6)** prefix before being written to disk and stored alongside the
event as `_ip_prefix`.

---

## How it works

1. On the first event, a singleton client lazily starts a **daemon
   thread** with a bounded `queue.Queue(maxsize=500)`.
2. Producers (`telemetry.record(...)`) enqueue events with
   **drop-oldest** semantics — they never block.
3. The worker thread batches up to **20 events** or flushes every
   **30 seconds**, whichever comes first.
4. Batches are POSTed to the configured endpoint with a **3 s timeout**
   using only the standard library (`urllib.request`).
5. **Every error is silently swallowed** — no exception ever propagates
   into the host application.
6. On shutdown, `telemetry.shutdown()` flushes any pending events with
   a short timeout (best-effort).

The whole module depends only on the Python standard library — no extra
packages are pulled in by enabling it.

---

## Lifecycle events

| Event | Emitted when | Example `props` |
|---|---|---|
| `llamphouse_init` | After `LLAMPHouse.__init__()` completes | `{ "agents": 2, "adapters": 1, "auth": false, "data_store": "InMemoryDataStore" }` |
| `llamphouse_ignite` | When `ignite()` is called | `{ "host": "127.0.0.1", "port": 8000 }` |
| `llamphouse_shutdown` | During the FastAPI lifespan finalizer | `{}` |

Only the lifecycle events above are emitted at the lifecycle tier.
Per-run, per-message, or per-tool-call events are **not** sent.

## Usage event (default)

At the default `usage` tier, the worker thread additionally flushes a
single rollup event every 5 minutes (and once on shutdown). It
contains **only counters and bucketed durations** — no IDs, no names,
no content:

```json
{
  "event": "llamphouse_usage",
  "ts": 1715300300.0,
  "install_id": "…",
  "session_id": "…",
  "props": {
    "interval_s": 300.0,
    "counters": {
      "threads_created":  27,
      "runs_created":    142,
      "runs_completed":  138,
      "runs_failed":       3,
      "runs_cancelled":    1,
      "run_dur_lt_1s":    24,
      "run_dur_lt_10s":   95,
      "run_dur_lt_60s":   18,
      "run_dur_lt_300s":   1,
      "run_dur_gte_300s":  0
    }
  }
}
```

| Counter | Meaning |
|---|---|
| `threads_created` | Successful `data_store.insert_thread` calls in the window |
| `runs_created` | Successful `data_store.insert_run` calls in the window |
| `runs_completed` / `runs_failed` / `runs_cancelled` / `runs_expired` | Terminal run-status transitions |
| `run_dur_<bucket>` | Histogram of run wall-clock duration (started_at → terminal). Buckets: `lt_1s`, `lt_10s`, `lt_60s`, `lt_300s`, `gte_300s` |

If no counters were touched in a window (idle process), no usage event
is emitted.

---

## Configuration

All settings are environment variables — there is no Python API to
configure telemetry from inside your app (intentional, so it stays out
of the way).

| Variable | Default | Description |
|---|---|---|
| `LLAMPHOUSE_TELEMETRY` | `usage` | `0`/`false`/`no`/`off` disables. `lifecycle`/`minimal`/`basic` sends only the three lifecycle events. `1`/`true`/`yes`/`on` or `usage`/`full`/`detailed`/`all` enables lifecycle **plus** aggregated usage counters (the default). |
| `NO_TRACKING` | _(unset)_ | Alias to disable. Accepts `1`/`true`/`yes`/`on`. Overridden by an explicit `LLAMPHOUSE_TELEMETRY` value. |
| `LLAMPHOUSE_TELEMETRY_ENDPOINT` | `https://api.llamp.ai/telemetry` | Override the collector URL — useful for self-hosting. |
| `LLAMPHOUSE_TRACKING_ID` | _(unset)_ | Optional UUID added as a top-level `tracking_id` on every event. **Must** be a valid UUID; any other value is silently dropped. |

### Generating a tracking ID

```bash
python -c "import uuid; print(uuid.uuid4())"
# or
uuidgen
```

Setting a tracking ID is purely opt-in and only useful if you want to
correlate your own deployment's events with the maintainers (e.g. when
filing a bug report or in an enterprise support context). Leaving it
unset keeps the install pseudonymous (`install_id` only).

---

## Self-hosting the collector

A reference PHP collector lives in [`docker/telemetry/`](https://github.com/llamp-ai/llamphouse/tree/main/docker/telemetry).
It is intentionally tiny — no database, no dependencies — and writes
JSON Lines to disk.

### Files

```
docker/telemetry/
├── collect.php   # the receiver
├── .htaccess     # rewrites POST /telemetry → collect.php + CORS
└── README.md
```

### Storage layout

```
$LLAMPHOUSE_TELEMETRY_DIR/
├── events-YYYY-MM-DD.jsonl    # anonymous events, daily-rotated
└── tracked/
    ├── <uuid-1>.jsonl          # one append-only file per tracking_id
    ├── <uuid-2>.jsonl
    └── …
```

- Anonymous events (no `tracking_id`) go to the daily rotated file.
- Events with a valid `tracking_id` go **only** to `tracked/<uuid>.jsonl`.
- The two streams are disjoint. If `tracked/` cannot be created or
  written to, tracked events fall back to the anonymous log so nothing
  is lost.
- `LLAMPHOUSE_TELEMETRY_DIR` defaults to `/var/log/llamphouse-telemetry`
  and falls back to the system temp dir if it is not writable.
- The collector adds `_received_at`, `_ip_prefix`, and `_user_agent` to
  each line.

### Deploy

Drop the files into any PHP-FPM / Apache host:

```bash
sudo mkdir -p /var/log/llamphouse-telemetry
sudo chown www-data:www-data /var/log/llamphouse-telemetry
```

Then point the framework at your collector:

```bash
export LLAMPHOUSE_TELEMETRY_ENDPOINT=https://telemetry.example.com/telemetry
```

The collector caps the request body at **1 MB**, validates that the
`tracking_id` (if present) matches a strict UUID regex (preventing path
traversal), uses `flock(LOCK_EX)` for safe concurrent appends, and
always responds with `204 No Content`.

---

## Verifying it works

To confirm telemetry is disabled in your environment:

```bash
LLAMPHOUSE_TELEMETRY=0 python -c "
from llamphouse.core.telemetry import is_enabled
print('enabled:', is_enabled())
"
# enabled: False
```

To inspect what would be sent without sending anything, point the
endpoint at a local listener:

```bash
export LLAMPHOUSE_TELEMETRY_ENDPOINT=http://127.0.0.1:9999/telemetry
# In another shell:
python -m http.server 9999
```

---

## FAQ

**Does telemetry slow down my app?**
No. All work runs on a daemon thread behind a bounded queue. Producers
never block, and any network or serialization error is dropped.

**Is the install ID linked to me personally?**
No. It is a random UUID stored locally at `~/.llamphouse/telemetry_id`.
Delete the file to rotate it. There is no other identifier sent.

**Can I see the source?**
Yes — the entire client is a single file:
[`llamphouse/llamphouse/core/telemetry/client.py`](https://github.com/llamp-ai/llamphouse/blob/main/llamphouse/llamphouse/core/telemetry/client.py).

**How do I disable telemetry for an entire team?**
Set `NO_TRACKING=1` in your shared CI / container base image / shell
profile — e.g. add it to your `Dockerfile` `ENV` or `.envrc`.
