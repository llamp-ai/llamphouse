# HTTP API

The native API is served under `/v2` by every API process (`llamphouse up --role api` or `all`). It offers the
engine's primitives over HTTP: threads, the messages on them, runs, their streams, files, agents and their
config, and subscriptions. Other protocols (OpenAI Responses and Assistants, A2A, AG-UI) are mounted next to it
as [adapters](guides/adapters.md), on the same threads and runs.

The API holds no state. Any number of API processes can serve one database.

## The model in one paragraph

Everything is a **thread** with one agent. A thread has at most one **active run**. Nothing starts a run by
itself: the app decides. There are three ways to put a message on a thread:

- **append** (`POST /v2/threads/{id}/messages`): for the record, while no run is active;
- **start** (`POST /v2/threads/{id}/runs`): the message is the input of a new run, while no run is active;
- **send** (`POST /v2/runs/{id}/messages`): to the active run's inbox, while it runs.

What a message means while a run is busy (start the next turn after it, steer it, answer its question, refuse)
is the app's decision. The API answers `409` instead of guessing.

## Authentication

Requests carry whatever the configured authenticators expect, usually `Authorization: Bearer <key or token>`
(see [Auth](guides/auth.md)). For an `EventSource`, which cannot send headers, an API key can go in the query
string: `?key=<key>`. With no authenticator configured, every caller is `anonymous` and may do everything.

A thread the caller may not touch answers `404`, the same as a thread that does not exist.

## Messages in requests

Wherever a request carries a message, its body is one of:

```json
{"text": "Compare Bangkok and Lisbon"}
{"data": {"city": "Lisbon", "month": "march"}}
{"parts": [{"kind": "text", "text": "Read this"}, {"kind": "file", "mime_type": "application/pdf", "uri": "files://f_..."}]}
```

with an optional `"sender": {"kind": "person", "id": "an", "name": "An"}`. Without one, the sender is the caller
(a person `user` when nobody authenticates). An authenticated caller is always the sender of its messages; only a
caller whose identity allows it (`metadata.act_as`) may name another sender.

Parts are in the SDK's wire format: `text` (`text`), `data` (`data`, optional `schema`), `file` (`mime_type`,
`name`, `uri`, or inline bytes as `"data": {"$bytes": "<base64>"}`), `tool_call`, `tool_result`, `artifact`.
Inline file bytes are moved to the file store before the message is written; a stored message never holds bytes.

## Messages in responses

Messages come back as:

```json
{
  "id": "msg_0ab7e263b82440308783e808f1557225",
  "sender": {"kind": "agent", "id": "talker", "name": null, "run": "run_31bfb3533f404fc9"},
  "parts": [{"kind": "text", "metadata": {}, "text": "Hey, world"}],
  "references": [],
  "metadata": {}
}
```

`sender.kind` is `person`, `agent`, `service`, `trigger` or `system`; `sender.run` is set for messages an
agent's run wrote.

## Threads

### `POST /v2/threads`

A new thread. `201` with the thread.

```json
{
  "agent": "desk",
  "metadata": {"org": "acme", "user": "an"},
  "state": {"currency": "EUR"},
  "config": {"model": "gpt-5"},
  "session": [{"text": "Earlier we agreed on March."}]
}
```

| Field | |
|---|---|
| `agent` | required: the agent's name |
| `metadata` | the app's labels for the thread (who it is for). Threads its runs create inherit them. Filterable in listings; agent code reads them as `ctx.thread.metadata`. The access policy adds its own stamp (`llamphouse.tenant`, `llamphouse.created_by`), which a caller cannot forge. |
| `state` | the thread's initial state (`ctx.state`) |
| `config` | the agent's config for runs on this thread, over the file's and the config store's; validated now. See [Config](guides/config.md). |
| `session` | messages the thread's first session starts from, e.g. a summary of a thread it continues |

`404` when there is no such agent; `422` when `config` doesn't validate.

### `GET /v2/threads`

Threads, newest first. Query: `agent`, `top=true` (only threads nobody's run created), `limit` (50),
`metadata.<key>=<value>` (any number). A caller under tenant access sees only its tenant's threads.

### `GET /v2/threads/{id}`

```json
{
  "id": "thread_4b7e094a26634203", "agent": "talker", "created_at": "2026-10-31T10:57:16.167718+00:00",
  "forked_from": null, "created_by": null, "closed": false, "last_position": 7,
  "metadata": {"org": "acme"}, "config": {"greeting": "Hey"},
  "active_run": null, "runs": ["run_31bfb3533f404fc9"], "created": []
}
```

`active_run` is the full run (below) while one is active. `created` lists the threads its runs created (forks,
threads of other agents). `created_by` is the run that created it, if any.

### `POST /v2/threads/{id}/messages`

Append a message. `201` with the message. `409 {"error": "thread_busy", "run": "<id>"}` while a run is active;
`409 thread_closed` on a closed thread.

### `POST /v2/threads/{id}/runs`

Start a run with this message as its input. `201` with the run (status `queued`). `409 thread_busy` while a run
is active. For a [Workflow](concepts/agents.md#workflow), send its input as `{"data": {...}}`; it is validated
against the workflow's `Input` before the run exists (`422` if it doesn't fit).

With `"stream": true` in the body, the answer is the run's stream from its first event (server-sent events,
status `201`): a `run` event with the run, then everything it does - `message.delta` text included - and `end`
when it ends. Nothing is missed, as nothing happens before the stream listens. If the run can't start (a run is
active, the thread is closed) the stream has one `error` event; a bad workflow input is still `422`.

### `POST /v2/runs`

A thread and its first run in one call: `{"agent": "...", <the message>, "metadata": {...}, "state": {...},
"config": {...}, "stream": false}`. The input is checked first: a bad one is `422` and nothing is made.
`201` with the run, or its stream with `"stream": true`.

### `GET /v2/threads/{id}/runs`

The thread's runs, oldest first.

### `POST /v2/threads/{id}/close`

No more messages: later appends, starts and sends answer `409 thread_closed`. Its active run (and what that run
started) is cancelled. Reading and streaming still work.

### `GET /v2/threads/{id}/events?after=N`

The thread's record: every event after position `N`, in order.

```json
{"position": 2, "kind": "run.created", "thread": "thread_...", "run": "run_...", "agent": "talker",
 "to": null, "at": "2026-10-31T10:57:16.168407+00:00", "message": null, "data": {"agent": "talker", "parent": null}}
```

Kinds include `message`, `run.created`, `run.waiting`, `run.input_required`, `run.input_answered`,
`run.completed`, `run.failed`, `run.cancelled`, `run.rejected`, `state.changed`, `session.*`, and custom
`x-...` kinds that agents post. `run.input_answered` has the answer in its data (`{"key", "by", "answer"}`;
never for an auth request: credentials aren't kept). `to` is set on a message sent to one run's inbox.

### `GET /v2/threads/{id}/events/stream?after=N`

The same, live, as server-sent events: what is stored after `N` first, then each new event as it is written.
Each SSE event has `id: <position>` and `event: <kind>`. Reconnect with `?after=<last id>` and nothing is
missed. A `: keep-alive` comment is sent every 15 seconds without news.

### `GET /v2/threads/{id}/state?at=N`

The thread's state, now or as it was at position `N`: `{"key": {"value": ..., "version": n}}`.

## Runs

### The run object

```json
{
  "id": "run_31bfb3533f404fc9", "thread": "thread_4b7e094a26634203", "agent": "talker",
  "status": "input_required", "parent": null,
  "input": {"id": "msg_...", "sender": {"kind": "person", "id": "user", "name": null, "run": null},
            "parts": [{"kind": "text", "metadata": {}, "text": "world"}], "references": [], "metadata": {}},
  "output": null, "error": null,
  "pending": {"reason": "Your name?", "prompt": [{"kind": "text", "metadata": {}, "text": "Your name?"}],
              "input_schema": null, "metadata": {}},
  "activations": 1, "children": [], "created": [], "unread": []
}
```

| Field | |
|---|---|
| `status` | `queued`, `running`, `waiting` (on the clock, a signal, other threads), `input_required`, `auth_required`, `completed`, `failed`, `cancelled`, `rejected` |
| `parent` | the run that started this one, for runs on forks and threads of other agents |
| `output` | what `run()` returned, once completed (typed by its return annotation) |
| `error` | why it failed or was rejected |
| `pending` | what it asks while `input_required` (`reason`, `prompt`, `input_schema`) or `auth_required` |
| `activations` | how many times it was executed (it executes again after each wait, and after a takeover) |
| `children`, `created` | runs it started; threads it created |
| `unread` | messages sent to its inbox it has not read yet |

### `GET /v2/runs/{id}`

The run.

### `POST /v2/runs/{id}/messages`

Send a message to the run's inbox (`ctx.next()` in agent code). `201` with the run. `409 {"error":
"run_ended"}` once it is not the thread's active run.

### `POST /v2/runs/{id}/answer`

Answer the run's question (`ctx.ask`) or auth request. `{"accepted": true, "run": {...}}`; `accepted` is false
when it was answered already: the first answer wins.

### `POST /v2/runs/{id}/signal`

`{"key": "approved", "value": true}`: wakes a run waiting in `ctx.run.wait_for_signal("approved")`.

### `POST /v2/runs/{id}/cancel`

Cancel the run and the runs it started.

### `GET /v2/runs/{id}/events?after=N`

The run's stream as stored: `message.started`, `message.completed`, `progress`, `usage`, `state.snapshot`,
`state.delta`, `step.*`, `thread.created`, custom `x-...` events, each with a `seq`. Text deltas are not stored.

```json
{"seq": 4, "kind": "message.completed", "metadata": {}, "message": {"id": "msg_...", "parts": [...], ...}}
```

### `GET /v2/runs/{id}/events/stream?after=N`

The run's stream live (SSE), including `message.delta` events with the text as it is written. What is stored is
sent first, then live events; the stream ends with `event: end`, `data: {"status": "completed"}` when the run
ends. Resume with `?after=<last seq>`. See [Streaming](guides/streaming.md).

### `GET /v2/runs/{id}/session`

The session the run worked with: its messages (each marked `own` when the agent wrote it), version, revisions,
and the session's log.

### `GET /v2/runs/{id}/trace`

The run's spans from the tracing store, every execution of it: `{"run": "<id>", "tracing": true, "spans":
[...]}`. Without a tracing store: `"tracing": false` and no spans. `502` when the store can't be reached. See
[Tracing](guides/tracing.md).

## Every run, live

### `GET /v2/stream/runs?agent=&metadata.<key>=<value>`

Every run's status changes in the deployment, live (SSE), filtered by agent and thread labels and narrowed to
what the caller may see:

```text
event: run.status
data: {"run": "run_...", "thread": "thread_...", "agent": "desk", "status": "completed", "error": null,
       "output": {...}, "parent": null, "metadata": {"org": "acme"}}
```

For a party that can't hold a stream open, use a [subscription](guides/subscriptions.md).

## Files

### `POST /v2/files?name=report.pdf`

Upload first, reference later. The body is the bytes, with their `Content-Type`. `201` with a file part:

```json
{"kind": "file", "metadata": {}, "mime_type": "text/plain", "name": "a.txt", "uri": "files://f_795e267dee944611",
 "size": 5, "sha256": "2cf24dba5fb0...", "data": null}
```

Put it in a message's `parts`. An upload nobody references expires (`upload_ttl`, an hour by default).

### `GET /v2/files/{id}`

The bytes, with the file's content type and name. A file in a message is readable by whoever may read the
thread; an upload nobody referenced yet, by its uploader. See [Files](guides/files.md).

## Agents

### `GET /v2/agents`

What can be talked to:

```json
[{"name": "talker", "type": "agent", "description": "Streams a greeting, then asks a question.",
  "version": null, "requires": [], "config_schema": {...}, "input_schema": null, "output_schema": null,
  "metadata": {}}]
```

`type` is `agent` or `workflow`; `input_schema` is a workflow's `Input`; `output_schema` comes from `run()`'s
return annotation.

### `GET /v2/agents/{name}/config`

The agent's config as runs get it, and what was changed at runtime:

```json
{"agent": "talker", "config": {"greeting": "Yo", "model": "gpt-5-mini"}, "changed": {"greeting": "Yo"},
 "version": 1, "updated_by": "anonymous", "updated_at": "2026-10-31T10:57:16.988690+00:00", "schema": {...}}
```

### `PUT /v2/agents/{name}/config`

`{"values": {"greeting": "Yo"}, "if_version": 0}`: replaces what the config store holds for the agent (`{}`
goes back to the file's), for every process, from their next run on. Validated against the agent's `Config`
(`422`). `409 config_conflict` when `if_version` is not the stored version. `403` when the caller may not
change it (under tenant access, only admins may). See [Config](guides/config.md).

## Subscriptions

| | |
|---|---|
| `POST /v2/subscriptions` | `{"url", "filter", "secret", "headers"}`: a URL that hears about run status changes. `201` with the subscription (without its secret). |
| `GET /v2/subscriptions` | the caller's subscriptions |
| `DELETE /v2/subscriptions/{id}` | |
| `GET /v2/subscriptions/{id}/deliveries?limit=50` | what was sent, what failed |
| `POST /v2/subscriptions/{id}/deliveries/{delivery}/retry` | send a failed or dead delivery again |

See [Subscriptions](guides/subscriptions.md).

## Other routes

| | |
|---|---|
| `GET /health` | `{"ok": true, "engine": "<id>", "workers": n}`, without authentication |
| `POST <webhook path>` | the webhooks agent files declare, outside `/v2`, each with its own secret. See [Triggers](guides/triggers.md). |
| `/compass`, `/v2/compass/*` | Compass and the API it reads. See [Compass](guides/compass.md). |
| adapters | under their own prefixes. See [Adapters](guides/adapters.md). |

## Errors

| Status | Body | When |
|---|---|---|
| `401` | `{"detail": "..."}` | credentials missing or wrong |
| `403` | `{"detail": "..."}` | an authenticated caller may not do this |
| `404` | `{"detail": "no thread ..."}` | it doesn't exist, or the caller may not see it |
| `409` | `{"error": "thread_busy", "detail", "run"}` | a run is active on the thread |
| `409` | `{"error": "no_active_run" \| "run_ended"}` | sending to a run that isn't active |
| `409` | `{"error": "thread_closed"}` | the thread is closed |
| `409` | `{"error": "config_conflict"}` | `if_version` doesn't match |
| `422` | `{"error": "invalid_input", "detail": [...]}` | a workflow input or a config that doesn't validate |
| `503` | `{"error": "store_unavailable"}` | a transaction could not be written; nothing of it happened |

## A session with curl

```bash
curl -s localhost:8000/v2/threads -H 'content-type: application/json' -d '{"agent": "desk"}'
# {"id": "thread_...", ...}
curl -s localhost:8000/v2/threads/thread_.../runs -H 'content-type: application/json' \
     -d '{"text": "Compare Bangkok and Lisbon"}'
# {"id": "run_...", "status": "queued", ...}
curl -N localhost:8000/v2/runs/run_.../events/stream     # watch it, live
curl -s localhost:8000/v2/runs/run_.../answer -H 'content-type: application/json' -d '{"text": "March"}'
```

See [Example 10_Server](../examples/10_Server) for a client that makes these calls with httpx.
