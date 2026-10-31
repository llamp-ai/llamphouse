# Threads and runs

Everything in llamphouse is a thread with one agent. This page is the model the rest builds on.

## The thread is the record

A thread belongs to one agent. Any number of people and services may talk to it, but only that agent
answers there. Other agents work on threads of their own (see [Working with other agents](multi-agent.md)).

A thread is an append-only log of events, each at a position (1, 2, 3, ...):

| event | what it is |
|---|---|
| `message` | something someone said: a person, the agent, a service, the system. A sender and typed parts (text, files, data, tool calls, artifacts) |
| `run.created`, `run.completed`, `run.failed`, `run.cancelled`, `run.rejected` | runs starting and ending |
| `run.input_required`, `run.auth_required`, `run.waiting` | a run waiting for someone or something |
| `state.changed` | a write to the thread's [state](state.md) |
| `x-...` | your own events, posted by the agent (`ctx.post`) |

The agent's working memory, its [session](sessions.md), is kept apart from the thread, with its own
log of entries and revisions.

Because the log is the truth, you can read a thread at any position, follow it live
(`GET /v2/threads/{id}/events/stream`), resume a stream by position without missing anything, and see in
Compass exactly what happened and in what order.

## A run is one piece of work

A run is the agent working on one message: its `ctx.input`. It starts `queued`, a worker picks it up
(`running`), it may wait, and it ends:

```
queued → running ⇄ waiting | input_required | auth_required → completed | failed | cancelled | rejected
```

- `waiting`: on the clock (`ctx.sleep`), a signal, other threads, or its inbox (`ctx.next`).
- `input_required`: it asked something (`ctx.ask`), or handed a tool call to the client.
- `auth_required`: it needs credentials or consent from the client.
- `completed`: `run` returned. A returned value is the run's output.
- `failed`: `run` raised. `AgentError(code, message)` gives clients a code to act on.
- `rejected`: the agent declined the work (`ctx.reject(reason)`). Terminal, not a failure.
- `cancelled`: someone cancelled it.

None of the waiting states holds a worker. A run can wait for days; when what it waits for arrives, a
worker picks it up again and it continues. See [Durable runs](durable-runs.md).

## At most one active run per thread

A run is active from `queued` until it ends. While a thread has an active run, it takes no new message
and no new run. This is the rule that keeps a conversation ordered: two runs never answer the same
thread at once. For parallel work, a run forks the thread (see [Forks](multi-agent.md#forks)).

Nothing starts a run by itself. A message on a thread doesn't wake the agent; something in front of it
(your app, an adapter, a trigger) decides to start a run.

## The primitives

Everything that happens to a thread from the outside is one of six things. The HTTP API has them one to
one:

| primitive | HTTP | when the thread is idle | when a run is active |
|---|---|---|---|
| **append** a message | `POST /v2/threads/{id}/messages` | on the thread, for the record; starts nothing | `409 thread_busy` (with the active run's id) |
| **start** a run with a message | `POST /v2/threads/{id}/runs` | a new run; the message is its `ctx.input` | `409 thread_busy` |
| **send** a message to the run | `POST /v2/runs/{id}/messages` | `409 run_ended` | in the run's inbox, waking it if it waits for one |
| **answer** its question | `POST /v2/runs/{id}/answer` | - | the run continues from its `ctx.ask`; the first answer wins |
| **signal** it | `POST /v2/runs/{id}/signal` `{"key", "value"}` | - | wakes a `ctx.run.wait_for_signal(key)` |
| **cancel** it | `POST /v2/runs/{id}/cancel` | - | ends it `cancelled`, and the runs it started |

A message body is `{"text": "..."}`, `{"data": {...}}` (structured data, for example a workflow's
input) or `{"parts": [...]}`, with an optional `"sender"`.

```bash
# a thread, with labels for your app, initial state, and config for this thread only
curl -X POST localhost:8000/v2/threads -H 'content-type: application/json' -d '{
  "agent": "support",
  "metadata": {"org": "acme", "user": "u42"},
  "state": {"plan": "pro"},
  "config": {"tone": "formal"}
}'
```

- `metadata`: your app's labels: who the thread is for. Listed and filtered by
  (`GET /v2/threads?metadata.org=acme`), read by the agent as `ctx.thread.metadata`, inherited by the
  threads its runs create.
- `state`: the thread's initial `ctx.state`.
- `config`: the agent's config for runs on this thread, over the agent file's, validated when the
  thread is made. One agent class serving many configurations.
- `session`: messages its first session starts from, for example the summary of an earlier thread
  this one continues.

## What a message means while a run is busy

When a person types while the agent works, llamphouse doesn't guess what they meant. The app in front
decides, and `thread_busy` tells it which run is active:

| the app wants to | it does |
|---|---|
| steer the run | **send** the message to the run |
| queue it | holds it, and **starts** the next run with it once the run ends |
| interrupt | **cancels** the run, then **starts** a new one |
| answer a question the run asked | **answers** the run |
| refuse | passes the 409 on to its client |

Each adapter makes this choice for its protocol. For example, the Responses adapter answers a run's
pending question with the next request's input.

## The inbox

What is sent to a run is its inbox. A message sent to a run goes on the thread when it is sent
(addressed to that run), and into the run's session when the run reads it:

```python
class Writer(Agent):
    """Drafts in sections, and takes notes from you while it works."""

    async def run(self, ctx) -> None:
        for section in ("intro", "body", "summary"):
            note = await ctx.next(timeout=0)  # a message sent to this run, if any: doesn't wait
            if note is not None:
                await ctx.send(f"Noted: {note.message.text}")
            await ctx.send(f"Writing the {section}.")
            await ctx.sleep(60)  # slow work, on the clock: no worker held
```

`ctx.next()` returns the next message sent to the run, waiting for one if there is none yet (without
holding a worker); `timeout=0` only looks. Reading records a `message.read` event on the run, so Compass
shows "sent 10:01, read 10:03". A message the run never reads stays on the thread, unread; the run's
`unread` list in the API shows it, and the app decides what to do with it (start the next run with it,
tell the user).

`ctx.next(*threads)` also waits on threads the run created: whichever comes first. See
[Working with other agents](multi-agent.md#listening).

## Hearing what happens

- A thread's events: `GET /v2/threads/{id}/events?after=N`, live as `.../events/stream`. In Python,
  `engine.follow(thread, after=position)`.
- A run's live stream (text deltas, progress, usage, your own events): `GET /v2/runs/{id}/events/stream`.
- Every run's status changes in the deployment: `GET /v2/stream/runs?agent=...&metadata.org=...`.
- Webhooks to other systems on run status changes: subscriptions (`POST /v2/subscriptions`), delivered
  from an outbox with retries.

## Next

- [Agents](agents.md): the class behind a thread.
- [The run context](context.md): what a run can do.
- [Durable runs](durable-runs.md): how waiting works.
