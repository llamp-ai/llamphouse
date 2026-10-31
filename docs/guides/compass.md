# Compass

Compass is the web app that comes with every llamphouse API process, at `/compass`. It is for the people who
build and run agents: to see what the agents do, to find out why a run is where it is, to try an agent and change
its config, and to learn the ideas of 2.0 on your own data.

```bash
llamphouse up --memory --agents agents.yaml     # then open http://127.0.0.1:8000/compass
```

Compass is built into the package: no CDN, no build step, it works offline. `--no-compass` leaves it out. It
reads the API at `/v2/compass` (and the rest of `/v2`), behind the deployment's auth and access: a caller sees
what it may see, and under tenant access a tenant sees only its own threads and runs.

When the deployment requires an API key, Compass asks for one and keeps it in the browser; you can also open
`/compass?key=<key>`.

## Overview

The home page answers "is everything all right?":

- how many threads and runs, and runs per hour by status;
- who is waiting for someone: runs that asked a question or need auth, and for how long;
- what failed recently, with the error;
- per agent: runs and p50 / p95 durations;
- this deployment's components (data store, queues, file store, tracing, adapters) and who you are.

The box at the top jumps to any id: a thread, a run, a Responses API response, a session.

## Why

For any run, Compass says in plain words why it is where it is:

- **queued**: when it was rung and when it will be rung again;
- **running**: which worker holds its lease, and until when;
- **waiting**: on what: other threads (which), a signal (which key), its inbox, a time;
- **asking**: the question it asked, or the client tool it called, and since when;
- **ended**: completed, handed over, cancelled, failed (with the error), rejected (with the reason);

and how many times it was executed (each wait means another execution).

## A run's story

A run's page tells what it did, in order: its input; what it said and asked, and what it was answered; the state
it changed; the threads and runs it started; the messages sent to it and when it read them. Next to the story:

- **Steps**: each `ctx.step`, recorded or replayed;
- **Waits**: what it waits on now;
- **Tree**: the runs from the root of the conversation down (forks, calls, handovers);
- **Session**: the messages the agent saw, with the session's revisions;
- **Trace**: its spans as a tree, with durations, status and attributes; model input and output
  (`gen_ai.*`, `input.value`, `output.value`) shown readable and foldable (see [Tracing](tracing.md));
- **Raw**: the run record as stored.

## Threads

A thread's page shows the conversation, live; the timeline of every event on it; the state at any position (pick
a position on the timeline); the tree of threads it created; and its runs.

At the bottom, a composer offers exactly what the thread allows right now:

- no active run: **append** a message, or **start** a run with it;
- an active run: **send** to its inbox, **answer** its question, **signal** it, or **cancel** it.

So you can drive any agent by hand the way an app would, and see what each primitive does.

## Agents

Each agent's page has:

- **Try it**: start a run on a new thread; for a workflow, a form built from its input schema;
- **Config**: the effective config, what differs from the agent file, and a form to change it or reset it
  for every process (see [Config](config.md));
- **Triggers**: its webhooks with a ready `curl` call, and its schedules with their next time and a
  "fire now" button (see [Triggers](triggers.md));
- its input, output and config schemas, its permissions and its session provider;
- its numbers: runs, failures, durations.

## Runs and traces

**Runs** lists runs across threads, filtered by agent and status. Pick two to **compare** them side by side
(input, story, steps, output, how long each took, how many executions).

**Traces** lists recent traces by agent, when a tracing store is configured.

## Triggers and subscriptions

**Triggers** lists every webhook and schedule of the deployment, with when each schedule fires next.

**Subscriptions** lists the URLs that hear about runs (see [Subscriptions](subscriptions.md)): their filters,
their deliveries (sent, failed, dead) and a button to retry one.

## Dashboards

Dashboards are pages of charts on SQL queries. A chart is one read-only query (`SELECT`, `WITH` or `EXPLAIN`)
and how to draw its result.

```sql
select agent, status, count(*) as runs
from llamphouse_runs
where created_at > now() - interval '7 days'
group by agent, status
order by runs desc
```

- Queries need the Postgres data store, and run on the data store's database.
- Queries read every tenant's data, so only an admin runs them: anyone under the default access, the
  `admin_subjects` under tenant access.
- Dashboards and charts are kept in the data store (`llamphouse_documents`), so every API process shows the same
  ones. `llamphouse migrate --dashboards --charts` brings 1.x Compass dashboards over.
- A dashboard can be shown full screen (present mode).

The 2.0 tables are `llamphouse_threads`, `llamphouse_runs` (with `created_at` and `ended_at`),
`llamphouse_events`, `llamphouse_run_events`, `llamphouse_sessions`, `llamphouse_files`,
`llamphouse_schedules`, `llamphouse_subscriptions`, `llamphouse_outbox` and `llamphouse_traces`; most of a
record is in its `data` JSON column.

## Learn

**Learn** walks through each idea of 2.0: threads, the primitives (append, start, send), the inbox, asking,
forks, calls and handovers, thread state, sessions, durability (steps and replay), thread and run events, config,
triggers and the scheduler, subscriptions. Each comes with a short piece of code and a link to where you can see
it in your own deployment.

## The API behind it

Compass's own endpoints, behind the same auth:

| | |
|---|---|
| `GET /v2/compass/info` | this deployment: version, components, adapters, the caller |
| `GET /v2/compass/overview?hours=24` | counts, statuses over time, per agent; what needs attention |
| `GET /v2/compass/lookup/{id}` | what any id is, and where |
| `GET /v2/compass/runs?agent=&status=` | runs across threads, newest first |
| `GET /v2/compass/runs/{id}/debug` | why it is where it is, its story, steps, waits, trees |
| `GET /v2/compass/threads/{id}/tree` | the threads created from a thread |
| `GET /v2/compass/agents[/{name}]` | agents with their config, permissions, triggers and numbers |
| `GET /v2/compass/traces[?agent=]`, `/v2/compass/traces/{run}` | recent traces; a run's spans |
| `GET /v2/compass/schedules`, `POST /v2/compass/schedules/{id}/fire` | schedules; fire one now |
| `GET /v2/compass/webhooks` | the webhooks agent files declare, with an example call |
| `GET/POST/PUT/DELETE /v2/compass/charts[/{id}]`, `/v2/compass/dashboards[/{id}]` | dashboards and charts |
| `POST /v2/compass/query` | `{"sql": "...", "limit": 1000}`: a read-only query (admins, Postgres) |

These are for Compass and may change between alphas; build apps on [the `/v2` API](../http-api.md).

## See also

- [Example 05_ResearchDesk](../../examples/05_ResearchDesk): forks, questions and steering to look at in Compass
- [Tracing](tracing.md)
