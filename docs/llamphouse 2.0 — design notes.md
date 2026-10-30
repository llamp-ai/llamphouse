# llamphouse 2.0 — design notes

Oct 22, 2026 · @Pieter van der Deen

## Purpose and principles

llamphouse 2.0 is the backbone that runs agents; it does not decide the rules of the agent game. Positioning line: *llamphouse runs your agents; it doesn't tell them how to play.*

- **Mechanism, not policy.** The runtime provides primitives (threads, runs, sessions, events, queues, scheduling, identity). Coordination, turn-taking and context strategy live in agent code or optional helpers.
- **Stable agent contract.** Agent code written against the SDK keeps running for years. When external APIs change, only adapters change.
- **Protocol-neutral core.** OpenAI Responses, A2A, webhooks and anything later are adapters that translate at the edge.
- **Same agents, any substrate.** Local single process, Docker Compose, Container Apps or Kubernetes, from the same agent files.
- **Workload scaling by default.** A shared worker pool stays the default; dedicated pools are an explicit opt-in.
- **The backbone enforces operational safety only:** ordering, versioning, permissions, limits, cancellation propagation.
- **One agent per thread.** Agents are processes: each owns its thread, works in parallel on forks of it, and talks to other agents by messages on their threads. Runs talk to threads, never to runs. Shared multi-agent conversations are a layer on top (see "One agent per thread", "Threads, forks and state").

## Where 1.3 stands today

The foundations exist; prod separation is hard because the YAML path and the role commands are two separate worlds. Findings from `llamp-ai/llamphouse` main/dev (release 1.3.0).

Already in place:

- `definitions` (agent types) and `agents` (instances) in `llamphouse.yaml`, with per-instance `config`, `env`, `secrets`, `execution`, `triggers`, `context.identity`.
- Config params declared on the class (`StringParam` etc.), resolved by the config store.
- Role commands `serve --no-workers`, `worker`, `compass`; `docker-compose.prod.yml` runs them as three services from one image.
- Redis Streams queue, Redis event queue, Postgres data store with Alembic, ClickHouse tracing.

Why the split is hard:

1. `up` reads YAML, but `worker` and `compass` only accept a code path (`app:app`).
2. The YAML loader never configures the run queue, event queue or config store, so they fall back to in-memory. `up --no-workers` from YAML enqueues into a queue no worker can reach.
3. The config store is in-memory only, so Compass config edits don't reach other processes. Standalone Compass also gets its own in-memory event queues.
4. `worker` and `compass` build the full API app and reach into `app.fastapi.state`.
5. One global stream and one consumer group; the `assistant_ids` filter in `RedisQueue.dequeue` doesn't skip messages, so every worker must load every agent.
6. All dependencies (Postgres drivers, Redis, OpenTelemetry) are mandatory; no extras.
7. The headline API is the OpenAI Assistants API, which OpenAI shut down on August 26, 2026.

What 1.4 adds before 2.0 (the last 1.x feature release):

- **Durable runs:** steps replayed by call key, leases with heartbeat and recovery, `context.sleep`, `wait_for_signal`, `wait_for_input` (a waiting run holds no worker), retry, durable LangGraph runs with a checkpointer in the data store.
- **AG-UI adapter:** streamed runs, shared state (`emit_state`), interrupts mapped onto `wait_for_input`.
- **Bug fixes** found along the way.

Sessions are not in 1.4: they land in 2.0, on the new core model (the 1.x prototype lives on the `feature/sessions` branch).

## 2.0 scope

Rule: 2.0 breaks contracts once; features that are purely additive wait for 2.x.

| Area | In 2.0 (breaking) | Later in 2.x (additive) |
| --- | --- | --- |
| API | Responses + Conversations as primary adapter, A2A and AG-UI kept, Assistants dropped | MCP exposure, more adapters |
| Core model | Threads, sessions, runs, events; 1.4's durable runs ported onto the event log (completed steps as events with their call key) | |
| Agent contract | `llamphouse.sdk`: typed config, `RunContext`, open event and part kinds, `FakeRun` test kit, no Assistants concepts | Pattern helpers (fan-out, supervisor, handover) |
| Config | Agent files split from runtime config; `apiVersion`/`kind` | API-created agents |
| Runtime | Config-driven roles, complete backbone in config, shared config store, file store | Kubernetes operator and CRDs |
| Queue | Run queue interface with `pool` (2.0 runs every pool as `shared`); scheduler interface with in-memory and Postgres backends | Stream per pool; Redis and Service Bus schedulers; sandboxes and mounts |
| Compass | Thread timeline (conversation and detailed views), run view, session revisions, workflow views (list, step timeline, start form) | Per-step metrics across runs |
| Packaging | Base install runs `llamphouse up` with in-memory backends; extras for the other backends | Official base image; `llamphouse build/deploy` |
| Name | Keep llamphouse | |

The agent file accepts `pool:` from 2.0 on, so per-pool queues land in 2.x without a format change.

Migration support: `llamphouse migrate` for YAML and database, a 1.x maintenance branch, a migration guide, rewritten examples, a `2.0.0a` pre-release.

## Migration from 1.x

`llamphouse migrate` converts the config and carries over the whole database, so the audit history survives the upgrade:

| 1.x | 2.0 |
| --- | --- |
| `llamphouse.yaml` `definitions` / `agents` | agent files (`AgentDefinition`, `Agent`) + a runtime file |
| thread | thread (same id, metadata kept) |
| message | `message` event (sender and parts: the 1.x role becomes a sender kind), position by `created_at` |
| run | run: `queued` → `queued`, `in_progress` → `running`, `requires_action` / waiting for input → `input_required`, parked on a timer or signal → `waiting`, `completed`, `failed`, `cancelled`, `expired` → `failed` |
| run step | events of its run: `tool.call` / `tool.result`, `message` links, `step.completed` with its call key (the 1.4 replay record) |
| signals, key-value records (LangGraph checkpoints) | kept as they are |
| tracing spans | not touched (tracing store) |

- Re-runnable: every migrated event carries the idempotency key `migrate:<1.x id>`, so a second run adds only what is new.
- In-flight runs (queued, running, waiting) are migrated as records, but their agents changed contract: `migrate` stops when it finds any, unless `--include-in-flight`. Drain the workers first; the migration guide says how.
- `--dry-run` reports the counts per type and what would not map.

## One agent per thread

> **Decision (Oct 29, 2026).** A thread belongs to one agent. Any number of people and callers may talk to it; the agent may work in parallel runs of itself; other agents are callers and callees, never co-owners of a thread. Conversations shared by several agents and users are a layer on top of llamphouse. This supersedes the multi-agent-on-one-thread design below; the sections it changes are listed at the end and are revised next.

**Why.** Almost every bug the rig and the scenario tests found came from several agents sharing one thread: an agent looping on its own earlier reply, two agents answering each other forever, "unread" meaning different things to different subscriptions, sessions out of order, "outside my run tree" across agents. A thread shared by several agents is a distributed system with shared memory — notoriously hard to get right. And who speaks when, who sees what, who moderates, is policy: product behaviour that differs per application. The protocols agree: in A2A a context belongs to the agent serving it and other agents are callers; AG-UI is one agent per thread.

**Agents as processes.** The model is Erlang/OTP's — share nothing, talk by messages, supervise in trees — the proven answer to distributed, long-running, failure-prone work, which is what agents do:

| Erlang | llamphouse |
| --- | --- |
| a process with its own state | an agent on its own thread, with its own session and state |
| its mailbox | the thread's inbox: people and callers send it messages |
| spawning workers | the agent starts parallel runs of itself |
| a supervision tree | a run supervises its runs: waits on them, steers, retries, cancels |
| share nothing, talk by messages | other agents are callers and callees, each on its own thread |
| a crash, restarted by its supervisor | leases, replay, recovery |

llamphouse is the runtime for these processes; multi-agent systems are built from them.

**What it means:**

- A thread has one agent and any number of people and callers (several users on one thread is fine: each is a `Sender`).
- **Parallel work of the same agent** stays: on forks of its thread (`ctx.fork`, see "Threads, forks and state"), supervised with waits, steering, `cancel` and `restart`.
- **Calling another agent** is sending it a message on its own thread — internal or through a gateway, with the same semantics either way; internal is a faster route, not different behaviour. Orchestration and permissions between agents can then live in a gateway outside llamphouse.
- Viewpoints become simple: this agent is OWN; people and callers are OTHER. "Outside" is "not this agent".
- Durability, workflows, sessions (one per agent per thread, plus a child run's own), files, artifacts, hub and spokes, external callers and agents: unchanged.

**What goes or shrinks:** thread participants and group-chat routing; the cross-tree subtleties of `Origin.OUTSIDE`; `ChildThread.SAME` for other agents; shared `ctx.state` and posts between different agents; most `Visibility` options.

**Shared conversations are a layer on top** — e.g. an application's rooms: several agents and users in one conversation. The room is the policy (participants, who sees what, turn-taking, moderation); each agent keeps its own thread per room; the room relays new messages to the agents that should see them and publishes their replies, which come back through subscriptions or webhooks. The room can be an app, or a moderator agent that owns the room's thread and calls the participating agents as callees — still one agent per thread. Each agent stays simple and testable: it has a conversation and answers; it doesn't know it is in a room. llamphouse gives the room what it needs: per-agent threads, streaming, subscriptions and webhooks, sender identities, durability. Rooms cost more round trips than a shared thread, and in exchange the room decides what each agent sees. It belongs in an optional package or a reference example, not the core.

**Revised accordingly** (and since revised again, see "Threads, forks and state"): the agent contract (`Filter`, `Origin`, `AgentCall` without `sees` and `escalate`, `ChildThread` and `ChildSession` for runs of the same agent, `ChildRun.thread`), "Parallel runs and calling other agents" (formerly multi-agent on one thread), sessions of child runs, where state lives, hub and spokes; in code the SDK, the `Scenario` runtime and its tests — the group-chat test is now a room built on top (`Room` in the scenario tests), and handover passes the thread to another agent.

## Threads, forks and state

> **Decision (Oct 29, 2026).** Everything is a thread with one agent, and runs talk to threads, never to runs. A run writes only on its own thread. It can create threads — a **fork** of its own thread (`ctx.fork`), or a new thread of **another agent** (`ctx.call`) — and gets a `Thread` handle to send to its run (`send`), start its next run (`start`) and listen to it (`subscribe`, `wait`, `watch`). Nothing starts a run by itself (see "Starting runs and the inbox"). Nothing comes back by itself: the creator listens, and reads what it wants from a fork's thread and state. Thread state is a sequence of events, so it can be shown at any position and forks with the thread. A run keeps its own bookkeeping in `ctx.run.state`; `restart()` takes no argument. Tested in the scenario tests, shown in `examples/v2/01_ResearchDesk` and `02_CodingAgent`. This replaces `AgentCall`, `ChildRun`, `ChildThread` (`SAME`, `FORK`, `NEW`), `ChildSession`, `ctx.send(..., to=)` and `ctx.agents.start/invoke/children/get`.

The case that drove it: a long-running coding agent. A main run with a session that gets compacted splits its work into subtasks running in parallel; a subtask may split again. Each subtask needs the same durability as the main run, and its work (hundreds of tool calls) must not flood the main session.

```python
fork    = await ctx.fork(task)                          # a new thread of mine, from here; a run starts there
ctx.forks                                                # the forks this run created, as handles - also after a restart
weather = await ctx.call("weather", "Rome in June")      # a new thread of another agent; its first run starts

await fork.send("use CSS variables")                            # a message on its thread: its run's inbox, or a new run
ev = await ctx.subscribe(*ctx.inbox,                            # the user here, and what happens there - at once,
                         Filter(thread=fork, origin=Origin.INSIDE, kinds=[EventKind.RUN_COMPLETED]))  # holding no worker
run = await weather.wait()                                      # the run handling my last message, finished
answer = (await weather.messages())[-1].text                    # what was said there: copied over by this run's code
```

**A fork is a thread.** `ctx.fork(...)` creates a thread of this agent that starts from this thread at the current position (`forked_from = (thread, position)`, by reference: nothing is copied, positions carry on from the fork point), with the given message on it; a run of this agent starts there, as this run's child. On it:

| | on a fork |
| --- | --- |
| history | the parent thread up to the fork, then its own events: reading it is one continuous thread |
| its run | the run of its thread like any other: its own inbox, `ask`, `sleep`, `restart`, steps and replay, recovery after a crash. `ctx.run.parent` says it was forked; nothing else does |
| session | `session=`: a copy of the parent's (`ctx.session`, the default: it is the same agent carrying on — what the parent said stays OWN), a view or messages the parent computed, or `None` (a clean start) |
| thread state | the parent's state at the fork; its writes stay in the fork |
| subtasks of its own | forks of the fork: a tree of threads |
| talking to it | `fork.send(...)`: a message on its thread, as if a person typed it — its run's inbox, or, once that run has ended, a new run ("could you also…"). A person can also talk to it directly, on its own thread |
| hearing from it | it talks on its own thread (`ctx.send`); nothing goes back by itself. The parent listens (`ctx.subscribe(Filter(thread=fork, ...))`, `fork.wait()`, `fork.watch()`) and reads it (`fork.messages()`, `fork.thread_events()`, `fork.state`), copying over what it wants into its own thread or state. No merge method |
| its end | on its own thread (`run.completed`, `run.failed`, …): a listening parent wakes — also for a fork that ended without a word. Nothing is copied to the parent's thread |
| after the parent ended | a detached fork (`detach=True`) outlives its parent, isolated on its branch: it never writes on the parent's thread. A later turn can't reach it (threads aren't continued; see below) |
| cancelling | cancelling a run cancels the runs its messages started, unless detached |
| in Compass | a thread tree: the conversation, with forks branching off where they were made |

**Another agent works on a new thread of its own**, created by `ctx.call(agent, message, session=None)`. It is talked to and listened to the same way (`send`, `subscribe`, `wait`, `stream`, `messages`), as an A2A caller talks to a task: `message/send`, `message/stream`, `tasks/resubscribe`, `tasks/get`. It is not read beyond its messages (`thread_events` and `state` are for forks only): another agent is asked, not inspected. A thread lives as long as the handle of the run that created it; threads are **not continued** across turns (a stored reference is just a reference in a later run) — if the other agent needs context, it goes in the message or the `session=`; if it needs memory, that is its own concern. Continuing conversations between agents can be added later, additively.

**Its forks, back.** `ctx.forks` gives a run the forks it created, oldest first, as handles — after `ctx.restart()` or a crash too, recorded like a step so a replay sees the same list. Only this run's: siblings and earlier turns are out of reach. What each fork is for (a city, a part of the plan) is the agent's to remember, in `ctx.run.state` or in the fork's input. Threads of other agents aren't listed: a run keeps those handles itself (they survive in run state as references).

**The everyday surface (Oct 30, 2026).** Agent code reads short: the things an agent does all the time are on `ctx` itself — `ctx.input`, `ctx.session` (`messages()`, `add()`, `revise()`, `settled()`, `is_own(m)`), `ctx.state`; `ctx.send` / `ctx.stream` (a message on my thread); `ctx.ask`; `ctx.emit`; `@step` (or `ctx.step`), `ctx.sleep`; `ctx.fork`, `ctx.forks`, `ctx.call`, `ctx.handover`; `ctx.next(*threads)` (whatever comes first: a message for me, or what is said on those threads, a run there asking something or ending), `ctx.restart`, `ctx.reject`. Facts and the rarer things sit on `ctx.run` (parent, depth, caller, request, cancelled, its own `state`, `request_input`, `request_auth`, `wait_for_signal`), `ctx.thread` (id, agent, forked_from, `messages()` — the record of my thread), `ctx.client` (tools, `call_tool`), `ctx.agents` (`list`, `as_tools`), and `ctx.subscribe` / `ctx.post` for full control. A thread handle: `send`, `wait`, `messages`, `events`, `watch` (live, not recorded), `respond`, `signal`, `close`; a fork also `thread_events` and `state`. Reading versus watching: `next`, `wait`, `messages` and `events` are durable and recorded, safe to decide on; `watch` is live, to pass on. Thread versus session: `ctx.thread.messages()` is the record of the thread; `ctx.session.load()` is the working memory the model sees (built by the provider, possibly compacted).

**Sessions (Oct 30, 2026).** Working memory is revised in the background, never on a read, and holds the agent's own entries next to the thread's messages: see "Sessions backbone".

**Who writes where.** A run writes on its own thread (`ctx.send`, `ctx.state`, `ctx.post`), and sends messages to the threads it created. It listens only to those (`Filter(thread=...)` elsewhere raises `not_your_thread`). A sibling's thread is out of reach: there is no handle. `Origin.OUTSIDE` means not written by a run working on the thread (a person, a caller, a parent sending to its fork); `Origin.INSIDE` the opposite (what a fork or another agent says there).

**State.** Three scopes, no snapshotting to manage:

| | scope | shared by | replay |
| --- | --- | --- | --- |
| `ctx.state` | the thread | every run of the agent on the thread, live; a fork gets the state at the fork | reads and writes recorded like steps |
| `ctx.run.state` | one run | that run only, across its restarts; gone when it ends | the same |
| `ctx.session` | one session (messages) | the runs on the thread | built by the provider |

- Every thread-state write is a thread event (`state.changed`: key, value, version, the run that wrote it); the key-value table is the latest view of those events. The state at any position is the events up to it: Compass can show it, an auditor can see which run changed what, and a fork starts from the state at its position for free.
- Writes are live — durable and visible once they return. There is no `commit()`: runs wait for hours and other runs coordinate through the state, so hidden writes would be stored anyway, and a forgotten commit would lose them. Several keys at once is a batch, all or nothing, one event: `async with ctx.state.batch() as s: s.set(...); s.set(...)`.
- Conflicts: per key versions (`set(..., if_version=)` → `StateConflict`, `update(key, fn)` atomic) — runs started side by side on one thread through the API share its state live.
- Values are JSON-like and may hold run and thread references: a `Thread` handle is stored as a reference (`{"$thread": id}`), and the run that created the thread gets its handle back when it reads it — `await ctx.run.state.set("forks", forks)`, later `for fork in await ctx.run.state.get("forks"): await fork.send(...)`.
- `ctx.state.at_start()`: the state as it was when the run started — one consistent picture for a long run, while the conversation moves on.
- Data flows explicitly: down as messages on the threads a run created, up by listening and reading. A change of a fact isn't a command: the supervising run decides to steer or cancel its subtasks.
- Storage can offer "history off" per key, or values by file reference, for large values (the data store's option).

**Sessions passed on.** `session=` on `ctx.fork`, `ctx.call` and `ctx.handover` says what the new session starts with: `ctx.session` (a copy, as it is now), a `SessionView` from `await ctx.session.messages()` (or a slice of it), any list of messages the agent computed (a summary, a slice), or `None` (built from the thread). Recorded when the call is made, so a replay doesn't recompute it, and Compass can show what the other side started with.

**Restart.** `ctx.restart()` takes no argument: the run's code starts from the top, replay history dropped; `ctx.run.state`, its session and subscriptions carry over. A long main loop keeps its bookkeeping in `ctx.run.state` and replays a few steps per wake-up, however long it runs.

**Loops.** Besides the runs-per-thread limit, a limit per run tree (`max_runs_per_tree`): forks have threads of their own, so an agent that keeps forking itself would never hit a per-thread limit.

**The working copy is the application's.** For a coding agent the real state is the repository: each fork gets its own working copy (a git worktree, a sandbox), referenced in the fork's thread state, and the parent merges a fork's changes when the fork is done (a durable step). llamphouse carries the reference; it doesn't manage the workspace.

## Starting runs and the inbox

> **Decision (Oct 30, 2026).** Nothing starts a run by itself, and a thread has at most one active run. While a run is active, the thread takes no new message and no new run — only messages sent to the run (its inbox). What to do with a message that arrives while a run works is up to whatever sits in front of the agent: an app, a connector, an adapter. llamphouse has no router. Built in the `Scenario` runtime; tested.

How other APIs do it: the OpenAI Assistants API refuses a message or a second run on a thread with an active run (only tool outputs reach a running run) — the rule we follow; LangGraph's threads offer a "multitask strategy" for a busy thread — reject (its default), enqueue, interrupt, rollback — which here is the app's code; A2A sends a message to a task by its id (that's how `input-required` continues) and allows several tasks per context (here: forks); AG-UI runs one turn at a time; coding agents queue what you type while they work and pick it up at the next step — here, sending to the run.

**The primitives** (the API, `Scenario`, the runtime):

| | when no run is active | when a run is active |
| --- | --- | --- |
| **append** a message to the thread | recorded (the history, the session); starts nothing | **`ThreadBusy`** (with the active run) |
| **start** a run with a message | a new run; the message is its `ctx.input` | **`ThreadBusy`** |
| **send** a message to the run (`POST /runs/{id}/messages`; in agent code `thread.send`) | `NoActiveRun` / the run has ended | in the run's inbox, waking it if it waits |

"Active" is anything not finished: queued, running, waiting, `input_required`, `auth_required` — also a run that finished its code but waits for attached forks. Detached forks run on threads of their own. In Postgres the rule is a partial unique index on `runs(thread_id)` where the status is not terminal: two clients starting at once can't both win. Parallel work is a fork.

**A message sent to a run is on the thread when sent, and in the run's session when read.** It is something one party said to another — the user steering, a parent steering its fork — so it belongs in the shared record at the moment it was said, with its sender and the run it was for (addressed). It enters the run's working memory when the run takes it in (`ctx.next()`, or a subscription that delivers it), where it influenced the work, and the run's record gets a `message.read` event (which message, when) — Compass shows "sent 10:01, read 10:03". One that is never read stays on the thread only, unread (`sc.unread(run)`; in the API, on the run): the sender decides what to do with it.

**What starts runs:** the API (whatever sits in front: a chat UI, A2A, AG-UI, an email or Teams connector, webhooks and schedules from the agent file); agent code — `ctx.call` and `ctx.fork` (a new thread and its first run), `thread.start(...)` (the next run on a thread it created); `ctx.handover`. **The inbox** is what is sent to a run, nothing else (no `Agent.inbox`); a run reads it with `ctx.next()` and listens further with `ctx.subscribe`.

**Left to the app** — each a few lines in front of the agent, with `ThreadBusy` telling it which run is active:

| a message arrives while a run works | the app |
| --- | --- |
| steer it | sends it to the run |
| queue it | holds it, and starts the next run with it once the run has ended |
| interrupt | cancels the run, then starts a new one with it |
| it's the answer to `ask()` | answers the run (the answer API) instead of sending |
| reject | lets `ThreadBusy` through (a 409 to its client) |
| a message the run never read | starts the next run with it, or tells the user |
| debounce, batch | waits until the thread is quiet, then starts one run |

Email or Teams: a thread per conversation and a run per message (steer while busy), one long-running run per mailbox that receives every event, or batches — the connector's choice. In the `Scenario` runtime, `sc.say` plays the simplest chat app: start a run when the thread is idle, otherwise send to the active run.

## Core model: threads, sessions, runs

The thread is the record, the session is the working memory, the run is the unit of work.

- **Thread:** append-only, immutable log of everything: user messages, replies, tool calls, `input_required`, session revisions. Can be shared by several agents. Built for audit, Compass and replay.
- **Session:** the context a run sees. Derived from the thread, mutable, versioned. A thread can have several sessions (handover, reset, per-agent views).
- **Run:** one unit of work toward one goal. Lifecycle `queued → running ⇄ waiting | input_required | auth_required → completed | failed | cancelled | rejected`. `waiting` is a wait on a timer or a signal (`sleep`, `wait_for_signal`), carrying what it waits for; `input_required` is a wait for an answer from the user or client; `auth_required` a wait for credentials from the client (a user's consent for a downstream system). None of them holds a worker. `rejected` means the agent declined the work: terminal, not a failure. Back-and-forth inside a run uses `input_required`; a new request after completion is a new run. Runs can spawn child runs.

  A2A (v1.0) mapping: `queued` → `SUBMITTED`, `running` and `waiting` → `WORKING`, `input_required` → `INPUT_REQUIRED`, `auth_required` → `AUTH_REQUIRED`, `completed` → `COMPLETED`, `failed` → `FAILED`, `cancelled` → `CANCELED`, `rejected` → `REJECTED`.
- **Event:** the unit written to the log and streamed to clients. Thread events make up the thread; run events belong to one run.

### What is on the thread

The thread holds what the conversation *is*: what someone said, did or produced that another party (the user, another agent, a session provider, an auditor) may need later. Every thread event takes a position in the thread.

| On the thread | Types | Notes |
| --- | --- | --- |
| Messages | `message` | the final message of a person, an agent, the system or a service: its sender plus parts (`TextPart`, `FilePart`, `DataPart`, `ToolCallPart`, `ToolResultPart`). Tool calls and results are messages with tool parts, as the agent reports them (models are not part of llamphouse). An artifact is an `ArtifactPart` in a message, next to its other parts (prose and an artifact in one reply); file contents live in the file store, referenced by URI. An answer to `input_required` is a message addressed to its run |
| Run lifecycle | `run.created`, `run.spawned`, `run.waiting`, `run.input_required`, `run.completed`, `run.failed`, `run.cancelled` | the timeline. `run.input_required` carries the prompt and the response schema |
| State changes | `state.changed` | every thread-state write (a batch is one event): key, value, version, the writing run. The state at a position is the events up to it |
| Session lifecycle | `session.created`, `session.revision.*`, … | see the sessions backbone |
| Agent items | `x-<namespace>.<name>`, e.g. `x-trip.plan_ready` | what an agent posts for other runs to subscribe to (`ctx.post`). The payload schema is the agent's |

Not on the thread, but per run, ordered by `seq` and with no thread position:

| Run events | Types | Persisted |
| --- | --- | --- |
| Run record | `step.started`, `step.completed` (with its call key: the durable replay record), `step.failed`, retries, error details, usage and cost, the last shared state (`state.snapshot`) | always |
| Run stream | `message.started`, `message.delta`, `state.delta`, `progress`, `child_run.*`, custom (`x-…`) emits | per the data store's option (below) |

Not in the event log at all: traces (tracing store), session data (session store), file contents (file store).

### Storage

Postgres for durability, Redis for live fan-out:

```sql
thread_events(
  thread_id   uuid,
  pos         bigint,                 -- position in the thread: gapless, in commit order
  id          uuid,                   -- stable event id (references)
  run_id     uuid null,              -- null for thread-level events (e.g. a user message)
  agent       text null, parent_run_id uuid null, session_id uuid null,
  idempotency_key text,               -- a retried run doesn't write twice
  type        text,
  payload     jsonb,
  trace_id    text, span_id text,
  created_at  timestamptz,
  primary key (thread_id, pos),
  unique (thread_id, idempotency_key)
);
run_events(
  run_id     uuid,
  seq         bigint,                 -- per-run order, assigned by the run's single writer
  thread_id   uuid,
  kind        text,                   -- record | stream
  idempotency_key text,
  type        text,
  payload     jsonb,
  created_at  timestamptz,
  primary key (run_id, seq),
  unique (run_id, idempotency_key)
);
threads(id, last_pos bigint, forked_from_thread uuid null, forked_at_pos bigint null, created_at, …);
runs(id, thread_id, session_id, parent_run_id, agent, agent_uid, agent_version, agent_revision, status, input, output, detached bool,
     run_state jsonb null, lease_owner, lease_until, created_at, …);   -- run_state: ctx.run.state, versioned
subscriptions(run_id, thread_id, filters jsonb, cursor bigint);  -- cursor = a thread position
system_events(id, kind, data jsonb, at);                         -- runtime-wide: no thread (see below)
sessions(id, thread_id, agent, scope text,                       -- scope: agent | run
         run_id uuid null, forked_from uuid null, forked_from_version int null,
         version, data jsonb, cursor bigint, updated_at);
```

Order: a Postgres `bigserial` is not commit order (id 101 can commit before id 100, and a reader at cursor 101 never sees 100). So each thread event takes the thread's next position in the same transaction:

```sql
begin;
  update threads set last_pos = last_pos + 1 where id = $1 returning last_pos;
  insert into thread_events(thread_id, pos, …) values ($1, last_pos, …);
commit;
```

The row lock lasts from the `update` to the commit, so the position is taken as the transaction's last statement: the rest of its work comes first. (Tested against Postgres with 20 concurrent writers, see `prototypes/v2_rig`: a `bigserial` cursor reader missed about half of 400 events; the per-thread position missed none; taken last, it is about 3× faster than taken first.) Only thread events take it — a few per run — so concurrent streaming runs on one thread never wait on each other: chunks are run events, numbered by their run. The contract for every data store: thread positions gapless and increasing in commit order; run `seq` increasing per run. There is no global order across threads; nothing needs one.

Queries: the timeline `where thread_id = $1 order by pos`; catch up `where thread_id = $1 and pos > $cursor`; one run `from run_events where run_id = $1 order by seq`. Clients resume a stream with `Last-Event-ID`: a thread position, or a run and seq for the stream of an active run.

Events are written at least once: a run that is retried (worker crash before ack) writes the same events again. Every event carries an idempotency key (run, restart generation, call key or attempt, seq), so a rewrite is a no-op. The restart generation matters: after `ctx.restart` call keys count from 1 again, and without it a reply in iteration 2 would collide with a message of iteration 1 and be silently dropped as a "retry" (found with the scenario tests).

Run stream events always go through the event queue, for live streaming and for resuming an active run's stream. Whether they are also persisted is the data store's option. The Postgres store:

```yaml
data_store:
  postgres:
    database_url: ${DATABASE_URL}
    stream_events: final      # final (default) | all | on_error
```

- `final`: no stream events; the final `message` (thread) and the run record are kept.
- `all`: every stream event, e.g. to replay a stream exactly or to debug a model's output.
- `on_error`: stream events are written only when the run fails, in one batch (from the event queue, which holds them for resume anyway), so a failure can be diagnosed; on success they are dropped.

The in-memory store keeps them all, with an optional retention limit. Thread events (messages, artifacts in them included) and the run record are always persisted.

Adapter mapping: A2A task → run, A2A context → thread (see "The A2A adapter"); Responses call → run, Conversation → thread; AG-UI run → run, AG-UI thread → thread, AG-UI interrupt → `input_required`; webhook → run (optionally on a thread).


### The first runtime (2.0.0a)

What runs today, to build real applications on:

- **Processes.** `python -m llamphouse.runtime app:AGENTS --role api|worker|all` (or `runtime.serve`). An `api` process serves the HTTP API and Compass and executes nothing; a `worker` executes runs; `all` is both. Any number of each, on one Postgres and one Redis. Without them: one process, in memory.
- **Trees and transactions.** State is grouped in trees: a thread people talk to and every thread its runs create, with their runs and sessions. Everything that happens touches one tree, so a tree is the unit of a transaction: `Engine._tx(tree)` takes the tree's lock in the store (a Postgres advisory lock held until commit), brings this process's copy up to date (only rows with a newer tree version), makes the change, and writes the changed rows with the tree's next version. Conversations never wait on each other. A process's copies are caches, evicted when idle; the store is the truth.
- **Leases.** A worker takes a run only if the store says it is queued, and takes a lease (`lease_owner`, `lease_until`) in the same transaction, renewed while it runs. A run executing here keeps its own record, merging in only what others may do meanwhile (cancel, answer, signal); a worker that finds its lease taken stops without writing. A graceful stop hands runs back at once; a dead worker's runs are queued again by the sweeper when the lease expires, and replay their recorded steps.
- **The sweeper** (every worker): waiting runs whose time came, runs whose lease expired, queued runs nobody took (`ring_at`), revisions to try again - all from indexed columns next to each row's JSON.
- **Queue and bus.** The job queue (`runtime.queue`: memory, a Redis list) is a doorbell: a lost or repeated job is harmless. The bus (`runtime.bus`: memory, Redis pub/sub) carries live news: `thread:<id>` for new thread events, `run:<id>` for a run's stream including text deltas (which are never stored).
- **Postgres** (`PostgresStore`): `llamphouse_trees`, `llamphouse_threads`, `llamphouse_events`, `llamphouse_runs` (with the one-active-run unique index), `llamphouse_sessions`, `llamphouse_run_events`; numbered migrations under an advisory lock. Files in `PostgresFileStore`; object storage later behind the same contract. Values go through `runtime.codec` (tagged JSON that reads back as the same types); agent classes must be importable by module path.
- **The HTTP API** (`runtime.api`): the primitives one to one, stateless; live thread and run streams (SSE) follow the bus. **Compass** (`/compass`): threads, a live timeline, runs with their stream and session, threads a run made, state, and a composer offering exactly the primitives.
- **The spec.** The scenario tests run on this engine in four ways: memory or Postgres, and one engine or split - runs executing in a second engine, as a worker process would (`LLAMPHOUSE_SCENARIO_STORE=postgres`, `LLAMPHOUSE_SCENARIO_SPLIT=1`). `tests/unit/runtime/test_processes.py` runs an API process and workers on Postgres and Redis, and replaces a worker that dies mid-run.

- **Tracing** (`runtime.tracing`, on the 1.x core's `llamphouse.core.tracing` and its stores). Each execution of a run is a root span `llamphouse.run` (`run.id`, `llamphouse.run_id`, `thread.id`, `llamphouse.agent`, `llamphouse.parent_run`; at its end `llamphouse.run.status`, `llamphouse.run.error`, `llamphouse.run.replayed_steps`), current while `run(ctx)` executes: spans agent code makes with any OpenTelemetry tracer nest under it, across awaits, tasks and `asyncio.to_thread`. A run executed again (it waited, or its lease was taken over) is a new span with the same `run.id`. A step executed is a `llamphouse.step` under it (`llamphouse.step.name`); a replayed step is none. A provider's revision is a root span `llamphouse.session.revise`. `Engine(tracing_store=...)` sets tracing up on `open()`; the server takes the store from the environment as 1.x does (`TRACING_STORE`, `LLAMPHOUSE_TRACING_ENABLED`). `GET /v2/runs/{id}/trace` reads a run's spans back from the store; Compass shows them as a tree, with a span's attributes and model input and output (`gen_ai.*.messages`, `input.value` / `output.value`) readable. Without a store or a tracer provider, nothing is imported and no span is made; tracing never fails a run.

Not yet: pools (which workers run which agents), per-tree encoding cost (every transaction re-encodes the tree's records to find what changed: fine for conversations, slow for trees with thousands of runs), retention, auth beyond an API key, and the A2A, AG-UI and Responses adapters on top.

## Agent contract (SDK)

Agents import only `llamphouse.sdk`: a small, stable module inside the `llamphouse` package, with no web, database or cloud dependencies. A test fails the build if `llamphouse.sdk` imports anything beyond pydantic and the standard library. If it ever needs its own version, `llamphouse.sdk` can move to a separate package that `llamphouse` re-exports, without changing agent imports. A code agent is a definition; its configured variants are instances (the 1.x `definitions` / `agents` split).

Rules the contract follows so it survives for years:

1. **Additive only.** Within a major version, fields, event kinds and part kinds may be added, never renamed, retyped or removed. The contract only breaks on a major release of llamphouse, and the changelog says so explicitly.
2. **Open by default.** Every type carries `metadata` for extensions, and `kind` is a plain string. Unknown kinds are passed through (`UnknownPart`, a plain `Event`) or ignored, never a crash. Custom kinds use an `x-` prefix.
3. **Feature detection over assumptions.** Optional features are asked for with `ctx.capability(...)` and may be `None`; an agent that can't run without one declares it with `@requires(...)`, and the runtime refuses to start it where it's missing.
4. **No vendor concepts.** Nothing in the SDK knows about OpenAI, Azure, A2A, AG-UI or MCP; adapters translate at the edge.
5. **Mechanism, not policy.** Models, retrieval and tools are not part of llamphouse: the agent brings its own clients. The SDK gives it the run, the thread, the session, durable waits and steps, and the config.
6. **Versioning is in the files, not the code.** The agent definition file declares its format with `apiVersion: llamphouse.io/v1alpha1` (read by different runtime versions over time, and later by the Kubernetes operator); the code pins its llamphouse major in its requirements (`llamphouse>=2,<3`), which decides the SDK it runs on.
7. **No magic strings.** A closed set of options is an `Enum` (`Origin.INSIDE`, `AgentType.WORKFLOW` for its known values). An open set — sender kinds, run statuses, event kinds — is a `StrEnum` for the known values, and its field also accepts a plain string, so a value from a newer runtime passes through (rule 2). Runs and threads are addressed by typed references (`ctx.run`, `ctx.parent`, `ctx.thread`, a `ChildRun`, `RunRef(id)`), never by a keyword string.

### Types

```python
"""llamphouse.sdk - the agent contract."""

Metadata = dict[str, Any]

# ── Vocabulary: enums for closed sets, StrEnums for open ones, typed references ──

class SenderKind(StrEnum):
    """Who wrote a message. A fact, the same for every reader - unlike a role, which depends on who reads it."""
    PERSON = "person"        # a human
    AGENT = "agent"          # an agent: one of ours, or a remote one (A2A)
    SYSTEM = "system"        # instructions or notices from the operator or the runtime
    SERVICE = "service"      # an outside system calling the API, an integration
    TRIGGER = "trigger"      # a trigger from the agent file (a webhook, a schedule): id = its name

class Viewpoint(Enum):
    """How a message relates to the run reading it - what a model prompt or a protocol calls its role."""
    OWN = "own"              # written by this run, or an earlier run sharing its session (its earlier turns)
    OTHER = "other"          # someone else in the conversation: a person, another agent, a service
    SYSTEM = "system"

class Origin(Enum):
    """Whose events a filter matches, relative to the subscribing run."""
    ANY = "any"
    OUTSIDE = "outside"      # not written by this agent: a person, a caller, a service

class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"       # waits for credentials from the client
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"                 # the agent declined the work: terminal, not a failure

class AgentType(StrEnum):
    """What an agent is, for Compass and the API. Open: a newer type passes through as a string."""
    AGENT = "agent"          # conversational: a thread, a session, user messages
    WORKFLOW = "workflow"    # a durable process: typed input and output, steps, no conversation

class EventKind(StrEnum):
    """Known kinds of thread and run events. Kinds stay open: `x-…` and newer kinds are plain strings."""
    MESSAGE = "message"
    RUN_CREATED = "run.created"
    RUN_SPAWNED = "run.spawned"
    RUN_WAITING = "run.waiting"
    RUN_INPUT_REQUIRED = "run.input_required"
    RUN_AUTH_REQUIRED = "run.auth_required"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_REJECTED = "run.rejected"
    # message.started, message.delta, progress, state.*, usage, child_run.*, session.*: see the event types

@dataclass(frozen=True, eq=False)
class RunRef:
    """A run to filter on: ctx.run, ctx.run.parent, a RunResult, or RunRef(id). Equal by id."""
    id: str

@dataclass(frozen=True, eq=False)
class ThreadRef:
    """A thread: ctx.thread, a Thread handle, or ThreadRef(id). Equal by id."""
    id: str

class Origin(Enum):
    ANY = "any"
    OUTSIDE = "outside"    # not written by a run working on the thread: a person, a caller, a parent sending to its fork
    INSIDE = "inside"      # written by a run working on the thread: what a fork or another agent says there

# ── Content: messages are made of typed parts, so new modalities are additive ──

@dataclass(frozen=True, kw_only=True)
class Part:
    kind: str
    metadata: Metadata = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class TextPart(Part):
    kind: str = "text"
    text: str

@dataclass(frozen=True, kw_only=True)
class FilePart(Part):
    """Documents, images, audio, video - stored by reference. `data` only hands bytes in: the runtime
    stores them, fills uri / size / sha256 and drops `data`; a stored message never holds file bytes."""
    kind: str = "file"
    mime_type: str
    name: str | None = None
    uri: str | None = None           # "files://…" (internal, resolved by the API) or an outside URL
    size: int | None = None          # filled by the file store
    sha256: str | None = None        # filled by the file store: integrity and deduplication
    data: bytes | None = None

@dataclass(frozen=True, kw_only=True)
class DataPart(Part):
    """Structured JSON-like content: form answers, results, UI payloads."""
    kind: str = "data"
    data: Any
    schema: Mapping[str, Any] | None = None       # JSON Schema, if the data has one

@dataclass(frozen=True, kw_only=True)
class ToolCallPart(Part):
    kind: str = "tool_call"
    call_id: str
    name: str
    arguments: Mapping[str, Any]

@dataclass(frozen=True, kw_only=True)
class ToolResultPart(Part):
    kind: str = "tool_result"
    call_id: str
    output: Sequence[Part]
    is_error: bool = False

@dataclass(frozen=True, kw_only=True)
class ArtifactPart(Part):
    """A named output of the run, carried in a message next to its other parts - what A2A calls an
    artifact. Versions, an artifact API and sharing rules are the app's, not llamphouse's."""
    kind: str = "artifact"
    name: str
    description: str | None = None
    parts: Sequence[Part]            # its content: text, a file (by URI, see Files), data

@dataclass(frozen=True, kw_only=True)
class UnknownPart(Part):
    """A part kind this SDK version doesn't know (sent by a newer runtime or client). Kept as is."""
    raw: Mapping[str, Any]

@dataclass(frozen=True, kw_only=True)
class Sender:
    """Who wrote a message: Sender.person(id), Sender.agent(id, run=...), Sender.system(), Sender.service(id)."""
    kind: SenderKind | str
    id: str                          # the person's subject, the agent instance's name, the service's name
    name: str | None = None          # for display
    run: RunRef | None = None        # the run that wrote it, for our own agents

@dataclass(frozen=True, kw_only=True)
class Message:
    id: str
    sender: Sender                   # who wrote it; a role is derived per reader (viewpoint)
    parts: Sequence[Part]
    references: Sequence[RunRef] = ()   # earlier runs this message refers to (A2A referenceTaskIds)
    metadata: Metadata = field(default_factory=dict)

    @classmethod
    def create(cls, sender, *content, references=(), metadata=None) -> Message: ...
    @property
    def artifacts(self) -> list[ArtifactPart]: ...

    @property
    def text(self) -> str: ...       # the text parts, joined

# ── Events: what a run emits. Adapters render them in whatever format an API streams ──

@dataclass(frozen=True, kw_only=True)
class Event:
    kind: str                        # unknown or custom ("x-…") kinds are passed through
    metadata: Metadata = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class MessageStarted(Event):
    kind: str = "message.started"
    message_id: str
    sender: Sender | None = None     # None: the run's own agent

@dataclass(frozen=True, kw_only=True)
class PartDelta(Event):
    """Incremental content while a message streams (usually text)."""
    kind: str = "message.delta"
    message_id: str
    index: int                       # which part of the message
    delta: str

@dataclass(frozen=True, kw_only=True)
class MessageCompleted(Event):
    """The final message: the only event that goes on the thread."""
    kind: str = "message.completed"
    message: Message

@dataclass(frozen=True, kw_only=True)
class Progress(Event):
    """Human-readable status, e.g. 'Searching documents'."""
    kind: str = "progress"
    label: str
    fraction: float | None = None

@dataclass(frozen=True, kw_only=True)
class StateSnapshot(Event):
    """State shared with the client (e.g. AG-UI shared state)."""
    kind: str = "state.snapshot"
    state: Mapping[str, Any]

@dataclass(frozen=True, kw_only=True)
class StateDelta(Event):
    kind: str = "state.delta"
    patch: Sequence[Mapping[str, Any]]           # JSON Patch (RFC 6902)

@dataclass(frozen=True, kw_only=True)
class Usage(Event):
    """What a model call cost, reported by the agent (llamphouse doesn't see model calls)."""
    kind: str = "usage"
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None

@dataclass(frozen=True, kw_only=True)
class Custom(Event):
    """An app's own event on the run's stream."""
    kind: str                                    # "x-…"
    data: Any = None

@dataclass(frozen=True, kw_only=True)
class ThreadCreated(Event):
    """Emitted by the runtime when the run creates a thread (a fork, another agent's), so a UI can draw the tree."""
    kind: str = "thread.created"
    thread_id: str
    agent: str
    forked_from: str | None = None
    run_id: str | None = None                    # the run its first message started

# ctx.emit takes the agent's kinds only - Progress, Usage, StateSnapshot, StateDelta, Custom - and raises
# for the rest: the message events come from ctx.send / ctx.stream, ThreadCreated from the runtime.
# None of them is on the thread; Usage and the last StateSnapshot are also kept on the run's record.

@dataclass(frozen=True, kw_only=True)
class UnknownEvent(Event):
    """An event kind this SDK version doesn't know. Kept as it came."""
    raw: Mapping[str, Any]

class Rejected(Exception):
    """Raise (or ctx.reject) to decline the work: the run ends `rejected`, terminal and not a failure."""
    def __init__(self, reason: str, *, metadata: Metadata | None = None): ...

class AgentError(Exception):
    """Raise to fail the run with a code clients can act on."""
    def __init__(self, code: str, message: str, *, retryable: bool = False, metadata: Metadata | None = None): ...

# ── The thread, as a run sees it ──

@dataclass(frozen=True, kw_only=True)
class ThreadEvent:
    """An event on the thread, as a subscription delivers it."""
    position: int
    kind: str                        # "message", "run.completed", "session.revision.completed", "x-trip.plan_ready", …
    run: RunRef | None = None        # the run that wrote it; None for the user or an outside system
    agent: str | None = None
    to: RunRef | None = None         # set when it is addressed to one run
    message: Message | None = None   # for kind "message"
    data: Any = None                 # for everything else
    at: datetime | None = None       # when it was written
    metadata: Metadata = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class Filter:
    """Which thread events a subscription delivers. Fields must all match; empty = any."""
    kinds: Sequence[EventKind | str] = ()   # exact kinds, or a pattern: "run.*", "x-*"
    senders: Sequence[SenderKind | str] = ()  # for messages: who wrote them
    agents: Sequence[str] = ()              # agent instance names
    runs: Sequence[RunRef] = ()
    to: RunRef | None = None         # events addressed to this run (an answer to its request)
    origin: Origin = Origin.ANY      # OUTSIDE / INSIDE: written by a run working on the thread or not
    thread: ThreadRef | None = None  # this run's thread (None), or a thread it created: one subscription, several threads

class Subscription(Protocol):
    """Durable: it has a cursor, survives a restart, and wakes a waiting run."""
    def __aiter__(self) -> AsyncIterator[ThreadEvent]: ...
    async def next(self, timeout: float | None = None) -> ThreadEvent | None: ...
    async def close(self) -> None: ...

class Session(Protocol):
    """The agent's working memory on this thread, built by the provider the agent file names."""
    id: str
    async def messages(self) -> SessionView: ...                         # as it is: no provider work; recorded
    async def add(self, *content: Content) -> Message: ...               # an entry: in the session, not on the thread
    async def revise(self, **options: Any) -> None: ...                  # ask for a revision (a job); returns at once
    async def settled(self) -> SessionView: ...                          # wait until no revision is in flight
    def is_own(self, message: Message) -> bool: ...                       # did I say it? (the model's assistant role)

class SessionView(Sequence[Message]):
    """A session as it is: an immutable snapshot of its messages and entries, newest last (it goes straight to a
    model), with its id, version and whether a revision was in flight. It can start another session."""

class State(Protocol):
    """Durable key-value state, scoped to the thread: shared by every run on it. Every key has a
    version (0: not set). Reads and writes are recorded like steps: a replay sees the same values
    and doesn't write again."""
    async def get(self, key: str, default: Any = None) -> Any: ...
    async def get_versioned(self, key: str) -> tuple[Any, int]: ...
    async def set(self, key: str, value: Any, *, if_version: int | None = None) -> int: ...   # StateConflict
    async def update(self, key: str, fn: Callable[[Any], Any], default: Any = None) -> Any: ...  # atomic
    async def delete(self, key: str, *, if_version: int | None = None) -> None: ...

class StateConflict(Exception):
    """A write with `if_version` found another version: someone else wrote it first."""

@dataclass(frozen=True, kw_only=True)
class Identity:
    subject: str                     # who started the run: a user, a service, a schedule, a parent run
    tenant: str | None = None
    metadata: Metadata = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class ToolSpec:
    """A tool described for a model. Used for client tools and agents-as-tools."""
    name: str
    description: str
    input_schema: Mapping[str, Any]              # JSON Schema
    metadata: Metadata = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class Request:
    """What the caller sent with the request that started (or resumed) the run."""
    accepts: Sequence[str] = ()      # the media types the caller accepts back; empty = anything
    extensions: Sequence[str] = ()   # the extension URIs the caller activated
    metadata: Metadata = field(default_factory=dict)   # the request's metadata, as the adapter received it

@dataclass(frozen=True, kw_only=True)
class AuthRequest:
    """Why a run pauses for credentials from the client, e.g. a user's consent for a downstream system."""
    reason: str
    scheme: str | None = None        # e.g. "oauth2", "openIdConnect"
    scopes: Sequence[str] = ()
    url: str | None = None           # where the client completes it
    metadata: Metadata = field(default_factory=dict)

@dataclass(frozen=True, kw_only=True)
class InputRequest:
    """Why a run pauses for a human or a client, e.g. approving a payment."""
    reason: str
    prompt: Sequence[Part] = ()
    input_schema: Mapping[str, Any] | None = None
    metadata: Metadata = field(default_factory=dict)

# ── The run ──

class RunContext(Protocol[ConfigT]):
    # The everyday things are short and on ctx itself; facts and the rarer things sit on ctx.run,
    # ctx.thread, ctx.client and ctx.agents.

    # what I'm working on
    input: Message                   # what started this run, always one message; its sender says who or
                                     # what (a person, an agent, the parent, a TRIGGER, a SERVICE);
                                     # for a Workflow, input.data is its validated Input
    session: Session                 # my working memory: messages(), add(), revise(), settled(), is_own()
    state: ThreadState               # the thread's: facts of the conversation; forks with the thread

    # talking, on my thread - the only thread a run writes on; whoever created it listens
    async def send(self, *content: Content, references: Sequence[RunRef] = ()) -> Message: ...
    def stream(self) -> MessageWriter: ...                  # the same, streamed: started, deltas, completed
    async def ask(self, text: str, *, schema: Mapping[str, Any] | None = None) -> Message: ...
                                                            # input_required: the run waits, then carries on here
    async def emit(self, event: Event) -> None: ...          # the live stream: Progress, Usage, State*, Custom

    # working
    async def step(self, name: str, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T: ...
                                                            # usually the @step decorator on a method
    async def sleep(self, seconds: float | None = None, *, until: datetime | None = None) -> None: ...
                                                            # durable, no worker held (asyncio.sleep: short pauses)

    # working with others
    async def fork(self, *content: Content, session: SessionSeed = ctx.session, detach: bool = False) -> Thread: ...
    forks: Sequence[Thread]                                 # this run's forks, as handles; also after a restart
    async def call(self, agent: str, *content: Content, session: SessionSeed = None,
                   detach: bool = False) -> Thread: ...     # a new thread of another agent, with this message
    async def handover(self, agent: str, *, session: SessionSeed = None) -> NoReturn: ...

    # waiting
    async def next(self, *threads: Thread, timeout: float | None = None) -> ThreadEvent | None: ...
                                                            # a message for me, or news from those threads
    async def restart(self) -> NoReturn: ...                # same run, code from the top, replay history dropped;
                                                            # ctx.run.state, forks, session, subscriptions carry over
    async def reject(self, reason: str) -> NoReturn: ...    # rejected: terminal, not a failure

    # facts, and the rest
    me: AgentIdentity                # who I am: name, type (agent or workflow), display name, version, revision
    config: ConfigT                  # the instance config, validated
    run: ThisRun                     # parent, depth, caller, request, cancelled, state (its own);
                                     # request_input(), request_auth(), wait_for_signal()
    thread: ThisThread               # id, agent, forked_from; messages() - the record of my thread
    client: Client                   # tools the client declared; call_tool()
    agents: Agents                   # list(), as_tools(): who this run may call
    inbox: tuple[Filter, ...]        # what this run receives without asking: for subscribe
    def subscribe(self, *filters: Filter) -> Subscription: ...   # full control: on my thread and threads I created
    async def post(self, kind: str, data: Any = None) -> ThreadEvent: ...   # an "x-…" item on my thread
    def capability(self, kind: type[T]) -> T | None: ...    # Files, Scheduler: None without the component
    def require(self, kind: type[T]) -> T: ...              # never None: raises CapabilityMissing

class MessageWriter(Protocol):
    """async with ctx.stream() as s: await s.write("…") - a message on my thread, streamed."""
    message_id: str
    async def write(self, text: str) -> None: ...
    async def add(self, part: Part) -> None: ...
    async def __aenter__(self) -> MessageWriter: ...
    async def __aexit__(self, *exc: Any) -> None: ...

def step(fn=None, *, name: str | None = None): ...   # decorator form of ctx.step, for agent methods

# ── The agent ──

class Config(BaseModel):
    """Typed instance config. The JSON Schema is published from the class."""
    model_config = ConfigDict(extra="forbid", frozen=True)

# ── What an agent publishes from its code: rendered in the API's agent list, Compass, an A2A card ──
# How it is presented through a protocol (an A2A card's skills, media type modes, extensions) is not
# the code's: it is the agent's `publish` settings and the adapter instance's (see "Adapters").

@dataclass(frozen=True, kw_only=True)
class AgentIdentity:
    """Who a run's agent is - ctx.me. The same for agents and workflows (see `type`); str() is the name."""
    name: str                        # the agent instance's name ("hr-thailand"): the reference everywhere, immutable
    type: AgentType | str = AgentType.AGENT
    display_name: str | None = None  # for people ("HR Assistant Thailand"); may change
    version: str | None = None       # the code's version
    revision: int | None = None      # the instance's config revision

@dataclass(frozen=True, kw_only=True)
class AgentInfo:
    name: str                        # from the agent definition
    type: AgentType | str = AgentType.AGENT
    description: str = ""
    version: str | None = None       # Agent.version
    requires: Sequence[str] = ()     # the capabilities it can't run without, by name ("Files")
    config_schema: Mapping[str, Any] | None = None
    input_schema: Mapping[str, Any] | None = None    # for a Workflow
    output_schema: Mapping[str, Any] | None = None
    metadata: Metadata = field(default_factory=dict)

class Agent(Generic[ConfigT]):
    type: ClassVar[AgentType] = AgentType.AGENT
    description: ClassVar[str | None] = None            # None: the class's own docstring (published: agent card, Compass)
    version: ClassVar[str | None] = None                 # the code's version, for reporting and logging
    inbox: ClassVar[tuple[Filter, ...]] = (            # what a run receives without asking (ctx.next):
        Filter(kinds=[EventKind.MESSAGE], origin=Origin.OUTSIDE),   # messages from outside its own run tree
    )

    @classmethod
    def describe(cls, name: str) -> AgentInfo: ...   # what gets published; the name comes from the definition

    async def run(self, ctx: RunContext[ConfigT]) -> Any:
        """Return = run completed (a returned value is its output); raise AgentError = run failed.
        May also be an async generator: every yielded Event is emitted, and there is no output.
        The return annotation declares the output (see "Output")."""

    @classmethod
    def config_schema(cls) -> dict[str, Any]: ...
    @classmethod
    def output_schema(cls) -> dict[str, Any] | None: ...   # from run()'s return annotation; published in describe()

class Workflow(Agent[ConfigT], Generic[ConfigT, InputT, OutputT]):
    """An agent that is a durable process rather than a conversation. Same contract, same
    durability; Compass shows it as steps instead of chat, and the API starts it with typed input."""
    type: ClassVar[AgentType] = AgentType.WORKFLOW
    inbox: ClassVar[tuple[Filter, ...]] = ()              # a process, not a conversation
    Input: ClassVar[type[BaseModel]]       # validated on start; its JSON Schema is published
    Output: ClassVar[type[BaseModel]]      # what run() returns; published too

    async def run(self, ctx: RunContext[ConfigT]) -> OutputT:
        """ctx.input.data is the validated Input. The returned Output becomes the run's output:
        a message with an ArtifactPart "output" holding it as data, and the `run.completed` payload."""

    @classmethod
    def input_schema(cls) -> dict[str, Any]: ...
    @classmethod
    def output_schema(cls) -> dict[str, Any]: ...

# ── Capabilities (optional; ctx.capability(X) returns None where absent) ──

SessionSeed = Session | Sequence[Message] | None   # ctx.session (a copy), a SessionView, messages, or None

class Agents(Protocol):
    """The other agents this run may talk to (ctx.call, ctx.handover)."""
    def list(self) -> Sequence[AgentInfo]: ...                      # the agents this run may talk to
    def as_tools(self) -> Sequence[ToolSpec]: ...                    # let a model decide to delegate

class ThisRun(RunRef, ABC):                                          # ctx.run
    parent: RunRef | None            # the run whose message started this one (a fact, not a channel)
    depth: int                       # 0 without a parent, +1 per level
    caller: Identity                 # who started it; a parent's caller passes on
    request: Request                 # what the caller sent: accepted media types, active extensions, metadata
    cancelled: bool
    state: State                     # its own: kept across restarts, gone when it ends
    async def request_input(self, request: InputRequest) -> Message: ...     # ctx.ask with a full request
    async def request_auth(self, request: AuthRequest) -> Message: ...       # auth_required
    async def wait_for_signal(self, key: str, *, timeout: float | None = None) -> Any: ...

class ThisThread(ThreadRef, ABC):                                    # ctx.thread
    agent: str
    forked_from: tuple[ThreadRef, int] | None
    async def messages(self) -> Sequence[Message]: ...               # the record; recorded like a step

class Client(Protocol):                                              # ctx.client
    tools: Sequence[ToolSpec]                                        # tools the client declared (AG-UI, Responses)
    async def call_tool(self, call: ToolCallPart) -> ToolResultPart: ...

@dataclass(frozen=True, eq=False)
class Thread(ThreadRef, ABC):
    """A thread this run created - a fork of its own, or another agent's - to talk to and listen to. Kept in
    state as a reference; the creating run gets the handle back. It lives as long as that run."""
    agent: str
    forked_from: tuple[ThreadRef, int] | None    # a fork: where it starts
    async def send(self, *content: Content) -> None: ...             # to the run working there (its inbox); NoActiveRun
    async def start(self, *content: Content) -> None: ...            # its next run, this run's child; ThreadBusy
    async def wait(self) -> RunResult: ...                           # the run handling my last message, finished; never raises
    async def messages(self) -> Sequence[Message]: ...               # what was said there (a fork: after the fork point)
    def events(self, *kinds: str) -> Subscription: ...               # durable: = ctx.subscribe(Filter(thread=self, kinds=kinds))
    def watch(self) -> AsyncIterator[Event]: ...                     # live run events of that run, to pass on; not recorded
    async def respond(self, answer: Message | Content) -> bool: ...  # answer its input_required; first answer wins
    async def signal(self, key: str, value: Any = None) -> None: ...
    async def close(self) -> None: ...                               # stop what runs there; later sends raise ThreadClosed
    async def thread_events(self, *kinds: str) -> Sequence[ThreadEvent]: ...   # a fork only: read its thread
    state: StateView                                                 # a fork only: read its state

@dataclass(frozen=True, eq=False)
class RunResult(RunRef):
    """A run, finished."""
    agent: str
    status: RunStatus | str          # completed, failed, cancelled, rejected
    output: Any = None               # what it returned: typed when run() declares the type (a Workflow: its Output)
    error: str | None = None

class Files(Protocol):
    """Stores file contents (an artifact's file, uploads) in the runtime's file store."""
    async def put(self, data: bytes, *, mime_type: str, name: str | None = None) -> FilePart: ...
    async def get(self, part: FilePart) -> bytes: ...

class Scheduler(Protocol):
    """Background jobs (no session, nothing user-facing) on the runtime's scheduler."""
    async def schedule(self, job: str, *, at: datetime | None = None, delay: float | None = None,
                       key: str | None = None, **kwargs: Any) -> str: ...
    async def cancel(self, key: str) -> None: ...

def job(name: str): ...        # @job("reindex") async def reindex(ctx: JobContext, index: str): ...
```

Requirements: a decorator, inherited and added to by subclasses — nothing on the `Agent` class itself; `requirements_of(cls)` reads them (the runtime's startup check, `describe()`). `ctx.require(X)` then returns the capability without a `None` check.

```python
@requires(Files, Scheduler)
class Reporter(Agent):
    async def run(self, ctx):
        files = ctx.require(Files)      # checked at startup: never None here
        ...

@requires(Agents)                       # TypeError: child runs are always there (ctx.agents)
```

Custom parts: a dataclass subclass of `Part` with an `x-` kind, registered with `@part` so it survives the wire (a worker, the data store, a replay) as itself:

```python
@part
@dataclass(frozen=True, kw_only=True)
class ChartPart(Part):
    kind: str = "x-chart"
    spec: Mapping[str, Any]
    title: str = ""
```

Registration is per process, at import; a process without the code (an API role without the agent, an older version) keeps it as an `UnknownPart` and passes it on unchanged. The `x-` prefix keeps custom kinds clear of the ones the SDK adds later; a kind can be registered by one class only. For most custom content a `DataPart` with a schema is enough and travels through every protocol unchanged; a custom part is worth it for a typed object your own agents use often.

Where events go: `MessageCompleted` puts the message on the thread; every other event goes to the run's stream (live, persisted per the data store's option), except `Usage` and the last `StateSnapshot`, which go to the run record. `ctx.post` puts an agent item on the thread.

Messages: a run writes only on its own thread — `ctx.send(...)`, or `ctx.stream()` to stream one. It talks to the threads it created with their handles: `await fork.send(...)`, `await weather.send(...)` — a message on that thread, in the inbox of the run active there, or starting a new run. Whoever created a thread listens to it; nothing is pushed back. Siblings have no handle to each other: they coordinate through their parent. A message is something one party says to another; a post (`ctx.post("x-…")`) is something runs coordinate on, not shown as chat and not in sessions by default.

An artifact is a part: `ArtifactPart(name, description, parts)` in a message, next to its other parts — prose and an artifact in one reply, as an A2A task has messages and artifacts (`await ctx.send("Here's the report.", ArtifactPart(name="report", parts=[report_file]))`). It is the minimum protocols need to round-trip; versions ("the latest report"), an artifact API and sharing rules are app-level — `ctx.state`, the app's own store, its own session provider — and llamphouse's visibility rules and `permissions` already decide who reads the thread and the file store. Adapters render it in their own terms (an A2A artifact; inline or as data where a protocol has none); file contents live in the file store, referenced by URI.

Client tools are a durable wait: `call_client_tool` sends the call to the client as `input_required` with a `ToolCallPart`, and the client's next input carries the `ToolResultPart`. That covers AG-UI frontend tools and Responses function tools with one mechanism.

What changed from 1.x: `context.reply` / `send_chunk` → `ctx.send` / `ctx.stream()`; `@step` stays; `wait_for_input` → `request_input` / `ask`; `call_agent()` → `Agents.invoke`; `handover_to_agent()` → `Agents.handover` (a new primary run on the same thread with its own session); `context.emit_state` → `StateSnapshot` / `StateDelta`.

A `Workflow` is an `Agent` with a type: see "Workflows".

The name comes from the agent definition file; the version from the code, because it identifies what actually ran — it travels with the code, and works without YAML (local dev, code-first setups, tests):

```python
class HrAssistant(Agent[HrConfig]):
    version = os.getenv("AGENT_VERSION", "dev")       # e.g. set by CI to the git commit
    # or: version = importlib.metadata.version("hr-agents")
```

Without one, the runtime uses `LLAMPHOUSE_AGENT_VERSION` from the environment, else the run is unversioned. The instance's `definition: hr-assistant@1.2` is a deployment reference (which image to run), not the reported version.

Every run records the agent version that ran it and the instance's config revision (`runs.agent_version` / `agent_revision`, in `run.created`, on its traces as `llamphouse.agent.version` / `llamphouse.agent.revision`, in `ctx.me.version` / `ctx.me.revision` for the agent's own logs) — "it behaves differently since yesterday" can be new code or a changed config — for reporting and logging: Compass shows it per run and breaks metrics down by version. It is not a routing key — llamphouse never sends a run back to the version that started it. In-flight runs meet the code that is deployed now; how they deal with a change is up to the agent's code.

### Output

A run's output is what `run()` returns, and its type is `run()`'s return annotation — no new API:

| `run()` is annotated | Output |
| --- | --- |
| `-> SomeModel`, `-> list[str]`, any type pydantic can store as JSON | checked when the agent is loaded (a type that can't be stored refuses the agent, before any run starts); the returned value is validated when the run ends; the JSON Schema is published in `describe()` (agent card, Compass); `wait()` gives it back typed |
| not at all | any value that can be stored as JSON, checked when the run ends |
| `-> None` (or an async generator) | no output: returning a value fails the run |
| a `Workflow` | its `Output`; an annotation on `run()` must say the same, or the workflow is refused when loaded |

The run record stores the output as JSON (`runs.output`), so it comes back as JSON would: a tuple returned without a type comes back as a list; with `-> tuple[str, int]` it comes back as a tuple. A value that doesn't fit fails the run, with what is wrong and where: `output_invalid` (not its declared type), `output_undeclared` (declared `-> None`), `output_not_storable` (not JSON — an object, a client). What the run already did stands: its messages are sent, its steps recorded. No lossy fallbacks (`repr`, pickle): an output is either what was declared, or the run says why not. The Scenario runtime and `FakeRun` store outputs the same way, so a test sees what a caller gets.

Output or message? Output is for whoever reads the run once it's finished (`wait()`, the API, Compass); a message is what the run says on its thread, for whoever listens now (the user, the run that created the thread).

### Examples

```python
from openai import AsyncOpenAI
from llamphouse.sdk import Agent, ArtifactPart, Config, DataPart, RunContext, Usage, Viewpoint

from .search import search              # the agent's own retrieval

class HrConfig(Config):
    instructions: str
    index: str
    model: str = "gpt-chat"
    top_k: int = 8

def role(ctx, m):                                       # the model's roles: which is the provider's convention
    return "assistant" if ctx.session.is_own(m) else "system" if m.sender.kind == SenderKind.SYSTEM else "user"

class HrAssistant(Agent[HrConfig]):
    """Answers HR questions from the policy index."""
    llm = AsyncOpenAI()

    async def run(self, ctx: RunContext[HrConfig]) -> None:
        question = ctx.input.text
        docs = await ctx.step("search", search, question, index=ctx.config.index, top_k=ctx.config.top_k)
        if not docs:
            year = (await ctx.ask("Which year's policy do you mean?")).text   # input_required
            docs = await ctx.step("search", search, f"{question} {year}", index=ctx.config.index)

        history = await ctx.session.messages()   # the conversation as the provider keeps it (revised in the background)
        stream = await self.llm.chat.completions.create(
            model=ctx.config.model, stream=True, stream_options={"include_usage": True},
            messages=[{"role": "system", "content": ctx.config.instructions},
                      {"role": "system", "content": "\n\n".join(d.text for d in docs)},
                      *[{"role": role(ctx, m), "content": m.text} for m in history]],
        )
        async with ctx.stream() as reply:
            async for chunk in stream:
                if chunk.choices:
                    await reply.write(chunk.choices[0].delta.content or "")
                if chunk.usage:
                    await ctx.emit(Usage(model=ctx.config.model, input_tokens=chunk.usage.prompt_tokens,
                                         output_tokens=chunk.usage.completion_tokens))

        await ctx.send(ArtifactPart(name="sources", parts=[DataPart(data=[d.ref for d in docs])]))
```

```python
import asyncio

from llamphouse.sdk import Agent, Progress, RunContext

class ResearchLead(Agent):

    async def run(self, ctx: RunContext) -> None:
        await ctx.emit(Progress(label="Researching"))
        news = await ctx.call("news-researcher", ctx.input)       # a new thread of each, with the question
        papers = await ctx.call("paper-researcher", ctx.input)
        await asyncio.gather(news.wait(), papers.wait())                  # both runs, finished
        news, papers = await news.messages(), await papers.messages()     # what was said there: their answers
        await ctx.post("x-research.findings", data={"news": len(news), "papers": len(papers)})
        await ctx.send(summarize(news, papers))       # the agent's own summary code
```

### Testing

`RunContext` is a protocol the runtime implements. `llamphouse.sdk.testing` ships `FakeRun`, an in-memory implementation for unit tests without the runtime. The test scripts what the outside world does — `answers` (input requests), `inputs` (messages for the run), `events` (thread events for subscriptions), `signals`, `client_results`, a seeded `conversation` (what `ctx.session.messages()` shows before the run), `state` and `run_state`, what forks reply (`forks=`), and fake capabilities (`FakeAgents` with a scripted reply per agent, `FakeFiles`, `FakeScheduler`) — and `FakeRun` records what the agent does: `emitted`, `replies`, `posts`, `steps`, `input_requests`, `sleeps`, `restarts`, `forked` and `agents.threads` (with what was sent to each), `session_entries`, `revisions` (with their options), `output`. Nothing waits: a wait with nothing scripted fails the test at once (`NotScripted`), so a test shows exactly what the agent asked for. `start()` runs the agent as the runtime would: required capabilities checked, a workflow's Input and Output validated, restarts followed.

What a run works with is two things: `ctx.input`, the one message that started it — its sender says who or what (a person, a calling agent, the parent, a trigger from the agent file, a service through the API) — and `await ctx.session.messages()`, its working memory: the conversation as the agent sees it, input included, plus its own entries, as the provider the agent file names keeps it. A run may also read the record of its own thread (`await ctx.thread.messages()`) and of the forks it created; reads are recorded, so a replay sees the same. Subscriptions deliver the thread events a run waits for (visibility-checked, recorded for replay); Compass, adapters and audit read the whole record.

For several agents, `llamphouse.sdk.testing.Scenario` runs real agents in memory, the way the runtime does: one agent per thread, forks of threads, inboxes, waiting runs that hold no worker, parallel runs and calls to other agents, handover, sessions (with a stand-in for compaction), thread state as events and run state, files, steps with replay, leases and recovery, limits per thread and per run tree, on a fake clock with crashes you inject. A test drives the outside world (`say` as a person or a calling agent — also on a fork, `answer` / `authorize`, `signal`, `advance` the clock, `crash` and `recover`) and asserts on what happened (`messages`, `timeline`, `runs` with their status, output, replays and stream, `session`, `state_of` a thread at any position, `forks`). It is the executable spec of the runtime's behaviour: the same scenario tests are meant to run against the real runtime.

```python
async with Scenario({"lead": Lead, "researcher": Researcher}) as sc:
    thread = await sc.thread("lead")                          # one agent per thread
    await sc.say(thread, "Plan my trip", by=Sender.person("an"))
    await sc.settle()
    await sc.say(thread, "cheaper", by=Sender.person("an"))   # steers the waiting run
    await sc.settle()
    assert sc.runs(thread)[0].text == "done: hotels, cheaper"
```

The scenario tests also build a shared conversation on top — a small `Room` that gives each agent its own thread and relays messages between them — as an example of that layer.

On the wire (`llamphouse.sdk.wire`), parts, messages and events become JSON-ready dicts and back; an unknown kind becomes `UnknownPart` / `UnknownEvent` and is written back unchanged, and a known kind with a field this SDK version doesn't know loses only that field.

```python
from llamphouse.sdk.testing import FakeRun

async def test_asks_for_the_year():
    run = FakeRun("What is the leave policy?", config=HrConfig(instructions="…", index="empty"),
                  answers=["2026"])
    await run.start(HrAssistant())
    assert [r.reason for r in run.input_requests] == ["Which year's policy do you mean?"]
    assert run.replies[-1].artifacts[0].name == "sources"
```

The config schema is published from the class so the API and portal can validate instance config without importing agent code.

## Configuration

Agent files describe what an agent is and needs; runtime files describe where and how it runs. The agent file never says whether workers share a process.

Agent file (travels with the agent, maps to CRDs later):

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: AgentDefinition
metadata: { name: hr-assistant }
spec:
  image: myacr.azurecr.io/hr-assistant:1.2.0   # ignored in local dev
  entrypoint: agents.hr:HrAssistant
  mode: serve            # serve | job
  pool: shared           # shared (default) | dedicated | <name>
---
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: { name: hr-thailand }
spec:
  definition: hr-assistant@1.2
  display_name: HR Assistant Thailand
  config: { instructions: "You help Bangkok staff…", index: hr-th }
  execution: { timeout: 60, retries: 2, concurrency: 20 }
  session:
    provider: my_sessions:SummarizingSession
    config: { budget: 100k }
  triggers:
    - webhook: { path: /triggers/hr-thailand }
    - schedule: { cron: "0 7 * * MON-FRI", input: { text: "Daily digest" } }
  requirements: { isolation: none, resources: { memory: 1Gi } }
  permissions:                           # what this instance may use - the runtime enforces it
    agents: [researcher, hotels]         # child runs it may start (ctx.agents)
    files: read-write                    # the file store (ctx.capability(Files))
  publish:                               # how it is presented through each protocol; the same in every environment
    a2a:
      skills:
        - { id: leave, name: Leave questions, tags: [hr], examples: ["How many leave days do I have?"] }
      input_modes: [text/plain]
      output_modes: [text/plain, application/pdf]
      extensions: [{ uri: "https://example.org/ext/citations/v1", required: false }]
```

An agent instance's identity follows Kubernetes: `metadata.name` is the reference everywhere (config, API paths, `ctx.me.name`, senders, filters, permissions) — a unique, URL-safe slug, **immutable** (renaming is creating a new agent); `metadata.uid` is assigned by the runtime when the agent is created, never reused, and recorded on every run (`runs.agent_uid`), so a recreated `hr-thailand` is told apart from the old one; `spec.display_name` is optional free text for UIs and the A2A card. Agent code sees the name, type, display name, version and revision as `ctx.me`; the uid is the runtime's, for audit and Compass.

What an agent *needs* is code (`@requires(Files)` on the class, inherited by subclasses, published by `describe()`); what an instance *may use* is deployment (`permissions` in the Agent file). The runtime checks both before anything runs.

Runtime file (one per environment, Helm values in Kubernetes):

```yaml
# llamphouse.dev.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Runtime
topology: all-in-one
run_queue: { in_memory: }
event_queue: { in_memory: }
scheduler:  { in_memory: }
session_store: { in_memory: }
config_store: { in_memory: }
file_store: { in_memory: }
data_store: { in_memory: }
adapters:
  - responses: { name: api }
  - a2a: { name: a2a, prefix: /a2a, agents: "*" }
  - ag_ui: { name: web, prefix: /ag-ui, agents: "*" }
  - compass: { name: compass, prefix: /compass }
```

```yaml
# llamphouse.prod.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Runtime
topology: distributed
run_queue:   { servicebus: { namespace: my-ns } }
event_queue:  { redis: { url: ${REDIS_URL} } }
scheduler:    { servicebus: {} }
session_store: { postgres: { database_url: ${DATABASE_URL} } }
config_store: { postgres: { database_url: ${DATABASE_URL} } }
file_store:   { azure_blob: { account_url: ${BLOB_URL}, container: llamphouse } }
data_store:   { postgres: { database_url: ${DATABASE_URL} } }
identity:     { azure_workload_identity: {} }        # outbound: how llamphouse authenticates to others
auth:                                                  # inbound: who may call; published as the A2A card's security schemes
  - oidc: { issuer: https://login.microsoftonline.com/<tenant>/v2.0, audience: api://llamphouse }
  - api_key: { header: X-API-Key, keys: ${API_KEYS} }
notifications: { webhook: { retries: 8, dead_letter: true } }   # delivery to external subscribers (A2A push, human notifications)
remote_agents:                                         # callable as child runs, like local ones
  - name: travel-partner
    a2a: { card: https://partner.example/.well-known/agent-card.json, auth: { oauth2_client: { client_id: ${PARTNER_ID}, secret: ${PARTNER_SECRET} } } }
adapters:                                              # named instances: the same kind may appear several times
  - a2a:
      name: partners                                   # public: two agents, OAuth, signed cards
      prefix: /a2a/partners
      agents:
        hr-thailand: { skills: [leave] }              # narrows the agent's own publish.a2a for this audience
        travel: {}
      auth: [oidc]                                     # entries from `auth` above
      bindings: [jsonrpc, rest]
      provider: { organization: StackWise, url: https://stack-wise.co.uk }
      sign_cards: { key: ${CARD_SIGNING_KEY} }
      push_notifications: true
  - a2a:
      name: internal                                   # internal: every agent, API keys, gRPC as well
      prefix: /a2a
      agents: "*"
      auth: [api_key]
      bindings: [jsonrpc, grpc]
  - responses: { name: api, agents: "*" }
  - ag_ui: { name: web, prefix: /ag-ui, agents: [hr-thailand] }
```

Every component uses the existing registry pattern (`{name: {kwargs}}`, kwargs validated against the constructor).

## External subscribers (notifications)

Some parties outside llamphouse want to hear about a run without holding a stream open: an A2A client that registered a push notification config for a long-running task, a human who should know a run waits for their input, another system. The runtime delivers run and thread events to external subscribers: a subscription (by run, thread, agent, kind — the same `Filter`) with a webhook URL and its authentication, delivered at least once from the event log (an outbox), with retries and backoff, a dead-letter list, and an idempotency key per delivery. The A2A adapter maps push notification configs (create, get, list, delete) onto it; Compass shows deliveries and failures. Mechanism only: who gets notified of what is configuration or agent code.

### System events

Next to thread events and run events, a third, small kind: **system events**, runtime-wide events that belong to no thread. Persisted for audit, followed by external subscribers and Compass. The first are the files service's:

| Event | When | Typical use |
| --- | --- | --- |
| `file.stored` | an upload, or a run's `put` | virus scanning, indexing it for search or retrieval, quotas |
| `file.claimed` | a message referenced it (with the thread and the message) | audit; an index learns which conversation it belongs to |
| `file.deleted` | with a reason: `thread_deleted`, `expired` | compliance and audit trails, cleaning derived indexes |

A file entering a conversation needs no thread event of its own: the message event carries its `FilePart`. Later candidates for system events: agents deployed or retired, workers joining or leaving.

## Adapters

Adapters are named instances in the runtime file, so the same kind can appear several times with different settings: a public A2A endpoint exposing two agents with OAuth and signed cards next to an internal one exposing every agent with API keys. Each instance has a unique `name`, a `prefix` (its endpoint), the `agents` it exposes (`"*"` or a list, optionally with per-agent overrides), its `auth` (entries from the runtime file's `auth`), and its protocol's own settings.

Parts a protocol doesn't know — custom `x-` parts, and kinds from a newer SDK — get a defined fallback per adapter: A2A sends a data part with the part's fields and `media_type: application/vnd.llamphouse.<kind>+json` (a llamphouse client on the other end rebuilds the part); AG-UI carries the same data in a custom event; Responses sends it as JSON text or leaves it out (the adapter instance's setting).

How an agent is presented through a protocol is not in its code. The agent's `publish.<protocol>` settings in its Agent file say it once, the same in every environment (an A2A card's skills, input and output modes, extensions); an adapter instance may narrow or override them for its audience. The code publishes only what it knows (`describe()`: type, description, version, requires, schemas).

## The A2A adapter (v1.0)

| A2A | llamphouse |
| --- | --- |
| Agent card | `Agent.describe()` (name, description, version, workflow schemas) + the agent's `publish.a2a` (skills, input/output modes, extensions) with the adapter instance's overrides + the adapter instance (`auth` → security schemes and requirements, provider, documentation URL, icon, push notification support, its interfaces and bindings, card signing) |
| Skills, per-skill security | `publish.a2a.skills`; per-skill security from the adapter instance's `auth` |
| Extensions (`uri`, `required`, `params`) | `publish.a2a.extensions`; the adapter refuses a request missing a required one (`A2A-Extensions`) before a run starts, and passes the activated ones in `ctx.request.extensions` |
| `SendMessageConfiguration.accepted_output_modes`, request metadata | `ctx.request.accepts`, `ctx.request.metadata` |
| Message (`message_id`, `context_id`, `task_id`, `reference_task_ids`, `extensions`) | `Message` (id, thread, addressed run, `references`, metadata) |
| Parts (text, raw / url with `media_type` and `filename`, data) | `TextPart`, `FilePart`, `DataPart` |
| Artifact (`artifact_id`, name, description, parts), streamed with `append` / `last_chunk` | an `ArtifactPart` in a message (its id derived from the run and its name); A2A artifact streaming is an adapter feature for later, if needed |
| Task states | run statuses, see the Run lifecycle |
| `TaskStatus.timestamp` | `ThreadEvent.at` and the run's status times |
| Send, Send streaming, Get, List (filters, pagination), Cancel, Subscribe | start or continue a run, the run's stream, the run API, cancel, `Last-Event-ID` resume |
| Push notification config | external subscribers |
| Extended agent card | the card with what the caller's identity may see (per-skill security) |
| `return_immediately`, `history_length` | the adapter: answer at once or wait for a terminal or interrupted state; trim the history |
| A direct message instead of a task | the adapter: a run that ends at once with one message and no artifact is answered as a message |
| Protocol bindings (JSON-RPC, gRPC, HTTP+JSON), `A2A-Version` negotiation, card signatures, `tenant` | the adapter (gRPC behind an extra); the tenant routes with `Identity.tenant` |
| Calling a remote A2A agent | `remote_agents` in the runtime file; `Agents.invoke` / `start` work the same as for local agents |

## Roles, CLI and packaging

One codebase and one image, several roles. Components talk only through the backbone (queue, stores, event queue), never in-process.

```bash
# local: everything in one process (today's fast experience)
llamphouse up --role all --config llamphouse.dev.yaml --agents ./agents

# prod: same image, one role per container
llamphouse up --role api       --config llamphouse.prod.yaml
llamphouse up --role worker    --config llamphouse.prod.yaml --pool shared
llamphouse up --role compass   --config llamphouse.prod.yaml
llamphouse up --role scheduler --config llamphouse.prod.yaml   # only if the backend needs a ticker
```

Test for each role: it starts alone on a separate machine with only config and backing services.

One package with extras. The base install is everything `llamphouse up --role all` needs with the in-memory backends (no SQLite: a dev setup keeps nothing across restarts), so a new user's first command works; extras add the other backends:

```toml
# pyproject.toml -> llamphouse
dependencies = ["pydantic>=2", "fastapi", "uvicorn", "pyyaml"]

[project.optional-dependencies]
postgres  = ["sqlalchemy", "asyncpg", "alembic"]
redis     = ["redis[hiredis]"]
servicebus = ["azure-servicebus"]
tracing   = ["opentelemetry-sdk", "opentelemetry-exporter-otlp"]
azure     = ["azure-identity", "azure-keyvault-secrets", "azure-storage-blob"]
s3        = ["aioboto3"]
all       = ["llamphouse[postgres,redis,servicebus,tracing,azure,s3]"]
```

Backends import lazily and fail with a clear message ("install `llamphouse[redis]`"). `llamphouse.sdk` never imports FastAPI or a backend, whatever is installed: that keeps schema publishing and `FakeRun` tests fast.

Official base image, users build a thin layer on top:

```dockerfile
FROM ghcr.io/llamp-ai/llamphouse:2.0-py3.12
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY agents/ /app/agents/
COPY agents.yaml /app/
CMD ["llamphouse", "up", "--role", "all"]
```

Image rules: pinned tags (`2.0`, `2.0.1`, `2.0-py3.12`), non-root, multi-arch, SBOM and signatures. Code mounting (volume, git-sync) is a dev mode only.

## Hub and spokes

The deployment model within one customer: one central **hub** and many **spoke** projects. Each spoke is a team's project with its own agent code and release cycle; its workers connect to the hub.

```
                         ┌─────────────────────── hub (one per customer) ───────────────────────┐
                         │  llamphouse api · worker gateway · Compass · scheduler · adapters     │
                         │  stores: data, sessions, files, events, queues · auth (e.g. Entra ID)  │
                         └───────▲──────────────────────▲──────────────────────▲─────────────────┘
                                 │ outbound, private link│                      │
                ┌────────────────┴───┐      ┌───────────┴────────┐   ┌─────────┴──────────┐
                │ spoke: hr          │      │ spoke: finance     │   │ spoke: travel      │
                │ project scope      │      │ project scope      │   │ project scope      │
                │ its agent code     │      │ its agent code     │   │ its agent code     │
                │ workers (pool: hr) │      │ workers (finance)  │   │ workers (travel)   │
                └────────────────────┘      └────────────────────┘   └────────────────────┘
```

| | Hub | Spoke (project) |
| --- | --- | --- |
| Roles | `api`, `compass`, `scheduler`, the adapters (A2A, AG-UI, Responses endpoints for all agents), the worker gateway | `worker`, for its own pool |
| Data | every store: threads, runs, sessions, files, events, queues | none: it reads and writes through the gateway |
| Agent code | none (shared hub agents excepted) | its own image, its own release cycle |
| Agent files | the registry: what is deployed where | the source: applied to the hub from CI (`llamphouse apply`) |
| Credentials | owns the stores' | per project, least privilege |

**Scopes: two levels.** Every agent, thread and file belongs to the **hub** (shared agents every project may use, e.g. a summariser) or a **project** (`hr`). Names are unique per scope and written `project/name` across scopes (`finance/budget-check`). A run lists and calls agents of its own project and the hub, within `permissions.agents`. Quotas (runs, threads, file storage) are set per project.

**Who owns a thread.** A thread belongs to its agent, its project and its creator (a person, a service, an external agent), plus optional members — other people or callers allowed in:

```sql
threads(id, …, agent, project, created_by, members jsonb)
```

Reading, continuing, answering its runs and downloading its files is for its creator and members (and operators in Compass); every adapter checks it on every call — A2A, Responses and REST alike — so a guessed thread id gives nothing. Push notifications only go to callers who may read the thread.

**The worker gateway.** Spoke workers connect outbound only, over private networking (spoke networks reaching the hub through private endpoints), and never get store credentials. Through the gateway a worker pulls its pool's runs (long polling or a stream) and does everything a run does — events, steps, state, session, files — as calls on the hub. Its credentials (mutual TLS or workload identity) are per project: it pulls only its own pool's runs and touches only those runs' threads and files. Leases and recovery work as before: a dropped connection lets the lease expire and the run is taken over. Agent code doesn't change — the SDK is the same; only the worker's plumbing differs. In-cluster workers (in the hub itself) keep talking to the backbone directly.

```yaml
# the hub's runtime file
pools:
  - { name: shared }                                  # in-cluster workers
  - { name: hr, project: hr, remote: true }           # spoke workers, through the gateway
  - { name: finance, project: finance, remote: true }
worker_gateway: { listen: 0.0.0.0:7443, auth: { mtls: { ca: ${SPOKE_CA} } } }
```

```bash
# in a spoke
llamphouse apply --hub https://hub.internal --project hr ./agents/*.yaml     # from CI
llamphouse up --role worker --pool hr --gateway https://hub.internal:7443
```

**Registration.** A spoke applies its agent files to the hub; its workers report their pool, llamphouse version and the agents they serve. The hub knows where each agent can run — Compass shows it — and refuses to route a run to a pool with no matching worker online (it waits queued, with a clear status).

**Calls between spokes go through the hub** (or a gateway, see "Parallel runs and calling other agents"). An agent in `hr` calling `finance/budget-check` creates a run of it on its own thread in the `finance` pool; `permissions.agents` (`agents: [finance/budget-check]`) decides whether it may, and the run's original caller travels along. Nothing connects spoke to spoke.

**A version window.** Spokes upgrade on their own schedule: the hub accepts workers within a supported range (the current and the previous minor version), and Compass shows which spoke runs which.

**Standalone.** A spoke still runs on its own for development — `llamphouse up --role all` with in-memory stores — with the same agent code it ships to the hub.

## Queue, pools and scheduler

One stream per pool replaces the single global stream, so workers only receive work they can run.

- Stream key `llamphouse:runs:{pool}`; definitions default to `pool: shared`.
- `llamphouse up --role worker --pool shared` loads only the definitions in that pool.
- The API resolves an agent instance to its definition's pool and enqueues there.
- Two execution modes: **serve** (long-running pool, many concurrent runs, scaled on queue depth) and **job** (a fresh process per run, for sandboxed or long work). Locally and in Docker Compose a job-mode run runs in a subprocess of the worker (no shared memory, killed on timeout); in Kubernetes it gets its own pod.

Run queue interface:

```python
class BaseRunQueue(ABC):
    async def enqueue(self, msg: RunMessage, pool: str) -> str: ...
    async def dequeue(self, pool: str, timeout: float) -> tuple[str, RunMessage] | None: ...
    async def ack(self, receipt: str) -> None: ...
    async def requeue(self, receipt: str, delay: float = 0) -> None: ...
    async def depth(self, pool: str) -> int: ...      # for autoscaling
```

Scheduler: a swappable component for delayed jobs, cron triggers, retries with backoff and background session work. When a job is due it is enqueued on the normal run queue, so workers don't know schedulers exist.

```python
class BaseScheduler(ABC):
    needs_ticker: bool = False

    async def schedule(self, job: Job, at: datetime, key: str | None = None) -> str: ...
    async def schedule_recurring(self, job: Job, cron: str, key: str) -> str: ...
    async def cancel(self, key: str) -> None: ...
    async def tick(self) -> int: ...   # release due jobs; returns count

@job("reindex")                         # llamphouse.sdk.job
async def reindex(ctx: JobContext, index: str): ...

scheduler = ctx.capability(Scheduler)
await scheduler.schedule("reindex", index="hr-th", delay=600, key="reindex:hr-th")
```

| Backend | How due jobs are released | Ticker needed | Scale to zero |
| --- | --- | --- | --- |
| `in_memory` | Loop inside `--role all` | Yes (built in) | No |
| `redis` | Sorted set by due time, moved to the stream | Yes | With ticker |
| `postgres` | `scheduled_jobs` table, `for update skip locked` | Yes, or KEDA query on `run_at` | Yes |
| `servicebus` | Native scheduled messages | No | Yes |

The ticker runs as `--role scheduler` (single replica) or inside the API with leader election (Redis lock or Postgres advisory lock). A child run (`Agents.invoke`) is an agent run; a scheduled job is a system job with no session and nothing user-facing.

Every queue and scheduler backend passes a shared contract test suite: idempotency, cancellation, at-least-once release, no double firing with two tickers.

## Sessions backbone

> **Decision (Oct 30, 2026).** A session is an agent's working memory on a thread: the thread's messages plus the agent's own **entries** that are not on the thread (tool calls and results, retrieved documents, notes). Reading it never computes. A **revision** (a summary, compaction, anything the provider does) is its own process — a job with a trigger and a result — started by the provider's policy as things are added, or on request (`ctx.session.revise(**options)`); runs carry on meanwhile, or wait for it (`settled()`). Its events go on the session's own log, not the thread. Built in the `Scenario` runtime; tested; shown in `examples/v2/02_CodingAgent`.

Users own the session implementation; llamphouse owns storage, access, lifecycle and guarantees. The backbone knows only generic operations; "compaction" or "summarize" is just a `reason` a provider gives.

**Thread and session, apart.**

| | what it is | who sees it |
| --- | --- | --- |
| the thread | the shared record: what was said between parties, run lifecycle, state changes | everyone with access: people, the run that created it, adapters (A2A, AG-UI), Compass, audit |
| the session | one agent's working memory on that thread: the thread's messages it may see, plus its own entries, as the provider shapes them | that agent's runs; Compass and audit through the session store |

Tool calls and their results, retrieved documents, scratch notes are working material, not conversation: they go in the session (`ctx.session.add(...)`), not on the thread, so the transcript the user and callers see stays the conversation. The provider decides how entries show in `messages()` — keep the last few tool results, summarise old calls, drop documents after use. Live UIs that show tool calls (AG-UI) get them from the run's stream (`ctx.emit`), not from the session.

**What agent code does:**

```python
view = await ctx.session.messages()        # the session as it is: no provider work; recorded; view.version, view.revising
await ctx.session.add(ToolCallPart(...))   # an entry: in my working memory, not on the thread; its own (is_own)
await ctx.session.add(ToolResultPart(...))
await ctx.session.revise(reason="before the big step", target_tokens=20_000)   # ask for a revision; returns at once
view = await ctx.session.settled()         # wait until no revision is in flight (no worker held)
ctx.session.is_own(message)                # did I say it - the model's assistant role
```

- **Reading is cheap and never computes.** `messages()` returns the current version plus what was added since. No hidden latency on a read, and a replay sees the same view.
- **Entries** are placed after the thread position they were added at, so the session is one line in the order things happened. They are durable and recorded like steps, the agent's own (signed by the run), copied into a fork with the session, and passed on in a handover only if `session=` says so. They are not shared: a run reading a fork's thread sees what the fork said there, not its entries.
- **`revise(**options)`** passes its options to the provider (a reason, a target, a hint). The provider may decline. One revision at a time per session: a request while one runs joins it.
- **`settled()`** is for a run that needs the revised version — before forking, before a big model call. Most agents never call it.

**Revisions.** The provider watches what is added (`on_event` for the thread's messages, `on_entry` for entries) and starts a revision when its policy says so — "over 60k tokens", "every 20 messages", "after each run". A revision is a **job**, not a run (runs belong to a thread and its agent): durable, retried, traced, with a cost, on a worker of the agent's pool (the provider's code is deployed with the agent), shown in Compass under its session. It works on a snapshot of the session, and its result is a new version, swapped in atomically: what was added meanwhile survives, after the revised part. A failed revision leaves the session as it was.

1. **Versioned session store**, opaque to the runtime, with compare-and-swap:

```python
class SessionStore(Protocol):
    async def get(self, session_id) -> tuple[dict, int]               # (data, version)
    async def put(self, session_id, data, if_version: int) -> int
```

2. **Generic operations**, each versioned and carrying `reason` + `metadata`:

```python
await session.append(items)                                 # thread messages and entries, in order
# ranges are positions (stable), not item indices (which shift after every revision)
await session.replace(range=(120, 480), items=[summary], reason="summarize",
                      metadata={"model": "a-small-model", "tokens_before": 41_000})
await session.remove(range=(10, 40), reason="redact")
```

3. **Thread access with cursors**, so providers catch up incrementally or rebuild from zero:

```python
events = await thread.events(after=session.cursor, types=["message"], limit=500)   # cursor = a thread position
```

4. **Provider contract** (`llamphouse.runtime.sessions`, built): override what you need; the defaults show everything, never revise and decline requests.

```python
class SessionProvider:
    def view(self, session: SessionInfo) -> Sequence[Message]                          # what messages() shows: cheap
    def on_event(self, session: SessionInfo, event: ThreadEvent) -> Revise | None      # a thread message came in
    def on_entry(self, session: SessionInfo, entry: Message) -> Revise | None          # an entry was added
    def after_run(self, session: SessionInfo, run: RunResult) -> Revise | None         # a run ended
    async def revise(self, session: SessionInfo, options) -> Sequence[Message] | None  # the job; None declines
```

`SessionInfo` is the session as a provider sees it, read-only: its messages (thread messages and entries, in order), version, revisions, whether one is in flight. The policy hooks are plain functions, called inline as things happen: they only decide, returning `Revise(options)` to ask for a revision. The work is `revise`, run as a job; it returns the new messages, or `None` when there's nothing to do; raising leaves the session as it was. A request from agent code (`ctx.session.revise(**options)`) goes straight to `revise`, with `reason="requested"` added to its options. Providers in the box: `RollingWindow(size)`, the default — the latest messages, no revisions; `Summarizing(summarize, max_messages=, keep=)`, a reference — past `max_messages`, everything but the last `keep` becomes one summary written by `summarize`, the app's own function (usually a model call; llamphouse calls no model; a request may pass `keep=`).

5. **Lifecycle events**, on the session's own log — not the thread — and streamed to clients that watch it (Compass, AG-UI front-ends showing "revising…"):

| Event | When |
| --- | --- |
| `session.created` | a session is made for a thread (first run, a fork, handover, reset) |
| `session.entry.added` | an entry was added: `{run_id, message_id, at}` (the thread position it follows) |
| `session.revision.started` | a revision begins: `{options, version, at}` (`options.reason`: `policy`, `requested`, the provider's own) |
| `session.revision.completed` | it was stored: `{options, version, before, after, metadata}` (e.g. `tokens_before`, `tokens_after`, `model`) |
| `session.revision.declined` | the provider had nothing to do (`revise` returned `None`); the session is unchanged |
| `session.revision.failed` | it failed; the session is unchanged: `{options, error}` |

Compass lines the session's log up with the thread by position, so a revision or an entry shows where it happened in the conversation.

Concurrency rules:

- A revision works on a snapshot and replaces that range: appends made meanwhile survive. The new version is swapped in with compare-and-swap; until then everyone reads the old one (`view.revising`).
- One revision in flight per session; a second trigger or request joins it.
- A run reading while a revision is in flight gets the current version. If it needs the revised one, it waits (`settled()`, no worker held); never block indefinitely: a provider sets a timeout on its jobs.
- A failed revision leaves the session unchanged; the thread is the source of truth for its messages, and entries are kept in the session store, so a session can always be rebuilt.

Where the session sits, end to end:

| Where | What |
| --- | --- |
| The agent file | which provider and its settings (`session: { provider: …, config: … }`; a rolling window by default) |
| `session=` on `ctx.fork`, `ctx.call`, `handover` | what a new session starts with: a copy of this one, a view or messages the agent computed, or nothing |
| `ctx.session.messages()`, `add()`, `revise()`, `settled()`, `is_own()` | what agent code does: read it, add entries, ask for a revision, wait for one |
| The session store | where it lives: versioned, compare-and-swap, one per agent per thread, with its log |
| The provider | who shapes it: `view`, the policy hooks, `revise` as a background job |

Agent code adds to its session and may ask for a revision; it doesn't edit what's there — what goes into the model's context is the provider's, so it can be swapped by configuration without touching agent code.

Ships in the box: `RollingWindow` (the default) and `Summarizing` (a reference); more reference providers (full history, token-budget compaction) can follow in an optional package. In the `Scenario` runtime, `Scenario(sessions=provider)` or `sessions={"agent-name": provider}` (as the agent file names it) sets them; revisions run as jobs on the same queue as runs.

### Sessions of forks and other agents

A session belongs to an agent on a thread, not to a run: the runs on a thread share their agent's session there, so the agent remembers across runs, and `ctx.restart` keeps it. A fork and another agent's new thread each get a session of their own, starting from what `session=` says:

| `session=` | The new session starts with | Default for |
| --- | --- | --- |
| `ctx.session` | a copy of this run's session, brought up to date first (compacted as it is). For a fork, what this agent said stays its own: it is the same agent carrying on | `ctx.fork` |
| a `SessionView` (`await ctx.session.messages()`, or a slice of it) or any list of messages | exactly those: a trimmed view, a slice, a summary the agent wrote | |
| `None` | nothing: built from the thread's own events (a fork: from the fork point) | `ctx.call`, `handover` |

- A session is bounded working memory, so copying it is cheap (unlike a thread, which is forked by reference). It is recorded when the call is made: a replay doesn't recompute it, and Compass shows what the other side started with.
- No shared session: a parent and a fork writing one session would conflict on every write and interleave their context. The parent hears from its forks by listening, and copies over what it wants.
- The default provider shows a run everything on its thread except what is addressed to another run. A fork's work stays on the fork's thread; the parent's session holds only what the parent itself wrote or copied.
- Handover passes the thread to another agent; `handover(…, session=ctx.session)` starts it from a copy of the previous agent's session (what that agent said is someone else's for the new one).
- One writer per session at a time, so a parent compacting its own session never conflicts with its forks. Their sessions emit lifecycle events too, and are kept after the run ends (Compass shows what each saw), for as long as the data store keeps them.

## Workflows

The run is the workflow: llamphouse gives durable execution as a mechanism, and every workflow primitive is a run primitive. A `Workflow` is an `Agent` subclass with `type = AgentType.WORKFLOW`: same contract, same durability, same deployment. The type tells Compass and the API what it is, so they can show and start it differently, without importing the code (the type and the Input/Output schemas are published with the config schema).

| Workflow concept | In llamphouse |
| --- | --- |
| a workflow instance | a run (durable, leased, replayed; `ctx.restart` for long-lived ones) |
| an activity | `ctx.step` / `@step` (recorded; runs at least once) |
| a timer | `ctx.sleep` |
| a signal, an external event | `ctx.wait_for_signal`, or a subscription on the thread |
| a human task | `ctx.request_input` / `ask` |
| a sub-workflow, fan-out / fan-in | child runs: `Agents.start` / `invoke` (a workflow can call agents and the other way around) |
| continue-as-new | `ctx.restart()`, with `ctx.run.state` |
| a trigger (cron, webhook, API) | `triggers` in the agent file, or `POST /v1/workflows/{name}/runs` with the Input as JSON |
| system work | `@job` on the scheduler (no thread, no run) |

What the type changes:

- **Typed input and output.** `Input` and `Output` are pydantic models: the input is validated before the run starts (a bad request never becomes a failed run), `ctx.input.data` holds it, and the `Output` that `run()` returns becomes the run's output (a message with an `ArtifactPart` named `output` holding it as data, and the `run.completed` payload). As a child run, a workflow is invoked with its Input and returns its Output.
- **No conversation by default.** A top-level workflow run gets a new thread (its record) and receives nothing without asking (`inbox = ()`); the session provider isn't loaded unless the workflow asks for `ctx.session.messages()`. A workflow invoked from a conversation runs as a child on that thread, like any child run.
- **Compass shows it as a process, not a chat:**
  - a workflow list: runs per workflow, their status, the current step, what a waiting run waits for (a timer and when it is due, a signal key, an input form);
  - a run view as a step timeline (Gantt): steps with durations and attempts, replayed vs. executed, waits, child runs nested as sub-workflows, the input and the output;
  - "Start" with a form generated from the Input schema; "Answer" on input requests; retry and cancel;
  - per workflow: step durations and failure rates across runs.

  Agents keep the conversation view; both share the detailed timeline. An unknown type (a newer `AgentType`) is shown as an agent.

Not in the core: declarative (YAML or visual) workflow definitions — policy; an agent that interprets a graph definition runs durably like any other — and graph engines, which plug in as integrations on top of the primitives (`llamphouse.integrations.langgraph` behind a `langgraph` extra: the checkpoint in the run's record or the thread's `State`, `interrupt()` as `request_input`).

Durable code has two rules, the same for agents and workflows:

- Steps run at least once (a crash after the work but before its record runs it again): side effects are idempotent or carry their own idempotency key.
- Everything outside steps is deterministic up to the last recorded step. A replay that doesn't match its record (a step called under another call key than recorded, e.g. after a deploy) fails the run clearly instead of taking another path silently. `ctx.restart` keeps the window small. Handling a code change for runs already in flight is the agent's own business, in its code (for example a flag in the restart state, or a branch on data the run already recorded); llamphouse only records the version that ran each run, for reporting.

```python
class InvoiceIn(BaseModel):
    invoice_id: str

class InvoiceOut(BaseModel):
    paid: bool
    reference: str | None = None

class PayInvoice(Workflow[Config, InvoiceIn, InvoiceOut]):
    async def run(self, ctx: RunContext[Config]) -> InvoiceOut:
        invoice = await ctx.step("fetch", erp.get_invoice, ctx.input.data.invoice_id)
        if invoice.amount > 10_000:
            approval = await ctx.run.request_input(InputRequest(reason="approve_payment",
                                                            input_schema=Approval.model_json_schema()))
            if not Approval.model_validate(approval.parts[0].data).approved:
                return InvoiceOut(paid=False)
        ref = await ctx.step("pay", bank.pay, invoice, idempotency_key=invoice.id)
        await ctx.sleep(until=invoice.due_date)                # holds no worker
        await ctx.step("confirm", erp.mark_paid, invoice.id, ref)
        return InvoiceOut(paid=True, reference=ref)
```

## Files and artifacts

**A message never holds file bytes.** A file in a message is a `FilePart` with a reference (`files://…`) and its facts (`mime_type`, `name`, `size`, `sha256`). Whatever hands bytes in — a client through an adapter, or an agent creating `FilePart(data=…)` — the runtime stores them in the file store, fills in `uri`, `size` and `sha256`, and drops `data` before the message is written; also inside artifacts and tool results. Text and data parts stay inline: they are the message's content. One path for every file means one way to access, check, download and delete it; and the same bytes are stored once (by `sha256`).

**Uploads,** two ways that end in the same message:

- *In one go:* the client sends the message with the file in it (A2A `raw`, a multipart post, an AG-UI attachment); the adapter stores the bytes and writes the message with the reference, in one transaction.
- *Upload first, reference later* (large files, a progress bar while the user types): `POST /v1/files` stores the file and returns its reference; the client puts it in a message's `FilePart`.
- A part pointing at an outside URL (`https://…`) stays as it is; the agent fetches it if it wants. Importing outside URLs would have the runtime fetch arbitrary addresses on someone's behalf: the app's decision.

**Ownership and lifetime.** A file belongs to the thread whose message references it, and is deleted with the thread (messages are append-only, so no reference counting). An upload, or a file an agent puts, belongs to nobody until a message references it, and expires if none does:

```sql
files(id, thread_id null,            -- null until a message references it; then the owner
      sha256, size, mime_type, name, -- the bytes in blob storage or on disk, once per sha256
      created_by, created_at,
      expires_at null)               -- set at upload (e.g. now + 1 hour); cleared when referenced
```

A message referencing a file claims it in the same transaction; a clean-up job deletes files whose `expires_at` passed (abandoned uploads). A message may only reference files of its own thread or unclaimed ones — otherwise deleting thread A would delete a file thread B points at; reusing a file elsewhere means storing it again (cheap: identical bytes are kept once). `Files.put` is recorded like a step: a replay gets the same reference back instead of storing again.

The store behind it (`llamphouse.runtime.files`, `file_store` in the runtime file):

```python
class BaseFileStore(ABC):
    async def put(self, data: bytes | AsyncIterable[bytes], *, mime_type: str, name: str | None = None,
                  created_by: str, thread_id: str | None = None) -> FilePart: ...   # streamed for big files
    async def stat(self, file_id: str) -> FileRecord | None: ...
    def read(self, file_id: str, *, start: int = 0, end: int | None = None) -> AsyncIterator[bytes]: ...  # ranges
    async def claim(self, file_ids: Sequence[str], thread_id: str) -> list[str]: ... # all or nothing; newly claimed
    async def signed_url(self, file_id: str, *, expires_in: float) -> str | None: ... # None: the API serves it
    async def delete_thread(self, thread_id: str) -> list[FileRecord]: ...
    async def cleanup(self) -> list[FileRecord]: ...                                 # expired, unclaimed; atomic
```

Backends only store bytes and never emit events: the **files service** (`FileService`) wraps any store and emits the `file.*` system events once, whatever the backend. A custom store adds its own behaviour by wrapping another — e.g. holding files back until a virus scan passed: its `read` raises **`FileUnavailable`** (with `retry_after`) meanwhile, which an agent gets from `Files.get` and the API answers with "try again later". Scanning is the custom store's business, not llamphouse's.

**No locks: files and artifacts never change.** A file is written once and read any number of times; a new version is a new file, in a new message. An artifact is a part of a message, and messages are append-only: "editing" one is sending a new message, so two agents revising "report" at once both produce a message, in a definite thread order, and nothing is overwritten. The only places two operations meet the same record are atomic, not locked:

| Race | What happens | How |
| --- | --- | --- |
| two threads claim the same upload | the first wins; the other message is refused (`FileAccessError`) | `claim` is one conditional update (only if still unowned or already this thread's) |
| a claim meets the clean-up at expiry | either the claim wins (the file stays) or the clean-up does (the claim fails, the message isn't written) — never a referenced file deleted | `cleanup` deletes only rows still unclaimed and expired, in one conditional delete |
| a thread is deleted while a run reads its file | the read fails (`FileNotFound`) | acceptable: the conversation is gone |

Coordination — "only one agent revises the report at a time" — is the app's, with existing mechanisms: a lease in `ctx.state` (`set(..., if_version=)`), or optimistically a `based_on` reference in the new artifact's metadata.

`InMemoryFileStore` is the dev default and what `FakeRun` and `Scenario` use; local disk, Azure Blob and S3 follow, and every backend passes the same contract tests (`tests/contract/file_store`).

**Access and download.** `files://` is an internal reference, never a link. A file can be read by whoever may read a message it is in — the thread's visibility rules and `permissions` — or, before that, by its uploader. Downloading goes through the API:

- *A browser* (an `<img src>`, a download link) gets a short-lived **signed URL**: an Azure Blob SAS or S3 presigned URL (straight from storage, with range requests), or a llamphouse-served `GET /v1/files/{id}/content?sig=…&exp=…` for the in-memory and disk stores.
- *A server-side client* uses `GET /v1/files/{id}/content` with its credentials; access is checked on every request.
- The response carries the file's content type and name (`Content-Disposition`).
- Agents read files with `ctx.capability(Files).get(part)`, under the same rules: a run reads the files of its thread (and its ancestors' threads), or what it put itself.

**The artifacts view.** A read-only view over messages, so apps and clients don't each scan them:

```
GET /v1/threads/{thread_id}/artifacts?name=report
GET /v1/runs/{run_id}/artifacts                  # an A2A task's artifacts; a run's outputs in Compass
```

Each item: the `ArtifactPart` (name, description, parts — file parts with a signed URL), where it came from (message, position, run, sender, time). A view, not a store; no version semantics (the same name may appear several times, in thread order — "the latest" is the client's reading); the same access rules as messages; in the API, not the SDK.

**Who does what:**

| | llamphouse (mechanism) | The application (policy, product) |
| --- | --- | --- |
| Storing | the file store; bytes moved there at the edge; files owned by their thread; abandoned uploads cleaned up | keeping files longer (its own library or document store) |
| Access | enforced: a file is readable by whoever may read its message | who that is in product terms: teams, roles, share links, per-artifact sharing |
| Delivery | signed URLs, the authenticated content path, `POST /v1/files`; adapters render artifacts per protocol | where and how files appear: an artifact panel, previews, downloads |
| Finding | the thread API and the artifacts view | "all my reports", search, versions |
| Operators | Compass shows artifacts and files in the timeline | the end-user UI |

## Where state lives

| Scope | What | Where it lives | Shared by |
| --- | --- | --- | --- |
| One run | local variables; `ctx.run.state` across restarts (versioned key-value) | the worker's memory; the run record (`runs.run_state`) | that run |
| One thread | `ctx.state`: durable key-value, with versions; every write a `state.changed` event | the data store: the events, and the latest view (thread-scoped, deleted with the thread) | the agent's runs on its thread (its turns, runs started side by side through the API); a fork starts with the state at the fork, and its creator can read it (other agents have their own threads, and their own state) |
| One agent on one thread | the session: the agent's working memory (what its model sees) | the session store, versioned, compare-and-swap | that agent's turns on the thread; a fork starts from a copy |
| The client | `StateSnapshot` / `StateDelta` (e.g. AG-UI shared state) | the run's events; the last snapshot on the run record | the client |
| Across threads | not in the core | the agent's own database or cache | the agent's choice |

An agent's runs on one thread share state through `ctx.state` (a blackboard) and tell each other about it with `ctx.post`; a fork shares nothing after the fork (its creator can read its state, and copies what it wants), and another agent shares nothing at all — they talk by messages on their threads. Concurrent runs never silently overwrite each other: `update(key, fn)` is atomic (the runtime retries `fn` on a conflict), and `set(..., if_version=)` raises `StateConflict` when someone wrote first. State reads and writes are recorded like steps, so a run replaying after a crash sees the same values and doesn't write twice — tested in the scenarios with five runs side by side on one thread updating one counter.

## Parallel runs and calling other agents

One agent per thread (see "One agent per thread", "Threads, forks and state"): an agent works in parallel on **forks of its thread**, and talks to **other agents** on new threads of theirs — always by messages on threads, never to runs. The backbone guarantees ordering, durability and lifetimes; how work is split and what to do with what comes back is agent code.

The backbone guarantees:

- **Ordering:** thread positions come from the database, never worker clocks. Only thread events take the short per-append lock; run events (chunks) are numbered by their run, so streaming runs never wait on each other.
- **Attribution:** every event carries `run_id`, `agent` and `parent_run_id`.
- **Senders, not roles:** a message records who wrote it (`Sender`: a person, an agent, the system, a service, a trigger) — a fact, the same for every reader. A role is derived per reader (`ctx.session.is_own(m)`: OWN, OTHER, SYSTEM), decided by the run that wrote it: OWN is this run or an earlier run sharing its session (the agent's earlier turns); a parent or a parallel run is OTHER, even of the same agent; a caller (a person, another agent over A2A or internally) is OTHER. Adapters derive their protocol's roles the same way.
- **The inbox:** the messages sent to a run (read with `ctx.next()`). Nothing else reaches it by itself; a run that wants more subscribes.
- **Starting runs:** never by itself — explicitly, at most one active run per thread (see "Starting runs and the inbox").
- **Subscriptions:** a run subscribes to thread events by filter (kind, sender kind, run, origin, and `thread`: its own, or one it created — one subscription may cover several). Stored with the run, with a cursor per thread: delivery is at least once, survives a restart, and a matching event wakes a `waiting` run. Within a thread, posts (`ctx.post("x-…")`) and `ctx.state` coordinate.
- **No gap when a run ends:** what was sent to a run and never read is known when it ends (its inbox cursor and subscriptions), so whatever sits in front can start the next run with it.
- **Addressed answers:** the answer to `input_required` carries its run and goes only to that run; the first answer wins.
- **Operational limits:** one active run per thread; max runs per thread and per run tree (checked wherever a run is created — also against an agent that keeps forking itself), spawn depth, timeouts, permissions, cancellation cascading to children by default (overridable).

### Threads a run creates

| Who | How | What happens |
| --- | --- | --- |
| this agent | `await ctx.fork(message, session=ctx.session)` | a fork: a thread that starts from this thread at the current position, by reference (`forked_from = (thread, position)`), with its state; a run of this agent starts there. Its work stays there; the parent listens and can read its thread and state (see "Threads, forks and state") |
| another agent | `await ctx.call(agent, message, session=None)` | a new thread of that agent, with the message: its first run starts there. Talked to and listened to; not read beyond its messages |

Either way the run gets a `Thread` handle. `send` is a message to the run working there (its inbox; `NoActiveRun` without one); `start` begins its next run (`ThreadBusy` while one is active) — the run that started it is its parent. The thread's runs talk only on their own thread; their status (needing input, their end) is recorded there too, and nothing is copied to the creator's thread. The creator listens: `ctx.subscribe(Filter(thread=t, ...))` (with its own inbox, at once), `t.events(...)`, `await t.wait()` (the run handling its last message, finished: `status`, `output` — a workflow's typed Output — and `error`, never raising for a failed run), `t.watch()` (live, to pass on), `await t.messages()` (what was said there); a fork also `t.thread_events()` and `t.state`. Threads are not continued across turns: the handle lives as long as the run that created it.

Calling another agent is the same whether it runs in this hub, in another spoke, or behind a gateway (A2A): a message to its thread, a status and a result back. Internal is a faster route, not different behaviour — so orchestration and permissions between agents can live in a gateway outside llamphouse.

**Handover** passes the thread to another agent: this run ends, the other agent continues the conversation on the same thread — one agent at a time — starting with the message this run was answering. Something to tell it goes on the thread first, where everyone sees it (`await ctx.send("Passing you to our travel desk: …")`, then `handover`), or for it alone in its session: `session=ctx.session` (a copy of this agent's), a view, or messages the agent wrote (a brief). There is no `input=`: one way to put something on the thread. What goes along: the thread's history and state; not the old run's own state, nor the threads it created (attached ones are cancelled with it).

**Forwarding.** The first message of `ctx.fork` and `ctx.call`, and `thread.send`, may be a full `Message`. One that is already on the run's own thread (its input, a message from the conversation) is forwarded as is, with its sender: the user did say it. Any other is signed by the run: an agent can say anything, but not as someone else.

Fan-out to other agents, written by the agent:

```python
class TripPlanner(Agent):
    async def run(self, ctx: RunContext) -> None:
        travel = await ctx.call("travel-agent", ctx.input)    # a new thread of each
        policy = await ctx.call("policy-agent", ctx.input)
        await asyncio.gather(travel.wait(), policy.wait())
        await ctx.send(f"Options: {(await travel.messages())[-1].text}\nPolicy: {(await policy.messages())[-1].text}")
```

Timeline as Compass would render it (each agent on its own thread, linked by the run that created it):

```
10:02:01  run-1  trip-planner   run.created     user: "Prepare my Bangkok trip"
10:02:02  run-1  trip-planner   run.spawned     run-2 (travel-agent, its thread), run-3 (policy-agent, its thread)
10:02:06  run-3  policy-agent   run.completed   "Hotel limit €150/night"
10:02:09  run-2  travel-agent   run.completed   3 flight options
10:02:11  run-1  trip-planner   message          assistant: "Here's your plan…"
```

### Supervising runs

A question comes in; the agent's run starts runs — of itself or of other agents — to work in parallel, and supervises them: it steers them on user input or on its own. It stays one run, waiting in between, never a new run per event: the run is what A2A, AG-UI and Responses clients track, cancelling it cancels what it started, and a user message steers this work instead of starting a parallel conversation.

```
running ──(reads its subscription, nothing new)──▶ waiting   (no worker held)
   ▲                                                  │
   └──── a matching event: a user message, something said on a thread it created, a run there ending
```

1. The run creates threads (`ctx.fork`, `ctx.call`: creating the thread, its first run and recording the parent's step happen in one transaction, so a crash never leaves a thread its creator doesn't know), then reads its subscription — on its own thread and those threads at once. A subscription delivers events from the start of the run (on a created thread: from its start), not from the `subscribe` call, so a run that finished before the parent subscribed is not missed. Nothing new: it goes to `waiting`, holding no worker.
2. A matching event arrives: the backbone queues the run again and a worker picks it up.
3. It handles the event (`thread.send`, `respond`, `cancel`, or a new fork as a retry) and waits again, or finishes.
4. It can't finish before the runs its messages started: returning while they are active keeps it `waiting` until they end. Failing or being cancelled cancels them, unless the thread was created with `detach=True`.

Continuing stably: a run continues by replay (1.4), which has two weak spots for a supervisor that wakes on every event — replay cost grows with each wake-up, and a deploy during a days-long run replays old history through new code. Two rules:

1. **Every delivery is recorded.** Each event a subscription delivers is stored as a step of the run, so a replay sees the same events in the same order, even if more arrived since.
2. **`ctx.restart()` bounds the history:** the same run id and status, the code starts from the top, and the replay history is dropped; `ctx.run.state` carries what the run keeps between iterations. A supervisor calls it once per iteration: each wake-up replays a few steps, and a deploy affects only the next iteration.

```python
class Coordinator(Agent):

    async def run(self, ctx: RunContext) -> None:
        mine = ctx.run.state                                                # this run's: kept across restarts
        if await mine.get("forks") is None:
            forks = [await ctx.fork(DataPart(data=part)) for part in self.plan(ctx.input)]  # a run starts on each
            await mine.set("forks", forks)                                  # handles, kept as references

        forks = await mine.get("forks")
        ev = await ctx.subscribe(
            *ctx.inbox,                                                     # the user, on this thread
            *[Filter(thread=f, origin=Origin.INSIDE, kinds=[EventKind.RUN_COMPLETED, EventKind.RUN_FAILED])
              for f in forks],                                              # a fork's run ending, on its thread
        ).next()                                                            # waiting: no worker until something arrives
        results = await mine.get("results", {})
        if ev.thread == ctx.thread:                                         # the user steers the work
            for f in forks:
                if f.id not in results:
                    await f.send(ev.message.text)                           # a message on its thread
        else:                                                               # a fork is done (or broke without a word)
            fork = next(f for f in forks if f == ev.thread)
            results[fork.id] = None if ev.kind == EventKind.RUN_FAILED else (await fork.messages())[-1].data
            await mine.set("results", results)                              # read from the fork: nothing came back by itself
            if len(results) == len(forks):
                await ctx.send(self.summary(results))                      # the run completes
                return
        await ctx.restart()                                                 # same run, nothing to replay next time

    # a fork just works and talks on its own thread: `await ctx.send(DataPart(data=found))`
```

A simpler supervisor writes `async for ev in ctx.subscribe(...)`; rule 1 keeps that correct, fine for a few dozen wake-ups.

Steering: `thread.send(...)` is a message on that thread — the run there gets it in its inbox and reads it at its own pace (`await ctx.next(timeout=0)` between steps); a person can also talk to a fork directly, on its own thread. Runs report by talking on their own thread; the creator listens. Interrupting at once is a `signal` (when it waits for one) or `cancel` and a new message. A run's `input_required` is recorded on its own thread; the creator sees it (`thread.events(EventKind.RUN_INPUT_REQUIRED)`) and answers it (`thread.respond`) or asks the user itself and passes the answer on — answers are idempotent, the first one wins.

The backbone guarantees:

- **No missed wake-up:** parking a run and checking its cursor for unread matching events happen in one transaction.
- **A waiting run wakes only on what it waits for:** the subscription it is reading, its signal, its timer, or its children. Its other subscriptions keep collecting events (their cursors) without waking it — otherwise a run waiting for a signal is woken by every message in its inbox, finds nothing, parks, and is woken again, forever (found with the scenario tests).
- **"Unread" means unread by any of the run's subscriptions:** a message the run read through its own filter doesn't also count as unread for its inbox when it completes (which would start a needless new run).
- **No double wake-up:** several events at once queue the run once; the lease lets one worker run it; the events wait on the cursor.
- **Events while running** wait on the cursor; the next read gets them, without an extra wake-up.
- **Crash:** the lease expires, recovery picks the run up, and it continues from replay or its last `restart` state; `start` returns the same child handles, so no child starts twice.
- **A child ending after its parent was cancelled** is recorded and wakes nothing.

The policy stays in agent code: how the work is split, which run a user message is for, when to retry, when the job is done.

Common patterns (fan-out/fan-in, supervisor, a router) ship as optional helpers or examples, outside the core; shared conversations of several agents and people (rooms) are a layer on top.

## Kubernetes and API-created agents (2.x)

2.0 designs the contract so these land without further breaking changes. The agent file format becomes the CRD schema; standalone mode reads the same files.

Platform installed once per cluster (CRDs, operator, shared API and Compass):

```bash
helm install llamphouse llamphouse/llamphouse -f values.prod.yaml
kubectl apply -f hr-assistant.yaml
```

What the operator (kopf, Python) generates per definition:

```
serve mode:
  Deployment     hr-assistant-worker   cmd: llamphouse up --role worker --pool hr-assistant
  ServiceAccount hr-assistant          workload identity -> Entra Agent ID
  ConfigMap      hr-assistant-spec     resolved spec + platform config
  ScaledObject   hr-assistant          KEDA on queue depth
job mode:
  Job per run   sandbox runtime class, mounts via sidecar/CSI, TTL cleanup
```

Instances of one definition share its pods; each definition gets its own pods (own image, scaling, blast radius). Small definitions can opt into a shared pool. Owner references clean up on delete; status shows ready replicas and endpoint; a validating webhook rejects specs broader than the agent's identity allows.

API-created agents are instances of a code definition, like Foundry prompt agents backed by real code:

```bash
POST /v1/agents
{ "name": "hr-belgium",
  "definition": "hr-assistant@1.2",
  "config": { "instructions": "You help Belgian staff…", "index": "hr-be" } }
```

- Config validated against the published schema; policy check so an instance never gets broader access than its creator.
- Instances pin a definition (`hr-assistant@1.2`, which image to run); `PATCH` creates a new config revision.
- Origin label `managed-by: git | api`: Git-managed agents are read-only via the API, API-managed ones excluded from GitOps sync.
- Per-instance identity and mount scope, swapped per request in shared pods.
- Source of truth for API agents: a database (or Hopline registry), synced by the operator.

## Build order and open questions

Start with the core model and contract, because every other piece depends on it.

- [ ] Core model: `events`, `runs`, `sessions` tables, Alembic migration, rename run → run throughout
- [x] `llamphouse.sdk` module: parts, messages, events, the wire format, `Agent[Config]`, `RunContext` (output, thread, durability), capabilities (`Agents`, `Files`, `Scheduler`), `FakeRun` test kit
- [ ] Event log: thread events (positions) and run events (`seq`), idempotency keys, Redis live fan-out, `Last-Event-ID` resume, Postgres `stream_events: final | all | on_error`
- [ ] Subscriptions: durable, with cursors; inboxes and the new-run rule (one agent per thread); `ctx.post` items between an agent's runs
- [x] `Scenario` (in `llamphouse.sdk.testing`): an in-memory simulator of the runtime's behaviour, the executable spec the runtime's tests reuse
- [ ] Hub and spokes: two scope levels (hub, project), thread ownership checked by every adapter, per-project quotas
- [ ] Worker gateway: remote workers (outbound, per-project credentials), registration (pool, version, agents), the version window; `llamphouse apply`
- [ ] Threads a run creates: `ctx.fork` and `ctx.call` with `Thread` handles, delivery to a thread (inbox or a new run), subscriptions across a run's own thread and the threads it created, `session=` seeds, waiting supervisors with recorded deliveries and `ctx.restart`, structured lifetime (a run waits for the runs its messages started, cancel cascades, `detach`)
- [ ] Session backbone: store with CAS, generic operations, provider hooks (`before_run`, `prepare`, `on_event`, `after_run`), lifecycle events, rolling-window default provider (port the prototype on `feature/sessions`)
- [ ] Config split: agent files vs runtime file, registry entries for run queue, event queue, scheduler, session store, config store, file store
- [x] File store: interface, in-memory, contract tests (`llamphouse.runtime.files`); `FakeRun` and `Scenario` use it
- [x] Files service with the `file.*` system events; `FileUnavailable`; race tests for claim and clean-up
- [ ] System events: persisted (`system_events`), followed by external subscribers and Compass
- [ ] File store backends: local disk, Azure Blob, S3
- [ ] `llamphouse up --role api|worker|compass|scheduler|all`, each role built from config only
- [ ] Run queue interface with `pool` + contract tests (every pool runs as `shared` in 2.0)
- [ ] Scheduler interface + in-memory and Postgres + contract tests
- [ ] Responses + Conversations adapter; port A2A (to v1.0: card, bindings, push notifications, extensions) and AG-UI to the new model; drop Assistants
- [ ] Inbound `auth` component (published as A2A security schemes), external subscribers (webhook outbox with retries and dead letters), `remote_agents`
- [ ] Workflows: `Workflow` with `AgentType`, Input/Output validation and schemas, `POST /v1/workflows/{name}/runs`, replay-mismatch detection
- [ ] Compass: thread timeline (conversation and detailed views), run view, session revisions; workflow list, step timeline, start form from the Input schema
- [x] First runtime: API and worker processes on Postgres + Redis, trees as the unit of a transaction, leases and a sweeper, HTTP API `/v2` with SSE, basic Compass, `python -m llamphouse.runtime` (see "The first runtime")
- [ ] Packaging: base install that runs `llamphouse up` in memory, backend extras, lazy imports, `llamphouse.sdk` import guard
- [ ] Migration: `llamphouse migrate` (config and the whole database, re-runnable, `--dry-run`), guide, rewritten examples, `2.0.0a` pre-release

After 2.0.0 (additive): stream per pool, Redis and Service Bus schedulers, official base image, pattern helpers, MCP exposure, Kubernetes operator, API-created agents.

Open questions:

- Keep the Assistants adapter as a legacy extra, or drop it now that OpenAI has shut the API down? -> drop it
- Rename the project for 2.0, or keep llamphouse? -> keep llamphouse
- Default visibility for spawned runs: own events only, or the thread's history? -> own events + the input from the parent
- Event order across concurrent writers? -> per-thread position, taken in the append transaction
- A user message while a run is active? -> the core starts nothing by itself and allows one active run per thread; while a run works, the thread takes no message and no run — only messages sent to the run (on the thread when sent, in its session when read). Steer, queue, interrupt, answer, reject: the app's choice, in front of the agent (see "Starting runs and the inbox")
- What is on the thread? -> messages (with their tool calls, results and artifacts as parts), run and session lifecycle, agent items; steps and chunks are run events
- Persisting chunks? -> chunks get no thread position; the Postgres store has `stream_events: final | all | on_error`
- Where do artifact contents live? -> an artifact is an `ArtifactPart` in a message; file contents go through the `Files` capability to a `file_store` in the runtime file (in-memory in dev; local disk, Azure Blob, S3)
- Does a child run get its own thread? -> yes, always: everything is a thread with one agent. A fork is a new thread of this agent from here (`ctx.fork`), another agent gets a new thread of its own (`ctx.call`); runs talk to threads, never to runs; nothing comes back by itself — the creator listens and reads (see "Threads, forks and state"). No `SAME` (parallel runs inside the conversation) and no continuing another agent's thread
- How does a supervising run wait for its children and the user? -> one run that waits on its inbox (results are addressed to it) or a subscription (no worker held) and wakes on matching events; deliveries are recorded steps; `ctx.restart()` bounds replay, `ctx.run.state` carries over; it can't finish before its children
- How do forks and other agents deal with sessions? -> their own session, never shared, starting from `session=`: a copy (`ctx.session`, a fork's default — the parent's messages its own), a `SessionView` or messages the agent computed, or `None` (another agent's default)
- Magic strings for options (`"parent"`, `"fork"`)? -> never: enums for closed sets, StrEnums for open ones, typed references (`RunRef`, `ThreadRef`) for addressing
- How does a run send a message? -> `ctx.send`, by default on its own thread (a child's lands where its parent sees it); `to=` narrows it to one run; `reply` = `send` without `to`; siblings coordinate through the parent or posts
- Where do workflows fit? -> the run is the workflow; `Workflow` is an `Agent` subclass with `type = AgentType.WORKFLOW`, typed Input/Output, no conversation by default, its own Compass views; declarative workflows and graph engines stay outside the core
- Are we future-proof against A2A? -> checked against v1.0: added `rejected` and `auth_required` run states (`ctx.reject`, `ctx.request_auth`), skills, input/output modes and extensions (`Agent.describe`), `ctx.request` (accepted media types, active extensions, metadata), `Message.references`, an `Artifact` type with streaming, `ThreadEvent.at`; in the runtime: inbound `auth`, external subscribers (push notifications), `remote_agents`. The rest is the adapter's
- Message roles? -> no stored role: a message records its `Sender` (person, agent, system, service); the role is derived per reader (`ctx.session.is_own`), because on a shared thread it depends on who reads; the default inbox is messages from outside the run's tree (`Origin.OUTSIDE`), not "user messages"
- `requires` in YAML? -> no: what the code needs stays in code (published by `describe()`); what an instance may use is `permissions` in the Agent file
- `ctx.capability(Agents)`? -> `ctx.agents`, always there (child runs are backbone); `capability()` is for components that may be absent (Files, Scheduler)
- Shared state between agents? -> none between agents (messages); within one agent: `ctx.state` on the thread, with versions, `update`, `if_version` and `batch()`, every write an event (snapshots at any position, forked with the thread); `ctx.run.state` for one run; see "Threads, forks and state", "Where state lives"
- Merging a fork back? -> no merge method: a fork talks on its own thread; the parent listens and reads the fork's messages, thread and state (`fork.messages()`, `fork.thread_events()`, `fork.state`), and copies over what it wants. A fork never writes on its parent's thread, also not after the parent ended
- Snapshots, `commit()`? -> state writes are thread events, so the state at any position (a run's start, a fork) is there for free; writes are live, no `commit()`; several keys at once is `batch()`
- Skills, media type modes, extensions in the agent's code? -> no: protocol presentation is the agent's `publish.a2a` settings in its Agent file, overridable per adapter instance; the code publishes only type, description, version, requires and schemas
- Several A2A adapters with different settings? -> yes: adapters are named instances in the runtime file, each with its endpoint, agents, auth, bindings and card settings
- Can people add their own parts? -> yes: a `Part` subclass with an `x-` kind, registered with `@part` so the wire rebuilds it; unregistered, it travels as an `UnknownPart`; adapters fall back to a data part with a vendor media type
- `ctx.input`, `input_data`, `history()`, `session`? -> two things: `ctx.input` (one `Message`, always: what started the run; a webhook or schedule is its sender, `SenderKind.TRIGGER`; a workflow's validated Input is `input.data`) and `await ctx.session.messages()` (its working memory, a `SessionView`); the record of its own thread with `ctx.thread.messages()`, and of the forks it created
- Artifacts as separate entities? -> no: an `ArtifactPart` (name, description, parts) in a message — the minimum to round-trip A2A artifacts; versions, an artifact API and sharing rules are app-level
- Uploaded files: artifacts or data? -> neither: a `FilePart` in the sender's message, by reference; artifacts are what a run produces
- Raw bytes in the messages table? -> never: the runtime stores them in the file store before writing the message and keeps a reference (uri, size, sha256); files belong to their thread, unreferenced uploads expire
- Who builds the artifact UI and sharing? -> the application; llamphouse enforces access at its boundary, serves signed downloads and uploads, and offers a read-only artifacts view
- Does the file store send events? -> the backends don't; the files service emits `file.stored`, `file.claimed`, `file.deleted` as system events (a third kind, next to thread and run events); quarantine is a custom store's business, signalled with `FileUnavailable`
- File or artifact locking? -> none: both never change (a new version is a new file, a new message); claim and clean-up are atomic; "one editor at a time" is app coordination via `ctx.state`
- How to declare requirements? -> `@requires(Files, Scheduler)` on the class: inherited, and a subclass's adds to its parent's; `ctx.require(X)` returns it without a `None` check; not by inheriting from `Files` (that would say the agent *is* a file store)
- Agent names and IDs? -> the instance's `metadata.name` (immutable slug, the reference everywhere) plus a runtime-assigned `uid` (on every run) and an optional `display_name`; code sees `ctx.me`
- `ctx.agent`, `ctx.run_id`, `ctx.identity`...? -> `ctx.me` (who I am: name, type, display name, version, revision - the same for workflows), `ctx.run` (with `parent`, `caller`, `depth`, its own `state`), `ctx.thread`; messages go to threads, never to runs
- Agent access to the full run and thread? -> the run's useful facts are on `ctx.run`; the record of its own thread with `ctx.thread.messages()` (recorded for replay) and of its forks; its working memory with `ctx.session.messages()`
- Session revisions on read? -> never: reading is cheap; a revision is a background job started by the provider's policy as things are added, or on request (`ctx.session.revise(**options)`); a run waits for one only if it asks (`settled()`)
- Tool calls on the thread? -> in the session (`ctx.session.add`), not on the thread: working material, not conversation; the provider decides how they show; live UIs get them from the run's stream
- Agents in different scopes, workers elsewhere? -> hub and spokes within one customer: the hub runs the API, Compass, scheduler, adapters and every store; each spoke project runs workers for its own pool, connecting outbound through a worker gateway with per-project credentials; scopes are the hub and the project; threads belong to their project and creator; calls between spokes go through the hub
- Several agents on one thread? -> no: one agent per thread, any number of people and callers; parallel runs of the same agent; other agents are callers and callees on their own threads; shared multi-agent conversations (rooms) are a layer on top
- Keep "task" or "run"? -> run (`RunContext`, `run_events`, `run_queue`)
- `AgentTemplate` or `AgentDefinition`? -> `AgentDefinition`, as in 1.x
- How to model waiting on a timer or signal? -> a `waiting` state next to `input_required`
- `mode: job` outside Kubernetes? -> a subprocess per run
- What is in 2.0.0 itself? -> the core model, SDK, config split, roles, adapters, sessions, Compass timeline, packaging extras, migration; per-pool streams, the Redis/Service Bus schedulers and the base image follow in 2.x
- What does `llamphouse migrate` carry over? -> everything: config, threads, messages, runs as runs, run steps as events
- A separate `llamphouse-sdk` package? -> no, `llamphouse.sdk` inside llamphouse for now (`llamphouse.sdk` imports only pydantic, guarded by a test; the base install still runs `llamphouse up`)
- Retention defaults for fine-grained events (7 or 30 days)? -> not part of llamphouse, except an option for in memory stores; whether deltas are persisted at all is up to each data store
- Sandboxes and mounts: build in llamphouse, or delegate to Hopline as the sandbox provider? -> keep out of llamphouse for now
