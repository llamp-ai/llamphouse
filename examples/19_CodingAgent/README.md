# Coding agent

A long-running coding agent that splits its work into subtasks running in parallel, each on a **fork of the
thread** (`ctx.fork`): a new thread of the coder, starting from this one. To a subtask its fork is a thread
like any other: it has a session (a copy of its parent's), state, an inbox; it can wait, crash and recover,
be talked to, and fork subtasks of its own. It talks only on its own thread; the parent listens, and reads
what it wants from the fork's thread and state. Nothing comes back by itself.

No model: `think` stands in for one, so it runs offline and the same way every time. The working copies are
the application's (`WORKTREES`, one per fork, merged back as subtasks finish); llamphouse only carries a
reference to them, in the thread's state.

## Run it

The scripted demo runs on the in-memory `Scenario` runtime (one process, a fake clock, crashes injected):

```bash
python demo.py
```

The demo: a request is split into three subtasks (one splits again), you talk to one subtask directly on its
fork, the worker running it dies and the run recovers, and the subtasks finish. A second request then finds
the session revised in the background, and its subtasks start from it. At the end it prints the thread
tree, the sessions and the working copy.

Served, on the real clock (each subtask builds for 60 seconds):

```bash
llamphouse up --memory --agents agents.yaml     # http://127.0.0.1:8000/compass
curl -s localhost:8000/v2/threads -H 'content-type: application/json' -d '{"agent": "coder"}'
curl -s localhost:8000/v2/threads/<thread id>/runs -H 'content-type: application/json' \
     -d '{"text": "Add dark mode to the settings page"}'
```

In Compass, open the thread: its **Tree** shows the forks (and the forks of `tests`), each a thread with its
own runs; the main run waits on them (**Waiting on**), and a fork's run shows its `edit` step (**Steps**)
and the tool call and result in its **Session** tab. Send a message to a fork's thread while it builds to steer it. The
`agents.yaml` gives the coder its session provider: `Summarizing`, with `summarize` from `agents.py`.

## What to look for

| mechanism | where |
|---|---|
| A subtask is a fork: a new thread of this agent, from here; a run starts there | `ctx.fork(DataPart(data={"task": part}))` |
| The fork carries on from the parent's history: what the parent said is its own | the forked sessions in the demo's output (`own` before the fork) |
| The fork starts with the parent's thread state; its writes stay in the fork | `ctx.state.set("worktree", ...)` before forking, read in `do` |
| The parent waits on you and on its forks at once, and notices a fork that failed | `ctx.next(*forks)` |
| Steering: a message on a fork's thread lands in its run's inbox | `f["thread"].send(...)`; `sc.say(ThreadRef(fork), ...)` in `demo.py` |
| Nothing comes back by itself: the parent reads the fork's messages and state, and copies what it wants | `fork.messages()`, `fork.state.get("changed")` → `ctx.state.update("changed", ...)` |
| A long main loop with short replays: its forks come back from the runtime, what each is for from its run state | `ctx.forks`, `ctx.run.state` + `ctx.restart()` (see `replays` in the tree) |
| Forks of forks | `tests` splits into `unit tests` and `e2e tests` |
| A crash in a fork: recovered like any run, its edit made once | `sc.crash(...)`, `sc.recover()`, `ctx.step("edit ...")` |
| Revisions: the session provider's job, in the background, as the session grows; forks start from the session as it is | `Scenario(..., sessions=Summarizing(summarize, max_messages=5, keep=3))` |
| Tool calls in working memory, not on the thread: the provider decides how they show | `ctx.session.add(ToolCallPart(...))`, `ctx.session.add(ToolResultPart(...))` in `do` |
| The application's side, done once per subtask | `ctx.step("merge ...", merge, ...)` |
