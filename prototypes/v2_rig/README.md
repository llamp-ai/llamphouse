# llamphouse 2.0 test rig

A small model of the 2.0 core, to test the design in
[`docs/llamphouse 2.0 — design notes.md`](../../docs/llamphouse%202.0%20—%20design%20notes.md)
before building it. Not llamphouse, and not imported by it.

- `rig.py`: threads with per-thread positions, runs, durable subscriptions
  with cursors, child runs (`Agents.start` / `ChildRun`), sessions, leases,
  replay by call key, `restart(state)`. In memory: one asyncio lock stands in
  for a database transaction, workers are asyncio tasks, time is a fake clock
  (`Runtime.advance`), and `Crash` simulates a worker process dying.
- `test_hypotheses.py`: one test (or a few) per hypothesis.
- `test_pg_ordering.py`: the ordering hypothesis against a real Postgres.

## Run it

```sh
cd prototypes/v2_rig
../../.venv/bin/python -m pytest -q -c /dev/null --rootdir=. test_hypotheses.py

# with Postgres
docker run -d --rm --name llamp-rig-pg -e POSTGRES_PASSWORD=rig -p 55432:5432 postgres:16-alpine
LLAMP_RIG_PG_DSN=postgresql://postgres:rig@127.0.0.1:55432/postgres \
  ../../.venv/bin/python -m pytest -q -s -c /dev/null --rootdir=. test_pg_ordering.py
docker stop llamp-rig-pg
```

## Hypotheses and results

| | Hypothesis | Result |
|---|---|---|
| H1 | A global `bigserial` is not commit order; a per-thread position taken in the append transaction is gapless and in commit order | **Holds.** Postgres, 20 writers, 400 events: the `bigserial` cursor reader missed 198–237 of them (every run); the per-thread position missed none. Taking the position first holds the thread lock for the whole transaction (1.5–1.6 s); **taken as the last statement** it is 0.42–0.56 s |
| H2 | Events with an idempotency key are written once, also when a run replays after a crash | Holds |
| H3 | Chunks are run events (`seq` per run) and take no thread position | Holds: 400 chunks, fewer than 10 thread positions |
| H4 | A waiting run holds no worker | Holds: a supervisor and three children finish on **one** worker |
| H5 | A supervisor is one run that waits and wakes on a child's post, a user message and a child finishing; a steering message starts no run | Holds |
| H6 | Checking for unread events and parking must be one transaction | Holds: with the check and the park split, an event arriving in between leaves the run `waiting` forever |
| H7 | A burst of events never activates a run on two workers at once | Holds (10 events, 4 workers) |
| H8 | Recorded deliveries make replay deterministic; completed steps don't run again | Holds: a crash after the first of three events replays the same first event; each effect ran once |
| H9 | Replay per wake-up grows without `restart`; with it, it stays flat | Holds: 30 wake-ups — without, the last replays 58 steps (870 in total, ≈ n²); with `restart`, 0. The cost: each restart re-queues the run only for it to park again (60 activations instead of 31) — see below |
| H10 | A run can't finish before its children; cancelling cascades, except to detached children; a child ending after its parent was cancelled wakes nothing | Holds |
| H11 | `start` creates the child and records the parent's step in one transaction, so a crash never duplicates a child | Holds |
| H12 | A user message with no subscriber starts a run; one unread when a run completes starts a new run | Holds |
| H13 | A child's question: parent and client may both answer, the first wins | Holds, both orders |
| H14 | Child sessions: `ChildSession.NEW` holds only the input; `AGENT` remembers across delegations; `FORK` copies the parent's and never changes it; the parent's session holds what its children send (not their stream) | Holds, after one fix (below) |
| H15 | Child threads `ChildThread.SAME` / `NEW` / `FORK`: the parent gets the result in each; a fork reads the parent's thread up to the fork point | Holds, after one rule (below) |
| H16 | `ctx.send` goes on the run's own thread by default, where a parent sees its child's messages and wakes on them mid-run; `to=ctx.parent` reaches the parent's thread also from a child on a new thread | Holds |

Options are enums (`ChildThread`, `ChildSession`, `Visibility`, `Escalation`) and runs are addressed by `RunRef` (`ctx.me`, `ctx.parent`, a `ChildRun`), as in the SDK: no magic strings.

All of `test_hypotheses.py` passed 30 runs out of 30 (26 tests; earlier, 24 tests 50 out of 50) (real asyncio interleaving, 4 workers in some tests).

## What the rig changed in the design

Things the design didn't say, found by building and testing it (now in the doc):

1. **Take the thread position last in the transaction.** The row lock lasts until commit.
2. **A subscription delivers from the start of the run**, not from the `subscribe` call: otherwise a child that finishes before its parent subscribes is missed.
3. **"Unread at completion" means unread by any of the run's subscriptions.** A supervisor that reads user messages through its own filter would otherwise start a needless new run when it completes (found while writing H5).
4. **A fork brings the parent's session up to date before copying it**, or the child gets stale working memory (H14 caught this).
5. **A child on another thread records its end on the parent's thread too**, so the parent's subscriptions see it (H15).
6. **`start` is atomic with the parent's step record** (H11).

Open: `restart` costs an extra activation per iteration (H9). The runtime
could park a restarted run directly when nothing is unread, instead of
queueing it, if the restarted code is known to wait on the same
subscriptions — worth deciding when building it.

Known and unchanged: a step whose function finished but whose result wasn't recorded before a crash runs again (at least once), as in 1.4.

## Not covered

Leases being renewed while a run is long-running, the scheduler, the
session store's compare-and-swap, permissions, Redis live fan-out, and
real multi-process workers. The in-memory transaction is a single lock,
so the rig shows the *logic* is right; H1 is the only one tested against a
database's actual concurrency.
