"""A long-running coding agent - forks of the thread as subtasks, the 2.0 SDK end to end.

One agent, `coder`. A request from you starts its main run: it plans, forks
itself once per part of the plan (`ctx.fork`: a new thread of the coder,
starting from this one), and listens to its forks while they work - and to
you. To a subtask its fork is a thread like any other: it has the session it
was forked with, its own state, it can wait, crash and recover, be talked to,
and fork subtasks of its own. It talks only on its own thread; the parent
listens, and reads what it wants from the fork's thread and state.

No model: `think` stands in for one, so the example runs offline and the same
way every time. The working copies (`WORKTREES`) are the application's:
llamphouse only carries a reference to them, in the thread's state.
"""

from __future__ import annotations

from llamphouse.sdk import Agent, DataPart, EventKind, Message, NoActiveRun, TextPart, ToolCallPart, ToolResultPart

# ── the application's side: working copies, one per fork ───────────────────

WORKTREES: dict[str, dict[str, str]] = {}  # thread id -> {path: content}
EDITS: list[str] = []  # every edit made, to show a recovered run doesn't edit twice


def worktree(thread_id: str, base: str | None) -> dict[str, str]:
    """A thread's working copy; a fork's starts as a copy of its parent's (a git worktree, in real life)."""
    if thread_id not in WORKTREES:
        WORKTREES[thread_id] = dict(WORKTREES[base]) if base else {"app.css": "body {}", "settings.tsx": "<Settings/>"}
    return WORKTREES[thread_id]


async def edit(thread_id: str, task: str, note: str) -> str:
    path = "app.css" if "style" in task else "settings.test.tsx" if "test" in task else "settings.tsx"
    files = WORKTREES[thread_id]
    files[path] = files.get(path, "") + f"\n/* {task}{note} */"
    EDITS.append(task)
    return path


async def merge(source: str, target: str) -> list[str]:
    """Bring a finished fork's changes into its parent's working copy (a merge, in real life)."""
    changed = []
    for path, content in WORKTREES[source].items():
        lines = WORKTREES[target].get(path, "").split("\n") if path in WORKTREES[target] else []
        new = [line for line in content.split("\n") if line and line not in lines]
        if new:
            WORKTREES[target][path] = "\n".join(lines + new)
            changed.append(path)
    return changed


# ── a stand-in for a model ──────────────────────────────────────────────────

def think(task: str) -> list[str]:
    """Which parts a task splits into; [] when it's small enough to just do."""
    if "dark mode" in task.lower():
        return ["styles", "settings toggle", "tests"]
    if "high contrast" in task.lower():
        return ["contrast styles", "tests"]
    return ["unit tests", "e2e tests"] if task == "tests" else []


def summarize(older: list[Message]) -> str:
    """What a model would write for the session provider (`Summarizing`): here, how many it covers."""
    return f"(summary of {len(older)} earlier messages)"


def outline(done: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {part}: " + text.replace("\n", "\n  ") for part, text in done)


# ── the agent ───────────────────────────────────────────────────────────────

class Coder(Agent):
    """Writes code: splits big tasks into subtasks that run in parallel on forks of the thread."""

    async def run(self, ctx) -> dict[str, str] | str:  # what each part came to; a file path for a small task
        task = ctx.input.data["task"] if isinstance(ctx.input.data, dict) else ctx.input.text
        mine = ctx.run.state  # this run's own: kept across restarts, gone when it ends
        if not ctx.forks:  # none yet: the first time through - plan, then fork or just do it
            parts = think(task)
            if not parts:
                return await self.do(ctx, task)
            await ctx.send(f"Plan: {', '.join(parts)} - in parallel.")  # before forking: the forks know the plan
            worktree(ctx.thread.id, base=await ctx.state.get("worktree"))
            await ctx.state.set("worktree", ctx.thread.id)  # thread state: every fork starts with it set
            for part in parts:
                await ctx.fork(DataPart(data={"task": part}))
            await mine.set("parts", parts)  # what each fork is for, in the order they were created
            await ctx.restart()  # the loop starts; replay stays a few steps, however long it runs

        # one iteration of the main loop: whatever comes first - a fork's run ending, or you
        forks = [{"thread": t, "part": p} for t, p in zip(ctx.forks, await mine.get("parts"))]
        ev = await ctx.next(*[f["thread"] for f in forks])  # whatever comes first: you, or news from a fork
        done = dict(await mine.get("done", []))
        open_ = [f for f in forks if f["part"] not in done]
        if ev.thread == ctx.thread:  # you: steer every subtask still working - a message on its thread
            for f in open_:
                try:
                    await f["thread"].send(ev.message.text)  # to the run working there: its inbox
                except NoActiveRun:
                    pass  # it has just finished: its news is on its way
            await ctx.send(f'Passing "{ev.message.text}" on to: {", ".join(f["part"] for f in open_)}.')
        elif ev.kind in (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED):  # a fork's run ended
            fork = next(f for f in forks if f["thread"] == ev.thread)
            if ev.kind == EventKind.RUN_FAILED:
                done[fork["part"]] = f"failed: {ev.data['error']}"
            else:  # nothing comes back by itself: read the fork, copy over what's wanted
                await ctx.step(f"merge {fork['part']}", merge, fork["thread"].id, ctx.thread.id)  # durable: once
                changed = await fork["thread"].state.get("changed", [])  # from the fork's state...
                await ctx.state.update("changed", lambda c: sorted(set(c) | set(changed)), [])  # ...into this one
                done[fork["part"]] = (await fork["thread"].messages())[-1].text  # what it said last, there
        await mine.set("done", list(done.items()))
        if len(done) == len(forks):
            await ctx.send(f"Done.\n{outline([(f['part'], done[f['part']]) for f in forks])}")
            return done
        await ctx.restart()

    async def do(self, ctx, task: str) -> str:
        """A task small enough to do: edit this thread's working copy."""
        worktree(ctx.thread.id, base=await ctx.state.get("worktree"))  # a copy of its parent's
        await ctx.sleep(60)  # build and run: waits on the clock, holding no worker
        steer = await ctx.next(timeout=0)  # did anyone say something here meanwhile?
        note = f" ({steer.message.text})" if steer is not None else ""
        await ctx.session.add(ToolCallPart(call_id=task, name="edit", arguments={"task": task}))  # working memory,
        path = await ctx.step(f"edit {task}", edit, ctx.thread.id, task, note)  # durable: never edited twice
        await ctx.session.add(ToolResultPart(call_id=task, output=[TextPart(text=path)]))  # not on the thread
        await ctx.state.set("changed", [path])  # its own state: the parent may read it
        await ctx.send(f"edited {path}{note}")  # on its own thread: the parent listens
        return path
