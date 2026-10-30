"""A scripted session with the coding agent. Run: python demo.py"""

import asyncio

from llamphouse.sdk import RunStatus, Sender, SenderKind, ThreadRef
from llamphouse.sdk.testing import Scenario

from agents import EDITS, WORKTREES, Coder, summarize
from llamphouse.runtime.sessions import Summarizing

YOU = Sender.person("you", name="You")
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def task_of(rec) -> str:
    data = rec.input.data
    return data["task"] if isinstance(data, dict) else rec.input.text


def tree(sc: Scenario, thread: ThreadRef, at: int = 0, depth: int = 0) -> None:
    """The thread and its forks, each with its runs and what was said on it after the fork."""
    pad = "  " * depth
    runs = sc.runs(thread)
    label = "main thread" if depth == 0 else f"fork from position {at}: {task_of(runs[0])}"
    print(f"{pad}{BOLD}{thread.id}{RESET} {DIM}({label}){RESET}")
    for r in runs:
        print(f"{pad}  {DIM}{r.id} {r.status}, activations {r.activations}, replays {r.replays}{RESET}")
    for e in sc.timeline(thread):
        if e.position > at and e.message is not None:
            to = f" → {e.to.id}" if e.to else ""
            text = e.message.text or e.message.data
            print(f"{pad}  {DIM}{e.position:>3}{RESET} {e.message.sender.id}{to}: {str(text).splitlines()[0]}")
    for fork, pos in sc.forks(thread):
        tree(sc, fork, pos, depth + 1)


async def main() -> None:
    async with Scenario({"coder": Coder}, workers=4, sessions=Summarizing(summarize, max_messages=5, keep=3)) as sc:
        thread = await sc.thread("coder")

        print(f"{BOLD}you{RESET}: Add dark mode to the settings page")
        await sc.say(thread, "Add dark mode to the settings page", by=YOU)
        await sc.settle()
        print(f"{BOLD}coder{RESET}: {sc.messages(thread)[-1][1]}")

        # a fork is a thread like any other: talk to the styles subtask on its own thread
        styles = next(r for r in sc.runs() if task_of(r) == "styles")
        print(f"\n{BOLD}you{RESET} (on {styles.thread_id}, the styles fork): use CSS variables")
        await sc.say(ThreadRef(styles.thread_id), "use CSS variables", by=YOU)

        # the worker running the styles subtask dies right after its edit is recorded
        sc.crash(agent="coder", after_step="edit styles")
        print(f"{DIM}   (60 seconds pass; the worker running the styles subtask dies){RESET}")
        sc.advance(60)
        await sc.settle()
        print(f"{DIM}   styles subtask: {styles.status} - its lease holds it until recovery{RESET}")
        print(f"{DIM}   (recovery: the lease expires, the run is picked up again){RESET}")
        await sc.recover()
        await sc.settle()

        main_run = sc.runs(thread)[0]
        assert main_run.status is RunStatus.COMPLETED
        print(f"\n{BOLD}coder{RESET}: {sc.messages(thread)[-1][1]}")

        # the next turn: the session has grown, so the provider revises it in the background
        print(f"\n{BOLD}you{RESET}: Also add a high contrast mode")
        await sc.say(thread, "Also add a high contrast mode", by=YOU)
        await sc.settle()
        sc.advance(60)
        await sc.settle()
        second = sc.runs(thread)[1]
        assert second.status is RunStatus.COMPLETED
        print(f"{BOLD}coder{RESET}: {sc.messages(thread)[-1][1]}\n")

        print(f"{BOLD}the thread tree{RESET}")
        tree(sc, thread)

        print(f"\n{BOLD}sessions{RESET} {DIM}(what the model would see: the thread's messages and the run's own "
              f"entries; revised in the background above 5){RESET}")
        contrast = next(r for r in sc.runs() if task_of(r) == "contrast styles")
        for r in [second, styles, contrast]:
            s = sc.session(r.ref)
            own = lambda m: "own" if m.id in s.inherited_own or (m.sender.run in s.runs) else (
                "system" if m.sender.kind == SenderKind.SYSTEM else "other")
            print(f"  {r.id} ({task_of(r)}), {s.revisions} revisions" +
                  (f", forked from {s.forked_from}" if s.forked_from else ""))
            for m in s.messages:
                part = m.parts[0]
                shown = (f"[tool call: {part.name}({part.arguments})]" if part.kind == "tool_call" else
                         f"[tool result: {part.output[0].text}]" if part.kind == "tool_result" else
                         str(m.text or m.data).splitlines()[0])
                print(f"    {DIM}{own(m):<6}{RESET} {shown}")

        changed = sc.state_of(thread)["changed"][0]
        print(f"\n{BOLD}changed{RESET}: {changed}  {DIM}(read from each fork's state, copied up the tree){RESET}")
        print(f"{BOLD}edits{RESET}: {EDITS}  {DIM}(each once, the crash included){RESET}")
        print(f"{BOLD}the main working copy{RESET} {DIM}(the application's; every fork merged into it){RESET}")
        for path, content in WORKTREES[thread.id].items():
            print(f"  {path}: {' '.join(content.splitlines())}")


if __name__ == "__main__":
    asyncio.run(main())
