"""Printing what happened - shared by the demo and the chat."""

from __future__ import annotations

from llamphouse.sdk import EventKind, RunStatus, SenderKind, ThreadRef
from llamphouse.sdk.testing import Scenario

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def brief(message) -> str:
    """One line: the text, else the data, else an artifact's data."""
    if message.text:
        return message.text.replace("\n", " / ")
    data = message.data if message.data is not None else next((a.data for a in message.artifacts), None)
    return str(data)


class Printer:
    """Prints what's new, in the order it happened: on the desk's thread (its replies) and on the threads
    it created (its forks, the weather agent's): progress, what the desk sends there, what is said there.
    And a question waiting for you."""

    def __init__(self, sc: Scenario, thread: ThreadRef):
        self.sc, self.thread = sc, thread
        self._seen = 0

    def _threads(self, thread: ThreadRef) -> set[ThreadRef]:
        out = {thread}
        for t in self.sc.created(thread):
            out |= self._threads(t)
        return out

    def _name(self, thread: ThreadRef) -> str:
        runs = self.sc.runs(thread)
        data = runs[0].input.data if runs else None
        if isinstance(data, dict) and "city" in data:
            return f"{data['city']} fork"
        return runs[0].agent if runs else thread.id

    def new(self) -> None:
        log = self.sc.log()
        ours = self._threads(self.thread)
        for e in log[self._seen:]:
            if e.thread not in ours:
                continue
            if e.kind == "x-desk.progress":
                print(f"{DIM}   … {e.data['label']}{RESET}")
            elif e.kind != EventKind.MESSAGE or e.message.sender.kind == SenderKind.PERSON:
                continue
            elif e.thread == self.thread:  # the desk, to you
                text = e.message.text or ""
                for a in e.message.artifacts:
                    files = ", ".join(p.name for p in a.parts if getattr(p, "name", None))
                    text += f"\n{DIM}   [artifact {a.name}: {a.description}{' - ' + files if files else ''}]{RESET}"
                print(f"{BOLD}{e.message.sender.id}{RESET}: {text}")
            elif self.sc.run(e.run).thread_id == e.thread.id:  # said there, by the run working on it
                print(f"{DIM}   {self._name(e.thread)}: {brief(e.message)}{RESET}")
            elif not any(r.input.id == e.message.id for r in self.sc.runs(e.thread)):  # sent there, not a first message
                print(f"{DIM}   {e.agent} → {self._name(e.thread)}: {brief(e.message)}{RESET}")
        self._seen = len(log)
        if (asking := self.pending()) is not None:
            print(f"{BOLD}desk asks{RESET}: {asking.pending_request.reason}")

    def pending(self):
        """The desk's run waiting for your answer, if any."""
        return next((r for r in self.sc.runs(self.thread) if r.status is RunStatus.INPUT_REQUIRED), None)

    def runs(self) -> None:
        print(f"{DIM}runs:{RESET}")
        for r in self.sc.runs():
            parent = f" ← {r.parent.id}" if r.parent else ""
            t = ThreadRef(r.thread_id)
            where = ("the desk's thread" if t == self.thread
                     else f"a fork from position {self.sc.threads[t.id].forked_from[1]}" if self.sc.threads[t.id].forked_from
                     else f"a new thread of {r.agent}")
            print(f"{DIM}  {r.id:<7} {r.agent:<8} {r.status:<10} on {where}{parent}{RESET}")
