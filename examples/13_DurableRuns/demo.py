"""The refund agent through three processes and two crashes, in one script. Run: python demo.py

Each `Scenario` is the llamphouse engine - the same code that serves agents - in memory, on a fake clock.
Closing one and opening another on the same store is a process stopping and a new one starting; `crash()`
kills the worker at a point in the run, and `recover()` lets its lease expire, as the sweeper would.
The store here is in memory (`MemoryStore`, `fresh()` for the next process); `PostgresStore(dsn)` works the
same way.
"""

import asyncio

from llamphouse.runtime.data_store import MemoryStore
from llamphouse.sdk import Sender
from llamphouse.sdk.testing import Scenario

from agents import EFFECTS, RefundAgent, RefundConfig

AGENTS = {"refunds": RefundAgent}
CUSTOMER = Sender.person("ana", name="Ana")
CLERK = Sender.person("sam", name="Sam")
CFG = RefundConfig()


def show(sc: Scenario, thread, seen: int) -> int:
    """What's new on the thread, and where the run is."""
    messages = sc.messages(thread)
    for who, text in messages[seen:]:
        print(f"  {who}: {text}")
    (run,) = sc.runs(thread)
    print(f"  [run {run.id}: {run.status}, executed {run.activations}x]")
    return len(messages)


async def main() -> None:
    store = MemoryStore()  # the data that outlives each process

    print("process 1: the request arrives; the worker dies right after the draft is recorded")
    async with Scenario(AGENTS, store=store) as sc:
        sc.crash(agent="refunds", after_step="draft_refund")
        thread = await sc.thread("refunds")
        await sc.say(thread, "My lamp from order A-1042 arrived broken, I'd like a refund.", by=CUSTOMER)
        await sc.settle()
        seen = show(sc, thread, 0)
        print(f"  its lease expires; the sweeper queues it again: {await sc.recover()} run")
        await sc.settle()
        seen = show(sc, thread, seen)
        (run,) = sc.runs(thread)
        print(f"  asks: {run.pending_request.reason}")

    print("\nprocess 2: a person approves; the run sleeps through the cool-off")
    async with Scenario(AGENTS, store=store.fresh()) as sc:
        (run,) = sc.runs(thread)
        await sc.answer(run.ref, {"approved": True, "note": "photo checks out"}, by=CLERK)
        await sc.settle()
        seen = show(sc, thread, seen)

    print(f"\nprocess 3: {CFG.cool_off_seconds:g} s later; the first payment call times out")
    async with Scenario(AGENTS, store=store.fresh()) as sc:
        sc.advance(CFG.cool_off_seconds)
        await sc.settle()
        seen = show(sc, thread, seen)
        print(f"  {CFG.backoff_seconds:g} s later the second call pays - and the worker dies before the result "
              "is recorded")
        sc.crash(agent="refunds", before_record="try_payment")
        sc.advance(CFG.backoff_seconds)
        await sc.settle()
        seen = show(sc, thread, seen)
        print(f"  recovered: {await sc.recover()} run; it calls again with the same idempotency key")
        await sc.settle()
        seen = show(sc, thread, seen)
        (run,) = sc.runs(thread)
        print(f"  output: {run.output!r}")

    print(f"\nwhat really happened: {dict(EFFECTS)}")
    assert EFFECTS["drafts"] == 1 and EFFECTS["refunds paid"] == 1


if __name__ == "__main__":
    asyncio.run(main())
