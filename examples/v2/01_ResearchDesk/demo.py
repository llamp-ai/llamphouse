"""A scripted conversation with the research desk. Run: python demo.py"""

import asyncio

from llamphouse.sdk import Sender
from llamphouse.sdk.testing import Scenario

from agents import AGENTS
from show import BOLD, DIM, RESET, Printer

YOU = Sender.person("you", name="You")


async def main() -> None:
    async with Scenario(AGENTS, workers=4) as sc:  # the in-memory runtime: one process, a fake clock
        thread = await sc.thread("desk")
        out = Printer(sc, thread)

        async def say(text: str) -> None:
            print(f"\n{BOLD}you{RESET}: {text}")
            await sc.say(thread, text, by=YOU)
            await sc.settle()
            out.new()

        async def wait(seconds: int) -> None:
            print(f"{DIM}   ({seconds} seconds pass){RESET}")
            sc.advance(seconds)
            await sc.settle()
            out.new()

        await say("Compare Bangkok and Lisbon for our team offsite")
        desk = sc.runs(thread)[0]
        print(f"{BOLD}you{RESET}: March")
        await sc.answer(desk.ref, "March", by=YOU)  # answers the waiting run: the same run carries on
        await sc.settle()
        out.new()

        await say("Focus on cost please")  # reaches the waiting desk run, which steers its city runs
        await wait(30)  # the city runs wake up, read the steering, ask prices

        await say("Also add Rome")  # a new turn: the desk remembers the cities and the month
        await wait(30)

        print()
        out.runs()


if __name__ == "__main__":
    asyncio.run(main())
