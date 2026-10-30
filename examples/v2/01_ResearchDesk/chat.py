"""Talk to the research desk yourself. Run: python chat.py

Type a message. When the desk asks something, your next line answers it.
  /wait [seconds]  let time pass (default 30) - the city runs sleep 30 seconds
  /runs            every run so far, and where it runs
  /timeline        the desk's thread, event by event
  /quit
"""

import asyncio

from llamphouse.sdk import Sender
from llamphouse.sdk.testing import Scenario

from agents import AGENTS
from show import Printer

YOU = Sender.person("you", name="You")


async def main() -> None:
    print(__doc__)
    async with Scenario(AGENTS, workers=4) as sc:
        thread = await sc.thread("desk")
        out = Printer(sc, thread)
        while True:
            line = (await asyncio.to_thread(input, "> ")).strip()
            if not line:
                continue
            if line == "/quit":
                break
            if line.startswith("/wait"):
                sc.advance(float(line.split()[1]) if " " in line else 30)
            elif line == "/runs":
                out.runs()
                continue
            elif line == "/timeline":
                sc.print_timeline(thread)
                continue
            elif (asking := out.pending()) is not None:
                await sc.answer(asking.ref, line, by=YOU)
            else:
                await sc.say(thread, line, by=YOU)
            await sc.settle()
            out.new()


if __name__ == "__main__":
    asyncio.run(main())
