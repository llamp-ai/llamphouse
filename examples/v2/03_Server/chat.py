"""Talk to the served desk from a terminal, over HTTP. Run server.py first.

    python chat.py                 # a new thread
    python chat.py <thread id>     # carry on with one (also after a server restart)

One app policy, written here and not in llamphouse: a message starts a run when the thread is idle,
answers the run's question when it asked one, and otherwise goes to the active run's inbox.
"""

import asyncio
import os
import sys

from client import ApiError, Client, text_of


async def show(client: Client, thread_id: str) -> None:
    async for ev in client.watch(thread_id):
        if ev["message"] and ev["message"]["sender"]["kind"] == "agent":
            print(f"\n{ev['message']['sender']['id']}: {text_of(ev['message'])}\n> ", end="", flush=True)
        elif ev["kind"] == "run.input_required":  # ask() is a request on the run, not a message
            pending = (await client.run(ev["run"]))["pending"]
            if pending:
                print(f"\n{ev['agent'] or 'agent'} asks: {pending['reason']}\n> ", end="", flush=True)
        elif ev["kind"] in ("run.completed", "run.failed", "run.cancelled"):
            print(f"\n[{ev['kind']}]\n> ", end="", flush=True)


async def say(client: Client, thread_id: str, text: str) -> None:
    active = (await client.get_thread(thread_id))["active_run"]
    try:
        if active is None:
            await client.start(thread_id, text, sender={"id": "you"})
        elif active["pending"] is not None:
            await client.answer(active["id"], text, sender={"id": "you"})
        else:
            await client.send(active["id"], text, sender={"id": "you"})
    except ApiError as e:
        if e.error in ("thread_busy", "no_active_run", "run_ended"):  # it changed meanwhile: once more
            return await say(client, thread_id, text)
        raise


async def main() -> None:
    async with Client(os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000"),
                      api_key=os.environ.get("LLAMPHOUSE_API_KEY")) as client:
        thread_id = sys.argv[1] if len(sys.argv) > 1 else (await client.thread("desk"))["id"]
        print(f"thread {thread_id} - Compass: {str(client.http.base_url).rstrip('/')}/compass#{thread_id}")
        watcher = asyncio.create_task(show(client, thread_id))
        loop = asyncio.get_running_loop()
        try:
            while True:
                line = (await loop.run_in_executor(None, input, "> ")).strip()
                if line in ("/quit", "/q"):
                    break
                if line:
                    await say(client, thread_id, line)
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            watcher.cancel()


if __name__ == "__main__":
    asyncio.run(main())
