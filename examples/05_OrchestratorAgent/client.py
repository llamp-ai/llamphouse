"""Central Orchestrator — streaming client with colored output.

Sends a request to the Orchestrator (Port 8000) and streams the
six-phase review-and-correct pipeline. Each agent has its own color:

  Researcher  (📚 / 🔄) — cyan
  Orchestrator (🔍)      — yellow
  Writer      (✍️ / ✏️)  — green
  Status                  — magenta
"""

import asyncio
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendStreamingMessageRequest,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

BASE_URL = "http://127.0.0.1:8000"


# ── ANSI helpers ─────────────────────────────────────────────────────────────────


class C:
    """ANSI color codes."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    GREEN   = "\033[32m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    WHITE   = "\033[97m"


# Agent → color mapping (same agent = same color)
AGENT_COLORS = {
    "researcher":   C.CYAN,
    "orchestrator": C.YELLOW,
    "writer":       C.GREEN,
}


def _part_text(part) -> str:
    root = getattr(part, "root", part)
    return getattr(root, "text", "") or ""


def _status_msg_text(event: TaskStatusUpdateEvent) -> str:
    if not event.status.message:
        return ""
    return "".join(_part_text(p) for p in event.status.message.parts)


def _detect_agent(text: str) -> str | None:
    """Detect which agent produced a phase-switch header."""
    if "##" not in text:
        return None
    if "\U0001f4da" in text:          # 📚  Draft Research      → Researcher
        return "researcher"
    if "\U0001f504" in text:          # 🔄  Revised Research    → Researcher
        return "researcher"
    if "\U0001f50d" in text:          # 🔍  Review             → Orchestrator
        return "orchestrator"
    if "\u270f" in text:              # ✏️   Final Article       → Writer
        return "writer"
    if "\u270d" in text:              # ✍️   Draft Article       → Writer
        return "writer"
    return None


async def stream_question(client: A2AClient, label: str, question: str):
    """Send a question via streaming and print with colored phases."""
    bar = "\u2501" * 60
    print(f"\n{C.BOLD}{C.WHITE}{bar}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}  {label}{C.RESET}")
    print(f"{C.BOLD}{C.WHITE}{bar}{C.RESET}")
    print(f"\n  {C.BOLD}User:{C.RESET} {question}\n")

    text_started = False
    current_color = C.RESET

    async for chunk in client.send_message_streaming(
        SendStreamingMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": question}],
                    "messageId": uuid4().hex,
                }
            ),
        )
    ):
        result = chunk.root.result

        if isinstance(result, TaskStatusUpdateEvent):
            msg = _status_msg_text(result)

            if result.final:
                if text_started:
                    print(C.RESET, flush=True)
                state = getattr(result.status.state, "value", str(result.status.state))
                if state != "completed":
                    print(f"  {C.MAGENTA}[{state}]{C.RESET}", flush=True)
                else:
                    print(f"\n  {C.GREEN}{C.BOLD}\u2713 Done{C.RESET}\n", flush=True)
            elif msg:
                if text_started:
                    print(C.RESET, flush=True)
                    text_started = False
                print(f"  {C.MAGENTA}{C.BOLD}\u25b8 {msg}{C.RESET}", flush=True)

        elif isinstance(result, TaskArtifactUpdateEvent):
            for part in result.artifact.parts:
                text = _part_text(part)
                if text:
                    agent = _detect_agent(text)
                    if agent:
                        if text_started:
                            print(C.RESET, flush=True)
                            text_started = False
                        current_color = AGENT_COLORS.get(agent, C.RESET)

                    if not text_started:
                        print(f"  {current_color}", end="", flush=True)
                        text_started = True
                    print(f"{current_color}{text}", end="", flush=True)

    print(C.RESET)


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as httpx_client:

        # ── Agent card discovery ───────────────────────────────────────────────
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()

        print(f"\n{C.BOLD}{C.WHITE}\u2501\u2501\u2501 Agent Card \u2501\u2501\u2501{C.RESET}")
        print(f"  {C.BOLD}Name:{C.RESET}        {card.name}")
        print(f"  {C.BOLD}Description:{C.RESET} {card.description}")
        if card.skills:
            print(f"  {C.BOLD}Skills:{C.RESET}      {[s.name for s in card.skills]}")
        print()

        client = A2AClient(httpx_client=httpx_client, agent_card=card)

        # ── Content-creation request ───────────────────────────────────────────
        await stream_question(
            client,
            "Review-and-Correct Pipeline (Researcher + Writer)",
            "Write a short, engaging blog post about the James Webb Space Telescope "
            "and its most exciting discoveries so far.",
        )


if __name__ == "__main__":
    asyncio.run(main())
