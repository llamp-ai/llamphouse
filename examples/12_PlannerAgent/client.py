"""Client for example 12 — PlannerAgent.

Demonstrates five planning patterns:

  1  single_tool       — one tool call (stock price lookup)
  2  parallel_tools    — multiple independent tools fired at the same time
  3  sequential_tools  — each step depends on the output of the previous one
  4  mixed             — parallel lookups followed by a derived calculation
  5  human_feedback    — ambiguous request; agent must ask before it can act
  6  portfolio         — multi-step: prices → compute share counts → leftover cash

Usage
-----
  python client.py              # interactive menu
  python client.py 3            # run sample #3 directly
  python client.py sequential   # run sample by name
"""

import asyncio
import sys
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

BASE_URL = "http://127.0.0.1:8000"

# ── Sample cases ──────────────────────────────────────────────────────────────
# Each entry: (short_label, description, question)

SAMPLES: list[tuple[str, str, str]] = [
    (
        "single_tool",
        "Single tool — one stock price lookup",
        "What is the current stock price of Tesla (TSLA)?",
    ),
    (
        "parallel_tools",
        "Parallel tools — fetch all stock prices at the same time",
        (
            "Get the current stock prices for AAPL, MSFT, TSLA, and GOOG "
            "in a single parallel step, then list them from most to least expensive."
        ),
    ),
    (
        "sequential_tools",
        "Sequential tools — each step feeds the next",
        (
            "Search the web for background information about Microsoft, "
            "then summarise what you found, "
            "and finally look up the current MSFT stock price. "
            "Give me all three results together."
        ),
    ),
    (
        "mixed",
        "Mixed — parallel lookups + derived calculation",
        (
            "Compare Apple and Microsoft: search for background on both companies "
            "and get the current stock prices for AAPL and MSFT in parallel. "
            "Then calculate what AAPL would cost if it rose 10%. "
            "Give me a concise side-by-side comparison."
        ),
    ),
    (
        "human_feedback",
        "Human feedback — agent must ask a clarifying question first",
        (
            "I want to look into a well-known tech stock, but I haven't decided which one yet. "
            "Ask me which company or ticker I mean before you do anything else, "
            "then search the web for background and get the current price."
        ),
    ),
    (
        "portfolio",
        "Portfolio — multi-step: prices → share counts → leftover cash",
        (
            "I have $10,000 to invest equally across AAPL, MSFT, and TSLA. "
            "Get each stock price, then calculate how many whole shares of each "
            "I can buy and how much cash is left over. Show the full breakdown."
        ),
    ),
]

SAMPLE_BY_NAME = {name: (label, q) for name, label, q in SAMPLES}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_sample() -> tuple[str, str]:
    """Return (label, question) — either from argv or an interactive menu."""
    if len(sys.argv) > 1:
        key = sys.argv[1]
        # Accept a 1-based index
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(SAMPLES):
                name, label, q = SAMPLES[idx]
                return label, q
            print(f"Index {key} out of range (1–{len(SAMPLES)}).")
            sys.exit(1)
        # Accept a name
        if key in SAMPLE_BY_NAME:
            label, q = SAMPLE_BY_NAME[key]
            return label, q
        print(f"Unknown sample '{key}'. Valid names: {', '.join(SAMPLE_BY_NAME)}")
        sys.exit(1)

    # Interactive menu
    print("Available sample cases:\n")
    for i, (name, label, _) in enumerate(SAMPLES, 1):
        print(f"  {i}. [{name}]  {label}")
    print()
    choice = input("Pick a number (or name): ").strip()
    if choice.isdigit():
        idx = int(choice) - 1
        name, label, q = SAMPLES[idx]
        return label, q
    if choice in SAMPLE_BY_NAME:
        label, q = SAMPLE_BY_NAME[choice]
        return label, q
    print("Invalid choice.")
    sys.exit(1)


async def main():
    label, question = pick_sample()

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()

        print(f"\n=== {card.name}  —  {label} ===\n")
        print(f"> User: {question}\n")
        print("> Agent:\n")

        factory = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=True))
        client = factory.create(card)

        msg = Message(
            messageId=uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
        )

        async for event in client.send_message(msg):
            if isinstance(event, tuple):
                _, streaming_event = event

                if isinstance(streaming_event, TaskArtifactUpdateEvent):
                    for part in streaming_event.artifact.parts:
                        if hasattr(part.root, "text") and part.root.text:
                            print(part.root.text, end="", flush=True)

                elif isinstance(streaming_event, TaskStatusUpdateEvent):
                    if streaming_event.final:
                        state = streaming_event.status.state.value
                        if state != "completed":
                            print(f"\n[{state}]")

        print()


if __name__ == "__main__":
    asyncio.run(main())
