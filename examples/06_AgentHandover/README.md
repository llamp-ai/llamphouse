# Example 11 — Agent Handover

Demonstrates how one agent can **hand over** a user request to a specialised agent at runtime, with both agents running on the **same LLAMPHouse server** and communicating via the **A2A protocol** — with the specialist's response streamed in real-time back to the client.

## Architecture

```
User
 │
 ▼
Receptionist (default assistant)
 │  General question? → answers and streams directly
 │  Coding question?  → calls handover_to_specialist tool
 │                            │
 │              internal A2A message/stream (same port)
 │              metadata: {assistant_id: "coding-specialist"}
 │                            │
 │           ┌────────────────┘
 │           │  AsyncWorker picks up the specialist run (create_task)
 │           ▼
 │    Coding Specialist
 │           │  streams answer chunk-by-chunk
 │           │
 │    each chunk → context._send_event(MESSAGE_DELTA)
 │           │       relayed live to the connected client
 │           │
 └───────────┘
    insert_message() persists text + signals MESSAGE_COMPLETED
```

Both assistants are registered on a single `LLAMPHouse` instance. When the Receptionist decides to hand over a query, it makes an internal A2A `message/stream` call to the **same server**, targeting the Coding Specialist with `metadata: {"assistant_id": "coding-specialist"}`.

The `AsyncWorker` dispatches each run as a concurrent `asyncio.create_task`, so the dequeue loop is never blocked. While the Receptionist awaits the Specialist's SSE stream, the worker immediately picks up the Specialist run — no deadlock, no extra worker configuration needed.

## How to run

**1. Start the server:**
```bash
python server.py
```

**2. Run the client:**
```bash
python client.py
```

## What to expect

```
=== General question (handled directly by Receptionist) ===
  User: What is the capital of Australia?
  Assistant: The capital of Australia is Canberra.

=== Coding question (handed over to Coding Specialist, streamed live) ===
  User: Write a Python function that checks whether a number is prime.
  Assistant: Here's a Python function that checks whether a number is prime:
  ...
```

## Key concepts

- **Single server, multiple agents** — both assistants are registered in one `LLAMPHouse([receptionist, specialist])` call.
- **A2A for internal calls** — agent-to-agent communication uses the same A2A protocol as external callers, keeping the architecture consistent.
- **Targeting an assistant** — pass `metadata: {"assistant_id": "<id>"}` in the A2A params to route a run to a specific assistant (falls back to the first assistant if omitted).
- **Real-time streaming relay** — the Receptionist subscribes to the Specialist's `message/stream` SSE feed and relays each text chunk as a `MESSAGE_DELTA` event via `context._send_event()`, so the client sees output as it is generated.
- **No duplication** — `insert_message()` only emits `MESSAGE_CREATED`, `MESSAGE_IN_PROGRESS`, and `MESSAGE_COMPLETED` — never `MESSAGE_DELTA` — so persisting the full text after streaming does not repeat any chunks on the wire.
- **asyncio.create_task concurrency** — the `AsyncWorker` fires each run as a non-blocking task; the dequeue loop stays free to pick up the Specialist run while the Receptionist is awaiting its stream.
- **LLM-driven routing** — the decision to hand over is made by the LLM via a tool call, not hard-coded logic.
