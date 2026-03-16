# Example 05 — Central Orchestrator with Review & Correction

Demonstrates a **central orchestrator** that checks every sub-agent’s output
and sends corrections back so the agent can revise its work.

## Architecture

```
User
 │
 ▼
Orchestrator
 │
 ├─ Phase 1: call_agent("researcher", topic)
 │            → researcher produces draft research (cyan / dimmed)
 │
 ├─ Phase 2: Orchestrator reviews the research
 │            → streams feedback to client (yellow)
 │
 ├─ Phase 3: call_agent("researcher", feedback, thread_id=…)
 │            → researcher revises on same thread (blue / bold)
 │
 ├─ Phase 4: call_agent("writer", revised_research)
 │            → writer produces draft article (cyan / dimmed)
 │
 ├─ Phase 5: Orchestrator reviews the article
 │            → streams feedback to client (yellow)
 │
 └─ Phase 6: call_agent("writer", feedback, thread_id=…)
              → writer revises on same thread (green / bold)
```

All three agents share one `LLAMPHouse` instance. The `AsyncWorker`
dispatches each run as a concurrent `asyncio.create_task`.

### Agents

| Agent | ID | Responsibility |
|-------|----|---------------|
| Orchestrator | `orchestrator` | Reviews sub-agent output, sends corrections |
| Researcher | `researcher` | Gathers factual information on any topic |
| Writer | `writer` | Crafts polished, engaging content from research |

### Key patterns demonstrated

| Pattern | How |
|---------|-----|
| **Draft → Review → Revise** | Both sub-agents go through this cycle |
| **Orchestrator as quality gate** | Own LLM call reviews each agent's output |
| **Feedback sent back to agent** | Second `call_agent()` reuses the same thread so the agent sees its draft |
| **Thread reuse** | `context.last_call_thread_id` captures the thread from the first call; passed back via `thread_id=` for revision |
| **Streamed output** | Every phase streams to the client in real-time |
| **Progress events** | `context.emit("progress", ...)` shows status updates |
| **Colored CLI output** | Client detects phase headers and applies ANSI colors |

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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Review-and-Correct Pipeline (Researcher + Writer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User: Write a blog post about the James Webb Space Telescope

  ▸ 📚 Phase 1: Researcher gathering facts...
  ## 📚 Draft Research                         ← cyan / dimmed
  The James Webb Space Telescope (JWST) is...

  ▸ 🔍 Phase 2: Reviewing research...
  ## 🔍 Research Review                        ← yellow
  Found 3 issues: 1. Launch date is wrong...

  ▸ 🔄 Phase 3: Researcher revising...
  ## 🔄 Revised Research                       ← blue / bold
  The JWST, launched on December 25, 2021...

  ▸ ✍️ Phase 4: Writer creating article...
  ## ✍️ Draft Article                          ← cyan / dimmed
  Looking Up: How JWST Is Rewriting Astronomy...

  ▸ 🔍 Phase 5: Reviewing article...
  ## 🔍 Article Review                         ← yellow
  Found 2 issues: 1. Missing conclusion...

  ▸ ✏️ Phase 6: Writer revising...
  ## ✏️ Final Article                          ← green / bold
  Looking Up: How JWST Is Rewriting Astronomy...

  ✓ Done
```

## Key concepts

- **`call_agent()` as async generator** — the orchestrator controls what happens
  with each chunk: collect, transform, or forward to the client.
- **`send_chunk(text)`** — pushes a MESSAGE_DELTA event to the client.
- **Review-and-correct loop** — the orchestrator reviews each sub-agent’s
  output using its own LLM call, then sends the feedback back to the same
  agent for revision on the **same thread** (via `thread_id=`), so the agent
  sees its original draft in the conversation history. Both the researcher
  and writer go through this cycle.
- **Six-phase pipeline** — draft → review → revise, repeated for each agent.
- **Thread reuse** — `context.last_call_thread_id` captures the thread created
  by the first `call_agent()` call. Passing it back as `thread_id=` in the
  revision call means the agent sees the full conversation, and the revision
  prompt only needs to include the editor's feedback — not re-paste the draft.
- **Colored CLI client** — detects emoji headers (📚, 🔍, 🔄, ✍️, ✏️) and
  switches ANSI colors to visually distinguish each phase.
