# Examples

The [examples/](https://github.com/llamp-ai/llamphouse/tree/main/examples) directory contains runnable samples for every major feature.
Each example includes an `agents.py` (or `server.py`), `client.py`, and `README.md` with instructions.

## Example index

| Example | Description | Key features |
|---|---|---|
| [01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld) | The simplest possible LLAMPHouse agent — no API keys, no LLM, just a | `Agent`, `A2AAdapter` |
| [02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat) | A LLAMPHouse agent that holds a real conversation using OpenAI's Chat | `run()`, `forward the conversation hi…` |
| [03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming) | A LLAMPHouse agent that streams its response token-by-token to the | `AsyncOpenAI`, `context.process_stream()` |
| [04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall) | This example demonstrates a minimal tool-calling loop inside a custom Agent running on a LLAMPHouse server, exposed v… |  |
| [05_OrchestratorAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/05_OrchestratorAgent) | Demonstrates a central orchestrator that checks every sub-agent’s output |  |
| [06_AgentHandover](https://github.com/llamp-ai/llamphouse/tree/main/examples/06_AgentHandover) | Demonstrates how one agent can hand over a user request to a specialised agent at runtime, with both agents running o… |  |
| [07_Tracing](https://github.com/llamp-ai/llamphouse/tree/main/examples/07_Tracing) | End-to-end OpenTelemetry tracing with an A2A client and server. The client creates a root span and injects W3C tracep… |  |
| [08_ConfigStore](https://github.com/llamp-ai/llamphouse/tree/main/examples/08_ConfigStore) | Demonstrates the config store feature with A2A streaming — define |  |
| [09_CustomAuth](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_CustomAuth) | Demonstrates how to implement a custom authenticator on your LLAMPHouse server with A2A streaming. |  |
| [10_DistributedWorker](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_DistributedWorker) | Compares the two worker modes in llamphouse using the A2A streaming protocol: |  |
| [11_WebhookTrigger](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_WebhookTrigger) | Trigger an agent via an HTTP POST instead of a human chat message. | `WebhookTrigger`, `secret_env` |
| [12_PlannerAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_PlannerAgent) | A generic ReAct-style Planner-Executor agent that you can drop into any project by providing a list of tools and a re… |  |
| [13_LLAMPHouseYAML](https://github.com/llamp-ai/llamphouse/tree/main/examples/13_LLAMPHouseYAML) | The config-driven way to run LLAMPHouse — no server.py required. | `server.py`, `The difference between a de…` |
| [14_WorkflowSteps](https://github.com/llamp-ai/llamphouse/tree/main/examples/14_WorkflowSteps) | Demonstrates the experimental @step decorator — the first piece of a | `@step`, `How input/output is capture…` |
| [15_LangGraph](https://github.com/llamp-ai/llamphouse/tree/main/examples/15_LangGraph) | This example shows a non-trivial LangGraph workflow running behind the LLAMPHouse runtime contract. |  |

## Running an example

Most examples follow the same pattern:

```bash
# Navigate to the example
cd examples/01_HelloWorld

# Install dependencies
pip install -r requirements.txt

# Start the server
llamphouse up   # or: python server.py

# In another terminal, run the client
python client.py
```

Some examples require environment variables (e.g., `OPENAI_API_KEY`). Check each example's `README.md` for specific instructions.

## Progression guide

If you're new to LLAMPHouse, we recommend working through the examples in this order:

2. **[01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld)** — 👋 Hello World
3. **[02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat)** — 💬 Chat
4. **[03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming)** — 🌊 Streaming
5. **[04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall)** — Tool Call Example
6. **[05_OrchestratorAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/05_OrchestratorAgent)** — Central Orchestrator with Review & Correction
7. **[06_AgentHandover](https://github.com/llamp-ai/llamphouse/tree/main/examples/06_AgentHandover)** — Agent Handover
8. **[07_Tracing](https://github.com/llamp-ai/llamphouse/tree/main/examples/07_Tracing)** — Tracing Example (A2A)
9. **[08_ConfigStore](https://github.com/llamp-ai/llamphouse/tree/main/examples/08_ConfigStore)** — Config Store Example
10. **[09_CustomAuth](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_CustomAuth)** — Custom Authenticator Example
11. **[10_DistributedWorker](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_DistributedWorker)** — AsyncWorker vs DistributedWorker
12. **[11_WebhookTrigger](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_WebhookTrigger)** — Webhook Trigger
13. **[12_PlannerAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_PlannerAgent)** — Planner Agent
14. **[13_LLAMPHouseYAML](https://github.com/llamp-ai/llamphouse/tree/main/examples/13_LLAMPHouseYAML)** — 📄 LLAMPHouse YAML
15. **[14_WorkflowSteps](https://github.com/llamp-ai/llamphouse/tree/main/examples/14_WorkflowSteps)** — 🪜 Workflow Steps
16. **[15_LangGraph](https://github.com/llamp-ai/llamphouse/tree/main/examples/15_LangGraph)** — LangGraph Branching Wrapper Example

## Next steps

- [Quickstart](getting-started/quickstart.md) — build your first agent from scratch
- [Core Concepts](concepts/agents.md) — understand the fundamentals
- [Guides](guides/streaming.md) — deep dives into specific features
