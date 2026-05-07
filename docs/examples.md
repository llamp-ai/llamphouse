# Examples

The [examples/](https://github.com/llamp-ai/llamphouse/tree/main/examples) directory contains runnable samples for every major feature.
Each example includes an `agents.py` (or `server.py`), `client.py`, and `README.md` with instructions.

## Example index

| Example | Description | Key features |
|---|---|---|
| [01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld) | Minimal agent — no LLM needed | Agent basics, `insert_message()` |
| [02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat) | OpenAI-powered conversational agent | LLM integration, conversation history |
| [03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming) | Real-time token streaming with SSE | `process_stream()`, stream adapters |
| [04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall) | Function calling with tool schemas | Tool schemas, `pending_tool_calls` |
| [05_OrchestratorAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/05_OrchestratorAgent) | Multi-agent orchestration | `call_agent()`, streaming |
| [06_AgentHandover](https://github.com/llamp-ai/llamphouse/tree/main/examples/06_AgentHandover) | Multi-agent handover | `handover_to_agent()` |
| [07_Tracing](https://github.com/llamp-ai/llamphouse/tree/main/examples/07_Tracing) | OpenTelemetry distributed tracing | OTel setup, trace propagation |
| [08_ConfigStore](https://github.com/llamp-ai/llamphouse/tree/main/examples/08_ConfigStore) | Runtime-tunable agent config | Config params, Compass UI |
| [09_CustomAuth](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_CustomAuth) | Custom authentication | `BaseAuth`, streaming |
| [10_DistributedWorker](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_DistributedWorker) | Redis-backed distributed workers | Redis queue, split-mode, Postgres |
| [11_WebhookSignal](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_WebhookSignal) | Webhook signal integration | Signals, external callbacks |
| [12_PlannerAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_PlannerAgent) | Planner/executor agent pattern | Planning, tool use |
| [13_LLAMPHouseYAML](https://github.com/llamp-ai/llamphouse/tree/main/examples/13_LLAMPHouseYAML) | Config-driven runtime — no server.py required | YAML config, CLI |

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

1. **[01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld)** — 👋 Hello World
2. **[02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat)** — 💬 Chat
3. **[03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming)** — 🌊 Streaming
4. **[04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall)** — Tool Call Example
5. **[05_OrchestratorAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/05_OrchestratorAgent)** — Central Orchestrator with Review & Correction
6. **[06_AgentHandover](https://github.com/llamp-ai/llamphouse/tree/main/examples/06_AgentHandover)** — Agent Handover
7. **[07_Tracing](https://github.com/llamp-ai/llamphouse/tree/main/examples/07_Tracing)** — Tracing Example (A2A)
8. **[08_ConfigStore](https://github.com/llamp-ai/llamphouse/tree/main/examples/08_ConfigStore)** — Config Store Example
9. **[09_CustomAuth](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_CustomAuth)** — Custom Authenticator Example
10. **[10_DistributedWorker](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_DistributedWorker)** — AsyncWorker vs DistributedWorker
11. **[11_WebhookSignal](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_WebhookSignal)** — Webhook Signal
12. **[12_PlannerAgent](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_PlannerAgent)** — Planner Agent
13. **[13_LLAMPHouseYAML](https://github.com/llamp-ai/llamphouse/tree/main/examples/13_LLAMPHouseYAML)** — 📄 LLAMPHouse YAML

## Next steps

- [Quickstart](getting-started/quickstart.md) — build your first agent from scratch
- [Core Concepts](concepts/agents.md) — understand the fundamentals
- [Guides](guides/streaming.md) — deep dives into specific features
