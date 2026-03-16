# Examples

The [examples/](https://github.com/llamp-ai/llamphouse/tree/main/examples) directory contains runnable samples for every major feature. Each example includes a `server.py`, `client.py`, and `README.md` with instructions.

## Example index

| Example | Description | Key features |
|---|---|---|
| [01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld) | Minimal agent — no LLM needed | Agent basics, `insert_message()` |
| [02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat) | OpenAI-powered conversational agent | LLM integration, conversation history |
| [03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming) | Real-time token streaming with SSE | `process_stream()`, stream adapters |
| [04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall) | Function calling with tool schemas | Tool schemas, `pending_tool_calls` |
| [06_GeminiStreaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/06_GeminiStreaming) | Streaming with Google Gemini | Gemini adapter, multi-provider |
| [08_Tracing](https://github.com/llamp-ai/llamphouse/tree/main/examples/08_Tracing) | OpenTelemetry distributed tracing | OTel setup, trace propagation |
| [09_A2A](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_A2A) | A2A protocol agent | A2A adapter, agent cards |
| [10_A2A_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/10_A2A_ToolCall) | A2A with tool calls | A2A + function calling |
| [11_AgentHandover](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_AgentHandover) | Multi-agent handover | `handover_to_agent()` |
| [12_CentralOrchestrator](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_CentralOrchestrator) | Central orchestrator pattern | `call_agent()`, multi-agent |
| [13_ConfigStore](https://github.com/llamp-ai/llamphouse/tree/main/examples/13_ConfigStore) | Runtime-tunable agent config | Config params, Compass UI |
| [14_DistributedWorker](https://github.com/llamp-ai/llamphouse/tree/main/examples/14_DistributedWorker) | Separate API and worker processes | Redis queues, split-mode |
| [15_A2A_AIFoundry](https://github.com/llamp-ai/llamphouse/tree/main/examples/15_A2A_AIFoundry) | A2A with Azure AI Foundry | Azure integration |
| [LangGraph](https://github.com/llamp-ai/llamphouse/tree/main/examples/LangGraph) | LangGraph integration | Framework integration |

## Running an example

Most examples follow the same pattern:

```bash
# Navigate to the example
cd examples/01_HelloWorld

# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py

# In another terminal, run the client
python client.py
```

Some examples require environment variables (e.g., `OPENAI_API_KEY`). Check each example's `README.md` for specific instructions.

## Progression guide

If you're new to LLAMPHouse, we recommend working through the examples in this order:

1. **[01_HelloWorld](https://github.com/llamp-ai/llamphouse/tree/main/examples/01_HelloWorld)** — understand the basics
2. **[02_Chat](https://github.com/llamp-ai/llamphouse/tree/main/examples/02_Chat)** — add an LLM
3. **[03_Streaming](https://github.com/llamp-ai/llamphouse/tree/main/examples/03_Streaming)** — enable streaming
4. **[04_ToolCall](https://github.com/llamp-ai/llamphouse/tree/main/examples/04_ToolCall)** — add function calling
5. **[09_A2A](https://github.com/llamp-ai/llamphouse/tree/main/examples/09_A2A)** — try the A2A protocol
6. **[11_AgentHandover](https://github.com/llamp-ai/llamphouse/tree/main/examples/11_AgentHandover)** — multi-agent basics
7. **[12_CentralOrchestrator](https://github.com/llamp-ai/llamphouse/tree/main/examples/12_CentralOrchestrator)** — orchestration patterns

## Next steps

- [Quickstart](getting-started/quickstart.md) — build your first agent from scratch
- [Core Concepts](concepts/agents.md) — understand the fundamentals
- [Guides](guides/streaming.md) — deep dives into specific features
