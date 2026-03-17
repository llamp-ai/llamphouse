# Tool Call Example

This example demonstrates a minimal tool-calling loop inside a custom `Agent` running on a LLAMPHouse server, exposed via the **A2A protocol**.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`

## Setup

1. Clone the repository and go to this example:

   ```bash
   git clone https://github.com/llamp-ai/llamphouse.git
   cd llamphouse/examples/04_ToolCall
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Create your `.env` from `.env.sample`:

   - `OPENAI_API_KEY=...` (required)
   - `DATABASE_URL=...` (optional; only for Postgres)

   ```bash
   cp .env.sample .env
   ```

## Tool Call (How it works)

- Tool function: [get_current_time](server.py#L15)
- Tool schema exposed to the model: [TOOL_SCHEMAS](server.py#L19)
- Tool registry used to execute tool locally: [TOOL_REGISTRY](server.py#L30)
- The assistant calls OpenAI with `tools=...` and `tool_choice="auto"`
- If the model returns `tool_calls`, the server executes the tool and appends a `"tool"` message with the result
- Loop is limited to 3 iterations

Client sends the [question](client.py#L14) defined in `client.py`.

## Running

1. Start the server:
   ```sh
   python server.py
   ```

2. In a second terminal, run the client:
   ```sh
   python client.py
   ```