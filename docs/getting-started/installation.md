# Installation

## Requirements

- **Python 3.10** or later
- pip (or any Python package manager)

## Install from PyPI

```bash
pip install llamphouse
```

This installs the core LLAMPHouse package with all required dependencies.

## Install from source

For development or to use the latest unreleased features:

```bash
git clone https://github.com/llamp-ai/llamphouse.git
cd llamphouse
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The `[dev]` extra installs testing and development dependencies (pytest, tox, etc.).

## Optional dependencies

Depending on your deployment setup, you may need additional packages:

| Package | When needed |
|---|---|
| `asyncpg` / `psycopg2` | Postgres data store |
| `redis` | Redis run queue / event queue |
| `opentelemetry-sdk` | Tracing (auto-installed) |
| `openai` | Using OpenAI as your LLM provider |
| `google-genai` | Using Google Gemini as your LLM provider |
| `anthropic` | Using Anthropic as your LLM provider |

## Verify the installation

```bash
python -c "import llamphouse; print('LLAMPHouse installed successfully')"
```

## Next steps

Head to the [Quickstart](quickstart.md) to create your first agent.
