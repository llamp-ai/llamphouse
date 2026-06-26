# 📄 LLAMPHouse YAML

The config-driven way to run LLAMPHouse — **no `server.py` required**.

Describe your agents and deployments in a `llamphouse.yaml` file and start
the server with a single command:

```sh
llamphouse up
```

This example shows how **one agent class can be deployed as multiple
independent instances**, each with its own personality, configured entirely
through `llamphouse.yaml`.

## What you'll learn

- How to replace `server.py` with a `llamphouse.yaml` config file
- The difference between a **definition** (the class) and an
  **agent** (a running instance with specific settings)
- How an agent reads its deployment config via `self.settings`
- How to declare a webhook trigger for one deployment
- How to choose a data store from YAML
- How to target a specific agent from a client

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |
| LLAMPHouse 1.2+ | `pip install -r requirements.txt` |

> **No API keys needed!** Both agents return deterministic replies, so
> there's nothing to configure.

## Project structure

```
13_LLAMPHouseYAML/
├── llamphouse.yaml   ← all server config lives here
├── agents.py         ← one reusable GreetingAgent class
├── client.py         ← discovers and talks to both deployments
└── requirements.txt
```

Notice: there is **no `server.py`**.

## Quick start

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Start the server

```sh
llamphouse up
```

You should see output like:

```
LLAMPHOUSE Loading deployment 'greeter-formal' ← agent 'greeting-agent' (agents.py:GreetingAgent)
LLAMPHOUSE Loading deployment 'greeter-casual' ← agent 'greeting-agent' (agents.py:GreetingAgent)
LLAMPHOUSE Project 'greeter-platform' — 2 agent(s) loaded.
LLAMPHOUSE We have light!
LLAMPHOUSE Server: http://0.0.0.0:8000
```

### 3. In a second terminal, run the client

```sh
python client.py
```

You'll see both agents reply to the same message with different
personalities:

```
Found 2 agent(s):

  • [greeter-formal]  greeter-formal
  • [greeter-casual]  greeter-casual

Sending to all agents: "Can you introduce yourself..."
============================================================

[greeter-formal]
----------------------------------------
Good day, esteemed visitor. I am at your service.

Speaking as your formal butler, I heard you say:
  "Can you introduce yourself..."
...

[greeter-casual]
----------------------------------------
Hey there! What's up, dude?

Speaking as your laid-back surfer, I heard you say:
  "Can you introduce yourself..."
...
```

## How it works

### `llamphouse.yaml`

The config has three sections:

```yaml
definitions:     # ← the class / entrypoint (reusable)
  - name: greeting-agent
    entrypoint: agents.py:GreetingAgent

agents:          # ← running instances with their own config
  - name: greeter-formal
    definition: greeting-agent
    config:
      persona: formal butler
      greeting: "Good day, esteemed visitor."
    triggers:
      - webhook:
          path: /triggers/greeter-formal
          idempotency:
            key: id
          thread_metadata:
            tenant_id: tenant.id
          run_metadata:
            event_type: type
            event_id: id

  - name: greeter-casual
    definition: greeting-agent
    config:
      persona: laid-back surfer
      greeting: "Hey there! What's up, dude?"

data_store:
  in_memory:
```

When you run `llamphouse up`, the CLI:

1. Parses and validates `llamphouse.yaml`.
2. Imports `GreetingAgent` from `agents.py`.
3. Creates **two separate instances** — `greeter-formal` and `greeter-casual`
   — and sets `agent.settings` on each from its `config` block.
4. Starts a LLAMPHouse server with both instances registered.

The `greeter-formal` deployment also exposes `POST /triggers/greeter-formal`.
Mapped payload fields are copied into thread/run metadata when present; missing
fields are ignored and the webhook still accepts the event. `idempotency.key`
uses a JSON body dot-path to dedupe normal webhook retries for the same
deployment and trigger path.

### `agents.py`

`GreetingAgent` reads its personality from `self.settings`:

```python
class GreetingAgent(Agent):
    async def run(self, context: Context):
        persona  = self.settings.get("persona",  "helpful assistant")
        greeting = self.settings.get("greeting", "Hello!")
        ...
```

The same class runs differently depending on which agent config it was
instantiated for — no if/else, no environment variables.

### `client.py`

The client:

1. **Discovers** all agents via `GET /agents`.
2. **Routes** to a specific agent by passing
   `"metadata": {"assistant_id": "<agent-name>"}` in the JSON-RPC
   request body.

## Supported entrypoint formats

| Format | Description |
|---|---|
| `agents.py:GreetingAgent` | An `Agent` subclass — instantiated directly |
| `agents.py:run` | An `async def run(context)` function — auto-wrapped |
| `agents.py:create` | A factory `(deployment_cfg: dict) -> Agent` |

## Next steps

| Topic | Details |
|---|---|
| More agents | Add a third entry under `agents:` with `definition: greeting-agent` |
| Secrets | Use `secrets:` + `secrets_store:` to inject API keys from Azure Key Vault or env vars |
| Global env | Use `globals.env` to set `LOG_LEVEL` or other shared settings |
| Worker options | Pass `--no-workers` to `llamphouse up` for API-only mode |
| Scaffold a new project | Run `llamphouse init` in an empty directory |
