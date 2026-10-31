# Config

An agent's settings (which model, which index, which instructions) are its **Config**: a typed model declared in
code, filled in by the agent file, changeable at runtime for everyone, and overridable per thread. One agent
class can serve many instances, each with its own config.

## Declaring it

```python
from llamphouse.sdk import Agent, Config


class AssistantConfig(Config):
    model: str = "gpt-5-mini"
    index: str = "default"
    instructions: str = "You answer HR questions briefly."
    max_turns: int = 8


class Assistant(Agent[AssistantConfig]):
    """Answers HR questions."""

    async def run(self, ctx) -> None:
        model = ctx.config.model          # typed: an AssistantConfig
        ...
```

`Config` is a pydantic model that forbids unknown fields and is frozen. Its JSON Schema is published
(`GET /v2/agents`, Compass), so the API and Compass validate an instance's config without importing your code.
An agent without a type argument has an empty `Config`.

## Values from the agent file

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: hr-thailand}
spec:
  entrypoint: agents.py:Assistant
  config: {index: hr-th, model: "${HR_MODEL:-gpt-5}"}
---
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: hr-belgium}
spec:
  entrypoint: agents.py:Assistant
  config: {index: hr-be, instructions: "You help Belgian staff. Answer in the language of the question."}
```

Two instances of one class, each with its own name and config. The values are validated against the `Config`
when the files load, so a typo fails `llamphouse check` and startup, not a run. Fields left out keep their
defaults.

## Per thread

A thread can carry config of its own, for runs of its agent on it: one agent instance, configured per customer,
per user, per conversation.

```bash
curl -s localhost:8000/v2/threads -H 'content-type: application/json' \
     -d '{"agent": "hr-thailand", "config": {"instructions": "Answer in Thai."}}'
```

In code: `await engine.thread("hr-thailand", config={"instructions": "Answer in Thai."})`.

- The values are validated together with the agent's config when the thread is made (`422` when they don't
  fit).
- Only the values given are kept, so the agent file's and the store's still apply to everything else.
- A fork of the thread inherits its config. After a handover to another agent, it no longer applies (it was
  for the first agent).
- `GET /v2/threads/{id}` shows it as `config`.

## Changed at runtime

An agent's config can be changed while the deployment runs, for every process, from their next run on. The
change is kept in the **config store**:

```yaml
# llamphouse.yaml
config_store: {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}
```

Without a runtime file entry, the config store follows the data store: Postgres when the process has a
database (`LLAMPHOUSE_DATABASE_URL`), so changes are shared by every process and kept over restarts; in memory
otherwise (one process, lost on restart). `config_store: {postgres: {dsn: ...}}` in the runtime file names one
explicitly.

Over HTTP:

```bash
curl -s localhost:8000/v2/agents/hr-thailand/config
# {"agent": "hr-thailand", "config": {...as runs get it...}, "changed": {}, "version": 0, "schema": {...}, ...}

curl -s -X PUT localhost:8000/v2/agents/hr-thailand/config -H 'content-type: application/json' \
     -d '{"values": {"model": "gpt-5", "max_turns": 4}, "if_version": 0}'
# {"config": {..., "model": "gpt-5", "max_turns": 4}, "changed": {"model": "gpt-5", "max_turns": 4}, "version": 1, ...}
```

- `values` replace what the store held for the agent; `{"values": {}}` goes back to the agent file's config.
- The result is validated against the `Config` before it is kept (`422` otherwise).
- `if_version` makes it a compare-and-set: `409 config_conflict` when someone changed it since you read it.
- Every process hears about the change at once (over the event queue) and uses it from its next run on. A
  process that misses the news reads the store again within 30 seconds.
- Who may change it is the access policy's decision: anyone authenticated under the default access, only the
  `admin_subjects` under tenant access (an agent serves every tenant). See [Auth](auth.md).

In code: `await engine.set_agent_config("hr-thailand", {"model": "gpt-5"}, by="ops", if_version=0)` and
`config, stored = await engine.agent_config("hr-thailand")`.

In [Compass](compass.md), an agent's page shows its effective config, what differs from the file, and a form to
change or reset it.

## Which value a run gets

For each run, `ctx.config` is, field by field, the first of:

1. the thread's config (if the thread has one for this agent);
2. the config store's values for the agent (changed at runtime);
3. the agent file's `config`;
4. the `Config` class's defaults.

The config is read when the run executes; a run that executes again after a wait sees a change made meanwhile.

## Testing

```python
from llamphouse.sdk.testing import FakeRun

run = FakeRun("How many leave days do I have?", config=AssistantConfig(index="hr-test"))
await run.start(Assistant())
```

## See also

- [Example 08_Config](../../examples/08_Config)
- [Configuration](../configuration.md): every key of the agent and runtime files
