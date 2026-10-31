# Agents

An agent is a Python class. An agent file gives it a name and settings, and the runtime serves it under
that name. The same class can be served several times under different names, with different config.

## Agent

```python
from llamphouse.sdk import Agent


class Echo(Agent):
    """Says back what it was told."""

    async def run(self, ctx) -> None:
        await ctx.send(f"You said: {ctx.input.text}")
```

- `run(self, ctx)` does the run's work, with the [run context](context.md). Return to complete the
  run; raise to fail it. When a run waits and then continues, `run` is executed again from the top, on
  a new instance, and replays what it already did (see [Durable runs](durable-runs.md)): keep nothing
  on `self`.
- The class docstring is the agent's description. It's published (the API's agent list, Compass, an A2A
  agent card), so write it for callers; keep notes for developers in comments. Set `description = "..."`
  to publish something else.
- `version = "1.4.2"` is your code's version, shown on every run. Without it, the runtime uses
  `LLAMPHOUSE_AGENT_VERSION` from the environment.
- The class must be importable by module path: define it at module level, not inside a function. A
  run's record refers to it, and another worker loads it from there.

### Ending a run

| in `run` | the run ends |
|---|---|
| `return` (or `return value`) | `completed`, with that value as its output |
| `raise AgentError("quota_exceeded", "...", retryable=True)` | `failed`, with a code clients can act on |
| any other exception | `failed` |
| `await ctx.reject("I only answer questions about billing")` (or `raise Rejected(...)`) | `rejected`: the agent declined. Not a failure |
| `await ctx.handover("billing")` | `completed`; another agent takes over the thread (see [handover](multi-agent.md#handover)) |

## Config

An agent's settings are a typed `Config`, a pydantic model:

```python
from llamphouse.sdk import Agent, Config


class SupportConfig(Config):
    model: str = "gpt-5-mini"
    tone: str = "friendly"
    max_refunds: int = 0


class Support(Agent[SupportConfig]):
    """Answers support questions."""

    async def run(self, ctx) -> None:
        await ctx.send(f"({ctx.config.tone}) How can I help?")
```

`Agent[SupportConfig]` ties the config to the agent; `ctx.config` is a validated, frozen
`SupportConfig`. Unknown keys are refused. Its JSON Schema is published, so the API and Compass check
a config without importing your code.

A run's config is built in layers, each over the one before:

1. the class's defaults;
2. the agent file's `config:`;
3. the config store, if the deployment has one: changed at runtime, for every process, from their next
   run on (`PUT /v2/agents/{name}/config`, or the agent's page in Compass);
4. the thread's `config` (given when the thread was created), for runs on that thread only.

So one class serves many instances: `support-nl` and `support-de` in two agent files, or one agent and
a config per customer thread.

## Typed output

The return annotation of `run` declares its output:

```python
from pydantic import BaseModel


class Verdict(BaseModel):
    approved: bool
    reason: str


class Reviewer(Agent):
    """Reviews an expense claim."""

    async def run(self, ctx) -> Verdict:
        approved = "receipt" in ctx.input.text.lower()
        return Verdict(approved=approved, reason="has a receipt" if approved else "no receipt")
```

- Any type pydantic can store as JSON works: a model, `list[str]`, `dict[str, int]`, a union. It's
  checked when the agent loads: a type that can't be stored refuses to start.
- When the run ends, the value is validated against it. The wrong type fails the run (`output_invalid`).
- The schema is published, and a caller waiting on the run (`await thread.wait()` in another agent)
  gets the value back typed.
- `-> None` declares no output: returning a value fails the run. No annotation: any storable value.

## Workflow

A workflow is an agent that is a process rather than a conversation: typed input, typed output, steps.
Same contract, same durability; Compass shows it as steps instead of a chat, and the API starts it with
its input as data.

```python
from pydantic import BaseModel

from llamphouse.sdk import Config, Workflow, step


class PriceIn(BaseModel):
    city: str
    month: str


class PriceOut(BaseModel):
    city: str
    per_day: int


class Prices(Workflow[Config, PriceIn, PriceOut]):
    """Estimates the cost per person per day for a city and month."""

    @step
    async def lookup(self, ctx, city: str, month: str) -> int:
        return 140 if month in ("june", "july", "august") else 110

    async def run(self, ctx) -> PriceOut:
        q: PriceIn = ctx.input.data  # validated before the run started
        return PriceOut(city=q.city, per_day=await self.lookup(ctx, q.city, q.month))
```

- `Workflow[ConfigT, Input, Output]`. `ctx.input.data` is the validated `Input`; an input that doesn't
  validate never becomes a run (`422 invalid_input`).
- Start it with data: `POST /v2/threads/{id}/runs` with `{"data": {"city": "Lisbon", "month": "march"}}`,
  or from another agent with `await ctx.call("prices", {"city": "Lisbon", "month": "march"})`.
- What `run` returns is its `Output`. It's the run's output, and it's also put on the thread as an
  artifact named `output`.

See the [Workflow example](../../examples/12_Workflow).

## Capabilities an agent needs

Some features depend on what the deployment has: a file store (`Files`), a scheduler (`Scheduler`).
`ctx.capability(Files)` returns `None` where there is none. An agent that can't work without one says so,
and the runtime refuses to start it there instead of failing mid-run:

```python
from llamphouse.sdk import Agent, Files, requires


@requires(Files)
class Reporter(Agent):
    """Writes reports as files."""

    async def run(self, ctx) -> None:
        files = ctx.require(Files)  # guaranteed by @requires
        part = await files.put(b"city,cost\nLisbon,140\n", mime_type="text/csv", name="report.csv")
        await ctx.send("Here is the report.", part)
```

## The agent file

An agent file describes agents for a deployment. It travels with the agents and is the same in every
environment; where and how they run is the runtime file's job (see [Configuration](../configuration.md)).

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: support}               # the name callers use: in the API, in ctx.call, in handovers
spec:
  entrypoint: agents.py:Support         # a file next to this one (path.py:Class), or a module (pkg.mod:Class)
  display_name: Customer support        # for people; may change
  config: {model: gpt-5, tone: formal}  # the Config, checked when the deployment loads
  session: {rolling_window: {size: 30}} # the session provider (see Sessions)
  env: {LOG_LEVEL: info}                # set in the process environment
  secrets: {OPENAI_API_KEY: openai-key} # an env var from the runtime file's secrets_store
  permissions:
    agents: [billing]                   # whom it may call or hand over to; '*' (the default): anyone
    files: read                         # none | read | read-write (the default)
  triggers:
    - schedule: {cron: "0 8 * * 1-5", timezone: Europe/Amsterdam, input: "Morning digest"}
```

Several agents go in one file as YAML documents separated by `---`, or in several files (`--agents`
takes files, directories and `module:NAME` for a Python dict of agents). Values may use `${VAR}` and
`${VAR:-default}`.

To serve one class as several instances, define it once as an `AgentDefinition` and refer to it:

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: AgentDefinition
metadata: {name: hr-assistant}
spec:
  entrypoint: agents.hr:HrAssistant
---
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: hr-thailand}
spec:
  definition: hr-assistant
  display_name: HR Assistant Thailand
  config: {index: hr-th}
```

`llamphouse check --agents agents.yaml` loads the files, imports the classes and validates every config,
without starting anything.

Triggers (webhooks and schedules that start runs) are covered in the
[Triggers example](../../examples/11_Triggers); permissions in [Working with other agents](multi-agent.md#permissions).

## Testing

`llamphouse.sdk.testing.FakeRun` runs one agent against a scripted context: the answers to its questions,
the replies of the agents it calls, the state it finds.

```python
from llamphouse.sdk.testing import FakeRun


async def test_greeter_asks_once():
    run = FakeRun("hello", answers=["Ada"])
    await run.start(Greeter())
    assert run.text == "Hello, Ada! You said: hello"
```

`llamphouse.sdk.scenario.Scenario` runs real agents on the real engine, in memory, on a fake clock, with
crashes you inject: for testing how agents behave together and across failures.
