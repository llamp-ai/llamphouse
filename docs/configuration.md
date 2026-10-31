# Configuration

A llamphouse deployment is described by two kinds of YAML files:

- **Agent files** say what the agents are: which class, its config, its session provider, its triggers and
  permissions. They travel with the agents and are the same in every environment.
- **A runtime file** says where and how they run: the data store, queues, file store, tracing, auth,
  adapters, engine options. You keep one per environment (laptop, staging, production).

Both are plain YAML documents with `apiVersion: llamphouse.io/v1alpha1` and a `kind`. They can share one file
(separated by `---`) or be spread over several files and directories.

```bash
llamphouse check --config llamphouse.yaml --agents agents.yaml   # load everything, start nothing
llamphouse up    --config llamphouse.yaml --agents agents.yaml   # run it
```

`--config` defaults to `LLAMPHOUSE_CONFIG`, else `./llamphouse.yaml` when it exists. `--agents` can be given
several times and takes a file, a directory (every `.yaml` / `.yml` in it) or `module:NAME` (a Python mapping
of agent names to classes). Without `--agents`, the agent documents in the runtime file are used.

## Agent files

There are two kinds of agent document: an `AgentDefinition` (the code) and an `Agent` (an instance that callers
talk to by name). Most deployments only need `Agent` with an `entrypoint`.

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: AgentDefinition                  # an agent type: its code
metadata: {name: hr-assistant}
spec:
  entrypoint: agents.py:Assistant
---
apiVersion: llamphouse.io/v1alpha1
kind: Agent                            # an instance: what callers talk to, by its name
metadata: {name: hr-thailand}
spec:
  definition: hr-assistant             # or `entrypoint:` directly
  display_name: HR Assistant Thailand
  config: {index: hr-th, model: "${HR_MODEL:-gpt-5-mini}"}
  session: {summarizing: {summarize: {import: agents:summarize}, max_messages: 40, keep: 10}}
  env: {LOG_LEVEL: debug}
  secrets: {OPENAI_API_KEY: openai-key}
  permissions: {agents: [digest], files: read}
  publish:
    a2a:
      skills: [{id: hr, name: HR questions, description: Leave, payroll and policies}]
---
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: digest}
spec:
  entrypoint: agents.py:Digest
  triggers:
    - schedule: {cron: "0 7 * * MON-FRI", timezone: Europe/Brussels, input: {text: Daily digest}, thread: {label: digest}}
    - webhook: {path: /hooks/digest, secret_env: DIGEST_SECRET, message: {text: $.summary}}
```

`metadata.name` is the agent's name in the API, in Compass and in `ctx.call(...)`. It must be a lowercase slug
(`a-z`, `0-9`, `-`, `_`, `.`). One class can serve several instances, each with its own name and config.

### `spec` keys

| Key | What it does |
|---|---|
| `entrypoint` | The agent class: `module:Class` (imported by module path) or `path/to/file.py:Class` (relative to the agent file; its folder joins `sys.path`). The class must be importable at module level, never defined inside a function: every process finds it again by the same path. |
| `definition` | The name of an `AgentDefinition` to take the entrypoint from, instead of `entrypoint`. `name@version` is accepted; the version is ignored for now. |
| `display_name` | A name for people: shown in Compass, A2A cards and the model list of the Responses adapter. |
| `config` | The agent's `Config` values. Checked against the class's `Config` model when the file is loaded. See [Config](guides/config.md). |
| `session` | The session provider for this agent, overriding the runtime file's `sessions`: `{rolling_window: {size: 50}}`, `{summarizing: {summarize: {import: module:fn}, max_messages: 50, keep: 10}}`, your own `{module:Class: {...}}`, or `{provider: module:Class, config: {...}}`. |
| `env` | Environment variables set in the process before the agents load. An agent's value wins over the runtime file's. |
| `secrets` | Environment variables filled from secrets: `VAR: secret-name`, looked up in the runtime file's `secrets_store` (else an environment variable of that name). |
| `triggers` | Webhooks and schedules that start runs of this agent. See [Triggers](guides/triggers.md). |
| `permissions` | What this instance may use, enforced by the runtime: `agents` (`"*"` or a list of agent names it may call or hand over to; also what `ctx.agents.list()` shows) and `files` (`none`, `read`, `read-write`; default `read-write`). Forks of itself are always allowed. |
| `publish` | How protocols present the agent. Today: `publish.a2a` with `skills`, `input_modes`, `output_modes` and `extensions` for its A2A card. |
| `execution` | Accepted and logged as "not enforced yet". |
| `pool`, `mode`, `image`, `requirements` | Accepted and reserved for later; they change nothing today. |

An unknown key is an error, with the list of keys it accepts.

The description of an agent comes from its class docstring (or a `description` class attribute), and its
version from a `version` class attribute, else `LLAMPHOUSE_AGENT_VERSION`.

### Agents from Python

`--agents my_app:AGENTS` loads a mapping of names to classes (or `(class, config)` tuples) instead of YAML:

```python
AGENTS = {"desk": Desk, "weather": Weather}
```

## The runtime file

```yaml
apiVersion: llamphouse.io/v1alpha1
kind: Runtime
data_store:   {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}
file_store:   {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}
config_store: {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}
run_queue:    {redis: {url: "${LLAMPHOUSE_REDIS_URL:-redis://localhost:6379/0}"}}
event_queue:  {redis: {url: "${LLAMPHOUSE_REDIS_URL:-redis://localhost:6379/0}"}}
tracing:      {postgres: {database_url: "${LLAMPHOUSE_DATABASE_URL}"}}
sessions:     {rolling_window: {size: 50}}
auth:
  - api_key: {keys: {"${PARTNER_KEY}": {subject: partner, tenant: acme}}}
access: {tenant: {admin_subjects: [ops]}}
adapters:
  - responses: {prefix: /v1, agent: hr-thailand}
  - a2a: {prefix: /a2a}
engine: {workers: 8, lease_ttl: 30, max_depth: 20}
env: {LOG_LEVEL: info}
secrets: {OPENAI_API_KEY: openai-key}
secrets_store:
  openai-key: {provider: env, name: OPENAI_KEY_PROD}
```

Only one `Runtime` document is allowed per deployment. Every key is optional: what is left out is in memory
(see [Deployment](deployment.md) for where each component comes from when it isn't in the file).

### Top-level keys

| Key | What it is |
|---|---|
| `data_store` | Where threads, runs, messages and state live. The truth of the deployment. |
| `run_queue` | The job queue that tells workers a run can make progress. |
| `event_queue` | Pub/sub for live events (streams) between processes. |
| `file_store` | Where file bytes live. See [Files](guides/files.md). |
| `config_store` | Where config changes made at runtime are kept. See [Config](guides/config.md). |
| `tracing` | The tracing store spans are written to and read back from. See [Tracing](guides/tracing.md). |
| `sessions` | The default session provider for every agent (an agent's `session` overrides it). |
| `auth` | A list of authenticators, tried in order. See [Auth](guides/auth.md). |
| `access` | What an authenticated caller may touch. See [Auth](guides/auth.md). |
| `adapters` | A list of protocols mounted next to `/v2`. See [Adapters](guides/adapters.md). |
| `triggers` | Accepted, but not used here: triggers belong to an agent, in its agent file. |
| `engine` | Engine options (below). |
| `env` | Environment variables set in the process. |
| `secrets` | Environment variables filled from `secrets_store`, as in agent files. |
| `secrets_store` | Where secrets come from, by name: `{provider: env, name: OTHER_VAR}` reads another environment variable; `{provider: my_pkg.secrets:fetch, ...}` calls your function `fetch(name, **options) -> str`. |
| `scheduler`, `identity`, `notifications`, `remote_agents`, `topology` | Reserved: accepted, not used yet. |

### The component registry

Every component is one entry `{name: {options}}`. `name` is either a name from the registry or your own class
as `module:Class`. A bare name (`data_store: in_memory`) means no options. The options are checked against the
class's constructor before anything starts: an unknown or missing option is an error that names the options it
takes.

| Key | Name | Class | Options |
|---|---|---|---|
| `data_store` | `in_memory` | `MemoryStore` | none |
| | `postgres` | `PostgresStore` | `dsn`, `min_size` (2), `max_size` (10), `migrate` (true) |
| `run_queue` | `in_memory` | `MemoryQueue` | none |
| | `redis` | `RedisQueue` | `url`, `key` (`llamphouse:jobs`) |
| `event_queue` | `in_memory` | `MemoryBus` | none |
| | `redis` | `RedisBus` | `url`, `prefix` (`llamphouse:`) |
| `file_store` | `in_memory` | `InMemoryFileStore` | `upload_ttl` (3600 s) |
| | `postgres` | `PostgresFileStore` | `dsn`, `migrate` (true), `upload_ttl` (3600 s) |
| `config_store` | `in_memory` | `MemoryConfigStore` | none |
| | `postgres` | `PostgresConfigStore` | `dsn`, `migrate` (true) |
| `tracing` | `in_memory` | `InMemoryTracingStore` | none |
| | `postgres` | `PostgresTracingStore` | `database_url` |
| | `clickhouse` | `ClickHouseTracingStore` | `clickhouse_url` |
| `sessions` | `rolling_window` | `RollingWindow` | `size` (50) |
| | `summarizing` | `Summarizing` | `summarize` (your function, via `{import: ...}`), `max_messages` (50), `keep` (10) |
| `auth` (list) | `api_key` | `ApiKeyAuth` | `keys`, `header` (`authorization`), `query` (`key`) |
| | `oidc` | `OidcAuth` | `issuer`, `audience`, `jwks_url`, `algorithms`, `subject_claim`, `tenant_claim`, `metadata_claims`, `required_scopes`, `leeway`, `jwks_ttl`, `jwks` |
| `access` | `open` | `Access` | none (the default) |
| | `tenant` | `TenantAccess` | `admin_subjects` |
| `triggers` (agent file) | `webhook` | `WebhookTrigger` | see [Triggers](guides/triggers.md) |
| | `schedule` | `ScheduleTrigger` | see [Triggers](guides/triggers.md) |
| `adapters` (list) | `responses` | `ResponsesAdapter` | `prefix` (`/v1`), `agents` (`"*"`), `agent` |
| | `assistants` | `AssistantsAdapter` | `prefix` (`/v1`), `agents` (`"*"`), `agent` |
| | `a2a` | `A2AAdapter` | `prefix` (`/a2a`), `agents` (`"*"`), `provider`, `documentation_url`, `icon_url` |
| | `ag_ui` | `AgUiAdapter` | `prefix` (`/ag-ui`), `agents` (`"*"`), `agent` |

Your own class goes in the same place:

```yaml
file_store: {my_pkg.stores:S3FileStore: {bucket: "${BUCKET}", upload_ttl: 7200}}
auth:
  - my_pkg.auth:GatewayAuth: {header: x-user}
```

Backends that need an extra say which one to install when it is missing (`pip install 'llamphouse[postgres]'`,
`[redis]`, `[tracing]`, `[oidc]`, `[ag-ui]`, or `[all]` for postgres, redis, tracing and oidc).

### Engine options

`engine:` takes the `Engine`'s options. Unknown options are an error.

| Option | Default | Meaning |
|---|---|---|
| `workers` | 4 under `llamphouse up` (2 in code) | How many runs this process executes at once. `0` for a process that only serves the API (`--role api` sets it). Also `--workers` or `LLAMPHOUSE_WORKERS`. |
| `lease_ttl` | 30 | Seconds a run's lease lasts. It is renewed while the run executes; when a worker dies, another takes the run over once the lease expires. |
| `redeliver` | 30 | Seconds a queued run may wait for a worker before its job is rung again. |
| `tick` | 0.5 | Seconds between sweeps for due work: sleeps that end, expired leases, schedules, deliveries. |
| `window` | 50 | The size of the default `RollingWindow` session, when no `sessions` is given. |
| `upload_ttl` | 3600 | Seconds an unreferenced upload lives, for the default file store. |
| `max_runs_per_root` | 1000 | Against loops: how many runs one run started from outside may lead to. |
| `max_depth` | 50 | Against loops: how deep runs may start runs. |
| `max_runs_per_thread` | none | A cap on runs per thread, if you want one. |
| `cache_ttl` | 300 | Seconds a process keeps its copy of a conversation nobody is using. |
| `delivery_attempts` | 8 | How many times a subscription delivery is tried before it is dead. |

## Values

Every string value in either file may use environment variables:

- `${VAR}`: the variable's value. If it is not set, loading fails and says which one.
- `${VAR:-default}`: the variable's value, or `default` when it is not set.

Mapping keys are expanded too (useful for API keys as keys of `keys:`).

A value written as `{import: module:attr}` is that Python object, for options that take a function or another
object:

```yaml
session: {summarizing: {summarize: {import: my_sessions:summarize}, max_messages: 40}}
```

## Checking before you deploy

`llamphouse check` loads the files the same way `llamphouse up` does: it imports every entrypoint, validates each
agent's config against its `Config`, checks that its output type can be stored as JSON, builds every component
and prints what it found. It does not open the data store or the queues (the Postgres tracing store is the one
exception: it tries to create its table when it is built, and logs an error if it can't reach the database).

```text
$ llamphouse check --config llamphouse.yaml --agents agents.yaml
agent    hr-thailand              agents:Assistant  session=Summarizing
agent    digest                   agents:Digest
data_store PostgresStore
run_queue RedisQueue
...
ok
```

A problem exits with status 2 and a message that says which file, which agent and what is wrong.

## See also

- [Deployment](deployment.md): processes, roles, Postgres and Redis
- [Config](guides/config.md): an agent's `Config`, per-thread config and changes at runtime
- [Example 10_Server](../examples/10_Server): agent and runtime files for a real deployment
