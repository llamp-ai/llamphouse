"""
Pydantic models for ``llamphouse.yaml``.

A llamphouse.yaml file has this top-level structure:

    version: "0.1"

    project:
      name: my-platform

    agents:          # agent *types* — reusable definitions
      - name: research-agent
        entrypoint: agent.py:ResearchAgent   # Agent subclass
        runtime: ...
        interface: ...

    deployments:     # instances of an agent type
      - name: research-fast
        agent: research-agent
        config:      # passed to the agent as agent.settings
          model: gpt-4o-mini
        env:         # injected into os.environ for this deployment
          MODEL: gpt-4o-mini
        secrets:     # ENVVAR_NAME: secret-store-ref
          OPENAI_API_KEY: openai-key
        execution:   # concurrency / timeout / retries
          timeout: 30
          retries: 2
          concurrency: 5

    adapters:        # API adapters to mount (uses adapter class constructors directly)
      - assistant_api:
      - compass:
          prefix: /dashboard

    workers:         # worker to run (first entry wins; kwargs map to class __init__)
      - asyncworker:
          time_out: 90

    tracing:         # tracing store (single entry; omit for env-var auto-detection)
      in_memory:     # or: postgres / clickhouse (with their respective kwargs)

    globals:
      env: {LOG_LEVEL: info}
      secrets: {OPENAI_API_KEY: openai-key}

    secrets_store:
      openai-key:
        provider: azure_keyvault
        name: my-openai-key

Each adapter/worker entry is a single-key mapping: ``{name: {kwargs}}``.
The kwargs are validated against the real class constructor — no separate
config models are needed.

Adapters available by default: ``assistant_api``, ``a2a``, ``compass``,
``dashboard``.
Workers available: ``asyncworker``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


# ── Low-level building blocks ────────────────────────────────────────────────


class ProjectConfig(BaseModel):
    name: str


class SchemaProperty(BaseModel):
    type: str
    description: Optional[str] = None


class SchemaDefinition(BaseModel):
    type: str = "object"
    properties: Optional[Dict[str, SchemaProperty]] = None
    required: Optional[List[str]] = None


class InterfaceConfig(BaseModel):
    input_schema: Optional[SchemaDefinition] = None
    output_schema: Optional[SchemaDefinition] = None


class RuntimeConfig(BaseModel):
    python: Optional[str] = None
    requirements: Optional[List[str]] = None


# ── Agent type definition ────────────────────────────────────────────────────


class AgentDefinition(BaseModel):
    """Defines a reusable agent *type*.

    ``entrypoint`` uses the same ``file:name`` convention as uvicorn:
    - ``agent.py:ResearchAgent``  – an ``Agent`` subclass
    - ``agent.py:run``            – an ``async def run(context)`` function
                                    (auto-wrapped in an Agent class)
    - ``agent.py:create_agent``   – a factory ``(deployment_cfg) -> Agent``
    """

    name: str
    entrypoint: str
    interface: Optional[InterfaceConfig] = None
    runtime: Optional[RuntimeConfig] = None


# ── Deployment (instance) ────────────────────────────────────────────────────


class ResourceConfig(BaseModel):
    name: str
    type: str
    provider: str
    config: Optional[Dict[str, Any]] = None


class DeploymentContextConfig(BaseModel):
    identity: Optional[str] = None
    resources: Optional[List[ResourceConfig]] = None


class ExecutionConfig(BaseModel):
    timeout: Optional[float] = None
    retries: Optional[int] = None
    concurrency: Optional[int] = None


class DeploymentConfig(BaseModel):
    """A concrete deployment — one running instance of an agent type."""

    name: str
    agent: str  # must match an AgentDefinition.name

    # Passed to the agent instance as ``agent.settings``
    config: Optional[Dict[str, Any]] = None

    # Merged into os.environ before the agent module is loaded
    env: Optional[Dict[str, str]] = None

    # Maps env-var names → keys in ``secrets_store``
    secrets: Optional[Dict[str, str]] = None

    context: Optional[DeploymentContextConfig] = None
    execution: Optional[ExecutionConfig] = None


# ── Globals & secret stores ──────────────────────────────────────────────────


class GlobalsConfig(BaseModel):
    # Applied to os.environ before any deployment env vars
    env: Optional[Dict[str, str]] = None
    # Maps env-var names → keys in ``secrets_store``
    secrets: Optional[Dict[str, str]] = None


class SecretProviderConfig(BaseModel):
    provider: str   # e.g. "azure_keyvault", "env"
    name: str       # the name/key within the provider


# ── Top-level config ─────────────────────────────────────────────────────────


class LLAMPHouseConfig(BaseModel):
    version: str
    project: Optional[ProjectConfig] = None

    agents: List[AgentDefinition] = []
    deployments: List[DeploymentConfig] = []

    # Each entry is a single-key mapping ``{adapter_name: {kwargs}}``.
    # ``None`` means "use LLAMPHouse defaults" (AssistantAPIAdapter + Compass).
    # An explicit list (even empty) replaces the defaults entirely.
    adapters: Optional[List[Dict[str, Any]]] = None

    # First entry wins; subsequent entries are ignored with a warning.
    # ``None`` means use the default AsyncWorker.
    workers: Optional[List[Dict[str, Any]]] = None

    # Single-key mapping ``{store_name: {kwargs}}``, e.g. ``{in_memory: {}}``,
    # ``{postgres: {db_url: ...}}``, or ``{clickhouse: {url: ...}}``.
    # ``None`` → auto-detected from TRACING_STORE / CLICKHOUSE_URL env vars.
    tracing: Optional[Dict[str, Any]] = None

    globals: Optional[GlobalsConfig] = None
    secrets_store: Optional[Dict[str, SecretProviderConfig]] = None

    @model_validator(mode="after")
    def _check_deployment_agent_refs(self) -> "LLAMPHouseConfig":
        agent_names = {a.name for a in self.agents}
        for dep in self.deployments:
            if dep.agent not in agent_names:
                raise ValueError(
                    f"Deployment '{dep.name}' references unknown agent '{dep.agent}'. "
                    f"Defined agents: {sorted(agent_names)}"
                )
        return self
