"""
Config loader for ``llamphouse.yaml``.

Responsibilities
----------------
1. Parse and validate the YAML file into a ``LLAMPHouseConfig``.
2. Resolve secrets (env look-up today; pluggable providers in the future).
3. Dynamically load agent entrypoints (class, async function, or factory).
4. Instantiate one ``Agent`` per deployment and inject ``agent.settings``.
5. Attach deployment triggers and instantiate runtime infrastructure.
6. Return a ready-to-run ``LLAMPHouse`` instance.

Entrypoint formats
------------------
``file.py:ClassName``   – an ``Agent`` subclass; instantiated directly.
``file.py:run``         – an ``async def run(context)`` function;
                          wrapped in a dynamic ``Agent`` subclass.
``file.py:create``      – a sync factory ``(deployment_cfg: dict) -> Agent``;
                          called with the deployment's merged config dict.

Module-style references (``mymodule:MyAgent``) are also supported — the
directory that contains ``llamphouse.yaml`` is prepended to ``sys.path``.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from ...core.assistant import Agent
from ...core.context import Context
from ...core.adapters import AssistantAPIAdapter, A2AAdapter, CompassAdapter
from ...core.data_stores import InMemoryDataStore, PostgresDataStore
from ...core.triggers import WebhookTrigger
from ...core.workers import AsyncWorker
from ...core.tracing.stores import InMemoryTracingStore, PostgresTracingStore, ClickHouseTracingStore
from .components import (
    apply_component_env_defaults,
    instantiate_from_registry,
    parse_component_entry,
)
from .schema import DeploymentConfig, LLAMPHouseConfig

logger = logging.getLogger("llamphouse.config")

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ── Component registries ──────────────────────────────────────────────────────
# Maps the YAML name (lower-case) to the real class.
# Kwargs are validated against the class ``__init__`` signature at load time —
# no separate Pydantic config models are required.

ADAPTER_REGISTRY: Dict[str, type] = {
    "assistant_api": AssistantAPIAdapter,
    "a2a": A2AAdapter,
    "compass": CompassAdapter,
}

WORKER_REGISTRY: Dict[str, type] = {
    "asyncworker": AsyncWorker,
}

TRACING_STORE_REGISTRY: Dict[str, type] = {
    "in_memory": InMemoryTracingStore,
    "memory": InMemoryTracingStore,
    "postgres": PostgresTracingStore,
    "clickhouse": ClickHouseTracingStore,
}

DATA_STORE_REGISTRY: Dict[str, type] = {
    "in_memory": InMemoryDataStore,
    "memory": InMemoryDataStore,
    "postgres": PostgresDataStore,
}

TRIGGER_REGISTRY: Dict[str, type] = {
    "webhook": WebhookTrigger,
}


def _instantiate_from_registry(
    registry: Dict[str, type],
    name: str,
    kwargs: Dict[str, Any],
    kind: str,
) -> Any:
    """
    Look up ``name`` in ``registry``, validate ``kwargs`` against the class
    ``__init__`` signature, and return an instance.

    Parameters
    ----------
    registry : mapping of name → class
    name     : key to look up in the registry
    kwargs   : constructor keyword arguments from the YAML
    kind     : human-readable label ("adapter" or "worker") for error messages
    """
    return instantiate_from_registry(registry, name, kwargs, kind)


def _parse_component_entry(entry: Any, kind: str) -> tuple[str, Dict[str, Any]]:
    """
    Parse a single ``{name: kwargs}`` entry from the adapters/workers list.

    Returns ``(name, kwargs_dict)``.
    """
    parsed = parse_component_entry(entry, kind)
    assert parsed is not None
    return parsed


# ── YAML loading ─────────────────────────────────────────────────────────────


def _expand_env_refs(value: Any, path: str = "") -> Any:
    """Recursively expand ``${ENV_VAR}`` references in YAML values."""
    if isinstance(value, dict):
        return {
            key: _expand_env_refs(child, f"{path}.{key}" if path else str(key))
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _expand_env_refs(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        env_value = os.environ.get(env_name)
        if env_value is None:
            location = path or "<root>"
            raise ValueError(
                f"Environment variable '{env_name}' referenced at '{location}' is not set."
            )
        return env_value

    return _ENV_REF_RE.sub(replace, value)


def load_config(config_path: str | Path) -> LLAMPHouseConfig:
    """Parse and validate a ``llamphouse.yaml`` file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"'{config_path}' must contain a YAML mapping at the top level.")

    raw = _expand_env_refs(raw)
    return LLAMPHouseConfig.model_validate(raw)


# ── Secret resolution ────────────────────────────────────────────────────────


def _resolve_secret(ref: str, secrets_store: dict | None, env_var_name: str) -> str | None:
    """
    Look up a secret value.

    Resolution order:
    1. The env var is already set → use it (caller-defined wins).
    2. ``secrets_store`` entry with ``provider: env`` → read from the
       env var named by ``config.name``.
    3. Other providers (azure_keyvault, …) → log a warning and skip;
       the env var will be left unset.

    Returns the resolved value, or ``None`` if it could not be resolved.
    """
    # 1. Already in the environment
    existing = os.environ.get(env_var_name)
    if existing is not None:
        return existing

    if secrets_store is None or ref not in secrets_store:
        logger.warning(
            "Secret '%s' for env var '%s' is not defined in secrets_store — skipping.",
            ref,
            env_var_name,
        )
        return None

    provider_cfg = secrets_store[ref]
    provider = provider_cfg.provider

    if provider == "env":
        # Read the secret value from another env var
        value = os.environ.get(provider_cfg.name)
        if value is None:
            logger.warning(
                "secrets_store['%s'] references env var '%s' which is not set — skipping.",
                ref,
                provider_cfg.name,
            )
        return value

    # Future providers: azure_keyvault, aws_secretsmanager, hashicorp_vault, …
    logger.warning(
        "Secret provider '%s' is not yet supported (secret '%s') — "
        "set the '%s' environment variable manually.",
        provider,
        ref,
        env_var_name,
    )
    return None


def _apply_secrets(
    secrets_mapping: dict[str, str] | None,
    secrets_store: dict | None,
) -> None:
    """Set env vars for all entries in a ``secrets`` mapping."""
    if not secrets_mapping:
        return
    for env_var_name, secret_ref in secrets_mapping.items():
        value = _resolve_secret(secret_ref, secrets_store, env_var_name)
        if value is not None:
            os.environ[env_var_name] = value


# ── Dynamic entrypoint loading ───────────────────────────────────────────────


def _load_entrypoint(entrypoint: str, base_dir: str | Path) -> Any:
    """
    Import the object at ``entrypoint`` (``file:name`` or ``module:name``).

    ``base_dir`` is used to resolve relative file paths and is prepended to
    ``sys.path`` for module-style references.
    """
    if ":" not in entrypoint:
        raise ValueError(
            f"Entrypoint '{entrypoint}' must use 'file_or_module:name' format "
            "(e.g. 'agent.py:ResearchAgent' or 'mymodule:ResearchAgent')."
        )

    module_ref, attr_name = entrypoint.rsplit(":", 1)
    base_dir = Path(base_dir)

    if module_ref.endswith(".py"):
        # File-based import — does not pollute sys.modules with collisions
        file_path = (base_dir / module_ref).resolve()
        if not file_path.exists():
            raise FileNotFoundError(
                f"Agent entrypoint file not found: {file_path}"
            )
        spec = importlib.util.spec_from_file_location(
            f"_llamphouse_agent_{file_path.stem}", file_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    else:
        # Module-based import
        base_str = str(base_dir)
        if base_str not in sys.path:
            sys.path.insert(0, base_str)
        module = importlib.import_module(module_ref)

    obj = getattr(module, attr_name, None)
    if obj is None:
        raise AttributeError(
            f"Entrypoint '{entrypoint}': module has no attribute '{attr_name}'."
        )
    return obj


def _make_agent(entrypoint_obj: Any, deployment: DeploymentConfig) -> Agent:
    """
    Turn an entrypoint object into a concrete ``Agent`` instance.

    Supported entrypoint types
    --------------------------
    * ``Agent`` subclass     → instantiated with ``id=deployment.name``.
    * ``async def run(ctx)`` → wrapped in a dynamic ``Agent`` subclass.
    * Factory callable       → called with the merged deployment config dict
                               and must return an ``Agent`` instance.
    """
    settings: Dict[str, Any] = deployment.config or {}

    # ── Agent subclass ──────────────────────────────────────────────────────
    if inspect.isclass(entrypoint_obj) and issubclass(entrypoint_obj, Agent):
        instance = entrypoint_obj(id=deployment.name, name=deployment.name)
        instance.settings = settings
        instance.execution_config = deployment.execution
        return instance

    # ── Async run function ──────────────────────────────────────────────────
    if inspect.iscoroutinefunction(entrypoint_obj):
        _run_fn = entrypoint_obj  # capture for closure

        async def _run(self: Agent, context: Context) -> None:  # type: ignore[override]
            await _run_fn(context)

        DynamicAgent = type(
            f"Agent_{deployment.name.replace('-', '_')}",
            (Agent,),
            {"run": _run},
        )
        instance = DynamicAgent(id=deployment.name, name=deployment.name)
        instance.settings = settings
        instance.execution_config = deployment.execution
        return instance

    # ── Factory function ────────────────────────────────────────────────────
    if callable(entrypoint_obj):
        deployment_cfg = {
            "name": deployment.name,
            "config": settings,
            "env": deployment.env or {},
            "secrets": deployment.secrets or {},
        }
        result = entrypoint_obj(deployment_cfg)
        if not isinstance(result, Agent):
            raise TypeError(
                f"Factory '{entrypoint_obj.__name__}' must return an Agent instance, "
                f"got {type(result).__name__}."
            )
        if not hasattr(result, "settings"):
            result.settings = settings
        result.execution_config = deployment.execution
        return result

    raise TypeError(
        f"Entrypoint object {entrypoint_obj!r} is not an Agent subclass, "
        "async function, or factory callable."
    )


# ── Main builder ─────────────────────────────────────────────────────────────


def build_app_from_config(
    config: LLAMPHouseConfig,
    config_dir: str | Path,
) -> "LLAMPHouse":  # noqa: F821 — imported lazily to avoid circular import
    """
    Build a ``LLAMPHouse`` instance from a validated ``LLAMPHouseConfig``.

    ``config_dir`` is the directory that contains ``llamphouse.yaml``; it is
    used to resolve relative entrypoint paths.
    """
    from ...core.llamphouse import LLAMPHouse  # local import avoids circular dep

    config_dir = Path(config_dir)
    secrets_store_raw = config.secrets_store  # Dict[str, SecretProviderConfig] | None

    # ── 1. Apply global env vars ────────────────────────────────────────────
    if config.globals and config.globals.env:
        for key, value in config.globals.env.items():
            os.environ.setdefault(key, value)

    # ── 2. Apply global secrets ─────────────────────────────────────────────
    if config.globals:
        _apply_secrets(config.globals.secrets, secrets_store_raw)

    # ── 3. Index agent definitions ──────────────────────────────────────────
    agent_defs = {a.name: a for a in config.definitions}

    # ── 4. Instantiate one Agent per deployment ─────────────────────────────
    agents = []
    for deployment in config.agents:
        agent_def = agent_defs[deployment.definition]  # validated in schema

        # Apply deployment-level env vars (override globals)
        if deployment.env:
            for key, value in deployment.env.items():
                os.environ[key] = value

        # Apply deployment-level secrets
        _apply_secrets(deployment.secrets, secrets_store_raw)

        # Load and instantiate
        logger.info(
            "Loading deployment '%s' ← agent '%s' (%s)",
            deployment.name,
            deployment.definition,
            agent_def.entrypoint,
        )
        entrypoint_obj = _load_entrypoint(agent_def.entrypoint, config_dir)
        agent = _make_agent(entrypoint_obj, deployment)
        if deployment.triggers is not None:
            agent.triggers = []
            for entry in deployment.triggers:
                name, kwargs = _parse_component_entry(entry, "trigger")
                trigger = _instantiate_from_registry(TRIGGER_REGISTRY, name, kwargs, "trigger")
                agent.triggers.append(trigger)
        agents.append(agent)

    project_name = config.project.name if config.project else "LLAMPHouse"
    logger.info(
        "Project '%s' — %d deployment(s) loaded.",
        project_name,
        len(agents),
    )

    # ── 5. Build adapters ───────────────────────────────────────────────────
    # None → pass None to LLAMPHouse so it falls back to its own defaults.
    # Explicit list (even []) → instantiate and pass through.
    adapters = None
    if config.adapters is not None:
        adapters = []
        for entry in config.adapters:
            name, kwargs = _parse_component_entry(entry, "adapter")
            adapter = _instantiate_from_registry(ADAPTER_REGISTRY, name, kwargs, "adapter")
            logger.info("Adapter '%s' registered (prefix=%r).", name, getattr(adapter, "prefix", ""))
            adapters.append(adapter)

    # ── 6. Build worker ─────────────────────────────────────────────────────
    worker = None
    if config.workers:
        if len(config.workers) > 1:
            logger.warning(
                "Multiple workers specified in llamphouse.yaml — only the first will be used."
            )
        name, kwargs = _parse_component_entry(config.workers[0], "worker")
        worker = _instantiate_from_registry(WORKER_REGISTRY, name, kwargs, "worker")
        logger.info("Worker '%s' configured.", name)

    # ── 7. Build tracing store ──────────────────────────────────────────────
    # None → LLAMPHouse falls back to env-var auto-detection.
    tracing_store = None
    if config.tracing is not None:
        name, kwargs = _parse_component_entry(config.tracing, "tracing_store")
        tracing_store = _instantiate_from_registry(TRACING_STORE_REGISTRY, name, kwargs, "tracing_store")
        logger.info("Tracing store '%s' configured from YAML.", name)

    # โ”€โ”€ 8. Build data store โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€โ”€
    # None -> LLAMPHouse uses its default InMemoryDataStore.
    data_store = None
    if config.data_store is not None:
        name, kwargs = _parse_component_entry(config.data_store, "data_store")
        kwargs = apply_component_env_defaults(name, kwargs, "data_store")
        data_store = _instantiate_from_registry(DATA_STORE_REGISTRY, name, kwargs, "data_store")
        logger.info("Data store '%s' configured from YAML.", name)

    return LLAMPHouse(
        agents=agents,
        adapters=adapters,
        worker=worker,
        tracing_store=tracing_store,
        data_store=data_store,
    )
