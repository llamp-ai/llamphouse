from __future__ import annotations

import asyncio
import inspect
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values
from pydantic import ValidationError

from ..core.assistant import Agent
from ..core.health import HealthCheckResult, HealthCheckStatus
from .config.loader import (
    ADAPTER_REGISTRY,
    DATA_STORE_REGISTRY,
    TRACING_STORE_REGISTRY,
    TRIGGER_REGISTRY,
    WORKER_REGISTRY,
    _load_entrypoint,
    load_config,
)
from .config.components import (
    apply_component_env_defaults,
    instantiate_from_registry,
    parse_component_entry,
    validate_registry_entry,
)


def _summary(results: Iterable[HealthCheckResult]) -> dict[str, int]:
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for result in results:
        counts[result.status.value] += 1
    return counts


def _overall_status(summary: dict[str, int]) -> str:
    if summary["fail"]:
        return "fail"
    if summary["warn"]:
        return "warn"
    return "pass"


def _result_to_dict(result: HealthCheckResult) -> dict:
    payload = asdict(result)
    payload["status"] = result.status.value
    return payload


def _format_validation_path(location: tuple) -> str:
    path = ""
    for part in location:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path = f"{path}.{part}" if path else str(part)
    return path or "<root>"


def _config_schema_failure(exc: Exception) -> HealthCheckResult:
    if isinstance(exc, ValidationError):
        errors = [
            {
                "path": _format_validation_path(tuple(error.get("loc", ()))),
                "message": error.get("msg", "Invalid value"),
            }
            for error in exc.errors(include_url=False)
        ]
        message = "Config schema validation failed"
        if errors:
            message += ": " + "; ".join(
                f"{error['path']}: {error['message']}" for error in errors
            )
        return HealthCheckResult.fail(
            "config.schema",
            "config",
            message,
            errors=errors,
        )
    return HealthCheckResult.fail(
        "config.schema",
        "config",
        f"Config validation failed: {exc}",
    )


def _load_check_dotenv(config_path: Path) -> None:
    dotenv_paths = [config_path.parent / ".env"]
    cwd_dotenv = Path.cwd() / ".env"
    if cwd_dotenv.resolve() != dotenv_paths[0].resolve():
        dotenv_paths.append(cwd_dotenv)

    for dotenv_path in dotenv_paths:
        if not dotenv_path.exists():
            continue
        for key, value in dotenv_values(dotenv_path).items():
            if value is not None:
                os.environ.setdefault(key, value)


def _check_components(config) -> HealthCheckResult:
    errors: list[str] = []

    component_groups = [
        ("adapter", ADAPTER_REGISTRY, config.adapters or []),
        ("worker", WORKER_REGISTRY, config.workers or []),
    ]
    if config.data_store is not None:
        component_groups.append(("data_store", DATA_STORE_REGISTRY, [config.data_store]))
    if config.tracing is not None:
        component_groups.append(("tracing_store", TRACING_STORE_REGISTRY, [config.tracing]))

    for kind, registry, entries in component_groups:
        for entry in entries:
            try:
                parsed = parse_component_entry(entry, kind, allow_none=True)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if parsed is None:
                continue
            name, kwargs = parsed
            error = validate_registry_entry(registry, name, kwargs, kind)
            if error:
                errors.append(error)

    for deployment in config.agents:
        for entry in deployment.triggers or []:
            try:
                parsed = parse_component_entry(entry, "trigger", allow_none=True)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if parsed is None:
                continue
            name, kwargs = parsed
            error = validate_registry_entry(TRIGGER_REGISTRY, name, kwargs, "trigger")
            if error:
                errors.append(error)
                continue
            try:
                trigger = instantiate_from_registry(TRIGGER_REGISTRY, name, kwargs, "trigger")
            except Exception as exc:
                errors.append(str(exc))
                continue
            secret_env = getattr(trigger, "secret_env", None)
            if secret_env and os.environ.get(secret_env) is None:
                errors.append(f"Webhook secret env '{secret_env}' is configured but not set.")

    if errors:
        return HealthCheckResult.fail(
            "config.components",
            "config",
            "; ".join(errors),
            errors=errors,
        )
    return HealthCheckResult.pass_(
        "config.components",
        "config",
        "Components validate",
    )


async def _run_component_health_check(component, timeout: float) -> HealthCheckResult | None:
    health_check = getattr(component, "health_check", None)
    if health_check is None:
        return None
    return await asyncio.wait_for(health_check(), timeout=timeout)


async def _check_data_store(config, timeout: float) -> HealthCheckResult:
    entry = parse_component_entry(config.data_store, "data_store", allow_none=True)
    if entry is None:
        return HealthCheckResult.pass_(
            "data_store.default",
            "data_store",
            "Using in_memory",
            backend="in_memory",
        )

    name, kwargs = entry
    if name not in DATA_STORE_REGISTRY:
        return HealthCheckResult.fail(
            f"data_store.{name}",
            "data_store",
            f"Unknown data_store '{name}'",
            available=sorted(DATA_STORE_REGISTRY),
        )
    if name in {"in_memory", "memory"}:
        return HealthCheckResult.pass_(
            "data_store.in_memory",
            "data_store",
            "No external dependency",
            backend="in_memory",
        )
    try:
        kwargs = apply_component_env_defaults(name, kwargs, "data_store")
        component = instantiate_from_registry(DATA_STORE_REGISTRY, name, kwargs, "data_store")
        result = await _run_component_health_check(component, timeout)
    except Exception as exc:
        return HealthCheckResult.fail(
            f"data_store.{name}",
            "data_store",
            f"Health check failed: {exc}",
            backend=name,
        )
    if result is not None:
        return result
    return HealthCheckResult.pass_(f"data_store.{name}", "data_store", "Configured", backend=name)


async def _check_tracing_store(config, timeout: float) -> HealthCheckResult:
    entry = parse_component_entry(config.tracing, "tracing_store", allow_none=True)
    if entry is None:
        return HealthCheckResult.pass_(
            "tracing.default",
            "tracing",
            "Not configured; tracing is optional",
        )

    name, kwargs = entry
    if name not in TRACING_STORE_REGISTRY:
        return HealthCheckResult.fail(
            f"tracing.{name}",
            "tracing",
            f"Unknown tracing store '{name}'",
            available=sorted(TRACING_STORE_REGISTRY),
        )
    if name in {"in_memory", "memory"}:
        return HealthCheckResult.pass_(
            "tracing.in_memory",
            "tracing",
            "No external dependency",
            backend="in_memory",
        )
    try:
        cls = TRACING_STORE_REGISTRY[name]
        dry_kwargs = dict(kwargs)
        if "ensure_table" in inspect.signature(cls.__init__).parameters:
            dry_kwargs.setdefault("ensure_table", False)
        component = instantiate_from_registry(TRACING_STORE_REGISTRY, name, dry_kwargs, "tracing_store")
        result = await _run_component_health_check(component, timeout)
    except Exception as exc:
        return HealthCheckResult.fail(
            f"tracing.{name}",
            "tracing",
            f"Health check failed: {exc}",
            backend=name,
        )
    if result is not None:
        return result
    return HealthCheckResult.pass_(f"tracing.{name}", "tracing", "Configured", backend=name)


def _normalize_path(path: str) -> str:
    return "/" + path.strip("/")


def _adapter_prefixes(config) -> list[str]:
    prefixes: list[str] = []
    if config.adapters is None:
        return prefixes
    for entry in config.adapters:
        try:
            name, kwargs = parse_component_entry(entry, "adapter")
        except ValueError:
            continue
        prefix = (kwargs or {}).get("prefix")
        if prefix:
            prefixes.append(_normalize_path(prefix))
    return prefixes


def _check_routes(config) -> HealthCheckResult:
    seen: dict[str, str] = {}
    adapter_prefixes = _adapter_prefixes(config)
    errors: list[str] = []

    for deployment in config.agents:
        for trigger_entry in deployment.triggers or []:
            try:
                trigger_name, kwargs = parse_component_entry(trigger_entry, "trigger")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if trigger_name != "webhook":
                continue

            raw_path = (kwargs or {}).get("path")
            if not raw_path:
                continue
            path = _normalize_path(raw_path)
            owner = deployment.name

            if path in seen:
                errors.append(
                    f"Duplicate webhook path '{path}' declared by '{owner}' and '{seen[path]}'."
                )
            else:
                seen[path] = owner

            for prefix in adapter_prefixes:
                if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
                    errors.append(
                        f"Webhook path '{path}' for '{owner}' falls under adapter prefix '{prefix}'."
                    )

    if errors:
        return HealthCheckResult.fail(
            "config.routes",
            "config",
            "; ".join(errors),
            errors=errors,
        )
    return HealthCheckResult.pass_(
        "config.routes",
        "config",
        "No route conflicts",
    )


def _check_entrypoints(config, config_dir: Path) -> HealthCheckResult:
    errors: list[str] = []
    for definition in config.definitions:
        try:
            entrypoint_obj = _load_entrypoint(definition.entrypoint, config_dir)
        except Exception as exc:
            errors.append(f"{definition.entrypoint}: {exc}")
            continue

        if (
            inspect.isclass(entrypoint_obj)
            and issubclass(entrypoint_obj, Agent)
        ):
            continue
        if inspect.iscoroutinefunction(entrypoint_obj):
            continue
        if callable(entrypoint_obj):
            continue
        errors.append(
            f"{definition.entrypoint}: entrypoint is not an Agent subclass, async function, or factory callable."
        )

    if errors:
        return HealthCheckResult.fail(
            "config.entrypoints",
            "config",
            "; ".join(errors),
            errors=errors,
        )
    return HealthCheckResult.pass_(
        "config.entrypoints",
        "config",
        "Entrypoints import successfully",
    )


async def _collect_results(config_path: Path, timeout: float) -> list[HealthCheckResult]:
    results: list[HealthCheckResult] = []

    try:
        config = load_config(config_path)
    except Exception as exc:
        results.append(_config_schema_failure(exc))
    else:
        results.append(
            HealthCheckResult.pass_(
                "config.schema",
                "config",
                "llamphouse.yaml is valid",
            )
        )
        try:
            results.append(_check_components(config))
            results.append(_check_entrypoints(config, config_path.parent))
            results.append(await _check_data_store(config, timeout))
            results.append(await _check_tracing_store(config, timeout))
            results.append(_check_routes(config))
        except Exception as exc:
            results.append(
                HealthCheckResult.fail(
                    "config.components",
                    "config",
                    f"Component validation failed: {exc}",
                )
            )
    return results


def run_check(
    config_path: str | Path,
    *,
    output_format: str = "text",
    verbose: bool = False,
    timeout: float = 5.0,
) -> int:
    config_path = Path(config_path).resolve()

    _load_check_dotenv(config_path)
    results = asyncio.run(_collect_results(config_path, timeout))

    summary = _summary(results)
    status = _overall_status(summary)

    if output_format == "json":
        print(
            json.dumps(
                {
                    "status": status,
                    "summary": summary,
                    "checks": [_result_to_dict(result) for result in results],
                },
                ensure_ascii=True,
            )
        )
    else:
        for result in results:
            print(f"{result.status.value.upper():<4} {result.name:<20} {result.message}")
            if verbose and result.details:
                print(f"     details: {json.dumps(result.details, ensure_ascii=True)}")
        print(
            f"Summary: {summary['pass']} passed, {summary['warn']} warned, {summary['fail']} failed"
        )

    return 1 if summary["fail"] else 0
