from __future__ import annotations

import inspect
import os
from typing import Any


def parse_component_entry(
    entry: Any,
    kind: str,
    *,
    allow_none: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    if entry is None:
        if allow_none:
            return None
        raise ValueError(
            f"Each {kind} entry must be a single-key mapping "
            f"(e.g. ``- compass:``), got {entry!r}."
        )
    if not isinstance(entry, dict) or len(entry) != 1:
        raise ValueError(
            f"Each {kind} entry must be a single-key mapping "
            f"(e.g. ``- compass:``), got {entry!r}."
        )

    name, raw_kwargs = next(iter(entry.items()))
    if raw_kwargs is None:
        return name, {}
    if not isinstance(raw_kwargs, dict):
        raise ValueError(
            f"{kind.capitalize()} '{name}' arguments must be a mapping, "
            f"got {type(raw_kwargs).__name__}."
        )
    return name, raw_kwargs


def validate_registry_entry(
    registry: dict[str, type],
    name: str,
    kwargs: dict[str, Any],
    kind: str,
) -> str | None:
    cls = registry.get(name)
    if cls is None:
        return f"Unknown {kind} '{name}'. Available {kind}s: {sorted(registry)}."

    sig = inspect.signature(cls.__init__)
    valid_params = {key for key in sig.parameters if key != "self"}
    invalid = set(kwargs) - valid_params
    if invalid:
        return (
            f"{kind.capitalize()} '{name}' does not accept argument(s) "
            f"{sorted(invalid)}. Valid parameters: {sorted(valid_params)}."
        )
    return None


def instantiate_from_registry(
    registry: dict[str, type],
    name: str,
    kwargs: dict[str, Any],
    kind: str,
) -> Any:
    error = validate_registry_entry(registry, name, kwargs, kind)
    if error:
        raise ValueError(error)

    cls = registry[name]
    return cls(**kwargs)


def apply_component_env_defaults(
    name: str,
    kwargs: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    if kind == "data_store" and name == "postgres":
        if kwargs:
            raise ValueError(
                "data_store.postgres reads DATABASE_URL from the environment; "
                "remove data_store.postgres.database_url."
            )
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError(
                "data_store.postgres requires DATABASE_URL."
            )
        return {**kwargs, "database_url": database_url}
    return kwargs
