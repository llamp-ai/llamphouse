"""Lightweight ``@tool`` decorator for the PlannerAgent example.

Converts an annotated Python function into an object that carries both its
callable and an auto-generated OpenAI tool schema. Nothing framework-specific
here — copy this file into any project you like.

Usage::

    from tools import tool, collect_tools
    from typing import Annotated

    @tool
    def search_web(query: Annotated[str, "Search query"]) -> str:
        \"\"\"Search the web for information about a topic.\"\"\"
        return ...

    @tool
    async def fetch_url(url: Annotated[str, "URL to fetch"]) -> dict:
        \"\"\"Fetch JSON from a URL.\"\"\"
        ...

    schemas, registry = collect_tools(search_web, fetch_url)
    # schemas  → list of OpenAI-compatible dicts  (pass to PlannerAgent tools=)
    # registry → {"search_web": <Tool>, ...}       (pass to tool_registry=)
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from typing import Any, Callable, get_type_hints


# ── Python → JSON Schema type mapping ────────────────────────────────────────

_PY_TO_JSON: dict[Any, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
    list:  "array",
    dict:  "object",
}


def _resolve_annotation(annotation: Any) -> tuple[str, str]:
    """Return ``(json_type, description)`` for a single parameter annotation."""
    description = ""

    # Annotated[X, "description", ...]
    if typing.get_origin(annotation) is typing.Annotated:
        args = typing.get_args(annotation)
        annotation = args[0]
        for meta in args[1:]:
            if isinstance(meta, str):
                description = meta
                break

    # Optional[X] = Union[X, None]
    if typing.get_origin(annotation) is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if non_none:
            annotation = non_none[0]

    return _PY_TO_JSON.get(annotation, "string"), description


# ── Tool ──────────────────────────────────────────────────────────────────────

class Tool:
    """Wrapper produced by ``@tool``.  Behaves like the original function,
    but also exposes ``.schema`` (the OpenAI tool dict) and ``.acall(**kwargs)``
    which handles both sync and async implementations.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__wrapped__ = fn
        self.schema: dict = self._build_schema()

    def _build_schema(self) -> dict:
        sig = inspect.signature(self._fn)
        try:
            hints = get_type_hints(self._fn, include_extras=True)
        except Exception:
            hints = {}

        description = next(
            (l.strip() for l in (self._fn.__doc__ or "").splitlines() if l.strip()),
            "",
        )

        properties: dict[str, dict] = {}
        required: list[str] = []

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue
            json_type, param_desc = _resolve_annotation(hints.get(name, str))
            prop: dict[str, Any] = {"type": json_type}
            if param_desc:
                prop["description"] = param_desc
            properties[name] = prop
            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    async def acall(self, **kwargs) -> Any:
        """Call the tool — awaits async tools, runs sync tools in a thread."""
        if asyncio.iscoroutinefunction(self._fn):
            return await self._fn(**kwargs)
        return await asyncio.to_thread(self._fn, **kwargs)

    def __repr__(self) -> str:
        return f"<Tool {self.__name__}>"


# ── Public API ────────────────────────────────────────────────────────────────

def tool(fn: Callable) -> Tool:
    """Decorator — wrap a function as a :class:`Tool` with an auto-generated schema.

    Use ``Annotated[type, "description"]`` for per-parameter descriptions::

        @tool
        def get_weather(
            city:  Annotated[str, "City name"],
            units: Annotated[str, "celsius or fahrenheit"] = "celsius",
        ) -> dict:
            \"\"\"Return the current weather for a city.\"\"\"
            ...
    """
    return Tool(fn)


def collect_tools(*tools: Tool) -> tuple[list[dict], dict[str, Tool]]:
    """Return ``(schemas, registry)`` ready to pass to ``PlannerAgent``::

        schemas, registry = collect_tools(search_web, get_weather, calculate)
    """
    return [t.schema for t in tools], {t.__name__: t for t in tools}
