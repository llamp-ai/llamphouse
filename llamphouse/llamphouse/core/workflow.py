"""Workflow primitives for LLAMPHouse.

This module provides the :func:`step` decorator, the first piece of a
workflow layer on top of the existing agent runtime. A ``@step``-decorated
function is recorded in the data store as a ``RunStepObject`` of type
``"step"``: its ``input`` and ``output`` are persisted, and the run step
transitions through ``in_progress`` → ``completed`` (or ``failed``).

The current ``Agent.run`` method is conceptually the ``@workflow`` — the
durable unit that owns a ``Run``. Each ``@step`` invoked inside it becomes
a child ``RunStep`` checkpoint that can be inspected, replayed or visualised
from the persisted run history.

Example::

    class MyAgent(Agent):
        @step
        async def fetch_data(self, context, query: str) -> dict:
            return await some_api.search(query)

        async def run(self, context):
            data = await self.fetch_data(context, "llamphouse")
            await context.reply(str(data))
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import traceback
from typing import Any, Callable, Optional

from .context import Context


def _find_context(args: tuple, kwargs: dict) -> Optional[Context]:
    """Locate a ``Context`` instance in a call's arguments.

    Searches positional args first (skipping a leading ``self``) then
    keyword arguments. Returns ``None`` when no context is found, in which
    case the decorated function still runs but no run step is recorded.
    """
    for value in args:
        if isinstance(value, Context):
            return value
    for value in kwargs.values():
        if isinstance(value, Context):
            return value
    return None


def _build_input_snapshot(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> dict:
    """Bind args/kwargs to parameter names and drop ``self`` and the context."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        snapshot: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            if name == "self":
                continue
            if isinstance(value, Context):
                continue
            snapshot[name] = value
        return snapshot
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": dict(kwargs)}


def step(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    timeout: Optional[float] = None,
):
    """Decorator that records a function call as a workflow run step.

    Wraps an async or sync function so that each invocation:

    1. Creates a ``RunStepObject`` of type ``"step"`` in ``in_progress`` state,
       capturing the call arguments as ``step_details.input``.
    2. Executes the wrapped function (optionally bounded by ``timeout``).
    3. Transitions the step to a terminal status:

       - ``completed`` on success — return value stored in ``step_details.output``.
       - ``failed`` on any ``Exception`` — error recorded in ``last_error``.
       - ``cancelled`` on ``asyncio.CancelledError`` — propagated to the caller.
       - ``expired`` on ``asyncio.TimeoutError`` (only when ``timeout`` is set) —
         re-raised as ``TimeoutError`` so callers can react.

    A ``Context`` instance must be present in the call (positionally or as a
    keyword) for persistence to occur. If no context is found the function
    still executes — useful for unit tests.

    :param name: Optional human-readable step name. Defaults to ``func.__qualname__``.
    :param timeout: Optional wall-clock limit in seconds. When exceeded, the
        step is marked ``expired`` and a ``TimeoutError`` is raised.
    """

    def decorator(func: Callable) -> Callable:
        step_name = name or getattr(func, "__qualname__", func.__name__)
        is_coro = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            ctx = _find_context(args, kwargs)
            if ctx is None:
                if timeout is not None:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                return await func(*args, **kwargs)

            input_snapshot = _build_input_snapshot(func, args, kwargs)
            run_step = await ctx.start_step(name=step_name, input=input_snapshot)
            step_id = run_step.id if run_step else None
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    result = await func(*args, **kwargs)
            except asyncio.TimeoutError:
                if step_id:
                    await ctx.complete_step(
                        step_id,
                        output=None,
                        error=f"Step '{step_name}' exceeded timeout of {timeout}s",
                        status="expired",
                    )
                raise TimeoutError(f"Step '{step_name}' exceeded timeout of {timeout}s")
            except asyncio.CancelledError:
                if step_id:
                    # Best-effort: persist cancellation before re-raising.
                    # Shielded so the data-store write isn't itself cancelled.
                    try:
                        await asyncio.shield(
                            ctx.complete_step(step_id, output=None, status="cancelled")
                        )
                    except asyncio.CancelledError:
                        pass
                raise
            except Exception as exc:
                if step_id:
                    err = f"{type(exc).__name__}: {exc}"
                    await ctx.complete_step(step_id, output=None, error=err)
                raise
            if step_id:
                await ctx.complete_step(step_id, output=result)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            ctx = _find_context(args, kwargs)
            if ctx is None:
                return func(*args, **kwargs)

            input_snapshot = _build_input_snapshot(func, args, kwargs)

            async def _run():
                run_step = await ctx.start_step(name=step_name, input=input_snapshot)
                step_id = run_step.id if run_step else None
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    if step_id:
                        err = f"{type(exc).__name__}: {exc}"
                        await ctx.complete_step(step_id, output=None, error=err)
                    raise
                if step_id:
                    await ctx.complete_step(step_id, output=result)
                return result

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_run())
            # Inside a running loop, returning a coroutine forces the caller
            # to await — which is fine for sync helpers called from async code.
            return _run()

        return async_wrapper if is_coro else sync_wrapper

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator


__all__ = ["step"]
