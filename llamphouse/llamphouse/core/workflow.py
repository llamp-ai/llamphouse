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
from .types.durable import WorkflowSuspended
from .types.enum import run_step_status


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
    retries: int = 0,
    backoff: float = 1.0,
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

    A run that runs again (resumed after a wait, retried, taken over after
    its worker died) does not run a step twice: a call that completed in an
    earlier attempt returns the output it stored then - as JSON data, so a
    step's return value should be JSON-serializable. See ``Context``'s notes
    on durable runs.

    A ``Context`` instance must be present in the call (positionally or as a
    keyword) for persistence to occur. If no context is found the function
    still executes — useful for unit tests.

    :param name: Optional human-readable step name. Defaults to ``func.__qualname__``.
    :param timeout: Optional wall-clock limit in seconds. When exceeded, the
        step is marked ``expired`` and a ``TimeoutError`` is raised.
    :param retries: How many times to run the step again after it raised an
        ``Exception`` (each attempt is its own failed step) before raising.
    :param backoff: Seconds before the first retry; doubled for each next one.
    """

    def decorator(func: Callable) -> Callable:
        step_name = name or getattr(func, "__qualname__", func.__name__)
        is_coro = asyncio.iscoroutinefunction(func)

        async def _call(args, kwargs):
            if not is_coro:
                return func(*args, **kwargs)
            if timeout is not None:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            return await func(*args, **kwargs)

        async def _run_recorded(ctx: Context, args, kwargs):
            call_key = ctx._call_key(step_name)
            prior = ctx._replayed(call_key)
            if prior is not None and prior.status == run_step_status.COMPLETED:
                return prior.step_details.output  # done by an earlier attempt
            if prior is not None and prior.status == run_step_status.IN_PROGRESS:
                # The attempt that started it stopped mid-step: run it again.
                await ctx.complete_step(prior.id, error="Interrupted: the run stopped during this step")

            input_snapshot = _build_input_snapshot(func, args, kwargs)
            attempt = 0
            while True:
                run_step = await ctx.start_step(
                    name=step_name,
                    input=input_snapshot,
                    call_key=call_key,
                    metadata={"attempt": attempt + 1} if retries else None,
                )
                step_id = run_step.id if run_step else None
                try:
                    result = await _call(args, kwargs)
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
                except WorkflowSuspended:
                    # The run waits inside this step: it is not done, and it
                    # runs again (its waits replay) when the run is resumed.
                    raise
                except Exception as exc:
                    if step_id:
                        err = f"{type(exc).__name__}: {exc}"
                        await ctx.complete_step(step_id, output=None, error=err)
                    if attempt >= retries:
                        raise
                    await asyncio.sleep(backoff * (2 ** attempt))
                    attempt += 1
                    continue
                if step_id:
                    await ctx.complete_step(step_id, output=result)
                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            ctx = _find_context(args, kwargs)
            if ctx is None:
                return await _call(args, kwargs)
            return await _run_recorded(ctx, args, kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            ctx = _find_context(args, kwargs)
            if ctx is None:
                return func(*args, **kwargs)

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_run_recorded(ctx, args, kwargs))
            # Inside a running loop, returning a coroutine forces the caller
            # to await — which is fine for sync helpers called from async code.
            return _run_recorded(ctx, args, kwargs)

        return async_wrapper if is_coro else sync_wrapper

    if _func is not None and callable(_func):
        return decorator(_func)
    return decorator


__all__ = ["step"]
