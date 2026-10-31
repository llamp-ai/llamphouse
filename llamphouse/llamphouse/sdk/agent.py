"""The agent, the workflow, typed config, and the `@step` and `@job` decorators."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import inspect
import json
import typing
from typing import Any, Awaitable, Callable, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from .card import AgentInfo
from .events import AgentError
from .vocab import AgentType

class Config(BaseModel):
    """Typed instance config. Its JSON Schema is published, so the API and Compass validate
    an instance's config without importing the agent's code."""

    model_config = ConfigDict(extra="forbid", frozen=True)


ConfigT = TypeVar("ConfigT", bound=Config)
InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Agent(Generic[ConfigT]):
    """Subclass it and implement `run`. The name comes from the agent definition file; the version
    from the code (`version`), for reporting and logging. The description is the class's docstring,
    unless `description` is set - it's published (agent card, Compass), so keep notes for developers
    in comments."""

    type: ClassVar[AgentType | str] = AgentType.AGENT
    description: ClassVar[str | None] = None  # None: the class's own docstring
    version: ClassVar[str | None] = None  # e.g. os.getenv("AGENT_VERSION") or the package's version;
    # None: the runtime uses LLAMPHOUSE_AGENT_VERSION from the environment, else the run is unversioned

    async def run(self, ctx: Any) -> Any:
        """Return = run completed (a returned value is its output); raise AgentError = run failed.
        May also be an async generator: every yielded Event is emitted, and there is no output.
        `ctx` is a `RunContext[ConfigT]`.

        The return annotation declares the output: a type pydantic can store as JSON (a model,
        `list[str]`, …) is checked when the agent is loaded, validated when a run ends, published in
        `describe()` and given back typed by `wait()`. No annotation: any storable value. `-> None`:
        no output - returning a value fails the run."""
        raise NotImplementedError

    @classmethod
    def output_type(cls) -> Any:
        """What `run` returns: its return annotation; Any without one; NoneType for `-> None` (and for
        an async generator)."""
        if inspect.isasyncgenfunction(cls.run):
            return type(None)
        raw = getattr(cls.run, "__annotations__", {}).get("return", inspect.Parameter.empty)
        if raw is inspect.Parameter.empty or cls.run is Agent.run:
            return Any
        if isinstance(raw, str):
            try:
                raw = eval(raw, getattr(cls.run, "__globals__", {}))  # noqa: S307 - an annotation, from the agent's own code
            except Exception as e:
                raise TypeError(f"{cls.__name__}.run: can't resolve its return annotation {raw!r} ({e}); "
                                "import the type at module level, not under TYPE_CHECKING") from e
        return type(None) if raw is None else raw

    @classmethod
    def check_output(cls) -> None:
        """When the agent is loaded: its output type must be storable as JSON."""
        t = cls.output_type()
        if t is Any or t is type(None):
            return
        try:
            TypeAdapter(t).json_schema(mode="serialization")
        except Exception as e:
            raise TypeError(f"{cls.__name__}.run returns {t!r}, which can't be stored as JSON: {e}") from e

    @classmethod
    def output_schema(cls) -> dict[str, Any] | None:
        t = cls.output_type()
        return None if t is Any or t is type(None) else TypeAdapter(t).json_schema(mode="serialization")

    @classmethod
    def output_to_json(cls, value: Any) -> Any:
        """A returned value as the run record stores it. Raises AgentError: `output_undeclared`,
        `output_invalid` (not its declared type), `output_not_storable` (not JSON)."""
        t = cls.output_type()
        if t is type(None):
            if value is not None:
                raise AgentError("output_undeclared", f"{cls.__name__}.run is declared `-> None` but returned "
                                                      f"{type(value).__name__}: annotate it with its type, or `-> Any`")
            return None
        adapter = TypeAdapter(t)
        try:
            if t is not Any:
                value = adapter.validate_python(value)
            return json.loads(adapter.dump_json(value))
        except ValidationError as e:
            raise AgentError("output_invalid", f"{cls.__name__}.run returned something that isn't {t!r}: {e}") from e
        except Exception as e:
            raise AgentError("output_not_storable", f"{cls.__name__}.run returned something that can't be stored "
                                                    f"as JSON: {e}") from e

    @classmethod
    def output_from_json(cls, value: Any) -> Any:
        """A stored output, as `wait()` gives it back: typed, when the type is declared."""
        t = cls.output_type()
        return value if t is Any or t is type(None) else TypeAdapter(t).validate_python(value)

    @classmethod
    def config_class(cls) -> type[Config]:
        """The Config class from `Agent[MyConfig]`; `Config` when none is given."""
        for klass in cls.__mro__:
            for base in getattr(klass, "__orig_bases__", ()):
                origin = typing.get_origin(base)
                if isinstance(origin, type) and issubclass(origin, Agent):
                    for arg in typing.get_args(base):
                        if isinstance(arg, type) and issubclass(arg, Config):
                            return arg
        return Config

    @classmethod
    def description_text(cls) -> str:
        """`description` when set, else the class's own docstring (never one inherited from a base)."""
        if cls.description is not None:
            return cls.description
        return inspect.cleandoc(cls.__dict__.get("__doc__") or "")

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return cls.config_class().model_json_schema()

    @classmethod
    def describe(cls, name: str) -> AgentInfo:
        """What gets published about it. The name comes from the agent definition."""
        return AgentInfo(name=name, type=cls.type, description=cls.description_text(), version=cls.version,
                         requires=tuple(sorted(c.__name__ for c in requirements_of(cls))),
                         config_schema=cls.config_schema(), output_schema=cls.output_schema())


class Workflow(Agent[ConfigT], Generic[ConfigT, InputT, OutputT]):
    """An agent that is a durable process rather than a conversation. Same contract, same
    durability; Compass shows it as steps instead of chat, and the API starts it with typed input.

    Input and Output come from the type arguments, `Workflow[MyConfig, MyInput, MyOutput]`.
    `ctx.input.data` is the validated Input; the Output that `run` returns becomes the run's
    output."""

    type: ClassVar[AgentType | str] = AgentType.WORKFLOW
    Input: ClassVar[type[BaseModel]]
    Output: ClassVar[type[BaseModel]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for base in cls.__dict__.get("__orig_bases__", ()):
            origin = typing.get_origin(base)
            if isinstance(origin, type) and issubclass(origin, Workflow):
                args = typing.get_args(base)
                if len(args) == 3:
                    for name, arg in (("Input", args[1]), ("Output", args[2])):
                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            setattr(cls, name, arg)

    @classmethod
    def describe(cls, name: str) -> AgentInfo:
        return dataclasses.replace(super().describe(name), input_schema=cls.input_schema())

    @classmethod
    def output_type(cls) -> Any:
        """Its Output - and `run`'s return annotation, if it has one, must say the same."""
        annotated = super().output_type()
        if annotated is not Any and annotated is not cls.Output:
            raise TypeError(f"{cls.__name__}.run returns {annotated!r}, but its Output is {cls.Output.__name__}")
        return cls.Output

    @classmethod
    def input_schema(cls) -> dict[str, Any]:
        return cls.Input.model_json_schema()


    @classmethod
    def parse_input(cls, value: Any) -> BaseModel:
        """Validate a start request's input (a model, a dict or JSON text)."""
        if isinstance(value, cls.Input):
            return value
        if isinstance(value, (str, bytes)):
            return cls.Input.model_validate_json(value)
        return cls.Input.model_validate(value)


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def step(fn: F | None = None, *, name: str | None = None, retries: int = 0, backoff: float = 1.0,
         retry_on: tuple[type[BaseException], ...] = (Exception,)) -> Any:
    """`ctx.step` as a decorator, for agent methods: `@step async def fetch(self, ctx, query)`.
    The context is the first argument after `self`.

    `retries`: tried again in place when it raises one of `retry_on`, after `backoff` seconds, doubling
    each time - inside one execution, holding its worker (for long waits between attempts, loop over
    `ctx.sleep` in the run instead). What it returns, or what it raises in the end, is recorded: a
    replay gives back the same result or raises the same error, and never calls it again."""

    def wrap(f: F) -> F:
        step_name = name or f.__name__

        async def attempts(self: Any, ctx: Any, *args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries + 1):
                try:
                    result = f(self, ctx, *args, **kwargs)
                    return await result if inspect.isawaitable(result) else result
                except retry_on:
                    if attempt == retries:
                        raise
                    await asyncio.sleep(backoff * 2 ** attempt)

        @functools.wraps(f)
        async def wrapper(self: Any, ctx: Any, *args: Any, **kwargs: Any) -> Any:
            return await ctx.step(step_name, functools.partial(attempts, self, ctx), *args, **kwargs)

        wrapper.__llamphouse_step__ = step_name  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return wrap(fn) if fn is not None else wrap


A = TypeVar("A", bound=type)


def requires(*capabilities: type) -> Callable[[A], A]:
    """The capabilities an agent can't run without - the runtime refuses to start it where one is
    missing, at startup rather than mid-run:

        @requires(Files, Scheduler)
        class Reporter(Agent): ...

    Inherited, and added to: a subclass decorated with @requires(...) needs its parent's
    capabilities and its own. Child runs (`ctx.agents`) are always there and are not one."""
    from .capabilities import Agents

    for c in capabilities:
        if c is Agents:
            raise TypeError("child runs are always available (ctx.agents): @requires(Agents) is not needed")
        if not isinstance(c, type):
            raise TypeError(f"@requires takes capability classes (Files, Scheduler), got {c!r}")

    def wrap(cls: A) -> A:
        if not (isinstance(cls, type) and issubclass(cls, Agent)):
            raise TypeError(f"@requires decorates an Agent class, got {cls!r}")
        own = cls.__dict__.get(_REQUIRES, frozenset())  # this class's own; its parents' are read through the MRO
        setattr(cls, _REQUIRES, frozenset(own) | frozenset(capabilities))
        return cls

    return wrap


_REQUIRES = "__llamphouse_requires__"


def requirements_of(agent: type[Agent] | Agent) -> frozenset[type]:
    """The capabilities an agent class (or instance) declared with @requires - its own and its parents'.
    What the runtime checks at startup, and what describe() publishes. What an instance may use is
    the agent definition's `permissions`."""
    cls = agent if isinstance(agent, type) else type(agent)
    return frozenset().union(*(klass.__dict__.get(_REQUIRES, frozenset()) for klass in cls.__mro__))


def job(name: str) -> Callable[[F], F]:
    """A background job the scheduler runs: `@job("reindex") async def reindex(ctx: JobContext, index: str)`.
    The runtime finds it by this mark; the SDK keeps no registry."""

    def wrap(f: F) -> F:
        f.__llamphouse_job__ = name  # type: ignore[attr-defined]
        return f

    return wrap
