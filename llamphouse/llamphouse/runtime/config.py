"""Deployment files: what agents there are (agent files) and where and how they run (a runtime file).

Agent files - they travel with the agents, the same in every environment:

    apiVersion: llamphouse.io/v1alpha1
    kind: AgentDefinition                     # an agent type: its code
    metadata: {name: hr-assistant}
    spec:
      entrypoint: agents.hr:HrAssistant       # module:Class, or path/to/file.py:Class (next to this file)
      pool: shared
    ---
    apiVersion: llamphouse.io/v1alpha1
    kind: Agent                               # an instance: what callers talk to, by its name
    metadata: {name: hr-thailand}
    spec:
      definition: hr-assistant                # or `entrypoint:` directly
      display_name: HR Assistant Thailand
      config: {index: hr-th}                  # the agent's Config
      session: {summarizing: {summarize: {import: my_sessions:summarize}, max_messages: 40}}
      env: {MODEL: gpt-5}                     # set in the process environment
      secrets: {OPENAI_API_KEY: openai-key}   # env var: a secret from the runtime file's secrets_store

A runtime file - one per environment:

    apiVersion: llamphouse.io/v1alpha1
    kind: Runtime
    data_store: {postgres: {dsn: ${LLAMPHOUSE_DATABASE_URL}}}
    config_store: {postgres: {dsn: ${LLAMPHOUSE_DATABASE_URL}}}
    auth: [{api_key: {keys: ["${LLAMPHOUSE_API_KEY}"]}}]
    access: {tenant: {}}
    run_queue:  {redis: {url: ${LLAMPHOUSE_REDIS_URL}}}
    event_queue: {redis: {url: ${LLAMPHOUSE_REDIS_URL}}}
    file_store: {postgres: {dsn: ${LLAMPHOUSE_DATABASE_URL}}}
    tracing:    {postgres: {database_url: ${LLAMPHOUSE_DATABASE_URL}}}
    sessions:   {rolling_window: {size: 50}}  # the default provider; an Agent's `session` overrides it
    engine:     {workers: 8, lease_ttl: 30}   # Engine options
    env: {LOG_LEVEL: info}
    secrets: {OPENAI_API_KEY: openai-key}
    secrets_store:
      openai-key: {provider: env, name: OPENAI_KEY_PROD}

Every component is `{name: {kwargs}}`: a name from the registry below, or your own class by import path
(`{my_pkg.stores:MyStore: {...}}`); its kwargs are checked against the class's constructor before anything
starts. Values may use `${VAR}` and `${VAR:-default}`; a value `{import: module:attr}` is that Python object
(a function, say). Agent and runtime documents may share a file, or be split over several (`--agents` takes
files, directories and `module:NAME` for a Python mapping of agents).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

log = logging.getLogger("llamphouse.runtime.config")

API_VERSION = "llamphouse.io/v1alpha1"

# kind -> name -> "module:Class": swap any of them by name, or give your own by import path
REGISTRY: dict[str, dict[str, str]] = {
    "data_store": {"in_memory": "llamphouse.runtime.store:MemoryStore",
                   "postgres": "llamphouse.runtime.store:PostgresStore"},
    "run_queue": {"in_memory": "llamphouse.runtime.queue:MemoryQueue",
                  "redis": "llamphouse.runtime.queue:RedisQueue"},
    "event_queue": {"in_memory": "llamphouse.runtime.bus:MemoryBus",
                    "redis": "llamphouse.runtime.bus:RedisBus"},
    "file_store": {"in_memory": "llamphouse.runtime.files:InMemoryFileStore",
                   "postgres": "llamphouse.runtime.files.postgres:PostgresFileStore"},
    "tracing": {"in_memory": "llamphouse.runtime.tracing.stores:InMemoryTracingStore",
                "postgres": "llamphouse.runtime.tracing.stores:PostgresTracingStore",
                "clickhouse": "llamphouse.runtime.tracing.stores:ClickHouseTracingStore"},
    "sessions": {"rolling_window": "llamphouse.runtime.sessions:RollingWindow",
                 "summarizing": "llamphouse.runtime.sessions:Summarizing"},
    "config_store": {"in_memory": "llamphouse.runtime.config_store:MemoryConfigStore",
                     "postgres": "llamphouse.runtime.config_store:PostgresConfigStore"},
    "auth": {"api_key": "llamphouse.runtime.auth:ApiKeyAuth",
             "oidc": "llamphouse.runtime.auth:OidcAuth"},
    "access": {"open": "llamphouse.runtime.auth:Access",
               "tenant": "llamphouse.runtime.auth:TenantAccess"},
}
LISTS = {"auth"}  # kinds given as a list of entries

ENGINE_OPTIONS = {"workers", "lease_ttl", "redeliver", "window", "upload_ttl", "max_runs_per_root", "max_depth",
                  "max_runs_per_thread", "tick", "cache_ttl"}


class ConfigError(ValueError):
    """A deployment file that can't be used: the message says where and why."""


# ── values ───────────────────────────────────────────────────────────────────

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand(value: Any, where: str = "") -> Any:
    """`${VAR}` and `${VAR:-default}` in every string; `{import: module:attr}` as that object."""
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"{where or 'a value'}: ${{{name}}} is not set (give a default: ${{{name}:-...}})")
        if (whole := _ENV.fullmatch(value)) and whole.group(1) not in os.environ and whole.group(2) is None:
            raise ConfigError(f"{where or 'a value'}: ${{{whole.group(1)}}} is not set")
        return _ENV.sub(sub, value)
    if isinstance(value, Mapping):
        if set(value) == {"import"}:
            return import_object(str(value["import"]), where)
        return {expand(k, where) if isinstance(k, str) else k: expand(v, f"{where}.{k}" if where else str(k))
                for k, v in value.items()}  # keys too: an API key is often one
    if isinstance(value, list):
        return [expand(v, f"{where}[{i}]") for i, v in enumerate(value)]
    return value


def import_object(path: str, where: str = "", base: Path | None = None) -> Any:
    """`module:attr`, or `path/to/file.py:attr` (relative to `base`): the file's folder joins `sys.path`
    and it is imported by its name, so every process finds its classes again by the same module path."""
    target, _, attr = path.rpartition(":")
    if not target or not attr:
        raise ConfigError(f"{where or path}: expected module:attr or file.py:attr, got {path!r}")
    try:
        if target.endswith(".py"):
            file = Path(target)
            if not file.is_absolute() and base is not None:
                file = base / file
            file = file.resolve()
            if not file.exists():
                raise ConfigError(f"{where or path}: no file {file}")
            if str(file.parent) not in sys.path:
                sys.path.insert(0, str(file.parent))
            module = importlib.import_module(file.stem)
        else:
            if base is not None and str(base) not in sys.path:
                sys.path.insert(0, str(base))
            module = importlib.import_module(target)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"{where or path}: can't import {target!r}: {e}") from e
    obj: Any = module
    for part in attr.split("."):
        if not hasattr(obj, part):
            raise ConfigError(f"{where or path}: {target!r} has no {attr!r}")
        obj = getattr(obj, part)
    return obj


# ── components ───────────────────────────────────────────────────────────────


def component(kind: str, entry: Any, where: str | None = None, *, base: Path | None = None) -> Any:
    """`{name: {kwargs}}` (or just `name`) for a kind: the component, its kwargs checked first."""
    where = where or kind
    if isinstance(entry, str):
        entry = {entry: {}}
    if not isinstance(entry, Mapping) or len(entry) != 1:
        raise ConfigError(f"{where}: expected one entry {{name: {{options}}}}, got {entry!r}")
    (name, kwargs), = entry.items()
    kwargs = expand(dict(kwargs or {}), f"{where}.{name}")
    if not isinstance(kwargs, dict):
        raise ConfigError(f"{where}.{name}: options must be a mapping")
    known = REGISTRY.get(kind, {})
    if name in known:
        cls = import_object(known[name], f"{where}.{name}")
    elif ":" in str(name):
        cls = import_object(str(name), f"{where}.{name}", base)
    else:
        raise ConfigError(f"{where}: unknown {kind} {name!r} - one of {sorted(known)}, or a class as module:Class")
    check_kwargs(cls, kwargs, f"{where}.{name}")
    try:
        return cls(**kwargs)
    except Exception as e:
        raise ConfigError(f"{where}.{name}: {e}") from e


def check_kwargs(cls: Any, kwargs: Mapping[str, Any], where: str) -> None:
    sig = inspect.signature(cls)
    params = sig.parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return
    valid = {p.name for p in params if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)}
    unknown = set(kwargs) - valid
    if unknown:
        raise ConfigError(f"{where}: unknown option(s) {sorted(unknown)} - it takes {sorted(valid)}")
    missing = [p.name for p in params if p.default is p.empty and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
               and p.name not in kwargs]
    if missing:
        raise ConfigError(f"{where}: missing option(s) {missing}")


# ── documents ────────────────────────────────────────────────────────────────


@dataclass
class AgentSpec:
    """An agent instance, as its file says: what the engine serves under `name`."""

    name: str
    cls: type
    config: dict[str, Any] = field(default_factory=dict)
    display_name: str | None = None
    session: Any = None  # its session provider
    env: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    pool: str = "shared"
    triggers: list[Any] = field(default_factory=list)
    permissions: dict[str, Any] = field(default_factory=dict)
    publish: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class RuntimeSpec:
    """A runtime file: the components, and the engine's options."""

    components: dict[str, Any] = field(default_factory=dict)  # kind -> its entry, built by `Deployment.build`
    engine: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    secrets: dict[str, str] = field(default_factory=dict)
    secrets_store: dict[str, Any] = field(default_factory=dict)
    rest: dict[str, Any] = field(default_factory=dict)  # auth, adapters, ... (read by their phases)
    source: str = ""


AGENT_SPEC_KEYS = {"definition", "entrypoint", "display_name", "config", "session", "env", "secrets", "pool",
                   "triggers", "permissions", "publish", "execution", "mode", "image", "requirements"}
RUNTIME_KEYS = {*REGISTRY, "engine", "env", "secrets", "secrets_store", "topology", "adapters", "scheduler",
                "identity", "notifications", "remote_agents"}


def read_documents(paths: Iterable[str | Path]) -> list[tuple[dict, Path]]:
    """Every YAML document in these files (and in .yaml / .yml files in these directories)."""
    import yaml

    out = []
    for p in paths:
        path = Path(p)
        files = sorted([*path.glob("*.yaml"), *path.glob("*.yml")]) if path.is_dir() else [path]
        for f in files:
            if not f.exists():
                raise ConfigError(f"no file {f}")
            for i, doc in enumerate(yaml.safe_load_all(f.read_text())):
                if doc is None:
                    continue
                if not isinstance(doc, dict):
                    raise ConfigError(f"{f} (document {i + 1}): expected a mapping")
                out.append((doc, f))
    return out


@dataclass
class Deployment:
    """What runs: the agents, and the runtime they run on."""

    agents: dict[str, AgentSpec] = field(default_factory=dict)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)

    @classmethod
    def load(cls, config: str | Path | None = None, agents: Iterable[str] | str | None = None) -> "Deployment":
        """From a runtime file and agent files, directories or `module:NAME` (a Python mapping of agents).
        The runtime file may hold agent documents too."""
        found = cls()
        sources = [config] if config else []
        python_specs = []
        for a in ([agents] if isinstance(agents, str) else list(agents or [])):
            if _is_python_spec(a):
                python_specs.append(a)
            else:
                sources.append(a)
        definitions: dict[str, tuple[dict, Path]] = {}
        instances: list[tuple[dict, Path]] = []
        runtimes = 0
        for doc, path in read_documents(sources):
            where = f"{path}"
            kind = doc.get("kind")
            version = doc.get("apiVersion", API_VERSION)
            if version != API_VERSION:
                raise ConfigError(f"{where}: apiVersion {version!r} - this llamphouse reads {API_VERSION!r}")
            if kind == "Runtime":
                runtimes += 1
                if runtimes > 1:
                    raise ConfigError(f"{where}: a second Runtime document - one per deployment")
                found.runtime = _runtime(doc, path)
            elif kind == "AgentDefinition":
                definitions[_name(doc, where)] = (doc, path)
            elif kind == "Agent":
                instances.append((doc, path))
            else:
                raise ConfigError(f"{where}: unknown kind {kind!r} - Runtime, AgentDefinition or Agent")
        for doc, path in instances:
            spec = _agent(doc, path, definitions)
            if spec.name in found.agents:
                raise ConfigError(f"{path}: agent {spec.name!r} is defined twice")
            found.agents[spec.name] = spec
        for spec_path in python_specs:
            mapping = import_object(spec_path, spec_path, Path.cwd())
            for name, spec in cls.from_mapping(dict(mapping)).agents.items():
                if name in found.agents:
                    raise ConfigError(f"{spec_path}: agent {name!r} is defined twice")
                spec.source = spec_path
                found.agents[name] = spec
        return found

    @classmethod
    def from_mapping(cls, agents: Mapping[str, Any], runtime: RuntimeSpec | None = None) -> "Deployment":
        """From a Python mapping of agents: name -> Agent class, or (class, its Config)."""
        found = cls(runtime=runtime or RuntimeSpec())
        for name, value in agents.items():
            agent_cls, cfg = value if isinstance(value, tuple) else (value, None)
            found.agents[name] = AgentSpec(name=name, cls=agent_cls, config=cfg.model_dump() if cfg is not None
                                           else {}, source="python")
        return found

    def authenticator(self, *, api_key: str | None = None) -> Any:
        """The runtime file's `auth` (in order), and an extra API key (the command line's, LLAMPHOUSE_API_KEY)."""
        from .auth import ApiKeyAuth, Authenticator

        auths = [component("auth", entry, f"{self.runtime.source}: auth[{i}]")
                 for i, entry in enumerate(self.runtime.components.get("auth") or [])]
        if api_key:
            auths.append(ApiKeyAuth([api_key]))
        return Authenticator(auths)

    def access(self) -> Any:
        from .auth import Access

        entry = self.runtime.components.get("access")
        return component("access", entry, f"{self.runtime.source}: access") if entry else Access()

    def apply_environment(self) -> None:
        """The env and secrets the files ask for, into this process's environment (agents' after the
        runtime's: an agent's own value wins)."""
        store = self.runtime.secrets_store
        _set_env(self.runtime.env, self.runtime.secrets, store, "runtime")
        for spec in self.agents.values():
            _set_env(spec.env, spec.secrets, store, f"agent {spec.name}")

    def build(self, *, role: str = "all", **overrides: Any) -> Any:
        """The engine for a process in this role (`api` executes nothing). `overrides`: Engine options,
        and components given as objects (store=..., queue=..., ...) - for tests and embedding."""
        from .engine import Engine

        self.apply_environment()
        components = self.runtime.components
        options: dict[str, Any] = {**self.runtime.engine}
        built = {
            "store": ("data_store", None), "queue": ("run_queue", None), "bus": ("event_queue", None),
            "file_store": ("file_store", None), "tracing_store": ("tracing", None),
            "config_store": ("config_store", None),
        }
        for option, (kind, _) in built.items():
            if option in overrides:
                options[option] = overrides.pop(option)
            elif kind in components:
                options[option] = component(kind, components[kind], f"{self.runtime.source}: {kind}")
        if "sessions" in components:
            default = component("sessions", components["sessions"], f"{self.runtime.source}: sessions")
        else:
            default = None
        per_agent = {name: spec.session for name, spec in self.agents.items() if spec.session is not None}
        if per_agent:
            from .sessions import RollingWindow

            options["sessions"] = {**{n: default or RollingWindow(options.get("window", 50)) for n in self.agents},
                                   **per_agent}
        elif default is not None:
            options["sessions"] = default
        options.update(overrides)
        if role == "api":
            options["workers"] = 0
        permissions = {}
        for name, spec in self.agents.items():
            if spec.permissions:
                from .engine import Permissions

                try:
                    permissions[name] = Permissions.parse(spec.permissions)
                except ValueError as e:
                    raise ConfigError(f"{spec.source}: agent {name!r}: {e}") from e
                unknown = [a for a in (permissions[name].agents or ()) if a not in self.agents]
                if unknown:
                    raise ConfigError(f"{spec.source}: agent {name!r}: permissions.agents names unknown agent(s) "
                                      f"{unknown}")
        if permissions:
            options.setdefault("permissions", permissions)
        agents = {}
        for name, spec in self.agents.items():
            config_cls = spec.cls.config_class() if hasattr(spec.cls, "config_class") else None
            if config_cls is not None:
                try:
                    agents[name] = (spec.cls, config_cls.model_validate(spec.config))
                except Exception as e:
                    raise ConfigError(f"{spec.source}: agent {name!r} config: {e}") from e
            else:
                agents[name] = spec.cls
        return Engine(agents, **options)


def _is_python_spec(value: str) -> bool:
    if ":" not in value or value.endswith((".yaml", ".yml")):
        return False
    target = value.rpartition(":")[0]
    return not Path(value).exists() and not target.endswith((".yaml", ".yml"))


def _name(doc: dict, where: str) -> str:
    name = (doc.get("metadata") or {}).get("name")
    if not name or not re.fullmatch(r"[a-z0-9]([a-z0-9-_.]*[a-z0-9])?", str(name)):
        raise ConfigError(f"{where}: metadata.name must be a lowercase slug (a-z, 0-9, -, _, .), got {name!r}")
    return str(name)


def _runtime(doc: dict, path: Path) -> RuntimeSpec:
    where = str(path)
    unknown = set(doc) - RUNTIME_KEYS - {"apiVersion", "kind", "metadata"}
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)} in Runtime")
    spec = RuntimeSpec(source=where)
    for kind in REGISTRY:
        if doc.get(kind) is not None:
            spec.components[kind] = doc[kind]
    engine = expand(doc.get("engine") or {}, f"{where}: engine")
    unknown = set(engine) - ENGINE_OPTIONS
    if unknown:
        raise ConfigError(f"{where}: engine: unknown option(s) {sorted(unknown)} - {sorted(ENGINE_OPTIONS)}")
    spec.engine = engine
    spec.env = {k: str(v) for k, v in expand(doc.get("env") or {}, f"{where}: env").items()}
    spec.secrets = dict(doc.get("secrets") or {})
    spec.secrets_store = dict(doc.get("secrets_store") or {})
    spec.rest = {k: doc[k] for k in ("adapters", "scheduler", "identity", "notifications", "remote_agents",
                                     "topology") if k in doc}
    if "auth" in spec.components and not isinstance(spec.components["auth"], list):
        raise ConfigError(f"{where}: auth: a list of entries, e.g. [{{api_key: {{keys: [...]}}}}]")
    return spec


def _agent(doc: dict, path: Path, definitions: Mapping[str, tuple[dict, Path]]) -> AgentSpec:
    where = str(path)
    name = _name(doc, where)
    where = f"{path}: agent {name!r}"
    spec = dict(doc.get("spec") or {})
    unknown = set(spec) - AGENT_SPEC_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)} - {sorted(AGENT_SPEC_KEYS)}")
    base = path.parent
    definition: dict = {}
    if "definition" in spec:
        ref = str(spec["definition"]).split("@", 1)[0]  # name@version: versions are pinned by the image, later
        if ref not in definitions:
            raise ConfigError(f"{where}: no AgentDefinition {ref!r}")
        def_doc, def_path = definitions[ref]
        definition, base = dict(def_doc.get("spec") or {}), def_path.parent
    entrypoint = spec.get("entrypoint") or definition.get("entrypoint")
    if not entrypoint:
        raise ConfigError(f"{where}: no entrypoint (give `definition:` or `entrypoint:`)")
    cls = import_object(str(entrypoint), where, base)
    from llamphouse.sdk.agent import Agent

    if not (isinstance(cls, type) and issubclass(cls, Agent)):
        raise ConfigError(f"{where}: {entrypoint} is not an Agent class")
    session = spec.get("session")
    provider = None
    if session is not None:
        if isinstance(session, Mapping) and "provider" in session:
            provider_cls = import_object(str(session["provider"]), f"{where}.session", base)
            kwargs = expand(dict(session.get("config") or {}), f"{where}.session.config")
            check_kwargs(provider_cls, kwargs, f"{where}.session")
            provider = provider_cls(**kwargs)
        else:
            provider = component("sessions", session, f"{where}.session", base=base)
    execution = dict(spec.get("execution") or {})
    if execution:
        log.warning("%s: execution (%s) is not enforced yet", where, ", ".join(sorted(execution)))
    return AgentSpec(
        name=name, cls=cls, config=expand(dict(spec.get("config") or {}), f"{where}.config"),
        display_name=spec.get("display_name"), session=provider,
        env={k: str(v) for k, v in expand(dict(spec.get("env") or {}), f"{where}.env").items()},
        secrets=dict(spec.get("secrets") or {}), pool=str(spec.get("pool") or definition.get("pool") or "shared"),
        triggers=list(spec.get("triggers") or []), permissions=dict(spec.get("permissions") or {}),
        publish=dict(spec.get("publish") or {}), execution=execution, source=str(path))


def _set_env(env: Mapping[str, str], secrets: Mapping[str, str], store: Mapping[str, Any], who: str) -> None:
    for k, v in env.items():
        os.environ[k] = str(v)
    for var, ref in secrets.items():
        value = resolve_secret(str(ref), store, f"{who}: secrets.{var}")
        if value is not None:
            os.environ[var] = value


def resolve_secret(ref: str, store: Mapping[str, Any], where: str) -> str | None:
    """A secret by its reference: through the runtime file's `secrets_store` (provider `env`: another
    environment variable), or an environment variable of that name."""
    if ref not in store:
        value = os.environ.get(ref)
        if value is None:
            log.warning("%s: secret %r is in no secrets_store and no environment variable", where, ref)
        return value
    entry = dict(store[ref] or {})
    provider = entry.get("provider", "env")
    if provider == "env":
        value = os.environ.get(str(entry.get("name", ref)))
        if value is None:
            log.warning("%s: secret %r: environment variable %r is not set", where, ref, entry.get("name", ref))
        return value
    if ":" in str(provider):  # your own: a callable (name, **options) -> str
        fetch = import_object(str(provider), where)
        return fetch(entry.get("name", ref), **{k: v for k, v in entry.items() if k not in ("provider", "name")})
    raise ConfigError(f"{where}: secret provider {provider!r} - `env`, or your own as module:function")
