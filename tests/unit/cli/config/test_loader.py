import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from llamphouse.cli.config.loader import build_app_from_config, load_config
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.adapters.compass import CompassAdapter
from llamphouse.core.data_stores import PostgresDataStore
from llamphouse.core.tracing.stores import InMemoryTracingStore
from llamphouse.core.triggers import WebhookTrigger
from llamphouse.core.workers import AsyncWorker


FIXTURES_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "yaml_runtime"


def _copy_project(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(FIXTURES_DIR, project)
    return project


def _write_yaml(project, body: str):
    path = project / "llamphouse.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _build_app(project, yaml_body: str):
    config_path = _write_yaml(project, yaml_body)
    config = load_config(config_path)
    return build_app_from_config(config, project)


def test_load_config_validates_definition_references(tmp_path):
    project = _copy_project(tmp_path)
    config_path = _write_yaml(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: missing-definition
""",
    )

    with pytest.raises(ValidationError, match="unknown definition"):
        load_config(config_path)


def test_build_app_from_config_builds_multiple_registered_agents_from_one_definition(tmp_path):
    project = _copy_project(tmp_path)
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
    config:
      tone: concise
      label: support
  - name: audit-agent
    definition: responder
    config:
      tone: strict
      label: audit
adapters: []
tracing:
  in_memory:
""",
    )

    assert [agent.id for agent in app.agents] == ["support-agent", "audit-agent"]
    assert [agent.name for agent in app.agents] == ["support-agent", "audit-agent"]
    assert app.agents[0].settings == {"tone": "concise", "label": "support"}
    assert app.agents[1].settings == {"tone": "strict", "label": "audit"}


def test_build_app_from_config_resolves_yaml_config_into_config_store(tmp_path):
    project = _copy_project(tmp_path)
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
    config:
      tone: concise
  - name: audit-agent
    definition: responder
    config:
      tone: strict
      label: audit
adapters: []
tracing:
  in_memory:
""",
    )

    config_store = app.fastapi.state.config_store
    assert config_store.resolve_config("support-agent") == {
        "tone": "concise",
        "label": "default-label",
    }
    assert config_store.resolve_config("audit-agent") == {
        "tone": "strict",
        "label": "audit",
    }


def test_build_app_from_config_supports_agent_subclass_entrypoint(tmp_path):
    project = _copy_project(tmp_path)
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: subclass-responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: subclass-agent
    definition: subclass-responder
    config:
      label: subclass
adapters: []
tracing:
  in_memory:
""",
    )

    assert app.agents[0].id == "subclass-agent"
    assert app.agents[0].settings == {"label": "subclass"}


def test_build_app_from_config_supports_async_function_entrypoint(tmp_path):
    project = _copy_project(tmp_path)
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: function-responder
    entrypoint: agents.py:function_agent
agents:
  - name: function-agent
    definition: function-responder
    config:
      label: function
adapters: []
tracing:
  in_memory:
""",
    )

    assert app.agents[0].id == "function-agent"
    assert app.agents[0].name == "function-agent"
    assert app.agents[0].settings == {"label": "function"}


def test_build_app_from_config_supports_factory_entrypoint(tmp_path):
    project = _copy_project(tmp_path)
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: factory-responder
    entrypoint: agents.py:create_agent
agents:
  - name: factory-agent
    definition: factory-responder
    config:
      label: factory
adapters: []
tracing:
  in_memory:
""",
    )

    assert app.agents[0].id == "factory-agent"
    assert app.agents[0].name == "factory-factory-agent"
    assert app.agents[0].settings == {"label": "factory"}


def test_build_app_from_config_instantiates_explicit_adapters_worker_and_tracing_store(tmp_path):
    project = _copy_project(tmp_path)
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
adapters:
  - a2a:
  - compass:
      prefix: /dashboard
workers:
  - asyncworker:
      time_out: 60
tracing:
  in_memory:
""",
    )

    assert [type(adapter) for adapter in app.adapters] == [A2AAdapter, CompassAdapter]
    assert app.adapters[0].prefix == ""
    assert app.adapters[1].prefix == "/dashboard"
    assert isinstance(app.worker, AsyncWorker)
    assert app.worker.time_out == 60
    assert isinstance(app.fastapi.state.tracing_store, InMemoryTracingStore)


def test_build_app_from_config_rejects_explicit_postgres_data_store_database_url(
    tmp_path,
    monkeypatch,
):
    project = _copy_project(tmp_path)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:pass@localhost/llamphouse",
    )

    with pytest.raises(ValueError, match="remove data_store.postgres.database_url"):
        _build_app(
            project,
            """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
adapters: []
data_store:
  postgres:
    database_url: postgresql+asyncpg://user:pass@localhost/llamphouse
tracing:
  in_memory:
""",
        )


def test_build_app_from_config_uses_database_url_env_for_postgres_data_store(
    tmp_path,
    monkeypatch,
):
    project = _copy_project(tmp_path)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:pass@localhost/llamphouse",
    )
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
adapters: []
data_store:
  postgres:
tracing:
  in_memory:
""",
    )

    assert isinstance(app.fastapi.state.data_store, PostgresDataStore)


def test_build_app_from_config_requires_database_url_for_postgres_data_store(
    tmp_path,
    monkeypatch,
):
    project = _copy_project(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError, match="data_store.postgres.*DATABASE_URL"):
        _build_app(
            project,
            """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
adapters: []
data_store:
  postgres:
tracing:
  in_memory:
""",
        )


def test_load_config_expands_env_placeholders(tmp_path, monkeypatch):
    project = _copy_project(tmp_path)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:pass@localhost/llamphouse",
    )
    config_path = _write_yaml(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
    config:
      database_url: ${DATABASE_URL}
""",
    )

    config = load_config(config_path)

    assert config.agents[0].config == {
        "database_url": "postgresql+asyncpg://user:pass@localhost/llamphouse",
    }


def test_load_config_reports_missing_env_placeholder(tmp_path, monkeypatch):
    project = _copy_project(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config_path = _write_yaml(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
    config:
      database_url: ${DATABASE_URL}
""",
    )

    with pytest.raises(ValueError, match="DATABASE_URL.*agents\\[0\\].config.database_url"):
        load_config(config_path)


def test_load_config_rejects_unknown_top_level_fields(tmp_path):
    project = _copy_project(tmp_path)
    config_path = _write_yaml(
        project,
        """
version: "0.1"
definitions: []
agents: []
unknown_section: {}
""",
    )

    with pytest.raises(ValidationError, match="unknown_section"):
        load_config(config_path)


def test_load_config_rejects_unknown_nested_fields(tmp_path):
    project = _copy_project(tmp_path)
    config_path = _write_yaml(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
    typo_field: true
""",
    )

    with pytest.raises(ValidationError, match="typo_field"):
        load_config(config_path)


def test_build_app_from_config_attaches_yaml_webhook_trigger_to_agent(tmp_path):
    project = _copy_project(tmp_path)
    app = _build_app(
        project,
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support-agent
    definition: responder
    triggers:
      - webhook:
          path: /triggers/support
          secret_env: SUPPORT_WEBHOOK_SECRET
          thread:
            id: thread_id
          message:
            text: message
adapters: []
tracing:
  in_memory:
""",
    )

    trigger = app.agents[0].triggers[0]
    assert isinstance(trigger, WebhookTrigger)
    assert trigger.path == "triggers/support"
    assert trigger.secret_env == "SUPPORT_WEBHOOK_SECRET"
    assert trigger.thread == {"id": "thread_id"}
    assert trigger.message == {"text": "message"}


def test_load_config_reads_utf8_yaml(tmp_path):
    project = _copy_project(tmp_path)
    project_name = "unicode-project-\u0e40\u0e02\u0e47\u0e21\u0e17\u0e34\u0e28"
    label = "\u0e44\u0e17\u0e22-\U0001f9ed"
    config_path = _write_yaml(
        project,
        f"""
version: "0.1"
project:
  name: "{project_name}"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: unicode-agent
    definition: responder
    config:
      label: "{label}"
""",
    )

    config = load_config(config_path)

    assert config.project.name == project_name
    assert config.agents[0].config["label"] == label
