from types import SimpleNamespace
import json
import os

import pytest

import llamphouse.cli as cli
from llamphouse.cli import check as check_module
from llamphouse.core.health import HealthCheckResult


def _args(config, fmt="text", verbose=False, timeout=5.0):
    return SimpleNamespace(
        config=str(config),
        format=fmt,
        verbose=verbose,
        timeout=timeout,
    )


def test_check_command_reports_valid_config_as_pass(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text('version: "0.1"\n', encoding="utf-8")

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS config.schema" in captured.out
    assert "PASS data_store.default" in captured.out
    assert "Summary:" in captured.out
    assert "0 failed" in captured.out


def test_check_command_outputs_json_shape(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text('version: "0.1"\n', encoding="utf-8")

    exit_code = cli._cmd_check(_args(config_path, fmt="json"))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "pass"
    assert payload["summary"] == {"pass": 6, "warn": 0, "fail": 0}
    assert payload["checks"][0]["name"] == "config.schema"
    assert payload["checks"][0]["status"] == "pass"


def test_check_command_reports_explicit_in_memory_data_store(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
data_store:
  in_memory:
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS data_store.in_memory" in captured.out
    assert "No external dependency" in captured.out


def test_check_command_exits_one_when_config_is_invalid(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text("version: 0.1\nunknown_section: {}\n", encoding="utf-8")

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.schema" in captured.out
    assert "unknown_section" in captured.out


def test_check_command_formats_schema_error_paths(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support
    definition: responder
    typo_field: true
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "agents[0].typo_field" in captured.out


def test_check_command_loads_dotenv_from_config_directory_before_cwd(
    tmp_path,
    monkeypatch,
    capsys,
):
    cwd = tmp_path / "cwd"
    project = tmp_path / "project"
    cwd.mkdir()
    project.mkdir()
    (cwd / ".env").write_text("DATABASE_URL=postgresql://cwd\n", encoding="utf-8")
    (project / ".env").write_text("DATABASE_URL=postgresql://config-dir\n", encoding="utf-8")
    config_path = project / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
project:
  name: ${DATABASE_URL}
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(cwd)

    exit_code = cli._cmd_check(_args(config_path, fmt="json"))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "pass"
    assert os.environ["DATABASE_URL"] == "postgresql://config-dir"


def test_check_command_runs_external_data_store_health_check(tmp_path, monkeypatch, capsys):
    class FakePostgresDataStore:
        def __init__(self, database_url):
            self.database_url = database_url

        async def health_check(self):
            return HealthCheckResult.pass_(
                "data_store.postgres",
                "data_store",
                "Connected",
                backend="postgres",
                operation="select 1",
            )

    monkeypatch.setitem(check_module.DATA_STORE_REGISTRY, "postgres", FakePostgresDataStore)
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
data_store:
  postgres:
    database_url: postgresql://user:pass@localhost/db
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path, fmt="json"))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert {
        "name": "data_store.postgres",
        "module": "data_store",
        "status": "pass",
        "message": "Connected",
        "details": {"backend": "postgres", "operation": "select 1"},
    } in payload["checks"]


def test_check_command_runs_external_tracing_store_health_check(tmp_path, monkeypatch, capsys):
    class FakePostgresTracingStore:
        def __init__(self, database_url):
            self.database_url = database_url

        async def health_check(self):
            return HealthCheckResult.pass_(
                "tracing.postgres",
                "tracing",
                "Connected",
                backend="postgres",
                operation="select 1",
            )

    monkeypatch.setitem(check_module.TRACING_STORE_REGISTRY, "postgres", FakePostgresTracingStore)
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
tracing:
  postgres:
    database_url: postgresql://user:pass@localhost/db
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path, fmt="json"))

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert {
        "name": "tracing.postgres",
        "module": "tracing",
        "status": "pass",
        "message": "Connected",
        "details": {"backend": "postgres", "operation": "select 1"},
    } in payload["checks"]


def test_check_command_fails_duplicate_yaml_webhook_paths(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: first
    definition: responder
    triggers:
      - webhook:
          path: /triggers/report
  - name: second
    definition: responder
    triggers:
      - webhook:
          path: triggers/report
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.routes" in captured.out
    assert "/triggers/report" in captured.out


def test_check_command_fails_webhook_path_under_adapter_prefix(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support
    definition: responder
    triggers:
      - webhook:
          path: /api/hooks/support
adapters:
  - compass:
      prefix: /api
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.routes" in captured.out
    assert "/api/hooks/support" in captured.out
    assert "/api" in captured.out


def test_check_command_imports_but_does_not_instantiate_agent_entrypoints(tmp_path, capsys):
    (tmp_path / "agents.py").write_text(
        """
from llamphouse.core import Agent, Context

class ExplodingInitAgent(Agent):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("should not instantiate during check")

    async def run(self, context: Context):
        return None
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ExplodingInitAgent
agents:
  - name: support
    definition: responder
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS config.entrypoints" in captured.out


def test_check_command_fails_when_agent_entrypoint_cannot_be_imported(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: missing.py:MissingAgent
agents:
  - name: support
    definition: responder
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.entrypoints" in captured.out
    assert "missing.py:MissingAgent" in captured.out


def test_check_command_fails_unknown_adapter_component(tmp_path, capsys):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
adapters:
  - made_up_adapter:
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.components" in captured.out
    assert "made_up_adapter" in captured.out


def test_check_command_fails_bad_webhook_trigger_kwargs(tmp_path, capsys):
    (tmp_path / "agents.py").write_text(
        """
from llamphouse.core import Agent, Context

class ConfigurableAgent(Agent):
    async def run(self, context: Context):
        return None
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support
    definition: responder
    triggers:
      - webhook:
          path: /triggers/support
          unsupported: true
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.components" in captured.out
    assert "unsupported" in captured.out


def test_check_command_fails_reserved_webhook_metadata_mapping_key(tmp_path, capsys):
    (tmp_path / "agents.py").write_text(
        """
from llamphouse.core import Agent, Context

class ConfigurableAgent(Agent):
    async def run(self, context: Context):
        return None
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support
    definition: responder
    triggers:
      - webhook:
          path: /triggers/support
          run_metadata:
            __webhook_idempotency_key: id
""",
        encoding="utf-8",
    )

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.components" in captured.out
    assert "reserved" in captured.out


def test_check_command_fails_webhook_secret_env_missing(tmp_path, monkeypatch, capsys):
    (tmp_path / "agents.py").write_text(
        """
from llamphouse.core import Agent, Context

class ConfigurableAgent(Agent):
    async def run(self, context: Context):
        return None
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text(
        """
version: "0.1"
definitions:
  - name: responder
    entrypoint: agents.py:ConfigurableAgent
agents:
  - name: support
    definition: responder
    triggers:
      - webhook:
          path: /triggers/support
          secret_env: SUPPORT_WEBHOOK_SECRET
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("SUPPORT_WEBHOOK_SECRET", raising=False)

    exit_code = cli._cmd_check(_args(config_path))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL config.components" in captured.out
    assert "SUPPORT_WEBHOOK_SECRET" in captured.out


def test_main_parses_check_command_options(monkeypatch):
    captured = {}

    def fake_cmd_check(args):
        captured["args"] = args

    monkeypatch.setattr(cli, "_cmd_check", fake_cmd_check)
    monkeypatch.setattr(
        "sys.argv",
        [
            "llamphouse",
            "check",
            "--config",
            "custom.yaml",
            "--format",
            "json",
            "--verbose",
            "--timeout",
            "2.5",
        ],
    )

    cli.main()

    args = captured["args"]
    assert args.config == "custom.yaml"
    assert args.format == "json"
    assert args.verbose is True
    assert args.timeout == 2.5


def test_main_uses_check_command_exit_code(monkeypatch):
    def fake_cmd_check(args):
        return 1

    monkeypatch.setattr(cli, "_cmd_check", fake_cmd_check)
    monkeypatch.setattr("sys.argv", ["llamphouse", "check"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
