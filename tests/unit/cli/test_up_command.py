from types import SimpleNamespace

import pytest

import llamphouse.cli as cli
from llamphouse.cli import config as cli_config


class FakeApp:
    def __init__(self):
        self._skip_worker = False
        self.ignite_calls = []

    def ignite(self, **kwargs):
        self.ignite_calls.append(kwargs)


def _args(config, host="127.0.0.1", port=8080, ws="wsproto", no_workers=False):
    return SimpleNamespace(
        config=str(config),
        host=host,
        port=port,
        ws=ws,
        no_workers=no_workers,
    )


def test_up_command_loads_config_builds_runtime_and_ignites_with_cli_options(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text('version: "0.1"\n', encoding="utf-8")
    fake_config = object()
    fake_app = FakeApp()
    calls = {}

    def fake_load_config(path):
        calls["config_path"] = path
        return fake_config

    def fake_build_app_from_config(config, config_dir):
        calls["config"] = config
        calls["config_dir"] = config_dir
        return fake_app

    monkeypatch.setattr(cli_config.loader, "load_config", fake_load_config)
    monkeypatch.setattr(cli_config.loader, "build_app_from_config", fake_build_app_from_config)

    cli._cmd_up(
        _args(
            config_path,
            host="127.0.0.1",
            port=9090,
            ws="websockets-sansio",
        )
    )

    assert calls["config_path"] == config_path.resolve()
    assert calls["config"] is fake_config
    assert calls["config_dir"] == config_path.resolve().parent
    assert fake_app._skip_worker is False
    assert fake_app.ignite_calls == [
        {"host": "127.0.0.1", "port": 9090, "ws": "websockets-sansio"}
    ]


def test_up_command_no_workers_sets_runtime_skip_worker(tmp_path, monkeypatch):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text('version: "0.1"\n', encoding="utf-8")
    fake_app = FakeApp()

    monkeypatch.setattr(cli_config.loader, "load_config", lambda _path: object())
    monkeypatch.setattr(cli_config.loader, "build_app_from_config", lambda _config, _dir: fake_app)

    cli._cmd_up(_args(config_path, no_workers=True))

    assert fake_app._skip_worker is True
    assert fake_app.ignite_calls == [{"host": "127.0.0.1", "port": 8080, "ws": "wsproto"}]


def test_up_command_exits_when_config_file_is_missing(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.yaml"

    def fake_load_config(_path):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(cli_config.loader, "load_config", fake_load_config)

    with pytest.raises(SystemExit) as exc:
        cli._cmd_up(_args(missing_path))

    assert exc.value.code == 1


def test_up_command_exits_when_config_is_invalid(tmp_path, monkeypatch):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text("not: valid enough for this test\n", encoding="utf-8")

    def fake_load_config(_path):
        raise ValueError("invalid config")

    monkeypatch.setattr(cli_config.loader, "load_config", fake_load_config)

    with pytest.raises(SystemExit) as exc:
        cli._cmd_up(_args(config_path))

    assert exc.value.code == 1


def test_up_command_exits_when_runtime_build_fails(tmp_path, monkeypatch):
    config_path = tmp_path / "llamphouse.yaml"
    config_path.write_text('version: "0.1"\n', encoding="utf-8")

    monkeypatch.setattr(cli_config.loader, "load_config", lambda _path: object())

    def fake_build_app_from_config(_config, _dir):
        raise RuntimeError("entrypoint failed")

    monkeypatch.setattr(cli_config.loader, "build_app_from_config", fake_build_app_from_config)

    with pytest.raises(SystemExit) as exc:
        cli._cmd_up(_args(config_path))

    assert exc.value.code == 1


def test_main_parses_up_command_options(monkeypatch):
    captured = {}

    def fake_cmd_up(args):
        captured["args"] = args

    monkeypatch.setattr(cli, "_cmd_up", fake_cmd_up)
    monkeypatch.setattr(
        "sys.argv",
        [
            "llamphouse",
            "up",
            "--config",
            "custom.yaml",
            "--host",
            "127.0.0.1",
            "--port",
            "9090",
            "--ws",
            "wsproto",
            "--no-workers",
        ],
    )

    cli.main()

    args = captured["args"]
    assert args.config == "custom.yaml"
    assert args.host == "127.0.0.1"
    assert args.port == 9090
    assert args.ws == "wsproto"
    assert args.no_workers is True
