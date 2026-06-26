import pytest

from llamphouse.cli.config.components import (
    instantiate_from_registry,
    parse_component_entry,
    validate_registry_entry,
)


class ExampleComponent:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled


def test_parse_component_entry_accepts_single_key_mapping():
    assert parse_component_entry({"example": {"enabled": False}}, "adapter") == (
        "example",
        {"enabled": False},
    )


def test_parse_component_entry_treats_null_kwargs_as_empty_mapping():
    assert parse_component_entry({"example": None}, "adapter") == ("example", {})


def test_parse_component_entry_can_allow_none():
    assert parse_component_entry(None, "data_store", allow_none=True) is None


def test_parse_component_entry_rejects_non_mapping_kwargs():
    with pytest.raises(ValueError, match="arguments must be a mapping"):
        parse_component_entry({"example": "bad"}, "adapter")


def test_validate_registry_entry_reports_unknown_component():
    error = validate_registry_entry({}, "missing", {}, "adapter")

    assert error == "Unknown adapter 'missing'. Available adapters: []."


def test_validate_registry_entry_reports_invalid_kwargs():
    error = validate_registry_entry(
        {"example": ExampleComponent},
        "example",
        {"enabled": True, "unknown": 1},
        "adapter",
    )

    assert "does not accept argument" in error
    assert "unknown" in error
    assert "enabled" in error


def test_instantiate_from_registry_returns_component_instance():
    component = instantiate_from_registry(
        {"example": ExampleComponent},
        "example",
        {"enabled": False},
        "adapter",
    )

    assert isinstance(component, ExampleComponent)
    assert component.enabled is False
