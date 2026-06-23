import pytest

from llamphouse.core.adapters import AssistantAPIAdapter


pytestmark = pytest.mark.unit


def test_assistant_api_adapter_emits_deprecation_warning(monkeypatch):
    warnings = []
    monkeypatch.setattr("llamphouse.core.adapters.assistant_api.adapter.logger.warning", warnings.append)

    with pytest.warns(DeprecationWarning, match="AssistantAPIAdapter is deprecated"):
        adapter = AssistantAPIAdapter(prefix="/v1")

    assert adapter.prefix == "/v1"
    assert warnings == [
        "AssistantAPIAdapter is deprecated and will be removed in a future release. "
        "Use A2AAdapter instead."
    ]
