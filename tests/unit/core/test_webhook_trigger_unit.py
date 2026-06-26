import pytest

from llamphouse.core.triggers import WebhookTrigger


def test_webhook_trigger_rejects_reserved_metadata_mapping_keys():
    with pytest.raises(ValueError, match="reserved"):
        WebhookTrigger(
            path="/triggers/report",
            run_metadata={"__webhook_idempotency_key": "id"},
        )


def test_webhook_trigger_rejects_invalid_metadata_mapping_paths():
    with pytest.raises(ValueError, match="non-empty string"):
        WebhookTrigger(
            path="/triggers/report",
            thread_metadata={"tenant_id": ""},
        )


def test_webhook_trigger_rejects_unknown_idempotency_options():
    with pytest.raises(ValueError, match="Unsupported idempotency option"):
        WebhookTrigger(
            path="/triggers/report",
            idempotency={"key": "id", "scope": "global"},
        )


def test_webhook_trigger_rejects_empty_path():
    with pytest.raises(ValueError, match="path must be a non-empty string"):
        WebhookTrigger(path="")
