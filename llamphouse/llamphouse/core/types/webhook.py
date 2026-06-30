from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class WebhookCommandConflict(Exception):
    """Raised when an idempotency key is reused for a different command."""


class WebhookThreadNotFound(Exception):
    """Raised when a webhook targets a well-formed thread id that does not exist."""


@dataclass(frozen=True)
class WebhookCommand:
    scope: str
    idempotency_key: str | None
    fingerprint: str | None
    agent_id: str
    trigger_path: str
    thread_id: str | None
    thread_metadata: dict[str, Any]
    message_text: str | None
    run_metadata: dict[str, Any]
    run_config_values: dict[str, Any] | None


@dataclass(frozen=True)
class WebhookCommandResult:
    run_id: str
    thread_id: str
    message_id: str | None
    deduped: bool
    thread_created: bool
    response_json: dict[str, Any]
