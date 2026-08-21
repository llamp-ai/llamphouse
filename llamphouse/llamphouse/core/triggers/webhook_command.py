from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .base import TriggerInfo
from ..types.webhook import WebhookCommand


class WebhookCommandPreparationError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class WebhookCommandPreparation:
    command: WebhookCommand


class WebhookCommandPreparer:
    def __init__(
        self,
        *,
        path: str,
        thread: dict[str, str] | None,
        message: dict[str, str] | None,
        thread_metadata: dict[str, str],
        run_metadata: dict[str, str],
        idempotency: dict[str, str] | None,
    ) -> None:
        self.path = path
        self.thread = thread
        self.message = message
        self.thread_metadata = thread_metadata
        self.run_metadata = run_metadata
        self.idempotency = idempotency

    def prepare(
        self,
        *,
        agent_id: str,
        data: dict[str, Any],
        run_config_values: dict[str, Any] | None,
    ) -> WebhookCommandPreparation:
        idempotency_key = self._idempotency_key_from_payload(data)
        thread_id = self._thread_id_from_payload(data)
        message_text = self._message_text_from_payload(data)
        thread_metadata = self._metadata_from_payload(data, self.thread_metadata)
        mapped_run_metadata = self._metadata_from_payload(data, self.run_metadata)

        try:
            fingerprint = (
                self._semantic_fingerprint(
                    thread_id=thread_id,
                    message_text=message_text,
                    thread_metadata=thread_metadata,
                    run_metadata=mapped_run_metadata,
                )
                if idempotency_key is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise WebhookCommandPreparationError(
                400,
                "Webhook command contains non-JSON-serializable values",
            ) from exc

        trigger_info = TriggerInfo(
            source="webhook",
            data=data,
            fired_at=datetime.now(timezone.utc).isoformat(),
        )
        run_metadata = dict(mapped_run_metadata)
        run_metadata["__trigger__"] = trigger_info.to_dict()
        if idempotency_key is not None:
            run_metadata["__webhook_idempotency_key"] = idempotency_key
            run_metadata["__webhook_trigger_path"] = f"/{self.path}"
            run_metadata["__webhook_agent_id"] = agent_id

        return WebhookCommandPreparation(
            command=WebhookCommand(
                scope=self._idempotency_scope(agent_id),
                idempotency_key=idempotency_key,
                fingerprint=fingerprint,
                agent_id=agent_id,
                trigger_path=f"/{self.path}",
                thread_id=thread_id,
                thread_metadata=thread_metadata,
                message_text=message_text,
                run_metadata=run_metadata,
                run_config_values=run_config_values or None,
            ),
        )

    def _metadata_from_payload(self, data: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for metadata_key, payload_path in mapping.items():
            found, value = self._resolve_payload_path(data, payload_path)
            if found:
                metadata[metadata_key] = value
        return metadata

    def _idempotency_key_from_payload(self, data: dict[str, Any]) -> str | None:
        if not self.idempotency:
            return None
        found, value = self._resolve_payload_path(data, self.idempotency["key"])
        if not found:
            raise WebhookCommandPreparationError(
                400,
                "Webhook idempotency key is missing",
            )
        if value is None or isinstance(value, (dict, list)):
            raise WebhookCommandPreparationError(
                400,
                "Webhook idempotency key must be a scalar value",
            )
        return str(value)

    def _thread_id_from_payload(self, data: dict[str, Any]) -> str | None:
        if not self.thread or not self.thread.get("id"):
            return None
        found, value = self._resolve_payload_path(data, self.thread["id"])
        if not found:
            return None
        if not isinstance(value, str) or not value:
            raise WebhookCommandPreparationError(
                400,
                "Webhook thread id must be a non-empty string",
            )
        return value

    def _message_text_from_payload(self, data: dict[str, Any]) -> str | None:
        if not self.message:
            return None
        found, value = self._resolve_payload_path(data, self.message["text"])
        if not found:
            raise WebhookCommandPreparationError(
                400,
                "Webhook message text is missing",
            )
        if not isinstance(value, str) or not value:
            raise WebhookCommandPreparationError(
                400,
                "Webhook message text must be a non-empty string",
            )
        return value

    @staticmethod
    def _canonical_json_bytes(value: Any) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def _hash_canonical_json(cls, value: Any) -> str:
        return hashlib.sha256(cls._canonical_json_bytes(value)).hexdigest()

    def _idempotency_scope(self, agent_id: str) -> str:
        return self._hash_canonical_json(
            {
                "type": "webhook",
                "agent_id": agent_id,
                "trigger_path": f"/{self.path}",
            }
        )

    def _semantic_fingerprint(
        self,
        *,
        thread_id: str | None,
        message_text: str | None,
        thread_metadata: dict[str, Any],
        run_metadata: dict[str, Any],
    ) -> str:
        return self._hash_canonical_json(
            {
                "thread_id": thread_id,
                "message_text": message_text,
                "thread_metadata": thread_metadata,
                "run_metadata": run_metadata,
            }
        )

    @staticmethod
    def _resolve_payload_path(data: dict[str, Any], path: str) -> tuple[bool, Any]:
        current: Any = data
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current
