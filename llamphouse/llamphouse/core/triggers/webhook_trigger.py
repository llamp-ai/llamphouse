from __future__ import annotations

import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .base import BaseTrigger, TriggerInfo
from ..types.run import RunCreateRequest
from ..types.thread import CreateThreadRequest

logger = logging.getLogger("llamphouse.triggers.webhook")


class WebhookTrigger(BaseTrigger):
    """Trigger an agent via an HTTP POST to a registered endpoint.

    Usage::

        class MyAgent(Agent):
            triggers = [
                WebhookTrigger(path="/triggers/my-agent", secret_env="WEBHOOK_SECRET"),
            ]

    The endpoint accepts ``POST /triggers/my-agent``.
    If ``secret_env`` is set, the request must include a matching
    ``Authorization: Bearer <token>`` header (token read from the env var).

    On success the endpoint returns ``202 Accepted`` with the new
    ``run_id`` and ``thread_id``.
    The request body (JSON) is available as ``context.trigger.data`` inside
    ``agent.run()``.
    """

    def __init__(
        self,
        path: str,
        secret_env: Optional[str] = None,
        thread_metadata: Optional[dict[str, str]] = None,
        run_metadata: Optional[dict[str, str]] = None,
        idempotency: Optional[dict[str, str]] = None,
    ) -> None:
        # Normalize: drop leading slash so we can prepend consistently.
        if not isinstance(path, str) or not path.strip("/"):
            raise ValueError("WebhookTrigger path must be a non-empty string.")
        self.path = path.lstrip("/")
        self.secret_env = secret_env
        self._validate_metadata_mapping(thread_metadata or {}, "thread_metadata")
        self._validate_metadata_mapping(run_metadata or {}, "run_metadata")
        self._validate_idempotency(idempotency)
        self.thread_metadata = thread_metadata or {}
        self.run_metadata = run_metadata or {}
        self.idempotency = idempotency

    @staticmethod
    def _validate_metadata_mapping(mapping: dict[str, str], name: str) -> None:
        for metadata_key, payload_path in mapping.items():
            if metadata_key.startswith("__"):
                raise ValueError(
                    f"{name} key '{metadata_key}' uses a reserved metadata prefix."
                )
            if not isinstance(payload_path, str) or not payload_path:
                raise ValueError(
                    f"{name} path for key '{metadata_key}' must be a non-empty string."
                )

    @staticmethod
    def _validate_idempotency(idempotency: Optional[dict[str, str]]) -> None:
        if idempotency is None:
            return
        if not isinstance(idempotency, dict):
            raise ValueError("idempotency must be a mapping.")
        unsupported = set(idempotency) - {"key"}
        if unsupported:
            raise ValueError(f"Unsupported idempotency option(s): {sorted(unsupported)}.")
        key = idempotency.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("idempotency.key must be a non-empty string.")

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
            raise HTTPException(
                status_code=400,
                detail="Webhook idempotency key is missing",
            )
        if value is None or isinstance(value, (dict, list)):
            raise HTTPException(
                status_code=400,
                detail="Webhook idempotency key must be a scalar value",
            )
        return str(value)

    async def _find_duplicate_run(self, db: Any, agent_id: str, idempotency_key: str):
        try:
            runs = await db.list_all_runs(
                limit=100,
                order="asc",
                after=None,
                before=None,
                filters=[
                    {
                        "field": "metadata",
                        "operator": "contains",
                        "value": idempotency_key,
                    }
                ],
                include_total=False,
            )
        except Exception as exc:
            logger.exception("Webhook idempotency lookup failed")
            raise HTTPException(
                status_code=500,
                detail="Webhook idempotency lookup failed",
            ) from exc

        if runs is None:
            raise HTTPException(
                status_code=500,
                detail="Webhook idempotency lookup failed",
            )

        trigger_path = f"/{self.path}"
        matches = [
            run
            for run in runs.data
            if (run.metadata or {}).get("__webhook_idempotency_key") == idempotency_key
            and (run.metadata or {}).get("__webhook_trigger_path") == trigger_path
            and (run.metadata or {}).get("__webhook_agent_id") == agent_id
        ]
        if len(matches) > 1:
            logger.warning(
                "Multiple webhook idempotency matches found for agent=%s path=%s key=%s; returning oldest run.",
                agent_id,
                trigger_path,
                idempotency_key,
            )
        return matches[0] if matches else None

    @staticmethod
    def _resolve_payload_path(data: dict[str, Any], path: str) -> tuple[bool, Any]:
        current: Any = data
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False, None
            current = current[part]
        return True, current

    # ── Route factory ─────────────────────────────────────────────────────────

    def get_router(self, agent_id: str) -> APIRouter:
        """Return a FastAPI router with the webhook POST endpoint wired up."""
        router = APIRouter()
        trigger = self  # capture for closure

        @router.post(f"/{trigger.path}", status_code=202)
        async def _webhook_endpoint(request: Request):
            # ── Auth ───────────────────────────────────────────────────────
            secret = (
                os.environ.get(trigger.secret_env) if trigger.secret_env else None
            )
            if trigger.secret_env and not secret:
                logger.error(
                    "Webhook secret env '%s' is configured but not set.",
                    trigger.secret_env,
                )
                raise HTTPException(
                    status_code=503,
                    detail="Webhook secret is not configured",
                )
            if secret:
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401, detail="Missing Authorization: Bearer header"
                    )
                token = auth_header[len("Bearer "):]
                # Constant-time comparison prevents timing attacks.
                if not hmac.compare_digest(token.encode(), secret.encode()):
                    raise HTTPException(status_code=403, detail="Invalid token")

            # ── Parse body ─────────────────────────────────────────────────
            try:
                data = await request.json()
                if not isinstance(data, dict):
                    data = {"payload": data}
            except Exception:
                data = {}

            idempotency_key = trigger._idempotency_key_from_payload(data)

            # ── Resolve agent ──────────────────────────────────────────────
            state = request.app.state
            assistants = getattr(state, "assistants", []) or []
            assistant = next((a for a in assistants if a.id == agent_id), None)
            if not assistant:
                raise HTTPException(
                    status_code=404, detail=f"Agent '{agent_id}' not found"
                )

            # ── Create thread + run with trigger metadata ──────────────────
            trigger_info = TriggerInfo(
                source="webhook",
                data=data,
                fired_at=datetime.now(timezone.utc).isoformat(),
            )

            db = state.data_store
            config_store = getattr(state, "config_store", None)
            config_values = config_store.resolve_config(agent_id) if config_store else None

            if idempotency_key is not None:
                duplicate_run = await trigger._find_duplicate_run(db, agent_id, idempotency_key)
                if duplicate_run is not None:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "run_id": duplicate_run.id,
                            "thread_id": duplicate_run.thread_id,
                            "deduped": True,
                        },
                    )

            thread_metadata = trigger._metadata_from_payload(
                data,
                trigger.thread_metadata,
            )
            run_metadata = trigger._metadata_from_payload(
                data,
                trigger.run_metadata,
            )
            run_metadata["__trigger__"] = trigger_info.to_dict()
            if idempotency_key is not None:
                run_metadata["__webhook_idempotency_key"] = idempotency_key
                run_metadata["__webhook_trigger_path"] = f"/{trigger.path}"
                run_metadata["__webhook_agent_id"] = agent_id

            thread = await db.insert_thread(CreateThreadRequest(metadata=thread_metadata))
            run_request = RunCreateRequest(
                assistant_id=agent_id,
                metadata=run_metadata,
                config_values=config_values or None,
            )
            run = await db.insert_run(thread.id, run_request, assistant)

            await state.run_queue.enqueue(
                {
                    "run_id": run.id,
                    "thread_id": thread.id,
                    "assistant_id": agent_id,
                    "metadata": {},
                }
            )

            logger.info(
                "Webhook trigger fired for agent '%s' (run=%s)", agent_id, run.id
            )
            return JSONResponse(
                status_code=202,
                content={"run_id": run.id, "thread_id": thread.id, "deduped": False},
            )

        return router

    # ── Lifespan hooks ────────────────────────────────────────────────────────
    # Routes are registered statically at init time via get_router(); there is
    # nothing async to start or stop.

    async def start(self, agent_id: str, fastapi_state: Any) -> None:
        pass

    async def stop(self) -> None:
        pass
