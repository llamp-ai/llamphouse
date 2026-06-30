from __future__ import annotations

import hmac
import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from .base import BaseTrigger
from .webhook_command import (
    WebhookCommandPreparationError,
    WebhookCommandPreparer,
)
from ..types.webhook import (
    WebhookCommandConflict,
    WebhookThreadNotFound,
)

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
        thread: Optional[dict[str, str]] = None,
        message: Optional[dict[str, str]] = None,
        thread_metadata: Optional[dict[str, str]] = None,
        run_metadata: Optional[dict[str, str]] = None,
        idempotency: Optional[dict[str, str]] = None,
    ) -> None:
        # Normalize: drop leading slash so we can prepend consistently.
        if not isinstance(path, str) or not path.strip("/"):
            raise ValueError("WebhookTrigger path must be a non-empty string.")
        self.path = path.lstrip("/")
        self.secret_env = secret_env
        self._validate_thread_mapping(thread)
        self._validate_message_mapping(message)
        self._validate_metadata_mapping(thread_metadata or {}, "thread_metadata")
        self._validate_metadata_mapping(run_metadata or {}, "run_metadata")
        self._validate_idempotency(idempotency)
        self.thread = thread
        self.message = message
        self.thread_metadata = thread_metadata or {}
        self.run_metadata = run_metadata or {}
        self.idempotency = idempotency
        self._command_preparer = WebhookCommandPreparer(
            path=self.path,
            thread=self.thread,
            message=self.message,
            thread_metadata=self.thread_metadata,
            run_metadata=self.run_metadata,
            idempotency=self.idempotency,
        )

    @staticmethod
    def _validate_thread_mapping(mapping: Optional[dict[str, str]]) -> None:
        if mapping is None:
            return
        if not isinstance(mapping, dict):
            raise ValueError("thread must be a mapping.")
        unsupported = set(mapping) - {"id"}
        if unsupported:
            raise ValueError(f"Unsupported thread option(s): {sorted(unsupported)}.")
        path = mapping.get("id")
        if path is not None and (not isinstance(path, str) or not path):
            raise ValueError("thread.id must be a non-empty string.")

    @staticmethod
    def _validate_message_mapping(mapping: Optional[dict[str, str]]) -> None:
        if mapping is None:
            return
        if not isinstance(mapping, dict):
            raise ValueError("message must be a mapping.")
        unsupported = set(mapping) - {"text"}
        if unsupported:
            raise ValueError(f"Unsupported message option(s): {sorted(unsupported)}.")
        path = mapping.get("text")
        if not isinstance(path, str) or not path:
            raise ValueError("message.text must be a non-empty string.")

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

            # ── Resolve agent ──────────────────────────────────────────────
            state = request.app.state
            assistants = getattr(state, "assistants", []) or []
            assistant = next((a for a in assistants if a.id == agent_id), None)
            if not assistant:
                raise HTTPException(
                    status_code=404, detail=f"Agent '{agent_id}' not found"
                )

            db = state.data_store
            config_store = getattr(state, "config_store", None)
            config_values = config_store.resolve_config(agent_id) if config_store else None

            try:
                prepared = trigger._command_preparer.prepare(
                    agent_id=agent_id,
                    data=data,
                    run_config_values=config_values,
                )
            except WebhookCommandPreparationError as exc:
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=exc.detail,
                ) from exc
            try:
                result = await db.execute_webhook_command(prepared.command)
            except WebhookCommandConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Webhook idempotency key was reused for a different command",
                ) from exc
            except WebhookThreadNotFound as exc:
                raise HTTPException(
                    status_code=404,
                    detail="Webhook thread was not found",
                ) from exc

            if not result.deduped:
                await state.run_queue.enqueue(
                    {
                        "run_id": result.run_id,
                        "thread_id": result.thread_id,
                        "assistant_id": agent_id,
                        "metadata": {},
                    }
                )

            logger.info(
                "Webhook trigger fired for agent '%s' (run=%s)", agent_id, result.run_id
            )
            return JSONResponse(
                status_code=200 if result.deduped else 202,
                content=result.response_json,
            )

        return router

    # ── Lifespan hooks ────────────────────────────────────────────────────────
    # Routes are registered statically at init time via get_router(); there is
    # nothing async to start or stop.

    async def start(self, agent_id: str, fastapi_state: Any) -> None:
        pass

    async def stop(self) -> None:
        pass
