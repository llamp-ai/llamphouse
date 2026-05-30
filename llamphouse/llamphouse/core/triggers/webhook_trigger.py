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
    ) -> None:
        # Normalize: drop leading slash so we can prepend consistently.
        self.path = path.lstrip("/")
        self.secret_env = secret_env

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

            # ── Create thread + run with trigger metadata ──────────────────
            trigger_info = TriggerInfo(
                source="webhook",
                data=data,
                fired_at=datetime.now(timezone.utc).isoformat(),
            )

            db = state.data_store
            config_store = getattr(state, "config_store", None)
            config_values = config_store.resolve_config(agent_id) if config_store else None

            thread = await db.insert_thread(CreateThreadRequest())
            run_request = RunCreateRequest(
                assistant_id=agent_id,
                metadata={"__trigger__": trigger_info.to_dict()},
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
                content={"run_id": run.id, "thread_id": thread.id},
            )

        return router

    # ── Lifespan hooks ────────────────────────────────────────────────────────
    # Routes are registered statically at init time via get_router(); there is
    # nothing async to start or stop.

    async def start(self, agent_id: str, fastapi_state: Any) -> None:
        pass

    async def stop(self) -> None:
        pass
