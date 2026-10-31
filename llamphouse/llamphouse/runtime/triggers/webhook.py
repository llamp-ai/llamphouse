"""Webhook triggers: a POST from another system starts (or reaches) a run."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Mapping

from fastapi import Request
from fastapi.responses import JSONResponse

from llamphouse.sdk import DataPart, Message, Sender, TextPart
from llamphouse.sdk.thread import Identity

from .base import check_busy, resolve, thread_label


class WebhookTrigger:
    """POST {path}: a run of the agent (see the module docstring). `secret` / `secret_env`: a bearer token;
    `signature`: an HMAC of the raw body ({header: X-Hub-Signature-256, secret_env: ..., prefix: "sha256="});
    `public: true` for neither. `tenant`: the tenant its threads belong to (`access: tenant`)."""

    kind = "webhook"

    def __init__(self, path: str, *, name: str | None = None, secret: str | None = None, secret_env: str | None = None,
                 signature: Mapping[str, Any] | None = None, public: bool = False, thread: Any = None,
                 message: Mapping[str, str] | None = None, thread_metadata: Mapping[str, str] | None = None,
                 idempotency: Mapping[str, str] | None = None, busy: str = "send", tenant: str | None = None):
        if not path.strip("/"):
            raise ValueError("webhook: path is required")
        if not (secret or secret_env or signature or public):
            raise ValueError("webhook: give secret, secret_env or signature - or public: true for an open endpoint")
        if signature is not None and not (signature.get("secret") or signature.get("secret_env")):
            raise ValueError("webhook: signature needs secret or secret_env")
        if message is not None and set(message) - {"text"}:
            raise ValueError(f"webhook: message takes text, not {sorted(set(message) - {'text'})}")
        if idempotency is not None and set(idempotency) != {"key"}:
            raise ValueError("webhook: idempotency takes key")
        self.path = "/" + path.strip("/")
        self.name = name or f"webhook{self.path.replace('/', ':')}"
        self.secret, self.secret_env, self.signature, self.public = secret, secret_env, signature, public
        self.label = thread_label(thread)
        self.message, self.thread_metadata = dict(message or {}), dict(thread_metadata or {})
        self.idempotency, self.busy, self.tenant = idempotency, check_busy(busy), tenant

    def _secret(self, given: str | None, env: str | None) -> str | None:
        return given or (os.environ.get(env) if env else None)

    def verify(self, headers: Mapping[str, str], body: bytes) -> tuple[int, str] | None:
        """None when the request may come in; else (status, why)."""
        if self.public:
            return None
        if self.signature is not None:
            secret = self._secret(self.signature.get("secret"), self.signature.get("secret_env"))
            if not secret:
                return 503, "webhook signature secret is not configured"
            header = str(self.signature.get("header", "X-Hub-Signature-256"))
            prefix = str(self.signature.get("prefix", "sha256="))
            given = headers.get(header.lower()) or headers.get(header) or ""
            expected = prefix + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            return None if hmac.compare_digest(given.encode(), expected.encode()) else (401, "bad signature")
        secret = self._secret(self.secret, self.secret_env)
        if not secret:
            return 503, "webhook secret is not configured"
        given = headers.get("authorization", "")
        if not given.lower().startswith("bearer "):
            return 401, "Authorization: Bearer <secret> required"
        return None if hmac.compare_digest(given[7:].strip().encode(), secret.encode()) else (403, "invalid token")

    def prepare(self, payload: Any) -> tuple[Message, str | None, dict[str, Any], str | None]:
        """(the input message, the thread label, the thread metadata, the idempotency key)."""
        parts: list[Any] = []
        if "text" in self.message:
            found, text = resolve(payload, self.message["text"])
            if found and text is not None:
                parts.append(TextPart(text=str(text)))
        parts.append(DataPart(data=payload))
        label = None
        if self.label:
            found, value = resolve(payload, self.label)
            if not found or value in (None, ""):
                raise ValueError(f"the payload has no {self.label} (thread label)")
            label = f"{self.name}:{value}"
        metadata = {}
        for key, path in self.thread_metadata.items():
            found, value = resolve(payload, path)
            if found:
                metadata[key] = value
        key = None
        if self.idempotency:
            found, value = resolve(payload, self.idempotency["key"])
            if not found or isinstance(value, (dict, list)) or value in (None, ""):
                raise ValueError(f"the payload has no {self.idempotency['key']} (idempotency key)")
            key = f"{self.name}:{value}"
        if self.tenant:
            metadata["llamphouse.tenant"] = self.tenant
        return Message.create(Sender.trigger(self.name), *parts), label, metadata, key

    def mount(self, app: Any, engine: Any, agent: str) -> None:
        trigger = self

        @app.post(self.path, status_code=202, include_in_schema=False)
        async def webhook(request: Request) -> Any:
            body = await request.body()
            refused = trigger.verify(request.headers, body)
            if refused is not None:
                return JSONResponse({"error": refused[1]}, status_code=refused[0])
            try:
                payload = json.loads(body) if body else {}
            except ValueError:
                return JSONResponse({"error": "the body is not JSON"}, status_code=400)
            try:
                message, label, metadata, key = trigger.prepare(payload)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=422)
            claim = f"webhook:{agent}:{key}" if key else None
            if claim:
                earlier = await engine.store.claim(claim, "pending")
                if earlier == "pending":
                    return JSONResponse({"error": "the same event is being handled"}, status_code=409)
                if earlier is not None:
                    return JSONResponse(json.loads(earlier), status_code=200, headers={"Idempotent-Replay": "true"})
            try:
                result = await engine.fire(agent, message, label=label, thread_metadata=metadata, busy=trigger.busy,
                                           identity=Identity(subject=f"trigger:{trigger.name}"))
            except Exception as e:
                if claim:
                    await engine.store.release_claim(claim)
                status = 409 if type(e).__name__ == "ThreadBusy" else 500
                return JSONResponse({"error": str(e)}, status_code=status)
            if claim:
                await engine.store.set_claim(claim, json.dumps(result))
            return JSONResponse(result, status_code=202)
