"""API keys: `Authorization: Bearer <key>` (or another header, or `?key=`)."""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Mapping, Sequence

from llamphouse.sdk.thread import Identity

from .base import Auth, AuthError


class ApiKeyAuth(Auth):
    """API keys. `keys`: a key, a list of keys, or key -> identity ({subject, tenant, metadata}). A key
    without an identity is `api-key:<its sha256, first 8>`. `header`: where the key is
    (`authorization` takes `Bearer <key>`); `query`: a query parameter that may carry it too (for an
    EventSource, which can't send headers), or None."""

    def __init__(self, keys: str | Sequence[str] | Mapping[str, Mapping[str, Any] | None], *,
                 header: str = "authorization", query: str | None = "key"):
        if isinstance(keys, str):
            keys = [keys]
        mapping = keys if isinstance(keys, Mapping) else {k: None for k in keys}
        self._keys = {k: self._identity(k, v) for k, v in mapping.items() if k}
        if not self._keys:
            raise ValueError("api_key: no keys")
        self.header, self.query = header.lower(), query

    @staticmethod
    def _identity(key: str, given: Mapping[str, Any] | None) -> Identity:
        given = dict(given or {})
        subject = given.get("subject") or f"api-key:{hashlib.sha256(key.encode()).hexdigest()[:8]}"
        return Identity(subject=str(subject), tenant=given.get("tenant"), metadata=dict(given.get("metadata") or {}))

    async def authenticate(self, request: Any) -> Identity | None:
        given = request.headers.get(self.header)
        if given is not None and self.header == "authorization":
            if not given.lower().startswith("bearer "):
                return None
            given = given[7:].strip()
            if given.count(".") == 2:  # a JWT: not an API key - another authenticator's
                return None
        if given is None and self.query:
            given = request.query_params.get(self.query)
        if given is None:
            return None
        for key, identity in self._keys.items():
            if hmac.compare_digest(key.encode(), given.encode()):
                return identity
        raise AuthError("invalid API key", 401)

    def scheme(self) -> dict[str, Any]:
        if self.header == "authorization":
            return {"type": "http", "scheme": "bearer"}
        return {"type": "apiKey", "in": "header", "name": self.header}
