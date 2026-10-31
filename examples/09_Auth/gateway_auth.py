"""A custom authenticator: trusts a token that the app's own gateway signs after it logged the user in.

The token is `user.tenant.expires.signature` in the `X-User-Token` header, the signature an HMAC-SHA256 of
`user.tenant.expires` with a secret the gateway and llamphouse share. Named in the runtime file as
`gateway_auth:GatewayTokenAuth`, with its options checked against `__init__` before anything starts.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

from llamphouse.runtime.auth import Auth, AuthError
from llamphouse.sdk import Identity


def sign(secret: str, user: str, tenant: str, ttl: int = 300) -> str:
    """What the gateway does (the client uses it to make a token)."""
    body = f"{user}.{tenant}.{int(time.time()) + ttl}"
    return f"{body}.{hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()}"


class GatewayTokenAuth(Auth):
    def __init__(self, secret_env: str, header: str = "x-user-token"):
        self.secret_env, self.header = secret_env, header.lower()

    async def authenticate(self, request: Any) -> Identity | None:
        token = request.headers.get(self.header)
        if token is None:
            return None  # not ours: the next authenticator may know the request
        secret = os.environ.get(self.secret_env)
        if not secret:
            raise AuthError(f"{self.secret_env} is not set", 503)
        try:
            user, tenant, expires, signature = token.split(".")
        except ValueError:
            raise AuthError("malformed token") from None
        expected = hmac.new(secret.encode(), f"{user}.{tenant}.{expires}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise AuthError("bad token signature")
        if int(expires) < time.time():
            raise AuthError("token expired")
        return Identity(subject=user, tenant=tenant, metadata={"via": "gateway"})

    def scheme(self) -> dict[str, Any]:
        return {"type": "apiKey", "in": "header", "name": self.header}
