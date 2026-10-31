"""Who calls (authentication) and what they may touch (access).

Authentication turns a request into an `Identity` (subject, tenant, metadata) - the caller of the runs it
starts (`ctx.run.caller`) and the sender of the messages it writes. Several authenticators may be
configured; the first that recognises the request's credentials decides:

    auth:
      - api_key: {keys: {"${PARTNER_KEY}": {subject: partner, tenant: acme}}}
      - oidc: {issuer: https://login.microsoftonline.com/<tenant>/v2.0, audience: api://llamphouse}

- `api_key`: `Authorization: Bearer <key>` (or a header of your choice, or `?key=` for an EventSource).
  `keys`: one key, a list, or key -> identity.
- `oidc`: a JWT bearer token, checked against the issuer's keys (JWKS, discovered); `pip install
  'llamphouse[oidc]'`.
- your own: `{my_pkg.auth:MyAuth: {...}}`, an `Auth` subclass.

With none configured, every caller is `anonymous`. Messages carry the caller as their sender; a caller may
name another sender only when its identity says so (`metadata: {act_as: true}` - a trusted service).

Access decides what an identity may touch. The default lets every authenticated caller see and act on every
thread (one team, one deployment). `access: {tenant: {}}` keeps tenants apart: a thread belongs to the tenant
of whoever created it (stamped in its metadata, `llamphouse.tenant`, and inherited by the threads its runs
create); a caller sees and acts only on its own tenant's threads. Or your own: `{my_pkg.access:MyAccess: {}}`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

from llamphouse.sdk.thread import Identity

ANONYMOUS = Identity(subject="anonymous")
TENANT_KEY = "llamphouse.tenant"
CREATOR_KEY = "llamphouse.created_by"


class AuthError(Exception):
    """Credentials that were given and are wrong (401), or a caller that may not do this (403)."""

    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


class Auth(ABC):
    """An authenticator. `authenticate` gets the HTTP request (Starlette): the caller's Identity, None
    when the request carries no credentials of this kind, or AuthError when it carries wrong ones."""

    @abstractmethod
    async def authenticate(self, request: Any) -> Identity | None: ...

    def scheme(self) -> dict[str, Any]:
        """How it is published (an OpenAPI / A2A security scheme)."""
        return {}


class Authenticator:
    """The configured authenticators, in order. None configured: everyone is `anonymous`."""

    def __init__(self, auths: Iterable[Auth] = ()):
        self.auths = list(auths)

    @property
    def open(self) -> bool:
        return not self.auths

    async def identify(self, request: Any) -> Identity:
        if not self.auths:
            return ANONYMOUS
        for auth in self.auths:
            found = await auth.authenticate(request)
            if found is not None:
                return found
        raise AuthError("credentials required", 401)


# ── access ───────────────────────────────────────────────────────────────────


def sender_of(identity: Identity, given: Mapping[str, Any] | None) -> dict[str, Any]:
    """The sender of a caller's message: the caller - a name for display is taken from the request; another
    sender only when the identity may act for others (`metadata.act_as`), or when nobody authenticates."""
    given = dict(given or {})
    if identity is ANONYMOUS or identity.metadata.get("act_as"):
        return given
    return {"kind": identity.metadata.get("kind", "person"), "id": identity.subject,
            "name": given.get("name") or identity.metadata.get("name")}
