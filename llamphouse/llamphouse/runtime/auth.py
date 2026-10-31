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

import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence

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


class OidcAuth(Auth):
    """OpenID Connect bearer tokens (JWT): signature from the issuer's keys (JWKS, from its discovery
    document unless `jwks_url` is given, cached `jwks_ttl` seconds), issuer, audience, expiry. The
    identity: `subject_claim` (sub), `tenant_claim` (e.g. tid), and the claims named in `metadata_claims`.
    `required_scopes`: all of them in `scp` / `scope`. `jwks`: the keys themselves (a JWKS document),
    instead of fetching them - for a deployment that can't reach the issuer."""

    def __init__(self, issuer: str, audience: str | Sequence[str], *, jwks_url: str | None = None,
                 algorithms: Sequence[str] = ("RS256",), subject_claim: str = "sub", tenant_claim: str | None = None,
                 metadata_claims: Sequence[str] = ("name", "email", "roles"), required_scopes: Sequence[str] = (),
                 leeway: float = 60, jwks_ttl: float = 3600, jwks: Mapping[str, Any] | None = None):
        self.issuer, self.audience, self.jwks_url = issuer.rstrip("/"), audience, jwks_url
        self.algorithms, self.subject_claim, self.tenant_claim = list(algorithms), subject_claim, tenant_claim
        self.metadata_claims, self.required_scopes = list(metadata_claims), list(required_scopes)
        self.leeway, self.jwks_ttl = leeway, jwks_ttl
        self._jwks: tuple[float, Any] | None = None
        self._static = dict(jwks) if jwks is not None else None

    async def _keys(self) -> Any:
        from ._extras import need

        jwt = need("jwt", "oidc")
        if self._static is not None:
            return jwt.PyJWKSet.from_dict(self._static)
        httpx = need("httpx", "oidc")
        if self._jwks is not None and time.monotonic() - self._jwks[0] < self.jwks_ttl:
            return self._jwks[1]
        async with httpx.AsyncClient(timeout=10) as client:
            url = self.jwks_url
            if url is None:
                discovery = (await client.get(f"{self.issuer}/.well-known/openid-configuration")).json()
                url = discovery["jwks_uri"]
            keys = jwt.PyJWKSet.from_dict((await client.get(url)).json())
        self._jwks = (time.monotonic(), keys)
        return keys

    async def authenticate(self, request: Any) -> Identity | None:
        given = request.headers.get("authorization", "")
        if not given.lower().startswith("bearer ") or given[7:].strip().count(".") != 2:
            return None  # no JWT here
        token = given[7:].strip()
        from ._extras import need

        jwt = need("jwt", "oidc")
        try:
            kid = jwt.get_unverified_header(token).get("kid")
            keys = await self._keys()
            key = next((k for k in keys.keys if k.key_id == kid), None)
            if key is None:  # rotated: once more, fresh
                self._jwks = None
                keys = await self._keys()
                key = next((k for k in keys.keys if k.key_id == kid), None)
            if key is None:
                raise AuthError("token signed with an unknown key")
            claims = jwt.decode(token, key.key, algorithms=self.algorithms, audience=self.audience,
                                issuer=self.issuer, leeway=self.leeway,
                                options={"require": ["exp", "iss", self.subject_claim]})
        except AuthError:
            raise
        except Exception as e:  # noqa: BLE001 - expired, wrong audience, bad signature, ...
            raise AuthError(f"invalid token: {e}") from e
        if self.required_scopes:
            scopes = set(str(claims.get("scp") or claims.get("scope") or "").split())
            missing = [s for s in self.required_scopes if s not in scopes]
            if missing:
                raise AuthError(f"token lacks scope(s) {missing}", 403)
        return Identity(subject=str(claims[self.subject_claim]),
                        tenant=str(claims[self.tenant_claim]) if self.tenant_claim and claims.get(self.tenant_claim)
                        else None,
                        metadata={k: claims[k] for k in self.metadata_claims if k in claims})

    def scheme(self) -> dict[str, Any]:
        return {"type": "openIdConnect", "openIdConnectUrl": f"{self.issuer}/.well-known/openid-configuration"}


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


class Access:
    """What an identity may touch. `allows(identity, action, thread_metadata)`: action is "read" or "write";
    `scope(identity)`: the metadata every thread it lists must have; `stamp(identity)`: what a thread it
    creates is stamped with. The default: everything, for every authenticated caller."""

    def allows(self, identity: Identity, action: str, metadata: Mapping[str, Any]) -> bool:
        return True

    def scope(self, identity: Identity) -> dict[str, Any]:
        return {}

    def may_configure(self, identity: Identity, agent: str) -> bool:
        """Whether it may change an agent's config for everyone."""
        return True

    def stamp(self, identity: Identity) -> dict[str, Any]:
        if identity is ANONYMOUS:  # nobody authenticates: nothing to say
            return {}
        stamp = {CREATOR_KEY: identity.subject}
        if identity.tenant:
            stamp[TENANT_KEY] = identity.tenant
        return stamp


class TenantAccess(Access):
    """Tenants apart: a caller touches only threads of its own tenant (`llamphouse.tenant` in the thread's
    metadata, stamped from the creator's identity, inherited by the threads its runs create). A caller
    without a tenant touches only threads without one. `admin_subjects`: callers that see every tenant."""

    def __init__(self, admin_subjects: Sequence[str] = ()):
        self.admins = set(admin_subjects)

    def allows(self, identity: Identity, action: str, metadata: Mapping[str, Any]) -> bool:
        return identity.subject in self.admins or metadata.get(TENANT_KEY) == identity.tenant

    def scope(self, identity: Identity) -> dict[str, Any]:
        return {} if identity.subject in self.admins else {TENANT_KEY: identity.tenant}

    def may_configure(self, identity: Identity, agent: str) -> bool:
        return identity.subject in self.admins  # an agent serves every tenant: its admins change it

    def stamp(self, identity: Identity) -> dict[str, Any]:
        return {CREATOR_KEY: identity.subject, TENANT_KEY: identity.tenant}  # no tenant: stamped as none


def sender_of(identity: Identity, given: Mapping[str, Any] | None) -> dict[str, Any]:
    """The sender of a caller's message: the caller - a name for display is taken from the request; another
    sender only when the identity may act for others (`metadata.act_as`), or when nobody authenticates."""
    given = dict(given or {})
    if identity is ANONYMOUS or identity.metadata.get("act_as"):
        return given
    return {"kind": identity.metadata.get("kind", "person"), "id": identity.subject,
            "name": given.get("name") or identity.metadata.get("name")}

