"""OpenID Connect bearer tokens (JWT), checked against the issuer's keys (`pip install 'llamphouse[oidc]'`)."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from llamphouse.sdk.thread import Identity

from .base import Auth, AuthError


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
        from .._extras import need

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
        from .._extras import need

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
