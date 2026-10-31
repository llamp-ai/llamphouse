# Auth

Two questions, two settings in the runtime file:

- **Who calls?** `auth`: a list of authenticators that turn a request into an **identity** (subject, tenant,
  metadata).
- **What may they touch?** `access`: a policy that decides which threads an identity may read and write.

And one in the agent files:

- **What may an agent use?** `permissions`: which other agents it may call, and whether it may read or store
  files.

With no `auth` configured, every caller is `anonymous` and may do everything: right for one process on a laptop,
wrong anywhere else.

## Identities

An identity is `Identity(subject, tenant, metadata)`. The caller's identity is:

- the **sender** of the messages it writes: an authenticated caller can't write as someone else, unless its
  identity says it may act for others (`metadata: {act_as: true}`, for a trusted backend that sends on behalf of
  its users);
- the **caller** of the runs it starts: `ctx.run.caller` in agent code. Runs started by those runs pass it on;
  runs started by a trigger have `trigger:<name>`.

## API keys

```yaml
auth:
  - api_key:
      keys:
        "${WEB_KEY}": {subject: web-app, tenant: acme}
        "${OPS_KEY}": {subject: ops}
        "${BACKEND_KEY}": {subject: backend, tenant: acme, metadata: {act_as: true}}
```

| Option | |
|---|---|
| `keys` | one key, a list of keys, or key → identity (`subject`, `tenant`, `metadata`). A key without an identity is `api-key:<first 8 of its sha256>`. |
| `header` | where the key is: `authorization` (default, as `Bearer <key>`) or another header name, e.g. `x-api-key` |
| `query` | a query parameter that may carry the key too (default `key`, for an `EventSource` that can't send headers); `null` to turn it off |

```bash
curl -H "Authorization: Bearer $WEB_KEY" localhost:8000/v2/agents
```

`llamphouse up --api-key KEY` (or `LLAMPHOUSE_API_KEY`) adds one more key, with no identity of its own, next to
whatever the runtime file configures. A wrong key answers `401`.

## OpenID Connect

```yaml
auth:
  - oidc:
      issuer: https://login.microsoftonline.com/<tenant-id>/v2.0
      audience: api://llamphouse
      tenant_claim: tid
      required_scopes: [agents.use]
```

`pip install 'llamphouse[oidc]'`. A JWT bearer token is checked against the issuer's keys (found through its
discovery document and cached), and for issuer, audience and expiry.

| Option | Default | |
|---|---|---|
| `issuer` | | required |
| `audience` | | required: one or a list |
| `jwks_url` | from discovery | where the keys are |
| `jwks` | | the keys themselves (a JWKS document), for a deployment that can't reach the issuer |
| `algorithms` | `[RS256]` | |
| `subject_claim` | `sub` | the identity's subject |
| `tenant_claim` | none | the identity's tenant, e.g. `tid` |
| `metadata_claims` | `[name, email, roles]` | claims copied into the identity's metadata |
| `required_scopes` | none | all must be in `scp` / `scope` (else `403`) |
| `leeway` | 60 | seconds of clock skew allowed |
| `jwks_ttl` | 3600 | seconds the keys are cached; an unknown key id fetches them again |

Several authenticators are tried in order; the first that recognises the request's credentials decides. An API
key authenticator ignores a JWT, so `api_key` and `oidc` work side by side: tokens for people, keys for services.
A request that none recognises answers `401`.

## Access

```yaml
access: {tenant: {admin_subjects: [ops]}}
```

| Policy | |
|---|---|
| `open` (default) | every authenticated caller sees and acts on every thread: one team, one deployment |
| `tenant` | tenants are kept apart |

Under `tenant`:

- A thread is stamped with its creator's tenant (`llamphouse.tenant` in its metadata) and subject
  (`llamphouse.created_by`). The stamp can't be set or forged by the caller. Threads a run creates (forks,
  threads of other agents) inherit it.
- A caller sees and acts only on its own tenant's threads; a caller without a tenant, only on threads without
  one. Anything else answers `404`, as if it didn't exist. Listings (`GET /v2/threads`, `/v2/stream/runs`,
  subscriptions, A2A `ListTasks`, Compass) are narrowed the same way.
- Files follow their thread; an upload nobody used yet is readable only by its uploader.
- `admin_subjects` see every tenant. Only they may change an agent's config at runtime (an agent serves every
  tenant) and run SQL in Compass dashboards.
- A webhook or schedule names the tenant its threads belong to with its own `tenant:` option.

## Agent permissions

In an agent file:

```yaml
spec:
  entrypoint: agents.py:Desk
  permissions:
    agents: [weather, prices]     # whom it may call or hand over to; "*" (default) for any
    files: read                   # none, read, or read-write (default)
```

The runtime enforces them: `ctx.call` / `ctx.handover` to an agent not in the list raises
`AgentError("not_permitted")` (which fails the run unless the agent catches it), and
`ctx.agents.list()` / `as_tools()` show only the allowed ones. Forks of itself are always allowed. With
`files: read`, `Files.put` fails; with `none`, the `Files` capability is not offered.

## Your own authenticator

An `Auth` turns a request into an identity: return `None` when the request carries no credentials of your kind
(the next authenticator is tried), raise `AuthError` when it carries wrong ones.

```python
# my_auth.py
from llamphouse.runtime.auth import Auth, AuthError
from llamphouse.sdk import Identity


class GatewayAuth(Auth):
    """Trusts the user headers an API gateway in front of llamphouse sets."""

    def __init__(self, header: str = "x-user", tenant_header: str = "x-org", secret: str = ""):
        self.header, self.tenant_header, self.secret = header, tenant_header, secret

    async def authenticate(self, request) -> Identity | None:
        user = request.headers.get(self.header)
        if user is None:
            return None
        if request.headers.get("x-gateway-secret") != self.secret:
            raise AuthError("not from the gateway", 401)
        return Identity(subject=user, tenant=request.headers.get(self.tenant_header))
```

```yaml
auth:
  - my_auth:GatewayAuth: {secret: "${GATEWAY_SECRET}"}
```

`scheme()` may return an OpenAPI security scheme, which the A2A adapter publishes in agent cards.

## Your own access policy

Subclass `Access` and override what you need:

```python
from llamphouse.runtime.auth import Access


class TeamAccess(Access):
    """Threads belong to teams; a caller's teams are in its identity's metadata."""

    def allows(self, identity, action, metadata) -> bool:       # action: "read" or "write"
        return metadata.get("team") in identity.metadata.get("teams", ())

    def scope(self, identity) -> dict:                           # what every listed thread must have
        return {}

    def stamp(self, identity) -> dict:                           # what a thread it creates is stamped with
        return {**super().stamp(identity), "team": identity.metadata.get("team")}

    def may_configure(self, identity, agent) -> bool:            # may it change an agent's config?
        return "admins" in identity.metadata.get("teams", ())
```

```yaml
access: {my_access:TeamAccess: {}}
```

## See also

- [Example 09_Auth](../../examples/09_Auth)
- [HTTP API](../http-api.md)
