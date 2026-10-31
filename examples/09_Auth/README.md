# Auth

Who may call, and what they may touch. The runtime file names the authenticators - API keys, each with its
own identity (subject, tenant), and a custom `Auth` class that trusts a token the app's own gateway signs -
and the access policy: `tenant` keeps tenants apart, so a caller sees and acts only on its own tenant's
threads, and an admin sees them all. The agent learns who called from `ctx.run.caller`. `client.py` makes
calls that are allowed and calls that are refused. No model and no key.

## Run

```bash
llamphouse up --memory --config llamphouse.yaml --agents agents.yaml     # http://127.0.0.1:8000/compass
python client.py                                                         # in a second terminal
```

```
Who are you?
  refused  401  no credentials (credentials required)
  refused  401  a wrong API key (invalid API key)
  refused  401  a forged gateway token (bad token signature)
  refused  401  an expired gateway token (token expired)

Each tenant on its own threads:
  allowed  201  ann (acme) makes a thread
           its labels: {'llamphouse.tenant': 'acme', 'llamphouse.created_by': 'ann'}
           Noted for ann (acme): 1 note(s) on this thread.
  allowed  201  carol (initech, gateway token) makes a thread
           Noted for carol (initech): 1 note(s) on this thread.

What another tenant may touch:
  allowed  200  bob (globex) lists threads
           bob sees 0 thread(s)
  refused  404  bob reads acme's thread (no thread thread_cac56728305044a8)
  refused  404  bob starts a run on it (no thread thread_cac56728305044a8)
  refused  404  bob reads its state (no thread thread_cac56728305044a8)
  refused  403  ann changes the agent's config (for every tenant) (ann may not change notes's config)

The admin:
  allowed  200  ops lists threads
           ops sees 2 thread(s): ['acme', 'initech']
  allowed  200  ops reads acme's thread
```

The keys and the gateway secret have defaults for the demo (`acme-key`, `globex-key`, `ops-key`,
`change-me`); set `ACME_KEY`, `GLOBEX_KEY`, `OPS_KEY` and `GATEWAY_SECRET` for both the server and the client
to use others. `LLAMPHOUSE_URL` points the client elsewhere (default `http://127.0.0.1:8000`).

## In Compass

Open `/compass?key=acme-key`: you are ann, and every page - Overview, Threads, Runs, Traces - shows acme's
threads only. With `?key=ops-key` you see every tenant's, and the tenant stamp in each thread's labels.

## The code

```yaml
# llamphouse.yaml
auth:                                       # tried in order: the first that knows the request decides
  - api_key:
      keys:
        "${ACME_KEY:-acme-key}": {subject: ann, tenant: acme}
        "${OPS_KEY:-ops-key}":   {subject: ops}
  - gateway_auth:GatewayTokenAuth: {secret_env: GATEWAY_SECRET}   # your own class, module:Class
access:
  tenant: {admin_subjects: [ops]}
```

```python
class GatewayTokenAuth(Auth):
    def __init__(self, secret_env: str, header: str = "x-user-token"): ...

    async def authenticate(self, request) -> Identity | None:
        token = request.headers.get(self.header)
        if token is None:
            return None                     # not mine: the next authenticator may know it
        ...                                 # check the signature and the expiry
        if bad:
            raise AuthError("bad token signature")       # 401
        return Identity(subject=user, tenant=tenant)
```

- `authenticate` returns the caller's `Identity`, `None` when the request carries no credentials of its kind,
  or raises `AuthError` when it carries wrong ones. Its options in the runtime file are checked against
  `__init__` before anything starts (`llamphouse check`).
- A thread is stamped with its creator's tenant and subject (`llamphouse.tenant`, `llamphouse.created_by`);
  a caller can't forge them, and threads its runs create inherit them. A thread of another tenant answers
  404, as one that doesn't exist.
- The caller is the run's caller (`ctx.run.caller`) and the sender of its messages; it names another
  sender only when its identity says `metadata: {act_as: true}` (a trusted service).
- Under `tenant` access only admins change an agent's config: an agent serves every tenant.
- Also built in: `oidc` (JWT bearer tokens checked against the issuer's keys, `pip install 'llamphouse[oidc]'`).
  `LLAMPHOUSE_API_KEY` (or `--api-key`) adds one more key. Your own access policy is an `Access` subclass
  (`access: {my_pkg.access:MyAccess: {}}`).
