"""Who calls, and what they may touch: API keys, OIDC tokens, tenants apart; agents' permissions."""

import json
import time

import httpx
import pytest

from llamphouse.runtime import Engine
from llamphouse.runtime.api import create_app
from llamphouse.runtime.auth import ApiKeyAuth, Authenticator, OidcAuth, TenantAccess
from llamphouse.sdk import Agent, AgentError, Files, Sender, requires
from llamphouse.sdk.testing import Scenario


class Who(Agent):
    """Says who started it, and who wrote its input."""

    async def run(self, ctx) -> dict:
        return {"caller": ctx.run.caller.subject, "tenant": ctx.run.caller.tenant, "sender": ctx.input.sender.id}


class Spawner(Agent):
    async def run(self, ctx) -> str:
        if ctx.run.parent is not None:
            return "the fork"
        helper = await ctx.fork("help")  # its own work: a fork, same labels
        await helper.wait()
        return "ok"


AGENTS = {"who": Who, "spawner": Spawner}
KEYS = {"key-acme": {"subject": "ann", "tenant": "acme"}, "key-globex": {"subject": "bob", "tenant": "globex"},
        "key-ops": {"subject": "ops"}, "key-svc": {"subject": "bridge", "metadata": {"act_as": True}}}


@pytest.fixture
async def api():
    engine = await Engine(AGENTS).open()
    app = create_app(engine, auth=Authenticator([ApiKeyAuth(KEYS)]), access=TenantAccess(admin_subjects=["ops"]))
    clients = {}
    for key in KEYS:
        clients[key] = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                         headers={"Authorization": f"Bearer {key}"})
    clients["none"] = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    yield engine, clients
    for c in clients.values():
        await c.aclose()
    await engine.close()


async def finished(c, run_id):
    for _ in range(200):
        run = (await c.get(f"/v2/runs/{run_id}")).json()
        if run["status"] == "completed":
            return run
        await __import__("asyncio").sleep(0.01)
    raise AssertionError(run)


async def test_no_credentials_or_wrong_ones_are_refused(api):
    _, c = api
    assert (await c["none"].get("/v2/agents")).status_code == 401
    bad = httpx.AsyncClient(transport=c["none"]._transport, base_url="http://t", headers={"Authorization": "Bearer x"})
    assert (await bad.get("/v2/agents")).status_code == 401
    assert (await c["none"].get("/v2/agents?key=key-acme")).status_code == 200  # an EventSource's way
    assert (await c["none"].get("/health")).status_code == 200


async def test_the_caller_starts_the_run_and_writes_its_messages(api):
    _, c = api
    t = (await c["key-acme"].post("/v2/threads", json={"agent": "who"})).json()
    assert t["metadata"] == {"llamphouse.created_by": "ann", "llamphouse.tenant": "acme"}
    run = (await c["key-acme"].post(f"/v2/threads/{t['id']}/runs",
                                    json={"text": "hi", "sender": {"id": "somebody-else", "name": "Ann"}})).json()
    run = await finished(c["key-acme"], run["id"])
    assert run["output"] == {"caller": "ann", "tenant": "acme", "sender": "ann"}  # not who it claimed to be
    assert run["input"]["sender"]["name"] == "Ann"  # a name for display is kept
    # a trusted service may act for its users
    t2 = (await c["key-svc"].post("/v2/threads", json={"agent": "who"})).json()
    run = (await c["key-svc"].post(f"/v2/threads/{t2['id']}/runs",
                                   json={"text": "hi", "sender": {"id": "teams:carla"}})).json()
    assert (await finished(c["key-svc"], run["id"]))["output"]["sender"] == "teams:carla"


async def test_tenants_dont_see_each_others_threads(api):
    _, c = api
    t = (await c["key-acme"].post("/v2/threads", json={"agent": "spawner",
                                                       "metadata": {"llamphouse.tenant": "globex"}})).json()
    assert t["metadata"]["llamphouse.tenant"] == "acme"  # the stamp can't be forged
    run = (await c["key-acme"].post(f"/v2/threads/{t['id']}/runs", json={"text": "go"})).json()
    await finished(c["key-acme"], run["id"])
    mine = [x["id"] for x in (await c["key-acme"].get("/v2/threads")).json()]
    assert t["id"] in mine and len(mine) == 2  # the fork too: it inherited the tenant
    assert (await c["key-globex"].get("/v2/threads")).json() == []
    for path in (f"/v2/threads/{t['id']}", f"/v2/threads/{t['id']}/events", f"/v2/runs/{run['id']}"):
        assert (await c["key-globex"].get(path)).status_code == 404  # as if it didn't exist
    assert (await c["key-globex"].post(f"/v2/threads/{t['id']}/messages", json={"text": "x"})).status_code == 404
    assert (await c["key-globex"].post(f"/v2/runs/{run['id']}/cancel")).status_code == 404
    assert len((await c["key-ops"].get("/v2/threads")).json()) == 2  # an admin sees every tenant


async def test_files_follow_their_thread_and_uploads_their_uploader(api):
    _, c = api
    part = (await c["key-acme"].post("/v2/files?name=a.txt", content=b"hi",
                                     headers={"Content-Type": "text/plain"})).json()
    fid = part["uri"].split(":")[-1].split("/")[-1]
    assert (await c["key-acme"].get(f"/v2/files/{fid}")).content == b"hi"
    assert (await c["key-globex"].get(f"/v2/files/{fid}")).status_code == 404


def rsa_key():
    from cryptography.hazmat.primitives.asymmetric import rsa

    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


async def test_oidc_tokens():
    import jwt
    from jwt.algorithms import RSAAlgorithm

    key = rsa_key()
    public = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    jwks = {"keys": [{**public, "kid": "k1", "use": "sig", "alg": "RS256"}]}
    issuer, audience = "https://login.example/tenant/v2.0", "api://llamphouse"

    def token(**claims):
        now = int(time.time())
        body = {"iss": issuer, "aud": audience, "sub": "u-1", "tid": "acme", "exp": now + 600, "iat": now,
                "name": "Ann", "scp": "agents.run", **claims}
        return jwt.encode(body, key, algorithm="RS256", headers={"kid": "k1"})

    oidc = OidcAuth(issuer, audience, tenant_claim="tid", required_scopes=["agents.run"], jwks=jwks)
    engine = await Engine(AGENTS).open()
    try:
        app = create_app(engine, auth=Authenticator([oidc, ApiKeyAuth(["svc-key"])]))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            ok = {"Authorization": f"Bearer {token()}"}
            t = (await c.post("/v2/threads", json={"agent": "who"}, headers=ok)).json()
            run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "hi"}, headers=ok)).json()
            for _ in range(200):
                run = (await c.get(f"/v2/runs/{run['id']}", headers=ok)).json()
                if run["status"] == "completed":
                    break
                await __import__("asyncio").sleep(0.01)
            assert run["output"] == {"caller": "u-1", "tenant": "acme", "sender": "u-1"}
            for bad, status in ((token(exp=1), 401), (token(aud="other"), 401), (token(iss="https://evil"), 401),
                                (token(scp="read"), 403)):
                assert (await c.get("/v2/agents", headers={"Authorization": f"Bearer {bad}"})).status_code == status
            assert (await c.get("/v2/agents", headers={"Authorization": "Bearer svc-key"})).status_code == 200
    finally:
        await engine.close()


# ── an agent instance's permissions ──────────────────────────────────────────

class Caller(Agent):
    async def run(self, ctx) -> list:
        seen = [a.name for a in ctx.agents.list()]
        try:
            await ctx.call("secret", "hi")
            called = "called"
        except AgentError as e:
            called = e.code
        return [seen, called]


class Secret(Agent):
    async def run(self, ctx) -> str:
        return "s"


@requires(Files)
class Reader(Agent):
    async def run(self, ctx) -> str:
        files = ctx.capability(Files)
        if files is None:
            return "no files"
        try:
            await files.put(b"x", mime_type="text/plain")
            return "stored"
        except AgentError as e:
            return e.code


async def test_permissions_limit_whom_an_agent_calls_and_what_it_does_with_files():
    agents = {"caller": Caller, "secret": Secret, "helper": Secret, "reader": Reader, "writer": Reader,
              "blind": Reader}
    perms = {"caller": {"agents": ["helper"]}, "reader": {"files": "read"}, "blind": {"files": "none"}}
    async with Scenario(agents, permissions=perms) as sc:
        for name in ("caller", "reader", "writer", "blind"):
            await sc.start(name, "go", by=Sender.person("an"))
        await sc.settle()
        out = {r.agent: r.output for r in sc.runs()}
        assert out["caller"] == [["helper"], "not_permitted"]
        assert out["reader"] == "not_permitted" and out["writer"] == "stored" and out["blind"] == "no files"
    with pytest.raises(ValueError, match="files"):
        Engine(agents, permissions={"reader": {"files": "all"}})
    with pytest.raises(ValueError, match="unknown agent"):
        Engine(agents, permissions={"nobody": {}})
