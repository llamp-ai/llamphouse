# Config

One agent class, set up three ways. A support agent declares what can be tuned as a `Config` (tone,
signature, max words, the model name); its agent file sets values for this deployment; a thread can carry
values of its own (one agent class, many customers); and an operator changes the config at runtime through
the config store - the API or Compass's Config tab - for every process, from the next run on. A run's config
is the file's, then the store's, then its thread's, each over the one before. No model and no key: the reply
is built from the config, and each run returns the config it got.

## Run

```bash
llamphouse up --memory --agents agents.yaml      # http://127.0.0.1:8000/compass
python client.py                                 # in a second terminal
```

```
1. The effective config (GET /v2/agents/support/config):
  {"tone": "friendly", "signature": "The support team", "max_words": 40, "model": "gpt-5-mini"}  changed={} version=0

2. Two threads: one as the file says, one with config of its own:
  plain thread:              tone=friendly signature='The support team'
  vip thread (tone=formal):  tone=formal   signature='The support team'
  a thread with tone=pirate: 422 (checked against the Config when it is made)

3. Changed at runtime, for every process, from the next run on (PUT, with if_version):
  200: changed={'signature': 'Ana, support'} version=1 by=anonymous
  plain thread:              tone=friendly signature='Ana, support'
  vip thread (tone=formal):  tone=formal   signature='Ana, support'
  the same if_version again: 409 config_conflict (someone changed it first)
  max_words=1: 422 (the Config says ge=5)

4. Back to the file's (PUT values {}):
  changed={} version=2
  plain thread:              tone=friendly signature='The support team'
```

`LLAMPHOUSE_URL` points the client elsewhere (default `http://127.0.0.1:8000`).

## In Compass

**Agents** -> `support` -> **Config**: the effective config as a form from the Config's schema, what differs
from the file, *Save for every process* and *Reset to the file's*. Change the signature there and run
`python client.py` again. Each run's output (Runs -> a run) holds the config it was given.

## The code

```python
class SupportConfig(Config):
    tone: Literal["plain", "friendly", "formal"] = "plain"
    signature: str = "Support"
    max_words: int = Field(default=40, ge=5, le=200)
    model: str = "gpt-5-mini"

class SupportAgent(Agent[SupportConfig]):
    async def run(self, ctx) -> dict:
        c = ctx.config                      # validated, every layer applied
        ...
```

```yaml
# agents.yaml - the file's values
spec:
  entrypoint: agents.py:SupportAgent
  config: {tone: friendly, signature: The support team}
```

```bash
# a thread's own values: over the file's and the store's
curl -X POST localhost:8000/v2/threads -d '{"agent": "support", "config": {"tone": "formal"}}' -H 'Content-Type: application/json'
# a change for everyone; if_version: only if nobody changed it since (else 409)
curl -X PUT localhost:8000/v2/agents/support/config -d '{"values": {"signature": "Ana"}, "if_version": 0}' -H 'Content-Type: application/json'
```

- Every value is checked against the Config before it is kept: the file's by `llamphouse check` and on
  start, a thread's when it is made, a change when it is saved (422).
- The store keeps only what was changed (`changed`), so the file's values still apply under it; `{}` goes
  back to the file's. A fork of a thread inherits the thread's config.
- In memory, a change lasts until the process stops. Across processes and restarts, keep it in Postgres:
  `config_store: {postgres: {dsn: "${LLAMPHOUSE_DATABASE_URL}"}}` in the runtime file.
- With `access: {tenant: ...}` (see `09_Auth`) only admins may change an agent's config: it serves every
  tenant.
