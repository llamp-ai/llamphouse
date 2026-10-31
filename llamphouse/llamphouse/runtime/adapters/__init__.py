"""Adapters: other protocols on the same engine, next to the native API (/v2).

    adapters:                                   # in the runtime file
      - assistants: {prefix: /v1}               # the OpenAI Assistants API
      - responses: {prefix: /v1, agent: desk}   # the OpenAI Responses + Conversations APIs
      - a2a: {prefix: /a2a}                     # A2A v1.0 (JSON-RPC), an agent card per agent
      - ag_ui: {prefix: /ag-ui}                 # AG-UI (pip install 'llamphouse[ag-ui]')
      - my_pkg.adapters:MyAdapter: {...}        # your own

Every adapter exposes `agents` ("*" or a list of names) under its `prefix`, behind the deployment's auth and
access (`llamphouse.runtime.gateway`). Several instances of one kind may be mounted, each with its own prefix.

    base.py         Adapter
    responses.py    ResponsesAdapter
    assistants.py   AssistantsAdapter
    a2a.py          A2AAdapter
    ag_ui.py        AgUiAdapter
    _common.py      what they share: an agent's description, parts without a protocol kind
"""

from .base import Adapter

__all__ = ["Adapter"]
