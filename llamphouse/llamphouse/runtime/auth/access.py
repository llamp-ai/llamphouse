"""Access: what an identity may touch - every thread (Access), or its own tenant's (TenantAccess)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from llamphouse.sdk.thread import Identity

from .base import ANONYMOUS, CREATOR_KEY, TENANT_KEY


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
