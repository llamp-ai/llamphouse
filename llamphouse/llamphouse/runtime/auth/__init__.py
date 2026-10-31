"""Who calls (authentication) and what they may touch (access) - see `base.py` for the whole story.

    base.py       Auth, Authenticator, AuthError, the anonymous identity, a caller's sender
    api_key.py    ApiKeyAuth
    oidc.py       OidcAuth
    access.py     Access (everything), TenantAccess (tenants apart)

The runtime file names them: `auth: [{api_key: {...}}, {oidc: {...}}]`, `access: {tenant: {}}`, or your own.
"""

from .access import Access, TenantAccess
from .api_key import ApiKeyAuth
from .base import ANONYMOUS, CREATOR_KEY, TENANT_KEY, Auth, AuthError, Authenticator, sender_of
from .oidc import OidcAuth

__all__ = ["Auth", "AuthError", "Authenticator", "ApiKeyAuth", "OidcAuth", "Access", "TenantAccess", "ANONYMOUS",
           "TENANT_KEY", "CREATOR_KEY", "sender_of"]
