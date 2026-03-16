from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Optional

from fastapi import Request


@dataclass
class AuthResult:
    """Result of an authentication attempt.

    Attributes:
        authenticated: Whether the request is authenticated.
        status_code:   HTTP status code to return on failure
                       (default 401). Use 403 for "valid identity
                       but insufficient permissions".
        identity:      Optional dict with caller identity info
                       (e.g. user_id, email, tenant, roles).
                       Stored on ``request.state.identity`` so
                       downstream handlers can access it.
        message:       Optional rejection reason (shown in error
                       responses when ``authenticated`` is False).
    """

    authenticated: bool = False
    status_code: int = 401
    identity: dict[str, Any] = field(default_factory=dict)
    message: str = "Authentication failed."


class BaseAuth(ABC):
    """Base class for authentication handlers.

    Subclass this and implement :meth:`authenticate`.  The method
    receives the full Starlette ``Request`` so you can inspect any
    header, query parameter, cookie, or body you need.

    Both sync and async implementations are supported — the middleware
    will ``await`` the result if it's a coroutine.

    Examples::

        # Simple API-key check (any header)
        class HeaderKeyAuth(BaseAuth):
            def authenticate(self, request):
                key = request.headers.get("X-API-Key")
                if key == "my-secret":
                    return AuthResult(authenticated=True)
                return AuthResult(message="Invalid X-API-Key header.")

        # Azure Managed Identity / Entra ID JWT
        class AzureAuth(BaseAuth):
            async def authenticate(self, request):
                token = request.headers.get("Authorization", "").removeprefix("Bearer ")
                claims = await verify_azure_token(token)
                if claims:
                    return AuthResult(
                        authenticated=True,
                        identity={"oid": claims["oid"], "tenant": claims["tid"]},
                    )
                return AuthResult(message="Invalid Azure AD token.")

        # Backward-compatible: return bool instead of AuthResult
        class LegacyAuth(BaseAuth):
            def authenticate(self, request):
                key = request.headers.get("Authorization", "").removeprefix("Bearer ")
                return key == "secret"
    """

    @abstractmethod
    def authenticate(self, request: Request) -> "AuthResult | bool":
        """Authenticate an incoming request.

        :param request: The full Starlette/FastAPI ``Request`` object.
        :returns: An :class:`AuthResult`, or ``True``/``False`` for
                  simple cases (converted to ``AuthResult`` by the
                  middleware).
        """
        ...