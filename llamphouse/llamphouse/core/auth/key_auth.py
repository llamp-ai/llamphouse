from fastapi import Request

from .base_auth import AuthResult, BaseAuth


class KeyAuth(BaseAuth):
    """Authenticate via ``Authorization: Bearer <key>`` header.

    This is the simplest built-in auth — it compares the Bearer token
    to a known API key string.

    Usage::

        app = llamphouse.App(authenticator=KeyAuth("my-secret-key"))
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def authenticate(self, request: Request) -> AuthResult:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return AuthResult(
                status_code=401,
                message="Missing or invalid API key.",
            )

        token = auth_header.removeprefix("Bearer ")
        if token != self.api_key:
            return AuthResult(
                status_code=403,
                message="Invalid API key.",
            )

        return AuthResult(authenticated=True)