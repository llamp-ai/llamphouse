from inspect import isawaitable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from .._exceptions import APIError
from ..auth.base_auth import AuthResult, BaseAuth


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth: BaseAuth):
        super().__init__(app)
        self.auth = auth

    async def dispatch(self, request: Request, call_next):
        result = self.auth.authenticate(request)
        if isawaitable(result):
            result = await result

        # Support legacy implementations that return a plain bool
        if isinstance(result, bool):
            result = AuthResult(
                authenticated=result,
                message="Authentication failed." if not result else "",
            )

        if not result.authenticated:
            return APIError(
                code=result.status_code,
                message=result.message or "Authentication failed.",
            ).to_JSON_response()

        # Expose identity info to downstream handlers
        request.state.identity = result.identity

        return await call_next(request)
