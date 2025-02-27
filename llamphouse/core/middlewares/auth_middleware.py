from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from .._exceptions import APIError
from ..auth.base_auth import BaseAuth

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth: BaseAuth):
        super().__init__(app)
        self.auth = auth

    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return APIError(code=401, message="Missing or invalid API key.").to_JSON_response()

        api_key = auth_header.split("Bearer ")[1]
        if not self.auth.authenticate(api_key):
            return APIError(code=403, message="Invalid API key.").to_JSON_response()

        return await call_next(request)
