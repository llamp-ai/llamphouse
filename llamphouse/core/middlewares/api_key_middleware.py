from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid API key.")

        api_key = auth_header.split("Bearer ")[1]
        if api_key != self.api_key:
            raise HTTPException(status_code=403, detail="Invalid API key.")

        response = await call_next(request)
        return response
