import traceback
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from .._exceptions import APIError

class CatchExceptionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)

        except APIError as e:
            response = e.to_JSON_response()

        except Exception as e:
            print(f"Unhandled exception: {str(e)}")
            print(traceback.format_exc())
            response = JSONResponse({
                "error": {
                    "message": str(e),
                    "type": "unhandled_internal_server_error",
                    "param": None,
                    "code": None
                }
            }, status_code=500)

        return response