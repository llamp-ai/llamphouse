from typing import Optional
from fastapi.responses import JSONResponse

class APIError(Exception):
    message: str
    code: int
    param: Optional[str] = None
    type: Optional[str] = 'invalid_request_error'

    def __init__(self, message: str, code: int, param: str = None, type: str = 'invalid_request_error'):
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param
        self.type = type

    def to_JSON_response(self):
        return JSONResponse({
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": str(self.code)
            }
        }, status_code=self.code)