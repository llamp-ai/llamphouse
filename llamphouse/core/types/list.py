from typing import Any, List, Literal, Optional
from pydantic import BaseModel

class ListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[Any]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False