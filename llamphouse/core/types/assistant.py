from typing import Optional, Union, Dict, List, Literal
from pydantic import BaseModel

class AssistantObject(BaseModel):
    id: str
    object: Literal["assistant"] = "assistant"
    model: str
    name: Optional[str] = None
    description: Optional[str] = None
    temperature: Optional[float] = 0.7 
    top_p: Optional[float] = 1.0
    instruction: Optional[str] = None
    tools: Optional[List[str]] = None

class AssistantListResponse(BaseModel):
    data: List[AssistantObject]
    after: Optional[str]
    before: Optional[str]

class AssistantListRequest(BaseModel):
    after: Optional[str] = None
    before: Optional[str] = None
    limit: Optional[int] = 20
    order: Optional[str] = "desc"

class AssistantCreateRequest(BaseModel):
    model: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    metadata: Optional[object] = {}
    name: Optional[str] = None
    response_format: Optional[Union[str, Dict]] = "auto"
    temperature: Optional[float] = 1.0
    tool_resources: Optional[dict] = None
    tools: Optional[List[str]] = []
    top_p: Optional[float] = 1.0
    extra_headers: Optional[dict] = None
    extra_query: Optional[dict] = None
    extra_body: Optional[dict] = None
    timeout: Optional[float] = None

class AssistantCreateResponse(BaseModel):
    id: str
    model: str
    name: str
    description: Optional[str]
    instructions: Optional[str]
    tools: List[str]
    file_ids: List[str]
    metadata: Optional[object] = {}
    object: str
    temperature: float
    top_p: float
    response_format: Union[str, Dict]

class ModifyAssistantRequest(BaseModel):
    model: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[Dict[str, str]]] = []
    file_ids: Optional[List[str]] = []
    metadata: Optional[object] = {}
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    response_format: Optional[Union[str, Dict[str, str]]] = "auto"