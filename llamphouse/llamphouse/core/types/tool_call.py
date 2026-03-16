from typing import Any, Dict, Optional, Union, List, Literal
from pydantic import BaseModel, RootModel


class CodeInterpreterOutputLogs(BaseModel):
    logs: str
    type: Literal["logs"]

class CodeInterpreterOutputImageImage(BaseModel):
    file_id: str

class CodeInterpreterOutputImage(BaseModel):
    image: CodeInterpreterOutputImageImage
    type: Literal["image"]

class CodeInterpreterOutput(RootModel[Union[CodeInterpreterOutputLogs, CodeInterpreterOutputImage]]):
    pass

class CodeInterpreter(BaseModel):
    input: str
    outputs: List[CodeInterpreterOutput]

class CodeInterpreterToolCall(BaseModel):
    id: str
    code_interpreter: CodeInterpreter
    type: Literal["code_interpreter"]

class FileSearchRankingOptions(BaseModel):
    ranker: Literal["default_2024_08_21"]
    score_threshold: float

class FileSearchResultContent(BaseModel):
    text: Optional[str] = None
    type: Optional[Literal["text"]] = None

class FileSearchResult(BaseModel):
    file_id: str
    file_name: str
    score: float
    content: Optional[List[FileSearchResultContent]] = None

class FileSearch(BaseModel):
    ranking_options: Optional[FileSearchRankingOptions] = None
    results: Optional[List[FileSearchResult]] = None

class FileSearchToolCall(BaseModel):
    id: str
    file_search: FileSearch
    type: Literal["file_search"]

class Function(BaseModel):
    arguments: str
    name: str
    output: Optional[str] = None

class FunctionToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"]

class ToolCall(RootModel[Union[CodeInterpreterToolCall, FileSearchToolCall, FunctionToolCall]]):
    pass


class GenericToolCall(BaseModel):
    """Protocol-agnostic tool call representation.

    Unlike the OpenAI-specific ``FunctionToolCall``, ``CodeInterpreterToolCall``,
    etc., this model works for any tool type: MCP tools, A2A agent calls,
    custom tool types, etc.

    The ``type`` field is free-form (not restricted to OpenAI's three types)
    and ``input`` is a generic dict instead of a JSON-encoded string.
    """
    id: str
    type: str = "function"
    name: str
    input: Dict[str, Any] = {}
    output: Optional[str] = None
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    metadata: Optional[Dict[str, Any]] = None

    def to_function_tool_call(self) -> FunctionToolCall:
        """Convert to OpenAI-compatible FunctionToolCall for the AssistantAPI adapter."""
        import json
        return FunctionToolCall(
            id=self.id,
            type="function",
            function=Function(
                name=self.name,
                arguments=json.dumps(self.input) if isinstance(self.input, dict) else str(self.input),
                output=self.output,
            ),
        )
