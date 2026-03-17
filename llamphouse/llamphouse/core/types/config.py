from typing import Any, List, Optional, Union
from pydantic import BaseModel


class BaseParam(BaseModel):
    """Base class for all config parameter types."""
    key: str
    label: str
    description: str = ""

    def param_type(self) -> str:
        raise NotImplementedError

    def default_value(self) -> Any:
        raise NotImplementedError

    def serialize(self) -> dict:
        """Serialize the param definition for the dashboard."""
        data = self.model_dump()
        data["type"] = self.param_type()
        return data


class NumberParam(BaseParam):
    """Numeric config parameter (int or float)."""
    default: float
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    def param_type(self) -> str:
        return "number"

    def default_value(self) -> float:
        return self.default


class StringParam(BaseParam):
    """Short text config parameter (single-line input)."""
    default: str = ""
    max_length: Optional[int] = None

    def param_type(self) -> str:
        return "string"

    def default_value(self) -> str:
        return self.default


class PromptParam(BaseParam):
    """Prompt/template config parameter.

    Renders as a multi-line editor in the dashboard with support for
    template variables like ``{{user_name}}``.
    """
    default: str = ""
    variables: Optional[List[str]] = None

    def param_type(self) -> str:
        return "prompt"

    def default_value(self) -> str:
        return self.default


class BooleanParam(BaseParam):
    """Boolean toggle config parameter."""
    default: bool = False

    def param_type(self) -> str:
        return "boolean"

    def default_value(self) -> bool:
        return self.default


class SelectParam(BaseParam):
    """Single-select config parameter with predefined options."""
    default: str
    options: List[str]

    def param_type(self) -> str:
        return "select"

    def default_value(self) -> str:
        return self.default


# Union of all concrete param types for type hints
ConfigParamTypes = Union[NumberParam, StringParam, PromptParam, BooleanParam, SelectParam]
