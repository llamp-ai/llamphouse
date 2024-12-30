from pydantic import BaseModel
from typing import Optional, Union, Dict, List

class CompletionCreateRequest(BaseModel):
    model: str
    prompt: str
    best_of: Optional[int] = 1
    echo: Optional[bool] = False 
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    stream: Optional[bool] = False 
    user: Optional[str] = None
    seed: Optional[int] = None
    suffix: Optional[str] = None
    stream_options: Optional[dict] = None

class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int
    audio_tokens: int
    reasoning_tokens: int
    rejected_prediction_tokens: int

class PromptTokensDetails(BaseModel):
    audio_tokens: int
    cached_tokens: int

class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    prompt_tokens_details: PromptTokensDetails

class Logprobs(BaseModel):
    text_offset: list[int] = []
    token_logprobs: list[float] = []
    tokens: list[str] = []
    top_logprobs: list[dict] = []

class Choice(BaseModel):
    text: str
    index: int
    finish_reason: str
    logprobs: Logprobs = None

class CompletionCreateResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: Choice
    usage: Usage