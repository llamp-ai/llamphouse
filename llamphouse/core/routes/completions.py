from fastapi import APIRouter, HTTPException
from ..types.completion import *
from openai import OpenAI
import os

router = APIRouter()

client = OpenAI(api_key=os.getenv("OPENAI_KEY", ""))

@router.post("/completions", response_model=CompletionCreateResponse)
async def create_completions(request: CompletionCreateRequest):
    prompt = request.prompt
    model = request.model
    temperature = request.temperature
    max_tokens = request.max_tokens
    top_p = request.top_p
    n = request.n
    stream = request.stream
    logprobs = request.logprobs
    stop = request.stop
    presence_penalty = request.presence_penalty
    frequency_penalty = request.frequency_penalty
    best_of = request.best_of
    echo = request.echo
    logit_bias = request.logit_bias
    seed = request.seed
    suffix = request.suffix
    user = request.user
    stream_options = request.stream_options

    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            logprobs=logprobs,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            seed=seed,
            user=user,
            stream=stream,
            stream_options=stream_options
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

