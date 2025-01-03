from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from ..types.run import RunObject, RunCreateRequest, CreateThreadAndRunRequest, RunListResponse, ModifyRunRequest
from ..assistant import Assistant
from ..database import database as db
from typing import List, Optional
import time
import asyncio
import json

router = APIRouter()


@router.post("/threads/{thread_id}/runs", response_model=RunObject)
async def create_run(
    thread_id: str,
    request: RunCreateRequest,
    req: Request
) -> RunObject:
    try:
        assistants = req.app.state.assistants
        assistant = get_assistant_by_id(assistants, request.assistant_id)
        # store run in db
        run = db.insert_run(thread_id, run=request, assistant=assistant)

        # Check if the task exists
        task_key = f"{request.assistant_id}:{thread_id}"
        if task_key not in req.app.state.task_queues:
            print(f"Creating queue for task {task_key} in RUN")
            req.app.state.task_queues[task_key] = asyncio.Queue(maxsize=1)
            # raise HTTPException(status_code=404, detail="Task not found")

        # check if stream is enabled
        if request.stream:
            output_queue: asyncio.Queue = req.app.state.task_queues[task_key]

            # Streaming generator for SSE
            async def event_stream():
                while True:
                    try:
                        event = await asyncio.wait_for(output_queue.get(), timeout=10.0)  # Set timeout in seconds
                        if event is None:  # Stream completion signal
                            break
                        print(f"Event: {event['event']}")
                        # output_queue.task_done()
                        yield f"event: {event['event']}\ndata: {event['data']}\n\n"
                    except asyncio.TimeoutError:
                        print("TimeoutError: No event received within the timeout period")
                        yield f'''event: error\ndata: {json.dumps({
                            "error": "TimeoutError",
                            "message": "No event received within the timeout period"
                        })}\n\n'''
                        break

            # Return the streaming response
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        return RunObject(
            id=run.id,
            created_at=time.mktime(run.created_at.timetuple()),
            thread_id=thread_id,
            assistant_id=assistant.id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error,
            expires_at=run.expires_at,
            started_at=run.started_at,
            cancelled_at=run.cancelled_at,
            failed_at=run.failed_at,
            completed_at=run.completed_at,
            incomplete_details=run.incomplete_details,
            model=run.model,
            instructions=run.instructions,
            tools=run.tools,
            metadata=run.meta,
            usage=run.usage,
            temperature=run.temperature,
            top_p=run.top_p,
            max_prompt_tokens=run.max_prompt_tokens,
            max_completion_tokens=run.max_completion_tokens,
            truncation_strategy=run.truncation_strategy,
            tool_choice=run.tool_choice,
            parallel_tool_calls=run.parallel_tool_calls,
            response_format=run.response_format,
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/threads/runs", response_model=RunObject)
async def create_thread_and_run(request: CreateThreadAndRunRequest, req: Request):
    try:
        thread = db.insert_thread(request.thread)

        for msg in request.thread.messages:
            if msg.role not in ["user", "assistant"]:
                raise HTTPException(status_code=400, detail="Invalid role. Must be 'user' or 'assistant'.")
            else:
                db.insert_message(thread_id=thread.id, message=msg)

        assistants = req.app.state.assistants
        assistant = get_assistant_by_id(assistants, request.assistant_id)
        # store run in db
        run = db.insert_run(thread.id, run=request, assistant=assistant)

        # Check if the task exists
        task_key = f"{request.assistant_id}:{thread.id}"
        if task_key not in req.app.state.task_queues:
            print(f"Creating queue for task {task_key} in RUN")
            req.app.state.task_queues[task_key] = asyncio.Queue(maxsize=1)
            # raise HTTPException(status_code=404, detail="Task not found")

        # check if stream is enabled
        if request.stream:
            output_queue: asyncio.Queue = req.app.state.task_queues[task_key]

            # Streaming generator for SSE
            async def event_stream():
                while True:
                    try:
                        event = await asyncio.wait_for(output_queue.get(), timeout=10.0)  # Set timeout in seconds
                        if event is None:  # Stream completion signal
                            break
                        print(f"Event: {event['event']}")
                        # output_queue.task_done()
                        yield f"event: {event['event']}\ndata: {event['data']}\n\n"
                    except asyncio.TimeoutError:
                        print("TimeoutError: No event received within the timeout period")
                        yield f'''event: error\ndata: {json.dumps({
                            "error": "TimeoutError",
                            "message": "No event received within the timeout period"
                        })}\n\n'''
                        break

            # Return the streaming response
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        return RunObject(
            id=run.id,
            created_at=time.mktime(run.created_at.timetuple()),
            thread=thread.id,
            assistant_id=assistant.id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error,
            expires_at=run.expires_at,
            started_at=run.started_at,
            cancelled_at=run.cancelled_at,
            failed_at=run.failed_at,
            completed_at=run.completed_at,
            incomplete_details=run.incomplete_details,
            model=run.model,
            instructions=run.instructions,
            tools=run.tools,
            metadata=run.meta,
            usage=run.usage,
            temperature=run.temperature,
            top_p=run.top_p,
            max_prompt_tokens=run.max_prompt_tokens,
            max_completion_tokens=run.max_completion_tokens,
            truncation_strategy=run.truncation_strategy,
            tool_choice=run.tool_choice,
            parallel_tool_calls=run.parallel_tool_calls,
            response_format=run.response_format,
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
  
def get_assistant_by_id(assistants: List[Assistant], assistant_id: str) -> Assistant:
    assistant = next((assistant for assistant in assistants if assistant.id == assistant_id), None)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return assistant

@router.get("/threads/{thread_id}/runs", response_model=RunListResponse)
async def list_runs(thread_id: str, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None) -> RunObject:
    try:
        runs = db.get_runs_by_thread_id(
            thread_id=thread_id,
            limit=limit + 1,
            order=order,
            after=after,
            before=before
        )
        has_more = len(runs) > limit
        first_id = runs[0].id if runs else None
        last_id = runs[-1].id if runs else None
        return  RunListResponse(
                    object="list",
                    data=[
                        RunObject(
                            id=run.id,
                            created_at=int(run.created_at.timestamp()),
                            thread_id=thread_id,
                            assistant_id=run.assistant_id,
                            status=run.status,
                            required_action=run.required_action,
                            last_error=run.last_error,
                            expires_at=run.expires_at,
                            started_at=run.started_at,
                            cancelled_at=run.cancelled_at,
                            failed_at=run.failed_at,
                            completed_at=run.completed_at,
                            incomplete_details=run.incomplete_details,
                            model=run.model,
                            instructions=run.instructions,
                            tools=run.tools,
                            metadata=run.meta,
                            usage=run.usage,
                            temperature=run.temperature,
                            top_p=run.top_p,
                            max_prompt_tokens=run.max_prompt_tokens,
                            max_completion_tokens=run.max_completion_tokens,
                            truncation_strategy=run.truncation_strategy,
                            tool_choice=run.tool_choice,
                            parallel_tool_calls=run.parallel_tool_calls,
                            response_format=run.response_format,
                        )
                        for run in runs
                    ],
                    first_id=first_id,
                    last_id=last_id,
                    has_more=has_more
                )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/runs/{run_id}", response_model=RunObject)
async def retrieve_run(
    thread_id: str,
    run_id: str,
) -> RunObject:
    try:
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        run = db.get_run_by_id(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        
        return RunObject(
            id=run.id,
            created_at=int(run.created_at.timestamp()),
            thread_id=thread_id,
            assistant_id=run.assistant_id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error,
            expires_at=run.expires_at,
            started_at=run.started_at,
            cancelled_at=run.cancelled_at,
            failed_at=run.failed_at,
            completed_at=run.completed_at,
            incomplete_details=run.incomplete_details,
            model=run.model,
            instructions=run.instructions,
            tools=run.tools,
            metadata=run.meta,
            usage=run.usage,
            temperature=run.temperature,
            top_p=run.top_p,
            max_prompt_tokens=run.max_prompt_tokens,
            max_completion_tokens=run.max_completion_tokens,
            truncation_strategy=run.truncation_strategy,
            tool_choice=run.tool_choice,
            parallel_tool_calls=run.parallel_tool_calls,
            response_format=run.response_format,
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.post("/threads/{thread_id}/runs/{run_id}", response_model=RunObject)
async def modify_run(thread_id: str, run_id: str, request: ModifyRunRequest):
    try:
        run = db.update_run_metadata(thread_id, run_id, request.metadata)
        if not run:
            raise HTTPException(status_code=404, detail="Message not found.")
        
        return  RunObject(
            id=run.id,
            created_at=int(run.created_at.timestamp()),
            thread_id=thread_id,
            assistant_id=run.assistant_id,
            status=run.status,
            required_action=run.required_action,
            last_error=run.last_error,
            expires_at=run.expires_at,
            started_at=run.started_at,
            cancelled_at=run.cancelled_at,
            failed_at=run.failed_at,
            completed_at=run.completed_at,
            incomplete_details=run.incomplete_details,
            model=run.model,
            instructions=run.instructions,
            tools=run.tools,
            metadata=run.meta,
            usage=run.usage,
            temperature=run.temperature,
            top_p=run.top_p,
            max_prompt_tokens=run.max_prompt_tokens,
            max_completion_tokens=run.max_completion_tokens,
            truncation_strategy=run.truncation_strategy,
            tool_choice=run.tool_choice,
            parallel_tool_calls=run.parallel_tool_calls,
            response_format=run.response_format,
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")