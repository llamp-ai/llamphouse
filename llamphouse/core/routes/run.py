from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from ..types.run import RunObject, RunCreateRequest, CreateThreadAndRunRequest, ModifyRunRequest, SubmitRunToolOutputRequest
from ..types.thread import CreateThreadRequest
from ..types.run_step import CreateRunStepRequest, RunStepObject
from ..types.enum import run_status, run_step_status, message_status, event_type
from ..types.list import ListResponse
from ..assistant import Assistant
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..streaming.event import Event, DoneEvent, ErrorEvent
from ..data_stores.base_data_store import BaseDataStore
from llamphouse.core.queue.base_queue import BaseQueue
from typing import List, Optional
import asyncio
import logging
import traceback

router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/threads/{thread_id}/runs", response_model=RunObject)
async def create_run(
    thread_id: str,
    request: RunCreateRequest,
    req: Request
) -> RunObject:
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store 

        # Get the assistant
        assistants = req.app.state.assistants
        assistant = get_assistant_by_id(assistants, request.assistant_id)

        # check if stream is enabled
        if request.stream:
            # Check if the task exists
            task_key = f"{request.assistant_id}:{thread_id}"
            if task_key not in req.app.state.event_queues:
                req.app.state.event_queues[task_key] = req.app.state.queue_class()
            output_queue: BaseEventQueue = req.app.state.event_queues[task_key]
        else:
            output_queue = None

        run_queue: BaseQueue = req.app.state.run_queue
        
        # store run in db
        run = await db.insert_run(thread_id, run=request, assistant=assistant, event_queue=output_queue)
        if not run:
            raise HTTPException(status_code=404, detail="Thread not found.")

        if run_queue:
            await run_queue.enqueue({
                "run_id": run.id,
                "thread_id": thread_id,
                "assistant_id": run.assistant_id,
            })

        if not output_queue:
            return run

        # check if stream is enabled
        if output_queue:

            # Streaming generator for SSE
            async def event_stream():
                while True:
                    try:
                        event: Event = await asyncio.wait_for(output_queue.get(), timeout=30.0)  # Set timeout in seconds
                        if event is None:  # Stream completion signal
                            break
                        yield event.to_sse()
                        if event.event == event_type.DONE:
                            logger.debug(f"Received DONE event for run {run.id}, ending stream.")
                            break
                    except asyncio.TimeoutError:
                        logger.debug("TimeoutError: No event received within the timeout period")
                        yield ErrorEvent({
                            "error": "TimeoutError",
                            "message": "No event received within the timeout period"
                        }).to_sse()
                        break
                    except Exception as e:
                        yield ErrorEvent({
                            "error": "InternalError",
                            "message": str(e)
                        }).to_sse()
                        break
                    
                # Cleanup the queue after the stream ends
                try:
                    while not output_queue.empty():
                        try:
                            await output_queue.get_nowait()
                        except Exception:
                            break
                finally:
                    await output_queue.close()

                # Remove the event queue after the stream ends
                if task_key in req.app.state.event_queues:
                    del req.app.state.event_queues[task_key]

            # Return the streaming response
            return StreamingResponse(event_stream(), media_type="text/event-stream")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/threads/runs", response_model=RunObject)
async def create_thread_and_run(request: CreateThreadAndRunRequest, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store 

        # create thread
        thread = await db.insert_thread(request.thread)
        if not thread:
            raise HTTPException(status_code=400, detail="Thread with the same ID already exists.")

        # Remove the thread from the run request
        del request.thread

        return await create_run(thread.id, RunCreateRequest(**request.model_dump()), req)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
  
def get_assistant_by_id(assistants: List[Assistant], assistant_id: str) -> Assistant:
    assistant = next((assistant for assistant in assistants if assistant.id == assistant_id), None)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found.")
    return assistant

@router.get("/threads/{thread_id}/runs", response_model=ListResponse)
async def list_runs(thread_id: str, req: Request, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None) -> RunObject:
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store
        
        # Fetch runs from the database
        runs: ListResponse = await db.list_runs(
            thread_id=thread_id,
            limit=limit,
            order=order,
            after=after,
            before=before
        )

        if not runs:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        return runs
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/runs/{run_id}", response_model=RunObject)
async def retrieve_run(
    thread_id: str,
    run_id: str,
    req: Request
) -> RunObject:
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        # Retrieve the run
        run = await db.get_run_by_id(thread_id, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found in thread.")
        
        return run
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.post("/threads/{thread_id}/runs/{run_id}", response_model=RunObject)
async def modify_run(thread_id: str, run_id: str, request: ModifyRunRequest, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store
        
        # Verify the run exists and update metadata
        run = await db.update_run(thread_id, run_id, request)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found in thread.")
        
        return run
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/threads/{thread_id}/runs/{run_id}/submit_tool_outputs", response_model=RunObject)
async def submit_tool_outputs_to_run(thread_id: str, run_id: str, request: SubmitRunToolOutputRequest, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        thread = await db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        run = await db.get_run_by_id(thread_id, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.status != run_status.REQUIRES_ACTION:
            raise HTTPException(status_code=400, detail="Run is not in 'requires_action' status.")

        latest_run_step = await db.get_latest_run_step_by_run_id(run_id)
        if not latest_run_step:
            raise HTTPException(status_code=404, detail="No run step found for this run.")

        step_details = latest_run_step.step_details
        tool_calls = getattr(step_details, "tool_calls", None)
        if tool_calls is None and isinstance(step_details, dict):
            tool_calls = step_details.get("tool_calls")
        if not tool_calls:
            raise HTTPException(status_code=400, detail="No tool calls found in the latest run step.")

        for tool_output in request.tool_outputs:
            for tool_call in tool_calls:
                # resolve tool_call_id from object or dict
                if isinstance(tool_call, dict):
                    tool_call_id = tool_call.get("id")
                elif hasattr(tool_call, "model_dump"):
                    tool_call_id = tool_call.model_dump().get("id")
                else:
                    tool_call_id = getattr(tool_call, "id", None)

                if tool_call_id != tool_output.tool_call_id:
                    continue

                # set output back on the tool call
                if isinstance(tool_call, dict):
                    tool_call.setdefault("function", {})["output"] = tool_output.output
                elif hasattr(tool_call, "function"):
                    tool_call.function.output = tool_output.output
                elif hasattr(tool_call, "model_dump"):
                    data = tool_call.model_dump()
                    data.setdefault("function", {})["output"] = tool_output.output
                    tool_call = data

        if hasattr(step_details, "tool_calls"):
            step_details.tool_calls = tool_calls
        else:
            latest_run_step.step_details = {"tool_calls": tool_calls}

        latest_run_step = await db.update_run_step_status(latest_run_step.id, run_step_status.COMPLETED)
        await db.update_run_status(thread_id, run_id, run_status.IN_PROGRESS)
        run = await db.get_run_by_id(thread_id, run_id)

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
            metadata=run.metadata,
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

@router.post("/threads/{thread_id}/runs/{run_id}/cancel", response_model=RunObject)
async def cancel_run(thread_id: str, run_id: str, req: Request):
    try:
        db: BaseDataStore = req.app.state.data_store

        run = await db.get_run_by_id(thread_id, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.status != run_status.QUEUED:
            raise HTTPException(status_code=400, detail="Run cannot be canceled unless it is in 'queued' status.")
        
        run = await db.update_run_status(thread_id, run_id, run_status.CANCELLED)

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
            metadata=run.metadata,
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

async def handle_stream_event(event: dict, db: BaseDataStore, thread_id: str, request: RunCreateRequest, assistant: Assistant):
    evt = event.get("event")
    data = event.get("data", {}) or {}

    if evt == event_type.THREAD_CREATED:
        thread_req = CreateThreadRequest(
            tool_resources=data.get("tool_resource", {}),
            metadat={**data.get("metadata", {}), "thread_id": data.get("id", thread_id)},
            messages=[]
        )
        await db.insert_thread(thread_req)

    elif evt == event_type.RUN_CREATED:
        await db.insert_run(thread_id, request, assistant)

    elif evt in {
        event_type.RUN_QUEUED,
        event_type.RUN_IN_PROGRESS,
        event_type.RUN_REQUIRES_ACTION,
        event_type.RUN_COMPLETED,
        event_type.RUN_FAILED,
        event_type.RUN_CANCELLING,
        event_type.RUN_CANCELLED,
        event_type.RUN_EXPIRED,
    }:
        status_map = {
            event_type.RUN_QUEUED: run_status.QUEUED,
            event_type.RUN_IN_PROGRESS: run_status.IN_PROGRESS,
            event_type.RUN_REQUIRES_ACTION: run_status.REQUIRES_ACTION,
            event_type.RUN_COMPLETED: run_status.COMPLETED,
            event_type.RUN_FAILED: run_status.FAILED,
            event_type.RUN_CANCELLING: run_status.CANCELLING,
            event_type.RUN_CANCELLED: run_status.CANCELLED,
            event_type.RUN_EXPIRED: run_status.EXPIRED,
        }
        run_id = data.get("id")
        if run_id:
            await db.update_run_status(thread_id, run_id, status_map[evt], data.get("error"))

    elif evt == event_type.RUN_STEP_CREATED:
        step_obj = RunStepObject.model_validate(data)
        step_req = CreateRunStepRequest(
            assistant_id=step_obj.assistant_id,
            metadata={**(step_obj.metadata or {}), "step_id": step_obj.id},
            step_details=step_obj.step_details,
        )
        await db.insert_run_step(step_obj.thread_id, step_obj.run_id, step_req, status=step_obj.status)

    elif evt in {
        event_type.RUN_STEP_IN_PROGRESS,
        event_type.RUN_STEP_COMPLETED,
        event_type.RUN_STEP_FAILED,
        event_type.RUN_STEP_CANCELLED,
        event_type.RUN_STEP_EXPIRED,
    }:
        step_status_map = {
            event_type.RUN_STEP_IN_PROGRESS: run_step_status.IN_PROGRESS,
            event_type.RUN_STEP_COMPLETED: run_step_status.COMPLETED,
            event_type.RUN_STEP_FAILED: run_step_status.FAILED,
            event_type.RUN_STEP_CANCELLED: run_step_status.CANCELLED,
            event_type.RUN_STEP_EXPIRED: run_step_status.EXPIRED,
        }
        step_id = data.get("id")
        if step_id:
            await db.update_run_step_status(step_id, step_status_map[evt], error=data.get("error"))

    elif evt == event_type.ERROR:
        run_id = data.get("id")
        if run_id:
            await db.update_run_status(thread_id, run_id, run_status.FAILED, data.get("error"))