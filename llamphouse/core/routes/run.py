from fastapi import APIRouter, HTTPException, Request
from llamphouse.core.database.database import DatabaseManager
from fastapi.responses import StreamingResponse
from ..types.run import RunObject, RunCreateRequest, CreateThreadAndRunRequest, ModifyRunRequest, SubmitRunToolOutputRequest
from ..types.enum import run_status, run_step_status, message_status, event_type
from ..types.list import ListResponse
from ..assistant import Assistant
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..streaming.event import Event, DoneEvent, ErrorEvent
from ..data_stores.base_data_store import BaseDataStore
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
        
        # store run in db
        run = await db.insert_run(thread_id, run=request, assistant=assistant, event_queue=output_queue)
        if not run:
            raise HTTPException(status_code=404, detail="Thread not found.")

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
                # Clear all remaining items in the janus queue
                while not output_queue.empty():
                    await output_queue.get()
                    output_queue.task_done()
                # Remove the event queue after the stream ends
                if task_key in req.app.state.event_queues:
                    del req.app.state.event_queues[task_key]

            # Return the streaming response
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        return run
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

        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        run = db.get_run_by_id(run_id)
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
                tool_call_id = getattr(tool_call, "id", None) or tool_call.get("id")
                if tool_call_id == tool_output.tool_call_id:
                    if hasattr(tool_call, "function"):      # Pydantic ToolCall
                        tool_call.function.output = tool_output.output
                    elif isinstance(tool_call, dict):        # dict fallback
                        tool_call["output"] = tool_output.output
                    break

        if hasattr(step_details, "tool_calls"):
            step_details.tool_calls = tool_calls
        else:
            latest_run_step.step_details = {"tool_calls": tool_calls}

        latest_run_step = await db.update_run_step_status(latest_run_step.id, run_step_status.COMPLETED)

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

def handle_stream_event(event, db: DatabaseManager, thread_id, request, assistant):
    if event['event'] == event_type.THREAD_CREATED:
        db.insert_stream_thread(event['data'])
    elif event['event'] == event_type.RUN_CREATED:
        db.insert_run(thread_id, run=request, assistant=assistant)
    elif event['event'] == event_type.RUN_QUEUED:
        db.update_run_status(event['data']["id"], run_status.QUEUED)
    elif event['event'] == event_type.RUN_IN_PROGRESS:
        db.update_run_status(event['data']["id"], run_status.IN_PROGRESS)
    elif event['event'] == event_type.RUN_REQUIRES_ACTION:
        db.update_run_status(event['data']["id"], run_status.REQUIRES_ACTION)
    elif event['event'] == event_type.RUN_COMPLETED:
        db.update_run_status(event['data']["id"], run_status.COMPLETED)
    elif event['event'] == event_type.RUN_FAILED:
        db.update_run_status(event['data']["id"], run_status.FAILED, event['data']["error"])
    elif event['event'] == event_type.RUN_CANCELLING:
        db.update_run_status(event['data']["id"], run_status.CANCELLING)
    elif event['event'] == event_type.RUN_CANCELLED:
        db.update_run_status(event['data']["id"], run_status.CANCELLED)
    elif event['event'] == event_type.RUN_EXPIRED:
        db.update_run_status(event['data']["id"], run_status.EXPIRED)
    elif event['event'] == event_type.RUN_STEP_CREATED:
        db.insert_run_step(event['data'])
    elif event['event'] == event_type.RUN_STEP_IN_PROGRESS:
        db.update_run_step_status(event['data']["id"], run_step_status.IN_PROGRESS)
    elif event['event'] == event_type.RUN_STEP_DELTA:
        pass
    elif event['event'] == event_type.RUN_STEP_COMPLETED:
        db.update_run_step_status(event['data']["id"], run_step_status.COMPLETED)
    elif event['event'] == event_type.RUN_STEP_FAILED:
        db.update_run_step_status(event['data']["id"], run_step_status.FAILED)
    elif event['event'] == event_type.RUN_STEP_CANCELLED:
        db.update_run_step_status(event['data']["id"], run_step_status.CANCELLED)
    elif event['event'] == event_type.RUN_STEP_EXPIRED:
        db.update_run_step_status(event['data']["id"], run_step_status.EXPIRED)
    elif event['event'] == event_type.MESSAGE_CREATED:
        pass
        # db.insert_stream_message(event['data']["id"], event['data']["thread_id"], event['data'])
    # elif event['event'] == event_type.MESSAGE_IN_PROGRESS:
    #     db.update_message_status(event['data']["id"], message_status.IN_PROGRESS)
    # elif event['event'] == event_type.MESSAGE_DELTA:
    #     pass
    #     # db.update_stream_message(event['data']["id"], event['data'].get("text", ""))
    # elif event['event'] == event_type.MESSAGE_COMPLETED:
    #     db.update_message_status(event['data']["id"], message_status.COMPLETED)
    # elif event['event'] == event_type.MESSAGE_INCOMPLETE:
    #     db.update_message_status(event['data']["id"], message_status.INCOMPLETE)
    elif event['event'] == event_type.ERROR:
        db.update_run_status(event['data']["id"], run_status.FAILED, event['data']["error"])
    elif event['event'] == event_type.DONE:
        db.update_run_status(event['data']["id"], run_status.COMPLETED)