from fastapi import APIRouter, HTTPException, Request
from llamphouse.core.database.database import DatabaseManager
from fastapi.responses import StreamingResponse
from ..types.run import RunObject, RunCreateRequest, CreateThreadAndRunRequest, RunListResponse, ModifyRunRequest, SubmitRunToolOutputRequest
from ..types.enum import run_status, run_step_status, message_status, event_type
from ..assistant import Assistant
from typing import List, Optional
import time
import asyncio
import json
import queue
import janus

router = APIRouter()

@router.post("/threads/{thread_id}/runs", response_model=RunObject)
async def create_run(
    thread_id: str,
    request: RunCreateRequest,
    req: Request
) -> RunObject:
    try:
        db = DatabaseManager()
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        assistants = req.app.state.assistants
        assistant = get_assistant_by_id(assistants, request.assistant_id)
        # store run in db
        run = db.insert_run(thread_id, run=request, assistant=assistant)

        # Check if the task exists
        task_key = f"{request.assistant_id}:{thread_id}"

        # Check if the task exists
        # task_key = f"{request.assistant_id}:{thread.id}"
        if task_key not in req.app.state.task_queues:
            req.app.state.task_queues[task_key] = janus.Queue()

        # check if stream is enabled
        if request.stream:
            output_queue: janus.Queue = req.app.state.task_queues[task_key]

            # Streaming generator for SSE
            async def event_stream():
                while True:
                    try:
                        event = await asyncio.wait_for(output_queue.async_q.get(), timeout=30.0)  # Set timeout in seconds
                        if event is None:  # Stream completion signal
                            break
                        yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
                        if event['event'] == event_type.DONE:
                            print("Stream completed: DONE event received", flush=True)
                            break
                    except asyncio.TimeoutError:
                        print("TimeoutError: No event received within the timeout period", flush=True)
                        yield f'''event: error\ndata: {json.dumps({
                            "error": "TimeoutError",
                            "message": "No event received within the timeout period"
                        })}\n\n'''
                        break
                    except Exception as e:
                        print(event, flush=True)
                        print(f"Error in event stream: {str(e)}", flush=True)
                        yield f'''event: error\ndata: {json.dumps({
                            "error": "InternalError",
                            "message": str(e)
                        })}\n\n'''
                        break
                # Cleanup the queue after the stream ends
                # Clear all remaining items in the janus queue
                while not output_queue.async_q.empty():
                    await output_queue.async_q.get()
                    output_queue.async_q.task_done()
                # Remove the task queue after the stream ends
                if task_key in req.app.state.task_queues:
                    del req.app.state.task_queues[task_key]

            # Return the streaming response
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        # wait_time = 0
        # max_wait = 3
        # poll_interval = 0.1

        # # Wait until worker creates the queue
        # while task_key not in req.app.state.task_queues and wait_time < max_wait:
        #     await asyncio.sleep(poll_interval)
        #     wait_time += poll_interval
        #     if task_key in req.app.state.task_queues:
        #         break
        # if task_key not in req.app.state.task_queues:
        #     raise HTTPException(status_code=500, detail="Worker did not register a task queue in time.")

        # output_queue = req.app.state.task_queues[task_key]

        # # check if stream is enabled
        # if request.stream:
        #     print(f"Stream enabled for run {run.id} in thread {thread_id}")

        #     # Async Worker
        #     async def async_event_stream():
        #         while True:
        #             event = None
        #             try:
        #                 event = await asyncio.wait_for(output_queue.get(), timeout=10.0)
        #                 if event is None:  # Stream completion signal
        #                     break
        #                 yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"

        #                 handle_stream_event(event, db, thread_id, request, assistant)
        #                 if event['event'] == event_type.DONE:
        #                     break
                            
        #             except asyncio.TimeoutError:
        #                 print("TimeoutError: No event received within the timeout period")
        #                 yield f'''event: error\ndata: {json.dumps({
        #                     "error": "TimeoutError",
        #                     "message": "No event received within the timeout period"
        #                 })}\n\n'''
        #                 event_id = event['data']["id"] if event else "No ID"
        #                 db.update_run_status(event_id, run_status.EXPIRED)
        #                 break

        #     # Thread Worker
        #     def thread_event_stream():
        #         while True:
        #             try:
        #                 event = output_queue.get(timeout=30.0)
        #                 if event is None:
        #                     print("Stream completed: None event received")
        #                     break
        #                 # print(f"Event: {event['event']}", flush=True)
        #                 yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        #                 # yield event
        #                 handle_stream_event(event, db, thread_id, request, assistant)
        #                 if event['event'] == event_type.DONE:
        #                     print("Stream completed: DONE event received")
        #                     break
        #             except queue.Empty:
        #                 print("TimeoutError: No event received within the timeout period")
        #                 yield f'''event: error\ndata: {json.dumps({
        #                     "error": "TimeoutError",
        #                     "message": "No event received within the timeout period"
        #                 })}\n\n'''
        #                 # event_id = event['data']["id"] if event else "No ID"
        #                 # db.update_run_status(event_id, run_status.EXPIRED)
        #                 break

        #     if isinstance(output_queue, asyncio.Queue):
        #         return StreamingResponse(async_event_stream(), media_type="text/event-stream")
        #     elif isinstance(output_queue, queue.Queue):
        #         return StreamingResponse(thread_event_stream(), media_type="text/event-stream")
        #     else:
        #         raise HTTPException(status_code=500, detail="Unsupported queue type for streaming.")
        
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
    finally:
        db.session.close()

@router.post("/threads/runs", response_model=RunObject)
async def create_thread_and_run(request: CreateThreadAndRunRequest, req: Request):
    try:
        db = DatabaseManager()
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
            # print(f"Creating queue for task {task_key} in RUN")
            req.app.state.task_queues[task_key] = asyncio.Queue(maxsize=1)
            # raise HTTPException(status_code=404, detail="Task not found")

        # check if stream is enabled
        if request.stream:
            print(f"Stream enabled for run {run.id} in thread {thread.id} THREAD_AND_RUN")
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
    finally:
        db.session.close()
  
def get_assistant_by_id(assistants: List[Assistant], assistant_id: str) -> Assistant:
    assistant = next((assistant for assistant in assistants if assistant.id == assistant_id), None)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return assistant

@router.get("/threads/{thread_id}/runs", response_model=RunListResponse)
async def list_runs(thread_id: str, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None) -> RunObject:
    try:
        db = DatabaseManager()
        runs = db.list_runs_by_thread_id(
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
    finally:
        db.session.close()

@router.get("/threads/{thread_id}/runs/{run_id}", response_model=RunObject)
async def retrieve_run(
    thread_id: str,
    run_id: str,
) -> RunObject:
    try:
        db = DatabaseManager()
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
    finally:
        db.session.close()
    
@router.post("/threads/{thread_id}/runs/{run_id}", response_model=RunObject)
async def modify_run(thread_id: str, run_id: str, request: ModifyRunRequest):
    try:
        db = DatabaseManager()
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
    finally:
        db.session.close()

@router.post("/threads/{thread_id}/runs/{run_id}/submit_tool_outputs", response_model=RunObject)
async def submit_tool_outputs_to_run(thread_id: str, run_id: str, request: SubmitRunToolOutputRequest):
    try:
        db = DatabaseManager()
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        run = db.get_run_by_id(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.status != run_status.REQUIRES_ACTION:
            raise HTTPException(status_code=400, detail="Run is not in 'requires_action' status.")

        latest_run_step = db.get_latest_run_step_by_run_id(run_id)
        if not latest_run_step:
            raise HTTPException(status_code=404, detail="No run step found for this run.")

        if not latest_run_step.step_details or "tool_calls" not in latest_run_step.step_details:
            raise HTTPException(status_code=400, detail="No tool calls found in the latest run step.")
        
        tool_calls = latest_run_step.step_details["tool_calls"]
        
        for tool_output in request.tool_outputs:
            for tool_call in tool_calls:
                if tool_call["id"] == tool_output.tool_call_id:
                    tool_call["output"] = tool_output.output
                    break

        latest_run_step.step_details = {"tool_calls": tool_calls}
        latest_run_step.status = run_step_status.COMPLETED
        db.session.commit()

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
    finally:
        db.session.close()

@router.post("/threads/{thread_id}/runs/{run_id}/cancel", response_model=RunObject)
async def cancel_run(thread_id: str, run_id: str):
    try:
        db = DatabaseManager()
        run = db.get_run_by_id(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run.status != run_status.QUEUED:
            raise HTTPException(status_code=400, detail="Run cannot be canceled unless it is in 'queued' status.")
        
        run.status = run_status.CANCELLED
        db.session.commit()

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
    finally:
        db.session.close()

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