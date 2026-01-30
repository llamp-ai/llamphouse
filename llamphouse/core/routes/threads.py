from fastapi import APIRouter, HTTPException, Request
from ..types.thread import ThreadObject, CreateThreadRequest, ModifyThreadRequest, DeleteThreadResponse
from ..data_stores.base_data_store import BaseDataStore
import json
import logging
from opentelemetry import propagate
from opentelemetry.trace import Status, StatusCode
from ..tracing import get_tracer, span_context

tracer = get_tracer("llamphouse.routes.threads")
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/threads", response_model=ThreadObject)
async def create_thread(request: CreateThreadRequest, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.threads.create",
        context=ctx,
        attributes={
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "thread.create",
        }
    ) as span:
        try:
            input_payload = request.model_dump(mode="json")
            span.set_attribute("input.value", json.dumps(input_payload, ensure_ascii=True, default=str))

            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store

            # Insert the thread
            thread = await db.insert_thread(request)
            if not thread:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("thread.already_exists")
                raise HTTPException(status_code=400, detail="Thread with the same ID already exists.")
            
            span.set_attribute("session.id", thread.id)
            span.set_attribute(
                "output.value",
                json.dumps({"thread_id": thread.id}, ensure_ascii=True)
            )
            span.set_status(Status(StatusCode.OK))
            return thread
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}", response_model=ThreadObject)
async def retrieve_thread(thread_id: str, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.threads.retrieve",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "thread.get",
        },
    ) as span:
        try:
            span.set_attribute(
                "input.value",
                json.dumps({"thread_id": thread_id}, ensure_ascii=True),
            )
            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store

            # Retrieve the thread
            thread = await db.get_thread_by_id(thread_id)
            if not thread:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("thread.not_found")
                raise HTTPException(status_code=404, detail="Thread not found.")
            
            span.set_attribute(
                "output.value",
                json.dumps({"thread_id": thread.id}, ensure_ascii=True),
            )
            span.set_status(Status(StatusCode.OK))
            return thread
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/threads/{thread_id}", response_model=ThreadObject)
async def modify_thread(thread_id: str, request: ModifyThreadRequest, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.threads.modify",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "thread.modify",
        },
    ) as span:
        try:
            input_payload = request.model_dump(mode="json")
            input_payload["thread_id"] = thread_id
            span.set_attribute(
                "input.value",
                json.dumps(input_payload, ensure_ascii=True, default=str),
            )
            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store

            # Retrieve the thread
            thread = await db.update_thread(thread_id, request)
            if not thread:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("thread.not_found")
                raise HTTPException(status_code=404, detail="Thread not found.")
            span.set_attribute(
                "output.value",
                json.dumps({"thread_id": thread.id}, ensure_ascii=True),
            )
            span.set_status(Status(StatusCode.OK))
            return thread
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/threads/{thread_id}", response_model=DeleteThreadResponse)
async def delete_thread(thread_id: str, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.threads.delete",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "thread.delete",
        },
    ) as span:
        try:
            span.set_attribute(
                "input.value",
                json.dumps({"thread_id": thread_id}, ensure_ascii=True),
            )
            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store

            deleted_id = await db.delete_thread(thread_id)
            if not deleted_id:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("thread.not_found")
                raise HTTPException(status_code=404, detail="Thread not found.")
            
            span.set_attribute(
                "output.value",
                json.dumps({"thread_id": deleted_id, "deleted": True}, ensure_ascii=True),
            )
            span.set_status(Status(StatusCode.OK))
            return DeleteThreadResponse(
                id=deleted_id,
                deleted=True
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")