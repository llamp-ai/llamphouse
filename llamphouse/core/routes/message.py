from fastapi import APIRouter, HTTPException, Request

from ..types.list import ListResponse
from ..data_stores.base_data_store import BaseDataStore
from ..types.message import DeleteMessageResponse, CreateMessageRequest, Attachment, MessageObject, TextContent, ImageFileContent, ModifyMessageRequest
from typing import List, Optional
from opentelemetry import propagate
from opentelemetry.trace import Status, StatusCode
from ..tracing import get_tracer, span_context
import json

tracer = get_tracer("llamphouse.routes.message")
router = APIRouter()

@router.post("/threads/{thread_id}/messages", response_model=MessageObject)
async def create_message(thread_id: str, request: CreateMessageRequest, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.messages.create",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "message.create",
            "gen_ai.message.role": request.role,
            }
        ) as span:
        try:
            input_payload = {
                "thread_id": thread_id,
                "role": request.role,
                "content": request.content,
            }
            span.set_attribute("input.value", json.dumps(input_payload, ensure_ascii=True, default=str))

            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store
            
            message = await db.insert_message(thread_id, request)
            if not message:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("thread.not_found")
                raise HTTPException(status_code=404, detail="Thread not found.")

            span.set_attribute("message.id", message.id)
            span.set_attribute(
                "output.value",
                json.dumps({"message_id": message.id, "status": message.status}, ensure_ascii=True)
            )
            span.set_status(Status(StatusCode.OK))
            return message
        
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/messages", response_model=ListResponse)
async def list_messages(thread_id: str, req: Request, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.messages.list",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "message.list",
        },
    ) as span:
        try:
            span.set_attribute(
                "input.value",
                json.dumps(
                    {"thread_id": thread_id, "limit": limit, "order": order, "after": after, "before": before},
                    ensure_ascii=True,
                    default=str,
                ),
            )
            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store
            
            messages: ListResponse = await db.list_messages(
                thread_id=thread_id,
                limit=limit,
                order=order,
                after=after,
                before=before
            )
            if not messages:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("thread.not_found")
                raise HTTPException(status_code=404, detail="Thread not found.")

            span.set_attribute(
                "output.value",
                json.dumps({"count": len(messages.data) if messages else 0}, ensure_ascii=True),
            )
            span.set_status(Status(StatusCode.OK))
            return messages
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def retrieve_message(thread_id: str, message_id: str, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.messages.retrieve",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "message.get",
        },
    ) as span:
        try:
            span.set_attribute(
                "input.value",
                json.dumps({"thread_id": thread_id, "message_id": message_id}, ensure_ascii=True),
            )
            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store

            message = await db.get_message_by_id(thread_id, message_id)
            if not message:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("message.not_found")
                raise HTTPException(status_code=404, detail="Message not found in thread.")
            span.set_attribute(
                "output.value",
                json.dumps({"message_id": message.id, "status": message.status}, ensure_ascii=True),
            )
            span.set_status(Status(StatusCode.OK))
            return message
        
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def modify_message(thread_id: str, message_id: str, request: ModifyMessageRequest, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.messages.modify",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "message.modify",
        },
    ) as span:
        try:
            input_payload = request.model_dump(mode="json")
            input_payload.update({"thread_id": thread_id, "message_id": message_id})
            span.set_attribute(
                "input.value",
                json.dumps(input_payload, ensure_ascii=True, default=str),
            )
            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store

            message = await db.update_message(thread_id, message_id, request)
            if not message:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("message.not_found")
                raise HTTPException(status_code=404, detail="Message not found in thread.")

            span.set_attribute(
                "output.value",
                json.dumps({"message_id": message.id, "status": message.status}, ensure_ascii=True),
            )
            span.set_status(Status(StatusCode.OK))
            return message

        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/threads/{thread_id}/messages/{message_id}", response_model=DeleteMessageResponse)
async def delete_message(thread_id: str, message_id: str, req: Request):
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None

    with span_context(
        tracer,
        "llamphouse.messages.delete",
        context=ctx,
        attributes={
            "session.id": thread_id,
            "gen_ai.system": "llamphouse",
            "gen_ai.operation.name": "message.delete",
        },
    ) as span:
        try:
            span.set_attribute(
                "input.value",
                json.dumps({"thread_id": thread_id, "message_id": message_id}, ensure_ascii=True),
            )
            # Get the data store from the app state
            db: BaseDataStore = req.app.state.data_store

            message_id = await db.delete_message(thread_id, message_id)
            if not message_id:
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("message.not_found")
                raise HTTPException(status_code=404, detail="Message not found in thread.")
            
            span.set_attribute(
                "output.value",
                json.dumps({"message_id": message_id, "deleted": True}, ensure_ascii=True),
            )
            span.set_status(Status(StatusCode.OK))
            return DeleteMessageResponse(
                id=message_id,
                deleted=True
            )
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
