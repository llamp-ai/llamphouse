from fastapi import APIRouter, HTTPException
from ..types.thread import ThreadObject, CreateThreadRequest, ModifyThreadRequest, DeleteThreadResponse
from llamphouse.core.database import database as db
import time

router = APIRouter()

@router.post("/threads", response_model=ThreadObject)
async def create_thread(request: CreateThreadRequest):
    thread = db.insert_thread(request)
    if request.messages:
        for msg in request.messages:
            if msg.role not in ["user", "assistant"]:
                raise HTTPException(status_code=400, detail="Invalid role. Must be 'user' or 'assistant'.")
            else:
                db.insert_message(thread_id=thread.id, message=msg)
    
    return ThreadObject(
        id=thread.id,
        created_at=time.mktime(thread.created_at.timetuple()),
        tool_resources=thread.tool_resources,
        metadata=thread.meta
    )

@router.get("/threads/{thread_id}", response_model=ThreadObject)
async def retrieve_thread(thread_id: str):
    thread = db.get_thread_by_id(thread_id)
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found.")
    
    return ThreadObject(
        id=thread.id,
        created_at=time.mktime(thread.created_at.timetuple()),
        tool_resources=thread.tool_resources,
        metadata=thread.meta
    )

@router.put("/threads/{thread_id}", response_model=ThreadObject)
async def modify_thread(thread_id: str, request: ModifyThreadRequest):
    thread = db.get_thread_by_id(thread_id)
    
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found.")
    
    if request.messages:
        for msg in request.messages:
            if msg.role not in ["user", "assistant"]:
                raise HTTPException(status_code=400, detail="Invalid role. Must be 'user' or 'assistant'.")
            
        db.insert_message(thread_id, request.messages)
    
    thread.name = request.name if request.name else thread.name
    db.commit()
    
    return ThreadObject(
        id=thread.id,
        created_at=time.mktime(thread.created_at.timetuple()),
        tool_resources=thread.tool_resources,
        metadata=thread.meta
    )

@router.delete("/threads/{thread_id}", response_model=DeleteThreadResponse)
async def delete_thread(thread_id: str):
    db.delete_messages_by_thread_id(thread_id)
    deleted = db.delete_thread_by_id(thread_id)
    
    if not deleted:
        return DeleteThreadResponse(
            id=thread_id,
            deleted=False
        )
    
    return DeleteThreadResponse(
        id=thread_id,
        deleted=True
    )