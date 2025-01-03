from fastapi import APIRouter, HTTPException
from llamphouse.core.database import database as db
from ..types.message import DeleteMessageResponse, CreateMessageRequest, MessageListResponse, Attachment, MessageObject, TextContent, ImageFileContent, ModifyMessageRequest
from typing import List, Optional


router = APIRouter()

@router.post("/threads/{thread_id}/messages", response_model=MessageObject)
async def create_message(thread_id: str, request: CreateMessageRequest):
    try:
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        message = db.insert_message(thread_id, request)
        
        return MessageObject(
                    id=message.id,
                    role=message.role,
                    content=[
                        TextContent(text=message.content) if isinstance(message.content, str) else ImageFileContent(image_file=message.content)
                        for message.content in [message.content]
                    ],
                    metadata=message.meta,
                    status=message.status,
                    incomplete_details=message.incomplete_details,
                    completed_at=message.completed_at,
                    incomplete_at=message.incomplete_at,
                    assistant_id=message.assistant_id,
                    run_id=message.run_id,
                    attachments=[
                        Attachment(file_id=attachment['file_id'], tool=attachment.get('tool')) for attachment in (message.attachments or [])
                    ],
                    created_at=int(message.created_at.timestamp()),
                    thread_id=message.thread_id
                ) 
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/messages", response_model=MessageListResponse)
async def list_messages(thread_id: str, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None):
    try:
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        messages = db.get_messages_by_thread_id(
            thread_id=thread_id,
            limit=limit + 1,
            order=order,
            after=after,
            before=before
        )
        has_more = len(messages) > limit
        first_id = messages[0].id if messages else None
        last_id = messages[-1].id if messages else None

        return MessageListResponse(
            object="list",
            data=[
                MessageObject(
                    id=msg.id,
                    role=msg.role,
                    content=[
                        TextContent(text=msg.content) if isinstance(msg.content, str) else ImageFileContent(image_file=msg.content)
                        for msg.content in [msg.content]
                    ],
                    metadata=msg.meta,
                    status=msg.status,
                    incomplete_details=msg.incomplete_details,
                    completed_at=msg.completed_at,
                    incomplete_at=msg.incomplete_at,
                    assistant_id=msg.assistant_id,
                    run_id=msg.run_id,
                    attachments=[
                        Attachment(file_id=attachment['file_id'], tool=attachment.get('tool')) for attachment in (msg.attachments or [])
                    ],
                    created_at=int(msg.created_at.timestamp()),
                    thread_id=msg.thread_id
                ) 
                for msg in messages
            ],
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def retrieve_message(thread_id: str, message_id: str):
    try:
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        message = db.get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found.")
        
        return  MessageObject(
                    id=message.id,
                    role=message.role,
                    content=[
                        TextContent(text=message.content) if isinstance(message.content, str) else ImageFileContent(image_file=message.content)
                        for message.content in [message.content]
                    ],
                    metadata=message.meta,
                    status=message.status,
                    incomplete_details=message.incomplete_details,
                    completed_at=message.completed_at,
                    incomplete_at=message.incomplete_at,
                    assistant_id=message.assistant_id,
                    run_id=message.run_id,
                    attachments=[
                        Attachment(file_id=attachment['file_id'], tool=attachment.get('tool')) for attachment in (message.attachments or [])
                    ],
                    created_at=int(message.created_at.timestamp()),
                    thread_id=message.thread_id
                ) 
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def modify_message(thread_id: str, message_id: str, request: ModifyMessageRequest):
    try:
        message = db.update_message_metadata(thread_id, message_id, request.metadata)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found.")
        
        return  MessageObject(
                    id=message.id,
                    role=message.role,
                    content=[
                        TextContent(text=message.content) if isinstance(message.content, str) else ImageFileContent(image_file=message.content)
                        for message.content in [message.content]
                    ],
                    metadata=message.meta,
                    status=message.status,
                    incomplete_details=message.incomplete_details,
                    completed_at=message.completed_at,
                    incomplete_at=message.incomplete_at,
                    assistant_id=message.assistant_id,
                    run_id=message.run_id,
                    attachments=[
                        Attachment(file_id=attachment['file_id'], tool=attachment.get('tool')) for attachment in (message.attachments or [])
                    ],
                    created_at=int(message.created_at.timestamp()),
                    thread_id=message.thread_id
                ) 
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/threads/{thread_id}/messages/{message_id}", response_model=DeleteMessageResponse)
async def delete_message(thread_id: str, message_id: str):
    try:
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        message = db.get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found.")

        deleted = db.delete_message(thread_id, message_id)
        if not deleted:
            return DeleteMessageResponse(
                id=thread_id,
                deleted=False
            )
        
        return DeleteMessageResponse(
            id=thread_id,
            deleted=True
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
