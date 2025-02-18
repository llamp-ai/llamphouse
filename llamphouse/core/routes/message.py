from fastapi import APIRouter, HTTPException
from llamphouse.core.database.database import DatabaseManager
from ..types.message import DeleteMessageResponse, CreateMessageRequest, MessageListResponse, Attachment, MessageObject, TextContent, ImageFileContent, ModifyMessageRequest
from typing import List, Optional

router = APIRouter()

@router.post("/threads/{thread_id}/messages", response_model=MessageObject)
async def create_message(thread_id: str, request: CreateMessageRequest):
    try:
        db = DatabaseManager()
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        message = db.insert_message(thread_id, request)
        return MessageObject.from_db_message(message)
     
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        db.session.close()

@router.get("/threads/{thread_id}/messages", response_model=MessageListResponse)
async def list_messages(thread_id: str, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None):
    try:
        db = DatabaseManager()
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        messages = db.list_messages_by_thread_id(
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
            data=[MessageObject.from_db_message(msg) for msg in messages],
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

@router.get("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def retrieve_message(thread_id: str, message_id: str):
    try:
        db = DatabaseManager()
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        message = db.get_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found.")
        
        return MessageObject.from_db_message(message)
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        db.session.close()

@router.post("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def modify_message(thread_id: str, message_id: str, request: ModifyMessageRequest):
    try:
        db = DatabaseManager()
        message = db.update_message_metadata(thread_id, message_id, request.metadata)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found.")
        
        return MessageObject.from_db_message(message)
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        db.session.close()

@router.delete("/threads/{thread_id}/messages/{message_id}", response_model=DeleteMessageResponse)
async def delete_message(thread_id: str, message_id: str):
    try:
        db = DatabaseManager()
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
    finally:
        db.session.close()
