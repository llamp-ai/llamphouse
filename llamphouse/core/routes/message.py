from fastapi import APIRouter, HTTPException, Request

from ..types.list import ListResponse
from ..data_stores.base_data_store import BaseDataStore
from ..types.message import DeleteMessageResponse, CreateMessageRequest, Attachment, MessageObject, TextContent, ImageFileContent, ModifyMessageRequest
from typing import List, Optional

router = APIRouter()

@router.post("/threads/{thread_id}/messages", response_model=MessageObject)
async def create_message(thread_id: str, request: CreateMessageRequest, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store
        
        message = await db.insert_message(thread_id, request)
        if not message:
            raise HTTPException(status_code=404, detail="Thread not found.")

        return message
     
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/messages", response_model=ListResponse)
async def list_messages(thread_id: str, req: Request, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None):
    try:
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
            raise HTTPException(status_code=404, detail="Thread not found.")

        return messages
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def retrieve_message(thread_id: str, message_id: str, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        message = await db.get_message_by_id(thread_id, message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found in thread.")
        
        return message
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        db.close()

@router.post("/threads/{thread_id}/messages/{message_id}", response_model=MessageObject)
async def modify_message(thread_id: str, message_id: str, request: ModifyMessageRequest, req: Request):
    try:
         # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        message = await db.update_message(thread_id, message_id, request)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found in thread.")

        return message

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/threads/{thread_id}/messages/{message_id}", response_model=DeleteMessageResponse)
async def delete_message(thread_id: str, message_id: str, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        message_id = await db.delete_message(thread_id, message_id)
        if not message_id:
            raise HTTPException(status_code=404, detail="Message not found in thread.")
        
        return DeleteMessageResponse(
            id=message_id,
            deleted=True
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
