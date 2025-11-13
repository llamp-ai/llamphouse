from fastapi import APIRouter, HTTPException, Request
from ..types.thread import ThreadObject, CreateThreadRequest, ModifyThreadRequest, DeleteThreadResponse
from ..data_stores.base_data_store import BaseDataStore
import time
import logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/threads", response_model=ThreadObject)
async def create_thread(request: CreateThreadRequest, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        # Insert the thread
        thread = await db.insert_thread(request)
        if not thread:
            raise HTTPException(status_code=400, detail="Thread with the same ID already exists.")
        
        return thread
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/threads/{thread_id}", response_model=ThreadObject)
async def retrieve_thread(thread_id: str, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        # Retrieve the thread
        thread = await db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        return thread
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/threads/{thread_id}", response_model=ThreadObject)
async def modify_thread(thread_id: str, request: ModifyThreadRequest, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        # Retrieve the thread
        thread = await db.update_thread(thread_id, request)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        return thread
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/threads/{thread_id}", response_model=DeleteThreadResponse)
async def delete_thread(thread_id: str, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        thread_id = await db.delete_thread(thread_id)
        if not thread_id:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        return DeleteThreadResponse(
            id=thread_id,
            deleted=True
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")