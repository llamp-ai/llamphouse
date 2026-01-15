from fastapi import APIRouter, HTTPException, Request
from ..types.run_step import RunStepObject
from ..types.list import ListResponse
from ..data_stores.base_data_store import BaseDataStore
from typing import Optional

router = APIRouter()


@router.get("/threads/{thread_id}/runs/{run_id}/steps", response_model=ListResponse)
async def list_run_steps(thread_id: str, req: Request, run_id: str, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store

        run_steps = db.list_run_steps(thread_id, run_id, limit, order, after, before)
        if not run_steps:
            raise HTTPException(status_code=404, detail="Could not retrieve run steps for the given thread and run.")
        
        return run_steps
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.get("/threads/{thread_id}/runs/{run_id}/steps/{step_id}", response_model=RunStepObject)
async def retrieve_run_step(thread_id: str, run_id: str, step_id: str, req: Request):
    try:
        # Get the data store from the app state
        db: BaseDataStore = req.app.state.data_store
        
        run_step = db.get_run_step_by_id(thread_id, run_id, step_id)
        if not run_step:
            raise HTTPException(status_code=404, detail="Run step not found in the given thread and run.")
        
        return  run_step
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
