from fastapi import APIRouter, HTTPException
from llamphouse.core.database.database import DatabaseManager
from ..types.run_step import RunStepObject, RunStepListResponse
from typing import Optional

router = APIRouter()


@router.get("/threads/{thread_id}/runs/{run_id}/steps", response_model=RunStepListResponse)
async def list_run_steps(thread_id: str, run_id: str, limit: int = 20, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None):
    try:
        db = DatabaseManager()
        run_steps = db.list_run_steps(
            thread_id=thread_id,
            run_id=run_id,
            limit=limit + 1,
            order=order,
            after=after,
            before=before
        )
        print(len(run_steps))
        has_more = len(run_steps) > limit
        first_id = run_steps[0].id if run_steps else None
        last_id = run_steps[-1].id if run_steps else None
        return  RunStepListResponse(
                    object="list",
                    data=[
                        RunStepObject(
                            id=run_step.id,
                            assistant_id=run_step.assistant_id,
                            cancelled_at=run_step.cancelled_at,
                            completed_at=run_step.completed_at,
                            created_at=int(run_step.created_at.timestamp()),
                            expires_at=run_step.expired_at,
                            failed_at=run_step.failed_at,
                            last_error=run_step.last_error,
                            metadata=run_step.meta,
                            run_id=run_step.run_id,
                            status=run_step.status,
                            step_details=run_step.step_details,
                            thread_id=run_step.thread_id,
                            type=run_step.type,
                            usage=run_step.usage,
                        )
                        for run_step in run_steps
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



@router.get("/threads/{thread_id}/runs/{run_id}/steps/{step_id}", response_model=RunStepObject)
async def retrieve_run_step(thread_id: str, run_id: str, step_id: str):
    try:
        db = DatabaseManager()
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found.")
        
        run = db.get_run_by_id(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        
        run_step = db.get_run_step_by_id(step_id)
        if not run_step:
            raise HTTPException(status_code=404, detail="Run step not found.")
        return  RunStepObject(
                    id=run_step.id,
                    assistant_id=run_step.assistant_id,
                    cancelled_at=run_step.cancelled_at,
                    completed_at=run_step.completed_at,
                    created_at=int(run_step.created_at.timestamp()),
                    expires_at=run_step.expired_at,
                    failed_at=run_step.failed_at,
                    last_error=run_step.last_error,
                    metadata=run_step.meta,
                    run_id=run_step.run_id,
                    status=run_step.status,
                    step_details=run_step.step_details,
                    thread_id=run_step.thread_id,
                    type=run_step.type,
                    usage=run_step.usage,
                )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        db.session.close()
