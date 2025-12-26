from fastapi import APIRouter, HTTPException, Request
from ..types.assistant import AssistantListResponse, AssistantObject
from typing import Optional, Literal

router = APIRouter()

@router.get("/assistants", response_model=AssistantListResponse)
async def list_assistants(
            req: Request,
            after: Optional[str] = None, # Optional assistant id to start after
            before: Optional[str] = None, # Optional assistant id to end before
            limit: Optional[int] = 10,
            order: Optional[Literal["asc", "desc"]] = "asc"
        ):
    try:
        # Get assistants from app state
        assistants = req.app.state.assistants

        # Apply ordering and pagination
        if order == "desc":
            assistants = assistants[::-1]
        # Apply pagination
        if after:
            try:
                index = next(i for i, assistant in enumerate(assistants) if assistant.id == after)
                assistants = assistants[index + 1:]
            except StopIteration:
                assistants = []
        if before:
            try:
                index = next(i for i, assistant in enumerate(assistants) if assistant.id == before)
                assistants = assistants[:index]
            except StopIteration:
                assistants = []
        # Apply limit
        assistants = assistants[:limit]
        return AssistantListResponse(
            data=[AssistantObject(
                created_at=assistant.created_at,
                id=assistant.id,
                object=assistant.object,
                model=assistant.model,
                name=assistant.name,
                description=assistant.description,
                temperature=assistant.temperature,
                top_p=assistant.top_p,
                tools=assistant.tools
            ) for assistant in assistants],
            after=after,
            before=before
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/assistants/{assistant_id}", response_model=AssistantObject)
async def retrieve_assistant(req: Request, assistant_id: str):
    try:
        assistant = next((a for a in req.app.state.assistants if a.id == assistant_id), None)
        if assistant is None:
            raise HTTPException(status_code=404, detail="Assistant not found.")

        return AssistantObject(
            created_at=assistant.created_at,
            id=assistant.id,
            object=assistant.object,
            model=assistant.model,
            name=assistant.name,
            description=assistant.description,
            temperature=assistant.temperature,
            top_p=assistant.top_p,
            tools=assistant.tools
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
