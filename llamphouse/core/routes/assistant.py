from fastapi import APIRouter, HTTPException, Depends, Request
from llamphouse.core.database import database as db
from ..types.assistant import AssistantListResponse, AssistantObject
from typing import Optional

router = APIRouter()

@router.get("/assistants", response_model=AssistantListResponse)
async def list_assistants(
            req: Request,
            after: Optional[str] = None,
            before: Optional[str] = None,
            limit: Optional[int] = 10,
            order: Optional[str] = "asc"
        ):
    try:
        assistants = req.app.state.assistants
        if order == "desc":
            assistants = assistants[::-1]

        assistants = assistants[:limit]
        return AssistantListResponse(
            data=[AssistantObject(
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
            raise HTTPException(status_code=404, detail="Assistant not found")

        return AssistantObject(
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
