from fastapi import APIRouter, HTTPException, Request

from app.schemas import IdentityRecord, RenameRequest

router = APIRouter()


@router.get("/identities", response_model=list[IdentityRecord])
def get_identities(request: Request) -> list[IdentityRecord]:
    return request.app.state.stream_manager.get_identities()


@router.patch("/identities/{name}/rename", response_model=IdentityRecord)
def rename_identity(name: str, req: RenameRequest, request: Request) -> IdentityRecord:
    result = request.app.state.stream_manager.rename_identity(name, req.new_name.strip())
    if result is None:
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found")
    return result
