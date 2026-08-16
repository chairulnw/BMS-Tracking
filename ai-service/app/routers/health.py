from fastapi import APIRouter, Request
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    models_loaded = (
        getattr(request.app.state, "detector",  None) is not None
        and getattr(request.app.state, "extractor", None) is not None
    )
    return HealthResponse(status="ok", models_loaded=models_loaded)
