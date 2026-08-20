import psutil
from fastapi import APIRouter, Request
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    models_loaded = (
        getattr(request.app.state, "detector",  None) is not None
        and getattr(request.app.state, "extractor", None) is not None
    )
    metrics = {}
    stream_manager = getattr(request.app.state, "stream_manager", None)
    if stream_manager is not None:
        metrics = stream_manager.get_batch_metrics() or {}
    return HealthResponse(
        status="ok",
        models_loaded=models_loaded,
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_percent=psutil.virtual_memory().percent,
        last_batch_ms=metrics.get("last_batch_ms"),
        avg_batch_ms=metrics.get("avg_batch_ms"),
        cameras_active=metrics.get("cameras_active"),
        cameras_total=metrics.get("cameras_total"),
    )
