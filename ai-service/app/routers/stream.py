from fastapi import APIRouter, HTTPException, Request

from app.schemas import StreamStartRequest, StreamStatusResponse

router = APIRouter(prefix="/stream")


@router.post("/start", response_model=StreamStatusResponse)
def stream_start(req: StreamStartRequest, request: Request) -> StreamStatusResponse:
    manager = request.app.state.stream_manager
    try:
        manager.start(
            detector_model=request.app.state.detector_model,
            reid_model=request.app.state.reid_model,
            conf_threshold=req.conf_threshold,
            reid_threshold=req.reid_threshold,
            skip_gallery_restore=req.skip_gallery_restore,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return manager.status()


@router.post("/stop", response_model=StreamStatusResponse)
def stream_stop(request: Request) -> StreamStatusResponse:
    manager = request.app.state.stream_manager
    manager.stop()
    return manager.status()


@router.get("/status", response_model=StreamStatusResponse)
def stream_status(request: Request) -> StreamStatusResponse:
    return request.app.state.stream_manager.status()
