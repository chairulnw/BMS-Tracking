from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.schemas import StreamStartRequest, StreamStatusResponse

router = APIRouter(prefix="/stream")


@router.post("/start", response_model=StreamStatusResponse)
def stream_start(req: StreamStartRequest, request: Request) -> StreamStatusResponse:
    manager = request.app.state.stream_manager
    try:
        manager.start(
            yolo_model=request.app.state.yolo_model,
            reid_model=request.app.state.reid_model,
            conf_threshold=req.conf_threshold,
            reid_threshold=req.reid_threshold,
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


# ── Ambiguous ReID cases ──────────────────────────────────────────────────────

@router.get("/ambiguous")
def list_ambiguous(request: Request) -> list[dict]:
    db = request.app.state.stream_manager.shared_db
    if db is None:
        return []
    return db.get_ambiguous_list()


@router.post("/ambiguous/{amb_id}/resolve")
def resolve_ambiguous(
    amb_id: str, request: Request,
    action: str = "confirm",
    target: str = "",
) -> dict:
    """action: 'confirm' | 'assign' (+ target=label) | 'reject'."""
    db = request.app.state.stream_manager.shared_db
    if db is None:
        raise HTTPException(503, "Stream belum berjalan")
    if action not in ("confirm", "assign", "reject"):
        raise HTTPException(400, "action harus 'confirm', 'assign', atau 'reject'")
    if action == "assign" and not target:
        raise HTTPException(400, "action 'assign' memerlukan query param 'target' (label orang)")
    ok = db.resolve_ambiguous(amb_id, action, target_label=target)
    if not ok:
        raise HTTPException(404, f"Ambiguous case '{amb_id}' tidak ditemukan atau target tidak valid")
    return {"status": "ok", "amb_id": amb_id, "action": action}


@router.get("/ambiguous/{amb_id}/snapshot")
def ambiguous_snapshot(amb_id: str) -> FileResponse:
    path = Path("thumbnails/ambiguous") / f"{amb_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "Snapshot tidak ditemukan")
    return FileResponse(str(path), media_type="image/jpeg")
