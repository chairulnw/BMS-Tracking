import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/thumbnails", tags=["thumbnails"])

# Resolve path relative to this file: backend/app/routers/ → BMS-IIP/ai-service/thumbnails/
_DEFAULT_DIR = Path(__file__).parents[3] / "ai-service" / "thumbnails"
THUMBNAILS_DIR = Path(os.getenv("THUMBNAILS_DIR", str(_DEFAULT_DIR)))


@router.get("/events/{filename}")
async def get_event_snapshot(filename: str) -> FileResponse:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = THUMBNAILS_DIR / "events" / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Event snapshot not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{filename}")
async def get_thumbnail(filename: str) -> FileResponse:
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = THUMBNAILS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path, media_type="image/jpeg")
