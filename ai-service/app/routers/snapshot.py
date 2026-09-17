import glob
from pathlib import Path
import cv2
from fastapi import APIRouter, HTTPException, Query, Request, Response

router = APIRouter(prefix="/snapshot", tags=["snapshot"])

_REPO_ROOT = Path(__file__).parents[2]
_CACHE_SNAPSHOT: dict[str, bytes] = {}


def _get_fallback_frame(camera_id: str) -> bytes | None:
    """Ambil frame acuan dari rekaman lokal jika stream RTSP tidak sedang live."""
    if camera_id in _CACHE_SNAPSHOT:
        return _CACHE_SNAPSHOT[camera_id]

    patterns = [
        _REPO_ROOT / "sample_0910" / camera_id / "*.avi",
        _REPO_ROOT / "sample_0910" / camera_id / "*.mp4",
        _REPO_ROOT / "output" / "clips" / f"clip_{camera_id}_*.avi",
        _REPO_ROOT / "output" / "clips" / f"clip_{camera_id}_*.mp4",
        _REPO_ROOT / "sample" / f"{camera_id}_sim" / "*.avi",
        _REPO_ROOT / "sample_0904" / camera_id / "*.avi",
    ]
    for pat in patterns:
        files = glob.glob(str(pat))
        if files:
            files.sort()
            cap = cv2.VideoCapture(files[0])
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    data = buf.tobytes()
                    _CACHE_SNAPSHOT[camera_id] = data
                    return data
    return None


@router.get("/{camera_id}")
def get_snapshot(
    camera_id: str,
    request: Request,
    rtsp_url: str | None = Query(None),
) -> Response:
    """Return frame JPEG terbaru dari kamera.
    Prioritas: running stream → direct RTSP capture (jika rtsp_url diberikan) → rekaman acuan kamera."""
    manager = request.app.state.stream_manager
    frame   = manager.get_snapshot(camera_id)

    if frame is None and rtsp_url:
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, cap_frame = cap.read()
            cap.release()
            if ret and cap_frame is not None:
                frame = cap_frame
        except Exception:
            pass

    if frame is not None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            return Response(content=buf.tobytes(), media_type="image/jpeg")

    # Fallback transparan ke frame rekaman kamera
    cached = _get_fallback_frame(camera_id)
    if cached is not None:
        return Response(content=cached, media_type="image/jpeg")

    raise HTTPException(404, "Kamera tidak ditemukan atau stream belum berjalan")

