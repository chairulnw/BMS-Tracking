import cv2
from fastapi import APIRouter, HTTPException, Query, Request, Response

router = APIRouter(prefix="/snapshot", tags=["snapshot"])


@router.get("/{camera_id}")
def get_snapshot(
    camera_id: str,
    request: Request,
    rtsp_url: str | None = Query(None),
) -> Response:
    """Return frame JPEG terbaru dari kamera.
    Prioritas: running stream → direct RTSP capture (jika rtsp_url diberikan)."""
    manager = request.app.state.stream_manager
    frame   = manager.get_snapshot(camera_id)

    if frame is None and rtsp_url:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise HTTPException(404, "Tidak dapat mengambil frame dari RTSP")

    if frame is None:
        raise HTTPException(404, "Kamera tidak ditemukan atau stream belum berjalan")

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise HTTPException(500, "Gagal encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")
