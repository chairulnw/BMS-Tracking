import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/clips", tags=["clips"])

CLIPS_DIR = Path("output/clips")
# ponytail: transcode-on-request + disk cache, cukup buat admin preview
# sesekali. Kalau nanti dipakai rame-rame, ganti ke pipeline transcode
# background pas clip selesai direkam.
CACHE_DIR = Path("output/clips_web")


def _find_clip(camera_id: str, target: datetime) -> Path | None:
    prefix = f"clip_{camera_id}_"
    best_path: Path | None = None
    best_ts:   datetime | None = None
    for f in CLIPS_DIR.glob(f"{prefix}*.avi"):
        stamp = f.name[len(prefix):-4]
        try:
            ts = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if ts <= target and (best_ts is None or ts > best_ts):
            best_ts, best_path = ts, f
    return best_path


def _transcode(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(dst)],
        check=True, capture_output=True,
    )


@router.get("/{camera_id}")
async def get_clip(camera_id: str, timestamp: datetime = Query(...)) -> FileResponse:
    """Cari klip rekaman kamera ini yang paling dekat (dan gak lebih baru dari)
    timestamp yang diminta, lalu transcode ke MP4 (klip asli MJPEG/AVI gak
    bisa diputar langsung di <video> browser). Hasil transcode di-cache.

    Nama file klip distempel pakai jam lokal (clip_recorder.py: datetime.now()
    naive, WIB) — timestamp yang masuk ke sini dari frontend selalu UTC-aware
    (dari kolom TIMESTAMPTZ). Harus dikonversi ke WIB dulu sebelum dibanding,
    kalau cuma di-strip tzinfo-nya selisihnya 7 jam dan klip gak akan ketemu."""
    target = timestamp.astimezone(ZoneInfo("Asia/Jakarta")).replace(tzinfo=None) \
        if timestamp.tzinfo else timestamp
    src = _find_clip(camera_id, target)
    if src is None:
        raise HTTPException(404, "Klip tidak ditemukan untuk kamera/waktu ini")

    dst = CACHE_DIR / (src.stem + ".mp4")
    if not dst.exists():
        try:
            await asyncio.to_thread(_transcode, src, dst)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, f"Gagal transcode klip: {exc.stderr.decode(errors='ignore')[:300]}")

    # content_disposition_type="inline" — tanpa ini FileResponse(filename=...)
    # default-nya "attachment", browser langsung download alih-alih memutar
    # di <video> tag.
    return FileResponse(dst, media_type="video/mp4", filename=dst.name, content_disposition_type="inline")
