import asyncio
import subprocess
from datetime import datetime, timedelta
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
    """Cari klip yang jendela [mulai, mulai+durasi]-nya BENERAN mencakup
    target. Sebelumnya cuma cek "klip terakhir yang mulai sebelum target" —
    itu salah pilih klip yang sudah berakhir kalau target jatuh di jeda
    IDLE setelah klip itu tutup tapi sebelum klip berikutnya (thumbnail
    jadi buka klip yang gak sesuai). Durasi nyata dibaca dari sidecar
    `.dur` yang ditulis clip_recorder.py._finalize(); kalau sidecar-nya gak
    ada (klip lama / gagal ditulis), fallback ke klip terakhir yang mulai
    sebelum target seperti dulu."""
    prefix = f"clip_{camera_id}_"
    containing: tuple[datetime, Path] | None = None
    fallback:   tuple[datetime, Path] | None = None
    for f in CLIPS_DIR.glob(f"{prefix}*.avi"):
        stamp = f.name[len(prefix):-4]
        try:
            ts = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if ts > target:
            continue
        if fallback is None or ts > fallback[0]:
            fallback = (ts, f)

        dur_sidecar = f.with_suffix(".dur")
        if not dur_sidecar.exists():
            continue
        try:
            duration = float(dur_sidecar.read_text().strip())
        except (ValueError, OSError):
            continue
        end = ts + timedelta(seconds=duration)
        if ts <= target <= end and (containing is None or ts > containing[0]):
            containing = (ts, f)

    if containing is not None:
        return containing[1]
    return fallback[1] if fallback else None


def _transcode(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y"]
    # Sidecar dari clip_recorder.py._finalize() — fps asli hasil hitung
    # frame_ditulis/durasi_nyata (AVI-nya sendiri sengaja gak disentuh, lihat
    # catatan di sana). "-r" SEBELUM "-i" nyuruh ffmpeg baca stream MJPEG ini
    # di laju itu, bukan percaya declared-fps di header AVI yang cuma tebakan
    # awal (di-set sebelum tau berapa lama klip beneran akan berjalan).
    fps_sidecar = src.with_suffix(".fps")
    if fps_sidecar.exists():
        try:
            cmd += ["-r", fps_sidecar.read_text().strip()]
        except Exception:
            pass
    cmd += ["-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-fps_mode", "cfr", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


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
