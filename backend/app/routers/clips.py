import asyncio
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

router = APIRouter(prefix="/clips", tags=["clips"])

# Klip direkam AI service ke ai-service/output/clips/. Backend baca langsung
# dari disk (pola sama dgn thumbnails.py) supaya klip tetap bisa dilihat pas
# AI service mati. Resolve relatif ke file ini: backend/app/routers/ → repo root.
_REPO_ROOT = Path(__file__).parents[3]
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", str(_REPO_ROOT / "ai-service" / "output" / "clips")))
# ponytail: transcode-on-request + disk cache, cukup buat admin preview sesekali.
# Kalau nanti dipakai rame-rame, ganti ke pipeline transcode background pas clip
# selesai direkam. Cache di folder AI service supaya retention.py-nya (yang
# nge-cap ukuran output/) ikut mrunning-in file lama.
CACHE_DIR = Path(os.getenv("CLIPS_WEB_DIR", str(_REPO_ROOT / "ai-service" / "output" / "clips_web")))


# Rekaman kini termotion-gate + tersegmentasi (clip_recorder.py) → banyak klip
# pendek dgn gap. Kalau target jatuh di gap, JANGAN balikin klip jauh sebelumnya
# (bisa orang lain sama sekali). Toleransi kecil aja buat jitter timestamp event.
FALLBACK_MAX_GAP = float(os.getenv("CLIP_FALLBACK_MAX_GAP", "20"))

# Klip file = rekaman scene utuh (bisa 90 dtk, banyak orang). Yang mau ditonton
# cuma sekitar momen event → potong jendela ini (detik sebelum/sesudah target).
WINDOW_BEFORE = float(os.getenv("CLIP_WINDOW_BEFORE", "30"))
WINDOW_AFTER  = float(os.getenv("CLIP_WINDOW_AFTER", "8"))


def _find_clip(camera_id: str, target: datetime) -> tuple[Path, datetime] | None:
    """Cari klip yang jendela [mulai, mulai+durasi]-nya BENERAN mencakup target.
    Durasi dibaca dari sidecar `.dur` (clip_recorder.py._finalize()); kalau
    nggak ada / target di gap, fallback ke klip terakhir sebelum target TAPI
    cuma kalau selisihnya <= FALLBACK_MAX_GAP detik (di luar itu: dianggap
    nggak ada rekaman, biar frontend bilang jujur bukan nampilin orang salah).
    `.mp4` = klip baru (ffmpeg langsung), `.avi` = klip lama (MJPEG, perlu transcode)."""
    prefix = f"clip_{camera_id}_"
    containing: tuple[datetime, Path] | None = None
    fallback:   tuple[datetime, Path, datetime] | None = None   # (start, path, end)
    for f in list(CLIPS_DIR.glob(f"{prefix}*.mp4")) + list(CLIPS_DIR.glob(f"{prefix}*.avi")):
        stamp = f.stem[len(prefix):]
        for fmt in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):   # baru (ada milidetik) / lama
            try:
                ts = datetime.strptime(stamp, fmt)
                break
            except ValueError:
                ts = None
        if ts is None:
            continue
        if ts > target:
            continue

        end = ts
        dur_sidecar = f.with_suffix(".dur")
        if dur_sidecar.exists():
            try:
                end = ts + timedelta(seconds=float(dur_sidecar.read_text().strip()))
            except (ValueError, OSError):
                pass

        if fallback is None or ts > fallback[0]:
            fallback = (ts, f, end)
        if ts <= target <= end and (containing is None or ts > containing[0]):
            containing = (ts, f)

    if containing is not None:
        return containing[1], containing[0]
    if fallback and (target - fallback[2]).total_seconds() <= FALLBACK_MAX_GAP:
        return fallback[1], fallback[0]
    return None


def _cut_segment(src: Path, start: float, dur: float, dst: Path) -> None:
    """Potong [start, start+dur] dari src → dst (MP4 H.264). Re-encode (bukan
    -c copy) biar mulai tepat di detik yang diminta, bukan lompat ke keyframe
    terdekat (~12 dtk meleset). Segmen pendek, ultrafast → cepat, hasil di-cache.
    `.fps` sidecar (klip .avi lama) dipakai buat laju baca yang bener."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    pre = []
    fps_sidecar = src.with_suffix(".fps")
    if src.suffix == ".avi" and fps_sidecar.exists():
        try:
            pre += ["-r", fps_sidecar.read_text().strip()]
        except OSError:
            pass
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", *pre, "-i", str(src), "-t", f"{dur:.3f}",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


@router.get("/{camera_id}")
async def get_clip(camera_id: str, timestamp: datetime = Query(...)) -> FileResponse:
    """Cari klip rekaman kamera ini yang mencakup timestamp yang diminta, lalu
    potong jendela [target-WINDOW_BEFORE, target+WINDOW_AFTER] dari file klip
    (yang isinya rekaman scene utuh) → MP4 pendek, di-cache.

    Nama file klip distempel pakai jam lokal (clip_recorder.py: datetime.now()
    naive, WIB) — timestamp dari frontend selalu UTC-aware (kolom TIMESTAMPTZ).
    Konversi ke WIB dulu sebelum dibanding, kalau cuma di-strip tzinfo selisih
    7 jam dan klip gak ketemu."""
    if "/" in camera_id or "\\" in camera_id or ".." in camera_id:
        raise HTTPException(400, "camera_id tidak valid")
    target = timestamp.astimezone(ZoneInfo("Asia/Jakarta")).replace(tzinfo=None) \
        if timestamp.tzinfo else timestamp
    found = _find_clip(camera_id, target)
    if found is None:
        raise HTTPException(404, "Klip tidak ditemukan untuk kamera/waktu ini")
    src, clip_start = found

    offset = max(0.0, (target - clip_start).total_seconds())
    seg_start = max(0.0, offset - WINDOW_BEFORE)
    seg_dur   = (offset - seg_start) + WINDOW_AFTER
    dst = CACHE_DIR / f"{src.stem}__{int(seg_start)}-{int(seg_start + seg_dur)}.mp4"
    if not dst.exists():
        try:
            await asyncio.to_thread(_cut_segment, src, seg_start, seg_dur, dst)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, f"Gagal potong klip: {exc.stderr.decode(errors='ignore')[:300]}")

    # content_disposition_type="inline" — tanpa ini FileResponse(filename=...)
    # default-nya "attachment", browser langsung download alih-alih memutar.
    return FileResponse(dst, media_type="video/mp4", filename=dst.name, content_disposition_type="inline")
