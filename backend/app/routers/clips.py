import asyncio
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix="/clips", tags=["clips"])

# Backend baca klip langsung dari disk AI service (pola sama dgn thumbnails.py)
# supaya tetap bisa dilihat pas AI service mati.
_REPO_ROOT = Path(__file__).parents[3]
CLIPS_DIR = Path(os.getenv("CLIPS_DIR", str(_REPO_ROOT / "ai-service" / "output" / "clips")))
# Transcode-on-request + disk cache di folder AI service, supaya retention.py
# ikut membersihkan cache lama bareng output/ lainnya.
CACHE_DIR = Path(os.getenv("CLIPS_WEB_DIR", str(_REPO_ROOT / "ai-service" / "output" / "clips_web")))


# Toleransi kalau target jatuh di gap antar-klip — jangan balikin klip yang
# jauh (bisa orang lain sama sekali).
FALLBACK_MAX_GAP = float(os.getenv("CLIP_FALLBACK_MAX_GAP", "20"))

# Jendela potong di sekitar momen event, bukan seluruh klip (bisa 90 dtk).
WINDOW_BEFORE = float(os.getenv("CLIP_WINDOW_BEFORE", "6"))
WINDOW_AFTER  = float(os.getenv("CLIP_WINDOW_AFTER", "8"))


_STAMP_ONLY_RE = re.compile(r"^(\d{8}_\d{6}(?:_\d+)?)$")


def _clip_stamp(f: Path, camera_id: str) -> str | None:
    """Cocokkan f cuma kalau stem-nya PERSIS clip_{camera_id}_{stamp}.
    Glob "clip_c10_*" juga menangkap kamera lain "clip_c10_0910_*"; validasi
    ulang di sini supaya bagian setelah prefix memang stempel waktu."""
    prefix = f"clip_{camera_id}_"
    if not f.stem.startswith(prefix):
        return None
    m = _STAMP_ONLY_RE.match(f.stem[len(prefix):])
    return m.group(1) if m else None


def _find_clip(camera_id: str, target: datetime) -> tuple[Path, datetime, float] | None:
    """Cari klip yang jendela [mulai, mulai+durasi]-nya mencakup target,
    fallback ke klip terdekat hari yang sama kalau target jatuh di gap."""
    candidates: list[tuple[datetime, datetime, Path]] = []
    all_files: list[Path] = [
        p for pat in (f"clip_{camera_id}_*.mp4", f"clip_{camera_id}_*.avi")
        for p in CLIPS_DIR.glob(pat)
    ]

    for f in all_files:
        stamp = _clip_stamp(f, camera_id)
        if stamp is None:
            continue
        ts = None
        for fmt in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
            try:
                ts = datetime.strptime(stamp, fmt)
                break
            except ValueError:
                ts = None
        if ts is None:
            continue

        dur_sidecar = f.with_suffix(".dur")
        dur = 10.0
        if dur_sidecar.exists():
            try:
                dur = float(dur_sidecar.read_text().strip())
            except (ValueError, OSError):
                pass
        end = ts + timedelta(seconds=dur)
        candidates.append((ts, end, dur, f))

    if not candidates:
        return None

    for ts, end, dur, f in candidates:
        if ts <= target <= end:
            return f, ts, dur

    # Fallback: klip terdekat pada hari yang sama
    best_file: Path | None = None
    best_ts: datetime | None = None
    best_dur: float = 0.0
    min_diff = float("inf")
    for ts, end, dur, f in candidates:
        if ts.date() != target.date():
            continue
        if target > end:
            diff = (target - end).total_seconds()
        else:
            diff = (ts - target).total_seconds()
        if diff < min_diff:
            min_diff = diff
            best_file = f
            best_ts = ts
            best_dur = dur

    if best_file and best_ts and min_diff <= FALLBACK_MAX_GAP:
        return best_file, best_ts, best_dur

    return None


def _cut_segment(src: Path, start: float, dur: float, dst: Path) -> None:
    """Potong [start, start+dur] dari src → dst (MP4 H.264). Re-encode (bukan
    -c copy) biar mulai tepat di detik yang diminta, bukan di keyframe terdekat."""
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
async def get_clip(camera_id: str, request: Request, timestamp: datetime = Query(...)) -> FileResponse:
    """Cari file klip di output/clips/ yang mencakup timestamp yang diminta,
    lalu potong segmen jendela di sekitar momen orang terekam."""
    if "/" in camera_id or "\\" in camera_id or ".." in camera_id:
        raise HTTPException(400, "camera_id tidak valid")
    target = timestamp.astimezone(ZoneInfo("Asia/Jakarta")).replace(tzinfo=None) \
        if timestamp.tzinfo else timestamp

    wall_start: datetime | None = None
    wall_end:   datetime | None = None
    pool = getattr(request.app.state, "pool", None)
    if pool:
        try:
            row = await pool.fetchrow(
                """SELECT d.id, d.thumbnail_url, d.created_at, t.started_at, t.ended_at 
                   FROM detections d 
                   LEFT JOIN tracklets t ON d.tracklet_id = t.id
                   WHERE d.camera_id = $1 AND d.timestamp = $2
                   LIMIT 1""",
                camera_id, timestamp
            )
            if not row:
                row = await pool.fetchrow(
                    """SELECT d.id, d.thumbnail_url, d.created_at, t.started_at, t.ended_at 
                       FROM detections d 
                       LEFT JOIN tracklets t ON d.tracklet_id = t.id
                       WHERE d.camera_id = $1 
                         AND abs(extract(epoch from (d.timestamp - $2))) < 30
                       ORDER BY abs(extract(epoch from (d.timestamp - $2)))
                       LIMIT 1""",
                    camera_id, timestamp
                )
            if row:
                # Prioritas row["ended_at"] (waktu frame terakhir tracklet
                # terlihat) atas created_at/nama thumbnail, yang distempel
                # saat tracklet finalisasi dan bisa telat dari momen aslinya.
                if row["ended_at"]:
                    wall_end = row["ended_at"].astimezone(ZoneInfo("Asia/Jakarta")).replace(tzinfo=None)
                else:
                    thumb = row["thumbnail_url"] or ""
                    m = re.search(r"(\d{8}_\d{6}(?:_\d+)?)", thumb)
                    if m:
                        for fmt in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
                            try:
                                wall_end = datetime.strptime(m.group(1), fmt)
                                break
                            except ValueError:
                                pass
                    elif row["created_at"]:
                        wall_end = row["created_at"].astimezone(ZoneInfo("Asia/Jakarta")).replace(tzinfo=None)

                if wall_end:
                    if row["started_at"] and row["ended_at"]:
                        # Clamp ke WINDOW_BEFORE, bukan durasi tracklet penuh:
                        # tracklet bisa nembus lebih dari satu file rekaman.
                        tracklet_dur = min(
                            max(1.0, (row["ended_at"] - row["started_at"]).total_seconds()),
                            WINDOW_BEFORE,
                        )
                    else:
                        tracklet_dur = 4.0
                    wall_start = wall_end - timedelta(seconds=tracklet_dur)
        except Exception as exc:
            print(f"[clips] DB lookup error: {exc}")

    found = None
    if wall_start:
        found = _find_clip(camera_id, wall_start)
    if found is None and wall_end:
        found = _find_clip(camera_id, wall_end)
    if found is None:
        found = _find_clip(camera_id, target)

    if found is None:
        raise HTTPException(404, "Klip tidak ditemukan untuk kamera/waktu ini")
    src, clip_start, clip_dur = found

    if wall_start and wall_end:
        p_start   = max(0.0, (wall_start - clip_start).total_seconds())
        p_end     = max(p_start + 1.0, (wall_end - clip_start).total_seconds())
        seg_start = max(0.0, p_start - 0.5)
        seg_dur   = max(3.0, (p_end - seg_start) + 1.0)
    else:
        offset    = max(0.0, (target - clip_start).total_seconds())
        seg_start = max(0.0, offset - 0.5)
        seg_dur   = max(3.0, 5.0)

    # Klip fallback: seg_start bisa lewat clip_dur asli. Clamp biar ffmpeg
    # gak keluarin potongan nyaris kosong dari ujung file.
    if seg_start >= clip_dur:
        seg_start = max(0.0, clip_dur - 1.0)
    seg_dur = max(1.0, min(seg_dur, clip_dur - seg_start))

    dst = CACHE_DIR / f"{src.stem}__{int(seg_start*10)}-{int((seg_start + seg_dur)*10)}.mp4"
    if not dst.exists():
        try:
            await asyncio.to_thread(_cut_segment, src, seg_start, seg_dur, dst)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, f"Gagal potong klip: {exc.stderr.decode(errors='ignore')[:300]}")

    return FileResponse(dst, media_type="video/mp4", filename=dst.name, content_disposition_type="inline")

