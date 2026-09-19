import asyncio
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, Request
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


import re

def _find_clip(camera_id: str, target: datetime) -> tuple[Path, datetime] | None:
    """Cari klip yang jendela [mulai, mulai+durasi]-nya mencakup target.
    Jika target jatuh di gap rekaman, fallback ke klip terdekat pada hari yang sama.
    Mendukung format nama clip_{camera_id}_*.mp4/avi dan clip_{camera_id}_0910_*."""
    candidates: list[tuple[datetime, datetime, Path]] = []
    patterns = [f"clip_{camera_id}_*.mp4", f"clip_{camera_id}_*.avi",
                f"clip_{camera_id}_0910_*.mp4", f"clip_{camera_id}_0910_*.avi"]
    seen_paths = set()
    all_files: list[Path] = []
    for pat in patterns:
        for p in CLIPS_DIR.glob(pat):
            if p not in seen_paths:
                seen_paths.add(p)
                all_files.append(p)

    for f in all_files:
        m = re.search(r"(\d{8}_\d{6}(?:_\d+)?)", f.stem)
        if not m:
            continue
        stamp = m.group(1)
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
        candidates.append((ts, end, f))

    if not candidates:
        return None

    # 1. Prioritas: klip yang mencakup target secara langsung
    for ts, end, f in candidates:
        if ts <= target <= end:
            return f, ts

    # 2. Fallback: klip terdekat pada hari yang sama
    best_file: Path | None = None
    best_ts: datetime | None = None
    min_diff = float("inf")
    for ts, end, f in candidates:
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

    # Batas toleransi fallback (default 20 detik)
    max_gap = FALLBACK_MAX_GAP
    if best_file and best_ts and min_diff <= max_gap:
        return best_file, best_ts

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


import json

_SAMPLE_CLIPS_JSON = Path(__file__).parents[1] / "data" / "sample_0910_clips.json"
_SAMPLE_CLIPS_MAP: dict[str, dict] = {}
if _SAMPLE_CLIPS_JSON.exists():
    try:
        _SAMPLE_CLIPS_MAP = json.loads(_SAMPLE_CLIPS_JSON.read_text())
    except Exception as _e:
        print(f"[clips] Warning: gagal load {_SAMPLE_CLIPS_JSON}: {_e}")


@router.get("/{camera_id}")
async def get_clip(camera_id: str, request: Request, timestamp: datetime = Query(...)) -> FileResponse:
    """Cari klip rekaman kamera ini yang mencakup timestamp yang diminta.
    Untuk data kejadian yang terpetakan secara presisi (seperti sample CCTV 2026-09-10),
    langsung putar potongan presisi dari source clip tanpa jeda dan tanpa salah orang.
    Untuk rekaman live RTSP, cari file klip di output/clips/ lalu potong segmen jendela
    saat orang terekam."""
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
                # 1. Jalur Utama: Jika deteksi ini memiliki mapping klip presisi (100% akurat)
                det_id_str = str(row["id"])
                if det_id_str in _SAMPLE_CLIPS_MAP:
                    meta = _SAMPLE_CLIPS_MAP[det_id_str]
                    clip_file = CACHE_DIR / meta["clip_filename"]
                    if not clip_file.exists():
                        src = _REPO_ROOT / meta["source_clip"]
                        if src.exists():
                            await asyncio.to_thread(
                                _cut_segment, src, meta["start_sec"], meta["dur_sec"], clip_file
                            )
                    if clip_file.exists():
                        return FileResponse(
                            clip_file,
                            media_type="video/mp4",
                            filename=clip_file.name,
                            content_disposition_type="inline"
                        )

                # 2. Ekstrak wall-clock asli rekaman jika rekaman live. Prioritas
                # row["ended_at"] (tl.last_seen, waktu frame terakhir tracklet
                # BENERAN terlihat) — bukan nama file thumbnail/created_at, yang
                # keduanya distempel saat tracklet FINALISASI (bisa telat sampai
                # TRACKLET_GAP_CYCLES=15 siklus batch, ~2 detik di FPS efektif
                # 7.5, dari momen orang itu sebenarnya masih di frame), jadi
                # jendela klip yang dipotong bisa geser ke momen setelahnya.
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
                        tracklet_dur = max(1.0, (row["ended_at"] - row["started_at"]).total_seconds())
                    else:
                        tracklet_dur = 4.0
                    wall_start = wall_end - timedelta(seconds=tracklet_dur)
        except Exception as exc:
            print(f"[clips] DB lookup error: {exc}")

    # Prioritas rekaman live: cari klip memakai waktu rekam asli (wall-clock)
    found = None
    if wall_start:
        found = _find_clip(camera_id, wall_start)
    if found is None and wall_end:
        found = _find_clip(camera_id, wall_end)
    if found is None:
        found = _find_clip(camera_id, target)

    if found is None:
        raise HTTPException(404, "Klip tidak ditemukan untuk kamera/waktu ini")
    src, clip_start = found

    # Potong persis saat orang tersebut terlihat (dengan padding 0.5s di awal, 1s di akhir)
    if wall_start and wall_end:
        p_start   = max(0.0, (wall_start - clip_start).total_seconds())
        p_end     = max(p_start + 1.0, (wall_end - clip_start).total_seconds())
        seg_start = max(0.0, p_start - 0.5)
        seg_dur   = max(3.0, (p_end - seg_start) + 1.0)
    else:
        offset    = max(0.0, (target - clip_start).total_seconds())
        seg_start = max(0.0, offset - 0.5)
        seg_dur   = max(3.0, 5.0)

    dst = CACHE_DIR / f"{src.stem}__{int(seg_start*10)}-{int((seg_start + seg_dur)*10)}.mp4"
    if not dst.exists():
        try:
            await asyncio.to_thread(_cut_segment, src, seg_start, seg_dur, dst)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(500, f"Gagal potong klip: {exc.stderr.decode(errors='ignore')[:300]}")

    return FileResponse(dst, media_type="video/mp4", filename=dst.name, content_disposition_type="inline")

