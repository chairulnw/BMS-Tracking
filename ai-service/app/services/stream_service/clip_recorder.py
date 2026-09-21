"""Rekam klip video per kamera saat ada orang terdeteksi, plus logger CSV
prediksi untuk mode file-playback (dibandingkan dengan ground truth).

Encoder: recorder emit `_latest` pada kadens TETAP OUT_FPS terkunci ke `now`
(wall clock yang di-pass, bukan wall clock ffmpeg — tahan walau recorder thread
ketinggalan), lalu di-pipe ke subprocess `ffmpeg` CFR yang encode langsung ke MP4.
Hasilnya:
  - playback PERSIS real-time — frames_written/OUT_FPS == detik `now` nyata;
  - burst cepat (source <= OUT_FPS) tetap kesimpan → gerak halus;
  - stall → `_latest` ditulis berulang selama gap (BEKU), bukan diregangkan;
  - output langsung MP4 → browser muter tanpa transcode terpisah, dan cv2
    VideoWriter mp4v yang nggak reliable di macOS nggak kepakai.
Sidecar `.dur` (durasi detik) tetap ditulis buat clips.py._find_clip().
"""

import csv
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CLIP_COOLDOWN  = 5.0
# klip < ini dibuang (bukan gerakan selesai, tapi giliran kamera habis atau klip
# sumber terlalu pendek) — nggak cukup buat _find_clip, cuma nyampah di clips/
MIN_CLIP_SEC   = float(os.getenv("CLIP_MIN_DURATION_SEC", "1.0"))
MAX_FRAME_GAP  = 3.0   # detik — gap nyata antar frame > ini → tutup klip
CLIP_MAX_DURATION = float(os.getenv("CLIP_MAX_DURATION", "90"))  # detik — segment klip lewat ini (standar CCTV)
CLIPS_DIR      = Path("output/clips")
# 1 → klip direkam dengan overlay bounding box + label; 0 → frame mentah.
CLIP_BBOX_OVERLAY = os.getenv("CLIP_BBOX_OVERLAY", "0").lower() not in ("0", "false", "no", "")
CLIP_ENCODE_PRESET = os.getenv("CLIP_ENCODE_PRESET", "ultrafast")  # x264 preset
CLIP_CRF           = os.getenv("CLIP_CRF", "26")                    # x264 quality (besar = kecil file)
OUT_FPS  = float(os.getenv("CLIP_OUT_FPS", "20"))   # kadens output klip (CFR)
_EMIT_DT = 1.0 / OUT_FPS

_CLIP_STOP = object()  # sentinel: finalize clip saat ganti file sumber


class _PredictionLogger:
    """Tulis hasil deteksi+tracking+ReID ke CSV, satu baris per box per frame,
    khusus untuk kamera yang jalan lewat _capture_loop_files (playlist file lokal).
    Dipakai untuk dibandingkan dengan ground truth hasil anotasi manual.

    Fase 2: identitas baru diketahui saat TRACKLET ditutup, bukan per-frame
    (lihat plan/07-fase2-detail.md §6). Jadi baris tiap frame di-buffer per
    (cam_id, track_id) lewat buffer(), baru benar-benar ditulis ke file lewat
    flush() sekali tracklet-nya resolve — dengan nama akhir yang sudah pasti,
    bukan placeholder f"t{track_id}"."""

    _HEADER = ["source_clip", "local_frame", "camera", "person_pred", "x", "y", "w", "h"]

    def __init__(self, path: Path) -> None:
        self._path    = path
        self._file    = None
        self._writer  = None
        self._pending: dict[tuple, list[tuple]] = {}

    def buffer(self, key: tuple, source_clip: str, local_frame: int, camera: str,
               x: int, y: int, w: int, h: int) -> None:
        self._pending.setdefault(key, []).append((source_clip, local_frame, camera, x, y, w, h))

    def flush(self, key: tuple, person_pred: str) -> None:
        rows = self._pending.pop(key, None)
        if not rows:
            return
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file   = open(self._path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self._HEADER)
        for source_clip, local_frame, camera, x, y, w, h in rows:
            self._writer.writerow([source_clip, local_frame, camera, person_pred, x, y, w, h])
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file   = None
            self._writer = None


class ClipRecorder:
    """
    State machine:
      IDLE      → orang muncul → RECORDING
      RECORDING → orang hilang → COOLING
      COOLING   → orang muncul lagi → RECORDING
      COOLING   → idle >= CLIP_COOLDOWN → IDLE (finalize)
    """

    _IDLE      = "IDLE"
    _RECORDING = "RECORDING"
    _COOLING   = "COOLING"

    def __init__(self, camera_id: str, fps: float, width: int, height: int) -> None:
        self._camera_id      = camera_id
        self._width          = width
        self._height         = height
        self._state          = self._IDLE
        self._proc:           "subprocess.Popen | None" = None
        self._clip_path:      "Path | None"             = None
        self._clip_start:     float                     = 0.0
        self._idle_since:     float                     = 0.0
        self._last_valid:     "np.ndarray | None"       = None
        self._latest:         "np.ndarray | None"       = None
        self._frames_written: int                       = 0
        self._next_emit:      float                     = 0.0
        self._last_update_at: float                     = 0.0
        self._lock            = threading.Lock()

    @staticmethod
    def _is_glitch(frame: np.ndarray) -> bool:
        return float(frame.mean()) < 15.0

    @staticmethod
    def _draw(frame: np.ndarray, annotations: list) -> np.ndarray:
        out = frame.copy()
        for x1, y1, x2, y2, label in annotations:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return out

    def update(self, frame: np.ndarray, has_person: bool, now: float,
               annotations: "list | None" = None) -> None:
        with self._lock:
            if (self._state != self._IDLE and self._last_update_at > 0
                    and now - self._last_update_at > MAX_FRAME_GAP):
                self._finalize()
            self._last_update_at = now

            drawn = self._draw(frame, annotations) if (CLIP_BBOX_OVERLAY and annotations) else frame
            if not self._is_glitch(frame):
                self._latest = drawn
                self._last_valid = drawn
            elif self._latest is None:
                self._latest = self._last_valid if self._last_valid is not None else drawn

            if self._state == self._IDLE:
                if has_person:
                    self._start(now)
            else:  # RECORDING / COOLING
                self._emit_until(now)
                if now - self._clip_start >= CLIP_MAX_DURATION:
                    # segment: tutup klip ini, langsung buka yang baru (mulus)
                    self._finalize()
                    if has_person:
                        self._start(now)
                elif self._state == self._RECORDING and not has_person:
                    self._idle_since = now
                    self._state = self._COOLING
                elif self._state == self._COOLING:
                    if has_person:
                        self._state = self._RECORDING
                    elif now - self._idle_since >= CLIP_COOLDOWN:
                        self._finalize()

    def _emit_until(self, now: float) -> None:
        """Tulis `_latest` ke ffmpeg pada kadens OUT_FPS terkunci ke `now`.
        Burst source > OUT_FPS → sebagian di-skip; stall → frame yang sama
        ditulis berkali-kali (BEKU). Gap > MAX_FRAME_GAP udah ditutup di update()."""
        proc = self._proc
        if proc is None or proc.stdin is None or self._latest is None:
            return
        buf = np.ascontiguousarray(self._latest).tobytes()
        try:
            while self._next_emit <= now:
                proc.stdin.write(buf)
                self._frames_written += 1
                self._next_emit += _EMIT_DT
        except (BrokenPipeError, ValueError, OSError):
            self._proc = None   # ffmpeg mati — jangan blokir

    def force_stop(self) -> None:
        with self._lock:
            if self._proc is not None:
                self._finalize()

    def _start(self, now: float) -> None:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        # milidetik ikut biar segment (CLIP_MAX_DURATION) / restart cepat di detik
        # yang sama nggak nabrak nama. clips.py._find_clip parse dua format.
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = CLIPS_DIR / f"clip_{self._camera_id}_{ts}.mp4"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}", "-r", f"{OUT_FPS:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", CLIP_ENCODE_PRESET, "-crf", CLIP_CRF,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except (OSError, FileNotFoundError) as exc:
            print(f"[clip:{self._camera_id}] ERROR: ffmpeg gagal start ({exc}) — klip tidak direkam")
            self._proc = None
            return
        self._clip_path      = path
        self._clip_start     = now
        self._next_emit      = now
        self._frames_written = 0
        self._state          = self._RECORDING
        print(f"[clip:{self._camera_id}] START → {path.name}  fps={OUT_FPS:g}")
        self._emit_until(now)

    def _finalize(self) -> None:
        proc   = self._proc
        path   = self._clip_path
        frames = self._frames_written
        dur    = max(frames / OUT_FPS, 0.04)   # CFR → durasi eksak dari jumlah frame
        self._proc           = None
        self._clip_path      = None
        self._idle_since     = 0.0
        self._frames_written = 0
        self._state          = self._IDLE

        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        if path is None:
            return
        if dur < MIN_CLIP_SEC:
            print(f"[clip:{self._camera_id}] BUANG {path.name}  {dur:.2f}s < {MIN_CLIP_SEC}s "
                  f"(giliran kamera abis / siklus deteksi kelewat lambat, bukan orang beneran hilang)")
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[clip:{self._camera_id}] gagal hapus klip terlalu pendek ({path.name}): {exc}")
            return
        size = path.stat().st_size if path.exists() else 0
        print(f"[clip:{self._camera_id}] SAVED {path.name}  {dur:.1f}s  frames={frames}  size={size//1024}KB")
        try:
            path.with_suffix(".dur").write_text(f"{dur:.3f}")
        except OSError as exc:
            print(f"[clip:{self._camera_id}] gagal tulis sidecar durasi ({path.name}): {exc}")


def _demo() -> None:
    """ponytail self-check: rekam laju IRREGULAR (burst + stall) lewat pipe
    ffmpeg, force_stop() pertengahan RECORDING, verifikasi MP4 valid & durasinya
    ngikut wall clock (bukan jumlah frame)."""
    import shutil
    import tempfile

    global CLIPS_DIR
    tmp = tempfile.mkdtemp()
    orig_dir = CLIPS_DIR
    CLIPS_DIR = Path(tmp)
    try:
        rec = ClipRecorder("test", fps=10.0, width=64, height=48)
        f = np.full((48, 64, 3), 200, dtype=np.uint8)
        t0 = 1000.0  # `now` sintetis — recorder pakai ini, bukan wall clock nyata
        # burst 30 frame di 1.0s (source 30fps), stall 1.5s, burst 30 frame lagi
        seq  = [t0 + i / 30 for i in range(30)]
        seq += [t0 + 2.5 + i / 30 for i in range(30)]
        for tt in seq:
            rec.update(f, has_person=True, now=tt)
        rec._emit_until(seq[-1])
        rec.force_stop()

        clips = list(Path(tmp).glob("*.mp4"))
        assert len(clips) == 1, f"expected 1 clip, got {len(clips)}"
        cap = cv2.VideoCapture(str(clips[0]))
        assert cap.isOpened(), "clip harusnya bisa dibuka, bukan corrupt"
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        assert rec._state == ClipRecorder._IDLE
        wall = seq[-1] - t0  # ~3.47s
        # durasi klip harus ngikut `now` span (real-time), bukan jumlah frame source
        assert abs(n / OUT_FPS - wall) < 0.3, f"durasi {n/OUT_FPS:.2f}s (frames={n}) jauh dari wall {wall:.2f}s"

        # Klip < MIN_CLIP_SEC (mis. giliran kamera abis / gap sesaat) dibuang,
        # bukan disimpan sebagai file sepersekian detik.
        rec2 = ClipRecorder("test2", fps=10.0, width=64, height=48)
        t1 = 2000.0
        for i in range(5):   # ~0.17s < MIN_CLIP_SEC — harus dibuang
            rec2.update(f, has_person=True, now=t1 + i / 30)
        rec2._emit_until(t1 + 4 / 30)
        rec2.force_stop()
        assert list(Path(tmp).glob("clip_test2_*.mp4")) == [], "klip < MIN_CLIP_SEC seharusnya dibuang, bukan disimpan"
    finally:
        CLIPS_DIR = orig_dir
        shutil.rmtree(tmp, ignore_errors=True)
    print("clip_recorder self-check OK")


if __name__ == "__main__":
    _demo()
