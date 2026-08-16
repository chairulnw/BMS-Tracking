"""
Stream service — pipeline RTSP multi-kamera dengan batch inference.

Arsitektur:
  _CamSlot      : data + capture thread per kamera (hanya cap.read(), ringan)
  BatchProcessor: satu thread inferensi, kumpulkan 1 frame per kamera →
                  model.predict(batch) → 1 GPU call untuk semua kamera sekaligus.
                  Tiap kamera punya BYTETracker sendiri di slot.tracker karena
                  model.track(batch) hanya buat 1 tracker untuk seluruh batch.
  StreamManager : public API
"""

import csv
import os
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;1048576",
)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

import cv2
import numpy as np
import requests
import torch
import torchreid
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from app.auth import create_service_token
from app.schemas import IdentityRecord, LineConfig, StreamStatusResponse
from app.services.pipeline_service import _debug_reid, IdentityDB, _extract_embedding

def _auth_headers() -> dict:
    """Header dipakai ai-service untuk memanggil endpoint backend yang dilindungi login."""
    return {"Authorization": f"Bearer {create_service_token()}"}

# ── Constants ─────────────────────────────────────────────────────────────────

RECONNECT_TRIES      = 5
RECONNECT_DELAY      = 2.0
CLIP_COOLDOWN        = 5.0
CROSSING_COOLDOWN    = 3.0   # detik minimum antar event crossing per (line, track)
CLIPS_DIR            = Path("output/clips")
THUMBNAILS_DIR       = Path("thumbnails")
PREDICTIONS_CSV      = Path("predictions.csv")  # log prediksi mode file-playback, utk dibanding ground truth


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _cross_side(px: int, py: int, lx1: int, ly1: int, lx2: int, ly2: int) -> float:
    """Signed distance dari titik (px,py) ke garis (lx1,ly1)-(lx2,ly2).
    Positif di satu sisi, negatif di sisi lain."""
    return float((lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1))


def _fetch_crossing_lines(camera_id: str) -> list[dict]:
    url = os.getenv("BACKEND_URL", "http://localhost:8002")
    try:
        r = requests.get(f"{url}/cameras/{camera_id}/lines", headers=_auth_headers(), timeout=3.0)
        if r.ok:
            return r.json()
    except Exception as exc:
        print(f"[{camera_id}] gagal load crossing lines: {exc}")
    return []


def _fetch_has_zone(camera_id: str) -> bool:
    """Cek apakah kamera punya entry di camera_zones (berarti crossing = masuk ruangan)."""
    url = os.getenv("BACKEND_URL", "http://localhost:8002")
    try:
        r = requests.get(f"{url}/cameras/{camera_id}/zone", headers=_auth_headers(), timeout=3.0)
        if r.ok:
            return bool(r.json())  # {} = tidak ada zona, {id:..., room_name:...} = ada
    except Exception as exc:
        print(f"[{camera_id}] gagal cek zone: {exc}")
    return False


# ── Backend client ────────────────────────────────────────────────────────────

class _BackendClient:
    @staticmethod
    def post_occupancy_event(
        camera_id:    str,
        line_id:      int,
        direction:    str,
        event_kind:   str = "crossing",
        snapshot_url: str | None = None,
        person_label: str | None = None,
        track_id:     int | None = None,
    ) -> None:
        url = os.getenv("BACKEND_URL", "http://localhost:8002")
        try:
            requests.post(
                f"{url}/occupancy-events",
                json={
                    "camera_id":    camera_id,
                    "line_id":      line_id,
                    "direction":    direction,
                    "event_kind":   event_kind,
                    "snapshot_url": snapshot_url,
                    "person_label": person_label,
                    "track_id":     track_id,
                },
                headers=_auth_headers(),
                timeout=0.8,
            )
        except Exception as exc:
            print(f"[backend] POST /occupancy-events error ({camera_id}): {exc}")

    @staticmethod
    def post_camera_event(
        camera_id:    str,
        event_type:   str,
        category:     str = "info",
        description:  str | None = None,
        snapshot_url: str | None = None,
        person_label: str | None = None,
    ) -> None:
        url = os.getenv("BACKEND_URL", "http://localhost:8002")
        try:
            requests.post(
                f"{url}/camera-events",
                json={
                    "camera_id":    camera_id,
                    "event_type":   event_type,
                    "category":     category,
                    "description":  description,
                    "snapshot_url": snapshot_url,
                    "person_label": person_label,
                },
                headers=_auth_headers(),
                timeout=0.8,
            )
        except Exception as exc:
            print(f"[backend] POST /camera-events error ({camera_id}): {exc}")

    @staticmethod
    def post_detection(
        person_name:   str,
        camera_id:     str,
        confidence:    float,
        method:        str = "appearance",
        thumbnail_url: str | None = None,
        unique_label:  str | None = None,
        track_id:      int | None = None,
    ) -> None:
        """unique_label adalah label date-aware (e.g. 'Unknown #1@20260622')
        agar PostgreSQL tidak menggabungkan orang dari hari berbeda."""
        url = os.getenv("BACKEND_URL", "http://localhost:8002")
        try:
            requests.post(
                f"{url}/detections",
                json={
                    "person_name":   person_name,
                    "person_label":  unique_label or person_name,
                    "camera_id":     camera_id,
                    "timestamp":     datetime.now(timezone.utc).isoformat(),
                    "confidence":    float(confidence),
                    "method":        method,
                    "thumbnail_url": thumbnail_url,
                    "track_id":      track_id,
                },
                headers=_auth_headers(),
                timeout=0.8,
            )
        except Exception as exc:
            print(f"[backend] POST /detections error ({camera_id}): {exc}")

    @staticmethod
    def update_thumbnail(
        person_label: str,
        camera_id:    str,
        thumb_url:    str,
    ) -> None:
        url = os.getenv("BACKEND_URL", "http://localhost:8002")
        try:
            requests.patch(
                f"{url}/detections/thumbnail",
                json={
                    "person_label":  person_label,
                    "camera_id":     camera_id,
                    "thumbnail_url": thumb_url,
                },
                headers=_auth_headers(),
                timeout=0.8,
            )
        except Exception as exc:
            print(f"[backend] PATCH /detections/thumbnail error ({camera_id}): {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_cameras() -> list[dict]:
    """Ambil kamera aktif dari database via backend. Return [] jika gagal."""
    url = os.getenv("BACKEND_URL", "http://localhost:8002")
    try:
        r = requests.get(f"{url}/cameras", params={"is_active": "true"}, headers=_auth_headers(), timeout=5.0)
        if r.ok:
            return r.json()
    except Exception as exc:
        print(f"[stream] fetch cameras from DB failed: {exc}")
    return []


def _camera_id_from_url(url: str) -> str:
    try:
        parts = [p for p in urlparse(url).path.split("/") if p]
        if "unicast" in parts:
            idx = parts.index("unicast")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return parts[-2] if len(parts) >= 2 else "cam"
    except Exception:
        return "cam"


def _open_capture(url: str) -> "cv2.VideoCapture | None":
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # was 1 — caused H.264 P-frame drops
    if cap.isOpened():
        return cap
    cap.release()
    return None


# ── Prediction logger (mode file-playback) ──────────────────────────────────────

class _PredictionLogger:
    """Tulis hasil deteksi+tracking+ReID ke CSV, satu baris per box per frame,
    khusus untuk kamera yang jalan lewat _capture_loop_files (playlist file lokal).
    Dipakai untuk dibandingkan dengan ground truth hasil anotasi manual.
    File dibuka lazy (baru dibuat saat baris pertama ditulis) dan di-flush tiap
    baris supaya tidak menunggu semua clip selesai baru bisa dibaca."""

    _HEADER = ["source_clip", "local_frame", "camera", "person_pred", "x", "y", "w", "h"]

    def __init__(self, path: Path) -> None:
        self._path   = path
        self._file    = None
        self._writer  = None

    def log(self, source_clip: str, local_frame: int, camera: str,
             person_pred: str, x: int, y: int, w: int, h: int) -> None:
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file   = open(self._path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self._HEADER)
        self._writer.writerow([source_clip, local_frame, camera, person_pred, x, y, w, h])
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file   = None
            self._writer = None


# ── Clip recorder ─────────────────────────────────────────────────────────────

_CLIP_STOP = object()  # sentinel: finalize clip saat ganti file sumber

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
        self._fps            = max(fps, 1.0)
        self._width          = width
        self._height         = height
        self._state          = self._IDLE
        self._writer:         cv2.VideoWriter | None = None
        self._clip_path:      Path | None            = None
        self._clip_start:     float                  = 0.0
        self._idle_since:     float                  = 0.0
        self._last_valid:     np.ndarray | None      = None
        self._frames_written: int                    = 0
        self._frame_times:    deque[float]           = deque(maxlen=60)
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

    def _measured_fps(self) -> float:
        if len(self._frame_times) >= 5:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0.1:
                measured = (len(self._frame_times) - 1) / elapsed
                return min(measured, self._fps)  # tidak bisa melebihi fps sumber
        return max(self._fps, 1.0)

    def update(self, frame: np.ndarray, has_person: bool, now: float,
               annotations: "list | None" = None) -> None:
        with self._lock:
            self._frame_times.append(now)
            # draw = self._draw(frame, annotations) if annotations else frame  # BBOX_OVERLAY
            draw = frame
            if not self._is_glitch(frame):
                self._last_valid = draw

            if self._state == self._IDLE:
                if has_person:
                    self._start(draw, now)
            elif self._state == self._RECORDING:
                self._write(draw)
                if not has_person:
                    self._idle_since = now
                    self._state = self._COOLING
            elif self._state == self._COOLING:
                self._write(draw)
                if has_person:
                    self._state = self._RECORDING
                elif now - self._idle_since >= CLIP_COOLDOWN:
                    self._finalize(now - self._clip_start)

    def force_stop(self) -> None:
        with self._lock:
            if self._writer is not None:
                self._finalize(time.time() - self._clip_start)

    def _start(self, frame: np.ndarray, now: float) -> None:
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_fps = self._measured_fps()
        path       = CLIPS_DIR / f"clip_{self._camera_id}_{ts}.avi"
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), actual_fps, (self._width, self._height)
        )
        if not self._writer.isOpened():
            print(f"[clip:{self._camera_id}] ERROR: VideoWriter gagal dibuka")
            self._writer = None
            return
        self._clip_path      = path
        self._clip_start     = now
        self._frames_written = 0
        self._state          = self._RECORDING
        print(f"[clip:{self._camera_id}] START → {path.name}  fps={actual_fps:.1f}")
        self._write(frame)

    def _write(self, frame: np.ndarray) -> None:
        if not self._writer:
            return
        if self._is_glitch(frame):
            if self._last_valid is not None:
                self._writer.write(self._last_valid)
                self._frames_written += 1
        else:
            self._last_valid = frame
            self._writer.write(frame)
            self._frames_written += 1

    def _finalize(self, duration: float) -> None:
        if self._writer:
            self._writer.release()
            self._writer = None

        path   = self._clip_path
        frames = self._frames_written
        self._clip_path      = None
        self._clip_start     = 0.0
        self._idle_since     = 0.0
        self._frames_written = 0
        self._state          = self._IDLE

        if path is None:
            return
        size = path.stat().st_size if path.exists() else 0
        print(f"[clip:{self._camera_id}] SAVED {path.name}  wall={duration:.1f}s  frames={frames}  size={size//1024}KB")


# ── Camera slot ───────────────────────────────────────────────────────────────

class _CamSlot:
    """Data + capture thread per kamera. Tidak ada inference di sini."""

    def __init__(
        self,
        camera_id:      str,
        rtsp_url:       str,
        reid_threshold: float,
        stop_event:     threading.Event,
    ) -> None:
        self.camera_id   = camera_id
        self._rtsp_url   = rtsp_url
        # Playlist: comma-separated file paths → sequential playback
        parts = [p.strip() for p in rtsp_url.split(",")]
        self._playlist   = [p for p in parts if Path(p).is_file()]
        # Event-aware playlist: di-set oleh StreamManager untuk koordinasi antar kamera
        self._playlist_events: list[tuple[str, threading.Event | None, threading.Event | None]] | None = None
        self._stop_event = stop_event

        self.db           = IdentityDB(reid_threshold, camera_id)
        self.tracker:     BYTETracker | None  = None
        self.recorder:    ClipRecorder | None = None
        self.best_conf:   dict[str, float]    = {}
        self.last_frame:  np.ndarray | None   = None
        # Metadata frame terbaru saat mode playlist file (untuk _PredictionLogger)
        self.last_source_clip: str | None     = None
        self.last_local_frame: int | None     = None

        # Line crossing
        self.crossing_lines: list[dict]                    = []
        self.has_zone:       bool                          = False
        self._side_hist:     dict[tuple, deque]            = {}
        self._last_dir:      dict[tuple, str]              = {}
        self._crossing_ts:   dict[tuple, float]            = {}  # cooldown per (line_id, track_id)

        self.fps    = 15.0
        self.width  = 1920
        self.height = 1080

        self._cap:        cv2.VideoCapture | None = None
        self.frame_q:     queue.Queue             = queue.Queue(maxsize=2)
        self.rec_q:       queue.Queue             = queue.Queue(maxsize=90)
        self._rec_has_person: bool                = False
        self._rec_annots: list                    = []
        self._rec_thread: threading.Thread | None = None
        self._cap_stop:   threading.Event         = threading.Event()
        self._cap_thread: threading.Thread | None = None
        self.online       = False

        self._lock  = threading.Lock()
        self._state = {
            "running": False, "frames_processed": 0, "identities": []
        }

    def connect(self) -> bool:
        if self._playlist:
            # File playlist mode: buka file pertama hanya untuk baca metadata
            cap = cv2.VideoCapture(self._playlist[0])
            if not cap.isOpened():
                cap.release()
                return False
            self.fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
            self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            self._cap = None
        else:
            cap = _open_capture(self._rtsp_url)
            if cap is None:
                return False
            self._cap   = cap
            self.fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
            self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.recorder is None:
            self.recorder = ClipRecorder(self.camera_id, self.fps, self.width, self.height)
        self._rec_thread = threading.Thread(
            target=self._recorder_loop,
            args=(self.rec_q, self._cap_stop),
            daemon=True, name=f"rec-{self.camera_id}",
        )
        self._rec_thread.start()
        self._launch_capture()
        with self._lock:
            self._state["running"] = True
        return True

    def start_reconnect(self, done_cb) -> None:
        """Reconnect async agar batch loop tidak berhenti menunggu."""
        self.online = False

        def _worker():
            self._cap_stop.set()
            if self._cap_thread:
                self._cap_thread.join(timeout=3.0)
            if self._cap:
                self._cap.release()
                self._cap = None
            for attempt in range(1, RECONNECT_TRIES + 1):
                if self._stop_event.is_set():
                    return
                print(f"[{self.camera_id}] reconnect {attempt}/{RECONNECT_TRIES}…")
                time.sleep(RECONNECT_DELAY)
                cap = _open_capture(self._rtsp_url)
                if cap is not None:
                    self._cap = cap
                    self.fps  = cap.get(cv2.CAP_PROP_FPS) or self.fps
                    self._launch_capture()
                    print(f"[{self.camera_id}] reconnected.")
                    done_cb(self, success=True)
                    return
            print(f"[{self.camera_id}] reconnect gagal.")
            done_cb(self, success=False)

        threading.Thread(target=_worker, daemon=True, name=f"recon-{self.camera_id}").start()

    def _launch_capture(self) -> None:
        self._cap_stop.clear()
        while not self.frame_q.empty():
            try:
                self.frame_q.get_nowait()
            except queue.Empty:
                break
        if self._playlist:
            ev_playlist = self._playlist_events or [(p, None, None) for p in self._playlist]
            target = _CamSlot._capture_loop_files
            args   = (ev_playlist, self.frame_q, self._cap_stop, self.rec_q)
        else:
            target = _CamSlot._capture_loop
            args   = (self._cap, self.frame_q, self._cap_stop, self.rec_q)
        self._cap_thread = threading.Thread(
            target=target, args=args, daemon=True, name=f"cap-{self.camera_id}"
        )
        self._cap_thread.start()
        self.online = True

    def shutdown(self) -> None:
        self.online = False
        self._cap_stop.set()
        if self._cap_thread:
            self._cap_thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        try:
            self.rec_q.put_nowait(None)
        except queue.Full:
            pass
        if self._rec_thread:
            self._rec_thread.join(timeout=3.0)
        if self.recorder:
            self.recorder.force_stop()
        with self._lock:
            self._state["running"] = False

    def _recorder_loop(self, rec_q: queue.Queue, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                frame = rec_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if frame is None:
                break
            if frame is _CLIP_STOP:
                if self.recorder:
                    self.recorder.force_stop()
                continue
            if self.recorder:
                self.recorder.update(
                    frame, self._rec_has_person, time.monotonic(),
                    list(self._rec_annots) if self._rec_annots else None,
                )

    @staticmethod
    def _capture_loop(
        cap: cv2.VideoCapture,
        frame_q: queue.Queue,
        stop: threading.Event,
        rec_q: "queue.Queue | None" = None,
    ) -> None:
        """Hanya baca cap.read() dan simpan frame terbaru. RTSP tetap hidup
        terlepas dari seberapa lambat batch inference berjalan."""
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                while True:
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        break
                frame_q.put(None)
                return
            if frame_q.full():
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                frame_q.put_nowait(frame)
            except queue.Full:
                pass
            if rec_q is not None:
                if rec_q.full():
                    try:
                        rec_q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    rec_q.put_nowait(frame)
                except queue.Full:
                    pass

    @staticmethod
    def _capture_loop_files(
        playlist: list[tuple[str, threading.Event | None, threading.Event | None]],
        frame_q: queue.Queue,
        stop: threading.Event,
        rec_q: "queue.Queue | None" = None,
    ) -> None:
        """Putar file video secara berurutan dengan throttle FPS asli video.
        Tiap entry: (path, wait_event, done_event).
        wait_event: tunggu event ini sebelum mulai (None = langsung).
        done_event: set event ini setelah clip selesai (None = tidak ada).
        Setelah semua selesai, kirim sentinel None."""
        for path, wait_ev, done_ev in playlist:
            if stop.is_set():
                break
            if wait_ev is not None:
                wait_ev.wait()  # tunggu clip sebelumnya selesai
            if stop.is_set():
                break
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[capture] gagal buka {Path(path).name}, skip")
                cap.release()
                if done_ev is not None:
                    done_ev.set()
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = 1.0 / fps
            clip_name = Path(path).name
            local_frame = 0
            print(f"[capture] memutar {clip_name}  fps={fps:.1f}")
            next_time = time.monotonic()
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_q.full():
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    # Tuple (frame, source_clip, local_frame) — dipakai BatchProcessor
                    # untuk logging prediksi; RTSP mode tetap kirim ndarray polos.
                    frame_q.put_nowait((frame, clip_name, local_frame))
                except queue.Full:
                    pass
                local_frame += 1
                if rec_q is not None:
                    if rec_q.full():
                        try:
                            rec_q.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        rec_q.put_nowait(frame)
                    except queue.Full:
                        pass
                next_time += frame_interval
                sleep_dur = next_time - time.monotonic()
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
            cap.release()
            if rec_q is not None:
                rec_q.put(_CLIP_STOP)  # finalize clip sebelum file berikutnya
            if done_ev is not None:
                done_ev.set()  # beri sinyal ke clip berikutnya
        # Semua clip selesai — kirim sentinel
        while True:
            try:
                frame_q.get_nowait()
            except queue.Empty:
                break
        frame_q.put(None)

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def update_state(self, frame_idx: int, identities: list) -> None:
        with self._lock:
            self._state["frames_processed"] = frame_idx
            self._state["identities"]        = identities


# ── Batch processor ───────────────────────────────────────────────────────────

class BatchProcessor:
    """Satu thread inferensi untuk semua kamera.

    Tiap siklus:
      1. Ambil frame terbaru dari setiap kamera (non-blocking)
      2. Kirim semua frame ke YOLO dalam 1 batch predict() call
      3. Jalankan per-camera BYTETracker (slot.tracker) untuk assign track ID
      4. Proses hasil per kamera (ReID, clip, backend POST)
    """

    def __init__(
        self,
        slots:          list[_CamSlot],
        yolo_model:     str,
        reid_model:     str,
        conf_threshold: float,
        stop_event:     threading.Event,
        auto_stop_cb    = None,
    ) -> None:
        self._slots            = slots
        self._conf             = conf_threshold
        self._stop             = stop_event
        self._auto_stop_cb       = auto_stop_cb
        self._file_slots_total:  set[str]              = set()  # slot playlist yang berhasil connect
        self._file_slots_done:   set[str]              = set()  # slot playlist yang sudah selesai
        self._thread: threading.Thread | None = None
        self._offline_reported: set[str]      = set()  # kamera yang sudah dilaporkan offline
        self._pred_logger = _PredictionLogger(PREDICTIONS_CSV)

        if torch.backends.mps.is_available():
            yolo_device = "mps"
        elif torch.cuda.is_available():
            yolo_device = "cuda"
        else:
            yolo_device = "cpu"
        reid_device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[batch] loading YOLO({yolo_model}) → {yolo_device}, ReID({reid_model}) → {reid_device}")
        self._detector = YOLO(yolo_model)
        self._detector.to(yolo_device)
        _msmt17 = Path.home() / ".cache/torch/checkpoints/osnet_ain_x1_0_msmt17.pt"
        self._extractor = torchreid.utils.FeatureExtractor(
            model_name=reid_model,
            model_path=str(_msmt17) if _msmt17.exists() else "",
            device=reid_device,
        )
        # PAR extractor disabled — pure OSNet ReID
        self._par = None
        # _rap1 = Path(__file__).parent.parent.parent / "par_checkpoints" / "RAP1.pth"
        # if _rap1.exists():
        #     try:
        #         from app.par.par_service import PARExtractor
        #         par_device = "cpu"
        #         self._par = PARExtractor(str(_rap1), device=par_device)
        #     except Exception as exc:
        #         print(f"[batch] PAR load failed ({exc}), continuing without PAR tie-breaker")
        # Load ByteTrack config untuk instans tracker per-kamera
        tracker_cfg = YAML.load(check_yaml("bytetrack.yaml"))
        self._tracker_args = IterableSimpleNamespace(**tracker_cfg)
        print("[batch] models ready")

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="batch-infer"
        )
        self._thread.start()

    def join(self, timeout: float = 15.0) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        for slot in self._slots:
            if slot.connect():
                print(f"[{slot.camera_id}] opened {slot.width}x{slot.height} @ {slot.fps:.1f}fps")
                slot.crossing_lines = _fetch_crossing_lines(slot.camera_id)
                slot.has_zone       = _fetch_has_zone(slot.camera_id)
                slot.tracker        = BYTETracker(args=self._tracker_args)
                print(f"[{slot.camera_id}] {len(slot.crossing_lines)} crossing line(s) loaded  has_zone={slot.has_zone}")
            else:
                print(f"[{slot.camera_id}] ERROR: tidak bisa membuka RTSP")

        if not any(s.online for s in self._slots):
            print("[batch] tidak ada kamera aktif.")
            return

        # Track hanya slot playlist yang berhasil connect
        self._file_slots_total = {s.camera_id for s in self._slots if s._playlist and s.online}

        frame_idx    = {s.camera_id: 0 for s in self._slots}
        current_date = datetime.now().date()

        try:
            while not self._stop.is_set():
                today = datetime.now().date()
                if today != current_date:
                    current_date = today
                    # Prune stale bank entries sebelum reset, lalu reset shared DB
                    self._slots[0].db.prune_banks()
                    self._slots[0].db.reset()
                    for slot in self._slots:
                        slot.best_conf.clear()
                        slot._side_hist.clear()
                        slot._last_dir.clear()
                        slot._crossing_ts.clear()
                        slot.tracker = BYTETracker(args=self._tracker_args)
                    print(f"[batch] midnight reset — identity DB dikosongkan untuk {today}")
                now           = time.time()
                batch_frames: list[np.ndarray] = []
                has_new:      list[bool]        = []

                for slot in self._slots:
                    placeholder = (
                        slot.last_frame
                        if slot.last_frame is not None
                        else np.zeros((slot.height, slot.width, 3), np.uint8)
                    )

                    if not slot.online:
                        batch_frames.append(placeholder)
                        has_new.append(False)
                        continue

                    try:
                        frame = slot.frame_q.get_nowait()
                    except queue.Empty:
                        # Belum ada frame baru — pakai placeholder
                        batch_frames.append(placeholder)
                        has_new.append(False)
                        continue

                    if frame is None:
                        if slot._playlist:
                            # File playlist habis — tidak perlu reconnect
                            print(f"[{slot.camera_id}] semua clip selesai")
                            slot.online = False
                            self._file_slots_done.add(slot.camera_id)
                            if self._file_slots_done >= self._file_slots_total and self._auto_stop_cb:
                                threading.Thread(
                                    target=self._auto_stop_cb, daemon=True, name="auto-stop"
                                ).start()
                        else:
                            # RTSP putus — reconnect seperti biasa
                            print(f"[{slot.camera_id}] disconnected")
                            slot.start_reconnect(self._on_reconnect)
                        batch_frames.append(placeholder)
                        has_new.append(False)
                        continue

                    if isinstance(frame, tuple):
                        # Mode file-playback: (frame, source_clip, local_frame)
                        frame, slot.last_source_clip, slot.last_local_frame = frame
                    else:
                        slot.last_source_clip = None
                        slot.last_local_frame = None

                    slot.last_frame = frame
                    batch_frames.append(frame)
                    has_new.append(True)

                if not any(has_new):
                    time.sleep(0.02)
                    continue

                # ── Batch YOLO detect — 1 GPU call untuk semua kamera ────────
                # Pakai predict() bukan track() karena batch track() berbagi 1
                # tracker untuk semua kamera (bug Ultralytics di non-stream mode).
                # Tiap kamera punya BYTETracker sendiri di slot.tracker.
                results = self._detector.predict(
                    batch_frames,
                    conf=self._conf,
                    classes=[0],
                    verbose=False,
                )

                # ── Per-camera post-processing ────────────────────────────────
                for slot, result, is_new in zip(self._slots, results, has_new):
                    if not is_new:
                        continue

                    frame = slot.last_frame
                    boxes = result.boxes

                    # Raw detection count — tidak butuh track ID.
                    # Dipakai untuk clip recorder agar rekaman tetap jalan.
                    n_raw = 0
                    if boxes is not None:
                        n_raw = sum(1 for b in boxes if int(b.cls[0]) == 0)

                    # Per-camera BYTETracker update → track ID per kamera
                    per_box: list[tuple[int, float, int, int, int, int, "np.ndarray | None", float]] = []
                    if boxes is not None and slot.tracker is not None and n_raw > 0:
                        det    = boxes.cpu().numpy()
                        tracks = slot.tracker.update(det, frame)
                        for t in tracks:
                            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                            track_id = int(t[4])
                            conf_val = float(t[5])
                            result   = _extract_embedding(self._extractor, frame, x1, y1, x2, y2,
                                                          track_id=track_id, cam_id=slot.camera_id)
                            emb, quality = result if result is not None else (None, 0.0)
                            per_box.append((track_id, conf_val, x1, y1, x2, y2, emb, quality))

                    active_tids = {tid for tid, *_ in per_box}
                    slot.db.update_active(slot.camera_id, active_tids)
                    for track_id, conf_val, x1, y1, x2, y2, emb, quality in per_box:
                        cam = slot.camera_id
                        if slot.db.name_of(track_id, cam) is None:
                            # Butuh embedding lulus quality gate; skip kalau None
                            if emb is None:
                                continue
                            _dbg_crop = frame[y1:y2, x1:x2].copy()  # selalu pass untuk ambiguous snapshot
                            name, id_new = slot.db.assign(track_id, emb, cam,
                                                          quality=quality, debug_crop=_dbg_crop,
                                                          active_track_ids=active_tids)
                            if name is None:
                                continue  # masih dalam delayed enrollment buffer
                            label        = slot.db.label_of(track_id, cam)
                            # Snapshot unik per detection — selalu disimpan, terlepas dari confidence
                            det_url   = self._save_unique_snapshot(frame, x1, y1, x2, y2, cam, track_id)
                            # Profile thumbnail (file shared, hanya diperbarui kalau confidence lebih tinggi)
                            self._save_thumbnail(slot, label, conf_val, frame, x1, y1, x2, y2)
                            _BackendClient.post_detection(name, cam, conf_val, "appearance", det_url, label, track_id)
                            if id_new:
                                _BackendClient.post_camera_event(
                                    cam, "person_detected", "info",
                                    person_label=label, snapshot_url=det_url,
                                )
                        else:
                            if emb is not None:
                                slot.db.refresh(track_id, emb, cam)
                            label = slot.db.label_of(track_id, cam)
                            if label and label not in slot.best_conf:
                                thumb_url = self._save_thumbnail(slot, label, conf_val, frame, x1, y1, x2, y2)
                                if thumb_url:
                                    _BackendClient.update_thumbnail(label, cam, thumb_url)

                    # ── Prediction logging (mode file-playback saja) ──────────
                    if slot.last_source_clip is not None:
                        for track_id, _, x1, y1, x2, y2, _, _ in per_box:
                            person_pred = slot.db.name_of(track_id, slot.camera_id) or f"Unknown #{track_id}"
                            self._pred_logger.log(
                                slot.last_source_clip, slot.last_local_frame, slot.camera_id,
                                person_pred, x1, y1, x2 - x1, y2 - y1,
                            )

                    # ── Line crossing check ───────────────────────────────────
                    if slot.crossing_lines:
                        for track_id, _, x1, y1, x2, y2, _, _ in per_box:
                            fx, fy = (x1 + x2) // 2, y2  # foot point
                            for line in slot.crossing_lines:
                                key  = (line["id"], track_id)
                                side = _cross_side(
                                    fx, fy,
                                    line["p1_x"], line["p1_y"],
                                    line["p2_x"], line["p2_y"],
                                )
                                hist = slot._side_hist.setdefault(key, deque(maxlen=4))
                                if abs(side) < 1:
                                    continue
                                hist.append(side)
                                if len(hist) >= 2 and hist[-2] * hist[-1] < 0:
                                    direction = "IN" if side * line["in_sign"] > 0 else "OUT"
                                    # Hysteresis: arah sama berturut-turut diabaikan
                                    if slot._last_dir.get(key) == direction:
                                        continue
                                    # Cooldown: minimal CROSSING_COOLDOWN detik antar event per (line, track)
                                    if now - slot._crossing_ts.get(key, 0.0) < CROSSING_COOLDOWN:
                                        continue
                                    slot._last_dir[key]    = direction
                                    slot._crossing_ts[key] = now
                                    event_kind   = "room_entry" if slot.has_zone else "passage"
                                    snap_url     = self._save_event_snapshot(frame, x1, y1, x2, y2, slot.camera_id)
                                    person_label = slot.db.label_of(track_id, slot.camera_id)
                                    _BackendClient.post_occupancy_event(
                                        slot.camera_id, line["id"], direction,
                                        event_kind, snap_url, person_label, track_id,
                                    )
                                    _BackendClient.post_camera_event(
                                        slot.camera_id, "zone_entry", "info",
                                        description=f"{direction} via garis {line['id']}",
                                        snapshot_url=snap_url,
                                        person_label=person_label,
                                    )

                    # Clip state diupdate di sini; frame ditulis oleh _recorder_loop
                    slot._rec_has_person = n_raw > 0
                    slot._rec_annots = [
                        (x1, y1, x2, y2, slot.db.name_of(tid, slot.camera_id) or f"#{tid}")
                        for tid, _, x1, y1, x2, y2, _, _ in per_box
                    ]

                    idx = frame_idx[slot.camera_id] + 1
                    frame_idx[slot.camera_id] = idx

                    if idx % 30 == 0:
                        h, w = frame.shape[:2]
                        tids  = [f"t{tid}({slot.db.name_of(tid, slot.camera_id) or '?'})" for tid, *_ in per_box]
                        print(
                            f"[{slot.camera_id}] f{idx}  {w}x{h}"
                            f"  raw={n_raw}  tracked={len(per_box)}"
                            + (f"  [{', '.join(tids)}]" if tids else "")
                        )

                    if idx % 15 == 0:
                        slot.update_state(idx, slot.db.to_records())

        finally:
            for slot in self._slots:
                slot.shutdown()
            self._pred_logger.close()

    def _on_reconnect(self, slot: _CamSlot, success: bool) -> None:
        if not success:
            print(f"[{slot.camera_id}] offline permanen")
            if slot.camera_id not in self._offline_reported:
                self._offline_reported.add(slot.camera_id)
                _BackendClient.post_camera_event(
                    slot.camera_id, "camera_offline", "critical",
                    description=f"Kamera {slot.camera_id} tidak merespons setelah {RECONNECT_TRIES} percobaan",
                )
        else:
            if slot.camera_id in self._offline_reported:
                self._offline_reported.discard(slot.camera_id)
                _BackendClient.post_camera_event(
                    slot.camera_id, "camera_online", "info",
                    description=f"Kamera {slot.camera_id} kembali online",
                )

    @staticmethod
    def _save_event_snapshot(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        camera_id: str,
    ) -> "str | None":
        """Simpan crop orang saat crossing terjadi ke thumbnails/events/.
        Crop diperbesar dari titik tengah bounding box untuk memastikan
        minimal 120x240 px sehingga orang selalu terlihat jelas."""
        try:
            events_dir = THUMBNAILS_DIR / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            fh, fw = frame.shape[:2]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half_w = max((x2 - x1) // 2 + 30, 60)   # min 120 px lebar
            half_h = max((y2 - y1) // 2 + 40, 120)  # min 240 px tinggi

            x1c = max(0, cx - half_w)
            y1c = max(0, cy - half_h)
            x2c = min(fw, cx + half_w)
            y2c = min(fh, cy + half_h)

            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_{ts}.jpg"
            cv2.imwrite(str(events_dir / filename), frame[y1c:y2c, x1c:x2c])
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")
            return f"{backend_url}/thumbnails/events/{filename}"
        except Exception as exc:
            print(f"[{camera_id}] event snapshot error: {exc}")
            return None

    @staticmethod
    def _save_unique_snapshot(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        camera_id: str,
        track_id: int,
    ) -> "str | None":
        """Simpan snapshot unik per detection track — selalu disimpan tanpa cek confidence.
        Dipakai sebagai thumbnail_url di tabel detections agar tiap baris punya foto sendiri."""
        try:
            fh, fw = frame.shape[:2]
            cx, cy  = (x1 + x2) // 2, (y1 + y2) // 2
            half_w  = max((x2 - x1) // 2 + 30, 60)
            half_h  = max((y2 - y1) // 2 + 40, 120)
            x1c = max(0, cx - half_w)
            y1c = max(0, cy - half_h)
            x2c = min(fw, cx + half_w)
            y2c = min(fh, cy + half_h)
            THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_t{track_id}_{ts}.jpg"
            cv2.imwrite(str(THUMBNAILS_DIR / filename), frame[y1c:y2c, x1c:x2c])
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")
            return f"{backend_url}/thumbnails/{filename}"
        except Exception as exc:
            print(f"[{camera_id}] detection snapshot error: {exc}")
            return None

    @staticmethod
    def _save_thumbnail(
        slot: _CamSlot, label: str, conf: float,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> "str | None":
        if conf <= slot.best_conf.get(label, -1.0):
            return None
        try:
            fh, fw = frame.shape[:2]
            pad    = 20
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(fw, x2+pad), min(fh, y2+pad)
            if x2-x1 < 8 or y2-y1 < 8:
                return None
            THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
            filename = re.sub(r"[^\w]+", "_", label).strip("_") + ".jpg"
            cv2.imwrite(str(THUMBNAILS_DIR / filename), frame[y1:y2, x1:x2])
            slot.best_conf[label] = conf
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")
            return f"{backend_url}/thumbnails/{filename}"
        except Exception as exc:
            print(f"[{slot.camera_id}] thumbnail error: {exc}")
            return None


# ── StreamManager ─────────────────────────────────────────────────────────────

class StreamManager:
    """Public API; satu instance di app.state."""

    def __init__(self) -> None:
        self._slots:            list[_CamSlot]       = []
        self._processor:        BatchProcessor | None = None
        self._stop_event:       threading.Event       = threading.Event()
        self._lock:             threading.Lock        = threading.Lock()
        self._display_names:    dict[str, str]        = {}
        self._running                                 = False
        self._video_identities: list[IdentityRecord]  = []
        self._shared_db:        IdentityDB | None     = None

    @property
    def shared_db(self) -> "IdentityDB | None":
        return self._shared_db

    def start(
        self,
        yolo_model:     str,
        reid_model:     str,
        conf_threshold: float,
        reid_threshold: float,
        line           = None,   # legacy, tidak digunakan — garis diambil dari DB
    ) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Stream sudah berjalan. Panggil /stream/stop dulu.")

            # Prioritas: ambil dari database, fallback ke env
            db_cameras = _fetch_cameras()
            if db_cameras:
                cam_configs = [
                    {
                        "camera_id": c.get("camera_id") or _camera_id_from_url(c["rtsp_url"]),
                        "rtsp_url":  c["rtsp_url"],
                        "name":      c.get("name", ""),
                    }
                    for c in db_cameras
                ]
                print(f"[stream] {len(cam_configs)} kamera dari database.")
            else:
                urls_raw = os.getenv("RTSP_URLS", "").strip()
                if not urls_raw:
                    raise RuntimeError(
                        "Tidak ada kamera di database dan RTSP_URLS tidak ditemukan di .env"
                    )
                urls = [u.strip() for u in urls_raw.split(",") if u.strip()]
                if not urls:
                    raise RuntimeError("RTSP_URLS kosong atau tidak valid.")
                cam_configs = [
                    {"camera_id": _camera_id_from_url(u), "rtsp_url": u, "name": ""}
                    for u in urls
                ]
                print(f"[stream] {len(cam_configs)} kamera dari .env (fallback).")

            self._stop_event.clear()
            self._slots = [
                _CamSlot(
                    camera_id      = cfg["camera_id"],
                    rtsp_url       = cfg["rtsp_url"],
                    reid_threshold = reid_threshold,
                    stop_event     = self._stop_event,
                )
                for cfg in cam_configs
            ]

            # Shared IdentityDB (PAR extractor wired in after BatchProcessor is created)
            shared_db = IdentityDB(reid_threshold)
            self._shared_db = shared_db
            for slot in self._slots:
                slot.db = shared_db

            # Bangun event chain antar clip berdasarkan urutan timestamp di nama file
            all_clips: list[tuple[_CamSlot, str]] = []
            for slot in self._slots:
                for path in slot._playlist:
                    all_clips.append((slot, path))
            if all_clips:
                def _ts_key(item: tuple) -> str:
                    m = re.search(r"\d{8}_\d{6}", item[1])
                    return m.group() if m else ""
                all_clips.sort(key=_ts_key)
                events = [threading.Event() for _ in range(len(all_clips) - 1)]
                per_slot: dict[str, list[tuple[str, threading.Event | None, threading.Event | None]]] = {
                    s.camera_id: [] for s in self._slots
                }
                for i, (slot, path) in enumerate(all_clips):
                    wait_ev = events[i - 1] if i > 0 else None
                    done_ev = events[i]     if i < len(all_clips) - 1 else None
                    per_slot[slot.camera_id].append((path, wait_ev, done_ev))
                for slot in self._slots:
                    if slot._playlist:
                        slot._playlist_events = per_slot[slot.camera_id]
            has_playlist = any(s._playlist for s in self._slots)
            self._processor = BatchProcessor(
                slots          = self._slots,
                yolo_model     = yolo_model,
                reid_model     = reid_model,
                conf_threshold = conf_threshold,
                stop_event     = self._stop_event,
                auto_stop_cb   = self.stop if has_playlist else None,
            )
            # Wire PAR extractor into shared IdentityDB after BatchProcessor is ready
            shared_db._par = self._processor._par
            self._processor.start()
            self._running = True
            print(f"[stream] {len(self._slots)} kamera dimulai (batch mode).")

    def stop(self) -> None:
        self._stop_event.set()
        if self._processor:
            self._processor.join()
        with self._lock:
            self._running = False

    def update_identities(self, identities: list) -> None:
        with self._lock:
            self._video_identities = list(identities)

    def get_identities(self) -> list[IdentityRecord]:
        records = self._all_identity_records()
        with self._lock:
            return [
                IdentityRecord(
                    name      = self._display_names.get(r.name, r.name),
                    track_ids = r.track_ids,
                )
                for r in records
            ]

    def rename_identity(self, old_name: str, new_name: str) -> "IdentityRecord | None":
        for record in self._all_identity_records():
            with self._lock:
                display = self._display_names.get(record.name, record.name)
            if display == old_name:
                with self._lock:
                    self._display_names[record.name] = new_name
                return IdentityRecord(name=new_name, track_ids=record.track_ids)
        return None

    def status(self) -> StreamStatusResponse:
        frames = sum(s.get_state()["frames_processed"] for s in self._slots)
        with self._lock:
            running = self._running
        return StreamStatusResponse(
            running          = running,
            count_in         = 0,
            count_out        = 0,
            frames_processed = frames,
            identities       = self.get_identities(),
            rtsp_configured  = bool(self._slots) or bool(os.getenv("RTSP_URLS", "").strip()),
        )

    def get_snapshot(self, camera_id: str) -> "np.ndarray | None":
        for slot in self._slots:
            if slot.camera_id == camera_id and slot.last_frame is not None:
                return slot.last_frame.copy()
        return None

    def _all_identity_records(self) -> list[IdentityRecord]:
        seen: set[str] = set()
        records: list[IdentityRecord] = []
        for slot in self._slots:
            for r in slot.get_state()["identities"]:
                if r.name not in seen:
                    seen.add(r.name)
                    records.append(r)
        with self._lock:
            records.extend(self._video_identities)
        return records
