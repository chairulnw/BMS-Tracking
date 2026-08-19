"""Rekam klip video per kamera saat ada orang terdeteksi, plus logger CSV
prediksi untuk mode file-playback (dibandingkan dengan ground truth)."""

import csv
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

CLIP_COOLDOWN  = 5.0
CLIPS_DIR      = Path("output/clips")

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

        # `actual_fps` di _start() adalah TEBAKAN dari sampel SEBELUM rekaman
        # mulai — kalau pipeline sempat lambat (beban CPU tinggi, dll) pas
        # sampel itu diambil, tebakan itu ke-"kunci" untuk SELURUH klip walau
        # kecepatan sebenarnya berubah-ubah, hasilnya playback kerasa lambat/
        # patah-patah. Di sini kita tahu PERSIS berapa frame ditulis dan
        # berapa detik nyata terpakai — hitung ulang fps yang jujur dan
        # remux (stream-copy, tanpa transcode ulang) supaya declared fps
        # klip cocok dengan kecepatan rekam yang sebenarnya terjadi.
        if frames >= 2 and duration > 0.1:
            true_fps = frames / duration
            # Thread terpisah, fire-and-forget — _finalize() dipanggil dari
            # dalam update() yang MEMEGANG self._lock; subprocess ffmpeg
            # sinkron di sini akan menahan lock dan bikin frame berikutnya
            # (klip BARU yang mungkin langsung mulai) ikut ketahan/hilang.
            threading.Thread(
                target=self._fix_container_fps, args=(path, true_fps),
                daemon=True, name=f"fixfps-{self._camera_id}",
            ).start()

    def _fix_container_fps(self, path: Path, true_fps: float) -> None:
        tmp = path.with_suffix(".fixfps.avi")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-c", "copy", "-r", f"{true_fps:.3f}", str(tmp)],
                check=True, capture_output=True, timeout=10,
            )
            tmp.replace(path)
        except Exception as exc:
            print(f"[clip:{self._camera_id}] gagal koreksi fps ({path.name}): {exc}")
            tmp.unlink(missing_ok=True)
