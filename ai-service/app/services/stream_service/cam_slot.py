"""Data + capture thread per kamera. Tidak ada inference di sini — cuma
cap.read() dan serah-terima frame lewat queue ke BatchProcessor."""

import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;1048576",
)
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from app.services.pipeline_service import IdentityDB
from app.services.stream_service.clip_recorder import _CLIP_STOP, ClipRecorder

RECONNECT_TRIES = 5      # percobaan cepat sebelum lapor camera_offline
RECONNECT_DELAY = 2.0    # jeda antar percobaan cepat
BACKGROUND_RETRY_DELAY = 30.0  # interval tetap pasca-give-up (bukan backoff — cukup buat retry tanpa-batas)


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


class _CamSlot:
    """Data + capture thread per kamera. Tidak ada inference di sini."""

    def __init__(
        self,
        camera_id:          str,
        rtsp_url:           str,
        reid_threshold:     float,
        stop_event:         threading.Event,
        analytics_enabled:  bool = True,
        skip_recording:     bool = False,   # True untuk evaluasi terisolasi — tidak tulis klip/thumbnail
    ) -> None:
        self.camera_id          = camera_id
        self.analytics_enabled  = analytics_enabled
        self.skip_recording     = skip_recording
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
        self.last_frame:  np.ndarray | None   = None
        # Metadata frame terbaru saat mode playlist file (untuk _PredictionLogger)
        self.last_source_clip: str | None     = None
        self.last_local_frame: int | None     = None
        self.last_decode_ms:   float          = 0.0   # durasi cap.read() terakhir (_LatencyLogger)

        # Zona (line + polygon) yang dipantau kamera ini
        self.zones: list[dict]                             = []
        self._side_hist:      dict[tuple, deque]           = {}
        self._last_dir:       dict[tuple, str]              = {}
        self._crossing_ts:    dict[tuple, float]            = {}  # cooldown per (zone_camera_id[+seg], track_id)
        self._polygon_inside: dict[tuple, bool]             = {}  # state dwell per (zone_camera_id, track_id)

        self.fps    = 15.0
        self.width  = 1920
        self.height = 1080

        self._cap:        cv2.VideoCapture | None = None
        # Dibatasi (bukan tak terbatas) — proses frame TERBARU, buang yang lama
        # kalau penuh. Antrian tak terbatas dicoba untuk file-playlist tapi bikin
        # backlog: frame yang di video aslinya berjarak sepersekian detik bisa
        # keproses berdetik-detik terpisah, salah-nutup gap-detection tracklet
        # (TRACKLET_GAP_CYCLES di pipeline_service.py). Ukuran 5 beri toleransi
        # jitter tanpa balik ke backlog tak terbatas.
        self.frame_q:     queue.Queue             = queue.Queue(maxsize=5)
        # Diisi LANGSUNG oleh BatchProcessor tiap siklus dengan (frame, has_person,
        # annots) sebagai satu paket atomik — bukan thread capture, karena capture
        # jauh lebih cepat dari analisis dan box akan selalu telat vs frame kalau
        # disuplai independen (3 pendekatan lain sudah dicoba, semua gagal).
        self.rec_q:       queue.Queue             = queue.Queue(maxsize=90)
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
        if not self.skip_recording:
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
        """Reconnect async agar batch loop tidak berhenti menunggu. Tidak
        pernah give-up permanen: percobaan cepat (RECONNECT_TRIES) untuk lapor
        offline secepatnya, lalu lanjut coba di background tiap
        BACKGROUND_RETRY_DELAY selama service masih jalan — begitu kamera
        beneran nyala lagi, otomatis connect tanpa restart manual."""
        self.online = False

        def _worker():
            self._cap_stop.set()
            if self._cap_thread:
                self._cap_thread.join(timeout=3.0)
            if self._cap:
                self._cap.release()
                self._cap = None
            attempt = 0
            while not self._stop_event.is_set():
                attempt += 1
                delay = RECONNECT_DELAY if attempt <= RECONNECT_TRIES else BACKGROUND_RETRY_DELAY
                print(f"[{self.camera_id}] reconnect attempt {attempt} (jeda {delay:.0f}s)…")
                time.sleep(delay)
                if self._stop_event.is_set():
                    return
                cap = _open_capture(self._rtsp_url)
                if cap is not None:
                    self._cap = cap
                    self.fps  = cap.get(cv2.CAP_PROP_FPS) or self.fps
                    self._launch_capture()
                    print(f"[{self.camera_id}] reconnected setelah {attempt} percobaan.")
                    done_cb(self, success=True)
                    return
                if attempt == RECONNECT_TRIES:
                    print(f"[{self.camera_id}] {RECONNECT_TRIES}x gagal — lapor offline, "
                          f"tetap coba reconnect di background tiap {BACKGROUND_RETRY_DELAY:.0f}s.")
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
            # rec_q di sini cuma buat sinyal _CLIP_STOP (klip sumber pindah
            # file) — frame rekaman sendiri disuplai BatchProcessor langsung.
            rec_q = None if self.skip_recording else self.rec_q
            ev_playlist = self._playlist_events or [(p, None, None) for p in self._playlist]
            target = _CamSlot._capture_loop_files
            args   = (ev_playlist, self.frame_q, self._cap_stop, rec_q)
        else:
            target = _CamSlot._capture_loop
            args   = (self._cap, self.frame_q, self._cap_stop)
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
        """rec_q diisi BatchProcessor dengan (frame, has_person, annots) —
        satu paket atomik per siklus, jadi box di sini SELALU cocok sama
        frame-nya, gak perlu dicocokkan/divalidasi lagi (lihat komentar di
        __init__ dan pemanggil di batch_processor.py)."""
        while not stop.is_set():
            try:
                item = rec_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            if item is _CLIP_STOP:
                if self.recorder:
                    self.recorder.force_stop()
                continue
            frame, has_person, annots = item
            if self.recorder:
                self.recorder.update(frame, has_person, time.monotonic(), annots or None)

    @staticmethod
    def _capture_loop(
        cap: cv2.VideoCapture,
        frame_q: queue.Queue,
        stop: threading.Event,
    ) -> None:
        """Hanya baca cap.read() dan simpan frame terbaru. RTSP tetap hidup
        terlepas dari seberapa lambat batch inference berjalan. Rekaman TIDAK
        disuplai dari sini lagi — BatchProcessor yang push ke rec_q langsung
        (lihat _recorder_loop), biar box selalu cocok sama frame-nya."""
        while not stop.is_set():
            _t0 = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter() - _t0) * 1000
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
                # item selalu (frame, decode_ms, source_clip, local_frame) —
                # None dua terakhir buat RTSP (bukan file-playlist), disamain
                # sama _capture_loop_files() biar consumer di batch_processor.py
                # unpack satu bentuk aja, bukan cek isinstance tuple/ndarray.
                frame_q.put_nowait((frame, decode_ms, None, None))
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
                _t0 = time.perf_counter()
                ok, frame = cap.read()
                decode_ms = (time.perf_counter() - _t0) * 1000
                if not ok:
                    break
                if frame_q.full():
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    # (frame, decode_ms, source_clip, local_frame) — source_clip/
                    # local_frame dipakai BatchProcessor buat logging prediksi;
                    # RTSP kirim bentuk sama dengan 2 field terakhir None.
                    # Rekaman TIDAK disuplai dari sini — lihat _recorder_loop.
                    frame_q.put_nowait((frame, decode_ms, clip_name, local_frame))
                except queue.Full:
                    pass
                local_frame += 1
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
