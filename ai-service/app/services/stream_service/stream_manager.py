"""Public API stream; satu instance di app.state."""

import re
import threading

import numpy as np

from app.schemas import IdentityRecord, StreamStatusResponse
from app.services.pipeline_service import IdentityDB
from app.services.stream_service.backend_client import _fetch_cameras, _fetch_tracklet_gallery
from app.services.stream_service.batch_processor import BatchProcessor, load_model_bundle
from app.services.stream_service.cam_slot import _CamSlot, _camera_id_from_url


class StreamManager:
    """Public API; satu instance di app.state."""

    def __init__(self) -> None:
        self._slots:            list[_CamSlot]       = []
        self._processor:        BatchProcessor | None = None
        self._stop_event:       threading.Event       = threading.Event()
        self._lock:              threading.Lock        = threading.Lock()
        self._display_names:    dict[str, str]        = {}
        self._running                                 = False
        self._video_identities: list[IdentityRecord]  = []
        self._shared_db:        IdentityDB | None     = None

        # Model YOLO/ReID di-cache di sini setelah pertama kali di-load, supaya
        # restart stream (yang sekarang otomatis kejadian tiap kamera di-save)
        # tidak reload model dari disk berulang-ulang. Lock terpisah dari
        # self._lock supaya status() tidak ikut ke-block selama loading.
        self._model_lock = threading.Lock()
        self._models = None  # _ModelBundle | None

    @property
    def shared_db(self) -> "IdentityDB | None":
        return self._shared_db

    def _get_or_load_models(self, detector_model: str, reid_model: str):
        with self._model_lock:
            if self._models is None:
                self._models = load_model_bundle(detector_model, reid_model)
            return self._models

    def start(
        self,
        detector_model: str,
        reid_model:     str,
        conf_threshold: float,
        reid_threshold: float,
        line           = None,   # legacy, tidak digunakan — garis diambil dari DB
        skip_gallery_restore: bool = False,          # True untuk evaluasi terisolasi
        skip_recording: "bool | None" = None,   # None → ikut skip_gallery_restore (perilaku lama)
    ) -> None:
        if skip_recording is None:
            skip_recording = skip_gallery_restore
        if self._running:
            raise RuntimeError("Stream sudah berjalan. Panggil /stream/stop dulu.")

        # Loading model (mahal, sekali per proses) dilakukan DI LUAR self._lock —
        # dia gak menyentuh state bersama, jadi status()/route lain gak perlu nunggu.
        models = self._get_or_load_models(detector_model, reid_model)

        with self._lock:
            if self._running:
                raise RuntimeError("Stream sudah berjalan. Panggil /stream/stop dulu.")

            db_cameras = _fetch_cameras()
            if not db_cameras:
                raise RuntimeError(
                    "Tidak ada kamera aktif di database. Tambahkan lewat /pengaturan."
                )
            cam_configs = [
                {
                    "camera_id":         c.get("camera_id") or _camera_id_from_url(c["rtsp_url"]),
                    "rtsp_url":          c["rtsp_url"],
                    "name":              c.get("name", ""),
                    "analytics_enabled": c.get("analytics_enabled", True),
                }
                for c in db_cameras
            ]
            print(f"[stream] {len(cam_configs)} kamera dari database.")

            self._stop_event.clear()
            self._slots = [
                _CamSlot(
                    camera_id         = cfg["camera_id"],
                    rtsp_url          = cfg["rtsp_url"],
                    reid_threshold    = reid_threshold,
                    stop_event        = self._stop_event,
                    analytics_enabled = cfg.get("analytics_enabled", True),
                    skip_recording    = skip_recording,
                )
                for cfg in cam_configs
            ]

            # Shared IdentityDB (PAR extractor wired in setelah models siap)
            shared_db = IdentityDB(reid_threshold)
            shared_db._par = models.par
            self._shared_db = shared_db
            for slot in self._slots:
                slot.db = shared_db

            # Pulihkan gallery ReID hari ini dari DB — supaya restart AI service
            # di tengah hari tidak membuat orang yang sama dapat Person ID baru
            # (plan/07-fase2-detail.md §7). Hanya tracklet hari ini, sesuai ADR-001.
            # Dilewati kalau skip_gallery_restore=True (evaluasi terisolasi,
            # mis. akurasi via file-playlist) — shared_db tetap kosong-baru,
            # tidak menyentuh atau terpengaruh data live di DB.
            if skip_gallery_restore:
                print("[stream] skip_gallery_restore=True — IdentityDB mulai kosong.")
            else:
                gallery_entries = _fetch_tracklet_gallery()
                if gallery_entries:
                    shared_db.load_gallery(gallery_entries)
                    print(f"[stream] gallery dipulihkan: {len(gallery_entries)} entri, "
                          f"{len(shared_db._embeddings)} orang")

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
                models         = models,
                conf_threshold = conf_threshold,
                stop_event     = self._stop_event,
                auto_stop_cb   = self.stop if has_playlist else None,
                skip_backend_persist = skip_gallery_restore,
            )
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
            rtsp_configured  = bool(self._slots),
        )

    def get_batch_metrics(self) -> dict | None:
        return self._processor.get_metrics() if self._processor else None

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
