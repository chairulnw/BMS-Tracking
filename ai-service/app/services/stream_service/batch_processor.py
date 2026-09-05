"""Satu thread inferensi untuk semua kamera — model YOLO/ReID di-load lewat
_ModelBundle (lihat load_model_bundle) yang di-cache di StreamManager supaya
restart stream tidak reload model dari disk tiap kali."""

import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
import torchreid
from ocsort.ocsort import OCSort
from ultralytics import RTDETR, YOLO
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from app.services.geometry import cross_side
from app.services.pipeline_service import _extract_embedding
from app.services.stream_service.backend_client import _BackendClient, _fetch_zones
from app.services.stream_service.cam_slot import _CamSlot

CROSSING_COOLDOWN = 3.0   # detik minimum antar event crossing per (garis/polygon, track)
# Cuma buat timestamp latency.csv (biar gampang dibaca manual, single-location
# Jakarta) — timestamp event ke backend/DB TETAP UTC (timezone.utc di tempat
# lain file ini), jangan ikut diganti.
_WIB = ZoneInfo("Asia/Jakarta")
THUMBNAILS_DIR    = Path("thumbnails")
PREDICTIONS_CSV   = Path("predictions.csv")  # log prediksi mode file-playback, utk dibanding ground truth

# Ukuran input YOLO (default ultralytics 640). Turunin → deteksi lebih cepat,
# tapi orang kecil/jauh lebih sering ke-miss. Env override buat tuning.
DETECT_IMGSZ = int(os.getenv("DETECT_IMGSZ", "480"))
# Ekstraksi embedding ReID cuma jalan 1x tiap N siklus per track (tracklet
# cuma nyimpan 16 sampel terbaik — ekstrak tiap frame itu mubazir & jadi
# bottleneck FPS pas rame). observe() tetap dipanggil tiap siklus.
REID_EVERY_N = int(os.getenv("REID_EVERY_N", "3"))


class _OCSortAdapter:
    """Bungkus OCSort (paket `ocsort`, interface array polos) biar bisa dipanggil
    persis sama seperti BYTETracker/BOTSORT: .update(det, frame), dengan det
    berupa objek Boxes ultralytics (.xyxy/.conf/.cls), balikin baris
    [x1,y1,x2,y2,track_id,conf] (indeks 0-5 dipakai kode pemanggil)."""

    def __init__(self, args=None) -> None:
        self._oc = OCSort()

    def update(self, det, frame=None) -> np.ndarray:
        if det is None or len(det) == 0:
            return np.empty((0, 6))
        dets = np.concatenate(
            [det.xyxy, det.conf.reshape(-1, 1), det.cls.reshape(-1, 1)], axis=1
        )
        # paket `ocsort` ini panggil .numpy() di dalam update() sendiri,
        # jadi butuh torch.Tensor sebagai input, bukan ndarray polos.
        tracks = self._oc.update(torch.as_tensor(dets), None)  # -> [x1,y1,x2,y2,id,cls,conf]
        if len(tracks) == 0:
            return np.empty((0, 6))
        return tracks[:, [0, 1, 2, 3, 4, 6]]   # -> [x1,y1,x2,y2,id,conf]


# ponytail: bytetrack/botsort masih lewat args+yaml bawaan Ultralytics;
# ocsort dibungkus _OCSortAdapter jadi interface-nya sama (tuple kedua None
# = tidak ada yaml config buat OC-SORT).
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "bytetrack")   # bytetrack | botsort | ocsort
_TRACKER_REGISTRY = {"bytetrack": (BYTETracker,      "bytetrack.yaml"),
                     "botsort":   (BOTSORT,          "botsort.yaml"),
                     "ocsort":    (_OCSortAdapter,   None)}


@dataclass
class _ModelBundle:
    """Model YOLO + ReID yang sudah di-load, plus config tracker. Dibuat sekali
    lewat load_model_bundle() dan dipakai ulang di setiap BatchProcessor —
    sebelumnya model ini di-load ulang dari disk tiap kali stream direstart."""
    detector:     YOLO
    extractor:    "torchreid.utils.FeatureExtractor"
    tracker_args: IterableSimpleNamespace
    tracker_cls:  type
    par:          None = None


def load_model_bundle(detector_model: str, reid_model: str) -> _ModelBundle:
    if torch.backends.mps.is_available():
        detector_device = "mps"
    elif torch.cuda.is_available():
        detector_device = "cuda"
    else:
        detector_device = "cpu"
    # ReID (OSNet) jalan per box terdeteksi, tiap siklus, di semua kamera —
    # jauh lebih sering dipanggil daripada YOLO per-frame. Sebelumnya cuma
    # cek cuda, jadi selalu jatuh ke CPU di Mac walau MPS ada — kemungkinan
    # besar inilah bottleneck asli di balik throughput rendah (klip slow-mo,
    # predictions.csv << ground truth).
    if torch.backends.mps.is_available():
        reid_device = "mps"
    elif torch.cuda.is_available():
        reid_device = "cuda"
    else:
        reid_device = "cpu"

    # RTDETR pakai kelas ultralytics beda dari YOLO — dipilih dari nama file
    # weight-nya (mis. "rtdetr-l.pt"), biar DETECTOR_MODEL tetap satu env var saja.
    detector_cls = RTDETR if "rtdetr" in detector_model.lower() else YOLO
    print(f"[batch] loading {detector_cls.__name__}({detector_model}) → {detector_device}, "
          f"ReID({reid_model}) → {reid_device}")
    detector = detector_cls(detector_model)
    detector.to(detector_device)
    # ponytail: pemetaan model_name -> checkpoint lokal masih manual satu-satu.
    # Kalau nambah model Re-ID baru, tambahkan baris di sini.
    _reid_checkpoints = {
        "osnet_ain_x1_0": "osnet_ain_x1_0_msmt17.pt",
        "resnet50":       "resnet50_market1501_converted.pth",
    }
    if reid_model == "transreid":
        from app.services.stream_service.transreid_extractor import TransReIDExtractor
        extractor = TransReIDExtractor(device=reid_device)
    elif reid_model == "bot_resnet50":
        from app.services.stream_service.bot_resnet50_extractor import BotResNet50Extractor
        extractor = BotResNet50Extractor(device=reid_device)
    else:
        _ckpt_name = _reid_checkpoints.get(reid_model)
        _ckpt_path = Path.home() / ".cache/torch/checkpoints" / _ckpt_name if _ckpt_name else None
        extractor = torchreid.utils.FeatureExtractor(
            model_name=reid_model,
            model_path=str(_ckpt_path) if _ckpt_path and _ckpt_path.exists() else "",
            device=reid_device,
        )
    tracker_cls, tracker_yaml = _TRACKER_REGISTRY[TRACKER_TYPE]
    print(f"[batch] tracker: {TRACKER_TYPE}")
    tracker_args = None
    if tracker_yaml is not None:
        tracker_cfg  = YAML.load(check_yaml(tracker_yaml))
        tracker_cfg["track_buffer"] = 90
        # FPS efektif rendah (~1-3/kamera) → orang loncat jauh antar siklus
        # inferensi, IoU box turun di bawah default 0.8 → ByteTrack keburu
        # ganti track_id (contoh: track 21→22 di tengah satu kemunculan).
        # Dilonggarkan biar asosiasi masih nyambung; naikkan lagi kalau mulai
        # nyambungin dua orang beda yang lewat berdekatan.
        tracker_cfg["match_thresh"] = 0.6
        tracker_args = IterableSimpleNamespace(**tracker_cfg)

    # PAR (atribut penampilan, Fase 3) — dijalankan sekali per tracklet pada
    # crop terbaik (lihat _resolve_tracklet di pipeline_service.py), bukan per
    # frame: ~2s/crop di CPU, per-frame akan melumpuhkan batch loop.
    par = None
    rap1_checkpoint = Path("checkpoints/par_checkpoints/RAP1.pth")
    if rap1_checkpoint.exists():
        from app.par.par_service import PARExtractor
        try:
            par = PARExtractor(str(rap1_checkpoint), device=reid_device)
        except Exception as exc:
            print(f"[batch] PAR gagal dimuat, lanjut tanpa atribut: {exc}")
    else:
        print(f"[batch] {rap1_checkpoint} tidak ditemukan — lanjut tanpa PAR")

    print("[batch] models ready")
    return _ModelBundle(detector=detector, extractor=extractor, tracker_args=tracker_args,
                         tracker_cls=tracker_cls, par=par)


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
        models:         _ModelBundle,
        conf_threshold: float,
        stop_event:     threading.Event,
        auto_stop_cb    = None,
        skip_backend_persist: bool = False,   # True untuk evaluasi terisolasi
    ) -> None:
        self._slots            = slots
        self._conf             = conf_threshold
        self._stop             = stop_event
        self._auto_stop_cb       = auto_stop_cb
        self._skip_backend_persist = skip_backend_persist
        self._file_slots_total:  set[str]              = set()  # slot playlist yang berhasil connect
        self._file_slots_done:   set[str]               = set()  # slot playlist yang sudah selesai
        self._thread: threading.Thread | None = None
        self._offline_reported: set[str]      = set()  # kamera yang sudah dilaporkan offline
        self._reid_tick: dict[tuple[str, int], int] = {}  # (cam,track) → hitungan siklus, buat throttle ReID
        # System Health (plan2/spesifikasi.md Fase 2) — waktu predict() batch
        # terakhir, buat tahu kapan BatchProcessor mulai jadi bottleneck.
        self._batch_ms: "deque[float]" = deque(maxlen=50)
        self._frame_idx: dict[str, int] = {}   # diisi _loop(), dibaca resource_sampler buat fps_effective
        # total_ai_ms frame TERAKHIR per kamera — approx kasar buat ditempel ke
        # payload camera_events (ai_latency_ms), BUKAN rata-rata seluruh
        # tracklet (itu butuh lebih banyak bookkeeping, lihat diskusi latency).
        self._last_ai_ms: dict[str, float] = {}
        # Toggle instrumentasi latency/resource — default nyala (evaluasi
        # kombinasi maupun produksi biasa sama-sama kepake), matiin cuma kalau
        # emang nggak mau overhead nulis CSV terus-terusan.
        self._latency_enabled = os.getenv("ENABLE_LATENCY_LOG", "1").lower() not in ("0", "false", "no")

        from app.services.stream_service.clip_recorder import _PredictionLogger
        self._pred_logger = _PredictionLogger(PREDICTIONS_CSV)

        from app.services.stream_service.latency_logger import _LatencyLogger
        self._latency_logger = _LatencyLogger()

        from app.services.stream_service.resource_sampler import ResourceSampler
        self._resource_sampler = ResourceSampler(
            get_total_frames=lambda: sum(self._frame_idx.values()),
            stop_event=stop_event,
        )

        self._detector      = models.detector
        self._extractor     = models.extractor
        self._tracker_args  = models.tracker_args
        self._tracker_cls   = models.tracker_cls
        self._par           = models.par

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="batch-infer"
        )
        self._thread.start()
        if self._latency_enabled:
            self._resource_sampler.start()

    def join(self, timeout: float = 15.0) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        for slot in self._slots:
            if slot.connect():
                print(f"[{slot.camera_id}] opened {slot.width}x{slot.height} @ {slot.fps:.1f}fps")
                slot.zones   = _fetch_zones(slot.camera_id)
                slot.tracker = self._tracker_cls(args=self._tracker_args)
                n_line = sum(1 for z in slot.zones if z["type"] == "line")
                n_poly = sum(1 for z in slot.zones if z["type"] == "polygon")
                print(f"[{slot.camera_id}] {n_line} line-zone(s), {n_poly} polygon-zone(s) loaded")
            else:
                print(f"[{slot.camera_id}] ERROR: tidak bisa membuka RTSP")

        if not any(s.online for s in self._slots):
            print("[batch] tidak ada kamera aktif.")
            return

        # Track hanya slot playlist yang berhasil connect
        self._file_slots_total = {s.camera_id for s in self._slots if s._playlist and s.online}

        frame_idx    = self._frame_idx = {s.camera_id: 0 for s in self._slots}
        current_date = datetime.now().date()

        try:
            while not self._stop.is_set():
                today = datetime.now().date()
                if today != current_date:
                    current_date = today
                    # Flush semua tracklet terbuka SEBELUM reset — kalau tidak,
                    # rebuild BYTETracker di bawah mendaur ulang track_id dari 1,
                    # dan tracklet kemarin yang belum ditutup akan menerima box
                    # orang lain hari ini (lihat plan/07-fase2-detail.md §3).
                    closed = self._slots[0].db.close_all()
                    if closed:
                        self._finalize_tracklets(closed)
                    self._slots[0].db.prune_banks()
                    self._slots[0].db.reset()
                    self._reid_tick.clear()
                    for slot in self._slots:
                        slot._side_hist.clear()
                        slot._last_dir.clear()
                        slot._crossing_ts.clear()
                        slot._polygon_inside.clear()
                        slot.tracker = self._tracker_cls(args=self._tracker_args)
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

                    # item selalu (frame, decode_ms, source_clip, local_frame) —
                    # 2 field terakhir None buat RTSP (lihat cam_slot.py).
                    frame, slot.last_decode_ms, slot.last_source_clip, slot.last_local_frame = frame

                    slot.last_frame = frame
                    batch_frames.append(frame)
                    has_new.append(True)

                if not any(has_new):
                    time.sleep(0.02)
                    continue

                # ── Batch YOLO detect — 1 GPU call, HANYA kamera yang punya
                # frame baru siklus ini. Kamera offline (permanen atau lagi
                # reconnect) numpang di batch_frames sebagai placeholder demi
                # alignment index dengan self._slots, tapi tidak pernah masuk
                # predict() — dulu ikut ke-infer padahal cuma frame hitam,
                # buang GPU/CPU cycle selama kamera itu mati.
                # Pakai predict() bukan track() karena batch track() berbagi 1
                # tracker untuk semua kamera (bug Ultralytics di non-stream mode).
                # Tiap kamera punya BYTETracker sendiri di slot.tracker.
                active_idx = [i for i, is_new in enumerate(has_new) if is_new]
                _batch_t0 = time.perf_counter()
                active_results = self._detector.predict(
                    [batch_frames[i] for i in active_idx],
                    conf=self._conf,
                    imgsz=DETECT_IMGSZ,
                    classes=[0],
                    verbose=False,
                )
                detection_ms = (time.perf_counter() - _batch_t0) * 1000
                self._batch_ms.append(detection_ms)
                results: list = [None] * len(self._slots)
                for i, r in zip(active_idx, active_results):
                    results[i] = r

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
                    # (skip kalau analytics dimatikan untuk kamera ini — capture &
                    # rekaman tetap jalan, cuma deteksi/tracking/event yang dilewati)
                    # Instrumentasi evaluasi (lihat latency_logger.py) — cuma
                    # nyatet waktu, gak ngubah urutan/hasil logika di bawah.
                    tracking_ms = 0.0
                    reid_ms     = 0.0
                    per_box: list[tuple[int, float, int, int, int, int, "np.ndarray | None", float]] = []
                    if slot.analytics_enabled and boxes is not None and slot.tracker is not None and n_raw > 0:
                        det = boxes.cpu().numpy()
                        _track_t0 = time.perf_counter()
                        tracks = slot.tracker.update(det, frame)
                        tracking_ms = (time.perf_counter() - _track_t0) * 1000
                        for t in tracks:
                            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                            track_id = int(t[4])
                            conf_val = float(t[5])
                            # Throttle: ekstrak embedding cuma 1x tiap REID_EVERY_N
                            # siklus per track, dan stop total kalau tracklet-nya
                            # sudah punya 16 sampel. observe() di bawah tetap jalan
                            # tiap siklus (emb None) buat n_det + titik lintasan +
                            # best_crop.
                            tkey = (slot.camera_id, track_id)
                            tick = self._reid_tick.get(tkey, 0)
                            self._reid_tick[tkey] = tick + 1
                            emb, quality = None, 0.0
                            if tick % REID_EVERY_N == 0 and not slot.db.samples_full(slot.camera_id, track_id):
                                _reid_t0 = time.perf_counter()
                                result   = _extract_embedding(self._extractor, frame, x1, y1, x2, y2,
                                                              track_id=track_id, cam_id=slot.camera_id)
                                reid_ms += (time.perf_counter() - _reid_t0) * 1000
                                if result is not None:
                                    emb, quality = result
                            per_box.append((track_id, conf_val, x1, y1, x2, y2, emb, quality))

                    active_tids = {tid for tid, *_ in per_box}
                    ts_now = datetime.now(timezone.utc)

                    # Kumpulkan bukti untuk tiap tracklet — keputusan identitas
                    # baru diambil saat tracklet DITUTUP (lihat pipeline_service.py).
                    # matching_ms nyaris 0 kecuali siklus ini nutup tracklet
                    # (cosine-similarity asosiasi cuma jalan di close_expired->
                    # _resolve_tracklet->associate(), bukan tiap frame — itu normal).
                    _match_t0 = time.perf_counter()
                    for track_id, conf_val, x1, y1, x2, y2, emb, quality in per_box:
                        slot.db.observe(slot.camera_id, track_id, emb, quality, conf_val,
                                        frame, x1, y1, x2, y2, ts_now)

                    slot.db.update_active(slot.camera_id, active_tids)
                    # Buang tick track yang sudah tidak aktif di kamera ini
                    # (track hilang → tracklet ditutup / track_id didaur ulang).
                    for k in [k for k in self._reid_tick
                              if k[0] == slot.camera_id and k[1] not in active_tids]:
                        del self._reid_tick[k]
                    closed = slot.db.close_expired(slot.camera_id, ts_now)
                    matching_ms = (time.perf_counter() - _match_t0) * 1000
                    if closed:
                        self._finalize_tracklets(closed)

                    # ── Prediction logging (mode file-playback saja) ──────────
                    # Baris di-buffer per (cam,track_id) — baru ditulis ke CSV saat
                    # tracklet-nya resolve (lihat _finalize_tracklets), supaya
                    # person_pred berisi identitas akhir, bukan placeholder track_id.
                    if slot.last_source_clip is not None:
                        for track_id, _, x1, y1, x2, y2, _, _ in per_box:
                            self._pred_logger.buffer(
                                (slot.camera_id, track_id),
                                slot.last_source_clip, slot.last_local_frame, slot.camera_id,
                                x1, y1, x2 - x1, y2 - y1,
                            )

                    # ── Zone check (line-crossing + polygon dwell) ────────────
                    for zone in slot.zones:
                        zc_id = zone["zone_camera_id"]

                        if zone["type"] == "line":
                            for track_id, _, x1, y1, x2, y2, _, _ in per_box:
                                fx, fy = (x1 + x2) // 2, y2  # foot point
                                for seg_i, seg in enumerate(zone["points"]):
                                    key  = (zc_id, seg_i, track_id)
                                    side = cross_side(
                                        fx, fy,
                                        seg["p1"]["x"], seg["p1"]["y"],
                                        seg["p2"]["x"], seg["p2"]["y"],
                                    )
                                    hist = slot._side_hist.setdefault(key, deque(maxlen=4))
                                    if abs(side) < 1:
                                        continue
                                    hist.append(side)
                                    if len(hist) >= 2 and hist[-2] * hist[-1] < 0:
                                        direction = "IN" if side * seg.get("in_sign", 1) > 0 else "OUT"
                                        # Hysteresis: arah sama berturut-turut diabaikan
                                        if slot._last_dir.get(key) == direction:
                                            continue
                                        # Cooldown: minimal CROSSING_COOLDOWN detik antar event per (garis, track)
                                        if now - slot._crossing_ts.get(key, 0.0) < CROSSING_COOLDOWN:
                                            continue
                                        slot._last_dir[key]    = direction
                                        slot._crossing_ts[key] = now
                                        snap_url     = self._save_event_snapshot(frame, x1, y1, x2, y2, slot.camera_id)
                                        person_label = slot.db.label_of(track_id, slot.camera_id)
                                        _BackendClient.post_occupancy_event(
                                            slot.camera_id, zc_id, direction, "room_entry",
                                            snap_url, person_label, track_id, fx, fy,
                                        )
                                        _BackendClient.post_camera_event(
                                            slot.camera_id, "zone_entry", "info",
                                            description=f"{direction} via {zone['name']}",
                                            snapshot_url=snap_url,
                                            person_label=person_label,
                                            timestamp=datetime.fromtimestamp(now, tz=timezone.utc),
                                            ai_latency_ms=self._last_ai_ms.get(slot.camera_id),
                                        )

                        elif zone["type"] == "polygon":
                            polygon_np = np.array(
                                [[p["x"], p["y"]] for p in zone["points"]], dtype=np.int32
                            )
                            for track_id, _, x1, y1, x2, y2, _, _ in per_box:
                                fx, fy = (x1 + x2) // 2, y2  # foot point
                                key = (zc_id, track_id)
                                inside = cv2.pointPolygonTest(polygon_np, (float(fx), float(fy)), False) >= 0
                                was_inside = slot._polygon_inside.get(key, False)
                                if inside == was_inside:
                                    continue
                                if now - slot._crossing_ts.get(key, 0.0) < CROSSING_COOLDOWN:
                                    continue
                                slot._polygon_inside[key] = inside
                                slot._crossing_ts[key]    = now
                                direction    = "IN" if inside else "OUT"
                                snap_url     = self._save_event_snapshot(frame, x1, y1, x2, y2, slot.camera_id)
                                person_label = slot.db.label_of(track_id, slot.camera_id)
                                _BackendClient.post_occupancy_event(
                                    slot.camera_id, zc_id, direction, "room_entry",
                                    snap_url, person_label, track_id, fx, fy,
                                )
                                _BackendClient.post_camera_event(
                                    slot.camera_id, "zone_entry", "info",
                                    description=f"{direction} via {zone['name']}",
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

                    self._last_ai_ms[slot.camera_id] = (
                        slot.last_decode_ms + detection_ms + tracking_ms + reid_ms + matching_ms
                    )
                    if self._latency_enabled:
                        self._latency_logger.log(
                            timestamp=datetime.now(_WIB).isoformat(),
                            camera_id=slot.camera_id,
                            frame_id=idx,
                            decode_ms=slot.last_decode_ms,
                            detection_ms=detection_ms,
                            tracking_ms=tracking_ms,
                            reid_ms=reid_ms,
                            matching_ms=matching_ms,
                        )

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
            # Flush semua tracklet terbuka sebelum berhenti — kalau tidak,
            # observasi yang sudah terkumpul hilang begitu saja tanpa POST.
            if self._slots:
                closed = self._slots[0].db.close_all()
                if closed:
                    self._finalize_tracklets(closed)
            for slot in self._slots:
                slot.shutdown()
            self._pred_logger.close()
            self._latency_logger.close()

    def get_metrics(self) -> dict:
        """System Health (plan2/spesifikasi.md Fase 2) — waktu predict() batch
        terakhir & rata-rata, buat diagnosa kapan BatchProcessor mulai
        keteteran sebelum FPS beneran drop."""
        ms = list(self._batch_ms)
        return {
            "last_batch_ms":    ms[-1] if ms else None,
            "avg_batch_ms":     sum(ms) / len(ms) if ms else None,
            "cameras_active":   sum(1 for s in self._slots if s.online),
            "cameras_total":    len(self._slots),
        }

    def _finalize_tracklets(self, closed: list[dict]) -> None:
        """Tracklet baru saja ditutup (lihat pipeline_service.py._resolve_tracklet).
        Simpan thumbnail sekali, lalu POST /detections + /tracklets (+ /camera-events
        kalau identitas baru) — dilewati kalau _skip_backend_persist=True (evaluasi
        terisolasi), supaya klip evaluasi tidak menulis baris baru ke persons/
        detections/tracklets di database live."""
        for result in closed:
            cam   = result["cam_id"]
            label = result["label"]
            det_url = None
            if not self._skip_backend_persist:
                if result["best_crop"] is not None and result["best_crop"].size > 0:
                    det_url = self._save_crop(result["best_crop"], cam, result["track_id"])

                _BackendClient.post_detection(
                    result["display_name"], cam, result["best_conf"], "appearance",
                    det_url, label, result["track_id"],
                    timestamp=result["ended_at"],
                )
                _BackendClient.post_tracklet(
                    cam, result["track_id"], label,
                    result["started_at"], result["ended_at"], result["n_detections"],
                    det_url, result["embedding"], result["assoc_score"],
                    par=result["par"], best_crop=result["best_crop"],
                    pos_x=result["pos_x"], pos_y=result["pos_y"],
                    positions=result["positions"],
                )
                if result["is_new"]:
                    _BackendClient.post_camera_event(
                        cam, "person_detected", "info",
                        person_label=label, snapshot_url=det_url,
                        timestamp=result["ended_at"],
                        ai_latency_ms=self._last_ai_ms.get(cam),
                    )
            # Tulis baris predictions.csv yang di-buffer selama tracklet ini
            # terbuka, sekarang dengan nama akhir yang sudah resolve (§6).
            # Tetap jalan walau _skip_backend_persist=True — inilah yang
            # dibutuhkan evaluasi akurasi.
            self._pred_logger.flush((cam, result["track_id"]), result["display_name"])

    def _on_reconnect(self, slot: _CamSlot, success: bool) -> None:
        if not success:
            print(f"[{slot.camera_id}] offline permanen")
            if slot.camera_id not in self._offline_reported:
                self._offline_reported.add(slot.camera_id)
                _BackendClient.post_camera_event(
                    slot.camera_id, "camera_offline", "critical",
                    description=f"Kamera {slot.camera_id} tidak merespons setelah 5 percobaan",
                )
        else:
            # Reconnect tidak me-rebuild BYTETracker — orang yang terekam sebelum
            # putus koneksi kemungkinan besar sudah pergi. Tutup tracklet terbuka
            # milik kamera ini (plan/07-fase2-detail.md §3).
            closed = slot.db.close_all(cam_id=slot.camera_id)
            if closed:
                self._finalize_tracklets(closed)
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
    def _save_crop(crop: np.ndarray, camera_id: str, track_id: int) -> "str | None":
        """Simpan crop yang sudah dipadding (tl.best_crop, lihat _padded_crop di
        pipeline_service.py) — satu-satunya thumbnail per tracklet. Menggantikan
        _save_unique_snapshot + _save_thumbnail/profile-thumbnail terpisah yang
        ada sebelum Fase 2 (lihat plan/07-fase2-detail.md §6)."""
        try:
            THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_t{track_id}_{ts}.jpg"
            cv2.imwrite(str(THUMBNAILS_DIR / filename), crop)
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")
            return f"{backend_url}/thumbnails/{filename}"
        except Exception as exc:
            print(f"[{camera_id}] snapshot error: {exc}")
            return None
