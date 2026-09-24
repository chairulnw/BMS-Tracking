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
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from app.services.geometry import cross_side, within_segment_span
from app.services.pipeline_service import _extract_embedding
from app.services.stream_service.backend_client import _BackendClient, _fetch_zones
from app.services.stream_service.cam_slot import _CamSlot

CROSSING_COOLDOWN = 3.0   # detik minimum antar event crossing per (garis/polygon, track)
CROSSING_ORPHAN_AGE = float(os.getenv("CROSSING_ORPHAN_AGE", "20"))  # flush crossing yg track-nya tak pernah resolve

# _WIB: cuma buat timestamp latency.csv; event ke backend/DB tetap timezone.utc
_WIB = ZoneInfo("Asia/Jakarta")
THUMBNAILS_DIR    = Path("thumbnails")
PREDICTIONS_CSV   = Path("predictions.csv")  # log prediksi mode file-playback, utk dibanding ground truth

DETECT_IMGSZ = int(os.getenv("DETECT_IMGSZ", "640"))
DARK_MEAN = float(os.getenv("DARK_MEAN", "40"))         # 0 = matikan gate; frame < brightness ini dilewati
OCCLUSION_FRAC = float(os.getenv("OCCLUSION_FRAC", "0.45"))  # box overlap > ini → skip embedding (crop campur)
REC_COAST_FRAMES = int(os.getenv("REC_COAST_FRAMES", "20"))  # tetap rekam N frame setelah track hilang (ByteTrack coast)


def _overlap_frac(a: tuple, b: tuple) -> float:
    """Luas irisan a∩b dibagi luas box terkecil, bukan IoU — biar box kecil di dalam box besar tetap kedeteksi."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    m = min(area_a, area_b)
    return inter / m if m > 0 else 0.0


@dataclass
class _ModelBundle:
    """Model + config sekali-load, dipakai ulang tiap restart stream via StreamManager."""
    detector:     YOLO
    extractor:    "TransReIDExtractor"
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
    # ReID dipanggil per box tiap siklus (jauh lebih sering dari YOLO) — cek MPS di sini juga, atau Mac jatuh ke CPU dan jadi bottleneck.
    if torch.backends.mps.is_available():
        reid_device = "mps"
    elif torch.cuda.is_available():
        reid_device = "cuda"
    else:
        reid_device = "cpu"

    print(f"[batch] loading YOLO({detector_model}) → {detector_device}, "
          f"ReID({reid_model}) → {reid_device}")
    detector = YOLO(detector_model)
    detector.to(detector_device)
    from app.services.stream_service.transreid_extractor import TransReIDExtractor
    extractor = TransReIDExtractor(device=reid_device)
    tracker_cls, tracker_yaml = BYTETracker, "bytetrack.yaml"
    print("[batch] tracker: bytetrack")
    tracker_cfg  = YAML.load(check_yaml(tracker_yaml))
    # Default ByteTrack config: tracker longgar bikin track_id melayang nyebrang orang saat occlusion, jadi tetap pakai default ketat.
    tracker_args = IterableSimpleNamespace(**tracker_cfg)

    # PAR jalan sekali per tracklet di crop terbaik, bukan per frame (~2s/crop di CPU)
    par = None
    rap1_checkpoint = Path("checkpoints/par_checkpoints/RAP1.pth")
    if rap1_checkpoint.exists():
        try:
            from app.par.par_service import PARExtractor
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
        self._batch_ms: "deque[float]" = deque(maxlen=50)  # waktu predict() batch terakhir, buat diagnosa bottleneck
        self._frame_idx: dict[str, int] = {}   # diisi _loop(), dibaca resource_sampler buat fps_effective
        self._last_ai_ms: dict[str, float] = {}  # latency frame TERAKHIR per kamera, approx kasar buat camera_events
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

        # Crossing di-buffer per (cam, track_id) sampai tracklet-nya resolve, baru di-POST dengan person_label FINAL.
        self._pending_crossings: dict[tuple[str, int], list[dict]] = {}

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
                    # Flush tracklet & crossing terbuka SEBELUM rebuild tracker di
                    # bawah mendaur ulang track_id dari 1 — kalau tidak, track_id
                    # baru bisa tabrakan dengan tracklet lama yang belum di-flush.
                    closed = self._slots[0].db.close_all()
                    if closed:
                        self._finalize_tracklets(closed)
                    self._sweep_orphan_crossings(time.time() + CROSSING_ORPHAN_AGE)
                    self._slots[0].db.prune_banks()
                    self._slots[0].db.reset()
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
                        batch_frames.append(placeholder)
                        has_new.append(False)
                        continue

                    if frame is None:
                        if slot._playlist:
                            print(f"[{slot.camera_id}] semua clip selesai")
                            slot.online = False
                            self._file_slots_done.add(slot.camera_id)
                            if self._file_slots_done >= self._file_slots_total and self._auto_stop_cb:
                                threading.Thread(
                                    target=self._auto_stop_cb, daemon=True, name="auto-stop"
                                ).start()
                        else:
                            print(f"[{slot.camera_id}] disconnected")
                            slot.start_reconnect(self._on_reconnect)
                        batch_frames.append(placeholder)
                        has_new.append(False)
                        continue

                    # 2 field terakhir None buat RTSP (lihat cam_slot.py)
                    frame, slot.last_decode_ms, slot.last_source_clip, slot.last_local_frame = frame

                    slot.last_frame = frame
                    if DARK_MEAN > 0 and float(frame.mean()) < DARK_MEAN:
                        batch_frames.append(frame)
                        has_new.append(False)
                        continue
                    batch_frames.append(frame)
                    has_new.append(True)

                if not any(has_new):
                    time.sleep(0.02)
                    continue

                # 1 GPU call untuk kamera dengan frame baru saja. predict() bukan
                # track(): batch track() berbagi 1 tracker lintas kamera (bug
                # Ultralytics non-stream) — tiap kamera pakai slot.tracker sendiri.
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

                for slot, result, is_new in zip(self._slots, results, has_new):
                    if not is_new:
                        continue

                    frame = slot.last_frame
                    boxes = result.boxes

                    n_raw = 0
                    if boxes is not None:
                        n_raw = sum(1 for b in boxes if int(b.cls[0]) == 0)

                    tracking_ms = 0.0
                    reid_ms     = 0.0
                    per_box: list[tuple[int, float, int, int, int, int, "np.ndarray | None", float]] = []
                    if slot.analytics_enabled and boxes is not None and slot.tracker is not None and n_raw > 0:
                        det = boxes.cpu().numpy()
                        _track_t0 = time.perf_counter()
                        tracks = slot.tracker.update(det, frame)
                        tracking_ms = (time.perf_counter() - _track_t0) * 1000
                        tboxes = [(int(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in tracks]
                        for i, t in enumerate(tracks):
                            x1, y1, x2, y2 = tboxes[i]
                            track_id = int(t[4])
                            conf_val = float(t[5])
                            occluded = any(_overlap_frac(tboxes[i], tboxes[j]) > OCCLUSION_FRAC
                                           for j in range(len(tboxes)) if j != i)
                            emb, quality = None, 0.0
                            if not occluded:
                                _reid_t0 = time.perf_counter()
                                result   = _extract_embedding(self._extractor, frame, x1, y1, x2, y2,
                                                              track_id=track_id, cam_id=slot.camera_id)
                                reid_ms += (time.perf_counter() - _reid_t0) * 1000
                                if result is not None:
                                    emb, quality = result
                            per_box.append((track_id, conf_val, x1, y1, x2, y2, emb, quality))

                    active_tids = {tid for tid, *_ in per_box}
                    ts_now = datetime.now(timezone.utc)
                    # idx = siklus lokal kamera untuk gap-detection tracklet (kebal lag); ts_now cuma MAX_DURATION safety-net & timestamp.
                    idx = frame_idx[slot.camera_id] + 1
                    frame_idx[slot.camera_id] = idx

                    _match_t0 = time.perf_counter()
                    for track_id, conf_val, x1, y1, x2, y2, emb, quality in per_box:
                        slot.db.observe(slot.camera_id, track_id, emb, quality, conf_val,
                                        frame, x1, y1, x2, y2, ts_now, cycle=idx)

                    slot.db.update_active(slot.camera_id, active_tids)
                    closed = slot.db.close_expired(slot.camera_id, ts_now, cycle=idx)
                    matching_ms = (time.perf_counter() - _match_t0) * 1000
                    if closed:
                        self._finalize_tracklets(closed)

                    # Mode file-playback: buffer per (cam,track_id), flush di _finalize_tracklets supaya person_pred pakai identitas akhir.
                    if slot.last_source_clip is not None:
                        for track_id, _, x1, y1, x2, y2, _, _ in per_box:
                            self._pred_logger.buffer(
                                (slot.camera_id, track_id),
                                slot.last_source_clip, slot.last_local_frame, slot.camera_id,
                                x1, y1, x2 - x1, y2 - y1,
                            )

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
                                        # cross_side = garis tak-hingga; pastikan lewat di RUAS-nya
                                        if not within_segment_span(
                                            fx, fy,
                                            seg["p1"]["x"], seg["p1"]["y"],
                                            seg["p2"]["x"], seg["p2"]["y"],
                                        ):
                                            continue
                                        direction = "IN" if side * seg.get("in_sign", 1) > 0 else "OUT"
                                        if slot._last_dir.get(key) == direction:
                                            continue
                                        if now - slot._crossing_ts.get(key, 0.0) < CROSSING_COOLDOWN:
                                            continue
                                        slot._last_dir[key]    = direction
                                        slot._crossing_ts[key] = now
                                        snap_url = self._save_event_snapshot(frame, x1, y1, x2, y2, slot.camera_id)
                                        self._buffer_crossing(
                                            slot.camera_id, track_id, zc_id, direction,
                                            zone["name"], snap_url, fx, fy, now,
                                            slot.db.label_of(track_id, slot.camera_id),
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
                                direction = "IN" if inside else "OUT"
                                snap_url  = self._save_event_snapshot(frame, x1, y1, x2, y2, slot.camera_id)
                                self._buffer_crossing(
                                    slot.camera_id, track_id, zc_id, direction,
                                    zone["name"], snap_url, fx, fy, now,
                                    slot.db.label_of(track_id, slot.camera_id),
                                )

                    # Track baru hilang < REC_COAST_FRAMES lalu = YOLO kedip, orang masih ada — rekam tanpa syarat gerak.
                    _tr = slot.tracker
                    recent_lost = any(
                        getattr(_tr, "frame_id", 0) - getattr(t, "end_frame", 0) <= REC_COAST_FRAMES
                        for t in getattr(_tr, "lost_stracks", ())
                    )
                    rec_has_person = n_raw > 0 or recent_lost
                    rec_annots = [
                        (x1, y1, x2, y2,
                         f"{slot.db.name_of(tid, slot.camera_id) or f'#{tid}'} {conf:.2f}")
                        for tid, conf, x1, y1, x2, y2, _, _ in per_box
                    ]
                    # frame+box disuplai langsung dari sini, bukan thread capture independen, supaya box selalu cocok framenya.
                    if slot.recorder is not None:   # None kalau skip_recording (evaluasi)
                        if slot.rec_q.full():
                            try:
                                slot.rec_q.get_nowait()
                            except queue.Empty:
                                pass
                        try:
                            slot.rec_q.put_nowait((frame, rec_has_person, rec_annots))
                        except queue.Full:
                            pass

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

                self._sweep_orphan_crossings(now)

        finally:
            if self._slots:
                closed = self._slots[0].db.close_all()
                if closed:
                    self._finalize_tracklets(closed)
            self._sweep_orphan_crossings(time.time() + CROSSING_ORPHAN_AGE)
            for slot in self._slots:
                slot.shutdown()
            self._pred_logger.close()
            self._latency_logger.close()

    def get_metrics(self) -> dict:
        """Waktu predict() batch terakhir & rata-rata, buat diagnosa bottleneck."""
        ms = list(self._batch_ms)
        return {
            "last_batch_ms":    ms[-1] if ms else None,
            "avg_batch_ms":     sum(ms) / len(ms) if ms else None,
            "cameras_active":   sum(1 for s in self._slots if s.online),
            "cameras_total":    len(self._slots),
        }

    def _buffer_crossing(self, cam_id: str, track_id: int, zc_id: int, direction: str,
                         zone_name: str, snap_url: "str | None", fx: int, fy: int,
                         ts: float, prov_label: "str | None") -> None:
        pc = {"zc_id": zc_id, "direction": direction, "zone_name": zone_name,
              "snap_url": snap_url, "fx": fx, "fy": fy, "ts": ts}
        # OUT diposting langsung; IN dibuffer supaya dapat identitas final saat tracklet resolve.
        if direction != "IN":
            self._post_crossing(cam_id, track_id, pc, prov_label or f"Unknown@{cam_id}")
            return
        self._pending_crossings.setdefault((cam_id, track_id), []).append(pc)

    def _flush_crossings(self, cam_id: str, track_ids, person_label: str) -> None:
        """POST semua crossing yang di-buffer buat track-track ini, pakai label final."""
        for tid in track_ids:
            for pc in self._pending_crossings.pop((cam_id, tid), []):
                self._post_crossing(cam_id, tid, pc, person_label)

    def _sweep_orphan_crossings(self, now: float) -> None:
        """Track yang hilang tanpa pernah nutup tracklet — flush crossing-nya pakai label terbaik yang ada, biar occupancy count tidak meleset."""
        for (cam_id, tid), pcs in list(self._pending_crossings.items()):
            if not pcs or now - pcs[0]["ts"] < CROSSING_ORPHAN_AGE:
                continue
            label = self._slots[0].db.label_of(tid, cam_id) if self._slots else None
            for pc in pcs:
                self._post_crossing(cam_id, tid, pc, label or f"Unknown@{cam_id}")
            del self._pending_crossings[(cam_id, tid)]

    @staticmethod
    def _post_crossing(cam_id: str, track_id: int, pc: dict, person_label: str) -> None:
        _BackendClient.post_occupancy_event(
            cam_id, pc["zc_id"], pc["direction"], "room_entry",
            pc["snap_url"], person_label, track_id, pc["fx"], pc["fy"],
        )
        _BackendClient.post_camera_event(
            cam_id, "zone_entry", "info",
            description=f"{pc['direction']} via {pc['zone_name']}",
            snapshot_url=pc["snap_url"], person_label=person_label,
            timestamp=datetime.fromtimestamp(pc["ts"], tz=timezone.utc),
        )

    def _finalize_tracklets(self, closed: list[dict]) -> None:
        """POST /detections + /tracklets (+ /camera-events kalau identitas baru); dilewati kalau _skip_backend_persist=True."""
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
            self._pred_logger.flush((cam, result["track_id"]), result["display_name"])
            for frag_tid in result.get("folded_track_ids", ()):
                self._pred_logger.flush((cam, frag_tid), result["display_name"])

            self._flush_crossings(
                cam, [result["track_id"], *result.get("folded_track_ids", ())], label,
            )

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
            # Reconnect tidak rebuild tracker — tutup tracklet terbuka kamera ini, orang sebelum putus kemungkinan sudah pergi.
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
        """Simpan crop orang saat crossing terjadi ke thumbnails/events/, diperbesar ke minimal 120x240 px."""
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
            # BACKEND_URL is ai-service's internal address (e.g. Docker service name); PUBLIC_BACKEND_URL overrides it for browser-facing URLs (e.g. "/api" via nginx).
            backend_url = os.getenv("PUBLIC_BACKEND_URL") or os.getenv("BACKEND_URL", "http://localhost:8002")
            return f"{backend_url}/thumbnails/events/{filename}"
        except Exception as exc:
            print(f"[{camera_id}] event snapshot error: {exc}")
            return None

    @staticmethod
    def _save_crop(crop: np.ndarray, camera_id: str, track_id: int) -> "str | None":
        """Simpan crop yang sudah dipadding (tl.best_crop, lihat _padded_crop di pipeline_service.py) — satu-satunya thumbnail per tracklet."""
        try:
            THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{camera_id}_t{track_id}_{ts}.jpg"
            cv2.imwrite(str(THUMBNAILS_DIR / filename), crop)
            # BACKEND_URL is ai-service's internal address (e.g. Docker service name); PUBLIC_BACKEND_URL overrides it for browser-facing URLs (e.g. "/api" via nginx).
            backend_url = os.getenv("PUBLIC_BACKEND_URL") or os.getenv("BACKEND_URL", "http://localhost:8002")
            return f"{backend_url}/thumbnails/{filename}"
        except Exception as exc:
            print(f"[{camera_id}] snapshot error: {exc}")
            return None
