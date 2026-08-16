"""
Pipeline service — callable version tanpa GUI.
Line coordinates diterima sebagai parameter, bukan dari klik mouse.
"""

import math
import os
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from app.schemas import IdentityRecord, ProcessVideoRequest, ProcessVideoResponse

EMBED_REFRESH      = 15
BUFFER_FRAMES      = 10
NEAR_LINE_DIST     = 40
MIN_CROP_PX        = 32
MIN_ENROLL_FRAMES  = 2      # delayed enrollment: tunggu N frame berkualitas
MIN_MARGIN         = 0.05   # gap minimum top1-top2 untuk confident match
MAX_BANK_SIZE      = 5      # maks entry per identitas di bank embedding
BANK_MERGE_SIM     = 0.90   # sim >= ini → update entry lama, bukan tambah baru
BANK_STALE_HOURS   = 6.0    # entry yang tidak jadi top-match selama N jam → kandidat pruning
QUALITY_MIN_H      = 80     # tinggi crop minimum (px)
QUALITY_MIN_W      = 40     # lebar crop minimum (px)
QUALITY_MIN_RATIO  = 1.8    # min h/w — person portrait aspect ratio
QUALITY_LAP_VAR    = 100.0  # min Laplacian variance — blur gate

DEBUG_CROPS_DIR = Path("debug_crops")

def _debug_reid() -> bool:
    return os.getenv("DEBUG_REID", "").lower() in ("1", "true")


# ── Identity database ─────────────────────────────────────────────────────────

PAR_AMBIG_LOW   = 0.54   # cosine sim zone where PAR veto is consulted
PAR_AMBIG_HIGH  = 0.63  # above this OSNet is trusted directly
PAR_ATTR_THR    = 0.6  # attr_match below this → PAR veto → NEW


class IdentityDB:
    def __init__(self, reid_threshold: float, camera_id: str = "",
                 par_extractor=None) -> None:
        self.threshold           = reid_threshold
        self._camera_id          = camera_id
        # Bank: name → list of {"emb": np.ndarray, "last_match": datetime, "cam_id": str}
        self._embeddings:      dict[str, list] = {}
        self._attr_gallery:    dict[str, "np.ndarray | None"] = {}  # PAR (disabled)
        self._track_to_name:   dict[int, str]        = {}
        self._frame_counter:   dict[int, int]        = {}
        # Maps display_name → globally-unique label ("Unknown #1@c8@20260622")
        # camera_id disertakan agar label tidak bentrok antar kamera di DB.
        self._display_to_label: dict[str, str]       = {}
        # Pending buffer: key (cam_id, track_id) → [(emb, laplacian_var, crop), ...]
        # Track yang belum mencapai MIN_ENROLL_FRAMES frame berkualitas disimpan di sini.
        self._pending:         dict                  = {}
        self._count = 0
        self._par = par_extractor  # PARExtractor | None
        self._color_gallery: dict[str, dict] = {}   # {"upper": str, "lower": str | None}
        self._name_to_owner:  dict[str, tuple]      = {}
        self._ambiguous:      dict[str, dict]       = {}  # amb_id → entry (belum dikonfirmasi)
        self._in_review:      set                   = set()  # key track yang sedang menunggu resolusi
        self._active_tracks:  dict[str, set[int]]   = {}  # cam_id → set track_id aktif saat ini
        self._last_top2_name: str | None            = None  # nama kandidat ke-2 (untuk logging)

    def _key(self, track_id: int, cam_id: str) -> tuple[str, int] | int:
        return (cam_id, track_id) if cam_id else track_id

    def _new_names(self, cam_id: str = "") -> tuple[str, str]:
        """Returns (display_name, unique_label).
        Label menyertakan camera_id dan tanggal agar unik di seluruh kamera."""
        self._count += 1
        today   = datetime.now().strftime("%Y%m%d")
        display = f"Unknown #{self._count}"
        cam     = cam_id or self._camera_id or "cam"
        label   = f"Unknown #{self._count}@{cam}@{today}"
        return display, label

    def _best_match(
        self, emb: np.ndarray, *, track_id=None, cam_id: str = "",
    ) -> tuple[str | None, float, str | None, float]:
        """Returns (name_top1, sim_top1, verdict, sim_top2). verdict always None (pure OSNet).
        Similarity per identitas = MAX similarity terhadap seluruh entry di banknya."""
        top1_name, top1_sim = None, -1.0
        top2_name, top2_sim = None, -1.0
        for name, bank in self._embeddings.items():
            best_e = max((float(np.dot(emb, e["emb"])) for e in bank), default=-1.0)
            if _debug_reid():
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[reid.cmp] {ts} cam={cam_id} t{track_id} ↔ {name!r}  sim={best_e:.4f} (bank={len(bank)})")
            if best_e > top1_sim:
                top2_sim, top2_name = top1_sim, top1_name
                top1_sim, top1_name = best_e, name
            elif best_e > top2_sim:
                top2_sim, top2_name = best_e, name
        self._last_top2_name = top2_name   # simpan untuk logging di assign()
        return top1_name, top1_sim, None, top2_sim

    def assign(
        self, track_id: int, emb: np.ndarray, cam_id: str = "",
        *, quality: float = 1.0, debug_crop: "np.ndarray | None" = None,
        active_track_ids: "set[int] | None" = None,
    ) -> tuple[str | None, bool]:
        """Assign identity; returns (display_name, is_new_identity).
        Returns (None, False) jika track masih dalam delayed enrollment buffer."""
        key = self._key(track_id, cam_id)
        if key in self._track_to_name:
            return self._track_to_name[key], False
        if key in self._in_review:
            return None, False   # sedang menunggu resolusi operator

        # ── Delayed enrollment buffer ──────────────────────────────────────────
        buf = self._pending.setdefault(key, [])
        buf.append((emb, quality))
        if len(buf) < MIN_ENROLL_FRAMES:
            print(f"[reid] {cam_id}/t{track_id} buffering ({len(buf)}/{MIN_ENROLL_FRAMES})")
            return None, False

        # Cukup frame — ambil embedding dengan Laplacian variance tertinggi
        best_entry = max(buf, key=lambda x: x[1])
        best_emb   = best_entry[0]
        del self._pending[key]

        # ── Gallery matching dengan best embedding ─────────────────────────────
        n_gallery = len(self._embeddings)
        name, sim, _, sim2 = self._best_match(best_emb, track_id=track_id, cam_id=cam_id)
        top2_name = self._last_top2_name
        margin    = sim - sim2
        ts        = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ts_fn     = datetime.now().strftime("%H%M%S_%f")

        # Header baris log: waktu + lokasi + ringkasan gallery
        if n_gallery == 0:
            _hdr = f"[reid] {ts} {cam_id}/t{track_id} gallery=∅"
        elif name is not None:
            top2_str = f" 2nd={top2_name!r}@{sim2:.3f}" if top2_name else ""
            _hdr = f"[reid] {ts} {cam_id}/t{track_id} gallery={n_gallery} top1={name!r}@{sim:.3f}{top2_str} margin={margin:.3f}"
        else:
            _hdr = f"[reid] {ts} {cam_id}/t{track_id} gallery={n_gallery}"

        # Collision guard: cek intra-kamera dan lintas kamera.
        # Identitas tidak boleh diklaim track baru selama track pemiliknya masih aktif,
        # baik di kamera yang sama maupun kamera lain.
        if name is not None and sim >= self.threshold:
            owner = self._name_to_owner.get(name)
            if owner is not None and owner != key:
                owner_cam, owner_tid = owner
                if owner_cam == cam_id:
                    owner_active = active_track_ids or set()
                else:
                    owner_active = self._active_tracks.get(owner_cam, set())
                if owner_tid in owner_active:
                    scope = "intra-cam" if owner_cam == cam_id else f"cross-cam({owner_cam})"
                    print(f"{_hdr} → CONFLICT {scope} (owner t{owner_tid}) → NEW")
                    name = None

        # ── 3-way decision ─────────────────────────────────────────────────────
        is_confident = name is not None and sim >= self.threshold and margin >= MIN_MARGIN
        is_ambiguous = name is not None and sim >= self.threshold and margin < MIN_MARGIN

        if is_confident:
            print(f"{_hdr} → MATCH ✓")
            if _debug_reid() and debug_crop is not None:
                safe = name.replace(" ", "_").replace("#", "")
                fname = f"{cam_id}_t{track_id}_{ts_fn}_MATCH_{safe}.jpg"
                DEBUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(DEBUG_CROPS_DIR / fname), debug_crop)
            self._track_to_name[key]  = name
            self._frame_counter[key]  = 0
            self._name_to_owner[name] = key
            self._bank_update(name, best_emb, cam_id, touch_only=True)
            return name, False

        if is_ambiguous:
            print(f"{_hdr} → AMBIGUOUS⚠ (margin tipis, auto-match ke top1 {name!r})")
            self._track_to_name[key]  = name
            self._frame_counter[key]  = 0
            self._name_to_owner[name] = key
            self._bank_update(name, best_emb, cam_id, touch_only=True)
            return name, False

        # NEW
        display, label = self._new_names(cam_id)
        self._embeddings[display]       = [{"emb": best_emb, "last_match": datetime.now(), "cam_id": cam_id}]
        self._display_to_label[display] = label
        reason = f"sim={sim:.3f} < thr={self.threshold}" if name else "gallery kosong"
        print(f"{_hdr} → NEW {display!r}  ({reason})")
        if _debug_reid() and debug_crop is not None:
            safe  = display.replace(" ", "_").replace("#", "")
            fname = f"{cam_id}_t{track_id}_{ts_fn}_NEW_{safe}.jpg"
            DEBUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(DEBUG_CROPS_DIR / fname), debug_crop)
        self._track_to_name[key]     = display
        self._frame_counter[key]     = 0
        self._name_to_owner[display] = key
        return display, True

    def get_ambiguous_list(self) -> list[dict]:
        return [
            {k: v for k, v in e.items() if k != "emb"}
            for e in self._ambiguous.values()
        ]

    def resolve_ambiguous(self, amb_id: str, action: str, *, target_label: str = "") -> bool:
        """action: 'confirm' → merge ke kandidat; 'assign' → merge ke target_label;
        'reject' → identitas baru."""
        entry = self._ambiguous.pop(amb_id, None)
        if entry is None:
            return False
        emb = entry["emb"]
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # Hapus track dari antrian review agar bisa assign ulang
        self._in_review.discard(self._key(entry["track_id"], entry["cam_id"]))

        if action in ("confirm", "assign"):
            if action == "confirm":
                name = entry["candidate"]
            else:
                # Cari display_name berdasarkan label
                name = next(
                    (dn for dn, lbl in self._display_to_label.items() if lbl == target_label),
                    None,
                )
                if name is None:
                    print(f"[reid] {ts} resolve {amb_id} → ASSIGN FAILED: label {target_label!r} tidak ditemukan")
                    return False
            if name in self._embeddings:
                self._bank_update(name, emb, entry["cam_id"])
            else:
                self._embeddings[name] = [{"emb": emb, "last_match": datetime.now(), "cam_id": entry["cam_id"]}]
            print(f"[reid] {ts} resolve {amb_id} → {'CONFIRM' if action == 'confirm' else 'ASSIGN'} as {name!r}")
        else:  # reject → new
            display, label = self._new_names(entry["cam_id"])
            self._embeddings[display]       = [{"emb": emb, "last_match": datetime.now(), "cam_id": entry["cam_id"]}]
            self._display_to_label[display] = label
            print(f"[reid] {ts} resolve {amb_id} → NEW {display!r}")
        return True

    def _bank_update(self, name: str, emb: np.ndarray, cam_id: str, *, touch_only: bool = False) -> None:
        """Perbarui bank embedding untuk `name` dengan embedding baru.
        touch_only=True → hanya update last_match tanpa mengubah vektor (dipakai saat MATCH pertama kali)."""
        bank = self._embeddings.get(name)
        if not bank:
            return
        # Cari entry paling mirip
        sims     = [float(np.dot(emb, e["emb"])) for e in bank]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        now = datetime.now()

        if touch_only or best_sim >= BANK_MERGE_SIM:
            # Update entry yang sudah ada
            if not touch_only:
                merged = 0.9 * bank[best_idx]["emb"] + 0.1 * emb
                bank[best_idx]["emb"] = merged / (np.linalg.norm(merged) + 1e-8)
            bank[best_idx]["last_match"] = now
        else:
            # Sudut pandang baru — tambah entry
            new_entry = {"emb": emb, "last_match": now, "cam_id": cam_id}
            if len(bank) >= MAX_BANK_SIZE:
                # LRU eviction: buang entry paling lama tidak jadi top-match
                lru_idx = min(range(len(bank)), key=lambda i: bank[i]["last_match"])
                bank[lru_idx] = new_entry
            else:
                bank.append(new_entry)
            ts = datetime.now().strftime("%H:%M:%S")
            evict = " (evict LRU)" if len(bank) >= MAX_BANK_SIZE else ""
            print(f"[reid.bank] {ts} {name!r} +angle cam={cam_id} nearest={best_sim:.3f} bank={len(bank)}{evict}")

    def label_of(self, track_id: int, cam_id: str = "") -> str | None:
        """Returns the date-unique label for a track (for DB storage)."""
        key     = self._key(track_id, cam_id)
        display = self._track_to_name.get(key)
        if display is None:
            return None
        return self._display_to_label.get(display, display)

    def refresh(self, track_id: int, emb: np.ndarray, cam_id: str = "") -> None:
        key = self._key(track_id, cam_id)
        cnt = self._frame_counter.get(key, 0) + 1
        self._frame_counter[key] = cnt
        if cnt % EMBED_REFRESH != 0:
            return
        name = self._track_to_name.get(key)
        if name is None or name not in self._embeddings:
            return
        # Hanya update bank untuk track yang sudah CONFIDENT match (sudah punya nama)
        self._bank_update(name, emb, cam_id)

    def update_active(self, cam_id: str, track_ids: "set[int]") -> None:
        """Dipanggil tiap siklus batch untuk update track yang aktif per kamera."""
        self._active_tracks[cam_id] = set(track_ids)

    def name_of(self, track_id: int, cam_id: str = "") -> str | None:
        return self._track_to_name.get(self._key(track_id, cam_id))

    def prune_banks(self) -> int:
        """Hapus entry stale dari bank embedding (jalankan sebelum reset harian).
        Returns jumlah entry yang dihapus."""
        removed = 0
        now     = datetime.now()
        for name, bank in self._embeddings.items():
            if len(bank) <= 1:
                continue
            fresh = [e for e in bank
                     if (now - e["last_match"]).total_seconds() / 3600 < BANK_STALE_HOURS]
            if not fresh:
                fresh = [max(bank, key=lambda e: e["last_match"])]
            removed += len(bank) - len(fresh)
            self._embeddings[name] = fresh
        if removed:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[reid.prune] {ts} pruned {removed} stale bank entries")
        return removed

    def reset(self) -> None:
        """Reset semua state setiap tengah malam. _count direset ke 0 karena
        _new_names() menyertakan tanggal di label sehingga tidak ada tabrakan
        label lintas hari di tabel persons PostgreSQL."""
        self._embeddings.clear()
        self._attr_gallery.clear()
        self._color_gallery.clear()
        self._track_to_name.clear()
        self._frame_counter.clear()
        self._display_to_label.clear()
        self._pending.clear()
        self._name_to_owner.clear()
        self._ambiguous.clear()
        self._in_review.clear()
        self._active_tracks.clear()
        self._count = 0

    def to_records(self) -> list[IdentityRecord]:
        records = []
        for name in self._embeddings:
            tids = [
                k[1] if isinstance(k, tuple) else k
                for k, n in self._track_to_name.items()
                if n == name
            ]
            records.append(IdentityRecord(name=name, track_ids=tids))
        return records


# ── Geometry ──────────────────────────────────────────────────────────────────

def _point_side(px: int, py: int, lx1: int, ly1: int, lx2: int, ly2: int) -> float:
    return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)

def _is_in_side(v: float, in_sign: int) -> bool:
    return v * in_sign >= 0

def _foot_point(box) -> tuple[int, int]:
    x1, _, x2, y2 = map(int, box.xyxy[0])
    return (x1 + x2) // 2, y2


# ── Embedding ─────────────────────────────────────────────────────────────────

def _extract_embedding(
    extractor, frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *, track_id=None, cam_id: str = "",
) -> "tuple[np.ndarray, float] | None":
    """Ekstrak embedding dari crop.
    Returns (emb_normalized, laplacian_var) jika lulus quality gate, else None."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    h, w = y2 - y1, x2 - x1

    # ── Quality gate ──────────────────────────────────────────────────────────
    if h < QUALITY_MIN_H or w < QUALITY_MIN_W:
        return None
    if h / (w + 1e-6) < QUALITY_MIN_RATIO:
        return None
    gray    = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < QUALITY_LAP_VAR:
        return None

    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feat = extractor([crop])
    emb  = feat[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8), lap_var


# ── Draw ──────────────────────────────────────────────────────────────────────

def _draw_line(frame: np.ndarray, p1: tuple, p2: tuple, in_sign: int) -> None:
    cv2.line(frame, p1, p2, (0, 200, 255), 2)
    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy) or 1
    nx = -dy / length * in_sign
    ny =  dx / length * in_sign
    off = 25
    cv2.putText(frame, "IN",  (int(mx + nx*off), int(my + ny*off)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)
    cv2.putText(frame, "OUT", (int(mx - nx*off), int(my - ny*off)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)


def _draw_tracks(frame: np.ndarray, result, db: IdentityDB, crossed: set[int]) -> None:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, track_id in zip(boxes, boxes.id.int().tolist()):
        if int(box.cls[0]) != 0:
            continue
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 100, 255) if track_id in crossed else (0, 200, 80)
        label = f"{db.name_of(track_id) or f'ID:{track_id}'}  {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - bl - 2), font, 0.6, (0, 0, 0), 2)


def _draw_counter(frame: np.ndarray, count_in: int, count_out: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"In : {count_in}",  (12, 30), font, 0.75, (0, 255, 120), 2)
    cv2.putText(frame, f"Out: {count_out}", (12, 58), font, 0.75, (0, 100, 255), 2)


# ── Public entry point ────────────────────────────────────────────────────────

def process_video(
    req: ProcessVideoRequest,
    detector: YOLO,
    extractor,
) -> ProcessVideoResponse:
    input_path  = Path(req.video_path)
    lp1         = req.line.p1
    lp2         = req.line.p2
    in_sign     = req.line.in_sign

    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    if req.output_path:
        output_path = Path(req.output_path)
    else:
        output_path = Path("output") / f"result_{input_path.stem}_processed.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db          = IdentityDB(req.reid_threshold)
    line_length = math.hypot(lp2[0] - lp1[0], lp2[1] - lp1[1]) or 1

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    side_history:     dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=4))
    track_last_frame: dict[int, int]          = {}
    counted_ids:      set[int]                = set()
    count_in  = 0
    count_out = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps

        results = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=req.conf_threshold,
            classes=[0],
            verbose=False,
        )

        crossed_this_frame: set[int] = set()
        active_ids:         set[int] = set()

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id.int().tolist()):
                if int(box.cls[0]) != 0:
                    continue

                active_ids.add(track_id)

                x1e, y1e, x2e, y2e = map(int, box.xyxy[0])
                result = _extract_embedding(extractor, frame, x1e, y1e, x2e, y2e)
                if result is not None:
                    emb, quality = result
                    if db.name_of(track_id) is None:
                        db.assign(track_id, emb, quality=quality)
                    else:
                        db.refresh(track_id, emb)

                fx, fy    = _foot_point(box)
                curr_side = _point_side(fx, fy, lp1[0], lp1[1], lp2[0], lp2[1])
                history   = side_history[track_id]

                if len(history) > 0 and track_id not in counted_ids:
                    prev       = history[-1]
                    prev_in    = _is_in_side(prev,      in_sign)
                    curr_in    = _is_in_side(curr_side, in_sign)
                    if not prev_in and curr_in:
                        count_in += 1
                        counted_ids.add(track_id)
                        crossed_this_frame.add(track_id)
                    elif prev_in and not curr_in:
                        count_out += 1
                        counted_ids.add(track_id)
                        crossed_this_frame.add(track_id)

                history.append(curr_side)
                track_last_frame[track_id] = frame_idx

        # Buffer crossing
        for tid, last_f in list(track_last_frame.items()):
            if tid in active_ids or tid in counted_ids:
                continue
            frames_gone = frame_idx - last_f
            if frames_gone < 1 or frames_gone > BUFFER_FRAMES:
                continue
            history = side_history[tid]
            if not history:
                continue
            last_side = history[-1]
            if abs(last_side) / line_length > NEAR_LINE_DIST:
                continue
            earliest = history[0]
            if abs(earliest) <= abs(last_side):
                continue
            if not _is_in_side(earliest, in_sign):
                count_in += 1
            else:
                count_out += 1
            counted_ids.add(tid)

        _draw_line(frame, lp1, lp2, in_sign)
        _draw_tracks(frame, results[0], db, crossed_this_frame)
        _draw_counter(frame, count_in, count_out)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return ProcessVideoResponse(
        output_path=str(output_path),
        count_in=count_in,
        count_out=count_out,
        identities=db.to_records(),
        frames_processed=frame_idx,
    )
