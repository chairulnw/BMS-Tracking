"""
Pipeline lengkap: YOLO26s + ByteTrack + OSNet ReID + user-defined line crossing.

Startup  : pilih garis crossing interaktif (3 phase).
Main loop: deteksi → tracking → ReID → crossing detection.
Log      : "Unknown #1 → IN" saat orang melewati garis.
Output   : output/result_full.mp4
"""

import math
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from pathlib import Path
from ultralytics import YOLO
import torchreid

INPUT      = Path("sample/test.mov")
OUTPUT     = Path("output/result_full.mp4")
YOLO_MODEL = "yolo26n.pt"
REID_MODEL = "osnet_x1_0"

PERSON_CLASS   = 0
CONF_THRESHOLD = 0.52

REID_THRESHOLD = 0.65
MIN_CROP_PX    = 32
EMBED_REFRESH  = 15

BUFFER_FRAMES  = 10
NEAR_LINE_DIST = 40

LINE_COLOR  = (0, 200, 255)
LINE_THICK  = 2
BOX_COLOR   = (0, 200, 80)
CROSS_COLOR = (0, 100, 255)
LABEL_BG    = (0, 140, 220)
THICKNESS   = 2
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.6


# ── Identity database ─────────────────────────────────────────────────────────

class IdentityDB:
    def __init__(self) -> None:
        self._embeddings:    dict[str, np.ndarray] = {}
        self._track_to_name: dict[int, str]        = {}
        self._frame_counter: dict[int, int]        = {}
        self._count = 0

    def _new_name(self) -> str:
        self._count += 1
        return f"Unknown #{self._count}"

    def _best_match(self, emb: np.ndarray) -> tuple[str | None, float]:
        best_name, best_sim = None, -1.0
        for name, stored in self._embeddings.items():
            sim = float(np.dot(emb, stored))
            if sim > best_sim:
                best_sim, best_name = sim, name
        return best_name, best_sim

    def assign(self, track_id: int, emb: np.ndarray) -> str:
        if track_id in self._track_to_name:
            return self._track_to_name[track_id]
        name, sim = self._best_match(emb)
        if name is None or sim < REID_THRESHOLD:
            name = self._new_name()
            self._embeddings[name] = emb
            print(f"  [ReID] New: {name}")
        else:
            print(f"  [ReID] Track {track_id} → {name}  (sim={sim:.3f})")
        self._track_to_name[track_id] = name
        self._frame_counter[track_id] = 0
        return name

    def refresh(self, track_id: int, emb: np.ndarray) -> None:
        cnt = self._frame_counter.get(track_id, 0) + 1
        self._frame_counter[track_id] = cnt
        if cnt % EMBED_REFRESH != 0:
            return
        name = self._track_to_name.get(track_id)
        if name is None:
            return
        merged = 0.9 * self._embeddings[name] + 0.1 * emb
        self._embeddings[name] = merged / (np.linalg.norm(merged) + 1e-8)

    def name_of(self, track_id: int) -> str | None:
        return self._track_to_name.get(track_id)

    def summary(self) -> None:
        print(f"\nIdentity DB — {len(self._embeddings)} identities:")
        for name in self._embeddings:
            tids = [t for t, n in self._track_to_name.items() if n == name]
            print(f"  {name}  ← track IDs {tids}")


# ── Embedding extraction ──────────────────────────────────────────────────────

def extract_embedding(extractor, frame: cv2.typing.MatLike, box) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
        return None
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feat = extractor([crop])
    emb  = feat[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)


# ── Geometry ──────────────────────────────────────────────────────────────────

def point_side(px: int, py: int, lx1: int, ly1: int, lx2: int, ly2: int) -> float:
    return (lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1)

def is_in_side(v: float, in_sign: int) -> bool:
    return v * in_sign >= 0

def foot_point(box) -> tuple[int, int]:
    x1, _, x2, y2 = map(int, box.xyxy[0])
    return (x1 + x2) // 2, y2


# ── Interactive line selector (3 phase) ──────────────────────────────────────

def select_crossing_line(
    first_frame: cv2.typing.MatLike,
) -> tuple[tuple[int, int], tuple[int, int], int]:
    window    = "Define Crossing Line"
    points:   list[tuple[int, int]]  = []
    in_click: tuple[int, int] | None = None
    in_sign:  int = 1
    phase:    int = 1

    def overlay_text(img: cv2.typing.MatLike, msgs: list[str]) -> None:
        h = img.shape[0]
        for i, msg in enumerate(msgs):
            y = h - 14 - i * 24
            cv2.putText(img, msg, (9, y + 1), FONT, 0.52, (0, 0, 0),      2, cv2.LINE_AA)
            cv2.putText(img, msg, (9, y),     FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_garis_labeled(img: cv2.typing.MatLike, sign: int) -> None:
        p1, p2 = points[0], points[1]
        cv2.line(img, p1, p2, LINE_COLOR, LINE_THICK)
        for pt in (p1, p2):
            cv2.circle(img, pt, 6, (0, 0, 255), -1)
            cv2.circle(img, pt, 6, (255, 255, 255), 1)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy) or 1
        nx = -dy / length * sign
        ny =  dx / length * sign
        off = 35
        cv2.putText(img, "IN",  (int(mx + nx*off), int(my + ny*off)), FONT, 0.7, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(img, "OUT", (int(mx - nx*off), int(my - ny*off)), FONT, 0.7, (0, 100, 255), 2, cv2.LINE_AA)

    def redraw() -> None:
        img = first_frame.copy()
        if phase == 1:
            for pt in points:
                cv2.circle(img, pt, 7, (0, 0, 255), -1)
                cv2.circle(img, pt, 7, (255, 255, 255), 1)
            if len(points) == 2:
                cv2.line(img, points[0], points[1], LINE_COLOR, LINE_THICK)
            hint = "R: reset  |  Enter/Space: lanjut" if len(points) == 2 else "Klik titik ke-2"
            overlay_text(img, [hint, "Phase 1/3 — Klik 2 titik untuk menentukan garis crossing"])
        elif phase == 2:
            p1, p2 = points[0], points[1]
            cv2.line(img, p1, p2, LINE_COLOR, LINE_THICK)
            for pt in (p1, p2):
                cv2.circle(img, pt, 6, (0, 0, 255), -1)
            h, w = img.shape[:2]
            cv2.putText(img, "Klik sisi IN", (w // 2 - 110, h // 2),
                        FONT, 1.2, (0, 0, 0),      5, cv2.LINE_AA)
            cv2.putText(img, "Klik sisi IN", (w // 2 - 110, h // 2),
                        FONT, 1.2, (0, 255, 120),  3, cv2.LINE_AA)
            overlay_text(img, ["R: ulang", "Phase 2/3 — Klik area yang ingin dihitung sebagai IN (masuk)"])
        elif phase == 3:
            draw_garis_labeled(img, in_sign)
            if in_click:
                cv2.drawMarker(img, in_click, (0, 255, 120), cv2.MARKER_CROSS, 24, 2)
            overlay_text(img, ["R: ulang  |  Enter/Space: mulai proses", "Phase 3/3 — Konfirmasi label IN/OUT"])
        cv2.imshow(window, img)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param) -> None:
        nonlocal phase, in_sign, in_click
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if phase == 1 and len(points) < 2:
            points.append((x, y))
            redraw()
        elif phase == 2:
            in_click = (x, y)
            raw      = point_side(x, y, points[0][0], points[0][1], points[1][0], points[1][1])
            in_sign  = 1 if raw >= 0 else -1
            phase    = 3
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    print("Phase 1/3: Klik 2 titik untuk menentukan garis crossing.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            if phase == 3:
                break
            raise RuntimeError("Window ditutup sebelum konfigurasi selesai.")
        if key in (ord("r"), ord("R")):
            points.clear()
            in_click = None
            in_sign  = 1
            phase    = 1
            redraw()
            print("Reset. Phase 1/3.")
        elif key in (13, 32):
            if phase == 1:
                if len(points) == 2:
                    phase = 2
                    redraw()
                    print("Phase 2/3: Klik area yang ingin dihitung sebagai IN (masuk).")
                else:
                    print("Pilih 2 titik terlebih dahulu.")
            elif phase == 3:
                break

    cv2.destroyAllWindows()
    return points[0], points[1], in_sign


# ── Draw helpers ──────────────────────────────────────────────────────────────

def draw_line(frame: cv2.typing.MatLike, p1: tuple, p2: tuple, in_sign: int) -> None:
    cv2.line(frame, p1, p2, LINE_COLOR, LINE_THICK)
    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy) or 1
    nx = -dy / length * in_sign
    ny =  dx / length * in_sign
    off = 25
    cv2.putText(frame, "IN",  (int(mx + nx*off), int(my + ny*off)), FONT, 0.5, (0, 255, 120), 2)
    cv2.putText(frame, "OUT", (int(mx - nx*off), int(my - ny*off)), FONT, 0.5, (0, 100, 255), 2)


def draw_counter(frame: cv2.typing.MatLike, count_in: int, count_out: int) -> None:
    cv2.putText(frame, f"In : {count_in}",  (12, 30), FONT, 0.75, (0, 255, 120), 2)
    cv2.putText(frame, f"Out: {count_out}", (12, 58), FONT, 0.75, (0, 100, 255), 2)


def draw_tracks(
    frame: cv2.typing.MatLike,
    result,
    db: IdentityDB,
    crossed_ids: set[int],
) -> None:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return
    for box, track_id in zip(boxes, boxes.id.int().tolist()):
        if int(box.cls[0]) != PERSON_CLASS:
            continue
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name  = db.name_of(track_id) or f"ID:{track_id}"
        label = f"{name}  {conf:.2f}"
        color = CROSS_COLOR if track_id in crossed_ids else BOX_COLOR

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
        (tw, th), bl = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - bl - 2), FONT, FONT_SCALE, (0, 0, 0), THICKNESS)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Loading OSNet ({REID_MODEL})…")
    extractor = torchreid.utils.FeatureExtractor(model_name=REID_MODEL, device=device)
    print("OSNet loaded.\n")

    cap = cv2.VideoCapture(str(INPUT))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {INPUT}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Tidak bisa membaca frame pertama.")

    lp1, lp2, in_sign = select_crossing_line(first_frame)
    print(f"Garis crossing: {lp1} → {lp2}  (in_sign={in_sign:+d})\n")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    detector    = YOLO(YOLO_MODEL)
    db          = IdentityDB()
    line_length = math.hypot(lp2[0] - lp1[0], lp2[1] - lp1[1]) or 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps, (width, height))
    print(f"Processing {total} frames  ({width}×{height} @ {fps:.1f} fps)\n")

    side_history:     dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=4))
    track_last_frame: dict[int, int]          = {}
    counted_ids:      set[int]                = set()
    count_in  = 0
    count_out = 0

    def log_crossing(name: str, track_id: int, direction: str, ts: float, buf: bool = False) -> None:
        tag = " [buffer]" if buf else ""
        print(f"[{ts:7.2f}s] {name} (ID:{track_id})  → {direction}{tag}")

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
            conf=CONF_THRESHOLD,
            classes=[PERSON_CLASS],
            verbose=False,
        )

        crossed_this_frame: set[int] = set()
        active_ids:         set[int] = set()

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id.int().tolist()):
                if int(box.cls[0]) != PERSON_CLASS:
                    continue

                active_ids.add(track_id)

                # ReID
                emb = extract_embedding(extractor, frame, box)
                if emb is not None:
                    if db.name_of(track_id) is None:
                        db.assign(track_id, emb)
                    else:
                        db.refresh(track_id, emb)

                # Crossing
                fx, fy    = foot_point(box)
                curr_side = point_side(fx, fy, lp1[0], lp1[1], lp2[0], lp2[1])
                history   = side_history[track_id]

                if len(history) > 0 and track_id not in counted_ids:
                    prev       = history[-1]
                    prev_is_in = is_in_side(prev,      in_sign)
                    curr_is_in = is_in_side(curr_side, in_sign)
                    name       = db.name_of(track_id) or f"ID:{track_id}"
                    if not prev_is_in and curr_is_in:
                        count_in += 1
                        counted_ids.add(track_id)
                        crossed_this_frame.add(track_id)
                        log_crossing(name, track_id, "IN", timestamp)
                    elif prev_is_in and not curr_is_in:
                        count_out += 1
                        counted_ids.add(track_id)
                        crossed_this_frame.add(track_id)
                        log_crossing(name, track_id, "OUT", timestamp)

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
            direction = "IN" if not is_in_side(earliest, in_sign) else "OUT"
            name      = db.name_of(tid) or f"ID:{tid}"
            if direction == "IN":
                count_in += 1
            else:
                count_out += 1
            counted_ids.add(tid)
            log_crossing(name, tid, direction, timestamp, buf=True)

        draw_line(frame, lp1, lp2, in_sign)
        draw_tracks(frame, results[0], db, crossed_this_frame)
        draw_counter(frame, count_in, count_out)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames  |  In:{count_in}  Out:{count_out}  Identities:{db._count}")

    cap.release()
    writer.release()
    db.summary()
    print(f"\nFinal — In:{count_in}  Out:{count_out}")
    print(f"Output saved to: {OUTPUT}")


if __name__ == "__main__":
    run()
