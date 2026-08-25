"""Bandingkan predictions.csv (hasil sistem) dengan sample/output.csv (ground
truth). Jalankan setelah satu run penuh lewat mode file-playlist selesai.

Pencocokan antar GT dan prediksi dilakukan per frame lewat IoU (Hungarian
assignment), bukan cuma "sama-sama ada di frame ini" — jadi mendukung lebih
dari satu orang per frame, bukan cuma satu.

Metrik:
- False merge: satu person_pred (identitas sistem) mencakup > 1 person GT
  (dua orang berbeda digabung jadi satu identitas).
- False split: satu person GT tersebar ke > 1 person_pred (satu orang
  terpecah jadi beberapa identitas).
"""

import collections
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

GT_PATH       = Path("sample/output.csv")
PRED_PATH     = Path("predictions.csv")
IOU_THRESHOLD = 0.3   # box GT & prediksi di bawah ini dianggap tidak match


def load(path: Path, id_col: str) -> dict[tuple, list[tuple]]:
    """key (source_clip, local_frame) -> list of (id, x, y, w, h)."""
    rows = list(csv.DictReader(open(path)))
    out: dict[tuple, list[tuple]] = collections.defaultdict(list)
    for r in rows:
        key = (r["source_clip"], r["local_frame"])
        out[key].append((r[id_col], float(r["x"]), float(r["y"]),
                          float(r["w"]), float(r["h"])))
    return out


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_frame(gt_boxes: list[tuple], pred_boxes: list[tuple]) -> list[tuple[str, str]]:
    """Pasangkan box GT <-> prediksi dalam satu frame lewat Hungarian
    assignment (maksimalkan total IoU), buang pasangan di bawah threshold."""
    if not gt_boxes or not pred_boxes:
        return []
    cost = np.array([[1.0 - iou(g[1:], p[1:]) for p in pred_boxes] for g in gt_boxes])
    gt_idx, pred_idx = linear_sum_assignment(cost)
    return [
        (gt_boxes[i][0], pred_boxes[j][0])
        for i, j in zip(gt_idx, pred_idx)
        if cost[i, j] <= 1.0 - IOU_THRESHOLD
    ]


def main() -> None:
    gt   = load(GT_PATH, "person")
    pred = load(PRED_PATH, "person_pred")

    gt_frames, pred_frames = set(gt), set(pred)
    matched_frames = gt_frames & pred_frames
    n_gt_boxes = sum(len(v) for v in gt.values())

    print(f"GT frames: {len(gt_frames)} ({n_gt_boxes} box)  |  "
          f"Pred frames: {len(pred_frames)}  |  "
          f"Frame matched: {len(matched_frames)}  |  "
          f"GT frame tanpa prediksi: {len(gt_frames - pred_frames)}")

    pred_to_gt: dict[str, set[str]] = collections.defaultdict(set)
    gt_to_pred: dict[str, set[str]] = collections.defaultdict(set)
    n_box_matched = 0
    for key in matched_frames:
        for gt_id, pred_id in match_frame(gt[key], pred[key]):
            pred_to_gt[pred_id].add(gt_id)
            gt_to_pred[gt_id].add(pred_id)
            n_box_matched += 1

    print(f"Box GT tercocokkan ke prediksi (IoU >= {IOU_THRESHOLD}): "
          f"{n_box_matched}/{n_gt_boxes}")

    print("\n=== Per identitas sistem (person_pred) ===")
    n_merge = 0
    for pred_id, gt_ids in sorted(pred_to_gt.items()):
        flag = "  <-- FALSE MERGE" if len(gt_ids) > 1 else ""
        if flag:
            n_merge += 1
        print(f"  {pred_id!r:30s} -> GT person {sorted(gt_ids)}{flag}")

    print("\n=== Per orang ground truth (person) ===")
    n_split = 0
    for gt_id, pred_ids in sorted(gt_to_pred.items()):
        flag = "  <-- FALSE SPLIT" if len(pred_ids) > 1 else ""
        if flag:
            n_split += 1
        print(f"  person {gt_id} -> identitas sistem {sorted(pred_ids)}{flag}")

    all_gt_ids = {gid for boxes in gt.values() for gid, *_ in boxes}
    unmatched  = sorted(all_gt_ids - set(gt_to_pred))

    print("\n=== Ringkasan ===")
    print(f"  {len(pred_to_gt)} identitas sistem dihasilkan untuk {len(gt_to_pred)} orang GT")
    print(f"  False merge : {n_merge} identitas sistem mencakup >1 orang GT")
    print(f"  False split : {n_split} orang GT terpecah jadi >1 identitas sistem")
    if unmatched:
        print(f"  Orang GT tidak pernah tercocokkan sama sekali: {unmatched}")


if __name__ == "__main__":
    main()
