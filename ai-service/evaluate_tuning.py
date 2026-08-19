"""Bandingkan predictions.csv (hasil sistem) dengan sample3/output.csv (ground
truth) untuk T2.12. Jalankan setelah satu run penuh lewat sample3/ selesai.

Metrik:
- False merge: satu person_pred (identitas sistem) mencakup > 1 person GT
  (dua orang berbeda digabung jadi satu identitas).
- False split: satu person GT tersebar ke > 1 person_pred (satu orang
  terpecah jadi beberapa identitas).
"""

import csv
import collections
from pathlib import Path

GT_PATH   = Path("sample3/output.csv")
PRED_PATH = Path("predictions.csv")


def load(path: Path, id_col: str) -> dict[tuple, str]:
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        key = (r["source_clip"], r["local_frame"])
        out[key] = r[id_col]
    return out


def main() -> None:
    gt   = load(GT_PATH, "person")
    pred = load(PRED_PATH, "person_pred")

    matched = {k: (gt[k], pred[k]) for k in gt if k in pred}
    missing = [k for k in gt if k not in pred]

    print(f"GT frames: {len(gt)}  |  Pred frames: {len(pred)}  |  "
          f"Matched: {len(matched)}  |  GT tanpa prediksi: {len(missing)}")

    # person_pred -> set(GT person) — false merge kalau > 1
    pred_to_gt = collections.defaultdict(set)
    # GT person -> set(person_pred) — false split kalau > 1
    gt_to_pred = collections.defaultdict(set)
    for gt_id, pred_id in matched.values():
        pred_to_gt[pred_id].add(gt_id)
        gt_to_pred[gt_id].add(pred_id)

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

    print(f"\n=== Ringkasan ===")
    print(f"  {len(pred_to_gt)} identitas sistem dihasilkan untuk {len(gt_to_pred)} orang GT")
    print(f"  False merge : {n_merge} identitas sistem mencakup >1 orang GT")
    print(f"  False split : {n_split} orang GT terpecah jadi >1 identitas sistem")


if __name__ == "__main__":
    main()
