"""Analisis open-set yang sebenarnya: pakai skor associate() sungguhan dari log
debug (DEBUG_REID=1) run terhadap sample3, diberi label genuine/impostor lewat
ground truth — bukan rata-rata cos crop-pair yang lepas dari tracker/waktu.

genuine   = tracklet dan kandidat sama-sama berasal dari orang GT yang sama
impostor  = tracklet dan kandidat berasal dari orang GT yang berbeda

Menjawab: di ASSOC_THRESHOLD=0.62 sekarang, berapa genuine yang salah tertolak
(false reject) dan berapa impostor yang salah lolos (false accept)?
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LOG_PATH  = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ai_tuning.log")
GT_PATH   = Path("sample/output.csv")
PRED_PATH = Path("predictions.csv")

CMP_RE = re.compile(
    r"\[assoc\.cmp\] (\S+)/t(\d+) vs '([^']+)': cos=([\d.]+) dt=(-?\d+)s "
    r"f_time=([\d.]+) cam=(\S+)->(\S+) p_cam=([\d.]+) score=([\d.]+)"
)
CLOSE_RE = re.compile(r"(\S+)/t(\d+) tracklet ditutup .* → (?:NEW|MATCH) '([^']+)'")


def label_to_gt_person() -> dict[str, str]:
    """person_pred (mis. "Unknown #6") -> GT person mayoritas dari frame-frame miliknya."""
    gt = {(r["source_clip"], r["local_frame"]): r["person"]
          for r in csv.DictReader(open(GT_PATH))}
    label_gt: dict[str, Counter] = defaultdict(Counter)
    for r in csv.DictReader(open(PRED_PATH)):
        key = (r["source_clip"], r["local_frame"])
        if key in gt:
            label_gt[r["person_pred"]][gt[key]] += 1
    return {label: counter.most_common(1)[0][0] for label, counter in label_gt.items()}


def track_to_label() -> dict[tuple[str, int], str]:
    """(camera, track_id) -> nama identitas akhir tracklet ini, dari log."""
    out = {}
    with open(LOG_PATH) as f:
        for line in f:
            m = CLOSE_RE.search(line)
            if m:
                cam, tid, label = m.groups()
                out[(cam, int(tid))] = label
    return out


def main() -> None:
    label_gt   = label_to_gt_person()
    track_label = track_to_label()

    genuine, impostor = [], []
    with open(LOG_PATH) as f:
        for line in f:
            m = CMP_RE.search(line)
            if not m:
                continue
            cam, track_id, cand, cos, dt, ft, cam_from, cam_to, p_cam, score = m.groups()
            score = float(score)
            tl_label = track_label.get((cam, int(track_id)))
            if tl_label is None:
                continue
            tl_gt   = label_gt.get(tl_label)
            cand_gt = label_gt.get(cand)
            if tl_gt is None or cand_gt is None:
                continue
            (genuine if tl_gt == cand_gt else impostor).append(score)

    print(f"Perbandingan genuine (tracklet & kandidat = orang GT sama): {len(genuine)}")
    print(f"Perbandingan impostor (orang GT beda): {len(impostor)}\n")

    THR = 0.62
    if genuine:
        below = sum(1 for s in genuine if s < THR)
        print(f"Genuine  — avg={sum(genuine)/len(genuine):.3f}  min={min(genuine):.3f}  max={max(genuine):.3f}")
        print(f"  {below}/{len(genuine)} di BAWAH threshold {THR} (kandidat ini gagal menang, walau ini orang yang benar)")
        print(f"  Skor: {sorted(round(s,3) for s in genuine)}")
    if impostor:
        above = sum(1 for s in impostor if s >= THR)
        print(f"\nImpostor — avg={sum(impostor)/len(impostor):.3f}  min={min(impostor):.3f}  max={max(impostor):.3f}")
        print(f"  {above}/{len(impostor)} di ATAS threshold {THR} (kandidat salah ini lolos threshold — bahaya kalau menang margin)")
        if above:
            print(f"  Skor impostor yang lolos threshold: {sorted(round(s,3) for s in impostor if s >= THR)}")


if __name__ == "__main__":
    main()
