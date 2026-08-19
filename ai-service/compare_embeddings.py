"""Bandingkan kekuatan pembeda embedding dua checkpoint OSNet secara langsung,
lepas dari tracker/waktu/threshold — pakai crop ground truth sample3/output.csv.

Metrik: rata-rata cosine similarity untuk pasangan SAMA orang, vs pasangan
BEDA orang. Selisihnya (separation) mengukur seberapa baik model memisahkan
identitas — makin besar makin baik.
"""

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchreid

SAMPLE_DIR = Path("sample3")
GT_PATH    = Path("sample3/output.csv")


def load_crops() -> dict[str, list[np.ndarray]]:
    """person -> list of RGB crops (maks 40/orang, disebar merata, biar cepat)."""
    rows = list(csv.DictReader(open(GT_PATH)))
    by_clip = defaultdict(list)
    for r in rows:
        by_clip[(r["camera"], r["source_clip"])].append(r)

    crops_by_person: dict[str, list[np.ndarray]] = defaultdict(list)
    for (camera, source_clip), clip_rows in by_clip.items():
        clip_rows.sort(key=lambda r: int(r["local_frame"]))
        wanted = {int(r["local_frame"]): r for r in clip_rows}
        path = SAMPLE_DIR / f"{camera}_sim" / source_clip
        cap = cv2.VideoCapture(str(path))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                r = wanted[idx]
                x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = int(x + w), int(y + h)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops_by_person[r["person"]].append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()

    # Subsample biar tidak terlalu lama — ambil merata, maks 40 crop/orang
    for p, lst in crops_by_person.items():
        if len(lst) > 40:
            step = len(lst) // 40
            crops_by_person[p] = lst[::step][:40]
    return crops_by_person


def embed_all(extractor, crops_by_person: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for person, crops in crops_by_person.items():
        with torch.no_grad():
            feats = extractor(crops)
        embs = feats.cpu().numpy()
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        out[person] = embs
    return out


def separation(embs_by_person: dict[str, np.ndarray]) -> tuple[float, float, float]:
    same, diff = [], []
    persons = list(embs_by_person.keys())
    for p in persons:
        E = embs_by_person[p]
        for i, j in itertools.combinations(range(len(E)), 2):
            same.append(float(np.dot(E[i], E[j])))
    for p1, p2 in itertools.combinations(persons, 2):
        E1, E2 = embs_by_person[p1], embs_by_person[p2]
        for i in range(len(E1)):
            for j in range(len(E2)):
                diff.append(float(np.dot(E1[i], E2[j])))
    avg_same = sum(same) / len(same)
    avg_diff = sum(diff) / len(diff)
    return avg_same, avg_diff, avg_same - avg_diff


def main() -> None:
    print("Memuat crop dari video (sekali, dipakai untuk kedua model)...")
    crops_by_person = load_crops()
    for p, c in sorted(crops_by_person.items()):
        print(f"  person {p}: {len(c)} crop")

    print("\n=== Model A: osnet_ain_x1_0 + MSMT17 (yang dipakai sekarang) ===")
    fe_a = torchreid.utils.FeatureExtractor(
        model_name="osnet_ain_x1_0",
        model_path=str(Path.home() / ".cache/torch/checkpoints/osnet_ain_x1_0_msmt17.pt"),
        device="cpu",
    )
    embs_a = embed_all(fe_a, crops_by_person)
    same_a, diff_a, sep_a = separation(embs_a)
    print(f"  avg same-person cos = {same_a:.4f}")
    print(f"  avg diff-person cos = {diff_a:.4f}")
    print(f"  separation          = {sep_a:.4f}")

    print("\n=== Model B: osnet_x1_0 + Market-1501 ===")
    fe_b = torchreid.utils.FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="checkpoints/osnet_x1_0_market1501.pt",
        device="cpu",
    )
    embs_b = embed_all(fe_b, crops_by_person)
    same_b, diff_b, sep_b = separation(embs_b)
    print(f"  avg same-person cos = {same_b:.4f}")
    print(f"  avg diff-person cos = {diff_b:.4f}")
    print(f"  separation          = {sep_b:.4f}")

    print("\n=== Model C: osnet_x1_0 + DukeMTMC-reID ===")
    fe_c = torchreid.utils.FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="checkpoints/osnet_x1_0_dukemtmcreid.pt",
        device="cpu",
    )
    embs_c = embed_all(fe_c, crops_by_person)
    same_c, diff_c, sep_c = separation(embs_c)
    print(f"  avg same-person cos = {same_c:.4f}")
    print(f"  avg diff-person cos = {diff_c:.4f}")
    print(f"  separation          = {sep_c:.4f}")

    print("\n=== Kesimpulan ===")
    results = {
        "A (MSMT17, osnet_ain_x1_0 — dipakai sekarang)": sep_a,
        "B (Market-1501, osnet_x1_0)": sep_b,
        "C (DukeMTMC-reID, osnet_x1_0)": sep_c,
    }
    for name, sep in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {sep:.4f}  {name}")
    winner = max(results, key=results.get)
    print(f"\n  Terbaik: {winner}")


if __name__ == "__main__":
    main()
