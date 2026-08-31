"""Jalankan kombinasi #2 (YOLO26n + ByteTrack + TransReID) lewat data
sample3/ (kamera *_sim di DB sudah dialihkan ke sample3/, lihat
CLAUDE.md/pipeline-comparison copy.md), simpan predictions.csv ke
`predictions_runs copy/` dan evaluasi ke `predictions_runs copy/output.csv`.

Reuse run_comparison.py apa adanya (cuma override PRED_DIR/GT_PATH/
TOTAL_FRAMES lewat monkeypatch modul, bukan duplikasi logic run_one()).

Jalankan: python run_transreid_combo2.py
"""

import json
import re
from pathlib import Path

import run_comparison as rc

rc.PRED_DIR = Path("predictions_runs copy")
rc.GT_PATH  = Path("predictions_runs copy/output.csv")
rc.MD_PATH  = Path("predictions_runs copy/pipeline-comparison copy.md")
rc.TOTAL_FRAMES = 2168   # 28 klip sample3/ (3 klip 0622 sudah tidak ada)

COL_MAP = {  # index kolom tabel pipeline-comparison copy.md (0-based split "|")
    "status": 5, "identitas": 6, "precision": 7, "recall": 8, "f1": 9,
    "akurasi": 10, "false_merge": 11, "false_split": 12,
    "fps": 13, "cpu_peak": 14, "ram_peak": 15,
}


def parse_metrics_copy(stdout: str) -> dict:
    m = rc.parse_metrics(stdout)
    for raw in stdout.splitlines():
        line = raw.strip()
        id_m = re.match(r"(\d+) identitas sistem dihasilkan untuk (\d+) orang GT", line)
        if id_m:
            m["identitas"] = f"{id_m.group(1)}/{id_m.group(2)}"
        ak_m = re.match(r"Akurasi\s*:\s*(\d+/\d+).*\(([\d.]+%)\)", line)
        if ak_m:
            m["akurasi"] = f"{ak_m.group(1)} ({ak_m.group(2)})"
    return m


def update_copy_markdown(num: int, m: dict) -> None:
    lines = rc.MD_PATH.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.startswith(f"| {num} ") and not line.startswith(f"| {num}  "):
            continue
        cols = line.split("|")
        if len(cols) < 17:
            continue
        for key, idx in COL_MAP.items():
            cols[idx] = f" {m.get(key, '')} "
        lines[i] = "|".join(cols)
        break
    rc.MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    rc.PRED_DIR.mkdir(exist_ok=True)
    num, det_label, yolo_model, trk_label, tracker_type, reid_label, reid_model = (
        2, "YOLO26n", "yolo26n.pt", "ByteTrack", "bytetrack", "TransReID (ViT-B/16*)", "transreid",
    )
    print(f"[run] #{num}: {det_label} + {trk_label} + {reid_label} -> sample3/")
    res = rc.run_one(num, yolo_model, tracker_type, reid_model)
    print(res["stdout"])
    m = parse_metrics_copy(res["stdout"])
    m["fps"], m["cpu_peak"], m["ram_peak"] = res["fps"], res["cpu_peak"], res["ram_peak"]
    m["status"] = "Selesai"
    print("metrics:", json.dumps(m, indent=2, ensure_ascii=False))
    update_copy_markdown(num, m)
    print(f"[run] ditulis ke {rc.MD_PATH}")


if __name__ == "__main__":
    main()
