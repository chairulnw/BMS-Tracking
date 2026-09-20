"""Uji efek PLAYLIST_LIVE_SIM=1 (antrian frame dipaksa bounded, maxsize=2,
buang frame lama saat penuh -- meniru perilaku RTSP live) terhadap YOLO26n
vs YOLO26x, tracker+Re-ID dikunci sama dengan baris #2/#28 di
pipeline-comparison.md (ByteTrack + TransReID).

Beda dari run_comparison.py:
- PLAYLIST_LIVE_SIM=1 diset sebelum tiap run (lihat cam_slot.py) -- default
  mode file-playlist yang dipakai run_comparison.py TIDAK memakai ini
  (unbounded, tidak buang frame).
- idx dipakai 902/928 (bukan 2/28) supaya predictions_*.csv/latency_*.csv
  yang ditulis TIDAK menimpa hasil run_comparison.py yang sudah ada.
- Hasil ditulis ke pipeline-comparison-livesim.md (file baru), TIDAK
  menimpa pipeline-comparison.md.

Jalankan dari ai-service/: python benchmark/run_livesim.py
"""
import os

os.environ["PLAYLIST_LIVE_SIM"] = "1"

import run_comparison as rc

OUT_MD = "benchmark/pipeline-comparison-livesim.md"

# (idx, label_detektor, file_model, label_tracker, tracker_type, label_reid, reid_model)
RUNS = [
    (902, "YOLO26n", "yolo26n.pt", "ByteTrack", "bytetrack", "TransReID (ViT-B/16*)", "transreid"),
    (928, "YOLO26x", "yolo26x.pt", "ByteTrack", "bytetrack", "TransReID (ViT-B/16*)", "transreid"),
]


def main() -> None:
    rc.PRED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    rc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rc.LATENCY_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, det_label, detector_model, trk_label, tracker_type, reid_label, reid_model in RUNS:
        print(f"\n{'='*70}\n#{idx}: {det_label} + {trk_label} + {reid_label}  (PLAYLIST_LIVE_SIM=1)\n{'='*70}")
        res = rc.run_one(idx, detector_model, tracker_type, reid_model)
        metrics = rc.parse_metrics(res["stdout"])
        for key in ("fps", "cpu_peak", "ram_peak", "decode_ms", "detection_ms",
                    "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"):
            metrics[key] = res.get(key, "")
        row = {"idx": idx, "detektor": det_label, **metrics}
        rows.append(row)
        print(f"  -> {row}")

    with open(OUT_MD, "w") as f:
        f.write("# Uji Antrian Live-Sim (PLAYLIST_LIVE_SIM=1)\n\n")
        f.write(
            "Kombinasi sama dengan baris #2 (YOLO26n) dan #28 (YOLO26x) di "
            "`pipeline-comparison.md` (ByteTrack + TransReID), tapi dijalankan dengan "
            "`PLAYLIST_LIVE_SIM=1` -- antrian frame dipaksa bounded (maxsize=2, buang "
            "frame lama saat penuh), meniru perilaku antrian RTSP live "
            "(lihat cam_slot.py), meski sumber datanya tetap klip rekaman file-playlist "
            "yang sama. Tujuannya mengisolasi efek mekanisme buang-frame terhadap akurasi, "
            "terpisah dari efek karakteristik deteksi model.\n\n"
        )
        f.write(
            "| # | Detektor | Cakupan deteksi | Precision | Recall | F1 | Akurasi | "
            "False merge | False split | Detect ms | Total AI ms |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['idx']} | {r['detektor']} | {r.get('cakupan','')} | "
                f"{r.get('precision','')} | {r.get('recall','')} | {r.get('f1','')} | "
                f"{r.get('akurasi','')} | {r.get('false_merge','')} | {r.get('false_split','')} | "
                f"{r.get('detection_ms','')} | {r.get('total_ai_ms','')} |\n"
            )

    print(f"\nSelesai, ditulis ke {OUT_MD}")


if __name__ == "__main__":
    main()
