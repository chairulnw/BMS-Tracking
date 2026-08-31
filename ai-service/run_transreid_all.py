"""Lanjutan run_transreid_combo2.py: 8 kombinasi TransReID sisanya (full grid,
non-prioritas) ke data sample3/, simpan predictions.csv ke `predictions_runs
copy/` dan lengkapi `predictions_runs copy/pipeline-comparison copy.md`.

Jalankan: python run_transreid_all.py
"""

import json

import run_comparison as rc
import run_transreid_combo2 as rt2

# (num, Detektor label, YOLO_MODEL, Tracker label, TRACKER_TYPE) — Re-ID selalu TransReID.
# #2 (YOLO26n+ByteTrack) sudah selesai lewat run_transreid_combo2.py.
COMBOS = [
    (5,  "YOLO26n",    "yolo26n.pt",  "BoT-SORT",  "botsort"),
    (8,  "YOLO26n",    "yolo26n.pt",  "OC-SORT",   "ocsort"),
    (11, "YOLO11n",    "yolo11n.pt",  "ByteTrack", "bytetrack"),
    (14, "YOLO11n",    "yolo11n.pt",  "BoT-SORT",  "botsort"),
    (17, "YOLO11n",    "yolo11n.pt",  "OC-SORT",   "ocsort"),
    (20, "RTDETRv2-s", "rtdetr-l.pt", "ByteTrack", "bytetrack"),
    (23, "RTDETRv2-s", "rtdetr-l.pt", "BoT-SORT",  "botsort"),
    (26, "RTDETRv2-s", "rtdetr-l.pt", "OC-SORT",   "ocsort"),
]


def main() -> None:
    # import run_transreid_combo2 di atas sudah override rc.PRED_DIR/GT_PATH/
    # MD_PATH/TOTAL_FRAMES ke predictions_runs copy/ (top-level module code-nya).
    rc.PRED_DIR.mkdir(exist_ok=True)
    results_path = rc.PRED_DIR / "results_transreid.json"
    results: list[dict] = json.loads(results_path.read_text()) if results_path.exists() else []
    done_nums = {r["num"] for r in results if r.get("status") == "Selesai"}

    for num, det_label, yolo_model, trk_label, tracker_type in COMBOS:
        if num in done_nums:
            print(f"[skip] #{num} sudah selesai")
            continue
        print(f"\n{'='*70}\n#{num}: {det_label} + {trk_label} + TransReID (ViT-B/16*)\n{'='*70}")
        results = [r for r in results if r["num"] != num]
        try:
            res = rc.run_one(num, yolo_model, tracker_type, "transreid")
            print(res["stdout"])
            m = rt2.parse_metrics_copy(res["stdout"])
            m["fps"], m["cpu_peak"], m["ram_peak"] = res["fps"], res["cpu_peak"], res["ram_peak"]
            m["status"] = "Selesai"
            print("metrics:", json.dumps(m, indent=2, ensure_ascii=False))
            rt2.update_copy_markdown(num, m)
            results.append({"num": num, **m})
        except Exception as exc:
            print(f"  !! GAGAL #{num}: {exc}")
            results.append({"num": num, "status": f"Gagal: {exc}"})
        results_path.write_text(json.dumps(results, indent=2))

    print("\nSemua kombinasi TransReID selesai.")


if __name__ == "__main__":
    main()
