"""One-off: jalanin 2 kombinasi yang belum ada (yolo26s/yolo26x + ByteTrack +
OSNet) buat pipeline-comparison-yolo26-size.md, TANPA nyentuh COMBOS/markdown
utama di run_comparison.py. Jalankan dari ai-service/: python benchmark/run_extra_osnet.py
"""
import json
from pathlib import Path

import run_comparison as rc

EXTRA = [
    (30, "yolo26s.pt", "bytetrack", "osnet_ain_x1_0"),
    (31, "yolo26x.pt", "bytetrack", "osnet_ain_x1_0"),
]

if __name__ == "__main__":
    rc.PRED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    rc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rc.LATENCY_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for idx, detector_model, tracker_type, reid_model in EXTRA:
        print(f"\n{'='*70}\n#{idx}: {detector_model} + {tracker_type} + {reid_model}\n{'='*70}")
        res = rc.run_one(idx, detector_model, tracker_type, reid_model)
        metrics = rc.parse_metrics(res["stdout"])
        for key in ("fps", "cpu_peak", "ram_peak", "decode_ms", "detection_ms",
                    "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"):
            metrics[key] = res.get(key, "")
        metrics["status"] = "Selesai"
        print(f"  -> {metrics}")
        out[idx] = metrics
    Path("benchmark/run_extra_osnet_results.json").write_text(json.dumps(out, indent=2))
    print("\nHasil disimpan ke benchmark/run_extra_osnet_results.json")
