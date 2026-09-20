"""Rerun kombinasi #1 (YOLO26n + ByteTrack + OSNet) buat cek variasi
run-to-run terhadap nilai yang di-reuse di pipeline-comparison-yolo26-size.md."""
import run_comparison as rc

if __name__ == "__main__":
    res = rc.run_one(41, "yolo26n.pt", "bytetrack", "osnet_ain_x1_0")
    metrics = rc.parse_metrics(res["stdout"])
    for key in ("fps", "cpu_peak", "ram_peak", "decode_ms", "detection_ms",
                "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"):
        metrics[key] = res.get(key, "")
    print(metrics)
