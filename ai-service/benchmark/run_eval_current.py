"""Eval OSNet vs TransReID di beberapa preset konfigurasi, tanpa nimpa run lama.

Output ke predictions_runs_current/ — predictions_runs/ dan
benchmark/pipeline-comparison.md TIDAK disentuh.

Jalankan dari root ai-service/, matikan dulu AI service manual di port 8001.

    python benchmark/run_eval_current.py current   # config kode sekarang
    python benchmark/run_eval_current.py strict     # + MIN_MARGIN=0.05, FRAG_MIN_SIM=0.7
    python benchmark/run_eval_current.py oldcfg      # dijalankan lewat run_eval_oldcfg.sh
"""

import os
import sys
from pathlib import Path

import evaluate  # noqa: F401  (dipakai run_comparison lewat side-effect path)
import run_comparison as rc

MODE = sys.argv[1] if len(sys.argv) > 1 else "current"

PRESETS = {
    "current": {},
    "strict":  {"MIN_MARGIN": "0.05", "CONCURRENT_FRAGMENT_MIN_SIM": "0.7"},
    "oldcfg":  {},   # kode app/ di-checkout ke e725842^ oleh run_eval_oldcfg.sh
    "verify":  {},   # kode sekarang (oldcfg ditulis manual) — cek match sama oldcfg
    "delta3":  {},   # oldcfg + BANK_EXPAND_MIN 0.80 + min 2 sample + ASSOC_THRESHOLD 0.65
    "delta4":  {},   # delta3 + BANK_EXPAND_MIN diturunkan ke 0.67
}
REID_FILTER = sys.argv[2] if len(sys.argv) > 2 else None   # "osnet_ain_x1_0" | "transreid"
extra_env = PRESETS[MODE]
os.environ.update(extra_env)

OUT_DIR = Path("predictions_runs_current")
rc.PRED_DIR      = OUT_DIR
rc.PRED_CSV_DIR  = OUT_DIR / "predictions"
rc.LOG_DIR       = OUT_DIR / "logs"
rc.RESULTS_DIR   = OUT_DIR / "results"
rc.LATENCY_DIR   = OUT_DIR / "latency"
for d in (rc.PRED_CSV_DIR, rc.LOG_DIR, rc.RESULTS_DIR, rc.LATENCY_DIR):
    d.mkdir(parents=True, exist_ok=True)

BASE_IDX = {"current": 900, "strict": 910, "oldcfg": 920, "verify": 930, "delta3": 940, "delta4": 950}[MODE]
COMBOS = [
    (BASE_IDX + 1, "yolo26n.pt", "bytetrack", "osnet_ain_x1_0"),
    (BASE_IDX + 2, "yolo26n.pt", "bytetrack", "transreid"),
]
if REID_FILTER:
    COMBOS = [c for c in COMBOS if c[3] == REID_FILTER]

print(f"MODE={MODE}  extra_env={extra_env or '(none)'}")
summary = []
for idx, det, trk, reid in COMBOS:
    print(f"\n{'='*70}\n{MODE} :: {reid}  ({det} + {trk})\n{'='*70}")
    try:
        res = rc.run_one(idx, det, trk, reid)
    except Exception as exc:
        print(f"  !! GAGAL: {exc}")
        summary.append((reid, {}))
        continue
    (OUT_DIR / f"eval_{MODE}_{reid}.txt").write_text(res["stdout"])
    m = rc.parse_metrics(res["stdout"])
    m.update({k: res.get(k, "") for k in ("fps", "cpu_peak", "ram_peak", "reid_ms", "total_ai_ms")})
    summary.append((reid, m))
    print(res["stdout"])

print(f"\n{'='*70}\nRINGKASAN  (MODE={MODE})\n{'='*70}")
keys = ["cakupan", "identitas", "precision", "recall", "f1", "akurasi",
        "false_merge", "false_split", "fps", "cpu_peak", "total_ai_ms"]
print(f"{'metrik':<14}" + "".join(f"{r:<22}" for r, _ in summary))
for k in keys:
    print(f"{k:<14}" + "".join(f"{str(m.get(k, '')):<22}" for _, m in summary))
print(f"\nDetail: {OUT_DIR}/eval_{MODE}_*.txt")
