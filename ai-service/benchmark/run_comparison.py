"""Orkestrasi otomatis: jalanin AI Service per kombinasi (deteksi x tracker x
Re-ID), proses seluruh klip sample/ lewat file-playlist, simpan
predictions.csv per kombinasi (nama beda-beda) ke predictions_runs/predictions/,
evaluasi ke sample/output.csv, tulis hasilnya ke benchmark/pipeline-comparison.md.
Log server per kombinasi di predictions_runs/logs/, ringkasan JSON (buat
resume) di predictions_runs/results/.

Restart proses uvicorn penuh dibutuhkan tiap ganti kombinasi karena
DETECTOR_MODEL/REID_MODEL/TRACKER_TYPE dibaca sebagai konstanta modul saat
proses start (lihat main.py/batch_processor.py) — tidak bisa diganti lewat
API di proses yang sama.

Satu script buat semua kombinasi (dulu run_transreid_combo2.py/run_transreid_
all.py/run_bot_resnet50.py masing-masing hardcode subset beda-beda — sekarang
tinggal pilih nomornya lewat argumen):

    python benchmark/run_comparison.py            # semua 27 kombinasi (resume, skip yang udah Selesai)
    python benchmark/run_comparison.py 2          # cuma #2 (paksa run ulang walau udah Selesai)
    python benchmark/run_comparison.py 2 3 9      # #2, #3, #9

Jalankan dari root ai-service/ (path di bawah relatif ke cwd, bukan ke lokasi
file ini).
"""

import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import jwt
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

import evaluate

load_dotenv()

# /stream/* di ai-service butuh JWT valid (Depends(get_current_user)) — token
# apa pun yang ditandatangani SECRET_KEY yang sama diterima, jadi cukup mint
# sendiri di sini (bukan lewat /auth/login backend, kita nggak punya password).
_SECRET_KEY = os.environ["SECRET_KEY"]
_AUTH_HEADERS = {"Authorization": "Bearer " + jwt.encode(
    {"sub": "run_comparison", "exp": datetime.now(timezone.utc) + timedelta(hours=24)},
    _SECRET_KEY, algorithm="HS256",
)}

PORT           = 8001
BASE_URL       = f"http://localhost:{PORT}"
PRED_DIR       = Path("predictions_runs")          # artefak mentah per-run (banyak file kecil)
PRED_CSV_DIR   = PRED_DIR / "predictions"           # predictions_NN_*.csv
LOG_DIR        = PRED_DIR / "logs"                  # server_NN.log
RESULTS_DIR    = PRED_DIR / "results"               # results*.json
LATENCY_DIR    = PRED_DIR / "latency"               # latency_NN_*.csv (copy latency.csv tiap run)
GT_PATH        = Path("sample/output.csv")
MD_PATH        = Path("benchmark/pipeline-comparison.md")   # dokumen hasil (dibaca manusia), bareng script-nya
MODEL_LOAD_TIMEOUT_SEC   = 300
STREAM_DONE_TIMEOUT_SEC  = 1800
POLL_SEC       = 3
TOTAL_FRAMES = 2168
COMBOS = [
    (1,  "YOLO26n",    "yolo26n.pt",  "ByteTrack", "bytetrack", "OSNet",                "osnet_ain_x1_0"),
    (2,  "YOLO26n",    "yolo26n.pt",  "ByteTrack", "bytetrack", "TransReID (ViT-B/16*)", "transreid"),
    (3,  "YOLO26n",    "yolo26n.pt",  "ByteTrack", "bytetrack", "BoT (ResNet50)",        "bot_resnet50"),
    (4,  "YOLO26n",    "yolo26n.pt",  "BoT-SORT",  "botsort",   "OSNet",                "osnet_ain_x1_0"),
    (5,  "YOLO26n",    "yolo26n.pt",  "BoT-SORT",  "botsort",   "TransReID (ViT-B/16*)", "transreid"),
    (6,  "YOLO26n",    "yolo26n.pt",  "BoT-SORT",  "botsort",   "BoT (ResNet50)",        "bot_resnet50"),
    (7,  "YOLO26n",    "yolo26n.pt",  "OC-SORT",   "ocsort",    "OSNet",                "osnet_ain_x1_0"),
    (8,  "YOLO26n",    "yolo26n.pt",  "OC-SORT",   "ocsort",    "TransReID (ViT-B/16*)", "transreid"),
    (9,  "YOLO26n",    "yolo26n.pt",  "OC-SORT",   "ocsort",    "BoT (ResNet50)",        "bot_resnet50"),
    (10, "YOLO11n",    "yolo11n.pt",  "ByteTrack", "bytetrack", "OSNet",                "osnet_ain_x1_0"),
    (11, "YOLO11n",    "yolo11n.pt",  "ByteTrack", "bytetrack", "TransReID (ViT-B/16*)", "transreid"),
    (12, "YOLO11n",    "yolo11n.pt",  "ByteTrack", "bytetrack", "BoT (ResNet50)",        "bot_resnet50"),
    (13, "YOLO11n",    "yolo11n.pt",  "BoT-SORT",  "botsort",   "OSNet",                "osnet_ain_x1_0"),
    (14, "YOLO11n",    "yolo11n.pt",  "BoT-SORT",  "botsort",   "TransReID (ViT-B/16*)", "transreid"),
    (15, "YOLO11n",    "yolo11n.pt",  "BoT-SORT",  "botsort",   "BoT (ResNet50)",        "bot_resnet50"),
    (16, "YOLO11n",    "yolo11n.pt",  "OC-SORT",   "ocsort",    "OSNet",                "osnet_ain_x1_0"),
    (17, "YOLO11n",    "yolo11n.pt",  "OC-SORT",   "ocsort",    "TransReID (ViT-B/16*)", "transreid"),
    (18, "YOLO11n",    "yolo11n.pt",  "OC-SORT",   "ocsort",    "BoT (ResNet50)",        "bot_resnet50"),
    (19, "RTDETRv2-s", "rtdetr-l.pt", "ByteTrack", "bytetrack", "OSNet",                "osnet_ain_x1_0"),
    (20, "RTDETRv2-s", "rtdetr-l.pt", "ByteTrack", "bytetrack", "TransReID (ViT-B/16*)", "transreid"),
    (21, "RTDETRv2-s", "rtdetr-l.pt", "ByteTrack", "bytetrack", "BoT (ResNet50)",        "bot_resnet50"),
    (22, "RTDETRv2-s", "rtdetr-l.pt", "BoT-SORT",  "botsort",   "OSNet",                "osnet_ain_x1_0"),
    (23, "RTDETRv2-s", "rtdetr-l.pt", "BoT-SORT",  "botsort",   "TransReID (ViT-B/16*)", "transreid"),
    (24, "RTDETRv2-s", "rtdetr-l.pt", "BoT-SORT",  "botsort",   "BoT (ResNet50)",        "bot_resnet50"),
    (25, "RTDETRv2-s", "rtdetr-l.pt", "OC-SORT",   "ocsort",    "OSNet",                "osnet_ain_x1_0"),
    (26, "RTDETRv2-s", "rtdetr-l.pt", "OC-SORT",   "ocsort",    "TransReID (ViT-B/16*)", "transreid"),
    (27, "RTDETRv2-s", "rtdetr-l.pt", "OC-SORT",   "ocsort",    "BoT (ResNet50)",        "bot_resnet50"),
]


def wait_for(predicate, timeout: float, what: str) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return
        time.sleep(POLL_SEC)
    raise TimeoutError(f"timeout menunggu {what}")


def run_one(idx: int, detector_model: str, tracker_type: str, reid_model: str) -> dict:
    env = os.environ.copy()
    env["DETECTOR_MODEL"]    = detector_model
    env["TRACKER_TYPE"]     = tracker_type
    env["REID_MODEL"]       = reid_model
    env["DISABLE_AUTOSTART"] = "1"

    log_path = LOG_DIR / f"server_{idx}.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ["uvicorn", "app.main:app", "--port", str(PORT)],
            env=env, stdout=logf, stderr=subprocess.STDOUT,
        )

    try:
        def models_ready() -> bool:
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=3)
                return r.ok and r.json().get("models_loaded")
            except Exception:
                return False

        wait_for(models_ready, MODEL_LOAD_TIMEOUT_SEC, "model selesai load")

        try:
            requests.post(f"{BASE_URL}/stream/stop", headers=_AUTH_HEADERS, timeout=10)
        except Exception:
            pass
        time.sleep(2)

        stream_start_t = time.monotonic()
        # timeout tinggi: /stream/start me-load model Re-ID kedua kalinya (cache
        # StreamManager terpisah dari app.state), yang buat TransReID (ViT-B,
        # 86M param + first-run MPS kernel compile) bisa >15s.
        r = requests.post(f"{BASE_URL}/stream/start", json={"skip_gallery_restore": True},
                           headers=_AUTH_HEADERS, timeout=120)
        r.raise_for_status()

        # Sampel CPU/RAM selama stream jalan (dari /health, sumbernya psutil —
        # beban seluruh mesin, bukan cuma proses uvicorn ini, tapi cukup buat
        # bandingkan kombinasi karena cuma 1 stream evaluasi yang jalan tiap saat).
        samples: list[dict] = []

        def stream_done() -> bool:
            try:
                r = requests.get(f"{BASE_URL}/stream/status", headers=_AUTH_HEADERS, timeout=5)
                h = requests.get(f"{BASE_URL}/health", headers=_AUTH_HEADERS, timeout=5)
                if h.ok:
                    samples.append(h.json())
                return r.ok and not r.json().get("running", True)
            except Exception:
                return False

        wait_for(stream_done, STREAM_DONE_TIMEOUT_SEC, "stream selesai proses semua klip")
        wall_clock_sec = time.monotonic() - stream_start_t
        time.sleep(2)   # jeda kecil biar predictions.csv sempat ke-flush penuh
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    cpu_peak = max((s["cpu_percent"] for s in samples if s.get("cpu_percent") is not None), default=None)
    ram_peak = max((s["ram_percent"] for s in samples if s.get("ram_percent") is not None), default=None)
    # FPS end-to-end: total frame video / detik nyata dari /stream/start sampai
    # selesai — mencakup deteksi + tracking + Re-ID + posting, bukan cuma deteksi.
    fps = TOTAL_FRAMES / wall_clock_sec if wall_clock_sec > 0 else None

    detector_name = Path(detector_model).stem   # buang folder (checkpoints/) & ekstensi .pt
    slug = f"{detector_name}-{tracker_type}-{reid_model}"
    pred_dst = PRED_CSV_DIR / f"predictions_{idx:02d}_{slug}.csv"
    shutil.copy("predictions.csv", pred_dst)

    lat_src = Path("latency.csv")
    lat_avg = {}
    if lat_src.exists():
        lat_dst = LATENCY_DIR / f"latency_{idx:02d}_{slug}.csv"
        shutil.copy(lat_src, lat_dst)
        lat_avg = avg_latency(lat_dst)

    evaluate.GT_PATH   = GT_PATH
    evaluate.PRED_PATH = pred_dst
    buf = io.StringIO()
    with redirect_stdout(buf):
        evaluate.main()
    return {
        "pred_path": str(pred_dst), "stdout": buf.getvalue(),
        "cpu_peak": f"{cpu_peak:.0f}%" if cpu_peak is not None else "",
        "ram_peak": f"{ram_peak:.0f}%" if ram_peak is not None else "",
        "fps":      f"{fps:.2f}" if fps is not None else "",
        **lat_avg,
    }


def avg_latency(path: Path) -> dict:
    """Rata-rata tiap kolom ms di latency_NN_*.csv (instrumentasi per-frame
    dari batch_processor.py) — buat kolom breakdown latency di tabel."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    cols = ["decode_ms", "detection_ms", "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"]
    out = {}
    for col in cols:
        vals = [float(r[col]) for r in rows if r.get(col)]
        out[col] = f"{sum(vals) / len(vals):.1f}" if vals else ""
    return out


def parse_metrics(stdout: str) -> dict:
    """Ekstrak semua kolom yang dibutuhkan tabel pipeline-comparison.md dari
    stdout evaluate.py — cakupan/false_merge/false_split/precision/recall/f1
    (regex garis lurus) + identitas/akurasi (butuh 2 pola tambahan, dulu cuma
    ada di parse_metrics_copy() sebelum di-merge ke sini)."""
    m = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if line.startswith("Box GT tercocokkan ke prediksi"):
            m["cakupan"] = line.split(":")[-1].strip()
        elif line.startswith("False merge"):
            m["false_merge"] = line.split(":")[1].strip().split()[0]
        elif line.startswith("False split"):
            m["false_split"] = line.split(":")[1].strip().split()[0]
        elif line.startswith("Precision="):
            for part in line.split():
                k, v = part.split("=")
                m[k.lower()] = v
        else:
            id_m = re.match(r"(\d+) identitas sistem dihasilkan untuk (\d+) orang GT", line)
            if id_m:
                m["identitas"] = f"{id_m.group(1)}/{id_m.group(2)}"
            ak_m = re.match(r"Akurasi\s*:\s*(\d+/\d+).*\(([\d.]+%)\)", line)
            if ak_m:
                m["akurasi"] = f"{ak_m.group(1)} ({ak_m.group(2)})"
    return m


# Index kolom tabel pipeline-comparison.md (0-based split "|") — cocokin lagi
# kalau header tabelnya berubah lagi.
COL_MAP = {
    "status": 5, "cakupan": 6, "identitas": 7, "precision": 8, "recall": 9,
    "f1": 10, "akurasi": 11, "false_merge": 12, "false_split": 13,
    "fps": 14, "cpu_peak": 15, "ram_peak": 16,
    # Rata-rata per-frame dari latency.csv (lihat latency_logger.py) — kosong
    # buat kombinasi lama yang belum di-rerun setelah instrumentasi ini ada.
    "decode_ms": 17, "detection_ms": 18, "tracking_ms": 19,
    "reid_ms": 20, "matching_ms": 21, "total_ai_ms": 22,
}


def update_markdown(num: int, metrics: dict) -> None:
    lines = MD_PATH.read_text().splitlines()
    for i, line in enumerate(lines):
        if not re.match(rf"^\|\s*{num}\s*\|", line):
            continue
        cols = line.split("|")
        if len(cols) < len(COL_MAP) + 2:
            continue
        for key, idx in COL_MAP.items():
            cols[idx] = f" {metrics.get(key, '')} "
        lines[i] = "|".join(cols)
        break
    MD_PATH.write_text("\n".join(lines) + "\n")


def run_combo(num: int, results: list[dict]) -> list[dict]:
    combo = next((c for c in COMBOS if c[0] == num), None)
    if combo is None:
        print(f"  !! #{num} tidak ada di COMBOS, skip")
        return results
    _, det_label, detector_model, trk_label, tracker_type, reid_label, reid_model = combo
    print(f"\n{'='*70}\n#{num}: {det_label} + {trk_label} + {reid_label}\n{'='*70}")
    results = [r for r in results if r["num"] != num]   # buang hasil lama (mis. status Gagal) kalau ada
    try:
        res = run_one(num, detector_model, tracker_type, reid_model)
        metrics = parse_metrics(res["stdout"])
        for key in ("fps", "cpu_peak", "ram_peak", "decode_ms", "detection_ms",
                    "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"):
            metrics[key] = res.get(key, "")
        metrics["status"] = "Selesai"
        print(f"  -> {metrics}")
        results.append({"num": num, "detektor": det_label, "tracker": trk_label,
                         "reid": reid_label, **metrics})
    except Exception as exc:
        print(f"  !! GAGAL: {exc}")
        metrics = {"status": f"Gagal: {exc}"}
        results.append({"num": num, "detektor": det_label, "tracker": trk_label,
                         "reid": reid_label, **metrics})
    update_markdown(num, metrics)
    return results


def main() -> None:
    PRED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATENCY_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"
    results: list[dict] = json.loads(results_path.read_text()) if results_path.exists() else []

    requested = [int(a) for a in sys.argv[1:]]
    if requested:
        # Nomor eksplisit di argumen: SELALU dijalankan ulang (biar bisa
        # dipakai buat re-run kombinasi yang mau diperbaiki, mis. checkpoint
        # baru), gak peduli status "Selesai" sebelumnya.
        todo = requested
    else:
        # Tanpa argumen: mode batch penuh, resume-capable — skip yang udah
        # "Selesai" DAN sudah punya kolom fps (run lama sebelum kolom itu
        # ditambahkan tetap dianggap belum tuntas, diulang buat lengkapin).
        done_nums = {r["num"] for r in results if r.get("status") == "Selesai" and r.get("fps")}
        if done_nums:
            print(f"Resume: {len(done_nums)} kombinasi udah 'Selesai' sebelumnya, di-skip: {sorted(done_nums)}")
        todo = [c[0] for c in COMBOS if c[0] not in done_nums]

    for num in todo:
        results = run_combo(num, results)
        results_path.write_text(json.dumps(results, indent=2))

    print(f"\n{len(todo)} kombinasi selesai diproses. Hasil ditulis ke {MD_PATH}")


if __name__ == "__main__":
    main()
