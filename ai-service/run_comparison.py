"""Orkestrasi otomatis: jalanin AI Service per kombinasi (deteksi x tracker x
Re-ID), proses seluruh klip sample copy/ lewat file-playlist, simpan
predictions.csv per kombinasi (nama beda-beda), evaluasi ke sample
copy/output.csv, tulis hasilnya ke pipeline-comparison.md.

Restart proses uvicorn penuh dibutuhkan tiap ganti kombinasi karena
YOLO_MODEL/REID_MODEL/TRACKER_TYPE dibaca sebagai konstanta modul saat
proses start (lihat main.py/batch_processor.py) — tidak bisa diganti lewat
API di proses yang sama.

Jalankan: python run_comparison.py
"""

import io
import json
import os
import re
import shutil
import subprocess
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
PRED_DIR       = Path("predictions_runs")
GT_PATH        = Path("sample copy/output.csv")
MD_PATH        = Path("pipeline-comparison.md")
MODEL_LOAD_TIMEOUT_SEC   = 300
STREAM_DONE_TIMEOUT_SEC  = 1800
POLL_SEC       = 3
# Total frame di seluruh 31 klip sample copy/ (dihitung sekali, lihat komentar
# di bawah) — dipakai buat FPS end-to-end (total frame / detik wall-clock),
# BUKAN dari avg_batch_ms di /health, yang cuma ngukur waktu YOLO predict()
# doang, tidak termasuk tracking + ekstraksi embedding Re-ID yang justru jadi
# bottleneck sebenarnya (lihat catatan OSNet di CLAUDE.md/plan/06-decisions.md).
TOTAL_FRAMES = 2518

# (nomor tabel di pipeline-comparison.md, Detektor label, YOLO_MODEL,
#  Tracker label, TRACKER_TYPE, Re-ID label, REID_MODEL) — 18 kombinasi
# (27 penuh dikurangi 9 yang pakai TransReID, belum diwire).
COMBOS = [
    (1,  "YOLO26n",    "yolo26n.pt",  "ByteTrack", "bytetrack", "OSNet",          "osnet_ain_x1_0"),
    (3,  "YOLO26n",    "yolo26n.pt",  "ByteTrack", "bytetrack", "BoT (ResNet50)", "resnet50"),
    (4,  "YOLO26n",    "yolo26n.pt",  "BoT-SORT",  "botsort",   "OSNet",          "osnet_ain_x1_0"),
    (6,  "YOLO26n",    "yolo26n.pt",  "BoT-SORT",  "botsort",   "BoT (ResNet50)", "resnet50"),
    (7,  "YOLO26n",    "yolo26n.pt",  "OC-SORT",   "ocsort",    "OSNet",          "osnet_ain_x1_0"),
    (9,  "YOLO26n",    "yolo26n.pt",  "OC-SORT",   "ocsort",    "BoT (ResNet50)", "resnet50"),
    (10, "YOLO11n",    "yolo11n.pt",  "ByteTrack", "bytetrack", "OSNet",          "osnet_ain_x1_0"),
    (12, "YOLO11n",    "yolo11n.pt",  "ByteTrack", "bytetrack", "BoT (ResNet50)", "resnet50"),
    (13, "YOLO11n",    "yolo11n.pt",  "BoT-SORT",  "botsort",   "OSNet",          "osnet_ain_x1_0"),
    (15, "YOLO11n",    "yolo11n.pt",  "BoT-SORT",  "botsort",   "BoT (ResNet50)", "resnet50"),
    (16, "YOLO11n",    "yolo11n.pt",  "OC-SORT",   "ocsort",    "OSNet",          "osnet_ain_x1_0"),
    (18, "YOLO11n",    "yolo11n.pt",  "OC-SORT",   "ocsort",    "BoT (ResNet50)", "resnet50"),
    (19, "RTDETRv2-s", "rtdetr-l.pt", "ByteTrack", "bytetrack", "OSNet",          "osnet_ain_x1_0"),
    (21, "RTDETRv2-s", "rtdetr-l.pt", "ByteTrack", "bytetrack", "BoT (ResNet50)", "resnet50"),
    (22, "RTDETRv2-s", "rtdetr-l.pt", "BoT-SORT",  "botsort",   "OSNet",          "osnet_ain_x1_0"),
    (24, "RTDETRv2-s", "rtdetr-l.pt", "BoT-SORT",  "botsort",   "BoT (ResNet50)", "resnet50"),
    (25, "RTDETRv2-s", "rtdetr-l.pt", "OC-SORT",   "ocsort",    "OSNet",          "osnet_ain_x1_0"),
    (27, "RTDETRv2-s", "rtdetr-l.pt", "OC-SORT",   "ocsort",    "BoT (ResNet50)", "resnet50"),
]


def wait_for(predicate, timeout: float, what: str) -> None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return
        time.sleep(POLL_SEC)
    raise TimeoutError(f"timeout menunggu {what}")


def run_one(idx: int, yolo_model: str, tracker_type: str, reid_model: str) -> dict:
    env = os.environ.copy()
    env["YOLO_MODEL"]       = yolo_model
    env["TRACKER_TYPE"]     = tracker_type
    env["REID_MODEL"]       = reid_model
    env["DISABLE_AUTOSTART"] = "1"

    log_path = PRED_DIR / f"server_{idx}.log"
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
        r = requests.post(f"{BASE_URL}/stream/start", json={"skip_gallery_restore": True},
                           headers=_AUTH_HEADERS, timeout=15)
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

    slug = f"{yolo_model.replace('.pt', '')}-{tracker_type}-{reid_model}"
    pred_dst = PRED_DIR / f"predictions_{idx:02d}_{slug}.csv"
    shutil.copy("predictions.csv", pred_dst)

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
    }


def parse_metrics(stdout: str) -> dict:
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
    return m


def update_markdown(results: list[dict]) -> None:
    lines = MD_PATH.read_text().splitlines()
    by_num = {r["num"]: r for r in results}
    for i, line in enumerate(lines):
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        num = int(m.group(1))
        if num not in by_num:
            continue
        cols = line.split("|")
        if len(cols) < 17:
            continue
        r = by_num[num]
        cols[6]  = f" {r.get('status', '')} "
        cols[7]  = f" {r.get('precision', '')} "
        cols[8]  = f" {r.get('recall', '')} "
        cols[9]  = f" {r.get('f1', '')} "
        cols[10] = f" {r.get('false_merge', '')} "
        cols[11] = f" {r.get('false_split', '')} "
        cols[12] = f" {r.get('cakupan', '')} "
        cols[13] = f" {r.get('fps', '')} "
        cols[14] = f" {r.get('cpu_peak', '')} "
        cols[15] = f" {r.get('ram_peak', '')} "
        lines[i] = "|".join(cols)
    MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    PRED_DIR.mkdir(exist_ok=True)
    results_path = PRED_DIR / "results.json"
    results: list[dict] = json.loads(results_path.read_text()) if results_path.exists() else []
    # "Selesai" tapi belum ada fps (hasil dari run sebelum kolom FPS/CPU/RAM
    # ditambahkan) tetap dianggap belum tuntas, biar diulang buat lengkapin kolom itu.
    done_nums = {r["num"] for r in results if r.get("status") == "Selesai" and r.get("fps")}
    if done_nums:
        print(f"Resume: {len(done_nums)} kombinasi udah 'Selesai' sebelumnya, di-skip: {sorted(done_nums)}")

    for i, (num, det_label, yolo_model, trk_label, tracker_type, reid_label, reid_model) in enumerate(COMBOS, 1):
        if num in done_nums:
            continue
        print(f"\n{'='*70}\n[{i}/{len(COMBOS)}] #{num}: {det_label} + {trk_label} + {reid_label}\n{'='*70}")
        results = [r for r in results if r["num"] != num]   # buang hasil lama (mis. status Gagal) kalau ada
        try:
            res = run_one(num, yolo_model, tracker_type, reid_model)
            metrics = parse_metrics(res["stdout"])
            metrics["fps"], metrics["cpu_peak"], metrics["ram_peak"] = res["fps"], res["cpu_peak"], res["ram_peak"]
            print(f"  -> {metrics}")
            results.append({"num": num, "detektor": det_label, "tracker": trk_label,
                             "reid": reid_label, **metrics, "status": "Selesai"})
        except Exception as exc:
            print(f"  !! GAGAL: {exc}")
            results.append({"num": num, "detektor": det_label, "tracker": trk_label,
                             "reid": reid_label, "status": f"Gagal: {exc}"})
        results_path.write_text(json.dumps(results, indent=2))
        update_markdown(results)

    print("\nSemua kombinasi selesai. Hasil ditulis ke pipeline-comparison.md")


if __name__ == "__main__":
    main()
