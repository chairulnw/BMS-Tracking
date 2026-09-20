"""Rangkum laporan kapasitas sumber daya (CPU/RAM/FPS) dari resources.csv yang
udah di-generate ResourceSampler selama satu sesi stream/benchmark jalan.

Pakai:
    python benchmark/capacity_report.py resources.csv
    python benchmark/capacity_report.py resources.csv --log predictions_runs/logs/server_30.log
"""
import argparse
import csv
import re
import statistics as st
from pathlib import Path

# Sama dengan resource_sampler.SAMPLE_INTERVAL_SEC — dihardcode di sini biar
# gak perlu import app.* penuh (butuh env var SECRET_KEY dkk) cuma buat 1 angka.
SAMPLE_INTERVAL_SEC = 2.0


def summarize(csv_path: Path) -> None:
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print("resources.csv kosong")
        return

    cpu  = [float(r["cpu_percent"]) for r in rows if r["cpu_percent"]]
    ram  = [float(r["ram_percent"]) for r in rows if r["ram_percent"]]
    fps  = [float(r["fps_effective"]) for r in rows if r["fps_effective"]]
    gpu  = [float(r["gpu_util"]) for r in rows if r.get("gpu_util")]
    vram = [float(r["vram_mb"]) for r in rows if r.get("vram_mb")]

    print(f"Sampel: {len(rows)} baris (interval {SAMPLE_INTERVAL_SEC:.0f}s)")
    print(f"CPU   avg={st.mean(cpu):.1f}%  peak={max(cpu):.1f}%")
    print(f"RAM   avg={st.mean(ram):.1f}%  peak={max(ram):.1f}%")
    print(f"FPS diproses  avg={st.mean(fps):.2f}")
    if gpu:
        print(f"GPU   avg={st.mean(gpu):.1f}%  peak={max(gpu):.1f}%")
        print(f"VRAM  avg={st.mean(vram):.0f}MB  peak={max(vram):.0f}MB")
    else:
        print("GPU/VRAM: kosong (CUDA tidak tersedia — normal di Apple Silicon/MPS)")


def fps_input(log_path: Path) -> None:
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"\[(\w+)\] opened \d+x\d+ @ ([\d.]+)fps", text)
    if not matches:
        print("Tidak ada baris 'opened ... @ Xfps' di log ini")
        return
    print("\nFPS input (laju asli kamera/klip, dari log):")
    for cam, fps in matches:
        print(f"  {cam}: {fps} fps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("resources_csv", type=Path)
    ap.add_argument("--log", type=Path, default=None, help="server log — buat ekstrak FPS input tiap kamera")
    args = ap.parse_args()

    summarize(args.resources_csv)
    if args.log:
        fps_input(args.log)

    print("""
Network bandwidth (belum diinstrumentasi di kode — ukur manual, jalankan
sebelum & sesudah sesi lalu selisihkan Ibytes/Obytes):
    netstat -ib | grep en0
""")
