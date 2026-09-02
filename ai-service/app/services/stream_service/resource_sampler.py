"""Instrumentasi evaluasi — sampler resource (CPU/RAM/FPS, GPU kalau ada)
jalan di background thread selama stream aktif. Cuma nyatet, gak ngubah
algoritma apa pun.

CPU/RAM reuse psutil call yang sama persis dipakai `/health`
(app/routers/health.py) — bukan duplikasi logic baru.
"""

import csv
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil
import torch

RESOURCE_CSV = Path("resources.csv")
SAMPLE_INTERVAL_SEC = 2.0
_WIB = ZoneInfo("Asia/Jakarta")   # cuma buat timestamp CSV ini, gampang dibaca manual

_HEADER = ["timestamp", "cpu_percent", "ram_percent", "cpu_peak", "ram_peak",
           "fps_effective", "gpu_util", "vram_mb"]


class ResourceSampler:
    def __init__(self, get_total_frames, stop_event: threading.Event,
                 path: Path = RESOURCE_CSV, interval: float = SAMPLE_INTERVAL_SEC) -> None:
        self._get_total_frames = get_total_frames   # callable -> int, total frame semua kamera
        self._stop     = stop_event
        self._path     = path
        self._interval = interval
        self._thread: "threading.Thread | None" = None
        self._cpu_peak = 0.0
        self._ram_peak = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="resource-sampler")
        self._thread.start()

    def _loop(self) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_HEADER)
            f.flush()
            prev_frames = self._get_total_frames()
            while not self._stop.wait(self._interval):
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                self._cpu_peak = max(self._cpu_peak, cpu)
                self._ram_peak = max(self._ram_peak, ram)

                total_frames = self._get_total_frames()
                fps = (total_frames - prev_frames) / self._interval
                prev_frames = total_frames

                gpu_util, vram_mb = self._gpu_metrics()

                writer.writerow([
                    datetime.now(_WIB).isoformat(),
                    f"{cpu:.1f}", f"{ram:.1f}", f"{self._cpu_peak:.1f}", f"{self._ram_peak:.1f}",
                    f"{fps:.2f}", gpu_util if gpu_util is not None else "",
                    vram_mb if vram_mb is not None else "",
                ])
                f.flush()

    @staticmethod
    def _gpu_metrics() -> tuple["float | None", "float | None"]:
        # Cuma CUDA yang punya API utilization/VRAM yang reliable lewat torch.
        # MPS (Mac) gak ada API setara tanpa `sudo powermetrics` — dikosongin
        # daripada dipaksa isi angka yang gak akurat (lihat plan instrumentasi).
        if not torch.cuda.is_available():
            return None, None
        try:
            util = float(torch.cuda.utilization())
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            return util, vram_mb
        except Exception:
            return None, None
