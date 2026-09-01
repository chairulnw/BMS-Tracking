"""Instrumentasi evaluasi — cuma nyatet waktu, TIDAK mengubah algoritma
deteksi/tracking/Re-ID apa pun. Satu baris per kamera per siklus batch,
ditulis ke latency.csv (sejajar predictions.csv di root ai-service/).

total_ai_ms = jumlah 5 kolom lain (decode..matching), BUKAN satu span
wall-clock tunggal — decode_ms diukur di thread capture yang jalan konkuren
sama batch loop, jadi gak ada satu "jam" yang mencakup keduanya sekaligus.
Ini penjumlahan logis biaya tiap tahap, cukup buat bandingin kombinasi
komponen (tujuan instrumentasi ini), bukan pengukuran latensi end-to-end
presisi nanodetik.

detection_ms dicatat SAMA untuk semua kamera dalam satu siklus batch, karena
YOLO/RTDETR diproses sekali buat semua kamera aktif sekaligus (1 GPU call,
lihat batch_processor.py) — bukan per-kamera murni, ini bagian yang dibagi.
"""

import csv
from pathlib import Path

LATENCY_CSV = Path("latency.csv")

_HEADER = ["timestamp", "camera_id", "frame_id", "decode_ms", "detection_ms",
           "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"]


class _LatencyLogger:
    def __init__(self, path: Path = LATENCY_CSV) -> None:
        self._path   = path
        self._file   = None
        self._writer = None

    def log(self, timestamp: str, camera_id: str, frame_id: int,
             decode_ms: float, detection_ms: float, tracking_ms: float,
             reid_ms: float, matching_ms: float) -> None:
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file   = open(self._path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(_HEADER)
        total_ms = decode_ms + detection_ms + tracking_ms + reid_ms + matching_ms
        self._writer.writerow([
            timestamp, camera_id, frame_id,
            f"{decode_ms:.3f}", f"{detection_ms:.3f}", f"{tracking_ms:.3f}",
            f"{reid_ms:.3f}", f"{matching_ms:.3f}", f"{total_ms:.3f}",
        ])
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file   = None
            self._writer = None
