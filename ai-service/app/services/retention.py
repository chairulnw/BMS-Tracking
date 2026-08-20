"""Fase 1 plan2/spesifikasi.md: hapus clip/thumbnail lama + guard kapasitas.
Dijalankan sebagai background thread dari lifespan (app/main.py)."""

import os
import threading
import time
from pathlib import Path

CLIPS_DIR      = Path("output/clips")
THUMBNAILS_DIR = Path("thumbnails")  # termasuk thumbnails/events/ (rglob)

RETENTION_DAYS         = float(os.getenv("RETENTION_DAYS", "30"))
# ponytail: cap ukuran folder project sendiri (GB), BUKAN persentase disk
# seluruh sistem — disk Mac/server bisa 90%+ penuh gara-gara hal lain sama
# sekali (OS, app lain) yang gak ada hubungannya sama clip/thumbnail di sini.
# Guard berbasis persen-disk-seluruh-sistem pernah kejadian nyata: disk 93%
# (padahal project cuma raih beberapa ratus MB), guard nyoba turunin ke 90%
# dan gak akan PERNAH berhasil cuma dari folder ini — jadi dia hapus TERUS,
# termasuk file yang baru dibuat beberapa menit lalu.
MAX_STORAGE_GB         = float(os.getenv("MAX_STORAGE_GB", "5"))
CLEANUP_INTERVAL_HOURS = float(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))


def _files_by_age(dirs: list[Path]) -> list[Path]:
    """Semua file di `dirs`, terlama duluan."""
    files = [p for d in dirs if d.exists() for p in d.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def cleanup_once(
    dirs:            list[Path] | None = None,
    retention_days:  float             = RETENTION_DAYS,
    max_storage_gb:  float             = MAX_STORAGE_GB,
) -> int:
    """Hapus file lebih tua dari `retention_days`, lalu — kalau total ukuran
    `dirs` masih di atas `max_storage_gb` — hapus file terlama sampai di
    bawah batas. Return jumlah file yang dihapus."""
    dirs = dirs if dirs is not None else [CLIPS_DIR, THUMBNAILS_DIR]
    removed = 0
    now = time.time()
    max_age_s = retention_days * 86400

    for f in _files_by_age(dirs):
        if now - f.stat().st_mtime > max_age_s:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass

    max_bytes = max_storage_gb * 1024**3
    remaining = _files_by_age(dirs)
    total_bytes = sum(f.stat().st_size for f in remaining)
    if total_bytes > max_bytes:
        print(f"[retention] folder {total_bytes / 1024**3:.2f}GB > cap {max_storage_gb}GB — hapus file terlama")
        for f in remaining:
            if total_bytes <= max_bytes:
                break
            try:
                size = f.stat().st_size
                f.unlink()
            except OSError:
                continue
            total_bytes -= size
            removed += 1

    if removed:
        print(f"[retention] {removed} file dihapus (retensi {retention_days}d, cap {max_storage_gb}GB)")
    return removed


def start_background(stop_event: threading.Event) -> threading.Thread:
    def _loop():
        while not stop_event.is_set():
            try:
                cleanup_once()
            except Exception as exc:
                print(f"[retention] error: {exc}")
            stop_event.wait(CLEANUP_INTERVAL_HOURS * 3600)

    t = threading.Thread(target=_loop, daemon=True, name="retention")
    t.start()
    return t


def _demo() -> None:
    """ponytail self-check: file tua & cap-kapasitas beneran kehapus, file baru dalam batas selamat."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = d / "old.avi"
        new = d / "new.avi"
        old.write_bytes(b"x" * 1000)
        new.write_bytes(b"x" * 1000)
        old_time = time.time() - 20 * 86400  # 20 hari lalu
        os.utime(old, (old_time, old_time))

        removed = cleanup_once([d], retention_days=14, max_storage_gb=100)
        assert removed == 1, f"expected 1 file removed, got {removed}"
        assert not old.exists(), "file tua harusnya kehapus"
        assert new.exists(), "file baru harusnya selamat"

        # cap kapasitas: dua file baru, cap sangat kecil → file terlama diantaranya kehapus
        d2 = Path(tmp) / "cap"
        d2.mkdir()
        a, b = d2 / "a.avi", d2 / "b.avi"
        a.write_bytes(b"x" * 2000)
        os.utime(a, (time.time() - 10, time.time() - 10))
        b.write_bytes(b"x" * 2000)
        removed2 = cleanup_once([d2], retention_days=14, max_storage_gb=2000 / 1024**3)
        assert removed2 == 1, f"expected 1 file removed by cap, got {removed2}"
        assert not a.exists() and b.exists(), "file terlama harusnya kehapus, yang baru selamat"
    print("retention self-check OK")


if __name__ == "__main__":
    _demo()
