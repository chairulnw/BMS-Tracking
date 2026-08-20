"""Fase 1 plan2/01-roadmap.md: hapus clip/thumbnail lama + guard disk penuh.
Dijalankan sebagai background thread dari lifespan (app/main.py)."""

import os
import shutil
import threading
import time
from pathlib import Path

CLIPS_DIR      = Path("output/clips")
THUMBNAILS_DIR = Path("thumbnails")  # termasuk thumbnails/events/ (rglob)

RETENTION_DAYS         = float(os.getenv("RETENTION_DAYS", "14"))
DISK_GUARD_PCT         = float(os.getenv("DISK_GUARD_PCT", "90"))
CLEANUP_INTERVAL_HOURS = float(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))


def _files_by_age(dirs: list[Path]) -> list[Path]:
    """Semua file di `dirs`, terlama duluan."""
    files = [p for d in dirs if d.exists() for p in d.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def cleanup_once(
    dirs:           list[Path] | None = None,
    retention_days: float             = RETENTION_DAYS,
    disk_guard_pct: float             = DISK_GUARD_PCT,
) -> int:
    """Hapus file lebih tua dari `retention_days`, lalu — kalau disk masih di
    atas `disk_guard_pct` — hapus file terlama sampai di bawah threshold.
    Return jumlah file yang dihapus."""
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

    probe_dir = next((d for d in dirs if d.exists()), None)
    if probe_dir is not None:
        usage = shutil.disk_usage(probe_dir)
        pct_used = usage.used / usage.total * 100
        if pct_used > disk_guard_pct:
            print(f"[retention] disk {pct_used:.1f}% > guard {disk_guard_pct}% — hapus file terlama")
            for f in _files_by_age(dirs):
                try:
                    size = f.stat().st_size
                    f.unlink()
                    removed += 1
                except OSError:
                    continue
                usage = shutil.disk_usage(probe_dir)
                pct_used = usage.used / usage.total * 100
                if pct_used <= disk_guard_pct:
                    break

    if removed:
        print(f"[retention] {removed} file dihapus (retensi {retention_days}d, guard {disk_guard_pct}%)")
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
    """ponytail self-check: file tua & disk-guard beneran kehapus, file baru selamat."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        old = d / "old.avi"
        new = d / "new.avi"
        old.write_bytes(b"x" * 1000)
        new.write_bytes(b"x" * 1000)
        old_time = time.time() - 20 * 86400  # 20 hari lalu
        os.utime(old, (old_time, old_time))

        removed = cleanup_once([d], retention_days=14, disk_guard_pct=100)
        assert removed == 1, f"expected 1 file removed, got {removed}"
        assert not old.exists(), "file tua harusnya kehapus"
        assert new.exists(), "file baru harusnya selamat"
    print("retention self-check OK")


if __name__ == "__main__":
    _demo()
