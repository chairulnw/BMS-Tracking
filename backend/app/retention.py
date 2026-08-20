"""DB retention untuk detections/tracklets (plan2/spesifikasi.md Fase 4).
File clip/thumbnail sudah punya retensi sendiri (ai-service/app/services/
retention.py) — ini versi baris database, supaya detections/tracklets gak
tumbuh selamanya. Tidak menyentuh persons/camera_events (di luar scope item
ini)."""

import asyncio
import os

import asyncpg

DB_RETENTION_DAYS      = float(os.getenv("DB_RETENTION_DAYS", "90"))
DB_CLEANUP_INTERVAL_HOURS = float(os.getenv("DB_CLEANUP_INTERVAL_HOURS", "24"))


async def cleanup_old_rows(pool: asyncpg.Pool, retention_days: float = DB_RETENTION_DAYS) -> dict:
    """Hapus detections & tracklets lebih tua dari `retention_days`. Tidak ada
    FK yang cascade-delete gara-gara ini: `detections.tracklet_id` ON DELETE
    SET NULL, tidak ada tabel lain yang FK ke detections.id."""
    async with pool.acquire() as conn:
        det = await conn.fetchval(
            """
            WITH deleted AS (
                DELETE FROM detections
                 WHERE timestamp < NOW() - make_interval(days => $1::int)
                RETURNING 1
            )
            SELECT COUNT(*) FROM deleted
            """,
            retention_days,
        )
        trk = await conn.fetchval(
            """
            WITH deleted AS (
                DELETE FROM tracklets
                 WHERE ended_at < NOW() - make_interval(days => $1::int)
                RETURNING 1
            )
            SELECT COUNT(*) FROM deleted
            """,
            retention_days,
        )
    if det or trk:
        print(f"[db-retention] {det} detections, {trk} tracklets dihapus (retensi {retention_days:.0f}d)")
    return {"detections_removed": int(det or 0), "tracklets_removed": int(trk or 0)}


async def start_background(pool: asyncpg.Pool) -> asyncio.Task:
    async def _loop():
        while True:
            try:
                await cleanup_old_rows(pool)
            except Exception as exc:
                print(f"[db-retention] error: {exc}")
            await asyncio.sleep(DB_CLEANUP_INTERVAL_HOURS * 3600)

    return asyncio.create_task(_loop())
