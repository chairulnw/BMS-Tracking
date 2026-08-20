"""ponytail self-check untuk app/retention.py — jalan langsung ke DB dev
(DATABASE_URL), bikin baris dummy super-tua, pastikan cleanup_old_rows()
menghapusnya dan baris baru selamat. Bersih-bersih sendiri di finally."""

import asyncio
import os

import asyncpg
from dotenv import load_dotenv

load_dotenv()

from app.retention import cleanup_old_rows


async def main() -> None:
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=1)
    person_id = None
    try:
        person_id = await pool.fetchval(
            "INSERT INTO persons (label, name) VALUES ('retention-selfcheck', 'retention-selfcheck') RETURNING id"
        )
        old_id = await pool.fetchval(
            """
            INSERT INTO detections (person_id, camera_id, timestamp, confidence, method)
            VALUES ($1, 'selfcheck', NOW() - interval '200 days', 1.0, 'unknown')
            RETURNING id
            """,
            person_id,
        )
        new_id = await pool.fetchval(
            """
            INSERT INTO detections (person_id, camera_id, timestamp, confidence, method)
            VALUES ($1, 'selfcheck', NOW(), 1.0, 'unknown')
            RETURNING id
            """,
            person_id,
        )

        result = await cleanup_old_rows(pool, retention_days=90)
        assert result["detections_removed"] >= 1, f"expected >=1 removed, got {result}"

        old_exists = await pool.fetchval("SELECT 1 FROM detections WHERE id = $1", old_id)
        new_exists = await pool.fetchval("SELECT 1 FROM detections WHERE id = $1", new_id)
        assert old_exists is None, "detection tua harusnya kehapus"
        assert new_exists == 1, "detection baru harusnya selamat"
        print("db-retention self-check OK")
    finally:
        if person_id is not None:
            await pool.execute("DELETE FROM persons WHERE id = $1", person_id)
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
