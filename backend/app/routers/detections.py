from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app.schemas import DetectionCreate, DetectionResponse, ThumbnailUpdate

router = APIRouter(prefix="/detections", tags=["detections"])


@router.post("", response_model=DetectionResponse, status_code=201)
async def create_detection(
    req: DetectionCreate, request: Request
) -> DetectionResponse:
    pool = request.app.state.pool
    now  = req.timestamp or datetime.now(timezone.utc)
    label = (req.person_label or req.person_name).strip()
    name  = req.person_name.strip()

    async with pool.acquire() as conn:
        async with conn.transaction():
            person = await conn.fetchrow(
                "SELECT * FROM persons WHERE label = $1 OR name = $1 LIMIT 1",
                label,
            )
            if person is None and label != name:
                person = await conn.fetchrow(
                    "SELECT * FROM persons WHERE name = $1 AND is_known = true LIMIT 1", name
                )

            if person is None:
                thumbnail = req.thumbnail_url
                person = await conn.fetchrow(
                    """
                    INSERT INTO persons
                        (name, label, first_seen, last_seen, last_camera,
                         best_thumbnail_url, enrollment_date)
                    VALUES ($1, $2, $3, $3, $4, $5, $6)
                    RETURNING *
                    """,
                    name, label, now, req.camera_id, thumbnail, now.date(),
                )
            else:
                # GREATEST/CASE guard: tracklet close order via background
                # post-queue tidak menjamin urutan waktu, jadi last_seen tidak
                # boleh langsung ditimpa POST paling akhir.
                person = await conn.fetchrow(
                    """
                    UPDATE persons
                    SET last_seen   = GREATEST(last_seen, $1),
                        last_camera = CASE WHEN $1 >= last_seen THEN $2 ELSE last_camera END,
                        best_thumbnail_url = COALESCE($3, best_thumbnail_url)
                    WHERE id = $4
                    RETURNING *
                    """,
                    now, req.camera_id, req.thumbnail_url, person["id"],
                )

            detection = await conn.fetchrow(
                """
                INSERT INTO detections (person_id, camera_id, timestamp, confidence, method, thumbnail_url, track_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                """,
                person["id"], req.camera_id, now, req.confidence, req.method, req.thumbnail_url, req.track_id,
            )

    return dict(detection)


@router.patch("/thumbnail")
async def update_detection_thumbnail(req: ThumbnailUpdate, request: Request) -> dict:
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        person = await conn.fetchrow(
            "SELECT id FROM persons WHERE label = $1", req.person_label
        )
        if not person:
            return {"updated": False}
        # Update thumbnail di baris deteksi terbaru yang belum punya thumbnail
        await conn.execute(
            """
            UPDATE detections SET thumbnail_url = $1
            WHERE id = (
                SELECT id FROM detections
                WHERE person_id  = $2
                  AND camera_id  = $3
                  AND thumbnail_url IS NULL
                ORDER BY timestamp DESC
                LIMIT 1
            )
            """,
            req.thumbnail_url, person["id"], req.camera_id,
        )
        # Update thumbnail person juga
        await conn.execute(
            "UPDATE persons SET best_thumbnail_url = COALESCE(best_thumbnail_url, $1) WHERE id = $2",
            req.thumbnail_url, person["id"],
        )
    return {"updated": True}
