from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import MovementRecord, PersonDetail, PersonResponse, RenameRequest

router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("", response_model=list[PersonResponse])
async def list_persons(
    request: Request,
    date_filter: date | None = Query(None, alias="date"),
) -> list[PersonResponse]:
    pool = request.app.state.pool
    if date_filter is None:
        from zoneinfo import ZoneInfo
        date_filter = datetime.now(ZoneInfo("Asia/Jakarta")).date()

    rows = await pool.fetch(
        """
        SELECT p.* FROM persons p
        WHERE EXISTS (
            SELECT 1 FROM detections d
            WHERE d.person_id = p.id
              AND (d.timestamp AT TIME ZONE 'Asia/Jakarta')::date = $1
        )
        ORDER BY p.last_seen DESC NULLS LAST
        """,
        date_filter,
    )
    return [dict(r) for r in rows]


@router.get("/{person_id}", response_model=PersonDetail)
async def get_person(person_id: int, request: Request) -> PersonDetail:
    pool = request.app.state.pool
    row = await pool.fetchrow("SELECT * FROM persons WHERE id = $1", person_id)
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")

    detections = await pool.fetch(
        "SELECT id, camera_id, timestamp, confidence, method "
        "FROM detections WHERE person_id = $1 ORDER BY timestamp DESC LIMIT 200",
        person_id,
    )
    person = dict(row)
    person["detections"] = [dict(d) for d in detections]
    return person


@router.get("/{person_id}/movements", response_model=list[MovementRecord])
async def get_movements(
    person_id: int,
    request: Request,
    date_filter: date | None = Query(None, alias="date"),
) -> list[MovementRecord]:
    pool = request.app.state.pool

    exists = await pool.fetchval("SELECT id FROM persons WHERE id = $1", person_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Person not found")

    if date_filter is None:
        from zoneinfo import ZoneInfo
        date_filter = datetime.now(ZoneInfo("Asia/Jakarta")).date()

    # Detections — deduplicated consecutive same-camera runs, no crossing data mixed in
    det_rows = await pool.fetch(
        """
        SELECT
            d.camera_id,
            d.timestamp,
            c.name                       AS cam_name,
            cz.room_name,
            COALESCE(cz.floor, c.floor)  AS db_floor,
            cz.camera_id IS NOT NULL     AS has_zone,
            d.thumbnail_url              AS snapshot_url,
            d.track_id
        FROM detections d
        LEFT JOIN camera_zones cz ON cz.camera_id = d.camera_id
        LEFT JOIN cameras c       ON c.camera_id  = d.camera_id
        WHERE d.person_id = $1
          AND (d.timestamp AT TIME ZONE 'Asia/Jakarta')::date = $2
        ORDER BY d.timestamp DESC
        LIMIT 200
        """,
        person_id,
        date_filter,
    )

    # Crossing events — each is always its own row
    cross_rows = await pool.fetch(
        """
        SELECT
            oe.camera_id,
            oe.timestamp,
            c.name                       AS cam_name,
            cz.room_name,
            COALESCE(cz.floor, c.floor)  AS db_floor,
            cz.camera_id IS NOT NULL     AS has_zone,
            oe.direction,
            oe.event_kind,
            oe.snapshot_url,
            oe.track_id
        FROM occupancy_events oe
        JOIN persons p        ON p.label     = oe.person_label
        LEFT JOIN camera_zones cz ON cz.camera_id = oe.camera_id
        LEFT JOIN cameras c       ON c.camera_id  = oe.camera_id
        WHERE p.id = $1
          AND (oe.timestamp AT TIME ZONE 'Asia/Jakarta')::date = $2
        ORDER BY oe.timestamp DESC
        LIMIT 100
        """,
        person_id,
        date_filter,
    )

    # Deduplicate consecutive same-camera detections
    deduped_det: list[dict] = []
    prev_cam = None
    for row in det_rows:
        if row["camera_id"] != prev_cam:
            deduped_det.append({"_type": "det", **dict(row)})
            prev_cam = row["camera_id"]

    # Tag crossing rows
    cross_list = [{"_type": "cross", **dict(r)} for r in cross_rows]

    # Merge and sort by timestamp desc
    merged = sorted(deduped_det + cross_list, key=lambda r: r["timestamp"], reverse=True)[:30]

    result = []
    for i, row in enumerate(merged):
        cam_name  = row["cam_name"] or row["camera_id"]
        has_zone  = row["has_zone"]
        direction = row.get("direction")
        room_name = row.get("room_name")

        if direction == "IN" and has_zone and room_name:
            location = room_name
        else:
            location = cam_name

        event_kind = row.get("event_kind") or ("room_entry" if has_zone else "passage")

        result.append(MovementRecord(
            timestamp=row["timestamp"],
            location=location,
            floor=row["db_floor"] or "—",
            camera=cam_name,
            isCurrent=(i == 0),
            note="Posisi sekarang" if i == 0 else "",
            event_kind=event_kind,
            direction=direction,
            snapshot_url=row.get("snapshot_url"),
            track_id=row.get("track_id"),
        ))

    return result


@router.patch("/{person_id}/rename", response_model=PersonResponse)
async def rename_person(
    person_id: int, req: RenameRequest, request: Request
) -> PersonResponse:
    new_name = req.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=422, detail="new_name must not be empty")

    pool = request.app.state.pool
    row = await pool.fetchrow(
        "UPDATE persons SET name = $1, is_known = TRUE WHERE id = $2 RETURNING *",
        new_name,
        person_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")
    return dict(row)
