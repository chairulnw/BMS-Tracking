from datetime import date, datetime, timezone

from fastapi import APIRouter, Query, Request

from app.notify import notify_telegram
from app.schemas import CameraEventCreate, CameraEventResponse, StatsToday

router = APIRouter(tags=["camera-events"])


@router.post("/camera-events", response_model=CameraEventResponse, status_code=201)
async def create_camera_event(req: CameraEventCreate, request: Request) -> CameraEventResponse:
    pool = request.app.state.pool
    ts   = req.timestamp or datetime.now(timezone.utc)
    row  = await pool.fetchrow(
        """
        INSERT INTO camera_events
            (camera_id, event_type, category, description, snapshot_url, person_label, timestamp)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING *
        """,
        req.camera_id, req.event_type, req.category,
        req.description, req.snapshot_url, req.person_label, ts,
    )
    cam = await pool.fetchrow(
        "SELECT name FROM cameras WHERE camera_id = $1", req.camera_id
    )
    if req.category == "critical":
        cam_label = cam["name"] if cam else req.camera_id
        await notify_telegram(f"🔴 {cam_label}: {req.description or req.event_type}")
    return {**dict(row), "camera_name": cam["name"] if cam else None}


@router.get("/camera-events")
async def list_camera_events(
    request: Request,
    date_filter: date | None = Query(None, alias="date"),
    camera_id:   str | None = Query(None),
    event_type:  str | None = Query(None),
    category:    str | None = Query(None),
    page:        int        = Query(1, ge=1),
    limit:       int        = Query(4, ge=1, le=100),
) -> dict:
    pool     = request.app.state.pool
    date_val = date_filter or datetime.now(timezone.utc).date()

    conditions: list[str] = ["(ce.timestamp AT TIME ZONE 'UTC')::date = $1"]
    params: list           = [date_val]
    idx = 2

    if camera_id:
        conditions.append(f"ce.camera_id = ${idx}"); params.append(camera_id); idx += 1
    if event_type:
        conditions.append(f"ce.event_type = ${idx}"); params.append(event_type); idx += 1
    if category:
        conditions.append(f"ce.category = ${idx}"); params.append(category); idx += 1

    where  = " AND ".join(conditions)
    offset = (page - 1) * limit

    total = await pool.fetchval(
        f"SELECT COUNT(*) FROM camera_events ce WHERE {where}", *params
    )
    rows = await pool.fetch(
        f"""
        SELECT ce.*, c.name AS camera_name
        FROM   camera_events ce
        LEFT JOIN cameras c ON c.camera_id = ce.camera_id
        WHERE  {where}
        ORDER  BY ce.timestamp DESC
        LIMIT  {limit} OFFSET {offset}
        """,
        *params,
    )
    return {
        "events": [dict(r) for r in rows],
        "total":  int(total),
        "page":   page,
        "pages":  max(1, (int(total) + limit - 1) // limit),
        "limit":  limit,
    }


@router.get("/stats/today", response_model=StatsToday)
async def stats_today(request: Request) -> StatsToday:
    pool  = request.app.state.pool
    today = datetime.now(timezone.utc).date()

    cams = await pool.fetchrow(
        "SELECT COUNT(*) AS total, COUNT(*) FILTER (WHERE is_active) AS active FROM cameras"
    )
    events_count = await pool.fetchval(
        "SELECT COUNT(*) FROM camera_events WHERE (timestamp AT TIME ZONE 'UTC')::date = $1",
        today,
    )
    det_count = await pool.fetchval(
        "SELECT COUNT(*) FROM detections WHERE (timestamp AT TIME ZONE 'UTC')::date = $1",
        today,
    )
    person_rows = await pool.fetch(
        """
        SELECT p.is_known, COUNT(*) AS cnt
        FROM   persons p
        WHERE  EXISTS (
            SELECT 1 FROM detections d
            WHERE  d.person_id = p.id
              AND  (d.timestamp AT TIME ZONE 'UTC')::date = $1
        )
        GROUP BY p.is_known
        """,
        today,
    )
    known   = sum(int(r["cnt"]) for r in person_rows if r["is_known"])
    unknown = sum(int(r["cnt"]) for r in person_rows if not r["is_known"])

    return {
        "cameras_total":    int(cams["total"]),
        "cameras_active":   int(cams["active"]),
        "cameras_inactive": int(cams["total"]) - int(cams["active"]),
        "events_today":     int(events_count),
        "detections_today": int(det_count),
        "persons_today":    known + unknown,
        "persons_known":    known,
        "persons_unknown":  unknown,
    }
