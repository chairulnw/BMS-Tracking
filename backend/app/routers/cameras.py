import os
import re
from datetime import date, datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.auth import create_access_token
from app.schemas import (
    CameraIn,
    CameraResponse,
    CrossingLineIn,
    CrossingLineResponse,
    OccupancyEventCreate,
    OccupancyResponse,
    SaveZoneRequest,
)

router = APIRouter(tags=["cameras"])

_AI_URL = lambda: os.getenv("AI_SERVICE_URL", "http://localhost:8001")


def _extract_cam_id(rtsp_url: str) -> str | None:
    """Ekstrak camera_id dari RTSP URL: segment setelah 'unicast/' atau komponen kedua dari belakang."""
    try:
        path = urlparse(rtsp_url).path
        m = re.search(r"/unicast/([^/]+)", path)
        if m:
            return m.group(1)
        parts = [p for p in path.split("/") if p]
        return parts[-2] if len(parts) >= 2 else None
    except Exception:
        return None


# ── Camera CRUD ───────────────────────────────────────────────────────────────

@router.get("/cameras", response_model=list[CameraResponse])
async def list_cameras(
    request: Request,
    is_active: bool | None = Query(None),
) -> list[CameraResponse]:
    if is_active is None:
        rows = await request.app.state.pool.fetch(
            "SELECT * FROM cameras ORDER BY id"
        )
    else:
        rows = await request.app.state.pool.fetch(
            "SELECT * FROM cameras WHERE is_active = $1 ORDER BY id", is_active
        )
    return [dict(r) for r in rows]


@router.post("/cameras", response_model=CameraResponse, status_code=201)
async def create_camera(req: CameraIn, request: Request) -> CameraResponse:
    camera_id = req.camera_id or _extract_cam_id(req.rtsp_url)
    row = await request.app.state.pool.fetchrow(
        """
        INSERT INTO cameras (camera_id, name, zone_location, floor, rtsp_url, is_active)
        VALUES ($1, $2, $3, $4, $5, $6) RETURNING *
        """,
        camera_id, req.name, req.zone_location, req.floor, req.rtsp_url, req.is_active,
    )
    return dict(row)


@router.put("/cameras/{cam_id}", response_model=CameraResponse)
async def update_camera(cam_id: int, req: CameraIn, request: Request) -> CameraResponse:
    camera_id = req.camera_id or _extract_cam_id(req.rtsp_url)
    row = await request.app.state.pool.fetchrow(
        """
        UPDATE cameras
           SET camera_id     = $1,
               name          = $2,
               zone_location = $3,
               floor         = $4,
               rtsp_url      = $5,
               is_active     = $6
         WHERE id = $7
        RETURNING *
        """,
        camera_id, req.name, req.zone_location, req.floor, req.rtsp_url, req.is_active, cam_id,
    )
    if not row:
        raise HTTPException(404, "Camera not found")
    return dict(row)


@router.delete("/cameras/{cam_id}", status_code=204)
async def delete_camera(cam_id: int, request: Request) -> None:
    await request.app.state.pool.execute(
        "DELETE FROM cameras WHERE id = $1", cam_id
    )


@router.patch("/cameras/{cam_id}/toggle", response_model=CameraResponse)
async def toggle_camera(cam_id: int, request: Request) -> CameraResponse:
    row = await request.app.state.pool.fetchrow(
        "UPDATE cameras SET is_active = NOT is_active WHERE id = $1 RETURNING *",
        cam_id,
    )
    if not row:
        raise HTTPException(404, "Camera not found")
    return dict(row)


# ── Snapshot proxy ────────────────────────────────────────────────────────────

@router.get("/cameras/{camera_id}/snapshot")
async def get_snapshot(camera_id: str, request: Request) -> Response:
    """Proxy ke AI service. Jika stream tidak berjalan, kirim rtsp_url agar AI
    service bisa langsung capture dari RTSP."""
    row = await request.app.state.pool.fetchrow(
        "SELECT rtsp_url FROM cameras WHERE camera_id = $1", camera_id
    )
    params: dict = {}
    if row:
        params["rtsp_url"] = row["rtsp_url"]
    headers = {"Authorization": f"Bearer {create_access_token('backend-service')}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{_AI_URL()}/snapshot/{camera_id}", params=params, headers=headers)
        if r.status_code == 404:
            raise HTTPException(404, "Kamera tidak ditemukan atau stream belum berjalan")
        r.raise_for_status()
        return Response(content=r.content, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"AI service tidak dapat dijangkau: {exc}")


# ── Zone (room name) ──────────────────────────────────────────────────────────

@router.post("/cameras/{camera_id}/zone", status_code=201)
async def upsert_zone(camera_id: str, req: SaveZoneRequest, request: Request) -> dict:
    pool = request.app.state.pool
    row  = await pool.fetchrow(
        """
        INSERT INTO camera_zones (camera_id, room_name, floor)
        VALUES ($1, $2, $3)
        ON CONFLICT (camera_id)
        DO UPDATE SET room_name = EXCLUDED.room_name, floor = EXCLUDED.floor
        RETURNING *
        """,
        camera_id, req.room_name, req.floor,
    )
    return dict(row)


@router.get("/cameras/{camera_id}/zone")
async def get_zone(camera_id: str, request: Request) -> dict:
    row = await request.app.state.pool.fetchrow(
        "SELECT * FROM camera_zones WHERE camera_id = $1", camera_id
    )
    return dict(row) if row else {}


# ── Crossing lines ────────────────────────────────────────────────────────────

@router.post("/cameras/{camera_id}/lines", response_model=list[CrossingLineResponse], status_code=201)
async def save_lines(
    camera_id: str,
    lines: list[CrossingLineIn],
    request: Request,
) -> list[CrossingLineResponse]:
    """Ganti semua garis crossing untuk kamera ini (delete + re-insert)."""
    pool = request.app.state.pool
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM crossing_lines WHERE camera_id = $1", camera_id
            )
            rows = []
            for line in lines:
                row = await conn.fetchrow(
                    """
                    INSERT INTO crossing_lines (camera_id, p1_x, p1_y, p2_x, p2_y, in_sign)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING *
                    """,
                    camera_id, line.p1_x, line.p1_y, line.p2_x, line.p2_y, line.in_sign,
                )
                rows.append(dict(row))
    return rows


@router.get("/cameras/{camera_id}/lines", response_model=list[CrossingLineResponse])
async def get_lines(camera_id: str, request: Request) -> list[CrossingLineResponse]:
    rows = await request.app.state.pool.fetch(
        "SELECT * FROM crossing_lines WHERE camera_id = $1 ORDER BY id",
        camera_id,
    )
    return [dict(r) for r in rows]


# ── Occupancy events ──────────────────────────────────────────────────────────

@router.post("/occupancy-events", status_code=201)
async def create_occupancy_event(req: OccupancyEventCreate, request: Request) -> dict:
    pool = request.app.state.pool
    ts   = req.timestamp or datetime.now(timezone.utc)
    row  = await pool.fetchrow(
        """
        INSERT INTO occupancy_events
            (camera_id, line_id, direction, timestamp, event_kind, snapshot_url, person_label, track_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING *
        """,
        req.camera_id, req.line_id, req.direction.upper(), ts,
        req.event_kind or "crossing", req.snapshot_url, req.person_label, req.track_id,
    )
    return dict(row)


@router.get("/occupancy", response_model=list[OccupancyResponse])
async def get_occupancy(
    request: Request,
    date_filter: date | None = Query(None, alias="date"),
) -> list[OccupancyResponse]:
    pool = request.app.state.pool
    if date_filter is None:
        date_filter = datetime.now(timezone.utc).date()

    rows = await pool.fetch(
        """
        SELECT
            cz.camera_id,
            cz.room_name,
            COALESCE(cz.floor, c.floor) AS floor,
            COALESCE(SUM(CASE WHEN oe.direction = 'IN'  THEN 1 ELSE 0 END), 0)::int AS count_in,
            COALESCE(SUM(CASE WHEN oe.direction = 'OUT' THEN 1 ELSE 0 END), 0)::int AS count_out
        FROM camera_zones cz
        LEFT JOIN cameras c             ON c.camera_id = cz.camera_id
        LEFT JOIN crossing_lines cl     ON cl.camera_id = cz.camera_id
        LEFT JOIN occupancy_events oe
               ON oe.line_id = cl.id
              AND (oe.timestamp AT TIME ZONE 'UTC')::date = $1
        GROUP BY cz.camera_id, cz.room_name, COALESCE(cz.floor, c.floor)
        ORDER BY cz.room_name
        """,
        date_filter,
    )
    return [
        {**dict(r), "current_occupancy": r["count_in"] - r["count_out"]}
        for r in rows
    ]
