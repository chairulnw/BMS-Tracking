from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas import (
    OccupancyEventCreate,
    OccupancyResponse,
    ZoneCameraIn,
    ZoneCameraResponse,
    ZoneDetailResponse,
    ZoneForCameraResponse,
    ZoneHeatmapResponse,
    ZoneHistoryPoint,
    ZoneIn,
    ZoneResponse,
    ZoneUpdate,
)

router = APIRouter(tags=["zones"])

_ZONE_SELECT = """
    SELECT z.id, z.name, z.max_capacity, z.created_at,
           COUNT(DISTINCT zc.camera_id)::int AS camera_count,
           COALESCE(array_agg(DISTINCT c.camera_id) FILTER (WHERE c.camera_id IS NOT NULL), ARRAY[]::text[]) AS camera_ids,
           COALESCE(array_agg(DISTINCT zc.type) FILTER (WHERE zc.type IS NOT NULL), ARRAY[]::text[]) AS types
    FROM zones z
    LEFT JOIN zone_cameras zc ON zc.zone_id = z.id
    LEFT JOIN cameras c       ON c.id = zc.camera_id
"""


# ── Zone CRUD ─────────────────────────────────────────────────────────────────

@router.get("/zones", response_model=list[ZoneResponse])
async def list_zones(request: Request) -> list[ZoneResponse]:
    rows = await request.app.state.pool.fetch(
        f"{_ZONE_SELECT} GROUP BY z.id ORDER BY z.id"
    )
    return [dict(r) for r in rows]


@router.post("/zones", response_model=ZoneResponse, status_code=201)
async def create_zone(req: ZoneIn, request: Request) -> ZoneResponse:
    """Zona gak boleh kosong — dibuat sekaligus dengan sumber kamera + gambar
    pertamanya dalam satu transaksi."""
    if req.type not in ("line", "polygon"):
        raise HTTPException(400, "type harus 'line' atau 'polygon'")
    pool = request.app.state.pool
    cam_pk = await pool.fetchval("SELECT id FROM cameras WHERE camera_id = $1", req.camera_id)
    if cam_pk is None:
        raise HTTPException(404, "Camera not found")

    async with pool.acquire() as conn:
        async with conn.transaction():
            zone_id = await conn.fetchval(
                "INSERT INTO zones (name, max_capacity) VALUES ($1, $2) RETURNING id",
                req.name.strip(), req.max_capacity,
            )
            await conn.execute(
                "INSERT INTO zone_cameras (zone_id, camera_id, type, points) VALUES ($1, $2, $3, $4)",
                zone_id, cam_pk, req.type, req.points,
            )
    row = await pool.fetchrow(f"{_ZONE_SELECT} WHERE z.id = $1 GROUP BY z.id", zone_id)
    return dict(row)


@router.get("/zones/{zone_id}", response_model=ZoneDetailResponse)
async def get_zone(zone_id: int, request: Request) -> ZoneDetailResponse:
    pool = request.app.state.pool
    zone = await pool.fetchrow(f"{_ZONE_SELECT} WHERE z.id = $1 GROUP BY z.id", zone_id)
    if not zone:
        raise HTTPException(404, "Zone not found")
    cam_rows = await pool.fetch(
        """
        SELECT zc.id, zc.zone_id, zc.camera_id, c.camera_id AS camera_str_id,
               c.name AS camera_name, zc.type, zc.points
        FROM zone_cameras zc
        JOIN cameras c ON c.id = zc.camera_id
        WHERE zc.zone_id = $1
        ORDER BY zc.id
        """,
        zone_id,
    )
    result = dict(zone)
    result["cameras"] = [dict(r) for r in cam_rows]
    return result


@router.put("/zones/{zone_id}", response_model=ZoneResponse)
async def update_zone(zone_id: int, req: ZoneUpdate, request: Request) -> ZoneResponse:
    pool = request.app.state.pool
    row = await pool.fetchrow(
        "UPDATE zones SET name = $1, max_capacity = $2 WHERE id = $3 RETURNING id",
        req.name.strip(), req.max_capacity, zone_id,
    )
    if not row:
        raise HTTPException(404, "Zone not found")
    row = await pool.fetchrow(f"{_ZONE_SELECT} WHERE z.id = $1 GROUP BY z.id", zone_id)
    return dict(row)


@router.delete("/zones/{zone_id}", status_code=204)
async def delete_zone(zone_id: int, request: Request) -> None:
    await request.app.state.pool.execute("DELETE FROM zones WHERE id = $1", zone_id)


# ── Zone <-> camera geometry ────────────────────────────────────────────────────

@router.put("/zones/{zone_id}/cameras/{camera_id}", response_model=ZoneCameraResponse)
async def set_zone_camera(
    zone_id: int, camera_id: str, req: ZoneCameraIn, request: Request
) -> ZoneCameraResponse:
    """Set/replace geometri (+ tipe) kamera ini untuk zona ini. Tipe ditanyakan
    tiap kali mau gambar — dipakai buat kamera baru maupun gambar ulang yang lama.
    camera_id = string cameras.camera_id."""
    if req.type not in ("line", "polygon"):
        raise HTTPException(400, "type harus 'line' atau 'polygon'")
    pool = request.app.state.pool
    cam_pk = await pool.fetchval("SELECT id FROM cameras WHERE camera_id = $1", camera_id)
    if cam_pk is None:
        raise HTTPException(404, "Camera not found")
    if not await pool.fetchval("SELECT id FROM zones WHERE id = $1", zone_id):
        raise HTTPException(404, "Zone not found")

    row = await pool.fetchrow(
        """
        INSERT INTO zone_cameras (zone_id, camera_id, type, points)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (zone_id, camera_id) DO UPDATE SET type = EXCLUDED.type, points = EXCLUDED.points
        RETURNING id, zone_id, camera_id, type, points
        """,
        zone_id, cam_pk, req.type, req.points,
    )
    result = dict(row)
    result["camera_str_id"] = camera_id
    result["camera_name"]   = await pool.fetchval("SELECT name FROM cameras WHERE id = $1", cam_pk)
    return result


@router.delete("/zones/{zone_id}/cameras/{camera_id}", status_code=204)
async def remove_zone_camera(zone_id: int, camera_id: str, request: Request) -> None:
    """Zona wajib punya minimal 1 sumber — kalau ini kamera terakhir, tolak
    (user harus hapus zona-nya sekalian kalau memang mau)."""
    pool = request.app.state.pool
    count = await pool.fetchval("SELECT COUNT(*) FROM zone_cameras WHERE zone_id = $1", zone_id)
    if count <= 1:
        raise HTTPException(400, "Zona harus punya minimal 1 kamera — hapus zona-nya kalau mau lepas semua")
    cam_pk = await pool.fetchval("SELECT id FROM cameras WHERE camera_id = $1", camera_id)
    if cam_pk is not None:
        await pool.execute(
            "DELETE FROM zone_cameras WHERE zone_id = $1 AND camera_id = $2", zone_id, cam_pk
        )


@router.get("/zones/for-camera/{camera_id}", response_model=list[ZoneForCameraResponse])
async def get_zones_for_camera(camera_id: str, request: Request) -> list[ZoneForCameraResponse]:
    """Dipakai AI service: semua zona (+ geometri) yang dipantau kamera ini."""
    rows = await request.app.state.pool.fetch(
        """
        SELECT z.id AS zone_id, zc.id AS zone_camera_id, z.name, zc.type, zc.points, z.max_capacity
        FROM zone_cameras zc
        JOIN zones z ON z.id = zc.zone_id
        JOIN cameras c ON c.id = zc.camera_id
        WHERE c.camera_id = $1
        ORDER BY z.id
        """,
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
            (camera_id, zone_camera_id, direction, timestamp, event_kind,
             snapshot_url, person_label, track_id, point_x, point_y)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING *
        """,
        req.camera_id, req.zone_camera_id, req.direction.upper(), ts,
        req.event_kind or "crossing", req.snapshot_url, req.person_label, req.track_id,
        req.point_x, req.point_y,
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
            z.id   AS zone_id,
            z.name AS zone_name,
            z.max_capacity,
            COALESCE(SUM(CASE WHEN oe.direction = 'IN'  THEN 1 ELSE 0 END), 0)::int AS count_in,
            COALESCE(SUM(CASE WHEN oe.direction = 'OUT' THEN 1 ELSE 0 END), 0)::int AS count_out
        FROM zones z
        LEFT JOIN zone_cameras zc   ON zc.zone_id = z.id
        LEFT JOIN occupancy_events oe
               ON oe.zone_camera_id = zc.id
              AND (oe.timestamp AT TIME ZONE 'UTC')::date = $1
        GROUP BY z.id, z.name, z.max_capacity
        ORDER BY z.name
        """,
        date_filter,
    )
    return [
        {**dict(r), "current_occupancy": r["count_in"] - r["count_out"]}
        for r in rows
    ]


# ── History / heatmap / event log ────────────────────────────────────────────

@router.get("/zones/{zone_id}/history", response_model=list[ZoneHistoryPoint])
async def zone_history(
    zone_id: int,
    request: Request,
    date_from: date = Query(..., alias="from"),
    date_to:   date = Query(..., alias="to"),
) -> list[ZoneHistoryPoint]:
    rows = await request.app.state.pool.fetch(
        """
        SELECT
            (oe.timestamp AT TIME ZONE 'UTC')::date AS date,
            COALESCE(SUM(CASE WHEN oe.direction = 'IN'  THEN 1 ELSE 0 END), 0)::int AS count_in,
            COALESCE(SUM(CASE WHEN oe.direction = 'OUT' THEN 1 ELSE 0 END), 0)::int AS count_out
        FROM occupancy_events oe
        JOIN zone_cameras zc ON zc.id = oe.zone_camera_id
        WHERE zc.zone_id = $1
          AND (oe.timestamp AT TIME ZONE 'UTC')::date BETWEEN $2 AND $3
        GROUP BY (oe.timestamp AT TIME ZONE 'UTC')::date
        ORDER BY date
        """,
        zone_id, date_from, date_to,
    )
    return [dict(r) for r in rows]


@router.get("/zones/{zone_id}/heatmap", response_model=ZoneHeatmapResponse)
async def zone_heatmap(
    zone_id: int,
    request: Request,
    camera_id: str = Query(..., description="cameras.camera_id — heatmap dihitung per sudut pandang satu kamera"),
    date_from: date | None = Query(None, alias="from"),
    date_to:   date | None = Query(None, alias="to"),
    grid:      int = Query(20, ge=4, le=50),
) -> ZoneHeatmapResponse:
    conditions = ["zc.zone_id = $1", "c.camera_id = $2",
                  "oe.point_x IS NOT NULL", "oe.point_y IS NOT NULL"]
    params: list = [zone_id, camera_id]
    idx = 3
    if date_from:
        conditions.append(f"(oe.timestamp AT TIME ZONE 'UTC')::date >= ${idx}")
        params.append(date_from); idx += 1
    if date_to:
        conditions.append(f"(oe.timestamp AT TIME ZONE 'UTC')::date <= ${idx}")
        params.append(date_to); idx += 1
    where = " AND ".join(conditions)

    rows = await request.app.state.pool.fetch(
        f"""
        SELECT oe.point_x, oe.point_y
        FROM occupancy_events oe
        JOIN zone_cameras zc ON zc.id = oe.zone_camera_id
        JOIN cameras c       ON c.id  = zc.camera_id
        WHERE {where}
        """,
        *params,
    )

    cells = [[0] * grid for _ in range(grid)]
    if rows:
        xs, ys = [r["point_x"] for r in rows], [r["point_y"] for r in rows]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x, span_y = max(1, max_x - min_x), max(1, max_y - min_y)
        for r in rows:
            gx = min(grid - 1, int((r["point_x"] - min_x) / span_x * grid))
            gy = min(grid - 1, int((r["point_y"] - min_y) / span_y * grid))
            cells[gy][gx] += 1

    return {"grid_size": grid, "cells": cells}


@router.get("/zones/{zone_id}/events")
async def zone_events(
    zone_id: int,
    request: Request,
    date_from: date | None = Query(None, alias="from"),
    date_to:   date | None = Query(None, alias="to"),
    camera_id: str | None = Query(None),
    page:      int = Query(1, ge=1),
    limit:     int = Query(20, ge=1, le=100),
) -> dict:
    pool = request.app.state.pool
    conditions = ["zc.zone_id = $1"]
    params: list = [zone_id]
    idx = 2
    if date_from:
        conditions.append(f"(oe.timestamp AT TIME ZONE 'UTC')::date >= ${idx}")
        params.append(date_from); idx += 1
    if date_to:
        conditions.append(f"(oe.timestamp AT TIME ZONE 'UTC')::date <= ${idx}")
        params.append(date_to); idx += 1
    if camera_id:
        conditions.append(f"oe.camera_id = ${idx}")
        params.append(camera_id); idx += 1
    where  = " AND ".join(conditions)
    offset = (page - 1) * limit

    total = await pool.fetchval(
        f"""
        SELECT COUNT(*) FROM occupancy_events oe
        JOIN zone_cameras zc ON zc.id = oe.zone_camera_id
        WHERE {where}
        """,
        *params,
    )
    rows = await pool.fetch(
        f"""
        SELECT oe.id, oe.timestamp, oe.direction, oe.person_label, oe.snapshot_url,
               oe.camera_id, c.name AS camera_name
        FROM occupancy_events oe
        JOIN zone_cameras zc ON zc.id = oe.zone_camera_id
        LEFT JOIN cameras c  ON c.camera_id = oe.camera_id
        WHERE {where}
        ORDER BY oe.timestamp DESC
        LIMIT {limit} OFFSET {offset}
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
