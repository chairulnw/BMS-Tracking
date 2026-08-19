import asyncio
import os
import re
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.auth import create_access_token
from app.schemas import (
    CameraGroupAssign,
    CameraGroupIn,
    CameraGroupResponse,
    CameraIn,
    CameraResponse,
)

router = APIRouter(tags=["cameras"])

_AI_URL = lambda: os.getenv("AI_SERVICE_URL", "http://localhost:8001")

_RESTART_DEBOUNCE = 1.5  # detik — beberapa edit kamera beruntun cuma restart sekali
_restart_task: "asyncio.Task | None" = None


async def _restart_ai_stream() -> None:
    """Jadwalkan restart stream AI service (debounced) supaya perubahan kamera
    (aktif/nonaktif, RTSP, dihapus) langsung berlaku tanpa perlu tombol manual.
    Restart AI service reload model dari disk (mahal, beberapa detik) — kalau
    beberapa kamera diedit berturut-turut, di-debounce jadi satu restart saja,
    bukan sekali per save. Best-effort — kalau AI service down, CRUD kamera
    tetap sukses; endpoint juga tidak menunggu restart selesai."""
    global _restart_task
    if _restart_task is not None and not _restart_task.done():
        _restart_task.cancel()
    _restart_task = asyncio.create_task(_debounced_restart())


async def _debounced_restart() -> None:
    try:
        await asyncio.sleep(_RESTART_DEBOUNCE)
    except asyncio.CancelledError:
        return  # ada edit lain masuk, restart ini dibatalkan & digantikan yang baru
    headers = {"Authorization": f"Bearer {create_access_token('backend-service')}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(f"{_AI_URL()}/stream/stop", headers=headers)
            r = await client.post(f"{_AI_URL()}/stream/start", json={}, headers=headers)
            r.raise_for_status()
    except Exception as exc:
        print(f"[cameras] gagal restart AI stream: {exc}")


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


# ── Camera groups ─────────────────────────────────────────────────────────────

@router.get("/camera-groups", response_model=list[CameraGroupResponse])
async def list_camera_groups(request: Request) -> list[CameraGroupResponse]:
    rows = await request.app.state.pool.fetch(
        "SELECT * FROM camera_groups ORDER BY name"
    )
    return [dict(r) for r in rows]


@router.post("/camera-groups", response_model=CameraGroupResponse, status_code=201)
async def create_camera_group(req: CameraGroupIn, request: Request) -> CameraGroupResponse:
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Nama grup tidak boleh kosong")
    row = await request.app.state.pool.fetchrow(
        """
        INSERT INTO camera_groups (name) VALUES ($1)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING *
        """,
        name,
    )
    return dict(row)


@router.delete("/camera-groups/{group_id}", status_code=204)
async def delete_camera_group(group_id: int, request: Request) -> None:
    await request.app.state.pool.execute(
        "DELETE FROM camera_groups WHERE id = $1", group_id
    )


_CAMERA_SELECT = """
    SELECT
        c.id, c.camera_id, c.name, c.location, c.group_id, c.rtsp_url,
        c.is_active, c.analytics_enabled, c.created_at,
        cg.name AS group_name,
        COALESCE(zc.zone_count, 0)::int AS zone_count
    FROM cameras c
    LEFT JOIN camera_groups cg ON cg.id = c.group_id
    LEFT JOIN (
        SELECT camera_id, COUNT(*) AS zone_count
        FROM zone_cameras
        GROUP BY camera_id
    ) zc ON zc.camera_id = c.id
"""


# ── Camera CRUD ───────────────────────────────────────────────────────────────

@router.get("/cameras", response_model=list[CameraResponse])
async def list_cameras(
    request: Request,
    is_active: bool | None = Query(None),
) -> list[CameraResponse]:
    if is_active is None:
        rows = await request.app.state.pool.fetch(f"{_CAMERA_SELECT} ORDER BY c.id")
    else:
        rows = await request.app.state.pool.fetch(
            f"{_CAMERA_SELECT} WHERE c.is_active = $1 ORDER BY c.id", is_active
        )
    return [dict(r) for r in rows]


@router.post("/cameras", response_model=CameraResponse, status_code=201)
async def create_camera(req: CameraIn, request: Request) -> CameraResponse:
    pool = request.app.state.pool
    camera_id = req.camera_id or _extract_cam_id(req.rtsp_url)

    group_name = None
    if req.group_id is not None:
        group_name = await pool.fetchval(
            "SELECT name FROM camera_groups WHERE id = $1", req.group_id
        )

    async with pool.acquire() as conn:
        async with conn.transaction():
            new_id = await conn.fetchval(
                """
                INSERT INTO cameras
                    (camera_id, name, location, group_id, floor, rtsp_url, is_active, analytics_enabled)
                VALUES ($1, '', $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                camera_id, req.location, req.group_id, group_name, req.rtsp_url,
                req.is_active, req.analytics_enabled and req.is_active,
            )
            await conn.execute(
                "UPDATE cameras SET name = 'Camera ' || $1::text WHERE id = $1", new_id
            )
    await _restart_ai_stream()
    row = await pool.fetchrow(f"{_CAMERA_SELECT} WHERE c.id = $1", new_id)
    return dict(row)


@router.put("/cameras/{cam_id}", response_model=CameraResponse)
async def update_camera(cam_id: int, req: CameraIn, request: Request) -> CameraResponse:
    pool = request.app.state.pool
    camera_id = req.camera_id or _extract_cam_id(req.rtsp_url)

    group_name = None
    if req.group_id is not None:
        group_name = await pool.fetchval(
            "SELECT name FROM camera_groups WHERE id = $1", req.group_id
        )

    row = await pool.fetchrow(
        """
        UPDATE cameras
           SET camera_id         = $1,
               location          = $2,
               group_id          = $3,
               floor             = $4,
               rtsp_url          = $5,
               is_active         = $6,
               analytics_enabled = $7
         WHERE id = $8
        RETURNING id
        """,
        camera_id, req.location, req.group_id, group_name, req.rtsp_url,
        req.is_active, req.analytics_enabled and req.is_active, cam_id,
    )
    if not row:
        raise HTTPException(404, "Camera not found")
    await _restart_ai_stream()
    row = await pool.fetchrow(f"{_CAMERA_SELECT} WHERE c.id = $1", cam_id)
    return dict(row)


@router.patch("/cameras/{cam_id}/group", response_model=CameraResponse)
async def assign_camera_group(cam_id: int, req: CameraGroupAssign, request: Request) -> CameraResponse:
    pool = request.app.state.pool
    group_name = None
    if req.group_id is not None:
        group_name = await pool.fetchval(
            "SELECT name FROM camera_groups WHERE id = $1", req.group_id
        )
    row = await pool.fetchrow(
        "UPDATE cameras SET group_id = $1, floor = $2 WHERE id = $3 RETURNING id",
        req.group_id, group_name, cam_id,
    )
    if not row:
        raise HTTPException(404, "Camera not found")
    row = await pool.fetchrow(f"{_CAMERA_SELECT} WHERE c.id = $1", cam_id)
    return dict(row)


@router.delete("/cameras/{cam_id}", status_code=204)
async def delete_camera(cam_id: int, request: Request) -> None:
    await request.app.state.pool.execute(
        "DELETE FROM cameras WHERE id = $1", cam_id
    )
    await _restart_ai_stream()


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
