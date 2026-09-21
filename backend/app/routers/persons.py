import os
from datetime import date, datetime

import httpx
from fastapi import APIRouter, HTTPException, Query, Request

from app.auth import create_access_token
from app.schemas import (
    CameraPoint, DwellRecord, MovementRecord, NameSuggestion, PersonCrossingResponse,
    PersonDetail, PersonResponse, PersonUpdate, RenameRequest, TrajectoryPoint,
)

router = APIRouter(prefix="/persons", tags=["persons"])

_AI_URL = lambda: os.getenv("AI_SERVICE_URL", "http://localhost:8001")


@router.get("", response_model=list[PersonResponse])
async def list_persons(
    request: Request,
    date_filter: date | None = Query(None, alias="date", description="Alias satu-hari — dipakai kalau from/to kosong"),
    from_date:  date | None  = Query(None, alias="from"),
    to_date:    date | None  = Query(None, alias="to"),
    q:          str | None  = Query(None, description="Cari nama/label (identity search)"),
    known_only: bool         = Query(False, description="Hanya orang yang sudah bernama"),
    camera_id:  str | None   = Query(None, description="Boleh multi dipisah koma (OR) — 'c1,c2'"),
) -> list[PersonResponse]:
    pool = request.app.state.pool
    if from_date is None and to_date is None:
        if date_filter is None:
            from zoneinfo import ZoneInfo
            date_filter = datetime.now(ZoneInfo("Asia/Jakarta")).date()
        from_date = to_date = date_filter
    elif from_date is None:
        from_date = to_date
    elif to_date is None:
        to_date = from_date

    conditions = [
        """EXISTS (
            SELECT 1 FROM detections d
            WHERE d.person_id = p.id
              AND (d.timestamp AT TIME ZONE 'Asia/Jakarta')::date BETWEEN $1 AND $2
        )"""
    ]
    params: list = [from_date, to_date]

    if q:
        params.append(f"%{q}%")
        conditions.append(f"(p.name ILIKE ${len(params)} OR p.label ILIKE ${len(params)})")
    if known_only:
        conditions.append("p.is_known = TRUE")
    if camera_id:
        values = [v.strip() for v in camera_id.split(",") if v.strip()]
        if values:
            params.append(values)
            conditions.append(
                f"""EXISTS (
                    SELECT 1 FROM detections d2
                    WHERE d2.person_id = p.id AND d2.camera_id = ANY(${len(params)}::text[])
                )"""
            )

    where = " AND ".join(conditions)
    rows = await pool.fetch(
        f"""
        SELECT p.*, c.name AS last_camera_name, z.name AS last_zone_name,
               (SELECT COUNT(*) FROM detections d WHERE d.person_id = p.id) AS observation_count
        FROM persons p
        LEFT JOIN cameras c ON c.camera_id = p.last_camera
        LEFT JOIN LATERAL (
            SELECT zn.name
            FROM zone_cameras zc2
            JOIN zones zn ON zn.id = zc2.zone_id
            WHERE zc2.camera_id = c.id
            ORDER BY zn.id
            LIMIT 1
        ) z ON true
        WHERE {where}
        ORDER BY p.last_seen DESC NULLS LAST
        """,
        *params,
    )
    return [dict(r) for r in rows]


@router.get("/{person_id}", response_model=PersonDetail)
async def get_person(person_id: int, request: Request) -> PersonDetail:
    pool = request.app.state.pool
    row = await pool.fetchrow(
        """
        SELECT p.*, c.name AS last_camera_name, z.name AS last_zone_name,
               (SELECT COUNT(*) FROM detections d WHERE d.person_id = p.id) AS observation_count
        FROM persons p
        LEFT JOIN cameras c ON c.camera_id = p.last_camera
        LEFT JOIN LATERAL (
            SELECT zn.name
            FROM zone_cameras zc2
            JOIN zones zn ON zn.id = zc2.zone_id
            WHERE zc2.camera_id = c.id
            ORDER BY zn.id
            LIMIT 1
        ) z ON true
        WHERE p.id = $1
        """,
        person_id,
    )
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

    # Detections — deduplicated consecutive same-camera runs, no crossing data mixed in.
    # Zona kamera diambil dari zona PERTAMA (by id) yang memantau kamera ini, kalau ada
    # lebih dari satu — passive detection gak tau lagi di zona spesifik yang mana.
    det_rows = await pool.fetch(
        """
        SELECT
            d.camera_id,
            d.timestamp,
            c.name                       AS cam_name,
            z.name                       AS room_name,
            c.floor                      AS db_floor,
            (z.id IS NOT NULL)           AS has_zone,
            d.thumbnail_url              AS snapshot_url,
            d.track_id
        FROM detections d
        LEFT JOIN cameras c ON c.camera_id = d.camera_id
        LEFT JOIN LATERAL (
            SELECT zn.id, zn.name
            FROM zone_cameras zc2
            JOIN zones zn ON zn.id = zc2.zone_id
            WHERE zc2.camera_id = c.id
            ORDER BY zn.id
            LIMIT 1
        ) z ON true
        WHERE d.person_id = $1
          AND (d.timestamp AT TIME ZONE 'Asia/Jakarta')::date = $2
        ORDER BY d.timestamp DESC
        LIMIT 200
        """,
        person_id,
        date_filter,
    )

    # Crossing events — zona diketahui persis lewat zone_camera_id, gak perlu tebak
    cross_rows = await pool.fetch(
        """
        SELECT
            oe.camera_id,
            oe.timestamp,
            c.name                       AS cam_name,
            z.name                       AS room_name,
            c.floor                      AS db_floor,
            (z.id IS NOT NULL)           AS has_zone,
            oe.direction,
            oe.event_kind,
            oe.snapshot_url,
            oe.track_id
        FROM occupancy_events oe
        JOIN persons p              ON p.label     = oe.person_label
        LEFT JOIN zone_cameras zc   ON zc.id        = oe.zone_camera_id
        LEFT JOIN zones z           ON z.id         = zc.zone_id
        LEFT JOIN cameras c         ON c.camera_id  = oe.camera_id
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


@router.get("/{person_id}/crossings", response_model=list[PersonCrossingResponse])
async def get_crossings(
    person_id: int, request: Request, limit: int = Query(200, ge=1, le=500),
) -> list[PersonCrossingResponse]:
    """Event IN/OUT (zone crossing) milik orang ini — dipakai tab Timeline
    biar tampil sebagai kartu TERPISAH dari deteksi biasa, bukan ditebak lewat
    kedekatan waktu ke satu deteksi tertentu."""
    pool = request.app.state.pool
    exists = await pool.fetchval("SELECT id FROM persons WHERE id = $1", person_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Person not found")

    rows = await pool.fetch(
        """
        SELECT oe.id, oe.timestamp, oe.direction, oe.camera_id, oe.snapshot_url,
               c.name AS camera_name, z.name AS zone_name
        FROM occupancy_events oe
        JOIN persons p            ON p.label = oe.person_label
        LEFT JOIN cameras c       ON c.camera_id = oe.camera_id
        LEFT JOIN zone_cameras zc ON zc.id = oe.zone_camera_id
        LEFT JOIN zones z         ON z.id = zc.zone_id
        WHERE p.id = $1
        ORDER BY oe.timestamp DESC
        LIMIT $2
        """,
        person_id, limit,
    )
    return [dict(r) for r in rows]


@router.get("/{person_id}/trajectory", response_model=list[TrajectoryPoint])
async def get_trajectory(person_id: int, request: Request) -> list[TrajectoryPoint]:
    """Urutan kamera yang dilewati orang ini, dari tracklets (T4.2). Lintas
    hari sekaligus — sama seperti Timeline, tidak dibatasi tanggal."""
    pool = request.app.state.pool
    exists = await pool.fetchval("SELECT id FROM persons WHERE id = $1", person_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Person not found")

    rows = await pool.fetch(
        """
        SELECT
            t.camera_id,
            c.name AS camera_name,
            z.name AS zone_name,
            t.started_at,
            t.ended_at
        FROM tracklets t
        LEFT JOIN cameras c ON c.camera_id = t.camera_id
        LEFT JOIN LATERAL (
            SELECT zn.name
            FROM zone_cameras zc2
            JOIN zones zn ON zn.id = zc2.zone_id
            WHERE zc2.camera_id = c.id
            ORDER BY zn.id LIMIT 1
        ) z ON true
        WHERE t.person_id = $1
        ORDER BY t.started_at ASC
        """,
        person_id,
    )
    return [dict(r) for r in rows]


@router.get("/{person_id}/camera-points", response_model=list[CameraPoint])
async def get_camera_points(person_id: int, request: Request) -> list[CameraPoint]:
    """Titik-titik posisi orang ini per kamera, di ruang koordinat piksel asli
    kamera itu — dipakai tab Pergerakan (garis lintasan + heatmap), TIDAK
    butuh zona. Gabungan tiga sumber:
    - titik lintas (crossing) dari occupancy_events — butuh zona line/polygon,
      direction='IN'/'OUT', jarang ada
    - SELURUH titik kaki tiap tracklet dari tracklets.positions (JSONB array,
      urut waktu) — direction='TRACK'. Inilah yang bikin satu tracklet punya
      BANYAK titik (bisa disambung jadi garis), bukan cuma satu ringkasan.
      Timestamp per titik disintesis dari started_at + urutan (bukan waktu
      asli per-frame, yang tidak disimpan) — cukup buat urutan relatif.
    - fallback tracklets.pos_x/pos_y (satu titik) untuk baris lama yang
      direkam sebelum kolom `positions` ada."""
    pool = request.app.state.pool
    exists = await pool.fetchval("SELECT id FROM persons WHERE id = $1", person_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Person not found")

    rows = await pool.fetch(
        """
        SELECT oe.camera_id, c.name AS camera_name,
               oe.point_x AS x, oe.point_y AS y, oe.direction, oe.timestamp
        FROM occupancy_events oe
        JOIN persons p            ON p.label = oe.person_label
        LEFT JOIN cameras c       ON c.camera_id = oe.camera_id
        WHERE p.id = $1 AND oe.point_x IS NOT NULL AND oe.point_y IS NOT NULL

        UNION ALL

        SELECT t.camera_id, c2.name AS camera_name,
               (elem.value->>0)::int AS x, (elem.value->>1)::int AS y,
               'TRACK' AS direction,
               t.started_at + (elem.ordinality - 1) * interval '1 second' AS timestamp
        FROM (
            -- 10 baris lama (sebelum bug double-encode json.dumps() dibetulkan
            -- di tracklets.py) nyimpen `positions` sebagai STRING JSON, bukan
            -- array asli — jsonb_array_length() error keras kalau dipanggil ke
            -- situ. CASE di sini masukin NULL buat baris begitu; jsonb_array_
            -- elements(NULL) aman, otomatis 0 baris, gak perlu WHERE terpisah
            -- yang bisa "ditembus" planner pas di-gabung UNION ALL (AND biasa
            -- gak selalu short-circuit sebelum LATERAL function dipanggil).
            SELECT camera_id, started_at, person_id,
                   CASE WHEN jsonb_typeof(positions) = 'array' THEN positions END AS positions
            FROM tracklets
            WHERE person_id = $1
        ) t
        LEFT JOIN cameras c2 ON c2.camera_id = t.camera_id
        CROSS JOIN LATERAL jsonb_array_elements(t.positions) WITH ORDINALITY AS elem(value, ordinality)

        UNION ALL

        SELECT t.camera_id, c3.name AS camera_name,
               t.pos_x AS x, t.pos_y AS y, 'TRACK' AS direction, t.started_at AS timestamp
        FROM (
            SELECT camera_id, started_at, pos_x, pos_y,
                   CASE WHEN jsonb_typeof(positions) = 'array' THEN jsonb_array_length(positions) ELSE 0 END AS n_pos
            FROM tracklets
            WHERE person_id = $1
        ) t
        LEFT JOIN cameras c3 ON c3.camera_id = t.camera_id
        WHERE t.pos_x IS NOT NULL AND t.n_pos = 0

        ORDER BY timestamp
        """,
        person_id,
    )
    return [dict(r) for r in rows]


@router.get("/{person_id}/dwell", response_model=list[DwellRecord])
async def get_dwell(person_id: int, request: Request) -> list[DwellRecord]:
    """Total waktu tinggal, dua sumber digabung (T4.3), lintas hari sekaligus:

    - **per kamera** (`kind='camera'`) — langsung dari tracklets.started_at/
      ended_at, SELALU ADA, TIDAK butuh zona sama sekali. Ini yang bikin tab
      Durasi tetap kerja walau belum ada zona digambar (atau zona-nya nempel
      di kamera yang salah).
    - **per zona, line-crossing SAJA** (`kind='zone'`) — dipasangkan IN→OUT
      per zone_camera_id (per pintu fisik — satu ruangan bisa punya banyak
      pintu, jangan dipasangkan lintas pintu). Polygon SENGAJA tidak dihitung
      di sini — zona polygon (biasanya lorong) tumpang tindih 1:1 dengan
      pandangan kameranya, jadi durasinya sama saja dengan durasi kamera di
      atas; menghitungnya lagi di sini cuma duplikat."""
    pool = request.app.state.pool
    exists = await pool.fetchval("SELECT id FROM persons WHERE id = $1", person_id)
    if not exists:
        raise HTTPException(status_code=404, detail="Person not found")

    rows = await pool.fetch(
        """
        SELECT COALESCE(c.name, t.camera_id) AS zone_name, 'camera' AS kind,
               SUM(EXTRACT(EPOCH FROM (t.ended_at - t.started_at))) AS dwell_seconds
        FROM tracklets t
        LEFT JOIN cameras c ON c.camera_id = t.camera_id
        WHERE t.person_id = $1
        GROUP BY COALESCE(c.name, t.camera_id)

        UNION ALL

        SELECT zone_name, 'zone' AS kind, SUM(EXTRACT(EPOCH FROM (timestamp - prev_ts))) AS dwell_seconds
        FROM (
            SELECT
                z.name AS zone_name,
                oe.direction,
                oe.timestamp,
                LAG(oe.direction) OVER (PARTITION BY zc.id ORDER BY oe.timestamp) AS prev_dir,
                LAG(oe.timestamp) OVER (PARTITION BY zc.id ORDER BY oe.timestamp) AS prev_ts
            FROM occupancy_events oe
            JOIN persons p            ON p.label = oe.person_label
            JOIN zone_cameras zc      ON zc.id   = oe.zone_camera_id AND zc.type = 'line'
            JOIN zones z              ON z.id    = zc.zone_id
            WHERE p.id = $1
        ) ev
        WHERE direction = 'OUT' AND prev_dir = 'IN'
        GROUP BY zone_name

        ORDER BY dwell_seconds DESC
        """,
        person_id,
    )
    return [dict(r) for r in rows]


async def _propagate_rename_to_ai(old_name: str, new_name: str) -> None:
    """Beri tahu AI service bahwa identitas ini sudah dinamai, supaya deteksi
    berikutnya hari ini langsung pakai nama itu — bukan menunggu besok lewat
    upsert match-by-name. Fire-and-forget: kegagalan di sini tidak boleh
    menggagalkan PATCH-nya sendiri (AI service mungkin sedang tidak jalan)."""
    headers = {"Authorization": f"Bearer {create_access_token('backend-service')}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.patch(
                f"{_AI_URL()}/identities/{old_name}/rename",
                json={"new_name": new_name},
                headers=headers,
            )
    except Exception:
        pass  # AI service down atau identitas sudah tidak aktif — aman diabaikan


@router.patch("/{person_id}", response_model=PersonResponse)
async def update_person(
    person_id: int, req: PersonUpdate, request: Request
) -> PersonResponse:
    pool = request.app.state.pool

    existing = await pool.fetchrow("SELECT * FROM persons WHERE id = $1", person_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Person not found")

    new_name    = req.name.strip() if req.name is not None else None
    new_jabatan = req.jabatan.strip() if req.jabatan is not None else None

    if new_name is not None and not new_name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    # Operator memberi nama yang sama persis dengan orang lain yang sudah
    # dikenal — ini keputusan sadar operator (bukan tebakan algoritma ReID),
    # jadi digabung. Beda dengan ADR-002 (tidak ada merge OTOMATIS berbasis
    # skor kemiripan): ini merge eksplisit, dipicu tindakan manusia.
    if new_name is not None and new_name != existing["name"]:
        dup = await pool.fetchrow(
            "SELECT * FROM persons WHERE name = $1 AND is_known = TRUE AND id != $2",
            new_name, person_id,
        )
        if dup:
            target_id = dup["id"]
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE detections SET person_id = $1 WHERE person_id = $2", target_id, person_id
                    )
                    await conn.execute(
                        "UPDATE tracklets SET person_id = $1 WHERE person_id = $2", target_id, person_id
                    )
                    # occupancy_events tertaut lewat person_label (string), bukan person_id —
                    # tanpa ini, event lama jadi yatim begitu baris person_id lama dihapus.
                    await conn.execute(
                        "UPDATE occupancy_events SET person_label = $1 WHERE person_label = $2",
                        dup["label"], existing["label"],
                    )
                    await conn.execute(
                        """
                        UPDATE persons SET
                            first_seen  = LEAST(first_seen, $2),
                            last_seen   = GREATEST(last_seen, $3),
                            last_camera = CASE WHEN $3 >= last_seen THEN $4 ELSE last_camera END,
                            best_thumbnail_url = COALESCE(best_thumbnail_url, $5),
                            jabatan     = COALESCE(jabatan, $6)
                        WHERE id = $1
                        """,
                        target_id, existing["first_seen"], existing["last_seen"],
                        existing["last_camera"], existing["best_thumbnail_url"],
                        new_jabatan if req.jabatan is not None else existing["jabatan"],
                    )
                    await conn.execute("DELETE FROM persons WHERE id = $1", person_id)
            await _propagate_rename_to_ai(existing["name"], new_name)
            row = await pool.fetchrow("SELECT * FROM persons WHERE id = $1", target_id)
            merged = dict(row)
            merged["observation_count"] = await pool.fetchval(
                "SELECT COUNT(*) FROM detections WHERE person_id = $1", target_id
            )
            return merged

    set_clauses: list[str] = []
    params: list = []

    if new_name is not None:
        params.append(new_name)
        set_clauses.append(f"name = ${len(params)}")
        set_clauses.append("is_known = TRUE")
    if req.jabatan is not None:
        params.append(new_jabatan)
        set_clauses.append(f"jabatan = ${len(params)}")

    if not set_clauses:
        return {**dict(existing), "observation_count": await pool.fetchval(
            "SELECT COUNT(*) FROM detections WHERE person_id = $1", person_id
        )}

    params.append(person_id)
    row = await pool.fetchrow(
        f"UPDATE persons SET {', '.join(set_clauses)} WHERE id = ${len(params)} RETURNING *",
        *params,
    )
    if new_name is not None and new_name != existing["name"]:
        await _propagate_rename_to_ai(existing["name"], new_name)

    person = dict(row)
    person["observation_count"] = await pool.fetchval(
        "SELECT COUNT(*) FROM detections WHERE person_id = $1", person_id
    )
    return person


@router.patch("/{person_id}/rename", response_model=PersonResponse)
async def rename_person(
    person_id: int, req: RenameRequest, request: Request
) -> PersonResponse:
    """Alias lama — dipertahankan sampai frontend sepenuhnya pindah ke PATCH /persons/{id}."""
    return await update_person(person_id, PersonUpdate(name=req.new_name), request)


@router.get("/{person_id}/name-suggestions", response_model=list[NameSuggestion])
async def name_suggestions(
    person_id: int,
    request:   Request,
    days:      int = Query(7, ge=1, le=90, description="Cari di antara orang bernama N hari terakhir"),
    limit:     int = Query(5, ge=1, le=20),
) -> list[NameSuggestion]:
    """Kandidat nama untuk person_id yang belum dikenali, dari kemiripan
    embedding tracklet terhadap orang yang SUDAH bernama (Fase 3). Operator
    yang memutuskan lewat PATCH /persons/{id} — endpoint ini TIDAK PERNAH
    menerapkan nama secara otomatis (lihat plan/06-decisions.md)."""
    pool = request.app.state.pool

    ref_embedding = await pool.fetchval(
        """
        SELECT embedding::text FROM tracklets
        WHERE person_id = $1
        ORDER BY ended_at DESC LIMIT 1
        """,
        person_id,
    )
    if ref_embedding is None:
        return []

    rows = await pool.fetch(
        """
        SELECT p.id AS person_id, p.name, p.jabatan, p.best_thumbnail_url,
               MIN(t.embedding <=> $1::vector) AS distance
        FROM tracklets t
        JOIN persons p ON p.id = t.person_id
        WHERE p.is_known = TRUE
          AND p.id != $2
          AND t.started_at >= now() - make_interval(days => $3)
        GROUP BY p.id, p.name, p.jabatan, p.best_thumbnail_url
        ORDER BY distance ASC
        LIMIT $4
        """,
        ref_embedding, person_id, days, limit,
    )
    return [
        NameSuggestion(
            person_id=r["person_id"], name=r["name"], jabatan=r["jabatan"],
            similarity=1.0 - float(r["distance"]), thumbnail_url=r["best_thumbnail_url"],
        )
        for r in rows
    ]
