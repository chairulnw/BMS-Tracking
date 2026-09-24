from datetime import date, datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query, Request

from app.schemas import TrackletCreate, TrackletGalleryEntry, TrackletResponse

router = APIRouter(prefix="/tracklets", tags=["tracklets"])


def _to_vector_literal(emb: list[float]) -> str:
    return "[" + ",".join(str(x) for x in emb) + "]"


def _parse_vector(raw: str) -> list[float]:
    return [float(x) for x in raw.strip("[]").split(",")]


@router.post("", response_model=TrackletResponse, status_code=201)
async def create_tracklet(req: TrackletCreate, request: Request) -> TrackletResponse:
    """Dipanggil AI service saat tracklet ditutup.
    `person_id` diresolusi dari `person_label` lewat lookup sederhana — pembuatan
    baris `persons` sepenuhnya jadi tanggung jawab `POST /detections`, yang
    selalu dikirim lebih dulu."""
    pool = request.app.state.pool

    person_id = None
    if req.person_label:
        person_id = await pool.fetchval(
            "SELECT id FROM persons WHERE label = $1", req.person_label
        )

    row = await pool.fetchrow(
        """
        INSERT INTO tracklets
            (camera_id, track_id, person_id, started_at, ended_at, n_detections,
             best_thumbnail_url, embedding, assoc_score, attrs, pos_x, pos_y, positions)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9, $10, $11, $12, $13)
        RETURNING id, camera_id, track_id, person_id, started_at, ended_at,
                  n_detections, best_thumbnail_url, assoc_score
        """,
        req.camera_id, req.track_id, person_id, req.started_at, req.ended_at,
        req.n_detections, req.best_thumbnail_url,
        _to_vector_literal(req.embedding), req.assoc_score, req.attrs,
        req.pos_x, req.pos_y, req.positions,
    )

    # detections.tracklet_id tidak pernah dikirim AI service (POST /detections
    # dan POST /tracklets adalah dua request independen, tidak ada id bersama
    # di antara keduanya). Sambungkan di sini: POST /detections SELALU dikirim
    # lebih dulu untuk tracklet yang sama, dan _post_queue satu worker FIFO
    # menjamin urutan tiba di DB tetap sama — jadi deteksi ter-unlink TERBARU
    # milik kamera+orang ini pasti pasangannya.
    if person_id is not None:
        await pool.execute(
            """
            UPDATE detections SET tracklet_id = $1
            WHERE id = (
                SELECT id FROM detections
                WHERE camera_id = $2 AND person_id = $3 AND tracklet_id IS NULL
                ORDER BY timestamp DESC LIMIT 1
            )
            """,
            row["id"], req.camera_id, person_id,
        )

        # Crossing events dikirim sebelum tracklet ditutup, jadi person_label-nya
        # masih NULL. Backfill sekarang, longgar ±2 detik karena crossing di
        # frame terakhir bisa ke-timestamp sedikit setelah ended_at.
        await pool.execute(
            """
            UPDATE occupancy_events
            SET person_label = $1
            WHERE camera_id = $2 AND track_id = $3
              AND person_label IS NULL
              AND timestamp BETWEEN $4::timestamptz - interval '2 seconds'
                                AND $5::timestamptz + interval '2 seconds'
            """,
            req.person_label, req.camera_id, req.track_id, req.started_at, req.ended_at,
        )

    return dict(row)


@router.get("/gallery", response_model=list[TrackletGalleryEntry])
async def gallery(
    request: Request,
    date_filter: date | None = Query(None, alias="date"),
) -> list[TrackletGalleryEntry]:
    """Dipanggil AI service saat `stream/start` untuk memulihkan gallery ReID.
    Hanya tracklet **hari ini** (Asia/Jakarta) dan yang sudah beridentitas —
    label person tidak menembus batas hari. Maks 5 embedding terbaru per orang, mengikuti
    MAX_BANK_SIZE di AI service."""
    pool = request.app.state.pool
    if date_filter is None:
        date_filter = datetime.now(ZoneInfo("Asia/Jakarta")).date()

    rows = await pool.fetch(
        """
        SELECT t.person_id, p.label AS person_label, t.camera_id,
               t.started_at, t.ended_at, t.embedding::text AS embedding
        FROM tracklets t
        JOIN persons p ON p.id = t.person_id
        WHERE t.person_id IS NOT NULL
          AND (t.started_at AT TIME ZONE 'Asia/Jakarta')::date = $1
        ORDER BY t.person_id, t.ended_at DESC
        """,
        date_filter,
    )

    per_person_count: dict[int, int] = {}
    entries: list[TrackletGalleryEntry] = []
    for r in rows:
        pid = r["person_id"]
        n   = per_person_count.get(pid, 0)
        if n >= 5:   # MAX_BANK_SIZE di ai-service/app/services/pipeline_service.py
            continue
        per_person_count[pid] = n + 1
        entries.append(TrackletGalleryEntry(
            person_id=pid, person_label=r["person_label"], camera_id=r["camera_id"],
            started_at=r["started_at"], ended_at=r["ended_at"],
            embedding=_parse_vector(r["embedding"]),
        ))
    return entries
