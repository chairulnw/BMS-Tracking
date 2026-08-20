from datetime import date, datetime

from fastapi import APIRouter, Query, Request

from app.schemas import PeopleFeedItem, PeopleFeedResponse

router = APIRouter(prefix="/people", tags=["people"])


ATTR_THRESHOLD = 0.5  # skor mentah tracklets.attrs >= ini dianggap "punya atribut"
# Nama kunci di tracklets.attrs = string asli RAP1_ATTR_WORDS (lihat
# ai-service/app/par/par_service.py SELECTED_ATTRS), bukan nama ramah.
GENDER_ATTR = "female"


@router.get("/feed", response_model=PeopleFeedResponse)
async def people_feed(
    request:     Request,
    q:           str | None  = Query(None, description="Cari nama/label orang pemilik deteksi"),
    camera_id:   str | None  = Query(None, description="Boleh multi dipisah koma (OR) — 'c1,c2'"),
    zone_id:     int | None  = Query(None, description="Filter ke kamera yang tertaut ke zone ini"),
    person_id:   int | None  = Query(None, description="Semua deteksi satu orang — dipakai tab Timeline"),
    from_date:   date | None = Query(None, alias="from"),
    to_date:     date | None = Query(None, alias="to"),
    upper_color: str | None  = Query(None, description="Warna atasan, boleh multi dipisah koma (OR) — 'red,blue'"),
    lower_color: str | None  = Query(None, description="Warna bawahan, boleh multi dipisah koma (OR)"),
    gender:      str | None  = Query(None, description="'male' atau 'female', dari skor atribut PAR 'female'"),
    attrs:       list[str] | None = Query(None, description="Grup atribut PAR generik (mis. aksesoris: tas/topi/kacamata). Tiap value dipisah koma untuk OR di dalam grup; beberapa `attrs=` di-AND satu sama lain"),
    similar_to:  int | None  = Query(None, description="tracklet_id acuan — urutkan berdasar kemiripan penampilan (Fase 3, KNN pgvector)"),
    page:        int         = Query(1, ge=1),
    limit:       int         = Query(12, ge=1, le=100),
) -> PeopleFeedResponse:
    """Feed deteksi untuk halaman People. Tanpa filter = recent detections.
    `q`/`person_id` mempersempit ke satu orang. Filter atribut
    (upper_color/lower_color/gender/attrs) butuh tracklets.attrs terisi
    (Fase 3, PAR) — deteksi yang belum punya tracklet tertaut otomatis tidak
    lolos filter atribut apa pun (LEFT JOIN, attrs NULL). `attrs` generik
    untuk grup PAR lain (aksesoris dll) — OR di dalam satu `attrs=`, AND
    antar `attrs=` berbeda. `similar_to` mengubah urutan hasil dari
    terbaru → terdekat embedding-nya (appearance-by-example) — filter lain
    tetap berlaku bersamaan."""
    pool = request.app.state.pool

    conditions: list[str] = []
    params: list = []
    needs_tracklet_join = any(v is not None for v in (upper_color, lower_color, gender)) \
        or bool(attrs) or similar_to is not None

    ref_embedding = None
    if similar_to is not None:
        ref_embedding = await pool.fetchval(
            "SELECT embedding::text FROM tracklets WHERE id = $1", similar_to
        )
        if ref_embedding is None:
            return PeopleFeedResponse(items=[], total=0, page=page, pages=1, limit=limit)

    if camera_id:
        values = [v.strip() for v in camera_id.split(",") if v.strip()]
        if values:
            params.append(values)
            conditions.append(f"d.camera_id = ANY(${len(params)}::text[])")
    if zone_id is not None:
        params.append(zone_id)
        conditions.append(f"""EXISTS (
            SELECT 1 FROM zone_cameras zc2
            JOIN cameras c2 ON c2.id = zc2.camera_id
            WHERE c2.camera_id = d.camera_id AND zc2.zone_id = ${len(params)}
        )""")
    if person_id is not None:
        params.append(person_id)
        conditions.append(f"d.person_id = ${len(params)}")
    if q:
        params.append(f"%{q}%")
        conditions.append(f"(p.name ILIKE ${len(params)} OR p.label ILIKE ${len(params)})")
    if from_date:
        params.append(from_date)
        conditions.append(f"(d.timestamp AT TIME ZONE 'Asia/Jakarta')::date >= ${len(params)}")
    if to_date:
        params.append(to_date)
        conditions.append(f"(d.timestamp AT TIME ZONE 'Asia/Jakarta')::date <= ${len(params)}")
    if upper_color:
        values = [v.strip() for v in upper_color.split(",") if v.strip()]
        if values:
            params.append(values)
            conditions.append(f"t.attrs->>'upper_color' = ANY(${len(params)}::text[])")
    if lower_color:
        values = [v.strip() for v in lower_color.split(",") if v.strip()]
        if values:
            params.append(values)
            conditions.append(f"t.attrs->>'lower_color' = ANY(${len(params)}::text[])")
    if gender:
        op = ">=" if gender == "female" else "<"
        conditions.append(f"COALESCE((t.attrs->>'{GENDER_ATTR}')::float, 0) {op} {ATTR_THRESHOLD}")
    if attrs:
        for group in attrs:
            names = [n.strip() for n in group.split(",") if n.strip()]
            if not names:
                continue
            ors = []
            for name in names:
                params.append(name)
                ors.append(f"COALESCE((t.attrs->>${len(params)})::float, 0) >= {ATTR_THRESHOLD}")
            conditions.append("(" + " OR ".join(ors) + ")")
    if similar_to is not None:
        params.append(similar_to)
        conditions.append(f"d.tracklet_id != ${len(params)}")

    where         = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    join_kind     = "JOIN" if similar_to is not None else "LEFT JOIN"  # similar_to butuh embedding, tidak boleh NULL
    tracklet_join = f"{join_kind} tracklets t ON t.id = d.tracklet_id" if needs_tracklet_join else ""
    offset        = (page - 1) * limit

    # COUNT tidak butuh urutan, jadi dijalankan dengan params SEBELUM
    # ref_embedding ditambahkan — kalau tidak, jumlah parameter yang dikirim
    # tidak cocok dengan placeholder $N yang benar-benar dipakai di SQL COUNT.
    total = await pool.fetchval(
        f"""
        SELECT COUNT(*)
        FROM detections d
        JOIN persons p ON p.id = d.person_id
        {tracklet_join}
        {where}
        """,
        *params,
    )

    if similar_to is not None:
        params.append(ref_embedding)
        order_clause = f"t.embedding <=> ${len(params)}::vector"
    else:
        order_clause = "d.timestamp DESC"

    rows = await pool.fetch(
        f"""
        SELECT
            d.id            AS detection_id,
            d.person_id     AS person_id,
            p.label         AS person_label,
            p.name          AS person_name,
            p.is_known      AS is_known,
            d.camera_id     AS camera_id,
            c.name          AS camera_name,
            d.timestamp     AS timestamp,
            d.thumbnail_url AS thumbnail_url,
            d.tracklet_id   AS tracklet_id
        FROM detections d
        JOIN persons p       ON p.id = d.person_id
        LEFT JOIN cameras c  ON c.camera_id = d.camera_id
        {tracklet_join}
        {where}
        ORDER BY {order_clause}
        LIMIT {limit} OFFSET {offset}
        """,
        *params,
    )

    items = [PeopleFeedItem(**dict(r)) for r in rows]
    total_int = int(total or 0)
    return PeopleFeedResponse(
        items=items,
        total=total_int,
        page=page,
        pages=max(1, (total_int + limit - 1) // limit),
        limit=limit,
    )
