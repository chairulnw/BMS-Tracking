from datetime import date

from fastapi import APIRouter, Query, Request

from app.schemas import PeopleFeedItem, PeopleFeedResponse, PersonsFeedResponse

router = APIRouter(prefix="/people", tags=["people"])


ATTR_THRESHOLD = 0.5  # skor mentah tracklets.attrs >= ini dianggap "punya atribut"
# Nama kunci di tracklets.attrs = string asli RAP1_ATTR_WORDS (lihat
# ai-service/app/par/par_service.py SELECTED_ATTRS), bukan nama ramah.
GENDER_ATTR = "female"


async def _build_filters(
    pool,
    q:           str | None,
    camera_id:   str | None,
    zone_id:     int | None,
    person_id:   int | None,
    from_date:   date | None,
    to_date:     date | None,
    upper_color: str | None,
    lower_color: str | None,
    gender:      str | None,
    attrs:       list[str] | None,
    similar_to:  int | None,
) -> tuple[list[str], list, str, str, "str | None"]:
    """Bangun WHERE/params/join yang dipakai bareng oleh /feed (per-deteksi) dan
    /persons (per-orang) — kedua endpoint menerima filter yang persis sama."""
    conditions: list[str] = []
    params: list = []
    needs_tracklet_join = any(v is not None for v in (upper_color, lower_color, gender)) \
        or bool(attrs) or similar_to is not None

    ref_embedding = None
    if similar_to is not None:
        ref_embedding = await pool.fetchval(
            "SELECT embedding::text FROM tracklets WHERE id = $1", similar_to
        )

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

    return conditions, params, where, tracklet_join, ref_embedding


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
    similar_to:  int | None  = Query(None, description="tracklet_id acuan — urutkan berdasar kemiripan penampilan (KNN pgvector)"),
    page:        int         = Query(1, ge=1),
    limit:       int         = Query(12, ge=1, le=100),
) -> PeopleFeedResponse:
    """Feed deteksi untuk halaman People. Tanpa filter = recent detections.
    `q`/`person_id` mempersempit ke satu orang. Filter atribut
    (upper_color/lower_color/gender/attrs) butuh tracklets.attrs terisi
    (hasil PAR) — deteksi yang belum punya tracklet tertaut otomatis tidak
    lolos filter atribut apa pun (LEFT JOIN, attrs NULL). `attrs` generik
    untuk grup PAR lain (aksesoris dll) — OR di dalam satu `attrs=`, AND
    antar `attrs=` berbeda. `similar_to` mengubah urutan hasil dari
    terbaru → terdekat embedding-nya (appearance-by-example) — filter lain
    tetap berlaku bersamaan."""
    pool = request.app.state.pool

    conditions, params, where, tracklet_join, ref_embedding = await _build_filters(
        pool, q, camera_id, zone_id, person_id, from_date, to_date,
        upper_color, lower_color, gender, attrs, similar_to,
    )
    if similar_to is not None and ref_embedding is None:
        return PeopleFeedResponse(items=[], total=0, page=page, pages=1, limit=limit)

    offset = (page - 1) * limit

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
        params = [*params, ref_embedding]
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
            c.location      AS camera_location,
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


@router.get("/persons", response_model=PersonsFeedResponse)
async def people_persons(
    request:     Request,
    q:           str | None  = Query(None, description="Cari nama/label orang"),
    camera_id:   str | None  = Query(None, description="Boleh multi dipisah koma (OR) — 'c1,c2'"),
    zone_id:     int | None  = Query(None, description="Filter ke kamera yang tertaut ke zone ini"),
    from_date:   date | None = Query(None, alias="from"),
    to_date:     date | None = Query(None, alias="to"),
    upper_color: str | None  = Query(None, description="Warna atasan, boleh multi dipisah koma (OR) — 'red,blue'"),
    lower_color: str | None  = Query(None, description="Warna bawahan, boleh multi dipisah koma (OR)"),
    gender:      str | None  = Query(None, description="'male' atau 'female', dari skor atribut PAR 'female'"),
    attrs:       list[str] | None = Query(None, description="Grup atribut PAR generik, sama seperti /feed"),
    similar_to:  int | None  = Query(None, description="tracklet_id acuan — tiap orang diranking pakai deteksi TERDEKATNYA ke acuan ini"),
    page:        int         = Query(1, ge=1),
    limit:       int         = Query(6, ge=1, le=100),
) -> PersonsFeedResponse:
    """Sama filternya dengan /feed, tapi paginasi per ORANG (bukan per
    deteksi) — satu baris per person_id, pakai deteksi terbarunya (atau
    terdekat ke `similar_to` kalau dipakai). Independen dari paginasi
    /feed: total/pages di sini dihitung dari jumlah distinct person, bukan
    jumlah deteksi, jadi frontend bisa punya dua kontrol halaman terpisah
    untuk section Orang dan Deteksi."""
    pool = request.app.state.pool

    conditions, params, where, tracklet_join, ref_embedding = await _build_filters(
        pool, q, camera_id, zone_id, None, from_date, to_date,
        upper_color, lower_color, gender, attrs, similar_to,
    )
    if similar_to is not None and ref_embedding is None:
        return PersonsFeedResponse(items=[], total=0, page=page, pages=1, limit=limit)

    offset = (page - 1) * limit

    total = await pool.fetchval(
        f"""
        SELECT COUNT(DISTINCT d.person_id)
        FROM detections d
        JOIN persons p ON p.id = d.person_id
        {tracklet_join}
        {where}
        """,
        *params,
    )

    if similar_to is not None:
        pick_order = f"t.embedding <=> ${len(params) + 1}::vector"
        params = [*params, ref_embedding]
    else:
        pick_order = "d.timestamp DESC"

    # Kalau similar_to: rank ORANG berdasar seberapa dekat deteksi TERBAIKNYA
    # (yang dipilih di CTE `latest`) ke embedding acuan — bukan waktu.
    outer_order = "l.dist ASC" if similar_to is not None else "l.timestamp DESC"
    dist_select = f", {pick_order} AS dist" if similar_to is not None else ", NULL::float AS dist"

    rows = await pool.fetch(
        f"""
        WITH latest AS (
            SELECT DISTINCT ON (d.person_id)
                d.id AS detection_id, d.person_id, d.camera_id, d.timestamp,
                d.thumbnail_url, d.tracklet_id
                {dist_select}
            FROM detections d
            JOIN persons p ON p.id = d.person_id
            {tracklet_join}
            {where}
            ORDER BY d.person_id, {pick_order}
        )
        SELECT
            l.detection_id  AS detection_id,
            l.person_id     AS person_id,
            p.label         AS person_label,
            p.name          AS person_name,
            p.is_known      AS is_known,
            l.camera_id     AS camera_id,
            c.name          AS camera_name,
            c.location      AS camera_location,
            l.timestamp     AS timestamp,
            l.thumbnail_url AS thumbnail_url,
            l.tracklet_id   AS tracklet_id
        FROM latest l
        JOIN persons p       ON p.id = l.person_id
        LEFT JOIN cameras c  ON c.camera_id = l.camera_id
        ORDER BY {outer_order}
        LIMIT {limit} OFFSET {offset}
        """,
        *params,
    )

    items = [PeopleFeedItem(**dict(r)) for r in rows]
    total_int = int(total or 0)
    return PersonsFeedResponse(
        items=items,
        total=total_int,
        page=page,
        pages=max(1, (total_int + limit - 1) // limit),
        limit=limit,
    )
