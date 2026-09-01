from datetime import date, datetime
from pydantic import BaseModel


# ── Persons ───────────────────────────────────────────────────────────────────

class PersonResponse(BaseModel):
    id:                 int
    name:               str
    label:              str
    first_seen:         datetime | None
    last_seen:          datetime | None
    last_camera:        str | None
    last_camera_name:   str | None = None
    last_zone_name:     str | None = None
    best_thumbnail_url: str | None
    is_known:           bool
    enrollment_date:    date | None
    jabatan:            str | None = None
    observation_count:  int = 0


class DetectionRecord(BaseModel):
    id:         int
    camera_id:  str
    timestamp:  datetime
    confidence: float
    method:     str


class PersonDetail(PersonResponse):
    detections: list[DetectionRecord]


class RenameRequest(BaseModel):
    new_name: str


class PersonUpdate(BaseModel):
    name:    str | None = None
    jabatan: str | None = None


class PeopleFeedItem(BaseModel):
    detection_id:  int
    person_id:     int | None
    person_label:  str | None
    person_name:   str | None = None
    is_known:      bool = False
    camera_id:       str
    camera_name:     str | None
    camera_location: str | None = None
    timestamp:       datetime
    thumbnail_url: str | None
    tracklet_id:   int | None = None  # None = deteksi ini belum punya tracklet tertaut (appearance search tidak tersedia untuknya)


class PeopleFeedResponse(BaseModel):
    items: list[PeopleFeedItem]
    total: int
    page:  int
    pages: int
    limit: int


# Sama persis strukturnya dengan PeopleFeedResponse — tapi `items` di sini
# satu baris per PERSON (deteksi terakhirnya), bukan satu baris per deteksi.
# Tipe terpisah supaya frontend eksplisit soal endpoint mana yang dipanggil,
# walau isinya identik dengan PeopleFeedResponse.
class PersonsFeedResponse(BaseModel):
    items: list[PeopleFeedItem]
    total: int
    page:  int
    pages: int
    limit: int


class MovementRecord(BaseModel):
    timestamp:    datetime
    location:     str
    floor:        str
    camera:       str
    isCurrent:    bool
    note:         str
    event_kind:   str | None = None   # "room_entry" | "passage"
    direction:    str | None = None   # "IN" | "OUT"
    snapshot_url: str | None = None
    track_id:     int | None = None


# ── Camera groups ────────────────────────────────────────────────────────────

class CameraGroupIn(BaseModel):
    name: str


class CameraGroupResponse(CameraGroupIn):
    id:         int
    created_at: datetime


# ── Cameras (CRUD) ───────────────────────────────────────────────────────────

class CameraIn(BaseModel):
    camera_id:          str | None  = None
    location:            str | None  = None
    group_id:           int | None  = None
    rtsp_url:           str
    is_active:          bool        = True
    analytics_enabled:  bool        = True


class CameraGroupAssign(BaseModel):
    group_id: int | None = None


class CameraResponse(BaseModel):
    id:                 int
    camera_id:          str | None
    name:               str
    location:           str | None
    group_id:           int | None
    group_name:         str | None = None
    rtsp_url:           str
    is_active:          bool
    analytics_enabled:  bool
    zone_count:         int = 0
    created_at:         datetime
    health_status:      str | None = None       # 'online' | 'offline' | None (belum pernah lapor)
    last_seen:          datetime | None = None  # timestamp event health terakhir


# ── Zones (line or polygon occupancy areas) ────────────────────────────────────

class ZoneIn(BaseModel):
    """Buat zona baru — wajib langsung bawa minimal 1 sumber kamera + gambarnya,
    gak ada zona kosong. Tipe (line/polygon) melekat ke gambar ini, bukan ke zona."""
    name:          str
    max_capacity:  int | None = None
    camera_id:     str            # cameras.camera_id (string) — sumber pertama
    type:          str            # "line" | "polygon" — tipe gambar pertama ini
    points:        list[dict]


class ZoneUpdate(BaseModel):
    name:         str
    max_capacity: int | None = None


class ZoneResponse(BaseModel):
    id:                 int
    name:               str
    max_capacity:       int | None
    camera_count:       int = 0
    camera_ids:         list[str] = []   # cameras.camera_id (string) — dipakai utk filter
    types:              list[str] = []   # tipe gambar unik yang ada di zona ini
    created_at:         datetime


class ZoneCameraIn(BaseModel):
    type:   str          # "line" | "polygon" — ditanyakan tiap kali mau gambar
    points: list[dict]   # line: [{"p1":{},"p2":{},"in_sign":1}, ...]; polygon: [{"x":,"y":}, ...]


class ZoneCameraResponse(BaseModel):
    id:            int
    zone_id:       int
    camera_id:     int
    camera_str_id: str | None = None   # cameras.camera_id (string) — dipakai utk snapshot URL
    camera_name:   str | None = None
    type:          str
    points:        list[dict]


class ZoneDetailResponse(ZoneResponse):
    cameras: list[ZoneCameraResponse] = []


class ZoneForCameraResponse(BaseModel):
    """Geometri satu gambar zona untuk satu kamera tertentu — dipakai AI service saat stream start."""
    zone_id:        int
    zone_camera_id: int
    name:           str
    type:           str
    points:         list[dict]
    max_capacity:   int | None


class ZoneHistoryPoint(BaseModel):
    bucket:    str    # label siap-tampil: "14:00" (granularity=hour) atau "23/08" (granularity=day)
    count_in:  int
    count_out: int


class ZoneHeatmapResponse(BaseModel):
    grid_size: int
    cells:     list[list[int]]


class ZoneEventResponse(BaseModel):
    id:           int
    timestamp:    datetime
    direction:    str
    person_label: str | None
    snapshot_url: str | None
    camera_id:    str | None
    camera_name:  str | None


class PersonCrossingResponse(BaseModel):
    id:           int
    timestamp:    datetime
    direction:    str
    camera_id:    str | None
    camera_name:  str | None
    zone_name:    str | None
    snapshot_url: str | None


class ZoneOccupantResponse(BaseModel):
    person_id:     int | None
    person_label:  str | None   # None = tracklet belum resolve identitasnya saat crossing terjadi
    person_name:   str | None
    is_known:      bool = False
    since:         datetime      # timestamp event IN terakhirnya
    camera_name:   str | None
    thumbnail_url: str | None


# ── Occupancy ─────────────────────────────────────────────────────────────────

class OccupancyEventCreate(BaseModel):
    camera_id:      str
    zone_camera_id: int | None = None
    direction:      str               # "IN" or "OUT"
    timestamp:      datetime | None = None
    event_kind:     str | None      = None   # "room_entry" | "passage" | "crossing"
    snapshot_url:   str | None      = None
    person_label:   str | None      = None
    track_id:       int | None      = None
    point_x:        int | None      = None
    point_y:        int | None      = None


class OccupancyResponse(BaseModel):
    zone_id:            int
    zone_name:          str
    max_capacity:       int | None
    count_in:           int
    count_out:          int
    current_occupancy:  int     # count_in - count_out


# ── Camera events ────────────────────────────────────────────────────────────

class CameraEventCreate(BaseModel):
    camera_id:    str
    event_type:   str
    category:     str = "info"
    description:  str | None = None
    snapshot_url: str | None = None
    person_label: str | None = None
    timestamp:    datetime | None = None


class CameraEventResponse(BaseModel):
    id:           int
    camera_id:    str
    camera_name:  str | None
    event_type:   str
    category:     str
    description:  str | None
    snapshot_url: str | None
    person_label: str | None
    timestamp:    datetime
    created_at:   datetime   # jam backend insert — buat hitung network/backend delay vs `timestamp` (jam AI)
    acknowledged:    bool
    acknowledged_at: datetime | None


class StatsToday(BaseModel):
    cameras_total:    int
    cameras_active:   int
    cameras_inactive: int
    events_today:     int
    detections_today: int
    persons_today:    int
    persons_known:    int
    persons_unknown:  int


# ── Detections ────────────────────────────────────────────────────────────────

class DetectionCreate(BaseModel):
    person_name: str           # display name from AI service (may be renamed)
    person_label: str | None = None  # original AI label if known; falls back to person_name
    camera_id:   str
    timestamp:   datetime | None = None
    confidence:  float
    method:      str = "appearance"  # face / appearance / unknown
    thumbnail_url: str | None = None
    track_id:    int | None = None


class DetectionResponse(BaseModel):
    id:         int
    person_id:  int
    camera_id:  str
    timestamp:  datetime
    created_at: datetime   # jam backend insert — buat hitung network/backend delay vs `timestamp` (jam AI)
    confidence: float
    method:     str


class ThumbnailUpdate(BaseModel):
    person_label:  str
    camera_id:     str
    thumbnail_url: str


# ── Tracklets ────────────────────────────────────────────────────────────────

class TrackletCreate(BaseModel):
    camera_id:          str
    track_id:            int
    person_label:        str | None = None   # None = Unassociated (di bawah threshold)
    started_at:           datetime
    ended_at:             datetime
    n_detections:         int
    best_thumbnail_url:   str | None = None
    embedding:            list[float]         # 3840-dim TransReID (768*5, JPM+global), L2-normalized
    assoc_score:          float | None = None
    attrs:                dict | None = None  # skor mentah PAR (Fase 3), None kalau PAR nonaktif
    pos_x:                int | None = None  # titik kaki sampel ber-confidence tertinggi (fallback lama)
    pos_y:                int | None = None
    positions:            list[list[int]] = []  # [[x,y], ...] seluruh titik kaki, urut waktu (Fase 4, Pergerakan)


class TrackletResponse(BaseModel):
    id:                  int
    camera_id:           str
    track_id:            int
    person_id:           int | None
    started_at:          datetime
    ended_at:            datetime
    n_detections:        int
    best_thumbnail_url:  str | None
    assoc_score:         float | None


class TrackletGalleryEntry(BaseModel):
    """Satu baris gallery untuk dipulihkan ke IdentityDB saat stream/start —
    embedding + label + rentang waktu terakhir per person_id hari ini."""
    person_id:    int
    person_label: str
    camera_id:    str
    started_at:   datetime
    ended_at:     datetime
    embedding:    list[float]


class TrajectoryPoint(BaseModel):
    """Satu singgahan kamera di urutan pergerakan orang (Fase 4, T4.2)."""
    camera_id:   str
    camera_name: str | None
    zone_name:   str | None
    started_at:  datetime
    ended_at:    datetime


class DwellRecord(BaseModel):
    """Total waktu tinggal, dari tracklets per kamera ATAU pasangan IN/OUT
    occupancy_events per zona line-crossing (T4.3). `kind` bedain sumbernya —
    'camera' (selalu ada, gak butuh zona) vs 'zone' (cuma zona line-crossing;
    polygon sengaja gak dihitung di sini, itu tumpang tindih sama durasi
    kamera-nya sendiri)."""
    zone_name:     str
    kind:          str  # "camera" | "zone"
    dwell_seconds: float


class CameraPoint(BaseModel):
    """Satu titik lintas (crossing) orang ini di ruang koordinat piksel asli
    satu kamera — dipakai buat overlay trajectory/heatmap di atas snapshot kamera."""
    camera_id:   str
    camera_name: str | None
    x:           int
    y:           int
    direction:   str
    timestamp:   datetime


class NameSuggestion(BaseModel):
    """Kandidat nama untuk orang yang belum dikenali, dari kemiripan embedding
    terhadap orang yang SUDAH bernama N hari terakhir (Fase 3). Operator yang
    memutuskan — tidak pernah diterapkan otomatis."""
    person_id:     int
    name:          str
    jabatan:       str | None
    similarity:    float
    thumbnail_url: str | None


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
