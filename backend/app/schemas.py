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
    best_thumbnail_url: str | None
    is_known:           bool
    enrollment_date:    date | None


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
    date:      date
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
    confidence: float
    method:     str


class ThumbnailUpdate(BaseModel):
    person_label:  str
    camera_id:     str
    thumbnail_url: str


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
