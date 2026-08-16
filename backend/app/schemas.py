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


# ── Cameras (CRUD) ───────────────────────────────────────────────────────────

class CameraIn(BaseModel):
    camera_id:     str | None  = None
    name:          str
    zone_location: str | None  = None
    floor:         str | None  = None
    rtsp_url:      str
    is_active:     bool        = True


class CameraResponse(CameraIn):
    id:         int
    created_at: datetime


class SaveZoneRequest(BaseModel):
    room_name: str
    floor:     str | None = None


# ── Crossing lines ────────────────────────────────────────────────────────────

class CrossingLineIn(BaseModel):
    p1_x:    int
    p1_y:    int
    p2_x:    int
    p2_y:    int
    in_sign: int = 1


class CrossingLineResponse(CrossingLineIn):
    id:         int
    camera_id:  str
    created_at: datetime


# ── Occupancy ─────────────────────────────────────────────────────────────────

class OccupancyEventCreate(BaseModel):
    camera_id:    str
    line_id:      int
    direction:    str               # "IN" or "OUT"
    timestamp:    datetime | None = None
    event_kind:   str | None      = None   # "room_entry" | "passage" | "crossing"
    snapshot_url: str | None      = None
    person_label: str | None      = None
    track_id:     int | None      = None


class OccupancyResponse(BaseModel):
    camera_id:          str
    room_name:          str
    floor:              str | None
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
