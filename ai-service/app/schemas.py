from pydantic import BaseModel

DEFAULT_CONF_THRESHOLD = 0.6
ASSOC_THRESHOLD        = 0.62  # dipakai juga sebagai default reid_threshold; lihat pipeline_service.py


class LineConfig(BaseModel):
    p1: tuple[int, int]
    p2: tuple[int, int]
    in_sign: int = 1          # +1 atau -1, ditentukan user saat klik sisi IN


class ProcessVideoRequest(BaseModel):
    video_path: str
    line: LineConfig
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    reid_threshold: float = ASSOC_THRESHOLD
    output_path: str | None = None   # auto-generate jika None


class IdentityRecord(BaseModel):
    name: str
    track_ids: list[int]


class ProcessVideoResponse(BaseModel):
    output_path: str
    count_in: int
    count_out: int
    identities: list[IdentityRecord]
    frames_processed: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class StreamStartRequest(BaseModel):
    line:           LineConfig | None = None 
    conf_threshold: float             = DEFAULT_CONF_THRESHOLD
    reid_threshold: float             = ASSOC_THRESHOLD
    # Override bobot association score (Fase 2) — kosongkan untuk pakai default
    # modul (lihat ai-service/app/services/pipeline_service.py). Dipakai T2.12.
    w_reid:         float | None      = None
    w_time:         float | None      = None
    w_cam:          float | None      = None


class RenameRequest(BaseModel):
    new_name: str


class StreamStatusResponse(BaseModel):
    running: bool
    count_in: int
    count_out: int
    frames_processed: int
    identities: list[IdentityRecord]
    rtsp_configured: bool   # sengaja tidak expose URL-nya
