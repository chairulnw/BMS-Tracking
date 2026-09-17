from pydantic import BaseModel

DEFAULT_CONF_THRESHOLD = 0.47
ASSOC_THRESHOLD        = 0.67

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
    cpu_percent:      float | None = None
    ram_percent:      float | None = None
    last_batch_ms:    float | None = None
    avg_batch_ms:     float | None = None
    cameras_active:   int | None = None
    cameras_total:    int | None = None


class StreamStartRequest(BaseModel):
    line:                  LineConfig | None = None
    conf_threshold:        float             = DEFAULT_CONF_THRESHOLD
    reid_threshold:        float             = ASSOC_THRESHOLD
    # ponytail: true untuk evaluasi terisolasi (mis. file-playlist akurasi) —
    # gallery hari ini TIDAK dipulihkan dari DB, IdentityDB mulai kosong.
    # Default false: perilaku produksi tidak berubah.
    skip_gallery_restore:  bool              = False


class RenameRequest(BaseModel):
    new_name: str


class StreamStatusResponse(BaseModel):
    running: bool
    count_in: int
    count_out: int
    frames_processed: int
    identities: list[IdentityRecord]
    rtsp_configured: bool   # sengaja tidak expose URL-nya
