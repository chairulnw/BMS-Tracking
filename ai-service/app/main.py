import builtins
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
import torchreid
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

load_dotenv()   # baca .env sebelum apapun

# ponytail: monkeypatch print() sekali di titik masuk biar semua log terminal
# (main.py + stream_service/*) kepakai timestamp, tanpa ubah puluhan call
# site satu-satu. Kalau nanti butuh level/filter, ganti ke logging module.
_print = builtins.print
def _print_with_ts(*args, **kwargs):
    _print(f"[{datetime.now().strftime('%H:%M:%S')}]", *args, **kwargs)
builtins.print = _print_with_ts

import threading

from app.auth import get_current_user
from app.routers import clips, health, identities, snapshot, stream, video
from app.schemas import DEFAULT_CONF_THRESHOLD, ASSOC_THRESHOLD
from app.services import retention
from app.services.stream_service import StreamManager

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt")
REID_MODEL = os.getenv("REID_MODEL", "osnet_ain_x1_0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[startup] device: {device}")

    app.state.yolo_model = YOLO_MODEL
    app.state.reid_model = REID_MODEL

    print(f"[startup] loading YOLO ({YOLO_MODEL})…")
    app.state.detector = YOLO(YOLO_MODEL)

    print(f"[startup] loading OSNet ({REID_MODEL})…")
    _msmt17 = Path.home() / ".cache/torch/checkpoints/osnet_ain_x1_0_msmt17.pt"
    app.state.extractor = torchreid.utils.FeatureExtractor(
        model_name=REID_MODEL,
        model_path=str(_msmt17) if _msmt17.exists() else "",
        device=device,
    )

    app.state.stream_manager = StreamManager()
    print("[startup] models ready\n")

    # ponytail: dipakai run_comparison.py — auto-start di boot (tanpa
    # skip_gallery_restore) sempat nulis ke DB beneran sebelum sempat
    # di-/stream/stop, nyampur tracklet antar kombinasi (beda dimensi
    # embedding per model Re-ID). DISABLE_AUTOSTART matiin itu buat evaluasi.
    if os.getenv("DISABLE_AUTOSTART"):
        print("[startup] auto-start dilewati (DISABLE_AUTOSTART)")
    else:
        try:
            app.state.stream_manager.start(
                yolo_model=YOLO_MODEL, reid_model=REID_MODEL,
                conf_threshold=DEFAULT_CONF_THRESHOLD, reid_threshold=ASSOC_THRESHOLD,
            )
            print("[startup] stream auto-started")
        except RuntimeError as exc:
            print(f"[startup] stream tidak auto-start: {exc}")

    retention_stop = threading.Event()
    retention.start_background(retention_stop)

    yield

    print("[shutdown] stopping stream (if running)…")
    retention_stop.set()
    app.state.stream_manager.stop()
    app.state.detector  = None
    app.state.extractor = None


app = FastAPI(
    title="BMS AI Service",
    description="Person detection, tracking, ReID, dan line crossing untuk sistem kamera pengawas.",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(video.router, dependencies=[Depends(get_current_user)])
app.include_router(identities.router, dependencies=[Depends(get_current_user)])
app.include_router(stream.router, dependencies=[Depends(get_current_user)])
app.include_router(snapshot.router, dependencies=[Depends(get_current_user)])
app.include_router(clips.router, dependencies=[Depends(get_current_user)])
