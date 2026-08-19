import json
import os
from contextlib import asynccontextmanager

import asyncpg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from app.auth import get_current_user
from app.routers import auth, camera_events, cameras, detections, people, persons, thumbnails, tracklets, zones


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog", format="text"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"], min_size=2, max_size=10, init=_init_connection
    )
    # Backfill camera_id for rows added without it (extract from unicast/ RTSP path)
    await app.state.pool.execute(
        """
        UPDATE cameras
           SET camera_id = (regexp_match(rtsp_url, '/unicast/([^/]+)'))[1]
         WHERE camera_id IS NULL
           AND rtsp_url ~ '/unicast/'
        """
    )
    yield
    await app.state.pool.close()


app = FastAPI(title="BMS-IIP Backend", version="1.0.0", lifespan=lifespan)

cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:4200").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(persons.router, dependencies=[Depends(get_current_user)])
app.include_router(detections.router, dependencies=[Depends(get_current_user)])
app.include_router(thumbnails.router, dependencies=[Depends(get_current_user)])
app.include_router(cameras.router, dependencies=[Depends(get_current_user)])
app.include_router(camera_events.router, dependencies=[Depends(get_current_user)])
app.include_router(zones.router, dependencies=[Depends(get_current_user)])
app.include_router(people.router, dependencies=[Depends(get_current_user)])
app.include_router(tracklets.router, dependencies=[Depends(get_current_user)])


@app.get("/health")
async def health():
    return {"status": "ok"}
