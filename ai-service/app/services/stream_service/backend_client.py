"""Client HTTP ke backend (fetch kamera/zona, post deteksi/event). Semua POST
dikirim lewat satu worker thread background (bukan langsung dari pemanggil)
supaya thread inferensi (BatchProcessor._loop) tidak pernah menunggu round-trip
network — sebelumnya requests.post(timeout=0.8) dipanggil inline di loop utama,
jadi tiap event crossing/deteksi bisa nahan pemrosesan kamera lain sampai 0.8 detik."""

import os
import queue
import threading
from datetime import datetime, timezone
from typing import Callable

import requests

from app.auth import create_service_token
from app.services.pipeline_service import par_attrs


def _auth_headers() -> dict:
    """Header dipakai ai-service untuk memanggil endpoint backend yang dilindungi login."""
    return {"Authorization": f"Bearer {create_service_token()}"}


def _backend_url() -> str:
    return os.getenv("BACKEND_URL", "http://localhost:8002")


def _fetch_zones(camera_id: str) -> list[dict]:
    """Ambil semua zona (line/polygon) yang dipantau kamera ini, lengkap dengan
    geometri (zone_camera_id, type, points, max_capacity)."""
    try:
        r = requests.get(f"{_backend_url()}/zones/for-camera/{camera_id}", headers=_auth_headers(), timeout=3.0)
        if r.ok:
            return r.json()
    except Exception as exc:
        print(f"[{camera_id}] gagal load zones: {exc}")
    return []


def _fetch_cameras() -> list[dict]:
    """Ambil kamera aktif dari database via backend. Return [] jika gagal."""
    try:
        r = requests.get(f"{_backend_url()}/cameras", params={"is_active": "true"}, headers=_auth_headers(), timeout=5.0)
        if r.ok:
            return r.json()
    except Exception as exc:
        print(f"[stream] fetch cameras from DB failed: {exc}")
    return []


def _fetch_tracklet_gallery() -> list[dict]:
    """Ambil gallery tracklet hari ini (embedding + label per orang) untuk
    memulihkan IdentityDB saat stream/start (plan/07-fase2-detail.md §7).
    started_at/ended_at diparse jadi datetime aware; embedding tetap list[float]."""
    from datetime import datetime as _dt
    try:
        r = requests.get(f"{_backend_url()}/tracklets/gallery", headers=_auth_headers(), timeout=5.0)
        if not r.ok:
            return []
        entries = r.json()
        for e in entries:
            e["started_at"] = _dt.fromisoformat(e["started_at"])
            e["ended_at"]   = _dt.fromisoformat(e["ended_at"])
        return entries
    except Exception as exc:
        print(f"[stream] fetch tracklet gallery failed: {exc}")
        return []


class _PostQueue:
    """Antrean fire-and-forget: pemanggil cuma enqueue closure, satu worker
    thread background yang benar-benar melakukan request HTTP-nya."""

    def __init__(self, maxsize: int = 500) -> None:
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=maxsize)
        self._started = False
        self._lock = threading.Lock()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._started:
                return
            threading.Thread(target=self._drain, daemon=True, name="backend-post").start()
            self._started = True

    def _drain(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job()
            except Exception as exc:
                print(f"[backend] post error: {exc}")

    def submit(self, job: Callable[[], None]) -> None:
        self._ensure_worker()
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            print("[backend] antrean post penuh, event dibuang")


_post_queue = _PostQueue()


class _BackendClient:
    @staticmethod
    def post_occupancy_event(
        camera_id:      str,
        zone_camera_id: int,
        direction:      str,
        event_kind:     str = "crossing",
        snapshot_url:   str | None = None,
        person_label:   str | None = None,
        track_id:       int | None = None,
        point_x:        int | None = None,
        point_y:        int | None = None,
    ) -> None:
        def _do() -> None:
            try:
                requests.post(
                    f"{_backend_url()}/occupancy-events",
                    json={
                        "camera_id":      camera_id,
                        "zone_camera_id": zone_camera_id,
                        "direction":      direction,
                        "event_kind":     event_kind,
                        "snapshot_url":   snapshot_url,
                        "person_label":   person_label,
                        "track_id":       track_id,
                        "point_x":        point_x,
                        "point_y":        point_y,
                    },
                    headers=_auth_headers(),
                    timeout=0.8,
                )
            except Exception as exc:
                print(f"[backend] POST /occupancy-events error ({camera_id}): {exc}")
        _post_queue.submit(_do)

    @staticmethod
    def post_camera_event(
        camera_id:    str,
        event_type:   str,
        category:     str = "info",
        description:  str | None = None,
        snapshot_url: str | None = None,
        person_label: str | None = None,
    ) -> None:
        def _do() -> None:
            try:
                requests.post(
                    f"{_backend_url()}/camera-events",
                    json={
                        "camera_id":    camera_id,
                        "event_type":   event_type,
                        "category":     category,
                        "description":  description,
                        "snapshot_url": snapshot_url,
                        "person_label": person_label,
                    },
                    headers=_auth_headers(),
                    timeout=0.8,
                )
            except Exception as exc:
                print(f"[backend] POST /camera-events error ({camera_id}): {exc}")
        _post_queue.submit(_do)

    @staticmethod
    def post_detection(
        person_name:   str,
        camera_id:     str,
        confidence:    float,
        method:        str = "appearance",
        thumbnail_url: str | None = None,
        unique_label:  str | None = None,
        track_id:      int | None = None,
        timestamp:     "object" = None,
    ) -> None:
        """unique_label adalah label date-aware (e.g. 'Unknown #1@20260622')
        agar PostgreSQL tidak menggabungkan orang dari hari berbeda.

        `timestamp` HARUS waktu tracklet-nya beneran terjadi (mis. `ended_at`
        dari _resolve_tracklet), bukan waktu POST ini dieksekusi. Post ini
        jalan di worker _PostQueue di BELAKANG antrean — kalau timestamp-nya
        `datetime.now()` diambil di sini, itu jam saat worker akhirnya
        sempat proses jobnya, bisa telat beberapa detik dari kejadian
        aslinya. Klip footage dicari berdasarkan timestamp ini (lihat
        ai-service/app/routers/clips.py) — telat dikit saja bisa bikin
        klip yang ditemukan BUKAN klip yang benar-benar merekam momen itu."""
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
        def _do() -> None:
            try:
                requests.post(
                    f"{_backend_url()}/detections",
                    json={
                        "person_name":   person_name,
                        "person_label":  unique_label or person_name,
                        "camera_id":     camera_id,
                        "timestamp":     ts.astimezone(timezone.utc).isoformat(),
                        "confidence":    float(confidence),
                        "method":        method,
                        "thumbnail_url": thumbnail_url,
                        "track_id":      track_id,
                    },
                    headers=_auth_headers(),
                    timeout=0.8,
                )
            except Exception as exc:
                print(f"[backend] POST /detections error ({camera_id}): {exc}")
        _post_queue.submit(_do)

    @staticmethod
    def post_tracklet(
        camera_id:           str,
        track_id:            int,
        person_label:        "str | None",
        started_at:          "object",
        ended_at:             "object",
        n_detections:         int,
        best_thumbnail_url:   "str | None",
        embedding:            "object",
        assoc_score:          "float | None",
        par:                  "object" = None,
        best_crop:            "object" = None,
        pos_x:                "int | None" = None,
        pos_y:                "int | None" = None,
        positions:            "list | None" = None,
    ) -> None:
        """Dikirim sekali per tracklet ditutup, setelah post_detection (lihat
        pipeline_service.py._resolve_tracklet + batch_processor._finalize_tracklets).

        `par`/`best_crop` (bukan `attrs` yang sudah jadi) sengaja dilewatkan
        mentah — par_attrs() dipanggil DI DALAM _do() di bawah, yang jalan di
        worker thread _PostQueue (background), bukan thread inferensi utama.
        Ini yang membuat biaya ~2.3s/crop PAR tidak menahan pemrosesan kamera
        lain (lihat plan/04-tasks.md T3.2)."""
        def _do() -> None:
            attrs = None
            try:
                attrs = par_attrs(par, best_crop)
            except Exception as exc:
                print(f"[backend] PAR gagal dihitung ({camera_id}): {exc}")
            try:
                requests.post(
                    f"{_backend_url()}/tracklets",
                    json={
                        "camera_id":           camera_id,
                        "track_id":            track_id,
                        "person_label":        person_label,
                        "started_at":          started_at.astimezone(timezone.utc).isoformat(),
                        "ended_at":            ended_at.astimezone(timezone.utc).isoformat(),
                        "n_detections":        n_detections,
                        "best_thumbnail_url":  best_thumbnail_url,
                        "embedding":           embedding.tolist(),
                        "assoc_score":         assoc_score,
                        "attrs":               attrs,
                        "pos_x":               pos_x,
                        "pos_y":               pos_y,
                        "positions":           positions or [],
                    },
                    headers=_auth_headers(),
                    timeout=0.8,
                )
            except Exception as exc:
                print(f"[backend] POST /tracklets error ({camera_id}): {exc}")
        _post_queue.submit(_do)
