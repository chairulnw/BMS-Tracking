# BMS-IIP Kamera Pengawas

Submodul **Kamera Pengawas** dari Building Management System. Platform video
analytics yang membaca stream RTSP dari kamera CCTV, mendeteksi dan melacak
orang (YOLO + ReID + atribut penampilan), menyimpan klip/thumbnail, lalu
menampilkannya di dashboard web (overview, live view, roster orang, playback,
zona/occupancy).

## Arsitektur

Tiga service independen:

```
Kamera RTSP
    │
    ▼
ai-service (FastAPI, :8001)      deteksi YOLO + ReID + PAR, rekam klip,
    │  POST /detections,             lapor event ke backend
    │       /camera-events,
    │       /occupancy-events
    ▼
backend (FastAPI, :8002)         PostgreSQL (asyncpg): semua CRUD, stats, auth
    ▲
    │ HTTP (JWT)
sistem-kamera-pengawas (Angular 20, :4200)   dashboard web
```

Backend dan AI service berbagi satu `SECRET_KEY` untuk mint/verify JWT HS256 —
AI service tidak punya login sendiri, ia memakai *service token* untuk
memanggil endpoint backend yang terproteksi.

| Layanan | Path | Tech |
|---|---|---|
| AI service | `ai-service/` | Python 3.14, FastAPI, Ultralytics YOLO, torchreid |
| Backend | `backend/` | Python 3.14, FastAPI, asyncpg, PostgreSQL + pgvector |
| Frontend | `sistem-kamera-pengawas/` | Angular 20 (standalone components) |

## Prasyarat

- Python 3.13+ (masing-masing service pakai venv sendiri)
- Node.js + Angular CLI (`npm i -g @angular/cli`) untuk frontend
- PostgreSQL dengan ekstensi `pgvector`
- `ffmpeg` (dipakai AI service untuk encode klip)

## Setup

### 1. Database

```bash
createdb bms_tracking
psql -d bms_tracking -f backend/db/schema.sql   # sudah termasuk CREATE EXTENSION vector
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # isi DATABASE_URL & SECRET_KEY
python scripts/create_admin.py   # buat akun login pertama
```

### 3. AI service

```bash
cd ai-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # SECRET_KEY harus SAMA dengan backend/.env
```

Model yang wajib ada sebelum start (di-load saat startup):
- `ai-service/checkpoints/<DETECTOR_MODEL>` — checkpoint YOLO/RT-DETR (default `yolo26n.pt`)
- ReID checkpoint sesuai `REID_MODEL` (default `transreid`)
- `ai-service/checkpoints/par_checkpoints/RAP1.pth` — opsional; tanpa ini,
  service tetap jalan tapi tanpa Person Attribute Recognition

Minimal satu baris di tabel `cameras` (lihat `/pengaturan` di frontend atau
insert manual) — AI service menolak start kalau tidak ada kamera aktif.

### 4. Frontend

```bash
cd sistem-kamera-pengawas
npm install
```

## Menjalankan

Sekaligus (macOS, buka 3 tab Terminal):
```bash
./start.sh
```

Atau manual, masing-masing di direktori & venv-nya sendiri:
```bash
# Backend — :8002
cd backend && source .venv/bin/activate && uvicorn app.main:app --port 8002 --reload

# AI service — :8001
cd ai-service && source .venv/bin/activate && uvicorn app.main:app --port 8001 --reload

# Frontend — :4200
cd sistem-kamera-pengawas && ng serve
```

- Dashboard: http://localhost:4200
- Swagger backend: http://localhost:8002/docs
- Swagger AI service: http://localhost:8001/docs

## Struktur repo

```
ai-service/     FastAPI — inferensi YOLO/ReID/PAR per kamera, rekam klip,
                lapor event ke backend
  app/
    routers/            endpoint HTTP (/stream, /process-video, ...)
    services/
      stream_service/   pipeline live-stream (capture, batch inferensi, tracker, klip)
      pipeline_service.py   IdentityDB (ReID) + logika file-playback
      geometry.py       util garis/polygon buat deteksi crossing
    par/                model vendored PromptPAR (person attribute recognition)
  benchmark/            skrip & laporan perbandingan detector/tracker/Re-ID
  checkpoints/          bobot model (tidak masuk git — lihat Setup)

backend/        FastAPI — sumber kebenaran data (PostgreSQL)
  app/
    routers/            satu file per resource (cameras, persons, zones, ...)
    auth.py             JWT mint/verify
    retention.py        housekeeping baris lama
  db/schema.sql          skema database

sistem-kamera-pengawas/   Angular 20 SPA
  src/app/
    pages/              satu folder per halaman (overview, liveview, people, ...)
    layout/             sidebar & topbar
    services/           HTTP client + auth interceptor
    guards/             authGuard (proteksi semua route kecuali /login)
```

## Tests

Tidak ada suite pytest/Karma — self-check dijalankan langsung:
```bash
cd backend && python test_retention.py           # butuh DATABASE_URL nyata
cd backend && python test_clip_camera_match.py
```

Frontend:
```bash
cd sistem-kamera-pengawas
ng test          # unit test via Karma
npx tsc --noEmit # type-check
```
