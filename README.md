# Sistem Kamera Pengawas pada BMS

Sistem **Kamera Pengawas** dari Building Management System — video
analytics yang membaca stream RTSP dari CCTV, mendeteksi dan melacak orang
(YOLO + ReID + atribut penampilan), menyimpan klip/thumbnail, dan
menampilkannya di dashboard web.

## Arsitektur

```
Kamera RTSP → ai-service (:8001)  deteksi YOLO + ReID + PAR, rekam klip
                   │ POST /detections, /camera-events, /occupancy-events
                   ▼
              backend (:8002)     PostgreSQL + pgvector, CRUD, auth
                   ▲ HTTP (JWT)
              frontend (:4200)    Angular 20 dashboard
```

Backend dan AI service berbagi satu `SECRET_KEY` (JWT HS256)

| Layanan | Path | Tech |
|---|---|---|
| AI service | `ai-service/` | Python, FastAPI, Ultralytics YOLO (ByteTrack), TransReID |
| Backend | `backend/` | Python, FastAPI, asyncpg, PostgreSQL + pgvector |
| Frontend | `sistem-kamera-pengawas/` | Angular 20 |

## Menjalankan dengan Docker (cara utama)

Butuh [Docker Desktop](https://docs.docker.com/desktop/).
GPU NVIDIA opsional — kalau tidak ada, hapus blok `deploy:` pada service `ai-service`
di `docker-compose.yml`.

### 1. Clone repo

```bash
git clone https://github.com/chairulnw/BMS-Tracking.git
cd BMS-Tracking
```

### 2. Setup `.env`

```bash
cp backend/.env.example backend/.env
cp ai-service/.env.example ai-service/.env
```
isi `SECRET_KEY` sama persis di kedua file

### 3. Download checkpoint model

Download checkpoint ReID/PAR dari Google Drive: https://drive.google.com/drive/folders/1haDcq2ecycDS93rjlFQjj_UwuudRBAs8?usp=sharing, lalu taruh isinya di
`ai-service/checkpoints/`.

- Wajib: `vit_transreid_market1501.pth` + folder `TransReID/` lengkap
- opsional: `par_checkpoints/RAP1.pth` untuk atribut penampilan

### 4. Build & jalankan

```bash
docker compose up --build
```

### 5. Buat akun untuk login

```bash
docker compose exec backend python scripts/create_admin.py
```

### 6. Tambah kamera

Login ke `http://localhost:4200`, tambah minimal satu kamera aktif dengan RTSP di `/camera`

## Matiin & nyalain lagi

Docker:
```bash
docker compose down          # matiin
docker compose up            # nyalain lagi (tanpa --build kalau tidak ada perubahan kode)
```

Manual: `Ctrl+C` di tiap terminal buat matiin, jalankan lagi perintah `uvicorn`/`ng serve` masing-masing (lihat "Setup manual" di bawah) buat nyalain.

## Setup manual (development lokal)

```bash
# Database
createdb bms_tracking
psql -d bms_tracking -f backend/db/schema.sql

# Backend — :8002
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # isi DATABASE_URL & SECRET_KEY
python scripts/create_admin.py
uvicorn app.main:app --port 8002 --reload

# AI service — :8001
cd ai-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # SECRET_KEY harus sama dengan backend/.env
uvicorn app.main:app --port 8001 --reload

# Frontend — :4200
cd sistem-kamera-pengawas
npm install
ng serve
```

- Dashboard: http://localhost:4200
- Swagger backend: http://localhost:8002/docs
- Swagger AI service: http://localhost:8001/docs

## Struktur repo

```
ai-service/
  app/routers/            endpoint HTTP (/stream, /process-video, ...)
  app/services/stream_service/   pipeline live-stream (capture, inferensi, tracker, klip)
  app/services/pipeline_service.py   IdentityDB (ReID) + logika file-playback
  app/par/                model vendored PromptPAR (atribut penampilan)
  checkpoints/            bobot model (tidak masuk git)

backend/
  app/routers/            satu file per resource (cameras, persons, zones, ...)
  app/auth.py             JWT mint/verify
  db/schema.sql           skema database

sistem-kamera-pengawas/
  src/app/pages/          satu folder per halaman
  src/app/layout/         sidebar & topbar
  src/app/services/       HTTP client + auth interceptor
  src/app/guards/         authGuard
```
