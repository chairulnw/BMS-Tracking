# Sistem Kamera Pengawas pada BMS

Video analytics untuk Building Management System — baca stream RTSP dari
CCTV, deteksi & lacak orang (Obj Detection + Tracking + ReID + atribut penampilan), simpan
klip/thumbnail, tampilkan di dashboard web.

| Layanan | Path | Tech |
|---|---|---|
| AI service (:8001) | `ai-service/` | Python, FastAPI, Ultralytics YOLO (ByteTrack), TransReID |
| Backend (:8002) | `backend/` | Python, FastAPI, asyncpg, PostgreSQL + pgvector |
| Frontend (:4200) | `sistem-kamera-pengawas/` | Angular 20 |

## Setup

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

Download dari [Google Drive](https://drive.google.com/drive/folders/1haDcq2ecycDS93rjlFQjj_UwuudRBAs8?usp=sharing), taruh isinya di `ai-service/checkpoints/`.

- Wajib: `vit_transreid_market1501.pth` + folder `TransReID/` lengkap
- Opsional: `par_checkpoints/RAP1.pth` untuk atribut penampilan

## Cara 1. Menjalankan dengan Docker

Butuh [Docker Desktop](https://docs.docker.com/desktop/). 

Jika tidak ada GPU NVIDIA, hapus blok `deploy:` pada service `ai-service`
di `docker-compose.yml`.

```bash
docker compose up --build
docker compose exec backend python scripts/create_admin.py   # buat akun login
```

Login ke `http://localhost:4200`, tambah minimal satu kamera aktif dengan RTSP di `/camera`.

Matiin: `docker compose down`. Nyalain lagi: `docker compose up` (tanpa `--build` kalau tidak ada perubahan kode).

## Cara 2. Setup manual (development lokal)

```bash
# Database
createdb bms_tracking
psql -d bms_tracking -f backend/db/schema.sql

# Backend — :8002
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/create_admin.py #buat akun
uvicorn app.main:app --port 8002 --reload

# AI service — :8001
cd ai-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8001 --reload

# Frontend — :4200
cd sistem-kamera-pengawas
npm install
ng serve
```

Matiin: `Ctrl+C` di tiap terminal. Nyalain lagi: jalankan ulang perintah `uvicorn`/`ng serve` masing-masing.

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
