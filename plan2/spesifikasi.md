# Spesifikasi: VMS Hardening

Satu dokumen rujukan untuk sistem sebagai **Video Management System**, di
luar scope `plan/` (fokus ke People Workspace). `[x]` = sudah ada di kode,
dengan bukti singkat. `[ ]` = belum, dengan effort **Kecil**/**Sedang**/**Besar**.

---

## Area yang sudah solid — tidak perlu fase

Tidak ada kerjaan terbuka di area ini kecuali muncul bug baru.

- [x] **Camera Management** — CRUD kamera (nama, RTSP, floor, grup): `backend/app/routers/cameras.py`
- [x] **Playback / Evidence** — cari klip berdasar kamera+timestamp: `clips.py` `_find_clip`
- [x] **Event Management** — `camera_events` (type, category, timestamp, snapshot) + filter `GET /camera-events`
- [x] **Search** — filter waktu/kamera/atribut + appearance search KNN (`similar_to`): `backend/app/routers/people.py`
- [x] **Person Identity / Re-ID** — tracklet → association score → Person ID, inti sistem (`plan/` Fase 2)
- [x] **Person Attributes** — PAR aktif, `par_checkpoints/RAP1.pth` di-load saat startup (`batch_processor.py:78-88`). *`CLAUDE.md` masih bilang "disabled" — itu basi, belum diupdate sejak `plan/` Fase 3 selesai.* Catatan kualitas: warna baju tidak stabil antar kemunculan (`plan/06-decisions.md`), jangan dipakai sebagai sinyal akurasi ReID.
- [x] **Zone Management** — zone line + polygon per kamera: tabel `zones` + `zone_cameras`
- [x] **Multi-camera Zone** — satu zone bisa punya banyak baris `zone_cameras`, arsitektur "logical zone → beberapa kamera" sudah ada dari awal
- [x] **Kontrol per-kamera** (bukan global) — `analytics_enabled` per kamera
      sudah full-stack: `cameras.py` (backend) + toggle & filter di halaman
      `/camera` (frontend). Ini yang dipakai buat maintenance satu kamera,
      bukan tombol start/stop global yang dulu ada di `/pengaturan` — global
      switch kurang lazim di VMS lain (recording service biasanya auto-start
      terus-menerus, kontrolnya per-kamera), jadi sengaja tidak dikembalikan.

---

## Fase 1 — Retensi storage & keandalan kamera

- [x] Cron/job hapus clip & thumbnail lebih tua dari `RETENTION_DAYS` (default
      14 hari) + disk guard `DISK_GUARD_PCT` (default 90%) — **selesai
      2026-08-21**: `ai-service/app/services/retention.py`, background thread
      dari `main.py` lifespan *(Storage Management)*
- [x] Reconnect kamera tanpa batas — **selesai 2026-08-21**:
      `cam_slot.py` `start_reconnect()`, 5x cepat lapor `camera_offline`
      lalu retry tiap 30s selama service hidup, tidak pernah give-up permanen
      *(Camera Health)*
- [x] Kamera offline tidak ikut batch inferensi YOLO — **selesai
      2026-08-21**: `batch_processor.py` (`active_idx`/`active_results`),
      dulu buang GPU/CPU cycle terus-menerus untuk kamera mati *(Camera Health)*
- [x] Endpoint status ringkas per kamera — **selesai 2026-08-21**:
      `GET /cameras` sekarang balikin `health_status` (`online`/`offline`/
      `null`) + `last_seen`, derived dari event `camera_online`/`camera_offline`
      terakhir (`cameras.py` `_CAMERA_SELECT`, LATERAL JOIN ke `camera_events`
      — gak nambah tabel baru) *(Camera Health)*
- [x] Telegram Bot webhook saat `category == "critical"` — **selesai
      2026-08-21**: `backend/app/notify.py` (`notify_telegram`), dipanggil
      dari `POST /camera-events` saat category critical. Isi
      `TELEGRAM_BOT_TOKEN`+`TELEGRAM_CHAT_ID` di `.env` buat aktifkan (no-op
      kalau kosong) *(Notification)*
- [x] **Bug ditemukan & diperbaiki**: disk guard sebelumnya bandingin ke
      persen disk **seluruh sistem**, bukan folder project — di mesin dengan
      disk 90%+ penuh gara-gara hal lain (OS, app lain), guard itu gak akan
      pernah berhasil turunin ke threshold cuma dari folder ini, jadi
      **terus-menerus hapus file yang baru dibuat**. Diganti jadi cap ukuran
      folder sendiri: `MAX_STORAGE_GB` (default 5GB) — `retention.py`

**Definition of Done**
- [x] Simulasi disk penuh → clip lama otomatis kehapus, service tidak crash
      (diverifikasi via self-check `python app/services/retention.py`)
- [x] Matikan kamera dummy lalu nyalakan lagi → reconnect otomatis tanpa
      restart manual
- [x] `GET /cameras` balikin status online/offline/last_seen
- [ ] Matikan 1 kamera dummy → pesan Telegram masuk dalam <1 menit (perlu
      `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` diisi buat diverifikasi manual)

---

## Fase 2 — Korektnes pipeline & observability dasar

- [x] Graceful shutdown `ClipRecorder` saat SIGTERM — **sudah ada**: lifespan
      shutdown → `StreamManager.stop()` → `slot.shutdown()` →
      `ClipRecorder.force_stop()` (`clip_recorder.py:154`) finalize file
      dengan benar. Cuma berlaku shutdown biasa (SIGTERM/SIGINT), bukan `kill -9`.
- [x] `POST /stream/start` idempotent — **sudah ada**: `StreamManager.start()`
      double-checked locking (`stream_manager.py:57` & `:65`), raise
      `RuntimeError` kalau sudah jalan, tidak pernah buka koneksi dobel
- [x] Satukan timezone — **selesai 2026-08-21**: semua `AT TIME ZONE 'UTC'`
      di `zones.py` (occupancy, history, heatmap, events) diganti
      `'Asia/Jakarta'`, konsisten dengan `persons.py` *(Occupancy)*
- [x] Filter zone di Search — **selesai 2026-08-21**: ternyata belum ada,
      ditambahkan `zone_id` query param di `GET /people/feed` (EXISTS
      subquery ke `zone_cameras`) *(Search)*
- [x] Metric CPU/RAM/waktu-per-batch — **selesai 2026-08-21**: `GET /health`
      sekarang balikin `cpu_percent`/`ram_percent` (psutil) +
      `last_batch_ms`/`avg_batch_ms`/`cameras_active` (dari
      `BatchProcessor.get_metrics()`, rolling window 50 batch) *(System Health)*

**Definition of Done**
- [ ] Restart AI-service pertengahan recording → clip sebelumnya tetap valid, tidak corrupt
- [x] `/occupancy` dan `/persons` pakai timezone yang sama untuk tanggal yang sama
- [x] Ada angka konkret CPU/RAM/batch-time yang bisa dipantau, bukan cuma boolean `/health`

---

## Fase 3 — Alarm & akses

- [x] Status ack/unack per event — **selesai 2026-08-21**: kolom
      `acknowledged`/`acknowledged_at` di `camera_events`, endpoint
      `PATCH /camera-events/{id}/ack`, tombol "Tandai selesai" di tabel
      Kejadian Terakhir (`/overview`, bukan halaman baru) *(Alarm Management)*
- [x] Review JWT expiry — **dicek 2026-08-21, sudah aman**: backend
      `ACCESS_TOKEN_EXPIRE_MINUTES` (default 720 = 12 jam, via env), ai-service
      service token 24 jam hardcode (`ai-service/app/auth.py:14`). Tidak ada
      yang infinite, tidak perlu perubahan *(Akses/Login)*

**Definition of Done**
- [x] Event bisa ditandai "sudah ditindaklanjuti" dan itu kelihatan di UI

---

## Fase 4 — Kapasitas

- [ ] **Sedang** — dashboard kapasitas admin-facing (estimasi GB/hari,
      proyeksi kapan disk habis); belum ada tempat lihat ini dari UI
      *(Storage Management)*
- [x] Retention policy untuk tabel DB — **selesai 2026-08-21**:
      `backend/app/retention.py`, hapus `detections`/`tracklets` lebih tua
      dari `DB_RETENTION_DAYS` (default 90 hari), background task dari
      `main.py` lifespan, jalan tiap `DB_CLEANUP_INTERVAL_HOURS` (default 24j)
      *(Storage Management)*

**Definition of Done**
- [ ] Ada angka konkret "N kamera = X% CPU/GPU" untuk sizing hardware selanjutnya
- [ ] Admin bisa lihat proyeksi kapasitas disk dari UI

---
