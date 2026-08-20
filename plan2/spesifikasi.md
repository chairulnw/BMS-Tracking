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
- [ ] **Kecil** — endpoint status ringkas per kamera (online/offline/last_seen);
      sekarang cuma bisa ditarik manual dari histori `camera_events`
      *(Camera Health)*
- [ ] **Kecil** — Telegram Bot webhook saat `category == "critical"` tercatat
      (sekarang cuma `camera_offline`) — gratis, tanpa verifikasi bisnis,
      endpoint `camera_events` sudah ada, tinggal listener + `sendMessage`
      *(Notification)*

**Definition of Done**
- [x] Simulasi disk penuh → clip lama otomatis kehapus, service tidak crash
      (diverifikasi via self-check `python app/services/retention.py`)
- [x] Matikan kamera dummy lalu nyalakan lagi → reconnect otomatis tanpa
      restart manual
- [ ] `GET /cameras` (atau endpoint baru) balikin status online/offline/last_seen
- [ ] Matikan 1 kamera dummy → pesan Telegram masuk dalam <1 menit

---

## Fase 2 — Korektnes pipeline & observability dasar

- [x] Graceful shutdown `ClipRecorder` saat SIGTERM — **sudah ada**: lifespan
      shutdown → `StreamManager.stop()` → `slot.shutdown()` →
      `ClipRecorder.force_stop()` (`clip_recorder.py:154`) finalize file
      dengan benar. Cuma berlaku shutdown biasa (SIGTERM/SIGINT), bukan `kill -9`.
- [x] `POST /stream/start` idempotent — **sudah ada**: `StreamManager.start()`
      double-checked locking (`stream_manager.py:57` & `:65`), raise
      `RuntimeError` kalau sudah jalan, tidak pernah buka koneksi dobel
- [ ] **Kecil** — satukan timezone: `zones.py` default filter pakai
      Asia/Jakarta (baris 201) tapi agregasi occupancy/history pakai UTC
      (baris 239-318), sementara `persons.py` konsisten Jakarta *(Occupancy)*
- [ ] **Kecil** — verifikasi filter zone ada di endpoint search people, belum
      dicek langsung *(Search)*
- [ ] **Sedang** — metric CPU/RAM/waktu-per-batch; `GET /health` sekarang cuma
      boolean model-loaded, tidak ada metric apapun di `BatchProcessor`
      *(System Health)*

**Definition of Done**
- [ ] Restart AI-service pertengahan recording → clip sebelumnya tetap valid, tidak corrupt
- [ ] `/occupancy` dan `/persons` pakai timezone yang sama untuk tanggal yang sama
- [ ] Ada angka konkret CPU/RAM/batch-time yang bisa dipantau, bukan cuma boolean `/health`

---

## Fase 3 — Alarm & akses

- [ ] **Sedang** — status ack/unack per event, dashboard prioritas — sekarang
      severity cuma `camera_events.category` (info/critical) tanpa status
      ditindaklanjuti-atau-belum *(Alarm Management)*
- [ ] **Kecil** — review JWT expiry di `app/auth.py` kedua service, pastikan
      tidak infinite *(Akses/Login)*

**Definition of Done**
- [ ] Event bisa ditandai "sudah ditindaklanjuti" dan itu kelihatan di UI

---

## Fase 4 — Kapasitas & integrasi

- [ ] **Sedang** — dashboard kapasitas admin-facing (estimasi GB/hari,
      proyeksi kapan disk habis); belum ada tempat lihat ini dari UI
      *(Storage Management)*
- [ ] **Sedang** — retention policy untuk tabel DB (`detections`/`tracklets`),
      sekarang cuma file di disk yang di-cover Fase 1 *(Storage Management)*
- [ ] **Sedang** — dokumentasi kapasitas: berapa kamera max per instance
      `BatchProcessor` sebelum FPS drop (ukur, jangan tebak)
- [ ] **Sedang** — ONVIF discovery/PTZ kalau ada kamera yang mendukung —
      sekarang cuma RTSP statis dari tabel `cameras`

**Definition of Done**
- [ ] Ada angka konkret "N kamera = X% CPU/GPU" untuk sizing hardware selanjutnya
- [ ] Admin bisa lihat proyeksi kapasitas disk dari UI

---

## Fase 5 — Butuh keputusan produk dulu (Besar)

*Bukan soal kurang kode, tapi scope yang belum diputuskan — jangan mulai
coding sebelum ini jelas.*

- [ ] **System Configuration** — tidak ada tabel settings maupun UI untuk
      threshold. `conf_threshold`/`reid_threshold` cuma parameter
      `POST /stream/start`, default hardcode di `schemas.py`. Ganti threshold
      = ubah kode + restart. Fitur standar di VMS lain (Milestone Management
      Client, Genetec Config Tool, Hikvision System Settings), jadi layak
      dibangun — **tapi tidak perlu halaman baru**: cukup section kecil
      "Pengaturan Sistem" di halaman `/camera` yang sudah ada (reuse, bukan
      route terpisah). Tombol start/stop stream global **sengaja tidak**
      dikembalikan — lihat catatan "Kontrol per-kamera" di atas.

**Definition of Done**
- [ ] Admin bisa ubah threshold dari UI tanpa restart service

---

## Diputuskan tidak dikerjakan

- **Role-based access (`Akses/Login`)** — sistem cukup satu tingkat akses;
  login saja sudah menutup kebutuhan, menambah role berarti membangun user
  management ganda tanpa kebutuhan operasional yang jelas.
- **Audit Log** — butuh halaman frontend baru untuk ditampilkan; effort tidak
  sepadan untuk skala operasional saat ini.
- **WebSocket live view (`API/Integration`)** — live view di luar tanggung
  jawab/scope kerja saat ini.
