# 01 — Roadmap

## Fase 1 — Retensi storage & alerting dasar

**Kenapa:** `ClipRecorder` nulis `.avi` ke `output/clips/` dan snapshot ke
`thumbnails/events/` tanpa batas umur atau kuota. `camera_offline`/`camera_online`
sudah tercatat di `camera_events` tapi cuma bisa dilihat kalau buka dashboard —
tidak ada yang mendorong notifikasi keluar.

- [x] Cron/job hapus clip & thumbnail lebih tua dari N hari (config, default
      misal 14 hari) — **selesai 2026-08-21**: `ai-service/app/services/retention.py`
      (`cleanup_once`), background thread dari `main.py` lifespan, interval
      `CLEANUP_INTERVAL_HOURS` (default 6j), retensi `RETENTION_DAYS` (default 14)
- [x] Guard kapasitas: kalau disk di atas threshold (mis. 90%), hapus clip
      terlama duluan, log peringatan — **selesai 2026-08-21**, sama file,
      `DISK_GUARD_PCT` (default 90)
- [ ] Webhook/email sederhana saat `camera_offline` tercatat (endpoint sudah
      ada di `camera_events`, tinggal listener)
- [ ] Health check gabungan: `GET /health` AI-service + `GET /stats/today`
      backend di-poll external (uptime monitor / cron ping) — bukan cuma UI.
      *Catatan: endpoint-nya sudah ada, ini murni setup operasional (uptime
      monitor eksternal), bukan kode baru.*

**Definition of Done**
- [x] Simulasi disk penuh → clip lama otomatis kehapus, service tidak crash —
      diverifikasi via self-check `python app/services/retention.py`
- [ ] Matikan 1 kamera dummy → notifikasi keluar dalam <1 menit

---

## Fase 2 — Keandalan pipeline AI

**Kenapa:** `IdentityDB` cuma di RAM (`app/services/pipeline_service.py`) —
restart AI-service di jam sibuk = semua orang yang lagi ditrack dapat identity
baru, dan `ClipRecorder` yang lagi RECORDING kepotong tanpa file ke-close
dengan benar. `BatchProcessor` juga single-thread untuk semua kamera — makin
banyak kamera makin numpuk di 1 inference queue.

- [x] Graceful shutdown: `ClipRecorder` di state RECORDING di-flush/close saat
      `SIGTERM` — **sudah ada**: `main.py` lifespan shutdown → `StreamManager.stop()`
      → `BatchProcessor._loop` `finally` → `slot.shutdown()` →
      `ClipRecorder.force_stop()` (`clip_recorder.py:154`) finalize file dengan benar.
      Cuma berlaku untuk shutdown biasa (uvicorn SIGTERM/SIGINT), bukan `kill -9`.
- [x] Restart-safe stream: `POST /stream/start` idempotent — **sudah ada**:
      `StreamManager.start()` cek `self._running` dua kali (double-checked
      locking, `stream_manager.py:57` & `:65`), raise `RuntimeError` kalau
      sudah jalan — tidak pernah buka koneksi dobel
- [ ] Metric dasar untuk `BatchProcessor`: waktu per batch, biar kelihatan
      kapan mulai jadi bottleneck sebelum kamera nambah
- [ ] Timezone fix: `GET /occupancy` (UTC) vs `GET /persons` (Asia/Jakarta) —
      satukan, ini bug nyata bukan cuma inkonsistensi kosmetik. *Dicek ulang
      2026-08-21: masih ada — `zones.py` bahkan campur sendiri, default filter
      pakai Jakarta (baris 201) tapi agregasi occupancy/history pakai UTC
      (baris 239-318), sementara `persons.py` konsisten Jakarta.*
- [ ] **Kamera offline permanen tetap ikut batch YOLO.** Setelah 5x reconnect
      gagal (`cam_slot.py` `RECONNECT_TRIES`), `_on_reconnect(success=False)`
      cuma log + post event lalu berhenti — slot tidak dihapus dari
      `self._slots`, jadi tiap siklus `batch_processor.py` tetap kirim frame
      hitam/placeholder kamera itu ke `detector.predict()` bareng kamera yang
      masih hidup (lihat `_loop`, baris ~185-247). Buang GPU/CPU cycle
      selamanya sampai stream direstart manual. Perbaikan: keluarkan slot
      offline-permanen dari batch (skip predict untuk slot itu), plus retry
      berkala (mis. tiap 5 menit) bukan cuma 5x lalu nyerah total

**Definition of Done**
- [ ] Restart AI-service pertengahan recording → clip sebelumnya tetap valid
      (playable), tidak corrupt
- [ ] `/occupancy` dan `/persons` pakai timezone yang sama untuk tanggal yang sama

---

## Fase 3 — Keamanan & audit

**Kenapa:** semua user login = akses penuh (satu tabel `users`, tanpa role).
Tidak ada catatan siapa export/lihat footage — jadi masalah kalau video
dipakai sebagai bukti atau kalau ada lebih dari satu operator.

- [ ] Kolom `role` di `users` (mis. `admin`/`operator`/`viewer`), enforce di
      router yang sensitif (camera CRUD, user management, export)
- [ ] Audit log tabel baru: siapa akses thumbnail/clip/playback kapan —
      minimal untuk endpoint yang serve file video/snapshot
- [ ] Review token lifetime & refresh — cek JWT expiry saat ini di
      `app/auth.py` kedua service, pastikan tidak infinite

**Definition of Done**
- [ ] User role `viewer` tidak bisa hit endpoint CRUD kamera
- [ ] Query "siapa buka clip X" bisa dijawab dari DB

---

## Fase 4 — Kapasitas & integrasi kamera

**Kenapa:** relevan begitu jumlah kamera naik dari skala testing sekarang.
Belum blocker hari ini.

- [ ] Dokumentasi kapasitas: berapa kamera max per instance `BatchProcessor`
      sebelum FPS drop (ukur, jangan tebak)
- [ ] ONVIF discovery/PTZ control kalau ada kamera yang mendukung — saat ini
      cuma RTSP statis dari `cameras` table
- [ ] Storage capacity planning: estimasi GB/hari per kamera aktif, expose di
      `/pengaturan` biar admin tahu kapan disk habis

**Definition of Done**
- [ ] Ada angka konkret "N kamera = X% CPU/GPU" untuk sizing hardware
      selanjutnya
