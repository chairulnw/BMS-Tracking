# Laporan Latency Per-Layer — Person Detection Pipeline

Digenerate `tools/latency_report.py`. Sumber: tabel `camera_events`, `tracklets` (DB), `latency.csv` (per-frame, seluruh sesi), dan `end_to_end_ms` dari console log browser (satu-satunya sumbernya — jam browser terima response gak disimpan di mana pun selain DevTools Console pas testing).

## Ringkasan eksekutif — alur lengkap kamera sampai dashboard

**1. Capture** (`cam_slot.py`) — `cap.read()` per kamera, non-blocking (frame didrop kalau inferensi belum selesai). Dicatet `decode_ms`.

**2. Deteksi** (`batch_processor.py`) — YOLO dipanggil 1x per siklus buat SEMUA kamera aktif sekaligus, SELALU jalan tiap siklus terlepas ada orang apa nggak (dia yang nentuin). Dicatet `detection_ms`.

**3. Tracking** (per kamera) — BYTETracker/BoTSORT, CUMA jalan kalau `n_raw>0` (ada ≥1 orang di frame ini). Dicatet `tracking_ms`.

**4. Ekstraksi embedding ReID** (per track) — `_extract_embedding()` dipanggil per track HASIL tracking, tapi ada quality-gate duluan (ukuran/rasio/blur box) SEBELUM model beneran dipanggil — kalau gagal, balik cepat (`<1ms`) tanpa ekstraksi beneran. Dicatet `reid_ms`.

**5. Matching ke gallery** (`associate()`) — SELALU 'dipanggil' tiap siklus (bookkeeping `observe()`/`update_active()` gak ada syarat), tapi cosine-similarity beneran ke gallery CUMA jalan pas tracklet BENERAN ditutup (`close_expired()`). Dicatet `matching_ms`.

→ Langkah 1-5 = `total_ai_ms`, terjadi REAL-TIME selama orang kelihatan di kamera.

**6. Tracklet digantung 15 siklus** tanpa deteksi baru (`TRACKLET_GAP_CYCLES`) — mastiin orangnya bener pergi, bukan cuma keok sesaat.

**7. Resolve + POST ke backend** — rata-rata embedding, `associate()` final, antre `_PostQueue`, network, insert DB.

→ Langkah 6-7 = `backend_delay_ms` (`created_at - timestamp` di DB) — didominasi langkah 6, BUKAN komputasi.

**8. Browser nunggu polling berikutnya** (`interval(30000)`, tiap 30 detik) — ini yang paling dominan dari total end-to-end.

→ Total 6-8 = `end_to_end_ms` (`Date.now()` browser − `timestamp` AI).

**Kesimpulan posisi bottleneck**: AI (langkah 1-5) cepat & stabil (~250ms/frame). Backend (6-7) ~2,75 detik, didominasi nunggu 15-siklus. Paling lambat: langkah 8 (desain polling frontend, ~16 detik rata-rata) — bukan masalah pipeline AI/backend.

## 0. Statistik per-tahap AI — SELURUH sesi (bukan cuma 29 event)

Dihitung dari semua 70,604 baris `latency.csv` (semua frame yang diproses, semua kamera, seluruh sesi test). `decode_ms`/`detection_ms` SELALU jalan tiap siklus (YOLO nentuin ada-orang-apa-nggak duluan), jadi rata-rata "semua frame" sudah representatif — TIDAK dipisah per event_type di Bagian 6 (lihat catatan di situ). `tracking_ms`/`reid_ms` di-skip total kalau frame kosong (`n_raw==0`, `batch_processor.py:377`). `reid_ms` nonzero masih kepecah 2: ditolak quality-gate (`<1ms`, model GAK dipanggil) vs ekstraksi beneran (`≥1ms`). `matching_ms` nonzero juga kepecah 2: bookkeeping murah (`≤0,05ms`) vs `associate()` beneran (tracklet ditutup, `>0,05ms`). Kalau rata-rata "semua frame" dipake buat 3 kolom terakhir ini, KEDILUSI sama frame kosong/gak-kerja — makanya dipisah.

| Tahap | Semua frame | Breakdown |
|---|---|---|
| `decode_ms` | n=70,604 avg=42.226ms min=0.594ms max=2,029.63ms | selalu jalan, gak perlu dipisah |
| `detection_ms` | n=70,604 avg=39.721ms min=7.183ms max=1,625.12ms | selalu jalan, gak perlu dipisah |
| `tracking_ms` | n=70,604 avg=0.026ms min=0.000ms max=34.85ms | ada-orang: n=2,622 avg=0.701ms min=0.148ms max=34.85ms |
| `reid_ms` | n=70,604 avg=1.652ms min=0.000ms max=1,284.17ms | ekstraksi beneran (≥1.0ms): n=2,397 avg=48.657ms min=26.201ms max=1,284.17ms <br> ditolak quality-gate (<1.0ms): n=178 avg=0.005ms min=0.002ms max=0.06ms |
| `matching_ms` | n=70,604 avg=0.006ms min=0.000ms max=19.34ms | tracklet ditutup (>0.05ms): n=701 avg=0.279ms min=0.051ms max=19.34ms <br> bookkeeping doang (≤0.05ms): n=69,761 avg=0.003ms min=0.001ms max=0.05ms |
| `total_ai_ms` | n=70,604 avg=83.631ms min=8.556ms max=2,068.61ms | jumlah 5 kolom di atas per frame |

### Funnel — dari decode turun ke tiap tahap (n frame, % dari decode)

| Tahap | n frame | % dari decode |
|---|---:|---:|
| decode (baseline) | 70,604 | 100% |
| detection (selalu jalan) | 70,604 | 100% |
| tracking (ada ≥1 orang) | 2,622 | 3.7% |
| reid — fungsi dipanggil | 2,575 | 3.6% |
| ↳ reid — ekstraksi BENERAN (lolos quality-gate) | 2,397 | 3.4% |
| ↳ reid — ditolak quality-gate | 178 | 0.3% |
| matching — fungsi dipanggil | 70,604 | 100% |
| ↳ matching — beneran nutup tracklet | 701 | 1.0% |
| ↳ matching — bookkeeping kosong | 69,761 | 98.8% |

Bacanya: dari semua frame yang di-decode, cuma sebagian kecil yang beneran ada substansi buat diproses lebih jauh — sisanya sistem lagi "mantau" (scan kosong), bukan "sibuk kerja".

## 1. Tabel `camera_events`

| id  | camera  | type            | timestamp                        | created_at                       | ai_latency_ms (snapshot 1 frame) |
| --- | ------- | --------------- | -------------------------------- | -------------------------------- | -------------------------------: |
| 1   | c10     | person_detected | 2026-09-01 15:49:24.982107+07:00 | 2026-09-01 15:49:26.586259+07:00 |                            44.30 |
| 2   | c12     | person_detected | 2026-09-01 15:49:30.392655+07:00 | 2026-09-01 15:49:31.937218+07:00 |                            45.91 |
| 3   | c11     | person_detected | 2026-09-01 15:49:40.690029+07:00 | 2026-09-01 15:49:42.842322+07:00 |                            61.00 |
| 4   | c11     | zone_entry      | 2026-09-01 15:50:09.129096+07:00 | 2026-09-01 15:50:09.249589+07:00 |                            64.76 |
| 5   | c11     | person_detected | 2026-09-01 15:50:17.726512+07:00 | 2026-09-01 15:50:19.594392+07:00 |                            51.00 |
| 6   | c12     | person_detected | 2026-09-01 15:50:25.857215+07:00 | 2026-09-01 15:50:27.324548+07:00 |                            52.19 |
| 7   | c10     | person_detected | 2026-09-01 15:50:35.652045+07:00 | 2026-09-01 15:50:37.599167+07:00 |                            71.66 |
| 8   | c8      | person_detected | 2026-09-01 15:50:35.895691+07:00 | 2026-09-01 15:50:38.496473+07:00 |                           124.36 |
| 9   | c10     | person_detected | 2026-09-01 15:53:28.641034+07:00 | 2026-09-01 15:53:30.853814+07:00 |                           103.28 |
| 10  | c10     | person_detected | 2026-09-01 15:53:46.371457+07:00 | 2026-09-01 15:53:48.868538+07:00 |                            64.57 |
| 11  | c12     | person_detected | 2026-09-01 15:53:56.414890+07:00 | 2026-09-01 15:54:01.367842+07:00 |                           150.74 |
| 12  | c12     | person_detected | 2026-09-01 15:54:01.086542+07:00 | 2026-09-01 15:54:07.554272+07:00 |                           193.94 |
| 13  | c12     | person_detected | 2026-09-01 15:54:04.767157+07:00 | 2026-09-01 15:54:11.606469+07:00 |                           262.59 |
| 14  | c11     | person_detected | 2026-09-01 15:54:13.786625+07:00 | 2026-09-01 15:54:19.901516+07:00 |                           300.51 |
| 15  | c12     | person_detected | 2026-09-01 15:54:14.235692+07:00 | 2026-09-01 15:54:22.509934+07:00 |                           378.44 |
| 16  | c10     | person_detected | 2026-09-01 15:58:09.904807+07:00 | 2026-09-01 15:58:11.953954+07:00 |                            75.99 |
| 17  | c11     | zone_entry      | 2026-09-01 16:00:08.586251+07:00 | 2026-09-01 16:00:08.763967+07:00 |                            93.77 |
| 18  | c11     | zone_entry      | 2026-09-01 16:00:18.223146+07:00 | 2026-09-01 16:00:18.752753+07:00 |                           104.07 |
| 19  | c11     | zone_entry      | 2026-09-01 16:00:34.799538+07:00 | 2026-09-01 16:00:34.897282+07:00 |                            75.57 |
| 20  | c11     | zone_entry      | 2026-09-01 16:01:24.548301+07:00 | 2026-09-01 16:01:24.701205+07:00 |                            73.54 |
| 21  | c11     | person_detected | 2026-09-01 16:01:40.147893+07:00 | 2026-09-01 16:01:42.661076+07:00 |                            73.42 |
| 22  | c10     | person_detected | 2026-09-01 16:03:25.069434+07:00 | 2026-09-01 16:03:27.706163+07:00 |                            35.35 |
| 23  | c10     | person_detected | 2026-09-01 16:03:39.155742+07:00 | 2026-09-01 16:03:42.122859+07:00 |                            59.14 |
| 24  | c11     | zone_entry      | 2026-09-01 16:03:58.297682+07:00 | 2026-09-01 16:03:58.449579+07:00 |                           144.99 |
| 25  | c11     | zone_entry      | 2026-09-01 16:04:02.376832+07:00 | 2026-09-01 16:04:02.470377+07:00 |                           175.97 |
| 26  | c11     | person_detected | 2026-09-01 16:04:02.458440+07:00 | 2026-09-01 16:04:05.193515+07:00 |                            90.42 |
| 27  | c12     | person_detected | 2026-09-01 16:04:37.409379+07:00 | 2026-09-01 16:04:41.240528+07:00 |                           196.25 |
| 28  | c12     | person_detected | 2026-09-01 16:04:39.700930+07:00 | 2026-09-01 16:04:44.434703+07:00 |                           127.51 |
| 29  | c12     | person_detected | 2026-09-01 16:04:42.610162+07:00 | 2026-09-01 16:04:49.161785+07:00 |                           953.04 |


## 2. Tracklet terkait (`started_at`/`ended_at` dipakai buat rentang pencarian CSV)

| id  | camera  | started_at                       | ended_at (= timestamp event)     |
| --- | ------- | -------------------------------- | -------------------------------- |
| 1   | c10     | 2026-09-01 15:49:20.906251+07:00 | 2026-09-01 15:49:24.982107+07:00 |
| 2   | c12     | 2026-09-01 15:49:29.379560+07:00 | 2026-09-01 15:49:30.392655+07:00 |
| 3   | c11     | 2026-09-01 15:49:33.629208+07:00 | 2026-09-01 15:49:40.690029+07:00 |
| 4   | c11     | — (zone_entry, gak ada tracklet) | 2026-09-01 15:50:09.129096+07:00 |
| 5   | c11     | 2026-09-01 15:50:09.236559+07:00 | 2026-09-01 15:50:17.726512+07:00 |
| 6   | c12     | 2026-09-01 15:50:25.857215+07:00 | 2026-09-01 15:50:25.857215+07:00 |
| 7   | c10     | 2026-09-01 15:50:29.467859+07:00 | 2026-09-01 15:50:35.652045+07:00 |
| 8   | c8      | 2026-09-01 15:50:32.829873+07:00 | 2026-09-01 15:50:35.895691+07:00 |
| 9   | c10     | 2026-09-01 15:53:28.511816+07:00 | 2026-09-01 15:53:28.641034+07:00 |
| 10  | c10     | 2026-09-01 15:53:38.449723+07:00 | 2026-09-01 15:53:46.371457+07:00 |
| 11  | c12     | 2026-09-01 15:53:53.763192+07:00 | 2026-09-01 15:53:56.414890+07:00 |
| 12  | c12     | 2026-09-01 15:54:00.142266+07:00 | 2026-09-01 15:54:01.086542+07:00 |
| 13  | c12     | 2026-09-01 15:53:54.880507+07:00 | 2026-09-01 15:54:04.767157+07:00 |
| 14  | c11     | 2026-09-01 15:54:12.242282+07:00 | 2026-09-01 15:54:13.786625+07:00 |
| 15  | c12     | 2026-09-01 15:54:14.050944+07:00 | 2026-09-01 15:54:14.235692+07:00 |
| 16  | c10     | 2026-09-01 15:58:04.504426+07:00 | 2026-09-01 15:58:09.904807+07:00 |
| 17  | c11     | — (zone_entry, gak ada tracklet) | 2026-09-01 16:00:08.586251+07:00 |
| 18  | c11     | — (zone_entry, gak ada tracklet) | 2026-09-01 16:00:18.223146+07:00 |
| 19  | c11     | — (zone_entry, gak ada tracklet) | 2026-09-01 16:00:34.799538+07:00 |
| 20  | c11     | — (zone_entry, gak ada tracklet) | 2026-09-01 16:01:24.548301+07:00 |
| 21  | c11     | 2026-09-01 16:01:24.687137+07:00 | 2026-09-01 16:01:40.147893+07:00 |
| 22  | c10     | 2026-09-01 16:03:17.454127+07:00 | 2026-09-01 16:03:25.069434+07:00 |
| 23  | c10     | 2026-09-01 16:03:33.954182+07:00 | 2026-09-01 16:03:39.155742+07:00 |
| 24  | c11     | — (zone_entry, gak ada tracklet) | 2026-09-01 16:03:58.297682+07:00 |
| 25  | c11     | — (zone_entry, gak ada tracklet) | 2026-09-01 16:04:02.376832+07:00 |
| 26  | c11     | 2026-09-01 16:03:51.937362+07:00 | 2026-09-01 16:04:02.458440+07:00 |
| 27  | c12     | 2026-09-01 16:04:37.409379+07:00 | 2026-09-01 16:04:37.409379+07:00 |
| 28  | c12     | 2026-09-01 16:04:36.367476+07:00 | 2026-09-01 16:04:39.700930+07:00 |
| 29  | c12     | 2026-09-01 16:04:41.381037+07:00 | 2026-09-01 16:04:42.610162+07:00 |


## 3. Baris `latency.csv` yang dipakai

| event | camera  | baris CSV   | jumlah baris |
| ----- | ------- | ----------- | -----------: |
| #1    | c10     | 910-1262    |           73 |
| #2    | c12     | 1630-1716   |           19 |
| #3    | c11     | 2012-2528   |          107 |
| #4    | c11     | 4951        |            1 |
| #5    | c11     | 4956-5627   |          138 |
| #6    | c12     | 6268        |            1 |
| #7    | c10     | 6553-6992   |           91 |
| #8    | c8      | 6848-7011   |           35 |
| #9    | c10     | 21668-21678 |            3 |
| #10   | c10     | 22233-22548 |           66 |
| #11   | c12     | 22992-23082 |           20 |
| #12   | c12     | 23157       |            1 |
| #13   | c12     | 23044-23212 |           36 |
| #14   | c11     | 23319       |            1 |
| #15   | c12     | 23335       |            1 |
| #16   | c10     | 43619-43925 |           62 |
| #17   | c11     | 51602       |            1 |
| #18   | c11     | 51941       |            1 |
| #19   | c11     | 52577       |            1 |
| #20   | c11     | 55971       |            1 |
| #21   | c11     | 55975-56720 |          152 |
| #22   | c10     | 64140-64461 |           64 |
| #23   | c10     | 65134-65420 |           61 |
| #24   | c11     | 66249       |            1 |
| #25   | c11     | 66423       |            1 |
| #26   | c11     | 65959-66423 |           98 |
| #27   | c12     | 69523       |            1 |
| #28   | c12     | 69490-69598 |           25 |
| #29   | c12     | 69623-69667 |           10 |


## 4. Console log (browser) + ringkasan per-layer

`end_to_end_ms` sumbernya console log browser (di-paste manual ke `END_TO_END_MS` di script ini — gak ada di DB). `backend_delay_ms` dihitung ulang dari DB (`created_at - timestamp`), bukan dari log, biar konsisten walau log browser hilang/gak lengkap.

| Event | Kamera  | Tipe            | AI latency total (ms) | AI latency frame terakhir (ms) | Backend delay (ms) | Browser wait (ms) | End-to-end (ms) |
| ----- | ------- | --------------- | --------------------: | -----------------------------: | -----------------: | ----------------: | --------------: |
| #1    | c10     | person_detected |               6,785.9 |                         23.782 |              1,604 |            21,747 |          23,351 |
| #2    | c12     | person_detected |               1,811.9 |                        110.197 |              1,545 |            16,396 |          17,941 |
| #3    | c11     | person_detected |              10,678.8 |                        124.040 |              2,152 |             5,491 |           7,643 |
| #4    | c11     | zone_entry      |                  64.8 |                         64.761 |                120 |             9,071 |           9,191 |
| #5    | c11     | person_detected |              14,070.4 |                        102.000 |              1,868 |            28,735 |          30,603 |
| #6    | c12     | person_detected |                 600.1 |                        600.097 |              1,467 |            21,005 |          22,472 |
| #7    | c10     | person_detected |               8,933.4 |                        112.367 |              1,947 |            10,730 |          12,677 |
| #8    | c8      | person_detected |               3,550.0 |                        351.012 |              2,601 |             9,833 |          12,434 |
| #9    | c10     | person_detected |                 294.7 |                         69.537 |              2,213 |            17,559 |          19,772 |
| #10   | c10     | person_detected |              10,196.3 |                        121.472 |              2,497 |            29,491 |          31,988 |
| #11   | c12     | person_detected |               3,384.4 |                        203.339 |              4,953 |            16,992 |          21,945 |
| #12   | c12     | person_detected |                 159.9 |                        159.888 |              6,468 |            10,805 |          17,273 |
| #13   | c12     | person_detected |              10,535.3 |                      1,096.888 |              6,839 |             6,753 |          13,592 |
| #14   | c11     | person_detected |                 223.2 |                        223.197 |              6,115 |            28,404 |          34,519 |
| #15   | c12     | person_detected |                 182.1 |                        182.053 |              8,274 |            25,796 |          34,070 |
| #16   | c10     | person_detected |               7,455.7 |                         27.423 |              2,049 |             6,355 |           8,404 |
| #17   | c11     | zone_entry      |                  93.8 |                         93.770 |                178 |             9,555 |           9,733 |
| #18   | c11     | zone_entry      |                 104.1 |                        104.066 |                530 |            29,836 |          30,366 |
| #19   | c11     | zone_entry      |                  75.6 |                         75.571 |                 98 |            13,692 |          13,790 |
| #20   | c11     | zone_entry      |                  73.5 |                         73.538 |                153 |            23,643 |          23,796 |
| #21   | c11     | person_detected |              19,720.3 |                        131.392 |              2,513 |             5,684 |           8,197 |
| #22   | c10     | person_detected |               8,994.8 |                        128.899 |              2,637 |            20,660 |          23,297 |
| #23   | c10     | person_detected |               7,454.2 |                         64.463 |              2,967 |             6,244 |           9,211 |
| #24   | c11     | zone_entry      |                 145.0 |                        144.991 |                152 |            19,878 |          20,030 |
| #25   | c11     | zone_entry      |                 176.0 |                        175.966 |                 94 |            15,857 |          15,951 |
| #26   | c11     | person_detected |              14,147.1 |                        175.966 |              2,735 |            13,134 |          15,869 |
| #27   | c12     | person_detected |                 233.3 |                        233.275 |              3,831 |             7,308 |          11,139 |
| #28   | c12     | person_detected |               3,799.9 |                        123.026 |              4,734 |             4,114 |           8,848 |
| #29   | c12     | person_detected |               1,576.1 |                        135.115 |              6,552 |            29,148 |          35,700 |


### 4a. Versi filter — cuma `person_detected`

Kolom **`E2E dari frame terakhir masuk`** = delta waktu ASLI antara frame terakhir & frame SEBELUMNYA (kamera sama, dari timestamp baris `latency.csv`, bukan `frame_id` — itu cuma counter) + `End-to-End`. Awalnya dicoba pakai `sum(decode+detection+tracking+reid)` frame terakhir, tapi itu OVERESTIMATE — `decode_ms` diukur di thread capture terpisah/concurrent (lihat Bagian 0), jadi jumlahnya gak sama dengan selisih waktu asli antar-frame (terbukti: event #1 delta asli 12,28ms vs sum logis 23,77ms). Delta antar-frame ini approksimasi yang lebih deket ke kenyataan, walau masih belum termasuk jeda antre di `frame_q` sebelum frame ini MULAI diproses.

Kolom **`E2E + Total AI Last`** = `Total AI Last + End-to-End` = estimasi BATAS ATAS (upper bound) — `Total AI Last` overestimate lag frame terakhir (~2x, decode konkuren + detection di-batch), jadi pakai ini sebagai batas atas, `E2E dari frame terakhir masuk` sebagai best estimate.

| Event | Kamera | Total AI (ms) | Total AI Last (ms) | Backend Delay (ms) | Browser Wait (ms) | End-to-End (ms) | E2E dari frame terakhir masuk (ms) | E2E + Total AI Last (ms) | E2E dari awal kemunculan (ms) |
| ----- | ------ | ------------: | -----------------: | -----------------: | ----------------: | --------------: | ---------------------------------: | -----------------------: | ----------------------------: |
| #1    | c10    |       6,785.9 |             23.782 |              1,604 |            21,747 |          23,351 |                           23,363.3 |                 23,374.8 |                      27,426.9 |
| #2    | c12    |       1,811.9 |            110.197 |              1,545 |            16,396 |          17,941 |                           18,002.6 |                 18,051.2 |                      18,954.1 |
| #3    | c11    |      10,678.8 |            124.040 |              2,152 |             5,491 |           7,643 |                            7,734.0 |                  7,767.0 |                      14,703.8 |
| #5    | c11    |      14,070.4 |            102.000 |              1,868 |            28,735 |          30,603 |                           30,655.1 |                 30,705.0 |                      39,093.0 |
| #6    | c12    |         600.1 |            600.097 |              1,467 |            21,005 |          22,472 |                           23,070.5 |                 23,072.1 |                      22,472.0 |
| #7    | c10    |       8,933.4 |            112.367 |              1,947 |            10,730 |          12,677 |                           12,748.2 |                 12,789.4 |                      18,861.2 |
| #8    | c8     |       3,550.0 |            351.012 |              2,601 |             9,833 |          12,434 |                           12,633.0 |                 12,785.0 |                      15,499.8 |
| #9    | c10    |         294.7 |             69.537 |              2,213 |            17,559 |          19,772 |                           19,805.6 |                 19,841.5 |                      19,901.2 |
| #10   | c10    |      10,196.3 |            121.472 |              2,497 |            29,491 |          31,988 |                           32,056.7 |                 32,109.5 |                      39,909.7 |
| #11   | c12    |       3,384.4 |            203.339 |              4,953 |            16,992 |          21,945 |                           22,109.7 |                 22,148.3 |                      24,596.7 |
| #12   | c12    |         159.9 |            159.888 |              6,468 |            10,805 |          17,273 |                           17,436.2 |                 17,432.9 |                      18,217.3 |
| #13   | c12    |      10,535.3 |          1,096.888 |              6,839 |             6,753 |          13,592 |                           14,658.4 |                 14,688.9 |                      23,478.7 |
| #14   | c11    |         223.2 |            223.197 |              6,115 |            28,404 |          34,519 |                           34,773.1 |                 34,742.2 |                      36,063.3 |
| #15   | c12    |         182.1 |            182.053 |              8,274 |            25,796 |          34,070 |                           34,221.1 |                 34,252.1 |                      34,254.7 |
| #16   | c10    |       7,455.7 |             27.423 |              2,049 |             6,355 |           8,404 |                            8,430.0 |                  8,431.4 |                      13,804.4 |
| #21   | c11    |      19,720.3 |            131.392 |              2,513 |             5,684 |           8,197 |                            8,281.1 |                  8,328.4 |                      23,657.8 |
| #22   | c10    |       8,994.8 |            128.899 |              2,637 |            20,660 |          23,297 |                           23,418.3 |                 23,425.9 |                      30,912.3 |
| #23   | c10    |       7,454.2 |             64.463 |              2,967 |             6,244 |           9,211 |                            9,260.6 |                  9,275.5 |                      14,412.6 |
| #26   | c11    |      14,147.1 |            175.966 |              2,735 |            13,134 |          15,869 |                           16,016.5 |                 16,045.0 |                      26,390.1 |
| #27   | c12    |         233.3 |            233.275 |              3,831 |             7,308 |          11,139 |                           11,322.0 |                 11,372.3 |                      11,139.0 |
| #28   | c12    |       3,799.9 |            123.026 |              4,734 |             4,114 |           8,848 |                            8,944.4 |                  8,971.0 |                      12,181.5 |
| #29   | c12    |       1,576.1 |            135.115 |              6,552 |            29,148 |          35,700 |                           35,804.2 |                 35,835.1 |                      36,929.1 |

### Ringkasan (n=29 event)

| Layer | Rata-rata | Min | Max |
|---|---:|---:|---:|
| AI latency (akumulasi per tracklet) | 4,673.1 ms | 64.8 ms | 19,720.3 ms |
| Backend delay | 2,754.6 ms | 93.5 ms | 8,274.2 ms |
| Browser wait (poll) | 15,997.1 ms | 4,114.2 ms | 29,836.4 ms |
| End-to-end | 18,751.8 ms | 7,643.0 ms | 35,700.0 ms |
| AI latency per frame (per-event min/max; weighted avg = 126.4ms dari 135,520ms / 1072 frame) | 151.8 ms | 64.8 ms | 600.1 ms |

**Catatan**: AI latency = biaya compute (akumulasi banyak frame), BUKAN wall-clock — jangan dijumlah ke 3 kolom lain, beda rentang waktu (AI latency terjadi sebelum `ended_at`, tiga kolom lain sesudahnya).


## 5. Tabel gabungan — 1 baris per event, semua sumber

Semua kolom di atas (bagian 1-4) digabung jadi 1 baris per event, biar gak perlu gulir-gulir cocokin antar tabel.

| id  | camera  | type            | timestamp (AI)                   | started_at (tracklet)            | created_at (backend)             | csv baris   | AI latency total (ms) | AI latency/frame (ms) | AI latency frame terakhir (ms) | backend delay (ms) | browser wait (ms) | end-to-end (ms) |
| --- | ------- | --------------- | -------------------------------- | -------------------------------- | -------------------------------- | ----------- | --------------------: | --------------------: | -----------------------------: | -----------------: | ----------------: | --------------: |
| 1   | c10     | person_detected | 2026-09-01 15:49:24.982107+07:00 | 2026-09-01 15:49:20.906251+07:00 | 2026-09-01 15:49:26.586259+07:00 | 910-1262    |               6,785.9 |                  93.0 |                         23.782 |              1,604 |            21,747 |          23,351 |
| 2   | c12     | person_detected | 2026-09-01 15:49:30.392655+07:00 | 2026-09-01 15:49:29.379560+07:00 | 2026-09-01 15:49:31.937218+07:00 | 1630-1716   |               1,811.9 |                  95.4 |                        110.197 |              1,545 |            16,396 |          17,941 |
| 3   | c11     | person_detected | 2026-09-01 15:49:40.690029+07:00 | 2026-09-01 15:49:33.629208+07:00 | 2026-09-01 15:49:42.842322+07:00 | 2012-2528   |              10,678.8 |                  99.8 |                        124.040 |              2,152 |             5,491 |           7,643 |
| 4   | c11     | zone_entry      | 2026-09-01 15:50:09.129096+07:00 | —                                | 2026-09-01 15:50:09.249589+07:00 | 4951        |                  64.8 |                  64.8 |                         64.761 |                120 |             9,071 |           9,191 |
| 5   | c11     | person_detected | 2026-09-01 15:50:17.726512+07:00 | 2026-09-01 15:50:09.236559+07:00 | 2026-09-01 15:50:19.594392+07:00 | 4956-5627   |              14,070.4 |                 102.0 |                        102.000 |              1,868 |            28,735 |          30,603 |
| 6   | c12     | person_detected | 2026-09-01 15:50:25.857215+07:00 | 2026-09-01 15:50:25.857215+07:00 | 2026-09-01 15:50:27.324548+07:00 | 6268        |                 600.1 |                 600.1 |                        600.097 |              1,467 |            21,005 |          22,472 |
| 7   | c10     | person_detected | 2026-09-01 15:50:35.652045+07:00 | 2026-09-01 15:50:29.467859+07:00 | 2026-09-01 15:50:37.599167+07:00 | 6553-6992   |               8,933.4 |                  98.2 |                        112.367 |              1,947 |            10,730 |          12,677 |
| 8   | c8      | person_detected | 2026-09-01 15:50:35.895691+07:00 | 2026-09-01 15:50:32.829873+07:00 | 2026-09-01 15:50:38.496473+07:00 | 6848-7011   |               3,550.0 |                 101.4 |                        351.012 |              2,601 |             9,833 |          12,434 |
| 9   | c10     | person_detected | 2026-09-01 15:53:28.641034+07:00 | 2026-09-01 15:53:28.511816+07:00 | 2026-09-01 15:53:30.853814+07:00 | 21668-21678 |                 294.7 |                  98.2 |                         69.537 |              2,213 |            17,559 |          19,772 |
| 10  | c10     | person_detected | 2026-09-01 15:53:46.371457+07:00 | 2026-09-01 15:53:38.449723+07:00 | 2026-09-01 15:53:48.868538+07:00 | 22233-22548 |              10,196.3 |                 154.5 |                        121.472 |              2,497 |            29,491 |          31,988 |
| 11  | c12     | person_detected | 2026-09-01 15:53:56.414890+07:00 | 2026-09-01 15:53:53.763192+07:00 | 2026-09-01 15:54:01.367842+07:00 | 22992-23082 |               3,384.4 |                 169.2 |                        203.339 |              4,953 |            16,992 |          21,945 |
| 12  | c12     | person_detected | 2026-09-01 15:54:01.086542+07:00 | 2026-09-01 15:54:00.142266+07:00 | 2026-09-01 15:54:07.554272+07:00 | 23157       |                 159.9 |                 159.9 |                        159.888 |              6,468 |            10,805 |          17,273 |
| 13  | c12     | person_detected | 2026-09-01 15:54:04.767157+07:00 | 2026-09-01 15:53:54.880507+07:00 | 2026-09-01 15:54:11.606469+07:00 | 23044-23212 |              10,535.3 |                 292.6 |                      1,096.888 |              6,839 |             6,753 |          13,592 |
| 14  | c11     | person_detected | 2026-09-01 15:54:13.786625+07:00 | 2026-09-01 15:54:12.242282+07:00 | 2026-09-01 15:54:19.901516+07:00 | 23319       |                 223.2 |                 223.2 |                        223.197 |              6,115 |            28,404 |          34,519 |
| 15  | c12     | person_detected | 2026-09-01 15:54:14.235692+07:00 | 2026-09-01 15:54:14.050944+07:00 | 2026-09-01 15:54:22.509934+07:00 | 23335       |                 182.1 |                 182.1 |                        182.053 |              8,274 |            25,796 |          34,070 |
| 16  | c10     | person_detected | 2026-09-01 15:58:09.904807+07:00 | 2026-09-01 15:58:04.504426+07:00 | 2026-09-01 15:58:11.953954+07:00 | 43619-43925 |               7,455.7 |                 120.3 |                         27.423 |              2,049 |             6,355 |           8,404 |
| 17  | c11     | zone_entry      | 2026-09-01 16:00:08.586251+07:00 | —                                | 2026-09-01 16:00:08.763967+07:00 | 51602       |                  93.8 |                  93.8 |                         93.770 |                178 |             9,555 |           9,733 |
| 18  | c11     | zone_entry      | 2026-09-01 16:00:18.223146+07:00 | —                                | 2026-09-01 16:00:18.752753+07:00 | 51941       |                 104.1 |                 104.1 |                        104.066 |                530 |            29,836 |          30,366 |
| 19  | c11     | zone_entry      | 2026-09-01 16:00:34.799538+07:00 | —                                | 2026-09-01 16:00:34.897282+07:00 | 52577       |                  75.6 |                  75.6 |                         75.571 |                 98 |            13,692 |          13,790 |
| 20  | c11     | zone_entry      | 2026-09-01 16:01:24.548301+07:00 | —                                | 2026-09-01 16:01:24.701205+07:00 | 55971       |                  73.5 |                  73.5 |                         73.538 |                153 |            23,643 |          23,796 |
| 21  | c11     | person_detected | 2026-09-01 16:01:40.147893+07:00 | 2026-09-01 16:01:24.687137+07:00 | 2026-09-01 16:01:42.661076+07:00 | 55975-56720 |              19,720.3 |                 129.7 |                        131.392 |              2,513 |             5,684 |           8,197 |
| 22  | c10     | person_detected | 2026-09-01 16:03:25.069434+07:00 | 2026-09-01 16:03:17.454127+07:00 | 2026-09-01 16:03:27.706163+07:00 | 64140-64461 |               8,994.8 |                 140.5 |                        128.899 |              2,637 |            20,660 |          23,297 |
| 23  | c10     | person_detected | 2026-09-01 16:03:39.155742+07:00 | 2026-09-01 16:03:33.954182+07:00 | 2026-09-01 16:03:42.122859+07:00 | 65134-65420 |               7,454.2 |                 122.2 |                         64.463 |              2,967 |             6,244 |           9,211 |
| 24  | c11     | zone_entry      | 2026-09-01 16:03:58.297682+07:00 | —                                | 2026-09-01 16:03:58.449579+07:00 | 66249       |                 145.0 |                 145.0 |                        144.991 |                152 |            19,878 |          20,030 |
| 25  | c11     | zone_entry      | 2026-09-01 16:04:02.376832+07:00 | —                                | 2026-09-01 16:04:02.470377+07:00 | 66423       |                 176.0 |                 176.0 |                        175.966 |                 94 |            15,857 |          15,951 |
| 26  | c11     | person_detected | 2026-09-01 16:04:02.458440+07:00 | 2026-09-01 16:03:51.937362+07:00 | 2026-09-01 16:04:05.193515+07:00 | 65959-66423 |              14,147.1 |                 144.4 |                        175.966 |              2,735 |            13,134 |          15,869 |
| 27  | c12     | person_detected | 2026-09-01 16:04:37.409379+07:00 | 2026-09-01 16:04:37.409379+07:00 | 2026-09-01 16:04:41.240528+07:00 | 69523       |                 233.3 |                 233.3 |                        233.275 |              3,831 |             7,308 |          11,139 |
| 28  | c12     | person_detected | 2026-09-01 16:04:39.700930+07:00 | 2026-09-01 16:04:36.367476+07:00 | 2026-09-01 16:04:44.434703+07:00 | 69490-69598 |               3,799.9 |                 152.0 |                        123.026 |              4,734 |             4,114 |           8,848 |
| 29  | c12     | person_detected | 2026-09-01 16:04:42.610162+07:00 | 2026-09-01 16:04:41.381037+07:00 | 2026-09-01 16:04:49.161785+07:00 | 69623-69667 |               1,576.1 |                 157.6 |                        135.115 |              6,552 |            29,148 |          35,700 |


## 6. Statistik per-komponen, dipisah `person_detected` vs `zone_entry`

`decode_ms`/`detection_ms` SENGAJA gak dimasukin di sini — dua-duanya jalan tiap siklus terlepas dari ada event apa nggak (lihat Bagian 0), jadi ngerata-ratain dari subset frame yang kebetulan masuk rentang 29 event ini BUKAN angka yang lebih fair (sample ~1,5% dari total, berpotensi bias). **Pakai Bagian 0 buat `decode_ms`/`detection_ms`.** `reid_ms_real`/`matching_ms_real` cuma ngitung yang BENERAN kerja (lolos threshold `REID_REAL_MS`/`MATCH_REAL_MS`, sama kayak Bagian 0 — bukan sekadar nonzero) — juga tersedia sebagai `latency_stats.csv`.

| event_type | component | n | avg_ms | min_ms | max_ms |
|---|---|---:|---:|---:|---:|
| person_detected | `tracking_ms` | 895 | 0.679 | 0.148 | 34.85 |
| person_detected | `reid_ms_real` | 801 | 45.174 | 26.201 | 908.91 |
| person_detected | `matching_ms_real` | 243 | 0.210 | 0.051 | 15.88 |
| person_detected | `total_ai_ms` | 1065 | 126.561 | 10.092 | 1,096.89 |
| zone_entry | `tracking_ms` | 2 | 0.719 | 0.639 | 0.80 |
| zone_entry | `reid_ms_real` | 2 | 44.732 | 44.313 | 45.15 |
| zone_entry | `matching_ms_real` | 1 | 0.073 | 0.073 | 0.07 |
| zone_entry | `total_ai_ms` | 7 | 104.666 | 64.761 | 175.97 |

## 7. Breakdown per kejadian NYATA — 53 tracklet di DB (bukan 29 event sample, ini SEMUA yang beneran lolos sampai resolve)

Titik acuan `ended_at` = frame TERAKHIR orang itu beneran kedeteksi (sama kayak Bagian 1-5, bukan "frame terakhir didecode"). Per kejadian: total ms & jumlah frame tiap tahap dalam rentang `[started_at, ended_at]` kejadian itu di `latency.csv`. Kolom **`new?`** = tracklet PERTAMA (urut waktu) yang munculin `person_id` itu → jadi `Unknown #N` BARU (23 identitas unik); yang lain (`match`) = tracklet susulan yang ke-MATCH ke identitas yang udah ada (re-appearance). Kolom **`*_last`** = nilai di frame TERAKHIR doang (baris CSV paling akhir dalam rentang tracklet ini) — beda sama kolom tanpa suffix yang AKUMULASI semua frame.

| id | camera | track_id | person_id | new? | n_detections (DB) | n frame (CSV) | decode_ms | detection_ms | tracking_ms | reid_ms | matching_ms | total_ai_ms | decode_last | detection_last | tracking_last | reid_last | matching_last | total_ai_ms_last |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | c10 | 1 | 1 | **baru** | 64 | 73 | 3,401.20 | 1,884.49 | 20.78 | 1,478.06 | 1.37 | 6,785.88 | 12.252 | 11.237 | 0.276 | 0.003 | 0.014 | 23.782 |
| 2 | c12 | 4 | 2 | **baru** | 15 | 19 | 870.11 | 524.23 | 3.73 | 413.49 | 0.38 | 1,811.94 | 49.265 | 32.781 | 0.266 | 27.866 | 0.019 | 110.197 |
| 3 | c11 | 6 | 3 | **baru** | 80 | 107 | 4,986.38 | 2,833.58 | 24.46 | 2,832.08 | 2.28 | 10,678.79 | 34.077 | 31.026 | 0.412 | 58.501 | 0.024 | 124.040 |
| 4 | c11 | 9 | 1 | match | 112 | 119 | 5,251.32 | 4,034.32 | 39.68 | 4,024.37 | 3.91 | 13,353.62 | 4.067 | 34.709 | 0.388 | 29.721 | 0.031 | 68.916 |
| 5 | c11 | 9 | 4 | **baru** | 116 | 138 | 6,720.49 | 3,912.06 | 40.16 | 3,393.76 | 3.89 | 14,070.36 | 50.584 | 22.274 | 0.360 | 28.764 | 0.018 | 102.000 |
| 6 | c12 | 5 | 4 | match | 1 | 1 | 27.37 | 44.31 | 0.00 | 0.00 | 0.00 | 71.69 | 27.369 | 44.315 | 0.000 | 0.000 | 0.001 | 71.685 |
| 7 | c12 | 4 | 5 | **baru** | 1 | 1 | 1.97 | 598.12 | 0.00 | 0.00 | 0.00 | 600.10 | 1.972 | 598.124 | 0.000 | 0.000 | 0.001 | 600.097 |
| 8 | c10 | 1 | 6 | **baru** | 84 | 91 | 4,196.50 | 3,084.81 | 39.54 | 1,608.91 | 3.60 | 8,933.35 | 42.326 | 38.597 | 0.586 | 30.823 | 0.035 | 112.367 |
| 9 | c8 | 10 | 7 | **baru** | 18 | 35 | 1,779.63 | 1,231.92 | 9.53 | 528.03 | 0.94 | 3,550.05 | 288.375 | 27.745 | 0.670 | 34.190 | 0.031 | 351.012 |
| 10 | c10 | 12 | 7 | match | 15 | 33 | 2,963.79 | 964.36 | 6.06 | 471.88 | 0.75 | 4,406.84 | 5.792 | 11.411 | 0.000 | 0.000 | 0.006 | 17.210 |
| 11 | c11 | 15 | 3 | match | 82 | 104 | 4,327.59 | 5,469.50 | 56.53 | 3,419.68 | 3.97 | 13,277.26 | 54.715 | 39.004 | 0.458 | 33.438 | 0.029 | 127.644 |
| 12 | c11 | 16 | 3 | match | 170 | 249 | 11,010.16 | 12,738.78 | 122.60 | 7,150.86 | 13.84 | 31,036.23 | 32.084 | 45.483 | 0.729 | 33.438 | 0.042 | 111.777 |
| 13 | c10 | 17 | 8 | **baru** | 2 | 3 | 121.03 | 99.95 | 0.61 | 72.97 | 0.12 | 294.68 | 36.207 | 33.324 | 0.000 | 0.000 | 0.006 | 69.537 |
| 14 | c10 | 17 | 8 | match | 21 | 27 | 1,045.82 | 1,166.45 | 13.15 | 713.21 | 0.75 | 2,939.40 | 50.426 | 41.974 | 0.800 | 39.174 | 0.038 | 132.413 |
| 15 | c10 | 21 | 9 | **baru** | 56 | 66 | 2,588.62 | 3,681.02 | 48.08 | 3,874.74 | 3.82 | 10,196.27 | 54.049 | 66.755 | 0.649 | 0.005 | 0.015 | 121.472 |
| 16 | c10 | 22 | 8 | match | 43 | 50 | 2,053.64 | 2,823.06 | 42.37 | 3,617.29 | 3.47 | 8,539.83 | 36.403 | 48.267 | 0.730 | 39.356 | 0.038 | 124.793 |
| 17 | c12 | 26 | 9 | match | 2 | 1 | 44.59 | 52.67 | 1.49 | 74.72 | 0.25 | 173.73 | 44.594 | 52.672 | 1.493 | 74.724 | 0.248 | 173.730 |
| 18 | c12 | 25 | 10 | **baru** | 17 | 20 | 1,040.69 | 1,106.50 | 21.03 | 1,213.79 | 2.35 | 3,384.36 | 42.612 | 82.181 | 1.088 | 77.288 | 0.172 | 203.339 |
| 19 | c12 | 28 | 8 | match | 16 | 16 | 958.89 | 2,407.04 | 21.64 | 1,460.74 | 2.33 | 4,850.64 | 19.012 | 823.882 | 1.203 | 0.000 | 0.025 | 844.122 |
| 20 | c12 | 31 | 11 | **baru** | 2 | 1 | 2.40 | 65.25 | 0.97 | 90.60 | 0.67 | 159.89 | 2.401 | 65.248 | 0.974 | 90.597 | 0.667 | 159.888 |
| 21 | c12 | 27 | 12 | **baru** | 21 | 36 | 1,696.84 | 4,253.15 | 43.77 | 4,536.77 | 4.81 | 10,535.33 | 42.850 | 143.767 | 1.187 | 908.908 | 0.175 | 1,096.888 |
| 22 | c10 | 34 | 3 | match | 2 | 1 | 42.10 | 51.30 | 1.22 | 33.77 | 0.10 | 128.50 | 42.104 | 51.299 | 1.222 | 33.772 | 0.097 | 128.495 |
| 24 | c10 | 35 | 3 | match | 7 | 7 | 287.04 | 620.81 | 5.30 | 237.86 | 0.48 | 1,151.50 | 30.607 | 58.099 | 0.795 | 49.500 | 0.056 | 139.057 |
| 23 | c12 | 33 | 12 | match | 15 | 14 | 445.76 | 2,028.17 | 18.66 | 2,078.27 | 3.82 | 4,574.69 | 27.498 | 58.099 | 0.790 | 100.586 | 1.170 | 188.143 |
| 25 | c11 | 39 | 13 | **baru** | 2 | 1 | 29.30 | 56.51 | 0.92 | 136.25 | 0.23 | 223.20 | 29.300 | 56.506 | 0.916 | 136.248 | 0.227 | 223.197 |
| 26 | c12 | 29 | 8 | match | 40 | 40 | 1,363.73 | 5,548.95 | 45.33 | 6,452.99 | 7.60 | 13,418.60 | 4.359 | 56.506 | 0.805 | 346.426 | 0.053 | 408.150 |
| 27 | c12 | 40 | 14 | **baru** | 2 | 1 | 35.46 | 59.94 | 1.17 | 69.61 | 15.88 | 182.05 | 35.460 | 59.938 | 1.169 | 69.609 | 15.877 | 182.053 |
| 28 | c10 | 41 | 8 | match | 47 | 70 | 2,745.50 | 3,572.21 | 36.88 | 1,978.21 | 3.16 | 8,335.97 | 38.285 | 85.007 | 1.219 | 145.631 | 0.035 | 270.177 |
| 29 | c10 | 42 | 8 | match | 4 | 6 | 252.16 | 210.69 | 1.15 | 113.65 | 0.15 | 577.80 | 42.464 | 33.491 | 0.000 | 0.000 | 0.006 | 75.961 |
| 30 | c10 | 43 | 15 | **baru** | 58 | 62 | 2,929.37 | 2,734.83 | 41.31 | 1,746.25 | 3.96 | 7,455.73 | 1.722 | 25.692 | 0.000 | 0.000 | 0.008 | 27.423 |
| 31 | c12 | 44 | 5 | match | 14 | 20 | 1,075.01 | 704.87 | 7.83 | 186.96 | 0.45 | 1,975.12 | 20.210 | 76.952 | 0.544 | 0.004 | 0.045 | 97.754 |
| 32 | c12 | 45 | 15 | match | 13 | 12 | 417.54 | 466.97 | 8.09 | 417.28 | 1.43 | 1,311.32 | 29.617 | 45.558 | 0.704 | 38.809 | 0.860 | 115.548 |
| 33 | c12 | 47 | 15 | match | 456 | 461 | 20,715.03 | 24,529.14 | 277.73 | 22,025.16 | 17.95 | 67,565.02 | 16.970 | 38.065 | 0.442 | 47.026 | 0.027 | 102.530 |
| 34 | c10 | 48 | 8 | match | 6 | 19 | 923.04 | 778.34 | 2.80 | 265.71 | 0.39 | 1,970.28 | 45.996 | 40.617 | 0.000 | 0.000 | 0.011 | 86.624 |
| 35 | c11 | 39 | 13 | match | 18 | 18 | 612.25 | 1,391.89 | 12.74 | 806.51 | 1.25 | 2,824.66 | 32.446 | 46.100 | 0.588 | 34.931 | 0.036 | 114.101 |
| 36 | c10 | 48 | 8 | match | 35 | 34 | 1,261.58 | 3,193.39 | 26.81 | 1,895.49 | 2.07 | 6,379.34 | 68.706 | 55.766 | 0.695 | 33.297 | 0.033 | 158.498 |
| 37 | c11 | 39 | 1 | match | 80 | 80 | 3,006.17 | 4,377.98 | 58.48 | 3,402.41 | 5.15 | 10,850.18 | 46.443 | 39.162 | 0.590 | 35.877 | 0.041 | 122.113 |
| 38 | c11 | 51 | 8 | match | 47 | 58 | 2,225.51 | 3,657.00 | 43.34 | 2,081.98 | 3.27 | 8,011.11 | 47.432 | 41.538 | 0.685 | 37.708 | 0.036 | 127.400 |
| 39 | c9 | 53 | 1 | match | 73 | 78 | 4,842.17 | 4,837.43 | 53.28 | 4,315.73 | 5.17 | 14,053.78 | 62.117 | 42.154 | 0.533 | 34.195 | 0.042 | 139.041 |
| 40 | c8 | 56 | 1 | match | 53 | 57 | 2,388.43 | 3,072.28 | 37.01 | 2,085.13 | 2.89 | 7,585.74 | 41.061 | 27.384 | 0.602 | 34.111 | 0.031 | 103.189 |
| 41 | c11 | 51 | 16 | **baru** | 117 | 152 | 6,497.67 | 8,311.72 | 116.97 | 4,787.57 | 6.41 | 19,720.33 | 48.739 | 82.637 | 0.000 | 0.000 | 0.016 | 131.392 |
| 42 | c12 | 59 | 8 | match | 221 | 220 | 10,479.04 | 12,724.55 | 165.56 | 10,672.54 | 9.84 | 34,051.51 | 42.648 | 48.919 | 0.411 | 52.789 | 0.035 | 144.801 |
| 43 | c10 | 60 | 17 | **baru** | 60 | 64 | 3,175.88 | 3,330.87 | 49.01 | 2,435.86 | 3.16 | 8,994.78 | 14.934 | 49.782 | 0.812 | 63.334 | 0.036 | 128.899 |
| 44 | c10 | 62 | 18 | **baru** | 56 | 61 | 2,808.93 | 3,047.91 | 39.35 | 1,555.00 | 3.03 | 7,454.21 | 15.515 | 48.941 | 0.000 | 0.000 | 0.007 | 64.463 |
| 45 | c11 | 63 | 17 | match | 52 | 58 | 2,240.28 | 5,165.39 | 43.41 | 2,389.94 | 8.03 | 9,847.06 | 39.725 | 116.799 | 0.000 | 0.000 | 0.022 | 156.546 |
| 46 | c11 | 65 | 19 | **baru** | 84 | 98 | 4,327.49 | 6,048.46 | 71.64 | 3,690.95 | 8.56 | 14,147.09 | 30.351 | 100.589 | 0.639 | 44.313 | 0.073 | 175.966 |
| 47 | c12 | 68 | 20 | **baru** | 1 | 1 | 53.48 | 142.60 | 0.79 | 36.38 | 0.02 | 233.28 | 53.475 | 142.602 | 0.794 | 36.384 | 0.020 | 233.275 |
| 48 | c12 | 59 | 21 | **baru** | 26 | 25 | 936.42 | 1,717.65 | 24.75 | 1,118.31 | 2.83 | 3,799.95 | 30.160 | 44.518 | 0.757 | 46.602 | 0.989 | 123.026 |
| 49 | c12 | 69 | 21 | match | 1 | 1 | 41.26 | 411.69 | 1.64 | 0.00 | 0.03 | 454.62 | 41.263 | 411.689 | 1.644 | 0.000 | 0.026 | 454.622 |
| 50 | c12 | 71 | 21 | match | 3 | 3 | 111.32 | 133.48 | 2.46 | 230.39 | 0.49 | 478.14 | 28.802 | 49.394 | 0.909 | 109.239 | 0.309 | 188.652 |
| 51 | c12 | 70 | 22 | **baru** | 11 | 10 | 330.18 | 670.15 | 9.02 | 565.49 | 1.26 | 1,576.11 | 32.926 | 62.000 | 0.764 | 39.401 | 0.023 | 135.115 |
| 52 | c11 | 74 | 21 | match | 19 | 20 | 536.06 | 1,349.47 | 18.99 | 1,662.56 | 1.37 | 3,568.45 | 28.717 | 80.211 | 0.874 | 111.122 | 0.060 | 220.984 |
| 53 | c10 | 75 | 6 | match | 10 | 9 | 360.59 | 644.57 | 12.71 | 680.17 | 0.87 | 1,698.91 | 47.486 | 149.600 | 3.723 | 78.727 | 0.148 | 279.685 |

### Rata-rata per kejadian — SEMUA 53 tracklet (baru + match)

| Tahap | Total ms (akumulasi) | Rata-rata ms/kejadian (akumulasi) | Rata-rata ms (frame TERAKHIR aja) | Rata-rata frame/kejadian |
|---|---:|---:|---:|---:|
| `decode_ms` | 132,584.8 | 2,501.6 | 38.660 | 55.7 |
| `detection_ms` | 158,566.8 | 2,991.8 | 87.177 | 55.7 |
| `tracking_ms` | 1,792.6 | 33.8 | 0.677 | 55.7 |
| `reid_ms` | 121,130.4 | 2,285.5 | 61.631 | 55.7 |
| `matching_ms` | 174.8 | 3.3 | 0.417 | 55.7 |
| `total_ai_ms` | 414,249.2 | 7,816.0 | 188.562 | 55.7 |

Total frame terpakai (semua 53 kejadian): 2,951


## 8. Versi CUMA identitas BARU — 22 kejadian (bukan 53, filter `new?`)

Subset dari Bagian 7, cuma baris `new?` (tracklet yang beneran jadi `Unknown #1`..`Unknown #22` — cocok sama 22 `persons` di DB). Tracklet susulan (`match`, re-appearance orang yang sama) DIBUANG dari sini biar gak dobel-hitung 1 orang berkali-kali.

| id | camera | track_id | person_id | n_detections (DB) | n frame (CSV) | decode_ms | detection_ms | tracking_ms | reid_ms | matching_ms | total_ai_ms | decode_last | detection_last | tracking_last | reid_last | matching_last | total_ai_ms_last |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | c10 | 1 | 1 | 64 | 73 | 3,401.20 | 1,884.49 | 20.78 | 1,478.06 | 1.37 | 6,785.88 | 12.252 | 11.237 | 0.276 | 0.003 | 0.014 | 23.782 |
| 2 | c12 | 4 | 2 | 15 | 19 | 870.11 | 524.23 | 3.73 | 413.49 | 0.38 | 1,811.94 | 49.265 | 32.781 | 0.266 | 27.866 | 0.019 | 110.197 |
| 3 | c11 | 6 | 3 | 80 | 107 | 4,986.38 | 2,833.58 | 24.46 | 2,832.08 | 2.28 | 10,678.79 | 34.077 | 31.026 | 0.412 | 58.501 | 0.024 | 124.040 |
| 5 | c11 | 9 | 4 | 116 | 138 | 6,720.49 | 3,912.06 | 40.16 | 3,393.76 | 3.89 | 14,070.36 | 50.584 | 22.274 | 0.360 | 28.764 | 0.018 | 102.000 |
| 7 | c12 | 4 | 5 | 1 | 1 | 1.97 | 598.12 | 0.00 | 0.00 | 0.00 | 600.10 | 1.972 | 598.124 | 0.000 | 0.000 | 0.001 | 600.097 |
| 8 | c10 | 1 | 6 | 84 | 91 | 4,196.50 | 3,084.81 | 39.54 | 1,608.91 | 3.60 | 8,933.35 | 42.326 | 38.597 | 0.586 | 30.823 | 0.035 | 112.367 |
| 9 | c8 | 10 | 7 | 18 | 35 | 1,779.63 | 1,231.92 | 9.53 | 528.03 | 0.94 | 3,550.05 | 288.375 | 27.745 | 0.670 | 34.190 | 0.031 | 351.012 |
| 13 | c10 | 17 | 8 | 2 | 3 | 121.03 | 99.95 | 0.61 | 72.97 | 0.12 | 294.68 | 36.207 | 33.324 | 0.000 | 0.000 | 0.006 | 69.537 |
| 15 | c10 | 21 | 9 | 56 | 66 | 2,588.62 | 3,681.02 | 48.08 | 3,874.74 | 3.82 | 10,196.27 | 54.049 | 66.755 | 0.649 | 0.005 | 0.015 | 121.472 |
| 18 | c12 | 25 | 10 | 17 | 20 | 1,040.69 | 1,106.50 | 21.03 | 1,213.79 | 2.35 | 3,384.36 | 42.612 | 82.181 | 1.088 | 77.288 | 0.172 | 203.339 |
| 20 | c12 | 31 | 11 | 2 | 1 | 2.40 | 65.25 | 0.97 | 90.60 | 0.67 | 159.89 | 2.401 | 65.248 | 0.974 | 90.597 | 0.667 | 159.888 |
| 21 | c12 | 27 | 12 | 21 | 36 | 1,696.84 | 4,253.15 | 43.77 | 4,536.77 | 4.81 | 10,535.33 | 42.850 | 143.767 | 1.187 | 908.908 | 0.175 | 1,096.888 |
| 25 | c11 | 39 | 13 | 2 | 1 | 29.30 | 56.51 | 0.92 | 136.25 | 0.23 | 223.20 | 29.300 | 56.506 | 0.916 | 136.248 | 0.227 | 223.197 |
| 27 | c12 | 40 | 14 | 2 | 1 | 35.46 | 59.94 | 1.17 | 69.61 | 15.88 | 182.05 | 35.460 | 59.938 | 1.169 | 69.609 | 15.877 | 182.053 |
| 30 | c10 | 43 | 15 | 58 | 62 | 2,929.37 | 2,734.83 | 41.31 | 1,746.25 | 3.96 | 7,455.73 | 1.722 | 25.692 | 0.000 | 0.000 | 0.008 | 27.423 |
| 41 | c11 | 51 | 16 | 117 | 152 | 6,497.67 | 8,311.72 | 116.97 | 4,787.57 | 6.41 | 19,720.33 | 48.739 | 82.637 | 0.000 | 0.000 | 0.016 | 131.392 |
| 43 | c10 | 60 | 17 | 60 | 64 | 3,175.88 | 3,330.87 | 49.01 | 2,435.86 | 3.16 | 8,994.78 | 14.934 | 49.782 | 0.812 | 63.334 | 0.036 | 128.899 |
| 44 | c10 | 62 | 18 | 56 | 61 | 2,808.93 | 3,047.91 | 39.35 | 1,555.00 | 3.03 | 7,454.21 | 15.515 | 48.941 | 0.000 | 0.000 | 0.007 | 64.463 |
| 46 | c11 | 65 | 19 | 84 | 98 | 4,327.49 | 6,048.46 | 71.64 | 3,690.95 | 8.56 | 14,147.09 | 30.351 | 100.589 | 0.639 | 44.313 | 0.073 | 175.966 |
| 47 | c12 | 68 | 20 | 1 | 1 | 53.48 | 142.60 | 0.79 | 36.38 | 0.02 | 233.28 | 53.475 | 142.602 | 0.794 | 36.384 | 0.020 | 233.275 |
| 48 | c12 | 59 | 21 | 26 | 25 | 936.42 | 1,717.65 | 24.75 | 1,118.31 | 2.83 | 3,799.95 | 30.160 | 44.518 | 0.757 | 46.602 | 0.989 | 123.026 |
| 51 | c12 | 70 | 22 | 11 | 10 | 330.18 | 670.15 | 9.02 | 565.49 | 1.26 | 1,576.11 | 32.926 | 62.000 | 0.764 | 39.401 | 0.023 | 135.115 |

### Rata-rata per kejadian — CUMA identitas baru (n=22)

| Tahap | Total ms (akumulasi) | Rata-rata ms/kejadian (akumulasi) | Rata-rata ms (frame TERAKHIR aja) | Rata-rata frame/kejadian |
|---|---:|---:|---:|---:|
| `decode_ms` | 48,530.0 | 2,205.9 | 43.161 | 48.4 |
| `detection_ms` | 49,395.7 | 2,245.3 | 83.012 | 48.4 |
| `tracking_ms` | 607.6 | 27.6 | 0.560 | 48.4 |
| `reid_ms` | 36,184.9 | 1,644.8 | 76.947 | 48.4 |
| `matching_ms` | 69.6 | 3.2 | 0.839 | 48.4 |
| `total_ai_ms` | 134,787.7 | 6,126.7 | 204.519 | 48.4 |

Total frame terpakai (22 identitas baru): 1,065

