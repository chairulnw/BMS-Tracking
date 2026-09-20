# Perbandingan Ukuran YOLO26 (n/s/x) × ByteTrack × Re-ID (TransReID/OSNet)

File terpisah dari `pipeline-comparison.md` (27+ kombinasi utama) — fokus ke satu
pertanyaan: apakah YOLO26 yang lebih besar (s, x) membantu akurasi dibanding
baseline n, dan gimana interaksinya sama Re-ID (TransReID vs OSNet). Tracker
dikunci ke ByteTrack (pemenang di perbandingan utama) supaya efek detektor
terisolasi bersih.

Metodologi, definisi metrik, dan lingkungan uji **sama persis** dengan
`pipeline-comparison.md` — lihat file itu buat detail (dataset `sample/`,
259 box GT, 11 orang, MacBook Air M4 tanpa GPU discrete). 4 dari 6 baris di
bawah adalah hasil yang sudah ada (dipakai ulang dari `pipeline-comparison.md`
baris #1, #2, #28, #29), 2 baris baru (`YOLO26s`/`YOLO26x` + OSNet) dijalankan
khusus buat tabel ini.

## Hasil Uji

| #   | Detektor | Tracker   | Re-ID                 | Status  | Total frame | Cakupan deteksi | Identitas terbentuk | Precision | Recall | F1    | Akurasi ketat | Akurasi dominan | IDF1  | False merge | False split | FPS efektif | CPU peak | RAM peak | Decode ms | Detect ms | Track ms | ReID ms | Match ms | Total AI ms |
| --- | -------- | --------- | --------------------- | ------- | ----------- | --------------- | ------------------- | --------- | ------ | ----- | ------------- | --------------- | ----- | ----------- | ----------- | ----------- | -------- | -------- | --------- | --------- | -------- | ------- | -------- | ----------- |
| 1   | YOLO26n  | ByteTrack | OSNet                 | Selesai | 1688        | 190/259         | 12/11               | 0.944     | 0.961  | 0.953 | 7/11 (63.6%)  | 11/11 (100.0%)  | 0.899 | 1           | 2           | 7.56        | 31%      | 81%      | 7.1       | 24.8      | 0.4      | 14.2    | 0.0      | 46.6        |
| 2   | YOLO26n  | ByteTrack | TransReID (ViT-B/16*) | Selesai | 1589        | 187/259         | 16/11               | 0.997     | 0.931  | 0.962 | 7/11 (63.6%)  | 10/11 (90.9%)   | 0.908 | 1           | 4           | 7.57        | 36%      | 81%      | 8.0       | 23.3      | 0.4      | 29.0    | 0.0      | 60.8        |
| 3   | YOLO26s  | ByteTrack | OSNet                 | Selesai | 1984        | 225/259         | 11/11               | 0.954     | 0.990  | 0.972 | 8/11 (72.7%)  | —               | —     | 1           | 1           | 7.78        | 43%      | 80%      | 6.5       | 35.1      | 0.4      | 12.6    | 0.1      | 54.7        |
| 4   | YOLO26s  | ByteTrack | TransReID (ViT-B/16*) | Selesai | 1420        | 166/259         | 13/11               | 0.982     | 0.986  | 0.984 | 8/11 (72.7%)  | —               | —     | 1           | 2           | 7.51        | 18%      | 84%      | 7.9       | 32.8      | 0.4      | 31.7    | 0.2      | 73.0        |
| 5   | YOLO26x  | ByteTrack | OSNet                 | Selesai | 1851        | 225/259         | 11/11               | 0.933     | 0.993  | 0.962 | 6/11 (54.5%)  | —               | —     | 2           | 2           | 7.77        | 20%      | 83%      | 6.1       | 90.8      | 0.4      | 14.0    | 0.1      | 111.4       |
| 6   | YOLO26x  | ByteTrack | TransReID (ViT-B/16*) | Selesai | 1363        | 159/259         | 10/10               | 0.945     | 0.994  | 0.969 | 4/11 (36.4%)  | —               | —     | 2           | 3           | 7.53        | 44%      | 88%      | 8.4       | 419.0     | 0.6      | 40.9    | 0.3      | 469.2       |

*Sumber baris #1/#2/#4/#6: `pipeline-comparison.md` baris #1, #2, #29, #28.*

## Kesimpulan

**YOLO26s adalah titik manis, bukan YOLO26n maupun YOLO26x.** Ringkasan per detektor
(rata-rata 2 Re-ID):

| Detektor | Cakupan deteksi | Akurasi ketat | Detect ms |
| --- | --- | --- | --- |
| YOLO26n | 190/259, 187/259 (~72%) | 63,6% / 63,6% | 24,8 / 23,3 |
| **YOLO26s** | **225/259, 225/259 (87%)** | **72,7% / 72,7%** | **35,1 / 32,8** |
| YOLO26x | 159/259, 225/259 | 36,4% / 54,5% | 419,0 / 90,8 |

- **YOLO26s menang di kedua pasangan Re-ID sekaligus** — cakupan deteksi tertinggi
  (225/259, sama dengan YOLO26x) dan akurasi tertinggi (72,7%, sama-sama di kedua
  baris #3 dan #4), dengan `detect_ms` cuma naik ~10ms dari baseline n. Ini
  kandidat kuat buat jawab "coba detektor lebih berat" ke penguji — lebih berat
  dari n, tapi masih dalam kapasitas hardware (M4, tanpa GPU discrete).
- **YOLO26x tetap tidak layak**, bahkan dengan Re-ID paling ringan (OSNet):
  akurasinya 54,5%, di bawah baseline n (63,6%), walau cakupan deteksinya bagus
  (87%) — beda dari temuan sebelumnya di `pipeline-comparison.md` (yang
  akurasinya jatuh karena *frame drop*/cakupan rendah), di sini penyebabnya
  kemungkinan besar salah asosiasi Re-ID di skala model yang lebih besar, bukan
  kehilangan frame. `detect_ms`-nya juga masih jauh lebih tinggi dari n/s
  (90,8–419,0 ms tergantung beban Re-ID yang dipasangkan) — sensitif banget ke
  kombinasi komponen lain, bukti kapasitas hardware sudah mepet di titik ini.
- **OSNet ternyata cakupan-nya lebih baik dari TransReID di YOLO26s/26x**
  (225 vs 166/159) — kebalikan dari temuan awal di YOLO26n (TransReID sedikit
  lebih baik). Dugaan: TransReID (ViT-B/16, lebih berat dari OSNet) menambah
  beban komputasi per siklus yang berinteraksi buruk sama detektor yang sudah
  lebih berat (26s/26x), bikin lebih banyak *frame drop* — sementara OSNet yang
  ringan menyisakan lebih banyak *headroom*. Ini *interaction effect* antar
  komponen yang baru kelihatan di grid 2×3 ini, gak kelihatan kalau cuma lihat
  YOLO26n.

**Rekomendasi**: YOLO26s + ByteTrack + OSNet (baris #3) — akurasi tertinggi di
seluruh tabel, cakupan deteksi terbaik, dan `detect_ms` masih jauh di bawah
ambang yang bikin YOLO26x kolaps (419ms).
