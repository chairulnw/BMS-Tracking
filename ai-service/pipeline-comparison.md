# Perbandingan Alternatif Pipeline (Deteksi × Tracking × Re-ID)

Catatan hasil uji ganti komponen *pipeline* (model *pretrained*, tanpa
training/tuning ulang). Baseline saat ini: **YOLO26n + ByteTrack + OSNet**.

## Metodologi

27 kombinasi penuh (3 detektor × 3 tracker × 3 Re-ID) didaftarkan di bawah
buat referensi, tapi **pengujian diprioritaskan ke 7 kombinasi** yang ganti
cuma satu komponen dari baseline sekaligus (variabel lain dikontrol tetap) —
lihat kolom **Prioritas**. Full 27 kombinasi cuma dikerjakan kalau ada temuan
yang menunjukkan *interaction effect* antar komponen (baru kelihatan setelah
7 run prioritas selesai).

Tiap kombinasi diuji lewat `evaluate.py` (dataset `sample copy/output.csv`)
dan dicatat di lingkungan yang sama dengan Bab VI laporan TA (MacBook Air
M4, RAM 8GB, tanpa GPU *discrete*, inferensi via MPS). Metrik yang dicatat:

- **Precision / Recall / F1 (pairwise, B-cubed style)** — metrik utama.
  Dihitung per **pasangan kemunculan** (bukan per identitas): FP = pasangan
  yang diprediksi identitas sama tapi sebenarnya beda orang (*false merge*
  di level pasangan), FN = pasangan yang sebenarnya orang sama tapi
  diprediksi beda (*false split* di level pasangan). Dipilih dibanding
  MOTA/IDF1/HOTA (butuh GT padat per-frame + video kontinu, sedangkan data
  ini sample jarang lintas banyak klip terpisah) dan dibanding TTR/FTR
  (cuma didukung 1 paper spesifik) — lihat rujukan Bagga & Baldwin (1998)
  di bawah. Berbeda dari hitungan *false merge*/*split* per-identitas,
  metrik ini otomatis kasih penalti sebanding jumlah orang yang tercampur
  kalau 1 identitas salah gabung banyak orang GT sekaligus.
- **False merge / False split (per identitas)** — tetap dicatat sebagai
  pelengkap deskriptif: berapa identitas sistem yang bermasalah, terpisah
  dari precision/recall yang ngukur di level pasangan. Sesuai catatan
  tuning sebelumnya (`plan/06-decisions.md`, ADR-002), *false merge* lebih
  mahal daripada *false split* karena tidak ada jalan koreksi manual.
- **Akurasi** — definisi sama persis dengan angka 80% di Bab VI laporan TA:
  proporsi orang GT yang dapet **tepat 1** identitas prediksi, DAN identitas
  itu **cuma dipakai dia sendiri** (nggak ke-*merge*/*split* sama sekali).
  Beda dari Precision/Recall/F1 (per pasangan) — ini per **orang**, jadi bisa
  beda urutan rangking dari F1 (kombinasi F1 tinggi belum tentu akurasi
  tertinggi, tergantung sebaran kesalahannya).
- **Cakupan deteksi** — proporsi box *ground truth* yang berhasil dicocokkan
  ke prediksi (dicetak `evaluate.py`), sinyal kualitas deteksi tanpa perlu
  skrip evaluasi deteksi terpisah.
- **FPS efektif** — throughput sistem untuk seluruh kamera berjalan paralel.
- **CPU/RAM peak** — puncak penggunaan selama run, lebih relevan dibanding
  rata-rata untuk menilai risiko *swap*/*throttling* pada penggunaan jangka
  panjang.

**Rujukan:** Bagga, A. & Baldwin, B. (1998). *Entity-Based Cross-Document
Coreferencing Using the Vector Space Model.* — dasar akademis metrik
precision/recall gaya B-cubed yang diadaptasi di sini.

## Hasil Uji

| #   | Detektor   | Tracker   | Re-ID                 | Prioritas         | Identitas terbentuk | Status  | Precision | Recall | F1    | Akurasi | False merge | False split | Cakupan deteksi | FPS efektif | CPU peak | RAM peak | Catatan |
| --- | ---------- | --------- | --------------------- | ----------------- | --------------- | ------- | --------- | ------ | ----- | ------- | ----------- | ----------- | --------------- | ----------- | -------- | -------- | ------- |
| 1   | YOLO26n    | ByteTrack | OSNet                 | Ya (baseline)     | 15/14 | Selesai | 0.677 | 0.919 | 0.780 | 7/14 (50.0%) | 1 | 4 | 245/332 | 7.96        | 74%      | 66%      |         |
| 2   | YOLO26n    | ByteTrack | TransReID (ViT-B/16*) | Ya                | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 3   | YOLO26n    | ByteTrack | BoT (ResNet50)        | Ya                | 1/14 | Selesai | 0.087 | 1.000 | 0.160 | 0/14 (0.0%) | 1 | 0 | 245/332 | 7.95        | 50%      | 84%      |         |
| 4   | YOLO26n    | BoT-SORT  | OSNet                 | Ya                | 23/14 | Selesai | 0.892 | 0.723 | 0.799 | 5/14 (35.7%) | 4 | 8 | 246/332 | 7.95        | 12%      | 61%      |         |
| 5   | YOLO26n    | BoT-SORT  | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 6   | YOLO26n    | BoT-SORT  | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.086 | 1.000 | 0.159 | 0/14 (0.0%) | 1 | 0 | 246/332 | 7.95        | 23%      | 84%      |         |
| 7   | YOLO26n    | OC-SORT   | OSNet                 | Ya                | 22/14 | Selesai | 0.953 | 0.751 | 0.840 | 4/14 (28.6%) | 4 | 7 | 230/332 | 7.95        | 12%      | 66%      |         |
| 8   | YOLO26n    | OC-SORT   | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 9   | YOLO26n    | OC-SORT   | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.089 | 1.000 | 0.163 | 0/14 (0.0%) | 1 | 0 | 230/332 | 7.95        | 33%      | 84%      |         |
| 10  | YOLO11n    | ByteTrack | OSNet                 | Ya                | 14/14 | Selesai | 0.558 | 0.675 | 0.611 | 2/14 (14.3%) | 4 | 9 | 245/332 | 7.95        | 14%      | 61%      |         |
| 11  | YOLO11n    | ByteTrack | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 12  | YOLO11n    | ByteTrack | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.091 | 1.000 | 0.167 | 0/14 (0.0%) | 1 | 0 | 245/332 | 7.95        | 46%      | 84%      |         |
| 13  | YOLO11n    | BoT-SORT  | OSNet                 | Tidak (full grid) | 13/14 | Selesai | 0.612 | 0.712 | 0.658 | 4/14 (28.6%) | 4 | 8 | 249/332 | 7.95        | 12%      | 61%      |         |
| 14  | YOLO11n    | BoT-SORT  | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 15  | YOLO11n    | BoT-SORT  | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.091 | 1.000 | 0.167 | 0/14 (0.0%) | 1 | 0 | 248/332 | 7.95        | 46%      | 85%      |         |
| 16  | YOLO11n    | OC-SORT   | OSNet                 | Tidak (full grid) | 16/14 | Selesai | 0.759 | 0.830 | 0.793 | 3/14 (21.4%) | 4 | 9 | 238/332 | 7.75        | 17%      | 59%      |         |
| 17  | YOLO11n    | OC-SORT   | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 18  | YOLO11n    | OC-SORT   | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.093 | 1.000 | 0.170 | 0/14 (0.0%) | 1 | 0 | 238/332 | 7.81        | 14%      | 59%      |         |
| 19  | RTDETRv2-s | ByteTrack | OSNet                 | Ya                | 15/14 | Selesai | 0.735 | 0.880 | 0.801 | 5/14 (35.7%) | 2 | 4 | 292/332 | 7.82        | 10%      | 70%      |         |
| 20  | RTDETRv2-s | ByteTrack | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 21  | RTDETRv2-s | ByteTrack | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.090 | 1.000 | 0.165 | 0/14 (0.0%) | 1 | 0 | 292/332 | 7.80        | 12%      | 72%      |         |
| 22  | RTDETRv2-s | BoT-SORT  | OSNet                 | Tidak (full grid) | 16/14 | Selesai | 0.685 | 0.832 | 0.752 | 2/14 (14.3%) | 3 | 5 | 293/332 | 7.80        | 31%      | 70%      |         |
| 23  | RTDETRv2-s | BoT-SORT  | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 24  | RTDETRv2-s | BoT-SORT  | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.090 | 1.000 | 0.165 | 0/14 (0.0%) | 1 | 0 | 293/332 | 7.79        | 10%      | 76%      |         |
| 25  | RTDETRv2-s | OC-SORT   | OSNet                 | Tidak (full grid) | 18/14 | Selesai | 0.748 | 0.797 | 0.772 | 3/14 (21.4%) | 4 | 7 | 285/332 | 7.79        | 11%      | 73%      |         |
| 26  | RTDETRv2-s | OC-SORT   | TransReID (ViT-B/16*) | Tidak (full grid) | | Belum   |           |        |       |  |             |             |                 |             |          |          |         |
| 27  | RTDETRv2-s | OC-SORT   | BoT (ResNet50)        | Tidak (full grid) | 1/14 | Selesai | 0.090 | 1.000 | 0.166 | 0/14 (0.0%) | 1 | 0 | 285/332 | 7.78        | 11%      | 76%      |         |

## Kesimpulan

**Komponen terpilih: YOLO26n + ByteTrack + TransReID (ViT-B/16*)** — kombinasi
#2, menggantikan OSNet sebagai model Re-ID pada baseline.

Baris TransReID pada tabel di atas masih "Belum" karena prediksinya cuma
dijalankan untuk 27 dari 31 klip (skip 3 klip yang sama dengan yang
dikecualikan di `predictions_runs copy/pipeline-comparison copy.md`) —
mengevaluasinya ke GT 14-orang di sini akan salah menghukum person 8/9/10
dengan skor 0% padahal memang belum pernah diproses. Keputusan pemilihan
komponen di bawah ini didasarkan pada hasil lengkap 27 kombinasi di tabel
copy tersebut, bukan tabel ini.

Perbandingan langsung #1 (baseline, OSNet) vs #2 (TransReID) — detektor &
tracker sama persis, cuma model Re-ID yang beda — di dataset copy (11 orang):

| Metrik | #1 OSNet | #2 TransReID |
| --- | --- | --- |
| Precision | 0.863 | **0.997** |
| Recall | **0.943** | 0.906 |
| F1 | 0.902 | **0.949** |
| Akurasi | 6/11 (54.5%) | 6/11 (54.5%) |
| Identitas terbentuk | 12/11 | 17/11 |
| CPU peak | **74%** | **14%** |

**Alasan pemilihan:**
1. **F1 lebih tinggi** (0.949 vs 0.902) — metrik utama karena dihitung dari
   ratusan pasangan kemunculan, lebih stabil secara statistik dibanding
   Akurasi yang cuma dari 11 orang GT.
2. **Precision nyaris sempurna (0.997)** — TransReID nyaris tidak pernah
   salah menggabungkan dua orang berbeda jadi satu identitas, kesalahan yang
   paling mahal secara operasional karena tidak ada jalan koreksi otomatis
   (lihat `plan/06-decisions.md`, ADR-002).
3. **CPU peak jauh lebih rendah** (14% vs 74%) — krusial untuk stabilitas
   sistem yang berjalan 24/7 di hardware tanpa GPU *discrete*; OSNet berisiko
   *throttling*/*swap* kalau load kamera bertambah.

**Trade-off yang diterima:** TransReID menghasilkan lebih banyak identitas
(17/11) karena Recall-nya lebih rendah — orang yang sama kadang dianggap
identitas baru kalau sudut kamera/pencahayaan berubah signifikan
(*over-segmentation*, false split naik dari 3 ke 5). Ini dianggap trade-off
yang lebih bisa diterima dibanding risiko *false merge* (privasi/keamanan:
menyamakan dua orang berbeda), dan konsisten dengan prioritas F1/Precision
sebagai metrik utama evaluasi.
