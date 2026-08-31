# Perbandingan Alternatif Pipeline (Deteksi × Tracking × Re-ID)

Catatan hasil uji ganti komponen *pipeline* (model *pretrained*, tanpa
training/tuning ulang). Baseline saat ini: **YOLO26n + ByteTrack + OSNet**.

> **Versi ini mengecualikan 3 klip** (`clip_c11_20260622_093444.avi`,
> `clip_c8_20260622_093541.avi`, `clip_c9_20260622_093510.avi`) dari
> `output.csv` sebelum evaluasi — GT jadi 11 orang (dari 14). Ketiga klip ini
> satu-satunya yang berisi 3 orang GT (person 8, 9, & 10) yang beneran ketuker
> sistem di kamera & waktu yang sama (bukan artefak lintas-hari) — hasil di
> tabel ini nunjukkin performa "kasus bersih" tanpa skenario itu, buat
> dibandingkan sama versi lengkap (`../pipeline-comparison.md`).

## Metodologi

27 kombinasi penuh (3 detektor × 3 tracker × 3 Re-ID) didaftarkan di bawah
buat referensi, tapi **pengujian diprioritaskan ke 7 kombinasi** yang ganti
cuma satu komponen dari baseline sekaligus (variabel lain dikontrol tetap).
Full 27 kombinasi cuma dikerjakan kalau ada temuan yang menunjukkan
*interaction effect* antar komponen (baru kelihatan setelah 7 run prioritas
selesai).

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
- **Cakupan deteksi** — proporsi box *ground truth* yang berhasil dicocokkan
  ke prediksi (`Box GT tercocokkan ke prediksi` di stdout `evaluate.py`,
  IoU >= 0.3), dari total 259 box GT (11 orang, setelah 3 klip 0622
  dikecualikan). Beda dari jumlah frame mentah yang diproses sistem
  (`predictions.csv` bisa punya ribuan baris karena `BatchProcessor` proses
  tiap frame video, sedangkan GT cuma disampling manual 1 frame/detik lewat
  `label_gt.py`) — cakupan deteksi ini yang apples-to-apples dibanding
  jumlah baris `output.csv`, sinyal kualitas deteksi tanpa perlu skrip
  evaluasi deteksi terpisah.
- **Identitas terbentuk** — jumlah identitas sistem (`person_pred` unik)
  yang dihasilkan, dibanding jumlah orang GT sebenarnya (`X/11`). Sinyal
  cepat soal over-/under-segmentation: jauh di atas 11 berarti banyak orang
  yang sama dipecah jadi identitas berbeda (*over-segmentation*), jauh di
  bawah 11 berarti banyak orang beda digabung jadi satu identitas
  (*under-segmentation* / *false merge* masif).
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
- **FPS efektif** — throughput sistem untuk seluruh kamera berjalan paralel.
- **CPU/RAM peak** — puncak penggunaan selama run, lebih relevan dibanding
  rata-rata untuk menilai risiko *swap*/*throttling* pada penggunaan jangka
  panjang.

**Rujukan:** Bagga, A. & Baldwin, B. (1998). *Entity-Based Cross-Document
Coreferencing Using the Vector Space Model.* — dasar akademis metrik
precision/recall gaya B-cubed yang diadaptasi di sini.

## Hasil Uji

| #   | Detektor   | Tracker   | Re-ID                 | Status  | Cakupan deteksi | Identitas terbentuk | Precision | Recall | F1    | Akurasi      | False merge | False split | FPS efektif | CPU peak | RAM peak |
| --- | ---------- | --------- | --------------------- | ------- | --------------- | ------------------- | --------- | ------ | ----- | ------------ | ----------- | ----------- | ----------- | -------- | -------- |
| 1   | YOLO26n    | ByteTrack | OSNet                 | Selesai | 191/259         | 12/11               | 0.863     | 0.943  | 0.902 | 6/11 (54.5%) | 1           | 3           | 7.96        | 74%      | 66%      |
| 2   | YOLO26n    | ByteTrack | TransReID (ViT-B/16*) | Selesai | 187/259         | 17/11               | 0.997     | 0.906  | 0.949 | 6/11 (54.5%) | 1           | 5           | 7.63        | 14%      | 77%      |
| 3   | YOLO26n    | ByteTrack | BoT (ResNet50)        | Selesai | 191/259         | 1/11                | 0.109     | 1.000  | 0.197 | 0/11 (0.0%)  | 1           | 0           | 7.95        | 50%      | 84%      |
| 4   | YOLO26n    | BoT-SORT  | OSNet                 | Selesai | 192/259         | 15/11               | 0.934     | 0.856  | 0.893 | 4/11 (36.4%) | 4           | 6           | 7.95        | 12%      | 61%      |
| 5   | YOLO26n    | BoT-SORT  | TransReID (ViT-B/16*) | Selesai | 187/259         | 16/11               | 0.945     | 0.853  | 0.896 | 4/11 (36.4%) | 4           | 6           | 7.61        | 21%      | 80%      |
| 6   | YOLO26n    | BoT-SORT  | BoT (ResNet50)        | Selesai | 192/259         | 1/11                | 0.108     | 1.000  | 0.195 | 0/11 (0.0%)  | 1           | 0           | 7.95        | 23%      | 84%      |
| 7   | YOLO26n    | OC-SORT   | OSNet                 | Selesai | 182/259         | 16/11               | 0.977     | 0.806  | 0.883 | 5/11 (45.5%) | 2           | 5           | 7.95        | 12%      | 66%      |
| 8   | YOLO26n    | OC-SORT   | TransReID (ViT-B/16*) | Selesai | 177/259         | 17/11               | 0.940     | 0.911  | 0.925 | 4/11 (36.4%) | 2           | 6           | 7.61        | 42%      | 80%      |
| 9   | YOLO26n    | OC-SORT   | BoT (ResNet50)        | Selesai | 182/259         | 1/11                | 0.112     | 1.000  | 0.202 | 0/11 (0.0%)  | 1           | 0           | 7.95        | 33%      | 84%      |
| 10  | YOLO11n    | ByteTrack | OSNet                 | Selesai | 194/259         | 13/11               | 0.614     | 0.622  | 0.618 | 2/11 (18.2%) | 3           | 8           | 7.95        | 14%      | 61%      |
| 11  | YOLO11n    | ByteTrack | TransReID (ViT-B/16*) | Selesai | 189/259         | 14/11               | 0.769     | 0.757  | 0.763 | 5/11 (45.5%) | 2           | 6           | 7.60        | 60%      | 85%      |
| 12  | YOLO11n    | ByteTrack | BoT (ResNet50)        | Selesai | 194/259         | 1/11                | 0.116     | 1.000  | 0.209 | 0/11 (0.0%)  | 1           | 0           | 7.95        | 46%      | 84%      |
| 13  | YOLO11n    | BoT-SORT  | OSNet                 | Selesai | 195/259         | 12/11               | 0.671     | 0.632  | 0.651 | 3/11 (27.3%) | 4           | 8           | 7.95        | 12%      | 61%      |
| 14  | YOLO11n    | BoT-SORT  | TransReID (ViT-B/16*) | Selesai | 192/259         | 14/11               | 0.758     | 0.734  | 0.746 | 3/11 (27.3%) | 4           | 7           | 7.52        | 19%      | 77%      |
| 15  | YOLO11n    | BoT-SORT  | BoT (ResNet50)        | Selesai | 194/259         | 1/11                | 0.117     | 1.000  | 0.209 | 0/11 (0.0%)  | 1           | 0           | 7.95        | 46%      | 85%      |
| 16  | YOLO11n    | OC-SORT   | OSNet                 | Selesai | 186/259         | 13/11               | 0.920     | 0.838  | 0.877 | 3/11 (27.3%) | 3           | 6           | 7.75        | 17%      | 59%      |
| 17  | YOLO11n    | OC-SORT   | TransReID (ViT-B/16*) | Selesai | 183/259         | 16/11               | 0.946     | 0.939  | 0.943 | 5/11 (45.5%) | 3           | 5           | 7.60        | 30%      | 78%      |
| 18  | YOLO11n    | OC-SORT   | BoT (ResNet50)        | Selesai | 186/259         | 1/11                | 0.119     | 1.000  | 0.213 | 0/11 (0.0%)  | 1           | 0           | 7.81        | 14%      | 59%      |
| 19  | RTDETRv2-s | ByteTrack | OSNet                 | Selesai | 228/259         | 14/11               | 0.933     | 0.845  | 0.886 | 4/11 (36.4%) | 2           | 4           | 7.82        | 10%      | 70%      |
| 20  | RTDETRv2-s | ByteTrack | TransReID (ViT-B/16*) | Selesai | 160/259         | 11/10               | 0.702     | 0.933  | 0.801 | 3/11 (27.3%) | 3           | 4           | 7.59        | 56%      | 88%      |
| 21  | RTDETRv2-s | ByteTrack | BoT (ResNet50)        | Selesai | 228/259         | 1/11                | 0.114     | 1.000  | 0.204 | 0/11 (0.0%)  | 1           | 0           | 7.80        | 12%      | 72%      |
| 22  | RTDETRv2-s | BoT-SORT  | OSNet                 | Selesai | 228/259         | 14/11               | 0.827     | 0.841  | 0.834 | 2/11 (18.2%) | 3           | 4           | 7.80        | 31%      | 70%      |
| 23  | RTDETRv2-s | BoT-SORT  | TransReID (ViT-B/16*) | Selesai | 162/259         | 13/11               | 0.695     | 0.855  | 0.766 | 4/11 (36.4%) | 3           | 5           | 7.68        | 38%      | 88%      |
| 24  | RTDETRv2-s | BoT-SORT  | BoT (ResNet50)        | Selesai | 228/259         | 1/11                | 0.114     | 1.000  | 0.204 | 0/11 (0.0%)  | 1           | 0           | 7.79        | 10%      | 76%      |
| 25  | RTDETRv2-s | OC-SORT   | OSNet                 | Selesai | 223/259         | 16/11               | 0.937     | 0.808  | 0.868 | 4/11 (36.4%) | 2           | 5           | 7.79        | 11%      | 73%      |
| 26  | RTDETRv2-s | OC-SORT   | TransReID (ViT-B/16*) | Selesai | 159/259         | 13/11               | 0.947     | 0.933  | 0.940 | 8/11 (72.7%) | 1           | 3           | 7.60        | 55%      | 88%      |
| 27  | RTDETRv2-s | OC-SORT   | BoT (ResNet50)        | Selesai | 223/259         | 1/11                | 0.115     | 1.000  | 0.206 | 0/11 (0.0%)  | 1           | 0           | 7.78        | 11%      | 76%      |

## Kesimpulan

**Komponen terpilih: YOLO26n + ByteTrack + TransReID (ViT-B/16*)** — kombinasi
#2, menggantikan OSNet sebagai model Re-ID pada baseline (#1).

Perbandingan langsung #1 vs #2 — detektor & tracker sama persis, cuma model
Re-ID yang beda, jadi efek Re-ID-nya terisolasi bersih:

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
   ratusan pasangan kemunculan (per pasangan, bukan per identitas), lebih
   stabil secara statistik dibanding Akurasi yang cuma dari 11 orang GT.
2. **Precision nyaris sempurna (0.997)** — TransReID nyaris tidak pernah
   salah menggabungkan dua orang berbeda jadi satu identitas, kesalahan yang
   paling mahal secara operasional karena tidak ada jalan koreksi otomatis
   (lihat `plan/06-decisions.md`, ADR-002).
3. **CPU peak jauh lebih rendah** (14% vs 74%) — krusial untuk stabilitas
   sistem yang berjalan 24/7 di hardware tanpa GPU *discrete*; OSNet berisiko
   *throttling*/*swap* kalau load kamera bertambah.

**Kandidat lain yang dipertimbangkan tapi tidak dipilih:**
- **#26 (RTDETR-l + OC-SORT + TransReID)** — Akurasi tertinggi (8/11, 72.7%)
  dan identitas terbentuk paling dekat GT (13/11), tapi cakupan deteksinya
  jauh lebih rendah dari kombinasi RTDETR lain (159/259 vs ~223/259 untuk
  RTDETR+OSNet) — pola yang konsisten di semua kombinasi RTDETR+TransReID,
  diduga `BatchProcessor` (single-thread) men-drop frame saat detektor berat
  (RTDETR) digabung dengan Re-ID paling berat (TransReID). Akurasi tingginya
  kemungkinan sebagian ditolong oleh sampel yang lebih sedikit/lebih mudah,
  bukan murni performa lebih baik — perlu investigasi lebih lanjut sebelum
  dipakai sebagai basis keputusan.

**Trade-off yang diterima:** TransReID menghasilkan lebih banyak identitas
(17/11) karena Recall-nya lebih rendah — orang yang sama kadang dianggap
identitas baru kalau sudut kamera/pencahayaan berubah signifikan
(*over-segmentation*, false split naik dari 3 ke 5). Ini dianggap trade-off
yang lebih bisa diterima dibanding risiko *false merge* (privasi/keamanan:
menyamakan dua orang berbeda), dan konsisten dengan prioritas F1/Precision
sebagai metrik utama evaluasi.
