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

| #   | Detektor   | Tracker   | Re-ID                 | Status  | Identitas terbentuk | Precision | Recall | F1    | Akurasi | False merge | False split | FPS efektif | CPU peak | RAM peak |
| --- | ---------- | --------- | ---------------------- | ------- | -------------------- | --------- | ------ | ----- | ------- | ----------- | ----------- | ----------- | -------- | -------- |
| 1   | YOLO26n    | ByteTrack | OSNet                 | Selesai | 12/11 | 0.863 | 0.943 | 0.902 | 6/11 (54.5%) | 1 | 3 | 7.96 | 74% | 66% |
| 2   | YOLO26n    | ByteTrack | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 3   | YOLO26n    | ByteTrack | BoT (ResNet50)        | Selesai | 1/11 | 0.109 | 1.000 | 0.197 | 0/11 (0.0%) | 1 | 0 | 7.95 | 50% | 84% |
| 4   | YOLO26n    | BoT-SORT  | OSNet                 | Selesai | 15/11 | 0.934 | 0.856 | 0.893 | 4/11 (36.4%) | 4 | 6 | 7.95 | 12% | 61% |
| 5   | YOLO26n    | BoT-SORT  | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 6   | YOLO26n    | BoT-SORT  | BoT (ResNet50)        | Selesai | 1/11 | 0.108 | 1.000 | 0.195 | 0/11 (0.0%) | 1 | 0 | 7.95 | 23% | 84% |
| 7   | YOLO26n    | OC-SORT   | OSNet                 | Selesai | 16/11 | 0.977 | 0.806 | 0.883 | 5/11 (45.5%) | 2 | 5 | 7.95 | 12% | 66% |
| 8   | YOLO26n    | OC-SORT   | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 9   | YOLO26n    | OC-SORT   | BoT (ResNet50)        | Selesai | 1/11 | 0.112 | 1.000 | 0.202 | 0/11 (0.0%) | 1 | 0 | 7.95 | 33% | 84% |
| 10  | YOLO11n    | ByteTrack | OSNet                 | Selesai | 13/11 | 0.614 | 0.622 | 0.618 | 2/11 (18.2%) | 3 | 8 | 7.95 | 14% | 61% |
| 11  | YOLO11n    | ByteTrack | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 12  | YOLO11n    | ByteTrack | BoT (ResNet50)        | Selesai | 1/11 | 0.116 | 1.000 | 0.209 | 0/11 (0.0%) | 1 | 0 | 7.95 | 46% | 84% |
| 13  | YOLO11n    | BoT-SORT  | OSNet                 | Selesai | 12/11 | 0.671 | 0.632 | 0.651 | 3/11 (27.3%) | 4 | 8 | 7.95 | 12% | 61% |
| 14  | YOLO11n    | BoT-SORT  | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 15  | YOLO11n    | BoT-SORT  | BoT (ResNet50)        | Selesai | 1/11 | 0.117 | 1.000 | 0.209 | 0/11 (0.0%) | 1 | 0 | 7.95 | 46% | 85% |
| 16  | YOLO11n    | OC-SORT   | OSNet                 | Selesai | 13/11 | 0.920 | 0.838 | 0.877 | 3/11 (27.3%) | 3 | 6 | 7.75 | 17% | 59% |
| 17  | YOLO11n    | OC-SORT   | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 18  | YOLO11n    | OC-SORT   | BoT (ResNet50)        | Selesai | 1/11 | 0.119 | 1.000 | 0.213 | 0/11 (0.0%) | 1 | 0 | 7.81 | 14% | 59% |
| 19  | RTDETRv2-s | ByteTrack | OSNet                 | Selesai | 14/11 | 0.933 | 0.845 | 0.886 | 4/11 (36.4%) | 2 | 4 | 7.82 | 10% | 70% |
| 20  | RTDETRv2-s | ByteTrack | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 21  | RTDETRv2-s | ByteTrack | BoT (ResNet50)        | Selesai | 1/11 | 0.114 | 1.000 | 0.204 | 0/11 (0.0%) | 1 | 0 | 7.80 | 12% | 72% |
| 22  | RTDETRv2-s | BoT-SORT  | OSNet                 | Selesai | 14/11 | 0.827 | 0.841 | 0.834 | 2/11 (18.2%) | 3 | 4 | 7.80 | 31% | 70% |
| 23  | RTDETRv2-s | BoT-SORT  | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 24  | RTDETRv2-s | BoT-SORT  | BoT (ResNet50)        | Selesai | 1/11 | 0.114 | 1.000 | 0.204 | 0/11 (0.0%) | 1 | 0 | 7.79 | 10% | 76% |
| 25  | RTDETRv2-s | OC-SORT   | OSNet                 | Selesai | 16/11 | 0.937 | 0.808 | 0.868 | 4/11 (36.4%) | 2 | 5 | 7.79 | 11% | 73% |
| 26  | RTDETRv2-s | OC-SORT   | TransReID (ViT-B/16*) | Belum   |  |       |       |       |  |   |   |   |   |   |
| 27  | RTDETRv2-s | OC-SORT   | BoT (ResNet50)        | Selesai | 1/11 | 0.115 | 1.000 | 0.206 | 0/11 (0.0%) | 1 | 0 | 7.78 | 11% | 76% |

## Kesimpulan

_(diisi setelah 7 run prioritas selesai — komponen mana yang dipertahankan/diganti dari baseline, dan alasannya.)_
