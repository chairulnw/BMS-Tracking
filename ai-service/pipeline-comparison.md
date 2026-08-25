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

Tiap kombinasi diuji lewat `evaluate.py` (dataset `sample/output.csv`) dan
dicatat di lingkungan yang sama dengan Bab VI laporan TA (MacBook Air M4,
RAM 8GB, tanpa GPU *discrete*, inferensi via MPS). Metrik yang dicatat:

- **False merge / False split** — dua angka terpisah dari `evaluate.py`,
  bukan digabung jadi satu persentase, karena keduanya bukan kesalahan yang
  setara: sesuai catatan tuning sebelumnya (`plan/06-decisions.md`, ADR-002),
  *false merge* (dua orang ketuker jadi satu identitas) lebih mahal daripada
  *false split* (satu orang kepecah jadi beberapa identitas) karena tidak ada
  jalan koreksi manual. Angka ini sudah mencakup seluruh *pipeline*, bukan
  cuma tahap Re-ID — `person_pred` yang dibandingkan adalah keluaran akhir
  setelah deteksi → tracking → Re-ID.
- **Cakupan deteksi** — proporsi box *ground truth* yang berhasil dicocokkan
  ke prediksi (dicetak `evaluate.py`), sinyal kualitas deteksi tanpa perlu
  skrip evaluasi deteksi terpisah.
- **FPS efektif** — throughput sistem untuk seluruh kamera berjalan paralel.
- **CPU/RAM peak** — puncak penggunaan selama run, lebih relevan dibanding
  rata-rata untuk menilai risiko *swap*/*throttling* pada penggunaan jangka
  panjang.

## Hasil Uji

| # | Detektor | Tracker | Re-ID | Prioritas | Status | False merge | False split | Cakupan deteksi | FPS efektif | CPU peak | RAM peak | Catatan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | YOLO26n | ByteTrack | OSNet | Ya (baseline) | Belum |  |  |  |  |  |  |  |
| 2 | YOLO26n | ByteTrack | TransReID (ViT-B/16*) | Ya | Belum |  |  |  |  |  |  |  |
| 3 | YOLO26n | ByteTrack | BoT (ResNet50) | Ya | Belum |  |  |  |  |  |  |  |
| 4 | YOLO26n | BoT-SORT | OSNet | Ya | Belum |  |  |  |  |  |  |  |
| 5 | YOLO26n | BoT-SORT | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 6 | YOLO26n | BoT-SORT | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 7 | YOLO26n | OC-SORT | OSNet | Ya | Belum |  |  |  |  |  |  |  |
| 8 | YOLO26n | OC-SORT | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 9 | YOLO26n | OC-SORT | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 10 | YOLO11n | ByteTrack | OSNet | Ya | Belum |  |  |  |  |  |  |  |
| 11 | YOLO11n | ByteTrack | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 12 | YOLO11n | ByteTrack | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 13 | YOLO11n | BoT-SORT | OSNet | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 14 | YOLO11n | BoT-SORT | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 15 | YOLO11n | BoT-SORT | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 16 | YOLO11n | OC-SORT | OSNet | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 17 | YOLO11n | OC-SORT | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 18 | YOLO11n | OC-SORT | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 19 | RTDETRv2-s | ByteTrack | OSNet | Ya | Belum |  |  |  |  |  |  |  |
| 20 | RTDETRv2-s | ByteTrack | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 21 | RTDETRv2-s | ByteTrack | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 22 | RTDETRv2-s | BoT-SORT | OSNet | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 23 | RTDETRv2-s | BoT-SORT | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 24 | RTDETRv2-s | BoT-SORT | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 25 | RTDETRv2-s | OC-SORT | OSNet | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 26 | RTDETRv2-s | OC-SORT | TransReID (ViT-B/16*) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |
| 27 | RTDETRv2-s | OC-SORT | BoT (ResNet50) | Tidak (full grid) | Belum |  |  |  |  |  |  |  |

## Kesimpulan

_(diisi setelah 7 run prioritas selesai — komponen mana yang dipertahankan/diganti dari baseline, dan alasannya.)_
