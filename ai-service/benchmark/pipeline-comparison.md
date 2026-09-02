# Perbandingan Alternatif Pipeline (Deteksi × Tracking × Re-ID)

## Metodologi

27 kombinasi penuh (3 detektor × 3 tracker × 3 Re-ID) didaftarkan di bawah
buat referensi, tapi **pengujian diprioritaskan ke 7 kombinasi** yang ganti
cuma satu komponen dari baseline sekaligus (variabel lain dikontrol tetap).
Full 27 kombinasi cuma dikerjakan kalau ada temuan yang menunjukkan
*interaction effect* antar komponen (baru kelihatan setelah 7 run prioritas
selesai).

Tiap kombinasi diuji lewat `evaluate.py` (dataset `sample/output.csv`)
dan dicatat di lingkungan yang sama dengan Bab VI laporan TA (MacBook Air
M4, RAM 8GB, tanpa GPU *discrete*, inferensi via MPS). Metrik yang dicatat:

- **Precision / Recall / F1 (pairwise, B-cubed style)** — metrik utama.
  Dihitung per **pasangan kemunculan** (bukan per identitas): 
	- FP = pasangan  yang diprediksi identitas sama tapi sebenarnya beda orang (*false merge* di level pasangan)
	- FN = pasangan yang sebenarnya orang sama tapi diprediksi beda (*false split* di level pasangan). 
	Dipilih dibanding  MOTA/IDF1/HOTA (butuh GT padat per-frame + video kontinu, sedangkan data ini sample jarang lintas banyak klip terpisah) dan dibanding TTR/FTR (cuma didukung 1 paper spesifik). Berbeda dari hitungan *false merge*/*split* per-identitas, metrik ini otomatis kasih penalti sebanding jumlah orang yang tercampurkalau 1 identitas salah gabung banyak orang GT sekaligus.
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
- **Akurasi (3 versi)** — semua per **orang GT** (dari 11), bukan per pasangan
  seperti F1 — jadi urutan rangking bisa beda dari F1.
  - **Akurasi ketat** — proporsi orang GT yang dapet **tepat 1** identitas
    prediksi **DAN** identitas itu **cuma dipakai dia sendiri** (nggak
    ke-*merge*/*split* sama sekali). All-or-nothing: 1 split/merge → orang itu
    dihitung 0 walau 99% frame-nya benar. Ini definisi "akurasi 80%" di Bab VI
    laporan.
    Rumus: `n_bersih / 11`, `n_bersih = |{g : |pred(g)|=1 ∧ |gt(pred(g))|=1}|`.
  - **Akurasi dominan** — orang GT dihitung benar kalau **>50%** kemunculannya
    (yang ke-match IoU) jatuh ke **satu** identitas prediksi, tanpa syarat
    eksklusivitas. Setara konsep *Mostly Tracked* (MOTChallenge) tapi ambang
    mayoritas, bukan 80%. Lebih longgar — toleran terhadap minority split.
    Rumus: `|{g : max_p count(g,p) / Σ_p count(g,p) > 0.5}| / 11`.
  - **IDF1** — metrik identitas standar MOT (Ristani et al. 2016). Matching
    bipartit global GT-id ↔ pred-id yang memaksimalkan co-occurrence (IDTP),
    lalu `IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)` dengan IDFP/IDFN dihitung di
    frame yang ada di GT **dan** prediksi. Tunggal (bukan `X/11`), sebanding
    lintas paper.
- **FPS efektif** — throughput sistem untuk seluruh kamera berjalan paralel.
- **Decode/Detect/Track/ReID/Match/Total AI ms** — rata-rata durasi tiap
  tahap pipeline per frame (`time.perf_counter()`, lihat `latency_logger.py`),
  dari `latency.csv` yang di-generate otomatis tiap run. `Detect ms` sama
  buat semua kamera dalam satu siklus (YOLO/RTDETR diproses sekaligus multi-
  kamera, 1 GPU call — bukan per-kamera murni). `Total AI ms` = jumlah 5
  kolom lain (penjumlahan logis, bukan satu span wall-clock — decode terjadi
  di thread capture yang konkuren sama batch loop). Kombinasi yang belum
  di-rerun setelah instrumentasi ini ada masih kosong di kolom ini.
- **CPU/RAM peak** — puncak penggunaan selama run, lebih relevan dibanding
  rata-rata untuk menilai risiko *swap*/*throttling* pada penggunaan jangka
  panjang.

**Rujukan:** Bagga, A. & Baldwin, B. (1998). *Entity-Based Cross-Document
Coreferencing Using the Vector Space Model.* — dasar akademis metrik
precision/recall gaya B-cubed yang diadaptasi di sini.

## Hasil Uji

| #   | Detektor   | Tracker   | Re-ID                 | Status  | Cakupan deteksi | Identitas terbentuk | Precision | Recall | F1    | Akurasi ketat | Akurasi dominan | IDF1  | False merge | False split | FPS efektif | CPU peak | RAM peak | Decode ms | Detect ms | Track ms | ReID ms | Match ms | Total AI ms |
| --- | ---------- | --------- | --------------------- | ------- | --------------- | ------------------- | --------- | ------ | ----- | ------------- | --------------- | ----- | ----------- | ----------- | ----------- | -------- | -------- | --------- | --------- | -------- | ------- | -------- | ----------- |
| 1   | YOLO26n    | ByteTrack | OSNet                 | Selesai | 190/259         | 12/11               | 0.944     | 0.961  | 0.953 | 7/11 (63.6%)  | 11/11 (100.0%)  | 0.899 | 1           | 2           | 7.56        | 31%      | 81%      | 7.1       | 24.8      | 0.4      | 14.2    | 0.0      | 46.6        |
| 2   | YOLO26n    | ByteTrack | TransReID (ViT-B/16*) | Selesai | 187/259         | 16/11               | 0.997     | 0.931  | 0.962 | 7/11 (63.6%)  | 10/11 (90.9%)   | 0.908 | 1           | 4           | 7.57        | 36%      | 81%      | 8.0       | 23.3      | 0.4      | 29.0    | 0.0      | 60.8        |
| 3   | YOLO26n    | ByteTrack | BoT (ResNet50)        | Selesai | 190/259         | 20/11               | 0.711     | 0.607  | 0.655 | 1/11 (9.1%)   | 9/11 (81.8%)    | 0.706 | 4           | 9           | 7.68        | 34%      | 82%      | 8.3       | 26.2      | 0.5      | 14.7    | 0.0      | 49.7        |
| 4   | YOLO26n    | BoT-SORT  | OSNet                 | Selesai | 191/259         | 13/11               | 0.927     | 0.879  | 0.903 | 5/11 (45.5%)  | 11/11 (100.0%)  | 0.848 | 4           | 5           | 7.66        | 53%      | 83%      | 7.5       | 25.5      | 10.6     | 14.9    | 0.0      | 58.6        |
| 5   | YOLO26n    | BoT-SORT  | TransReID (ViT-B/16*) | Selesai | 187/259         | 16/11               | 0.945     | 0.853  | 0.896 | 4/11 (36.4%)  | 10/11 (90.9%)   | 0.823 | 4           | 6           | 7.50        | 68%      | 87%      | 10.3      | 28.1      | 13.6     | 33.6    | 0.1      | 85.7        |
| 6   | YOLO26n    | BoT-SORT  | BoT (ResNet50)        | Selesai | 187/259         | 18/11               | 0.723     | 0.599  | 0.655 | 1/11 (9.1%)   | 10/11 (90.9%)   | 0.712 | 6           | 10          | 7.62        | 79%      | 86%      | 9.9       | 30.9      | 13.1     | 16.6    | 0.1      | 70.6        |
| 7   | YOLO26n    | OC-SORT   | OSNet                 | Selesai | 181/259         | 15/11               | 0.934     | 0.807  | 0.866 | 4/11 (36.4%)  | 11/11 (100.0%)  | 0.834 | 3           | 5           | 7.33        | 82%      | 86%      | 9.5       | 29.7      | 2.3      | 18.4    | 0.0      | 60.0        |
| 8   | YOLO26n    | OC-SORT   | TransReID (ViT-B/16*) | Selesai | 174/259         | 17/11               | 0.959     | 0.916  | 0.937 | 4/11 (36.4%)  | 10/11 (90.9%)   | 0.890 | 2           | 6           | 7.39        | 51%      | 85%      | 10.3      | 29.6      | 1.4      | 32.9    | 0.0      | 74.2        |
| 9   | YOLO26n    | OC-SORT   | BoT (ResNet50)        | Selesai | 182/259         | 18/11               | 0.647     | 0.604  | 0.625 | 1/11 (9.1%)   | 10/11 (90.9%)   | 0.705 | 4           | 10          | 7.56        | 68%      | 86%      | 9.2       | 27.9      | 1.2      | 15.2    | 0.0      | 53.6        |
| 10  | YOLO11n    | ByteTrack | OSNet                 | Selesai | 193/259         | 12/11               | 0.475     | 0.730  | 0.575 | 1/11 (9.1%)   | 9/11 (81.8%)    | 0.706 | 4           | 8           | 7.48        | 74%      | 86%      | 8.8       | 30.7      | 0.5      | 17.1    | 0.1      | 57.3        |
| 11  | YOLO11n    | ByteTrack | TransReID (ViT-B/16*) | Selesai | 189/259         | 14/11               | 0.763     | 0.745  | 0.754 | 3/11 (27.3%)  | 10/11 (90.9%)   | 0.809 | 3           | 7           | 7.45        | 55%      | 86%      | 10.2      | 28.5      | 0.6      | 31.6    | 0.1      | 71.0        |
| 12  | YOLO11n    | ByteTrack | BoT (ResNet50)        | Selesai | 194/259         | 19/11               | 0.591     | 0.452  | 0.512 | 1/11 (9.1%)   | 6/11 (54.5%)    | 0.565 | 4           | 10          | 7.61        | 48%      | 86%      | 9.8       | 30.0      | 0.6      | 15.3    | 0.0      | 55.7        |
| 13  | YOLO11n    | BoT-SORT  | OSNet                 | Selesai | 195/259         | 13/11               | 0.733     | 0.632  | 0.679 | 3/11 (27.3%)  | 10/11 (90.9%)   | 0.739 | 4           | 8           | 7.61        | 57%      | 85%      | 7.7       | 27.6      | 10.7     | 13.9    | 0.0      | 59.9        |
| 14  | YOLO11n    | BoT-SORT  | TransReID (ViT-B/16*) | Selesai | 194/259         | 14/11               | 0.753     | 0.732  | 0.742 | 3/11 (27.3%)  | 10/11 (90.9%)   | 0.779 | 4           | 7           | 7.81        | 33%      | 83%      | 6.9       | 21.8      | 9.1      | 23.6    | 0.0      | 61.4        |
| 15  | YOLO11n    | BoT-SORT  | BoT (ResNet50)        | Selesai | 194/259         | 15/11               | 0.380     | 0.487  | 0.427 | 1/11 (9.1%)   | 8/11 (72.7%)    | 0.595 | 4           | 10          | 7.84        | 23%      | 81%      | 6.8       | 26.0      | 9.1      | 13.7    | 0.0      | 55.7        |
| 16  | YOLO11n    | OC-SORT   | OSNet                 | Selesai | 186/259         | 15/11               | 0.934     | 0.814  | 0.870 | 3/11 (27.3%)  | 10/11 (90.9%)   | 0.836 | 4           | 7           | 7.84        | 47%      | 81%      | 7.0       | 27.5      | 0.7      | 12.7    | 0.0      | 47.9        |
| 17  | YOLO11n    | OC-SORT   | TransReID (ViT-B/16*) | Selesai | 186/259         | 16/11               | 0.947     | 0.932  | 0.939 | 5/11 (45.5%)  | 10/11 (90.9%)   | 0.878 | 3           | 5           | 7.81        | 56%      | 82%      | 7.7       | 23.9      | 0.8      | 22.5    | 0.0      | 54.8        |
| 18  | YOLO11n    | OC-SORT   | BoT (ResNet50)        | Selesai | 186/259         | 17/11               | 0.568     | 0.602  | 0.584 | 1/11 (9.1%)   | 8/11 (72.7%)    | 0.688 | 4           | 10          | 7.85        | 36%      | 80%      | 7.5       | 27.1      | 0.8      | 13.7    | 0.0      | 49.1        |
| 19  | RTDETRv2-s | ByteTrack | OSNet                 | Selesai | 230/259         | 14/11               | 0.925     | 0.836  | 0.878 | 3/11 (27.3%)  | 11/11 (100.0%)  | 0.835 | 3           | 4           | 7.85        | 53%      | 86%      | 6.3       | 98.7      | 0.4      | 13.3    | 0.0      | 118.7       |
| 20  | RTDETRv2-s | ByteTrack | TransReID (ViT-B/16*) | Selesai | 230/259         | 16/11               | 0.940     | 0.876  | 0.907 | 3/11 (27.3%)  | 10/11 (90.9%)   | 0.878 | 4           | 7           | 7.81        | 32%      | 86%      | 7.2       | 110.3     | 0.5      | 22.1    | 0.0      | 140.1       |
| 21  | RTDETRv2-s | ByteTrack | BoT (ResNet50)        | Selesai | 230/259         | 25/11               | 0.940     | 0.550  | 0.694 | 0/11 (0.0%)   | 9/11 (81.8%)    | 0.664 | 4           | 11          | 7.85        | 34%      | 84%      | 7.1       | 99.8      | 0.4      | 9.0     | 0.0      | 116.3       |
| 22  | RTDETRv2-s | BoT-SORT  | OSNet                 | Selesai | 228/259         | 15/11               | 0.933     | 0.841  | 0.884 | 4/11 (36.4%)  | 11/11 (100.0%)  | 0.827 | 2           | 4           | 7.78        | 30%      | 87%      | 5.5       | 102.8     | 10.5     | 11.1    | 0.0      | 129.8       |
| 23  | RTDETRv2-s | BoT-SORT  | TransReID (ViT-B/16*) | Selesai | 226/259         | 16/11               | 0.888     | 0.872  | 0.880 | 2/11 (18.2%)  | 10/11 (90.9%)   | 0.809 | 4           | 6           | 7.80        | 40%      | 87%      | 5.6       | 155.0     | 10.7     | 24.5    | 0.0      | 196.0       |
| 24  | RTDETRv2-s | BoT-SORT  | BoT (ResNet50)        | Selesai | 228/259         | 22/11               | 0.657     | 0.572  | 0.612 | 0/11 (0.0%)   | 8/11 (72.7%)    | 0.670 | 3           | 10          | 7.76        | 19%      | 84%      | 5.7       | 136.8     | 10.4     | 8.1     | 0.0      | 160.9       |
| 25  | RTDETRv2-s | OC-SORT   | OSNet                 | Selesai | 223/259         | 15/11               | 0.940     | 0.844  | 0.889 | 5/11 (45.5%)  | 11/11 (100.0%)  | 0.841 | 2           | 4           | 7.85        | 61%      | 88%      | 7.6       | 122.1     | 1.0      | 15.7    | 0.0      | 146.4       |
| 26  | RTDETRv2-s | OC-SORT   | TransReID (ViT-B/16*) | Selesai | 222/259         | 15/11               | 0.873     | 0.878  | 0.876 | 4/11 (36.4%)  | 10/11 (90.9%)   | 0.805 | 3           | 6           | 7.70        | 33%      | 89%      | 7.7       | 188.4     | 1.0      | 26.9    | 0.0      | 224.0       |
| 27  | RTDETRv2-s | OC-SORT   | BoT (ResNet50)        | Selesai | 223/259         | 24/11               | 0.713     | 0.578  | 0.638 | 0/11 (0.0%)   | 8/11 (72.7%)    | 0.687 | 2           | 10          | 7.85        | 56%      | 87%      | 7.9       | 172.3     | 1.0      | 11.2    | 0.0      | 192.5       |

**BoT (ResNet50) — checkpoint:** SEMUA baris BoT-ResNet50 di run ini (#3, 6, 9,
12, 15, 18, 21, 24, 27) pakai **satu** checkpoint: `checkpoints/bot_resnet50_market1501.pth`
via `BotResNet50Extractor` (`REID_MODEL=bot_resnet50`). Itu checkpoint resmi
paper (Luo et al., *Bag of Tricks and a Strong Baseline for Deep Person Re-ID*,
CVPRW 2019, arXiv:1903.07071), Rank-1 94.5% di Market1501. Link resmi di README
[`michuanhaohao/reid-strong-baseline`](https://github.com/michuanhaohao/reid-strong-baseline)
mati (HTTP 500) — dipakai mirror komunitas dari
[issue #151](https://github.com/michuanhaohao/reid-strong-baseline/issues/151)
repo yang sama (`market_resnet50_model_120_rank1_945.pth`), diverifikasi cocok
struktur key-nya sama `Baseline.load_param()` kode asli. Checkpoint ini SEHAT
(bukan degenerate).

Hasilnya tetap terburuk di seluruh grid (F1 0.43–0.69, akurasi ≤ 9.1%) — bukan
karena checkpoint, tapi **domain gap Market1501 (outdoor, kamera setinggi
pinggang) → CCTV indoor (sudut atas)**. OSNet (MSMT17) & TransReID transfer
lebih baik ke domain ini. Lihat Kesimpulan.

> Catatan run lama: tabel versi sebelumnya menandai #3 dengan † (checkpoint
> resmi) dan #6,9,… dengan ‡ (checkpoint `torchreid` `resnet50_market1501_converted.pth`,
> rank-1 25.67%, degenerate). Run ini menyeragamkan semua ke checkpoint resmi,
> jadi tanda †/‡ dihapus.

## Kesimpulan

**Komponen terpilih: YOLO26n + ByteTrack + TransReID (ViT-B/16*)** — kombinasi
#2, menggantikan OSNet sebagai model Re-ID pada baseline (#1).

Perbandingan langsung #1 vs #2 — detektor & tracker sama persis, cuma model
Re-ID yang beda, jadi efek Re-ID-nya terisolasi bersih:

| Metrik | #1 OSNet | #2 TransReID |
| --- | --- | --- | --------- | --------- | -------- | ------- | -------- | ----------- |
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
- **BoT-ResNet50 (#3, 6, 9, 12, 15, 18, 21, 24, 27)** — semua pakai checkpoint
  resmi paper (Rank-1 94.5% di Market1501, `bot_resnet50_market1501.pth`).
  Checkpoint sehat (precision sampai 0.94 di #21, bukan lagi ~0.1 seperti
  checkpoint `torchreid` lama yang degenerate), tapi F1 tetap terburuk di
  seluruh grid (0.43–0.69 vs OSNet/TransReID 0.87–0.96) dan akurasi ≤ 9.1%.
  Mengonfirmasi BoT-ResNet50 bukan kandidat layak di data CCTV indoor ini —
  domain gap Market1501 → CCTV, bukan kualitas checkpoint.

**Trade-off yang diterima:** TransReID menghasilkan lebih banyak identitas
(17/11) karena Recall-nya lebih rendah — orang yang sama kadang dianggap
identitas baru kalau sudut kamera/pencahayaan berubah signifikan
(*over-segmentation*, false split naik dari 3 ke 5). Ini dianggap trade-off
yang lebih bisa diterima dibanding risiko *false merge* (privasi/keamanan:
menyamakan dua orang berbeda), dan konsisten dengan prioritas F1/Precision
sebagai metrik utama evaluasi.
