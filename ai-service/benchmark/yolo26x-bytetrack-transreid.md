# YOLO26x + ByteTrack + TransReID

| Detektor   | Tracker   | Re-ID                 | Total frame | Cakupan deteksi | Identitas terbentuk | Precision | Recall | F1    | Akurasi ketat | False merge | False split | FPS efektif | CPU peak | RAM peak | Decode ms | Detect ms | Track ms | ReID ms | Match ms | Total AI ms |
| ---------- | --------- | --------------------- | ----------- | --------------- | ------------------- | --------- | ------ | ----- | ------------- | ----------- | ----------- | ----------- | -------- | -------- | --------- | --------- | -------- | ------- | -------- | ----------- |
| YOLO26x    | ByteTrack | TransReID (ViT-B/16*)  | 899         | 125/259         | 11/11               | 0.980     | 0.964  | 0.972 | 6/11 (54.5%)  | 2           | 2           | 7.52        | 25%      | 83%      | 6.8       | 163.4     | 0.8      | 40.6    | 0.3      | 212.0       |

## Akuntansi Tracklet

Total tracklet ditutup: 39 — lolos gate (NEW+MATCH): 25, gagal gate (BUANG): 14, dilipat ke tracklet lain (FOLD): 0.

| Kamera | Track ID | Deteksi | Sample | Hasil | Identitas / keterangan |
| ------ | -------- | ------- | ------ | ----- | ----------------------- |
| c9_sim | t5 | 2 | 2 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t1 | 22 | 16 | NEW | Unknown #1 |
| c8_sim | t10 | 2 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t21 | 56 | 16 | NEW | Unknown #2 |
| c11_sim | t20 | 154 | 16 | BUANG | koherensi sample 0.29 < 0.35, embedding terkontaminasi — cegah magnet |
| c10_sim | t12 | 29 | 16 | MATCH | Unknown #2 |
| c10_sim | t14 | 25 | 16 | NEW | Unknown #3 |
| c10_sim | t29 | 2 | 2 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t23 | 20 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t24 | 4 | 4 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t26 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t33 | 54 | 16 | NEW | Unknown #4 |
| c9_sim | t6 | 120 | 16 | MATCH | Unknown #1 |
| c8_sim | t11 | 23 | 11 | MATCH | Unknown #1 |
| c8_sim | t37 | 29 | 16 | NEW | Unknown #5 |
| c10_sim | t32 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t27 | 39 | 12 | MATCH | Unknown #4 |
| c11_sim | t35 | 11 | 11 | NEW | Unknown #6 |
| c11_sim | t39 | 43 | 16 | NEW | Unknown #7 |
| c11_sim | t40 | 6 | 5 | MATCH | Unknown #7 |
| c12_sim | t15 | 17 | 16 | MATCH | Unknown #3 |
| c12_sim | t19 | 9 | 9 | MATCH | Unknown #3 |
| c10_sim | t38 | 24 | 16 | MATCH | Unknown #7 |
| c11_sim | t41 | 75 | 16 | BUANG | koherensi sample 0.29 < 0.35, embedding terkontaminasi — cegah magnet |
| c9_sim | t36 | 79 | 16 | MATCH | Unknown #5 |
| c8_sim | t48 | 6 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t49 | 25 | 15 | NEW | Unknown #8 |
| c10_sim | t44 | 22 | 9 | NEW | Unknown #9 |
| c11_sim | t45 | 52 | 16 | MATCH | Unknown #8 |
| c11_sim | t57 | 52 | 16 | NEW | Unknown #10 |
| c9_sim | t47 | 58 | 16 | MATCH | Unknown #8 |
| c8_sim | t64 | 1 | 1 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c12_sim | t42 | 3 | 2 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t52 | 43 | 16 | MATCH | Unknown #10 |
| c10_sim | t53 | 7 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t54 | 2 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t60 | 22 | 16 | NEW | Unknown #11 |
| c9_sim | t61 | 21 | 16 | NEW | Unknown #12 |
| c8_sim | t65 | 18 | 16 | NEW | Unknown #13 |

## Klip

Klip rekaman (dengan overlay bbox, `CLIP_BBOX_OVERLAY=1`) ada di `output/clips/clip_*.mp4` untuk kamera-kamera `sample/` (`c8_sim`..`c12_sim`) selama run ini — cek modified-time-nya biar tau file mana yang dari run ini.
