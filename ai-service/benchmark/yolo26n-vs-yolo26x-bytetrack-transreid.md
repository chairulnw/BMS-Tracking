# YOLO26n vs YOLO26x (ByteTrack + TransReID)

| Detektor   | Tracker   | Re-ID                 | Total frame | Cakupan deteksi | Identitas terbentuk | Precision | Recall | F1    | Akurasi ketat | False merge | False split | FPS efektif | CPU peak | RAM peak | Decode ms | Detect ms | Track ms | ReID ms | Match ms | Total AI ms |
| ---------- | --------- | --------------------- | ----------- | --------------- | ------------------- | --------- | ------ | ----- | ------------- | ----------- | ----------- | ----------- | -------- | -------- | --------- | --------- | -------- | ------- | -------- | ----------- |
| YOLO26n    | ByteTrack | TransReID (ViT-B/16*)  | 1438        | 182/259         | 12/11               | 0.957     | 0.984  | 0.970 | 5/11 (45.5%)  | 2           | 3           | 7.50        | 50%      | 82%      | 8.1       | 23.9      | 0.5      | 32.7    | 0.2      | 65.4        |
| YOLO26x    | ByteTrack | TransReID (ViT-B/16*)  | 886         | 127/259         | 13/11               | 0.977     | 0.940  | 0.958 | 5/11 (45.5%)  | 2           | 4           | 7.51        | 36%      | 87%      | 8.2       | 173.4     | 1.0      | 42.1    | 0.5      | 225.2       |

## Akuntansi Tracklet

### YOLO26n + ByteTrack + TransReID (ViT-B/16*)

Total tracklet ditutup: 39 — lolos gate (NEW+MATCH): 27, gagal gate (BUANG): 12, dilipat ke tracklet lain (FOLD): 0.

| Kamera | Track ID | Deteksi | Sample | Hasil | Identitas / keterangan |
| ------ | -------- | ------- | ------ | ----- | ----------------------- |
| c8_sim | t1 | 48 | 16 | NEW | Unknown #1 |
| c8_sim | t5 | 10 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t8 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c12_sim | t11 | 8 | 7 | NEW | Unknown #2 |
| c11_sim | t16 | 51 | 16 | MATCH | Unknown #2 |
| c11_sim | t17 | 32 | 16 | MATCH | Unknown #2 |
| c11_sim | t15 | 243 | 16 | NEW | Unknown #3 |
| c11_sim | t23 | 80 | 4 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t9 | 40 | 16 | MATCH | Unknown #3 |
| c10_sim | t10 | 43 | 16 | MATCH | Unknown #2 |
| c10_sim | t28 | 29 | 3 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t29 | 1 | 1 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t24 | 10 | 10 | NEW | Unknown #4 |
| c11_sim | t27 | 2 | 2 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t32 | 57 | 16 | NEW | Unknown #5 |
| c9_sim | t3 | 256 | 16 | MATCH | Unknown #1 |
| c8_sim | t6 | 54 | 16 | MATCH | Unknown #1 |
| c8_sim | t36 | 28 | 16 | NEW | Unknown #6 |
| c10_sim | t30 | 24 | 16 | MATCH | Unknown #5 |
| c11_sim | t41 | 5 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t39 | 44 | 16 | NEW | Unknown #7 |
| c11_sim | t42 | 6 | 5 | MATCH | Unknown #7 |
| c12_sim | t14 | 5 | 4 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t29 | 24 | 16 | MATCH | Unknown #7 |
| c11_sim | t43 | 68 | 16 | BUANG | koherensi sample 0.21 < 0.35, embedding terkontaminasi — cegah magnet |
| c9_sim | t33 | 63 | 16 | MATCH | Unknown #6 |
| c8_sim | t49 | 24 | 11 | NEW | Unknown #8 |
| c10_sim | t45 | 22 | 6 | NEW | Unknown #9 |
| c11_sim | t46 | 43 | 16 | MATCH | Unknown #8 |
| c11_sim | t55 | 47 | 16 | NEW | Unknown #10 |
| c11_sim | t57 | 5 | 1 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c9_sim | t48 | 36 | 16 | MATCH | Unknown #8 |
| c8_sim | t61 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c12_sim | t12 | 20 | 16 | MATCH | Unknown #3 |
| c10_sim | t50 | 25 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t52 | 27 | 15 | MATCH | Unknown #10 |
| c11_sim | t58 | 68 | 16 | NEW | Unknown #11 |
| c9_sim | t59 | 66 | 16 | NEW | Unknown #12 |
| c8_sim | t62 | 53 | 16 | NEW | Unknown #13 |


### YOLO26x + ByteTrack + TransReID (ViT-B/16*)

Total tracklet ditutup: 46 — lolos gate (NEW+MATCH): 27, gagal gate (BUANG): 19, dilipat ke tracklet lain (FOLD): 0.

| Kamera | Track ID | Deteksi | Sample | Hasil | Identitas / keterangan |
| ------ | -------- | ------- | ------ | ----- | ----------------------- |
| c9_sim | t4 | 3 | 3 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t1 | 23 | 16 | NEW | Unknown #1 |
| c8_sim | t3 | 3 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t2 | 2 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t20 | 54 | 16 | NEW | Unknown #2 |
| c11_sim | t19 | 151 | 16 | BUANG | koherensi sample 0.29 < 0.35, embedding terkontaminasi — cegah magnet |
| c10_sim | t12 | 28 | 16 | MATCH | Unknown #2 |
| c10_sim | t14 | 25 | 16 | NEW | Unknown #3 |
| c10_sim | t32 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t33 | 6 | 6 | NEW | Unknown #4 |
| c11_sim | t25 | 21 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t27 | 7 | 7 | MATCH | Unknown #4 |
| c11_sim | t31 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t38 | 55 | 16 | NEW | Unknown #5 |
| c9_sim | t7 | 111 | 16 | MATCH | Unknown #1 |
| c9_sim | t9 | 4 | 4 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t11 | 23 | 15 | MATCH | Unknown #1 |
| c8_sim | t42 | 29 | 16 | NEW | Unknown #6 |
| c10_sim | t35 | 14 | 14 | MATCH | Unknown #5 |
| c10_sim | t37 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t32 | 7 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t40 | 12 | 12 | NEW | Unknown #7 |
| c11_sim | t44 | 45 | 16 | NEW | Unknown #8 |
| c11_sim | t45 | 6 | 5 | MATCH | Unknown #8 |
| c11_sim | t46 | 4 | 4 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c12_sim | t15 | 18 | 16 | MATCH | Unknown #4 |
| c12_sim | t18 | 10 | 10 | MATCH | Unknown #4 |
| c10_sim | t43 | 27 | 16 | MATCH | Unknown #8 |
| c11_sim | t44 | 70 | 16 | BUANG | koherensi sample 0.24 < 0.35, embedding terkontaminasi — cegah magnet |
| c9_sim | t41 | 78 | 16 | MATCH | Unknown #6 |
| c8_sim | t53 | 6 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t54 | 25 | 15 | NEW | Unknown #9 |
| c10_sim | t49 | 22 | 9 | NEW | Unknown #10 |
| c11_sim | t50 | 53 | 16 | MATCH | Unknown #9 |
| c11_sim | t62 | 56 | 16 | NEW | Unknown #11 |
| c9_sim | t52 | 60 | 16 | MATCH | Unknown #9 |
| c9_sim | t65 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t68 | 1 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c8_sim | t69 | 1 | 1 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c12_sim | t47 | 3 | 2 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t57 | 42 | 16 | MATCH | Unknown #11 |
| c10_sim | t58 | 7 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c10_sim | t59 | 2 | 0 | BUANG | < 5 crop berkualitas, bukti visual terlalu tipis |
| c11_sim | t64 | 21 | 16 | NEW | Unknown #12 |
| c9_sim | t66 | 21 | 16 | NEW | Unknown #13 |
| c8_sim | t70 | 15 | 14 | NEW | Unknown #14 |


## Klip

Klip rekaman (dengan overlay bbox, `CLIP_BBOX_OVERLAY=1`) ada di `output/clips/clip_*_sim_*.mp4` untuk kamera-kamera `sample/` (`c8_sim`..`c12_sim`) selama run ini — cek modified-time-nya biar tau file mana yang dari run ini.
