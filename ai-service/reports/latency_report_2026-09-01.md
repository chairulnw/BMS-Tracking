# Laporan Latency — Sesi 2026-09-01 (Restore)

Direkonstruksi dari hasil analisis di percakapan (sumber CSV mentah sesi ini
sudah tertimpa setelah AI service di-restart — lihat catatan di bawah).
Cakupan: 22 event `person_detected` dari 29 event yang di-log (`zone_entry`
dikeluarkan, gak ada tracklet-nya).

**Definisi kolom:**
- **AI Latency Total** = akumulasi `decode+detection+tracking+reid+matching`
  seluruh frame dalam rentang `[started_at, ended_at]` tracklet itu — biaya
  compute total, BUKAN wall-clock (bisa lebih gede dari durasi asli, karena
  `detection_ms` di-duplikasi ke semua kamera yang ikut 1 siklus batch).
- **AI Latency Frame Terakhir** = `total_ai_ms` kolom asli di baris CSV
  TERAKHIR dalam rentang tracklet itu (1 frame doang, bukan akumulasi).
- **End-to-End** = `Date.now() browser − timestamp (ended_at)` = udah
  otomatis `Backend Delay + Browser Wait`. Ini yang valid dipakai buat "dari
  orang terakhir kedeteksi sampai muncul di dashboard" — AI Latency (dua-duanya)
  TIDAK perlu/boleh dijumlahin ke sini (beda rentang waktu, lihat diskusi:
  AI Latency terjadi SEBELUM `ended_at`, End-to-End sesudahnya).

| Event | Kamera | AI Latency Total (ms) | AI Latency Frame Terakhir (ms) | End-to-End (ms) |
|---|---|---:|---:|---:|
| #1 | c10 | 6.785,9 | 23,782 | 23.351 |
| #2 | c12 | 1.811,9 | 110,197 | 17.941 |
| #3 | c11 | 10.678,8 | 124,040 | 7.643 |
| #5 | c11 | 14.070,4 | 102,000 | 30.603 |
| #6 | c12 | 600,1 | 600,097 | 22.472 |
| #7 | c10 | 8.933,4 | 112,367 | 12.677 |
| #8 | c8 | 3.550,0 | 351,012 | 12.434 |
| #9 | c10 | 294,7 | 69,537 | 19.772 |
| #10 | c10 | 10.196,3 | 121,472 | 31.988 |
| #11 | c12 | 3.384,4 | 203,339 | 21.945 |
| #12 | c12 | 159,9 | 159,888 | 17.273 |
| #13 | c12 | 10.535,3 | 1.096,888 | 13.592 |
| #14 | c11 | 223,2 | 223,197 | 34.519 |
| #15 | c12 | 182,1 | 182,053 | 34.070 |
| #16 | c10 | 7.455,7 | 27,423 | 8.404 |
| #21 | c11 | 19.720,3 | 131,392 | 8.197 |
| #22 | c10 | 8.994,8 | 128,899 | 23.297 |
| #23 | c10 | 7.454,2 | 64,463 | 9.211 |
| #26 | c11 | 14.147,1 | 175,966 | 15.869 |
| #27 | c12 | 233,3 | 233,275 | 11.139 |
| #28 | c12 | 3.799,9 | 123,026 | 8.848 |
| #29 | c12 | 1.576,1 | 135,115 | 35.700 |

## Ringkasan

| Layer | Rata-rata | Min | Max |
|---|---:|---:|---:|
| AI Latency Total | 4.673,1 ms | 64,8 ms | 19.720,3 ms |
| Backend Delay | 2.754,6 ms | 93,5 ms | 8.274,2 ms |
| Browser Wait | 15.997,1 ms | 4.114,2 ms | 29.836,4 ms |
| End-to-End | 18.751,8 ms | 7.643,0 ms | 35.700,0 ms |

**Catatan soal file yang hilang:** `latency.csv` mentah (70.604 baris) buat
sesi ini udah ketimpa waktu AI service di-restart (file ditulis mode
overwrite tiap restart, bukan append) — jadi tabel di atas gak bisa
di-generate ULANG dari nol pakai `tools/latency_report.py` lagi. Ini disalin
dari angka yang udah dihitung & diverifikasi bolak-balik di sesi diskusi
sebelumnya (termasuk perbaikan bug double-count `AI Latency Total`). Bagian
lain dari laporan lengkap (per-tahap decode/detect/track/reid/matching,
Bagian 0/6/7/8) TIDAK bisa direstore karena butuh baris mentah CSV yang
sudah hilang — cuma tabel ringkas ini yang tersimpan di riwayat percakapan.
