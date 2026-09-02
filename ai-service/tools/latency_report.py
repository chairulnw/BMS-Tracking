"""Laporan latency per-layer (AI / backend / browser / end-to-end) buat N event
`camera_events`. Gabungin 4 sumber: tabel `camera_events`, tabel `tracklets`,
`latency.csv` (per-frame), dan `end_to_end_ms` dari console log browser (satu-
satunya sumbernya — waktu browser terima response gak disimpan di mana pun,
cuma keliatan di DevTools Console pas testing).

Jalanin dari ai-service/ (path CSV & DB relatif ke situ):
    cd ai-service && source .venv/bin/activate && python3 tools/latency_report.py

END_TO_END_MS di bawah HARUS diisi manual dari console log browser (paste
`[latency] event #N ...` satu-satu, key = id event) — gak ada cara ambil ini
dari DB, browser gak pernah lapor balik jam dia sendiri ke server.
"""
import csv
import subprocess
from datetime import datetime
from pathlib import Path

AI_SERVICE_DIR = Path(__file__).resolve().parents[1]
LATENCY_CSV = AI_SERVICE_DIR / "latency.csv"
OUT_MD = AI_SERVICE_DIR / "latency_report.md"

# reid_ms nonzero yang di bawah ini = ditolak quality-gate (_extract_embedding,
# pipeline_service.py:609-628 — box kekecilan/rasio aneh/buram, return None
# SEBELUM model dipanggil), bukan ekstraksi beneran. Batas ini dari gap jelas
# di distribusi (p5=0.005ms, p10=27.9ms — bimodal).
REID_REAL_MS = 1.0
# matching_ms nonzero yang di bawah ini = cuma observe()/update_active()
# (bookkeeping murah, batch_processor.py:401-408), bukan associate() beneran
# (cosine similarity ke gallery, cuma jalan pas tracklet ditutup).
MATCH_REAL_MS = 0.05

# end_to_end_ms per event id — dari console log browser (overview.ts:223),
# gak ada di DB. Isi ulang manual tiap kali mau laporan baru.
END_TO_END_MS = {
    1: 23351, 2: 17941, 3: 7643, 4: 9191, 5: 30603, 6: 22472, 7: 12677,
    8: 12434, 9: 19772, 10: 31988, 11: 21945, 12: 17273, 13: 13592,
    14: 34519, 15: 34070, 16: 8404, 17: 9733, 18: 30366, 19: 13790,
    20: 23796, 21: 8197, 22: 23297, 23: 9211, 24: 20030, 25: 15951,
    26: 15869, 27: 11139, 28: 8848, 29: 35700,
}


def psql(query: str) -> list[str]:
    return subprocess.run(
        ["psql", "-d", "bms_tracking", "-t", "-A", "-F,", "-c", query],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip().replace(" ", "T"))


def load_camera_events() -> list[dict]:
    rows = psql(
        "select id, camera_id, event_type, category, timestamp, created_at, "
        "ai_latency_ms from camera_events order by id;"
    )
    out = []
    for line in rows:
        eid, cam, etype, cat, ts, created, ai_ms = line.split(",", 6)
        out.append({
            "id": int(eid), "camera_id": cam, "event_type": etype,
            "category": cat, "timestamp": parse_ts(ts),
            "created_at": parse_ts(created), "ai_latency_ms_snapshot": float(ai_ms),
        })
    return out


def load_tracklets() -> dict:
    rows = psql("select camera_id, ended_at, started_at from tracklets;")
    out = {}
    for line in rows:
        cam, ended, started = line.split(",")
        out[(cam, parse_ts(ended))] = parse_ts(started)
    return out


def load_all_tracklets() -> list[dict]:
    """Semua tracklet (53 kejadian nyata di DB) — beda dari load_tracklets()
    yang cuma dict lookup buat 29 event yang punya console-log. Ini yang dipakai
    Bagian 7 (breakdown per kejadian, bukan per event yang ke-log browser)."""
    rows = psql(
        "select id, camera_id, track_id, person_id, started_at, ended_at, "
        "n_detections from tracklets order by ended_at;"
    )
    out = []
    for line in rows:
        tid, cam, trk, pid, started, ended, ndet = line.split(",")
        out.append({
            "id": int(tid), "camera_id": cam, "track_id": int(trk),
            "person_id": int(pid), "started_at": parse_ts(started),
            "ended_at": parse_ts(ended), "n_detections": int(ndet),
        })
    return out


def load_latency_csv() -> list[tuple]:
    rows = []
    with open(LATENCY_CSV) as f:
        r = csv.reader(f)
        next(r)
        for lineno, row in enumerate(r, start=2):
            t = datetime.fromisoformat(row[0])
            vals = [float(x) if x else 0.0 for x in row[3:9]]  # decode..total_ai
            rows.append((lineno, row[1], t, vals))
    return rows


def build_cam_index(rows: list[tuple]) -> dict:
    """cam -> list of (lineno, timestamp), urut waktu — buat cari frame
    SEBELUM frame tertentu (delta antar-frame asli, bukan sum durasi logis)."""
    idx: dict[str, list[tuple]] = {}
    for ln, cam, t, _ in rows:
        idx.setdefault(cam, []).append((ln, t))
    for cam in idx:
        idx[cam].sort(key=lambda x: x[0])
    return idx


def prev_frame_gap_ms(cam_index: dict, cam: str, last_ln: int, last_t: "datetime") -> "float | None":
    """Selisih waktu ASLI (wall-clock) antara frame terakhir dan frame
    SEBELUMNYA di kamera yang sama — lebih akurat drpd sum(decode..reid),
    karena decode_ms diukur di thread terpisah/concurrent (lihat catatan
    Bagian 0), jadi sum-nya bisa overestimate durasi asli antar-frame."""
    rows = cam_index.get(cam, [])
    pos = next((i for i, (ln, _) in enumerate(rows) if ln == last_ln), None)
    if pos is None or pos == 0:
        return None
    prev_t = rows[pos - 1][1]
    return (last_t - prev_t).total_seconds() * 1000


def find_csv_range(rows, cam, lo, hi):
    matches = [(ln, t, vals) for ln, c, t, vals in rows if c == cam and lo <= t <= hi]
    if not matches:
        cand = [(ln, t, vals) for ln, c, t, vals in rows if c == cam and t <= hi]
        if cand:
            matches = [max(cand, key=lambda x: x[1])]
    return matches


def main():
    events = load_camera_events()
    tracklet_by_key = load_tracklets()
    csv_rows = load_latency_csv()
    cam_index = build_cam_index(csv_rows)

    report_rows = []
    for ev in events:
        eid, cam, etype, ts = ev["id"], ev["camera_id"], ev["event_type"], ev["timestamp"]
        started = tracklet_by_key.get((cam, ts)) if etype == "person_detected" else ts
        lo, hi = (started, ts) if started else (ts, ts)
        matches = find_csv_range(csv_rows, cam, lo, hi)
        # vals = [decode,detection,tracking,reid,matching,total_ai_ms] — index 5
        # (total_ai_ms) UDAH jumlah 4 kolom sebelumnya di baris yang sama, jadi
        # ambil index 5 doang per baris, JANGAN sum(vals) (itu bakal 2x lipat:
        # 4 kolom + total_ai_ms yang notabene = jumlah 4 kolom itu lagi).
        ai_ms = sum(vals[5] for _, _, vals in matches)
        # total_ai_ms KOLOM ASLI di baris CSV terakhir (bukan dihitung ulang) —
        # cocokin via (camera_id, timestamp) EXACT ke tracklets.ended_at, JANGAN
        # pasang manual berdasar urutan baris (kejadian yang gagal post_tracklet,
        # lihat Bagian 7, bikin urutan geser kalau dipasang positional).
        last_ln, last_t, last_row = max(matches, key=lambda m: m[1]) if matches else (None, None, None)
        ai_last = last_row[5] if last_row else None
        # Delta ANTAR-FRAME ASLI (frame terakhir vs frame sebelumnya, kamera
        # sama) — bukan sum(decode..reid) logis. decode_ms diukur di thread
        # capture TERPISAH/concurrent (lihat Bagian 0), jadi sum durasi 5
        # kolom itu OVERESTIMATE durasi asli antar-frame (dibuktikan: delta
        # asli ~12ms vs sum logis ~24ms buat event #1). Ini yang lebih akurat.
        pre_ended_ms = prev_frame_gap_ms(cam_index, cam, last_ln, last_t) if last_row else None
        line_lo = min(m[0] for m in matches) if matches else None
        line_hi = max(m[0] for m in matches) if matches else None

        backend_delay_ms = (ev["created_at"] - ts).total_seconds() * 1000
        e2e = END_TO_END_MS.get(eid)
        browser_wait = (e2e - backend_delay_ms) if e2e is not None else None
        true_e2e = (pre_ended_ms + e2e) if (pre_ended_ms is not None and e2e is not None) else None

        report_rows.append({
            **ev, "started_at": started, "ai_latency_ms": ai_ms, "ai_latency_last_ms": ai_last,
            "pre_ended_ms": pre_ended_ms, "true_e2e_from_last_frame_ms": true_e2e,
            "csv_line_lo": line_lo, "csv_line_hi": line_hi, "csv_n": len(matches),
            "ai_latency_per_frame_ms": (ai_ms / len(matches)) if matches else 0.0,
            "backend_delay_ms": backend_delay_ms, "end_to_end_ms": e2e,
            "browser_wait_ms": browser_wait, "_matched_frames": matches,
        })

    cols = ["decode_ms", "detection_ms", "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"]
    col_vals = {c: [vals[i] for _, _, _, vals in csv_rows] for i, c in enumerate(cols)}
    n_total = len(csv_rows)

    def fmt_stats(vals):
        if not vals:
            return "n=0"
        return f"n={len(vals):,} avg={sum(vals)/len(vals):,.3f}ms min={min(vals):,.3f}ms max={max(vals):,.2f}ms"

    trk_nz    = [v for v in col_vals["tracking_ms"] if v > 0]
    reid_nz   = [v for v in col_vals["reid_ms"] if v > 0]
    reid_real = [v for v in reid_nz if v >= REID_REAL_MS]
    reid_rej  = [v for v in reid_nz if v < REID_REAL_MS]
    mat_nz    = [v for v in col_vals["matching_ms"] if v > 0]
    mat_close = [v for v in mat_nz if v > MATCH_REAL_MS]
    mat_cheap = [v for v in mat_nz if v <= MATCH_REAL_MS]

    lines = []
    lines.append("# Laporan Latency Per-Layer — Person Detection Pipeline\n")
    lines.append(
        "Digenerate `tools/latency_report.py`. Sumber: tabel `camera_events`, "
        "`tracklets` (DB), `latency.csv` (per-frame, seluruh sesi), dan "
        "`end_to_end_ms` dari console log browser (satu-satunya sumbernya — "
        "jam browser terima response gak disimpan di mana pun selain DevTools "
        "Console pas testing).\n"
    )

    lines.append("## Ringkasan eksekutif — alur lengkap kamera sampai dashboard\n")
    lines.append(
        "**1. Capture** (`cam_slot.py`) — `cap.read()` per kamera, non-blocking "
        "(frame didrop kalau inferensi belum selesai). Dicatet `decode_ms`.\n\n"
        "**2. Deteksi** (`batch_processor.py`) — YOLO dipanggil 1x per siklus "
        "buat SEMUA kamera aktif sekaligus, SELALU jalan tiap siklus terlepas "
        "ada orang apa nggak (dia yang nentuin). Dicatet `detection_ms`.\n\n"
        "**3. Tracking** (per kamera) — BYTETracker/BoTSORT, CUMA jalan kalau "
        "`n_raw>0` (ada ≥1 orang di frame ini). Dicatet `tracking_ms`.\n\n"
        "**4. Ekstraksi embedding ReID** (per track) — `_extract_embedding()` "
        "dipanggil per track HASIL tracking, tapi ada quality-gate duluan "
        "(ukuran/rasio/blur box) SEBELUM model beneran dipanggil — kalau gagal, "
        "balik cepat (`<1ms`) tanpa ekstraksi beneran. Dicatet `reid_ms`.\n\n"
        "**5. Matching ke gallery** (`associate()`) — SELALU 'dipanggil' tiap "
        "siklus (bookkeeping `observe()`/`update_active()` gak ada syarat), "
        "tapi cosine-similarity beneran ke gallery CUMA jalan pas tracklet "
        "BENERAN ditutup (`close_expired()`). Dicatet `matching_ms`.\n\n"
        "→ Langkah 1-5 = `total_ai_ms`, terjadi REAL-TIME selama orang kelihatan "
        "di kamera.\n\n"
        "**6. Tracklet digantung 15 siklus** tanpa deteksi baru (`TRACKLET_GAP_"
        "CYCLES`) — mastiin orangnya bener pergi, bukan cuma keok sesaat.\n\n"
        "**7. Resolve + POST ke backend** — rata-rata embedding, `associate()` "
        "final, antre `_PostQueue`, network, insert DB.\n\n"
        "→ Langkah 6-7 = `backend_delay_ms` (`created_at - timestamp` di DB) — "
        "didominasi langkah 6, BUKAN komputasi.\n\n"
        "**8. Browser nunggu polling berikutnya** (`interval(30000)`, tiap 30 "
        "detik) — ini yang paling dominan dari total end-to-end.\n\n"
        "→ Total 6-8 = `end_to_end_ms` (`Date.now()` browser − `timestamp` AI).\n"
    )

    lines.append(
        "**Kesimpulan posisi bottleneck**: AI (langkah 1-5) cepat & stabil "
        "(~250ms/frame). Backend (6-7) ~2,75 detik, didominasi nunggu 15-siklus. "
        "Paling lambat: langkah 8 (desain polling frontend, ~16 detik rata-rata) "
        "— bukan masalah pipeline AI/backend.\n"
    )

    lines.append("## 0. Statistik per-tahap AI — SELURUH sesi (bukan cuma 29 event)\n")
    lines.append(
        f"Dihitung dari semua {n_total:,} baris `latency.csv` (semua frame yang "
        "diproses, semua kamera, seluruh sesi test). `decode_ms`/`detection_ms` "
        "SELALU jalan tiap siklus (YOLO nentuin ada-orang-apa-nggak duluan), "
        "jadi rata-rata \"semua frame\" sudah representatif — TIDAK dipisah per "
        "event_type di Bagian 6 (lihat catatan di situ). `tracking_ms`/`reid_ms` "
        "di-skip total kalau frame kosong (`n_raw==0`, `batch_processor.py:377`). "
        "`reid_ms` nonzero masih kepecah 2: ditolak quality-gate (`<1ms`, model "
        "GAK dipanggil) vs ekstraksi beneran (`≥1ms`). `matching_ms` nonzero "
        "juga kepecah 2: bookkeeping murah (`≤0,05ms`) vs `associate()` beneran "
        "(tracklet ditutup, `>0,05ms`). Kalau rata-rata \"semua frame\" dipake "
        "buat 3 kolom terakhir ini, KEDILUSI sama frame kosong/gak-kerja — "
        "makanya dipisah.\n"
    )

    lines.append("| Tahap | Semua frame | Breakdown |")
    lines.append("|---|---|---|")
    lines.append(f"| `decode_ms` | {fmt_stats(col_vals['decode_ms'])} | selalu jalan, gak perlu dipisah |")
    lines.append(f"| `detection_ms` | {fmt_stats(col_vals['detection_ms'])} | selalu jalan, gak perlu dipisah |")
    lines.append(f"| `tracking_ms` | {fmt_stats(col_vals['tracking_ms'])} | ada-orang: {fmt_stats(trk_nz)} |")
    lines.append(
        f"| `reid_ms` | {fmt_stats(col_vals['reid_ms'])} | ekstraksi beneran (≥{REID_REAL_MS}ms): "
        f"{fmt_stats(reid_real)} <br> ditolak quality-gate (<{REID_REAL_MS}ms): {fmt_stats(reid_rej)} |"
    )
    lines.append(
        f"| `matching_ms` | {fmt_stats(col_vals['matching_ms'])} | tracklet ditutup (>{MATCH_REAL_MS}ms): "
        f"{fmt_stats(mat_close)} <br> bookkeeping doang (≤{MATCH_REAL_MS}ms): {fmt_stats(mat_cheap)} |"
    )
    lines.append(f"| `total_ai_ms` | {fmt_stats(col_vals['total_ai_ms'])} | jumlah 5 kolom di atas per frame |")

    lines.append("\n### Funnel — dari decode turun ke tiap tahap (n frame, % dari decode)\n")
    lines.append("| Tahap | n frame | % dari decode |")
    lines.append("|---|---:|---:|")
    lines.append(f"| decode (baseline) | {n_total:,} | 100% |")
    lines.append(f"| detection (selalu jalan) | {n_total:,} | 100% |")
    lines.append(f"| tracking (ada ≥1 orang) | {len(trk_nz):,} | {len(trk_nz)/n_total*100:.1f}% |")
    lines.append(f"| reid — fungsi dipanggil | {len(reid_nz):,} | {len(reid_nz)/n_total*100:.1f}% |")
    lines.append(f"| ↳ reid — ekstraksi BENERAN (lolos quality-gate) | {len(reid_real):,} | {len(reid_real)/n_total*100:.1f}% |")
    lines.append(f"| ↳ reid — ditolak quality-gate | {len(reid_rej):,} | {len(reid_rej)/n_total*100:.1f}% |")
    lines.append(f"| matching — fungsi dipanggil | {n_total:,} | 100% |")
    lines.append(f"| ↳ matching — beneran nutup tracklet | {len(mat_close):,} | {len(mat_close)/n_total*100:.1f}% |")
    lines.append(f"| ↳ matching — bookkeeping kosong | {len(mat_cheap):,} | {len(mat_cheap)/n_total*100:.1f}% |")
    lines.append(
        "\nBacanya: dari semua frame yang di-decode, cuma sebagian kecil yang "
        "beneran ada substansi buat diproses lebih jauh — sisanya sistem lagi "
        "\"mantau\" (scan kosong), bukan \"sibuk kerja\".\n"
    )

    lines.append("## 1. Tabel `camera_events`\n")
    lines.append("| id | camera | type | timestamp | created_at | ai_latency_ms (snapshot 1 frame) |")
    lines.append("|---|---|---|---|---|---:|")
    for ev in events:
        lines.append(
            f"| {ev['id']} | {ev['camera_id']} | {ev['event_type']} | "
            f"{ev['timestamp']} | {ev['created_at']} | {ev['ai_latency_ms_snapshot']:.2f} |"
        )

    lines.append("\n## 2. Tracklet terkait (`started_at`/`ended_at` dipakai buat rentang pencarian CSV)\n")
    lines.append("| id | camera | started_at | ended_at (= timestamp event) |")
    lines.append("|---|---|---|---|")
    for r in report_rows:
        if r["event_type"] == "person_detected":
            lines.append(f"| {r['id']} | {r['camera_id']} | {r['started_at']} | {r['timestamp']} |")
        else:
            lines.append(f"| {r['id']} | {r['camera_id']} | — (zone_entry, gak ada tracklet) | {r['timestamp']} |")

    lines.append("\n## 3. Baris `latency.csv` yang dipakai\n")
    lines.append("| event | camera | baris CSV | jumlah baris |")
    lines.append("|---|---|---|---:|")
    for r in report_rows:
        rng = f"{r['csv_line_lo']}-{r['csv_line_hi']}" if r['csv_line_lo'] != r['csv_line_hi'] else str(r['csv_line_lo'])
        lines.append(f"| #{r['id']} | {r['camera_id']} | {rng} | {r['csv_n']} |")

    lines.append("\n## 4. Console log (browser) + ringkasan per-layer\n")
    lines.append(
        "`end_to_end_ms` sumbernya console log browser (di-paste manual ke "
        "`END_TO_END_MS` di script ini — gak ada di DB). `backend_delay_ms` "
        "dihitung ulang dari DB (`created_at - timestamp`), bukan dari log, "
        "biar konsisten walau log browser hilang/gak lengkap.\n"
    )
    lines.append("| Event | Kamera | Tipe | AI latency total (ms) | AI latency frame terakhir (ms) | Backend delay (ms) | Browser wait (ms) | End-to-end (ms) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in report_rows:
        e2e = f"{r['end_to_end_ms']:,}" if r['end_to_end_ms'] is not None else "n/a"
        bw = f"{r['browser_wait_ms']:,.0f}" if r['browser_wait_ms'] is not None else "n/a"
        ai_last = f"{r['ai_latency_last_ms']:,.3f}" if r['ai_latency_last_ms'] is not None else "n/a"
        lines.append(
            f"| #{r['id']} | {r['camera_id']} | {r['event_type']} | {r['ai_latency_ms']:,.1f} | {ai_last} | "
            f"{r['backend_delay_ms']:,.0f} | {bw} | {e2e} |"
        )

    lines.append("\n### 4a. Versi filter — cuma `person_detected`\n")
    lines.append(
        "Kolom **`E2E dari frame terakhir masuk`** = delta waktu ASLI antara "
        "frame terakhir & frame SEBELUMNYA (kamera sama, dari timestamp baris "
        "`latency.csv`, bukan `frame_id` — itu cuma counter) + `End-to-End`. "
        "Awalnya dicoba pakai `sum(decode+detection+tracking+reid)` frame "
        "terakhir, tapi itu OVERESTIMATE — `decode_ms` diukur di thread "
        "capture terpisah/concurrent (lihat Bagian 0), jadi jumlahnya gak sama "
        "dengan selisih waktu asli antar-frame (terbukti: event #1 delta asli "
        "12,28ms vs sum logis 23,77ms). Delta antar-frame ini approksimasi yang "
        "lebih deket ke kenyataan, walau masih belum termasuk jeda antre di "
        "`frame_q` sebelum frame ini MULAI diproses.\n\n"
        "Kolom **`E2E + Total AI Last`** = `Total AI Last + End-to-End` = "
        "estimasi BATAS ATAS (upper bound) — `Total AI Last` overestimate lag "
        "frame terakhir (~2x, decode konkuren + detection di-batch), jadi pakai "
        "ini sebagai batas atas, `E2E dari frame terakhir masuk` sebagai best "
        "estimate.\n"
    )
    lines.append(
        "| Event | Kamera | Total AI (ms) | Total AI Last (ms) | Backend Delay (ms) | Browser Wait (ms) | "
        "End-to-End (ms) | E2E dari frame terakhir masuk (ms) | E2E + Total AI Last (ms) | E2E dari awal kemunculan (ms) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in report_rows:
        if r["event_type"] != "person_detected":
            continue
        e2e = f"{r['end_to_end_ms']:,}" if r['end_to_end_ms'] is not None else "n/a"
        bw = f"{r['browser_wait_ms']:,.0f}" if r['browser_wait_ms'] is not None else "n/a"
        ai_last = f"{r['ai_latency_last_ms']:,.3f}" if r['ai_latency_last_ms'] is not None else "n/a"
        true_e2e = f"{r['true_e2e_from_last_frame_ms']:,.1f}" if r['true_e2e_from_last_frame_ms'] is not None else "n/a"
        # durasi ASLI orang itu kelihatan (started_at -> ended_at, wall-clock)
        # + End-to-End yang udah ada — BUKAN AI Latency (itu compute-cost,
        # beda dimensi, udah dibahas kenapa gak boleh dipakai buat ini).
        if r["started_at"] is not None and r["end_to_end_ms"] is not None:
            wall_ms = (r["timestamp"] - r["started_at"]).total_seconds() * 1000
            e2e_from_start = f"{wall_ms + r['end_to_end_ms']:,.1f}"
        else:
            e2e_from_start = "n/a"
        if r["ai_latency_last_ms"] is not None and r["end_to_end_ms"] is not None:
            e2e_plus_ai_last = f"{r['ai_latency_last_ms'] + r['end_to_end_ms']:,.1f}"
        else:
            e2e_plus_ai_last = "n/a"
        lines.append(
            f"| #{r['id']} | {r['camera_id']} | {r['ai_latency_ms']:,.1f} | {ai_last} | "
            f"{r['backend_delay_ms']:,.0f} | {bw} | {e2e} | {true_e2e} | {e2e_plus_ai_last} | {e2e_from_start} |"
        )

    valid = [r for r in report_rows if r["end_to_end_ms"] is not None]
    n = len(valid)
    def stats(key):
        vals = [r[key] for r in valid]
        return sum(vals) / n, min(vals), max(vals)

    total_ai_ms = sum(r["ai_latency_ms"] for r in report_rows)
    total_frames = sum(r["csv_n"] for r in report_rows)
    weighted_per_frame = total_ai_ms / total_frames if total_frames else 0.0
    per_frame_vals = [r["ai_latency_per_frame_ms"] for r in report_rows if r["csv_n"]]

    lines.append(f"\n### Ringkasan (n={n} event)\n")
    lines.append("| Layer | Rata-rata | Min | Max |")
    lines.append("|---|---:|---:|---:|")
    for label, key in [
        ("AI latency (akumulasi per tracklet)", "ai_latency_ms"),
        ("Backend delay", "backend_delay_ms"),
        ("Browser wait (poll)", "browser_wait_ms"),
        ("End-to-end", "end_to_end_ms"),
    ]:
        a, mn, mx = stats(key)
        lines.append(f"| {label} | {a:,.1f} ms | {mn:,.1f} ms | {mx:,.1f} ms |")
    lines.append(
        f"| AI latency per frame (per-event min/max; weighted avg = {weighted_per_frame:,.1f}ms "
        f"dari {total_ai_ms:,.0f}ms / {total_frames} frame) | "
        f"{sum(per_frame_vals)/len(per_frame_vals):,.1f} ms | {min(per_frame_vals):,.1f} ms | {max(per_frame_vals):,.1f} ms |"
    )
    lines.append(
        "\n**Catatan**: AI latency = biaya compute (akumulasi banyak frame), "
        "BUKAN wall-clock — jangan dijumlah ke 3 kolom lain, beda rentang waktu "
        "(AI latency terjadi sebelum `ended_at`, tiga kolom lain sesudahnya).\n"
    )

    lines.append("\n## 5. Tabel gabungan — 1 baris per event, semua sumber\n")
    lines.append(
        "Semua kolom di atas (bagian 1-4) digabung jadi 1 baris per event, "
        "biar gak perlu gulir-gulir cocokin antar tabel.\n"
    )
    lines.append(
        "| id | camera | type | timestamp (AI) | started_at (tracklet) | "
        "created_at (backend) | csv baris | AI latency total (ms) | AI latency/frame (ms) | "
        "AI latency frame terakhir (ms) | backend delay (ms) | browser wait (ms) | end-to-end (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in report_rows:
        rng = (f"{r['csv_line_lo']}-{r['csv_line_hi']}"
               if r['csv_line_lo'] != r['csv_line_hi'] else str(r['csv_line_lo']))
        started = r['started_at'] if r['event_type'] == 'person_detected' else '—'
        e2e = f"{r['end_to_end_ms']:,}" if r['end_to_end_ms'] is not None else "n/a"
        bw = f"{r['browser_wait_ms']:,.0f}" if r['browser_wait_ms'] is not None else "n/a"
        ai_last = f"{r['ai_latency_last_ms']:,.3f}" if r['ai_latency_last_ms'] is not None else "n/a"
        lines.append(
            f"| {r['id']} | {r['camera_id']} | {r['event_type']} | {r['timestamp']} | "
            f"{started} | {r['created_at']} | {rng} | {r['ai_latency_ms']:,.1f} | "
            f"{r['ai_latency_per_frame_ms']:,.1f} | {ai_last} | {r['backend_delay_ms']:,.0f} | {bw} | {e2e} |"
        )

    stats_rows = compute_stats_rows(report_rows)
    lines.append("\n## 6. Statistik per-komponen, dipisah `person_detected` vs `zone_entry`\n")
    lines.append(
        "`decode_ms`/`detection_ms` SENGAJA gak dimasukin di sini — dua-duanya "
        "jalan tiap siklus terlepas dari ada event apa nggak (lihat Bagian 0), "
        "jadi ngerata-ratain dari subset frame yang kebetulan masuk rentang 29 "
        "event ini BUKAN angka yang lebih fair (sample ~1,5% dari total, "
        "berpotensi bias). **Pakai Bagian 0 buat `decode_ms`/`detection_ms`.** "
        "`reid_ms_real`/`matching_ms_real` cuma ngitung yang BENERAN kerja "
        "(lolos threshold `REID_REAL_MS`/`MATCH_REAL_MS`, sama kayak Bagian 0 — "
        "bukan sekadar nonzero) — juga tersedia sebagai `latency_stats.csv`.\n"
    )
    lines.append("| event_type | component | n | avg_ms | min_ms | max_ms |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in stats_rows:
        etype, c, n, avg, mn, mx = row
        if n == 0:
            lines.append(f"| {etype} | `{c}` | 0 | — | — | — |")
        else:
            lines.append(f"| {etype} | `{c}` | {n} | {avg:,.3f} | {mn:,.3f} | {mx:,.2f} |")

    all_tracklets = load_all_tracklets()  # udah urut by ended_at (query di load_all_tracklets)
    seen_person: set[int] = set()
    for tk in all_tracklets:
        tk["is_new"] = tk["person_id"] not in seen_person  # tracklet PERTAMA buat person_id ini = identitas baru
        seen_person.add(tk["person_id"])

    lines.append(
        "\n## 7. Breakdown per kejadian NYATA — 53 tracklet di DB "
        "(bukan 29 event sample, ini SEMUA yang beneran lolos sampai resolve)\n"
    )
    lines.append(
        "Titik acuan `ended_at` = frame TERAKHIR orang itu beneran kedeteksi "
        "(sama kayak Bagian 1-5, bukan \"frame terakhir didecode\"). Per "
        "kejadian: total ms & jumlah frame tiap tahap dalam rentang "
        "`[started_at, ended_at]` kejadian itu di `latency.csv`. Kolom **`new?`** "
        "= tracklet PERTAMA (urut waktu) yang munculin `person_id` itu → jadi "
        "`Unknown #N` BARU (23 identitas unik); yang lain (`match`) = tracklet "
        "susulan yang ke-MATCH ke identitas yang udah ada (re-appearance). Kolom "
        "**`*_last`** = nilai di frame TERAKHIR doang (baris CSV paling akhir "
        "dalam rentang tracklet ini) — beda sama kolom tanpa suffix yang "
        "AKUMULASI semua frame.\n"
    )
    lines.append(
        "| id | camera | track_id | person_id | new? | n_detections (DB) | n frame (CSV) | "
        "decode_ms | detection_ms | tracking_ms | reid_ms | matching_ms | total_ai_ms | "
        "decode_last | detection_last | tracking_last | reid_last | matching_last | total_ai_ms_last |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    tk_sums = {c: [] for c in COLS}
    tk_n_frames = []
    for tk in all_tracklets:
        matches = find_csv_range(csv_rows, tk["camera_id"], tk["started_at"], tk["ended_at"])
        sums = [0.0] * len(COLS)
        for _, _, vals in matches:
            for i in range(len(COLS)):
                sums[i] += vals[i]
        # last_vals[5] = total_ai_ms KOLOM ASLI di latency.csv buat baris ini —
        # itu udah jumlah 5 kolom di baris yang SAMA, gak perlu dihitung ulang.
        last_vals = max(matches, key=lambda m: m[1])[2] if matches else [0.0] * len(COLS)
        for i, c in enumerate(COLS):
            tk_sums[c].append(sums[i])
        tk_n_frames.append(len(matches))
        tk["_sums"] = sums
        tk["_last"] = last_vals
        tk["_n_frame"] = len(matches)
        newflag = "**baru**" if tk["is_new"] else "match"
        lines.append(
            f"| {tk['id']} | {tk['camera_id']} | {tk['track_id']} | {tk['person_id']} | {newflag} | "
            f"{tk['n_detections']} | {len(matches)} | " +
            " | ".join(f"{sums[i]:,.2f}" for i in range(len(COLS))) + " | " +
            " | ".join(f"{last_vals[i]:,.3f}" for i in range(6)) + " |"
        )

    n_tk = len(all_tracklets)
    lines.append(f"\n### Rata-rata per kejadian — SEMUA 53 tracklet (baru + match)\n")
    lines.append("| Tahap | Total ms (akumulasi) | Rata-rata ms/kejadian (akumulasi) | Rata-rata ms (frame TERAKHIR aja) | Rata-rata frame/kejadian |")
    lines.append("|---|---:|---:|---:|---:|")
    avg_frames = sum(tk_n_frames) / n_tk
    for i, c in enumerate(COLS):
        total_c = sum(tk_sums[c])
        last_avg = sum(tk["_last"][i] for tk in all_tracklets) / n_tk
        lines.append(f"| `{c}` | {total_c:,.1f} | {total_c/n_tk:,.1f} | {last_avg:,.3f} | {avg_frames:,.1f} |")
    lines.append(f"\nTotal frame terpakai (semua 53 kejadian): {sum(tk_n_frames):,}\n")

    new_only = [tk for tk in all_tracklets if tk["is_new"]]
    n_new = len(new_only)
    lines.append(f"\n## 8. Versi CUMA identitas BARU — {n_new} kejadian (bukan 53, filter `new?`)\n")
    lines.append(
        f"Subset dari Bagian 7, cuma baris `new?` (tracklet yang beneran jadi "
        f"`Unknown #1`..`Unknown #{n_new}` — cocok sama {n_new} `persons` di DB). "
        "Tracklet susulan (`match`, re-appearance orang yang sama) DIBUANG dari "
        "sini biar gak dobel-hitung 1 orang berkali-kali.\n"
    )
    lines.append(
        "| id | camera | track_id | person_id | n_detections (DB) | n frame (CSV) | "
        "decode_ms | detection_ms | tracking_ms | reid_ms | matching_ms | total_ai_ms | "
        "decode_last | detection_last | tracking_last | reid_last | matching_last | total_ai_ms_last |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tk in new_only:
        sums, last_vals = tk["_sums"], tk["_last"]
        lines.append(
            f"| {tk['id']} | {tk['camera_id']} | {tk['track_id']} | {tk['person_id']} | "
            f"{tk['n_detections']} | {tk['_n_frame']} | " +
            " | ".join(f"{sums[i]:,.2f}" for i in range(len(COLS))) + " | " +
            " | ".join(f"{last_vals[i]:,.3f}" for i in range(6)) + " |"
        )

    lines.append(f"\n### Rata-rata per kejadian — CUMA identitas baru (n={n_new})\n")
    lines.append("| Tahap | Total ms (akumulasi) | Rata-rata ms/kejadian (akumulasi) | Rata-rata ms (frame TERAKHIR aja) | Rata-rata frame/kejadian |")
    lines.append("|---|---:|---:|---:|---:|")
    new_frames = [tk["_n_frame"] for tk in new_only]
    avg_frames_new = sum(new_frames) / n_new
    for i, c in enumerate(COLS):
        total_c = sum(tk["_sums"][i] for tk in new_only)
        last_avg = sum(tk["_last"][i] for tk in new_only) / n_new
        lines.append(f"| `{c}` | {total_c:,.1f} | {total_c/n_new:,.1f} | {last_avg:,.3f} | {avg_frames_new:,.1f} |")
    lines.append(f"\nTotal frame terpakai ({n_new} identitas baru): {sum(new_frames):,}\n")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Ditulis ke {OUT_MD}")

    write_stats_csv(stats_rows)


COLS = ["decode_ms", "detection_ms", "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"]
# decode_ms/detection_ms SENGAJA gak dimasukin ke breakdown per-event_type
# (compute_stats_rows) — dua-duanya jalan tiap siklus TERLEPAS dari ada event
# apa nggak (lihat Bagian 0 di laporan), jadi ngerata-ratain mereka dari subset
# frame yang kebetulan masuk rentang 29 event itu BUKAN angka yang lebih akurat,
# malah sample lebih kecil (~1.5% dari total) & berpotensi bias. Angka yang
# bener buat 2 kolom ini ada di Bagian 0 (seluruh 70k+ frame).
STATS_COLS = ["tracking_ms", "reid_ms_real", "matching_ms_real", "total_ai_ms"]
# tahap yang di-skip kalau nilainya 0 (frame kosong / gak ada tracklet nutup —
# lihat batch_processor.py:374-408) — 0 di sini artinya "gak kepake", bukan
# "kerja instan", jadi dibuang dari itungan biar avg gak kedilusi.
SKIP_ZERO = {"tracking_ms"}


def compute_stats_rows(report_rows: list[dict]) -> list[tuple]:
    """min/max/avg/n per komponen latency, dipisah per event_type
    (person_detected vs zone_entry). `reid_ms_real`/`matching_ms_real` cuma
    ngitung yang BENERAN kerja (lolos REID_REAL_MS/MATCH_REAL_MS threshold,
    lihat Bagian 0) — bukan sekadar nonzero, biar konsisten sama funnel global.
    Return list of (event_type, component, n, avg, min, max)."""
    idx = {c: COLS.index(c) for c in ["tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"]}
    by_type: dict[str, dict[str, list[float]]] = {}
    for r in report_rows:
        bucket = by_type.setdefault(r["event_type"], {c: [] for c in STATS_COLS})
        for _, _, vals in r["_matched_frames"]:
            trk, reid, match, total = (vals[idx["tracking_ms"]], vals[idx["reid_ms"]],
                                        vals[idx["matching_ms"]], vals[idx["total_ai_ms"]])
            if trk > 0:
                bucket["tracking_ms"].append(trk)
            if reid >= REID_REAL_MS:
                bucket["reid_ms_real"].append(reid)
            if match > MATCH_REAL_MS:
                bucket["matching_ms_real"].append(match)
            bucket["total_ai_ms"].append(total)

    out = []
    for etype, bucket in by_type.items():
        for c in STATS_COLS:
            vals = bucket[c]
            if not vals:
                out.append((etype, c, 0, None, None, None))
            else:
                out.append((etype, c, len(vals), sum(vals) / len(vals), min(vals), max(vals)))
    return out


def write_stats_csv(stats_rows: list[tuple]) -> None:
    out_path = AI_SERVICE_DIR / "latency_stats.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "component", "n", "avg_ms", "min_ms", "max_ms"])
        for etype, c, n, avg, mn, mx in stats_rows:
            if n == 0:
                w.writerow([etype, c, 0, "", "", ""])
            else:
                w.writerow([etype, c, n, round(avg, 3), round(mn, 3), round(mx, 3)])
    print(f"Ditulis ke {out_path}")


if __name__ == "__main__":
    main()
