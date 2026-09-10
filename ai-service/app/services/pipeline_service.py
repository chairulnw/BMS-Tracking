"""
Pipeline service — callable version tanpa GUI.
Line coordinates diterima sebagai parameter, bukan dari klik mouse.

Dua IdentityDB hidup di file ini (lihat plan/07-fase2-detail.md):
- `IdentityDB`        — live stream, berbasis tracklet (Fase 2).
- `_LegacyIdentityDB` — dipakai HANYA oleh `process_video()` (analisis file
  offline). Di luar cakupan Fase 2 (lihat 07-fase2-detail.md); logikanya
  sengaja dibiarkan identik dengan versi sebelum Fase 2 supaya alat offline
  ini tidak ikut berubah perilakunya.
"""

import math
import os
import re
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO

from app.schemas import (
    ASSOC_THRESHOLD,
    IdentityRecord,
    ProcessVideoRequest,
    ProcessVideoResponse,
)
from app.services.geometry import cross_side, foot_point_xyxy, is_in_side

BUFFER_FRAMES      = 10
NEAR_LINE_DIST     = 40
MIN_CROP_PX        = 32
MIN_MARGIN         = 0.04   # gap minimum top1-top2 untuk confident match
SEED_MIN_COHERENCE = 0.35   # tracklet yang mau jadi identitas BARU: cosine PAIRWISE
                            # TERENDAH antar sample-nya harus >= ini. Rendah = ada
                            # sample yang jauh dari yang lain (box kadang isi 2 orang /
                            # occlusion berat) → embedding mean degenerate → kalau jadi
                            # bibit jadi "magnet". Angka aktual muncul di log [reid],
                            # tune dari situ.
MAX_BANK_SIZE      = 5      # maks entry per identitas di bank embedding
BANK_MERGE_SIM     = 0.90   # sim >= ini → update entry lama, bukan tambah baru
BANK_EXPAND_MIN    = 0.67   # skor match tracklet < ini → JANGAN tambah prototipe bank baru,
BANK_STALE_HOURS   = 6.0    # entry yang tidak jadi top-match selama N jam → kandidat pruning
QUALITY_MIN_H      = 80     # tinggi crop minimum (px)
QUALITY_MIN_W      = 40     # lebar crop minimum (px)
QUALITY_MIN_RATIO  = 1.8    # min h/w — person portrait aspect ratio
QUALITY_LAP_VAR    = 100.0  # min Laplacian variance — blur gate
SAMPLE_MIN_CONF    = 0.62   # conf YOLO minimum buat sebuah box jadi SAMPLE embedding.

DEBUG_CROPS_DIR = Path("debug_crops")

def _debug_reid() -> bool:
    return os.getenv("DEBUG_REID", "").lower() in ("1", "true")


# ── Tracklet association (Fase 2) ──────────────────────────────────────────────
# Skor asosiasi = cosine similarity murni (dulu ada juga suku waktu-antar-
# kemunculan & transisi-antar-kamera, dihapus — w_reid dipakai 1.0, dua suku
# lain 0, jadi gak pernah nyumbang skor apa pun; lihat git history kalau perlu
# dihidupkan lagi).
# ASSOC_THRESHOLD didefinisikan di app.schemas (satu sumber, dipakai juga sebagai
# default reid_threshold di ProcessVideoRequest/StreamStartRequest).
TRACKLET_GAP_CYCLES    = 15       # siklus batch berturut-turut track hilang → tutup tracklet
TRACKLET_MAX_DURATION  = 600.0    # detik — tutup paksa + buka tracklet baru dengan key sama
TRACKLET_MAX_SAMPLES   = 16       # maks embedding disimpan per tracklet (top-K by quality)
TRACKLET_MAX_POSITIONS = 120      # maks titik kaki disimpan per tracklet (garis lintasan/heatmap)
# Fold fragmen tracker: di RTSP FPS rendah, 1 orang bisa dipecah jadi >1 track_id
# yang overlap waktu di kamera SAMA. Tracklet pendek (n_det <= MAX) yang jadi NEW
# tapi overlap waktu dgn tracklet terbuka lain (lebih panjang) di kamera sama, dan
# embedding-nya tidak jelas beda orang (>= MIN_SIM), dilipat ke tracklet terbuka
# itu — bukan bikin identitas terpisah. Bar SIM longgar: prior spatio-temporal kuat.
CONCURRENT_FRAGMENT_MAX_DET = 12
CONCURRENT_FRAGMENT_MIN_SIM = 0.52
# ponytail: cap keras + FIFO drop titik TERTUA kalau kepenuhan — cukup buat tracklet
# normal (detik-menit). Kalau nanti perlu path presisi untuk tracklet super panjang
# (mendekati TRACKLET_MAX_DURATION), ganti ke downsampling merata bukan FIFO.
# ponytail: top-K by quality, linear scan atas TRACKLET_MAX_SAMPLES entri. Kalau
# nanti butuh keragaman pose (bukan sekadar ketajaman), ganti ke clustering —
# tapi jangan sebelum ada bukti top-K saja tidak cukup.


@dataclass
class Tracklet:
    cam_id:     str
    track_id:   int
    started_at: datetime
    last_seen:  datetime
    n_det:      int                       = 0
    samples:    list                      = field(default_factory=list)  # [(emb, quality)], top-K
    best_conf:  float                     = 0.0
    best_crop:  "np.ndarray | None"       = None
    best_x:     "int | None"              = None  # titik kaki (foot point) sampel best_conf — fallback lama
    best_y:     "int | None"              = None
    positions:  list                      = field(default_factory=list)  # [(x,y), ...] tiap observe(), urut waktu
    missing:    int                       = 0   # siklus batch berturut-turut tanpa track ini
    folded_track_ids: list                = field(default_factory=list)  # track_id fragmen yang dilipat ke sini


def par_attrs(par, crop: "np.ndarray | None") -> "dict | None":
    """PAR (Fase 3) — atribut penampilan + warna baju dari satu crop terbaik.
    Fungsi lepas (bukan method IdentityDB) supaya bisa dipanggil dari thread
    mana pun — dipakai dari worker background _PostQueue (backend_client.py
    post_tracklet), BUKAN dari thread inferensi utama: ~2.3s/crop (didominasi
    CLIP ViT-L/14) akan menahan semua kamera kalau dijalankan di sana.
    None kalau PAR tidak aktif atau crop tidak ada. Skor mentah (sigmoid),
    bukan boolean — threshold bisa diubah belakangan tanpa hitung ulang (ADR-006)."""
    if par is None or crop is None or crop.size == 0:
        return None
    from app.par.par_service import ATTR_NAMES
    probs = par.extract(crop)
    if probs is None:
        return None
    attrs: dict = {name: float(p) for name, p in zip(ATTR_NAMES, probs)}
    upper = par.detect_color_scored(crop, "upper")
    if upper is not None:
        attrs["upper_color"], attrs["upper_color_score"] = upper[0], float(upper[1])
    lower = par.detect_color_scored(crop, "lower")
    if lower is not None:
        attrs["lower_color"], attrs["lower_color_score"] = lower[0], float(lower[1])
    return attrs


def _padded_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop diperbesar dari titik tengah box, minimal 120x240 px. Satu-satunya
    sumber thumbnail tracklet — menggantikan snapshot-per-deteksi dan
    profile-thumbnail terpisah yang ada sebelum Fase 2."""
    fh, fw = frame.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half_w = max((x2 - x1) // 2 + 30, 60)
    half_h = max((y2 - y1) // 2 + 40, 120)
    x1c = max(0, cx - half_w)
    y1c = max(0, cy - half_h)
    x2c = min(fw, cx + half_w)
    y2c = min(fh, cy + half_h)
    return frame[y1c:y2c, x1c:x2c].copy()


class IdentityDB:
    """Identity database berbasis tracklet untuk live stream pipeline.
    Lihat plan/07-fase2-detail.md untuk lifecycle lengkap. Bukan dipakai oleh
    process_video() — itu pakai _LegacyIdentityDB di bawah."""

    def __init__(
        self, reid_threshold: float, camera_id: str = "",
        par_extractor=None,
    ) -> None:
        self.threshold  = reid_threshold  # dipakai sebagai ASSOC_THRESHOLD
        self._camera_id = camera_id

        # Bank: name → list of {"emb": np.ndarray, "last_match": datetime, "cam_id": str}
        self._embeddings:      dict[str, list]                  = {}
        self._display_to_label: dict[str, str]                  = {}
        self._track_to_name:   dict[tuple[str, int], str]       = {}
        self._name_to_owner:   dict[str, tuple[str, int]]       = {}  # dipakai to_records()/identities saja
        self._active_tracks:   dict[str, set[int]]              = {}

        self._open: dict[tuple[str, int], Tracklet] = {}

        self._count = 0
        self._par = par_extractor  # PARExtractor | None — dipakai Fase 3

    def _key(self, track_id: int, cam_id: str) -> tuple[str, int]:
        return (cam_id or self._camera_id, track_id)

    def _new_names(self, cam_id: str = "") -> tuple[str, str]:
        """Returns (display_name, unique_label). Label menyertakan camera_id
        dan tanggal agar unik di seluruh kamera (ADR-001 — tidak menembus hari)."""
        self._count += 1
        today   = datetime.now().strftime("%Y%m%d")
        display = f"Unknown #{self._count}"
        cam     = cam_id or self._camera_id or "cam"
        label   = f"Unknown #{self._count}@{cam}@{today}"
        return display, label

    # ── Observe / close (lifecycle tracklet) ────────────────────────────────

    def observe(
        self, cam_id: str, track_id: int, emb: "np.ndarray | None", quality: float,
        conf: float, frame: "np.ndarray | None", x1: int, y1: int, x2: int, y2: int,
        ts: datetime,
    ) -> None:
        """Kumpulkan bukti untuk tracklet ini. Dipanggil tiap box per siklus
        batch, termasuk box yang gagal quality gate (emb None) atau conf <
        SAMPLE_MIN_CONF — itu tetap menaikkan n_det, last_seen, titik lintasan
        tapi TIDAK menambah sample embedding (conf ganda: 0.45 buat deteksi+
        tracking+rekam, 0.66 buat bukti identitas)."""
        key = self._key(track_id, cam_id)
        tl = self._open.get(key)
        if tl is None:
            tl = Tracklet(cam_id=cam_id or self._camera_id, track_id=track_id,
                           started_at=ts, last_seen=ts)
            self._open[key] = tl
        tl.last_seen = ts
        tl.n_det += 1
        if conf > tl.best_conf and frame is not None:
            tl.best_conf = conf
            tl.best_crop = _padded_crop(frame, x1, y1, x2, y2)
            tl.best_x, tl.best_y = foot_point_xyxy(x1, y1, x2, y2)
        # Titik kaki DISIMPAN TIAP OBSERVE, bukan cuma sampel best_conf — ini
        # yang membuat "garis lintasan" beneran punya beberapa titik untuk
        # disambung, bukan cuma satu titik ringkasan per tracklet.
        if len(tl.positions) >= TRACKLET_MAX_POSITIONS:
            tl.positions.pop(0)
        tl.positions.append(foot_point_xyxy(x1, y1, x2, y2))
        if emb is not None and conf >= SAMPLE_MIN_CONF:
            self._add_sample(tl, emb, quality)

    @staticmethod
    def _add_sample(tl: Tracklet, emb: np.ndarray, quality: float) -> None:
        # Top-K by quality (Laplacian var): kalau penuh, ganti sample terburuk
        # bila yang baru lebih tajam.
        if len(tl.samples) < TRACKLET_MAX_SAMPLES:
            tl.samples.append((emb, quality))
            return
        worst_idx = min(range(len(tl.samples)), key=lambda i: tl.samples[i][1])
        if quality > tl.samples[worst_idx][1]:
            tl.samples[worst_idx] = (emb, quality)

    def update_active(self, cam_id: str, track_ids: "set[int]") -> None:
        """Dipanggil tiap siklus batch. Update tracklet mana yang masih 'hidup'
        (missing=0) dan mana yang mulai hilang (missing += 1)."""
        self._active_tracks[cam_id] = set(track_ids)
        for key, tl in self._open.items():
            if key[0] != cam_id:
                continue
            tl.missing = 0 if tl.track_id in track_ids else tl.missing + 1

    def close_expired(self, cam_id: str, now: "datetime | None" = None) -> list[dict]:
        """Tutup tracklet kamera ini yang track-nya hilang >= TRACKLET_GAP_CYCLES
        siklus, atau yang sudah melebihi TRACKLET_MAX_DURATION (lalu langsung
        buka tracklet baru dengan key sama — track-nya masih hidup)."""
        now = now or datetime.now(timezone.utc)
        closed: list[dict] = []
        for key in [k for k in list(self._open) if k[0] == cam_id]:
            tl = self._open[key]
            gap_expired      = tl.missing >= TRACKLET_GAP_CYCLES
            duration_expired = (now - tl.started_at).total_seconds() > TRACKLET_MAX_DURATION
            if not (gap_expired or duration_expired):
                continue
            del self._open[key]
            result = self._resolve_tracklet(tl)
            if result is not None:
                closed.append(result)
            if duration_expired and not gap_expired:
                self._open[key] = Tracklet(cam_id=tl.cam_id, track_id=tl.track_id,
                                            started_at=now, last_seen=now)
        return closed

    def close_all(self, cam_id: "str | None" = None) -> list[dict]:
        """Flush semua tracklet terbuka — dipakai saat stream stop, reconnect
        kamera, atau reset tengah malam."""
        keys = [k for k in list(self._open) if cam_id is None or k[0] == cam_id]
        closed: list[dict] = []
        for key in keys:
            tl = self._open.pop(key)
            result = self._resolve_tracklet(tl)
            if result is not None:
                closed.append(result)
        return closed

    # ── Asosiasi ─────────────────────────────────────────────────────────────

    def associate(self, emb: np.ndarray, tl: Tracklet) -> tuple["str | None", "str | None", float, "str | None", float]:
        """score = cosine similarity murni terhadap bank embedding tiap orang.
        Return: (match_atau_None, top1_name, top1_score, top2_name, top2_score).
        match None kalau top1 < threshold atau margin top1-top2 < MIN_MARGIN.
        ponytail: hard constraint interval-overlap (dua tracklet beririsan
        waktu di kamera sama tidak pernah dianggap orang yang sama) dicabut
        atas permintaan eksplisit — sekarang murni threshold+margin di bawah,
        gak ada guard lain. Konsekuensi yang sudah didiskusikan: dua orang
        beda yang crossing berdekatan bisa ke-gabung jadi satu identitas."""
        best_name, best_score = None, -1.0
        second_name, second_score = None, -1.0
        for name, bank in self._embeddings.items():
            score = max(float(np.dot(emb, e["emb"])) for e in bank)
            if _debug_reid():
                print(f"[assoc.cmp] {tl.cam_id}/t{tl.track_id} vs {name!r}: cos={score:.3f}")
            if score > best_score:
                second_name, second_score = best_name, best_score
                best_score, best_name = score, name
            elif score > second_score:
                second_name, second_score = name, score

        margin = best_score - second_score
        matched = best_name if (best_name is not None and best_score >= self.threshold
                                and margin >= MIN_MARGIN) else None
        return matched, best_name, best_score, second_name, second_score

    def _tl_mean_emb(self, tl: Tracklet) -> np.ndarray:
        mean = np.mean([e for e, _ in tl.samples], axis=0)
        return mean / (np.linalg.norm(mean) + 1e-8)

    def _fold_fragment_into_open(self, tl: Tracklet, emb: np.ndarray) -> bool:
        """Fragmen tracker (lihat CONCURRENT_FRAGMENT_*): `tl` pendek & overlap
        waktu dgn tracklet terbuka lain di kamera sama yg lebih panjang &
        embedding-nya tidak jelas beda orang → lipat sampel+posisi ke tracklet
        terbuka itu, return True (pemanggil buang `tl` tanpa emit identitas)."""
        if tl.n_det > CONCURRENT_FRAGMENT_MAX_DET:
            return False
        for (cam, _), o in self._open.items():
            if cam != tl.cam_id or o is tl or len(o.samples) <= len(tl.samples):
                continue
            if not (o.started_at <= tl.last_seen and tl.started_at <= o.last_seen):
                continue  # tidak overlap waktu
            if float(np.dot(emb, self._tl_mean_emb(o))) < CONCURRENT_FRAGMENT_MIN_SIM:
                continue  # jelas beda orang
            for s_emb, s_q in tl.samples:
                self._add_sample(o, s_emb, s_q)
            o.n_det += tl.n_det
            o.positions.extend(tl.positions)
            del o.positions[:-TRACKLET_MAX_POSITIONS]
            o.folded_track_ids.append(tl.track_id)
            o.folded_track_ids.extend(tl.folded_track_ids)
            print(f"[reid] fragmen {tl.cam_id}/t{tl.track_id} ({tl.n_det} det) dilipat "
                  f"ke tracklet terbuka t{o.track_id}")
            return True
        return False

    @staticmethod
    def _sample_coherence(tl: Tracklet) -> float:
        """Cosine PAIRWISE terendah antar sample embedding tracklet. Rendah = ada
        sample yang beda jauh dari yang lain → box kadang isi 2 orang / occlusion
        berat → embedding mean degenerate. (mean-cosine-ke-mean punya floor ~0.7
        walau kontaminasi 50/50, pairwise-min rentangnya penuh 0..1.)"""
        E = np.stack([e for e, _ in tl.samples])
        sims = E @ E.T
        n = len(E)
        return float(sims[np.triu_indices(n, k=1)].min()) if n > 1 else 1.0

    def _resolve_tracklet(self, tl: Tracklet) -> "dict | None":
        if len(tl.samples) < 5:
            return None  # < 5 crop berkualitas — bukti visual terlalu tipis, buang (tidak ada POST)

        emb = self._tl_mean_emb(tl)

        name, top1_name, score, top2_name, top2_score = self.associate(emb, tl)
        is_new = name is None
        if is_new and self._fold_fragment_into_open(tl, emb):
            return None
        if is_new:
            coh = self._sample_coherence(tl)
            if coh < SEED_MIN_COHERENCE:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[reid] {ts} {tl.cam_id}/t{tl.track_id} tracklet ditutup "
                      f"({tl.n_det} det, {len(tl.samples)} sample) → BUANG "
                      f"(koherensi sample {coh:.2f} < {SEED_MIN_COHERENCE}, embedding terkontaminasi — cegah magnet)")
                return None
            name, label = self._new_names(tl.cam_id)
            self._embeddings[name]       = [{"emb": emb, "last_match": tl.last_seen, "cam_id": tl.cam_id}]
            self._display_to_label[name] = label
        else:
            self._bank_update(name, emb, tl.cam_id, match_score=score)
            label = self._display_to_label.get(name, name)

        key = self._key(tl.track_id, tl.cam_id)
        self._track_to_name[key]  = name
        self._name_to_owner[name] = key

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        margin = score - top2_score
        # top1/top2 = kandidat terdekat di galeri + skor cosine-nya; margin = selisihnya.
        cand = (f"top1={top1_name!r}:{score:.3f} top2={top2_name!r}:{top2_score:.3f} "
                f"margin={margin:.3f}")
        # score >= threshold tapi tetap NEW berarti ditolak gate margin (MIN_MARGIN),
        # bukan skor kurang — dua kandidat teratas terlalu dekat untuk dipercaya.
        if is_new and score >= self.threshold:
            reason = f"{cand} → margin < {MIN_MARGIN}, tolak"
        elif is_new:
            reason = f"{cand} → top1 < thr={self.threshold} (atau gallery kosong)"
        else:
            reason = cand
        coh_str = f", koh={self._sample_coherence(tl):.2f}" if is_new else ""
        print(f"[reid] {ts} {tl.cam_id}/t{tl.track_id} tracklet ditutup ({tl.n_det} det, {len(tl.samples)} sample{coh_str}) "
              f"→ {'NEW' if is_new else 'MATCH'} {name!r}  ({reason})")

        return {
            "display_name": name,
            "label":        label,
            "is_new":       is_new,
            "cam_id":       tl.cam_id,
            "track_id":     tl.track_id,
            "started_at":   tl.started_at,
            "ended_at":     tl.last_seen,
            "n_detections": tl.n_det,
            "best_crop":    tl.best_crop,
            "best_conf":    tl.best_conf,
            "pos_x":        tl.best_x,
            "pos_y":        tl.best_y,
            "positions":    tl.positions,
            "embedding":    emb,
            "assoc_score":  score,
            "folded_track_ids": tl.folded_track_ids,
            # PAR (Fase 3) TIDAK dihitung di sini — ~2.3s/crop akan menahan
            # thread inferensi utama untuk SEMUA kamera. par_attrs() dipanggil
            # nanti di worker background _PostQueue (backend_client.py).
            "par":          self._par,
        }


    def _bank_update(self, name: str, emb: np.ndarray, cam_id: str, *, match_score: float = 1.0) -> None:
        # Aware UTC — harus konsisten dengan started_at/ended_at yang dipakai
        # associate()/load_gallery(), kalau tidak prune_banks() crash saat
        # membandingkan entry lokal (naive) dengan entry hasil load_gallery (aware).
        bank = self._embeddings.get(name)
        if not bank:
            return
        sims     = [float(np.dot(emb, e["emb"])) for e in bank]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        now = datetime.now(timezone.utc)

        if best_sim >= BANK_MERGE_SIM:
            merged = 0.9 * bank[best_idx]["emb"] + 0.1 * emb
            bank[best_idx]["emb"]        = merged / (np.linalg.norm(merged) + 1e-8)
            bank[best_idx]["last_match"] = now
        elif match_score < BANK_EXPAND_MIN:
            # match tipis → jangan ekspansi bank, cukup jaga entry terdekat tetap fresh
            bank[best_idx]["last_match"] = now
        else:
            new_entry = {"emb": emb, "last_match": now, "cam_id": cam_id}
            if len(bank) >= MAX_BANK_SIZE:
                lru_idx = min(range(len(bank)), key=lambda i: bank[i]["last_match"])
                bank[lru_idx] = new_entry
            else:
                bank.append(new_entry)

    # ── Pemulihan gallery dari DB (§7) ───────────────────────────────────────

    def load_gallery(self, entries: list[dict]) -> None:
        """entries: [{person_id, person_label, camera_id, started_at, ended_at,
        embedding}] — hasil GET /tracklets/gallery, maks 5 entri terbaru/orang
        (sudah dibatasi backend). Dipanggil sekali saat stream/start. Hanya
        tracklet hari ini yang dimuat — konsekuensi ADR-001."""
        max_count = 0
        for e in entries:
            label   = e["person_label"]
            display = label.split("@")[0]
            m = re.match(r"Unknown #(\d+)$", display)
            if m:
                max_count = max(max_count, int(m.group(1)))

            self._display_to_label[display] = label
            bank = self._embeddings.setdefault(display, [])
            emb_arr = np.asarray(e["embedding"], dtype=np.float32)
            ended_at = e["ended_at"]
            bank.append({"emb": emb_arr, "last_match": ended_at, "cam_id": e["camera_id"]})

        self._count = max(self._count, max_count)

    # ── Lookup / maintenance ─────────────────────────────────────────────────

    def label_of(self, track_id: int, cam_id: str = "") -> "str | None":
        key     = self._key(track_id, cam_id)
        display = self._track_to_name.get(key)
        if display is None:
            return None
        return self._display_to_label.get(display, display)

    def name_of(self, track_id: int, cam_id: str = "") -> "str | None":
        return self._track_to_name.get(self._key(track_id, cam_id))

    def prune_banks(self) -> int:
        """Hapus entry stale dari bank embedding (jalankan sebelum reset harian)."""
        removed = 0
        now     = datetime.now(timezone.utc)
        for name, bank in self._embeddings.items():
            if len(bank) <= 1:
                continue
            fresh = [e for e in bank
                     if (now - e["last_match"]).total_seconds() / 3600 < BANK_STALE_HOURS]
            if not fresh:
                fresh = [max(bank, key=lambda e: e["last_match"])]
            removed += len(bank) - len(fresh)
            self._embeddings[name] = fresh
        return removed

    def reset(self) -> None:
        """Reset semua state setiap tengah malam. Panggil close_all() DULU
        (lihat batch_processor.py) supaya tracklet yang masih terbuka tidak
        hilang begitu saja — ini hanya defensive clear."""
        self._open.clear()
        self._embeddings.clear()
        self._track_to_name.clear()
        self._display_to_label.clear()
        self._name_to_owner.clear()
        self._active_tracks.clear()
        self._count = 0

    def to_records(self) -> list[IdentityRecord]:
        records = []
        for name in self._embeddings:
            tids = [k[1] for k, n in self._track_to_name.items() if n == name]
            records.append(IdentityRecord(name=name, track_ids=tids))
        return records


# ── Legacy: dipakai HANYA oleh process_video() (analisis file offline) ────────
# Di luar cakupan Fase 2. Logika identik dengan sebelum Fase 2 — lihat
# plan/07-fase2-detail.md untuk alasan kenapa dipisah dari IdentityDB di atas.

EMBED_REFRESH      = 15
MIN_ENROLL_FRAMES  = 2      # delayed enrollment: tunggu N frame berkualitas


class _LegacyIdentityDB:
    def __init__(self, reid_threshold: float, camera_id: str = "",
                 par_extractor=None) -> None:
        self.threshold           = reid_threshold
        self._camera_id          = camera_id
        self._embeddings:      dict[str, list] = {}
        self._track_to_name:   dict[int, str]        = {}
        self._frame_counter:   dict[int, int]        = {}
        self._display_to_label: dict[str, str]       = {}
        self._pending:         dict                  = {}
        self._count = 0
        self._par = par_extractor
        self._name_to_owner:  dict[str, tuple]      = {}
        self._active_tracks:  dict[str, set[int]]   = {}
        self._last_top2_name: str | None            = None

    def _key(self, track_id: int, cam_id: str) -> tuple[str, int] | int:
        return (cam_id, track_id) if cam_id else track_id

    def _new_names(self, cam_id: str = "") -> tuple[str, str]:
        self._count += 1
        today   = datetime.now().strftime("%Y%m%d")
        display = f"Unknown #{self._count}"
        cam     = cam_id or self._camera_id or "cam"
        label   = f"Unknown #{self._count}@{cam}@{today}"
        return display, label

    def _best_match(
        self, emb: np.ndarray, *, track_id=None, cam_id: str = "",
    ) -> tuple[str | None, float, str | None, float]:
        top1_name, top1_sim = None, -1.0
        top2_name, top2_sim = None, -1.0
        for name, bank in self._embeddings.items():
            best_e = max((float(np.dot(emb, e["emb"])) for e in bank), default=-1.0)
            if best_e > top1_sim:
                top2_sim, top2_name = top1_sim, top1_name
                top1_sim, top1_name = best_e, name
            elif best_e > top2_sim:
                top2_sim, top2_name = best_e, name
        self._last_top2_name = top2_name
        return top1_name, top1_sim, None, top2_sim

    def assign(
        self, track_id: int, emb: np.ndarray, cam_id: str = "",
        *, quality: float = 1.0, debug_crop: "np.ndarray | None" = None,
        active_track_ids: "set[int] | None" = None,
    ) -> tuple[str | None, bool]:
        key = self._key(track_id, cam_id)
        if key in self._track_to_name:
            return self._track_to_name[key], False

        buf = self._pending.setdefault(key, [])
        buf.append((emb, quality))
        if len(buf) < MIN_ENROLL_FRAMES:
            return None, False

        best_entry = max(buf, key=lambda x: x[1])
        best_emb   = best_entry[0]
        del self._pending[key]

        name, sim, _, sim2 = self._best_match(best_emb, track_id=track_id, cam_id=cam_id)
        margin = sim - sim2

        if name is not None and sim >= self.threshold:
            owner = self._name_to_owner.get(name)
            if owner is not None and owner != key:
                owner_cam, owner_tid = owner
                owner_active = active_track_ids or set() if owner_cam == cam_id \
                    else self._active_tracks.get(owner_cam, set())
                if owner_tid in owner_active:
                    name = None

        is_confident = name is not None and sim >= self.threshold and margin >= MIN_MARGIN
        is_ambiguous = name is not None and sim >= self.threshold and margin < MIN_MARGIN

        if is_confident or is_ambiguous:
            self._track_to_name[key]  = name
            self._frame_counter[key]  = 0
            self._name_to_owner[name] = key
            self._bank_update(name, best_emb, cam_id, touch_only=True)
            return name, False

        display, label = self._new_names(cam_id)
        self._embeddings[display]       = [{"emb": best_emb, "last_match": datetime.now(), "cam_id": cam_id}]
        self._display_to_label[display] = label
        self._track_to_name[key]     = display
        self._frame_counter[key]     = 0
        self._name_to_owner[display] = key
        return display, True

    def _bank_update(self, name: str, emb: np.ndarray, cam_id: str, *, touch_only: bool = False) -> None:
        bank = self._embeddings.get(name)
        if not bank:
            return
        sims     = [float(np.dot(emb, e["emb"])) for e in bank]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        now = datetime.now()

        if touch_only or best_sim >= BANK_MERGE_SIM:
            if not touch_only:
                merged = 0.9 * bank[best_idx]["emb"] + 0.1 * emb
                bank[best_idx]["emb"] = merged / (np.linalg.norm(merged) + 1e-8)
            bank[best_idx]["last_match"] = now
        else:
            new_entry = {"emb": emb, "last_match": now, "cam_id": cam_id}
            if len(bank) >= MAX_BANK_SIZE:
                lru_idx = min(range(len(bank)), key=lambda i: bank[i]["last_match"])
                bank[lru_idx] = new_entry
            else:
                bank.append(new_entry)

    def label_of(self, track_id: int, cam_id: str = "") -> str | None:
        key     = self._key(track_id, cam_id)
        display = self._track_to_name.get(key)
        if display is None:
            return None
        return self._display_to_label.get(display, display)

    def refresh(self, track_id: int, emb: np.ndarray, cam_id: str = "") -> None:
        key = self._key(track_id, cam_id)
        cnt = self._frame_counter.get(key, 0) + 1
        self._frame_counter[key] = cnt
        if cnt % EMBED_REFRESH != 0:
            return
        name = self._track_to_name.get(key)
        if name is None or name not in self._embeddings:
            return
        self._bank_update(name, emb, cam_id)

    def update_active(self, cam_id: str, track_ids: "set[int]") -> None:
        self._active_tracks[cam_id] = set(track_ids)

    def name_of(self, track_id: int, cam_id: str = "") -> str | None:
        return self._track_to_name.get(self._key(track_id, cam_id))

    def prune_banks(self) -> int:
        removed = 0
        now     = datetime.now()
        for name, bank in self._embeddings.items():
            if len(bank) <= 1:
                continue
            fresh = [e for e in bank
                     if (now - e["last_match"]).total_seconds() / 3600 < BANK_STALE_HOURS]
            if not fresh:
                fresh = [max(bank, key=lambda e: e["last_match"])]
            removed += len(bank) - len(fresh)
            self._embeddings[name] = fresh
        return removed

    def reset(self) -> None:
        self._embeddings.clear()
        self._track_to_name.clear()
        self._frame_counter.clear()
        self._display_to_label.clear()
        self._pending.clear()
        self._name_to_owner.clear()
        self._active_tracks.clear()
        self._count = 0

    def to_records(self) -> list[IdentityRecord]:
        records = []
        for name in self._embeddings:
            tids = [
                k[1] if isinstance(k, tuple) else k
                for k, n in self._track_to_name.items()
                if n == name
            ]
            records.append(IdentityRecord(name=name, track_ids=tids))
        return records


# ── Geometry ────────────────────────────────────────────────────────────────

def _foot_point(box) -> tuple[int, int]:
    x1, _, x2, y2 = map(int, box.xyxy[0])
    return foot_point_xyxy(x1, 0, x2, y2)


# ── Embedding ─────────────────────────────────────────────────────────────────

def _extract_embedding(
    extractor, frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *, track_id=None, cam_id: str = "",
) -> "tuple[np.ndarray, float] | None":
    """Ekstrak embedding dari crop.
    Returns (emb_normalized, laplacian_var) jika lulus quality gate, else None."""
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    h, w = y2 - y1, x2 - x1

    if h < QUALITY_MIN_H or w < QUALITY_MIN_W:
        return None
    if h / (w + 1e-6) < QUALITY_MIN_RATIO:
        return None
    gray    = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < QUALITY_LAP_VAR:
        return None

    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        feat = extractor([crop])
    emb  = feat[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8), lap_var


# ── Draw ──────────────────────────────────────────────────────────────────────

def _draw_line(frame: np.ndarray, p1: tuple, p2: tuple, in_sign: int) -> None:
    cv2.line(frame, p1, p2, (0, 200, 255), 2)
    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy) or 1
    nx = -dy / length * in_sign
    ny =  dx / length * in_sign
    off = 25
    cv2.putText(frame, "IN",  (int(mx + nx*off), int(my + ny*off)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)
    cv2.putText(frame, "OUT", (int(mx - nx*off), int(my - ny*off)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)


def _draw_tracks(frame: np.ndarray, result, db: "_LegacyIdentityDB", crossed: set[int]) -> None:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, track_id in zip(boxes, boxes.id.int().tolist()):
        if int(box.cls[0]) != 0:
            continue
        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 100, 255) if track_id in crossed else (0, 200, 80)
        label = f"{db.name_of(track_id) or f'ID:{track_id}'}  {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(label, font, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - bl - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - bl - 2), font, 0.6, (0, 0, 0), 2)


def _draw_counter(frame: np.ndarray, count_in: int, count_out: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"In : {count_in}",  (12, 30), font, 0.75, (0, 255, 120), 2)
    cv2.putText(frame, f"Out: {count_out}", (12, 58), font, 0.75, (0, 100, 255), 2)


# ── Public entry point ────────────────────────────────────────────────────────

def process_video(
    req: ProcessVideoRequest,
    detector: YOLO,
    extractor,
) -> ProcessVideoResponse:
    input_path  = Path(req.video_path)
    lp1         = req.line.p1
    lp2         = req.line.p2
    in_sign     = req.line.in_sign

    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    if req.output_path:
        output_path = Path(req.output_path)
    else:
        output_path = Path("output") / f"result_{input_path.stem}_processed.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db          = _LegacyIdentityDB(req.reid_threshold)
    line_length = math.hypot(lp2[0] - lp1[0], lp2[1] - lp1[1]) or 1

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 15
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    side_history:     dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=4))
    track_last_frame: dict[int, int]          = {}
    counted_ids:      set[int]                = set()
    count_in  = 0
    count_out = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=req.conf_threshold,
            classes=[0],
            verbose=False,
        )

        crossed_this_frame: set[int] = set()
        active_ids:         set[int] = set()

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id in zip(boxes, boxes.id.int().tolist()):
                if int(box.cls[0]) != 0:
                    continue

                active_ids.add(track_id)

                x1e, y1e, x2e, y2e = map(int, box.xyxy[0])
                result = _extract_embedding(extractor, frame, x1e, y1e, x2e, y2e)
                if result is not None:
                    emb, quality = result
                    if db.name_of(track_id) is None:
                        db.assign(track_id, emb, quality=quality)
                    else:
                        db.refresh(track_id, emb)

                fx, fy    = _foot_point(box)
                curr_side = cross_side(fx, fy, lp1[0], lp1[1], lp2[0], lp2[1])
                history   = side_history[track_id]

                if len(history) > 0 and track_id not in counted_ids:
                    prev       = history[-1]
                    prev_in    = is_in_side(prev,      in_sign)
                    curr_in    = is_in_side(curr_side, in_sign)
                    if not prev_in and curr_in:
                        count_in += 1
                        counted_ids.add(track_id)
                        crossed_this_frame.add(track_id)
                    elif prev_in and not curr_in:
                        count_out += 1
                        counted_ids.add(track_id)
                        crossed_this_frame.add(track_id)

                history.append(curr_side)
                track_last_frame[track_id] = frame_idx

        for tid, last_f in list(track_last_frame.items()):
            if tid in active_ids or tid in counted_ids:
                continue
            frames_gone = frame_idx - last_f
            if frames_gone < 1 or frames_gone > BUFFER_FRAMES:
                continue
            history = side_history[tid]
            if not history:
                continue
            last_side = history[-1]
            if abs(last_side) / line_length > NEAR_LINE_DIST:
                continue
            earliest = history[0]
            if abs(earliest) <= abs(last_side):
                continue
            if not is_in_side(earliest, in_sign):
                count_in += 1
            else:
                count_out += 1
            counted_ids.add(tid)

        _draw_line(frame, lp1, lp2, in_sign)
        _draw_tracks(frame, results[0], db, crossed_this_frame)
        _draw_counter(frame, count_in, count_out)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return ProcessVideoResponse(
        output_path=str(output_path),
        count_in=count_in,
        count_out=count_out,
        identities=db.to_records(),
        frames_processed=frame_idx,
    )
