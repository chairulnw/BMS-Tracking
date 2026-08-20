"""Self-check untuk asosiasi identitas level-tracklet (Fase 2).
Tanpa framework — jalankan langsung: python test_association.py

Lihat plan/07-fase2-detail.md §8 untuk daftar kasus yang wajib ada.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from app.services.pipeline_service import (
    ASSOC_THRESHOLD,
    Tracklet,
    IdentityDB,
)


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def _tl(cam: str, track_id: int, start: datetime, end: datetime, emb: np.ndarray) -> Tracklet:
    tl = Tracklet(cam_id=cam, track_id=track_id, started_at=start, last_seen=end)
    tl.samples = [(emb, 500.0)]
    tl.n_det = 5
    return tl


def test_below_threshold_returns_none():
    """§8 kasus 1 — tracklet dengan embedding acak terhadap gallery berisi satu
    orang → associate() mengembalikan (None, score), tidak dipaksakan match."""
    db = IdentityDB(reid_threshold=ASSOC_THRESHOLD)
    t0 = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)

    gallery_emb = _unit(1)
    db._embeddings["Unknown #1"] = [{"emb": gallery_emb, "last_match": t0, "cam_id": "c1"}]
    db._last_interval["Unknown #1"] = (t0, t0 + timedelta(seconds=5))
    db._last_cam["Unknown #1"] = "c1"

    # Embedding acak, jauh dari gallery_emb, tracklet lama sesudahnya (dt besar
    # → f_time menuju 0) sehingga skor gabungan pasti < threshold.
    query_emb = _unit(999)
    query_tl  = _tl("c2", 42, t0 + timedelta(hours=2), t0 + timedelta(hours=2, seconds=5), query_emb)

    name, score = db.associate(query_emb, query_tl)
    assert name is None, f"expected no match, got {name!r} (score={score})"
    assert score < db.threshold, f"score {score} should be below threshold {db.threshold}"
    print(f"  ok: below-threshold tracklet → (None, {score:.3f})")


def test_overlapping_tracklets_never_merge():
    """§8 kasus 2 — dua tracklet dengan interval waktu beririsan, walau
    embedding identik, tidak pernah dianggap orang yang sama."""
    db = IdentityDB(reid_threshold=ASSOC_THRESHOLD)
    t0 = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)

    emb = _unit(7)
    db._embeddings["Unknown #1"] = [{"emb": emb, "last_match": t0, "cam_id": "c1"}]
    # Interval "Unknown #1": 09:00:00 - 09:00:10 di kamera c1
    db._last_interval["Unknown #1"] = (t0, t0 + timedelta(seconds=10))
    db._last_cam["Unknown #1"] = "c1"

    # Tracklet KEDUA di kamera LAIN dengan embedding SAMA PERSIS, tapi interval
    # waktunya beririsan (09:00:05 - 09:00:15) — dua orang tidak mungkin sama
    # persis embeddingnya DAN hidup bersamaan, jadi ini harus ditolak.
    overlapping_tl = _tl("c2", 99, t0 + timedelta(seconds=5), t0 + timedelta(seconds=15), emb)

    name, score = db.associate(emb, overlapping_tl)
    assert name is None, f"tracklet beririsan waktu seharusnya tidak pernah digabung, tapi match ke {name!r}"
    print(f"  ok: tracklet beririsan waktu (embedding identik) tetap tidak digabung (score={score:.3f} dibuang)")


def test_empty_samples_discarded():
    """§8 kasus 3 — tracklet tanpa sample (semua box gagal quality gate)
    dibuang, tidak ada POST, tidak ada identitas baru."""
    db = IdentityDB(reid_threshold=ASSOC_THRESHOLD)
    t0 = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)

    empty_tl = Tracklet(cam_id="c1", track_id=1, started_at=t0, last_seen=t0 + timedelta(seconds=3))
    assert empty_tl.samples == []

    result = db._resolve_tracklet(empty_tl)
    assert result is None, "tracklet tanpa sample harus dibuang (None), bukan menghasilkan identitas"
    assert db._count == 0, "tidak boleh ada label baru dipakai untuk tracklet yang dibuang"
    print("  ok: tracklet tanpa sample → dibuang, tidak ada identitas baru")


def test_count_restored_after_gallery_reload():
    """§8 kasus 4 — _count pulih setelah reload gallery, supaya label baru
    tidak menabrak label yang sudah dipakai (mis. restart di tengah hari)."""
    db = IdentityDB(reid_threshold=ASSOC_THRESHOLD)
    t0 = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
    today = datetime.now().strftime("%Y%m%d")

    gallery_entries = [
        {
            "person_id": 5, "person_label": f"Unknown #1@c1@{today}",
            "camera_id": "c1", "started_at": t0, "ended_at": t0 + timedelta(seconds=5),
            "embedding": _unit(1).tolist(),
        },
        {
            "person_id": 8, "person_label": f"Unknown #4@c1@{today}",
            "camera_id": "c1", "started_at": t0, "ended_at": t0 + timedelta(seconds=5),
            "embedding": _unit(4).tolist(),
        },
    ]
    db.load_gallery(gallery_entries)
    assert db._count == 4, f"_count harus jadi 4 (nomor tertinggi di gallery), dapat {db._count}"

    # Tracklet baru yang tidak match siapa pun harus dapat nomor #5, bukan #1
    # (yang akan menabrak label yang sudah ada di persons kalau _count tidak dipulihkan).
    new_tl = _tl("c1", 200, t0 + timedelta(minutes=1), t0 + timedelta(minutes=1, seconds=5), _unit(999))
    result = db._resolve_tracklet(new_tl)
    assert result is not None
    assert result["display_name"] == "Unknown #5", f"expected Unknown #5, got {result['display_name']!r}"
    print(f"  ok: _count dipulihkan ke 4, tracklet baru dapat {result['display_name']!r} (bukan tabrakan)")


def test_reversed_resolve_order_still_matches():
    """Regresi: tracklet yang mulai lebih awal dari sighting terakhir gallery,
    tapi baru SELESAI DIPROSES belakangan (urutan resolve lintas kamera bisa
    meleset dari urutan kejadian aslinya — lihat plan/08-pipeline-flow.md §9b),
    dt jadi negatif walau interval-nya TIDAK beririsan. Sebelum fix, _f_time
    mengembalikan 0.0 untuk dt<0 dan menjatuhkan skor di bawah threshold
    walau kemiripan visual tinggi (kasus nyata: Unknown #1 vs #6, 2026-08-20)."""
    db = IdentityDB(reid_threshold=ASSOC_THRESHOLD, w_reid=0.70, w_time=0.15, w_cam=0.15)
    t0 = datetime(2026, 8, 20, 22, 42, tzinfo=timezone.utc)

    emb = _unit(3)
    # "Unknown #1": terlihat 22:42:18-24, SUDAH di-gallery (resolve duluan).
    db._embeddings["Unknown #1"] = [{"emb": emb, "last_match": t0, "cam_id": "c8"}]
    db._last_interval["Unknown #1"] = (t0 + timedelta(seconds=18), t0 + timedelta(seconds=24))
    db._last_cam["Unknown #1"] = "c8"
    db._camera_group = {"c8": 4, "c9": 4}   # satu grup kamera → p_cam=0.8

    # Tracklet baru: video-time-nya LEBIH AWAL (22:42:02-17) — TIDAK beririsan
    # dengan Unknown #1 (17 < 18) — tapi baru diproses/resolve SEKARANG,
    # setelah Unknown #1. Embedding sama (orang yang sama secara visual).
    query_tl = _tl("c9", 7, t0 + timedelta(seconds=2), t0 + timedelta(seconds=17), emb)

    name, score = db.associate(emb, query_tl)
    assert name == "Unknown #1", f"harusnya match ke Unknown #1 (dt<0 tapi non-overlap), dapat {name!r} (score={score:.3f})"
    assert score >= db.threshold, f"score {score:.3f} harus lolos threshold {db.threshold}"
    print(f"  ok: dt<0 non-overlap tetap match ({name!r}, score={score:.3f})")


if __name__ == "__main__":
    tests = [
        test_below_threshold_returns_none,
        test_overlapping_tracklets_never_merge,
        test_empty_samples_discarded,
        test_count_restored_after_gallery_reload,
        test_reversed_resolve_order_still_matches,
    ]
    for t in tests:
        print(f"{t.__name__} ...")
        t()
    print(f"\n{len(tests)}/{len(tests)} test lulus.")
