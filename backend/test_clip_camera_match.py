"""ponytail self-check untuk backend/app/routers/clips.py:

1. _clip_stamp() harus TOLAK file kamera lain yang namanya kebetulan diawali
   camera_id yang lebih pendek (mis. "c10" vs "c10_0910") — bug lama: glob
   "clip_c10_*" ikut nangkep "clip_c10_0910_*.mp4", klip kamera lain
   terputar (salah scene sama sekali).
2. Jendela potong ffmpeg (seg_start/seg_dur) harus di-clamp ke durasi asli
   file (clip_dur) — kalau _find_clip balikin klip FALLBACK (target di luar
   jendela klip itu), seg_start bisa nyeek lewat akhir file dan hasilnya
   klip nyaris kosong (<1 detik). Diuji lewat aritmetika clamp-nya langsung
   (sama persis logic di get_clip), bukan lewat HTTP (butuh DB+file asli).

Jalankan: python test_clip_camera_match.py (gak butuh DB/file asli).
"""

from pathlib import Path

from app.routers.clips import _clip_stamp


def test_rejects_longer_camera_id_prefix():
    f = Path("clip_c10_0910_20260911_062449_836.mp4")
    assert _clip_stamp(f, "c10") is None, "harusnya ditolak, ini klip c10_0910 bukan c10"


def test_accepts_exact_camera_id():
    f = Path("clip_c10_20260918_145055_634.mp4")
    assert _clip_stamp(f, "c10") == "20260918_145055_634"


def test_accepts_matching_longer_camera_id():
    f = Path("clip_c10_0910_20260911_062449_836.mp4")
    assert _clip_stamp(f, "c10_0910") == "20260911_062449_836"


def _clamp(seg_start: float, seg_dur: float, clip_dur: float) -> tuple[float, float]:
    """Logic clamp yang sama persis dengan get_clip() di clips.py."""
    if seg_start >= clip_dur:
        seg_start = max(0.0, clip_dur - 1.0)
    seg_dur = max(1.0, min(seg_dur, clip_dur - seg_start))
    return seg_start, seg_dur


def test_clamp_keeps_segment_inside_short_fallback_clip():
    # Diminta detik 12-20, tapi klip fallback cuma 5 detik durasinya —
    # sebelum fix: ffmpeg nyeek ke detik 12 di file 5 detik → nyaris kosong.
    seg_start, seg_dur = _clamp(seg_start=12.0, seg_dur=8.0, clip_dur=5.0)
    assert seg_start == 4.0, seg_start   # mundur biar muat 1s terakhir
    assert seg_dur == 1.0, seg_dur
    assert seg_start + seg_dur <= 5.0


def test_clamp_noop_when_segment_already_fits():
    seg_start, seg_dur = _clamp(seg_start=1.0, seg_dur=4.0, clip_dur=20.0)
    assert (seg_start, seg_dur) == (1.0, 4.0)


if __name__ == "__main__":
    test_rejects_longer_camera_id_prefix()
    test_accepts_exact_camera_id()
    test_accepts_matching_longer_camera_id()
    test_clamp_keeps_segment_inside_short_fallback_clip()
    test_clamp_noop_when_segment_already_fits()
    print("OK — camera_id gak collide, dan jendela potong selalu muat di durasi file asli.")
