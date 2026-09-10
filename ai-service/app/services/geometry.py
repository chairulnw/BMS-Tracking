"""Geometri garis/titik dipakai bareng oleh live pipeline (stream_service) dan
offline pipeline (pipeline_service) — sebelumnya diduplikasi di dua tempat."""


def cross_side(px: int, py: int, lx1: int, ly1: int, lx2: int, ly2: int) -> float:
    """Signed distance dari titik (px,py) ke garis (lx1,ly1)-(lx2,ly2).
    Positif di satu sisi, negatif di sisi lain."""
    return float((lx2 - lx1) * (py - ly1) - (ly2 - ly1) * (px - lx1))


def within_segment_span(px: int, py: int, lx1: int, ly1: int, lx2: int, ly2: int,
                        margin: float = 0.15) -> bool:
    """True kalau proyeksi (px,py) ke garis jatuh di dalam rentang RUAS p1-p2
    (bukan perpanjangannya). `cross_side` pakai garis tak-hingga, jadi orang yang
    lewat jauh di luar ujung garis tetap kena flip sisi → IN/OUT palsu. Ini
    gate-nya. margin: toleransi di luar ujung (0.15 = 15% panjang ruas)."""
    dx, dy = lx2 - lx1, ly2 - ly1
    seg_sq = dx * dx + dy * dy
    if seg_sq == 0:
        return True
    t = ((px - lx1) * dx + (py - ly1) * dy) / seg_sq
    return -margin <= t <= 1.0 + margin


def is_in_side(v: float, in_sign: int) -> bool:
    return v * in_sign >= 0


def foot_point_xyxy(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Titik kaki bounding box — dipakai sebagai titik uji crossing/dwell."""
    return (x1 + x2) // 2, y2
