"""Self-check: rekaman disuplai LANGSUNG oleh BatchProcessor sebagai paket
atomik (frame, has_person, annots) lewat slot.rec_q — bukan dicocokkan
belakangan sama frame dari thread capture yang independen. Ini fix final
setelah 3 pendekatan sebelumnya (exact key match, umur jam dinding, toleransi
jarak frame) gagal karena akar masalahnya: frame_q mode file-playlist TANPA
BATAS, backlog batch loop vs rec_q makin menjauh sepanjang run, bukan cuma di
awal — jadi APAPUN skema pencocokan belakangan pasti gagal di beberapa titik.
Dengan rec_q diisi langsung dari titik box dihitung, gak ada yang perlu
dicocokkan sama sekali: box dan frame SELALU satu paket yang sama.

Jalankan: python test_rec_annot_sync.py
"""

import queue
import threading
import types

from dotenv import load_dotenv
load_dotenv()

from app.services.stream_service.cam_slot import _CamSlot


def _make_slot() -> _CamSlot:
    slot = _CamSlot.__new__(_CamSlot)  # skip __init__ (butuh rtsp_url dkk)
    slot.recorder = types.SimpleNamespace(updates=[])
    slot.recorder.update = lambda frame, has_person, ts, annots: slot.recorder.updates.append((frame, has_person, annots))
    return slot


def test_annots_passed_through_untouched():
    slot = _make_slot()
    rec_q = queue.Queue()
    annots = [(0, 0, 10, 10, "Unknown #1 0.90")]
    rec_q.put(("frame_A", True, annots))
    rec_q.put(None)  # sentinel stop
    slot._recorder_loop(rec_q, threading.Event())
    assert slot.recorder.updates == [("frame_A", True, annots)], slot.recorder.updates


def test_empty_annots_becomes_none():
    slot = _make_slot()
    rec_q = queue.Queue()
    rec_q.put(("frame_B", False, []))  # gak ada orang siklus ini
    rec_q.put(None)
    slot._recorder_loop(rec_q, threading.Event())
    frame, has_person, annots = slot.recorder.updates[0]
    assert annots is None, annots   # [] -> None, bukan list kosong ke _draw()


if __name__ == "__main__":
    test_annots_passed_through_untouched()
    test_empty_annots_becomes_none()
    print("OK — rec_q selalu bawa box yang cocok sama framenya, gak ada pencocokan lagi.")
