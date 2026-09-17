"""Self-check: crossing di-buffer sampai tracklet resolve, orphan tetap di-flush."""
import types
from app.services.stream_service import batch_processor as bp

posts = []
class FakeClient:
    @staticmethod
    def post_occupancy_event(*a): posts.append(("occ",) + a)
    @staticmethod
    def post_camera_event(cam, et, cat, **kw): posts.append(("cam", cam, kw.get("person_label")))
bp._BackendClient = FakeClient

# stub BatchProcessor cukup buat method crossing
class DB:
    def label_of(self, tid, cam): return f"prov{tid}"
slot = types.SimpleNamespace(db=DB())
proc = object.__new__(bp.BatchProcessor)
proc._pending_crossings = {}
proc._slots = [slot]

proc._buffer_crossing("c1", 5, 10, "IN", "Lobby", "s.jpg", 100, 200, 1000.0, "prov5")
assert not posts, "IN belum boleh POST sebelum resolve"

# OUT: langsung POST, nggak nunggu
proc._buffer_crossing("c1", 7, 10, "OUT", "Lobby", None, 1, 2, 1500.0, "prov7")
assert posts and posts[0][6] == "prov7" and posts[0][3] == "OUT", posts
posts.clear()

# tracklet resolve → flush pakai label final
proc._flush_crossings("c1", [5], "Budi")
assert posts and posts[0][6] == "Budi" and posts[1][2] == "Budi", posts
posts.clear()

# orphan: IN yang track-nya nggak pernah resolve
proc._buffer_crossing("c1", 9, 10, "IN", "Lobby", None, 1, 2, 2000.0, "prov9")
proc._sweep_orphan_crossings(2000.0 + 5)      # belum cukup tua
assert not posts
proc._sweep_orphan_crossings(2000.0 + 999)    # lewat CROSSING_ORPHAN_AGE
assert posts and posts[0][6] == "prov9", posts   # label terbaik yg ada
assert not proc._pending_crossings
print("OK")
