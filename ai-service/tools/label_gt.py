"""Alat bantu labeling ground truth semi-otomatis untuk klip di GT_DIR.

Box orang dideteksi otomatis pakai YOLO (model produksi, yolo26n.pt) —
kamu tinggal kasih person ID per box yang kedeteksi, atau tolak box yang
salah deteksi. Jalankan langsung (butuh display, nggak bisa headless):

    python label_gt.py

Alur per frame yang ditampilkan (tiap ~SAMPLE_INTERVAL_SEC detik, dihitung
ulang per klip dari FPS aslinya masing-masing — FPS klip di sample/
bervariasi dari 5 sampai 36, jadi step tetap dalam jumlah frame bakal bikin
densitas label beda-beda antar klip):
    ENTER  -> terima semua box, lanjut diminta person ID satu-satu di terminal
              (kosong = tolak box itu, u = undo baris terakhir yang ditulis)
    n      -> skip frame ini (nggak ada box yang match), lanjut ke frame berikutnya
    m      -> gambar box manual tambahan (buat box YOLO yang salah/kurang,
              mis. 1 box nutupin 2 orang) — bisa dipencet berkali-kali,
              box lama tetap ada, box baru ditambahkan di atasnya
    u      -> undo baris terakhir yang ditulis (bisa dipencet berkali-kali)
    q      -> berhenti, keluar dari klip yang lagi jalan

Tiap box yang sudah dikasih ID langsung ditulis (append + flush) ke OUT_CSV,
jadi aman kalau tiba-tiba berhenti di tengah jalan.

local_frame dihitung dari SETIAP cap.read() (bukan cuma yang ditampilkan),
supaya penomorannya identik dengan yang dipakai app/services/stream_service
/cam_slot.py:_capture_loop_files saat klip ini nanti diproses AI Service.

# ponytail: undo cuma nyimpen riwayat baris yang ditulis SELAMA skrip ini
# jalan (di RAM) — keluar/restart skrip berarti riwayat undo-nya ilang,
# walau baris di CSV-nya sendiri tetap ada.
# ponytail: nulis ulang ke OUT_CSV yang sama tanpa cek duplikat lintas
# sesi — kalau satu klip dilabel dua kali di run yang beda, barisnya
# numpuk. Hapus dulu baris lama (atau OUT_CSV-nya) kalau mau re-label
# satu klip dari awal.
"""

import csv
import sys
import termios
from pathlib import Path

import cv2
from ultralytics import YOLO

GT_DIR    = Path("sample")
OUT_CSV   = GT_DIR / "output.csv"
YOLO_PATH = "checkpoints/yolo26n.pt"
SAMPLE_INTERVAL_SEC = 1.0   # jarak waktu antar frame yang ditampilkan buat dilabel
CONF      = 0.4   # ambang confidence YOLO — agak longgar, manusia tetap kurasi

HEADER = ["frame", "source_clip", "local_frame", "camera",
          "person", "x", "y", "w", "h"]


class _GTWriter:
    """Bungkus file+writer OUT_CSV, plus riwayat undo (di RAM, per sesi)."""

    def __init__(self, path: Path) -> None:
        # Cek isi header-nya sendiri, bukan cuma "file udah ada" — kalau proses
        # sebelumnya sempat mati sebelum header ke-flush, file bisa "ada" tapi
        # kosong/tanpa header, dan run berikutnya bakal salah kira header udah
        # ditulis lalu langsung nulis data dari baris pertama.
        if not path.exists() or path.stat().st_size == 0:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(HEADER)
        else:
            with open(path, newline="") as f:
                first_row = next(csv.reader(f), None)
            if first_row != HEADER:
                with open(path, newline="") as f:
                    rest = f.read()
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(HEADER)
                    f.write(rest)
                print(f"  [perbaikan] header hilang di {path}, sudah disisipkan ulang.")

        self._f = open(path, "a", newline="")
        self._writer = csv.writer(self._f)
        self.frame_counter = sum(1 for _ in open(path)) - 1
        self._undo_stack: list[tuple[int, int, str]] = []  # (offset, counter_sebelum, deskripsi)

    def write_row(self, clip_name: str, local_frame: int, camera: str,
                  person: str, x: int, y: int, w: int, h: int) -> None:
        offset = self._f.tell()
        prev_counter = self.frame_counter
        self.frame_counter += 1
        self._writer.writerow([self.frame_counter, clip_name, local_frame, camera,
                                person, x, y, w, h])
        self._f.flush()
        self._undo_stack.append((offset, prev_counter, f"{clip_name} frame {local_frame} person={person}"))

    def undo(self) -> str | None:
        if not self._undo_stack:
            return None
        offset, prev_counter, desc = self._undo_stack.pop()
        self._f.seek(offset)
        self._f.truncate()
        self._f.flush()
        self.frame_counter = prev_counter
        return desc

    def close(self) -> None:
        self._f.close()


def _flush_stdin() -> None:
    """Buang keystroke nyasar yang kebawa dari jendela cv2 (beda fokus
    keyboard antara jendela gambar dan terminal), biar input() nggak
    langsung "makan" ketikan lama sebagai jawaban kosong."""
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        pass


def camera_from_folder(folder: str) -> str:
    return folder.removesuffix("_sim")


def already_labeled_clips(path: Path) -> set[str]:
    """source_clip yang udah punya minimal 1 baris di OUT_CSV — dipakai buat
    skip klip yang udah dikerjakan kalau skrip di-restart.

    # ponytail: klip yang keburu 'q' di tengah jalan (baru sebagian
    # frame-nya kelabel) ikut ke-skip penuh juga, bukan cuma sisa
    # frame-nya — trade-off biar resume-nya simpel (per-klip, bukan
    # per-frame). Kalau itu kejadian, hapus baris klip itu dari CSV
    # dulu baru jalanin ulang biar diulang dari awal klip tersebut."""
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {row["source_clip"] for row in csv.DictReader(f)}


def detect_boxes(model: YOLO, frame) -> list[tuple[int, int, int, int]]:
    """Deteksi orang di frame, kembalikan list (x, y, w, h) integer."""
    result = model.predict(frame, classes=[0], conf=CONF, verbose=False)[0]
    boxes = []
    for xyxy in result.boxes.xyxy.tolist():
        x1, y1, x2, y2 = xyxy
        boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return boxes


def _handle_undo(gt: _GTWriter) -> None:
    desc = gt.undo()
    print(f"  [undo] dihapus: {desc}" if desc else "  [undo] nggak ada yang bisa di-undo")


def label_clip(path: Path, camera: str, model: YOLO, gt: _GTWriter) -> bool:
    """Return False kalau user tekan 'q' (berhenti total)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"gagal buka {path}")
        return True

    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * SAMPLE_INTERVAL_SEC))

    clip_name = path.name
    local_frame = 0
    print(f"\n=== {clip_name} ({camera}) — fps={fps:.1f}, step={step} frame ===")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if local_frame % step == 0:
            boxes = detect_boxes(model, frame)
            while True:
                display = frame.copy()
                for i, (x, y, w, h) in enumerate(boxes):
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, str(i), (x, max(0, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"frame {local_frame}  [ENTER]=label  [n]=skip  [m]=box manual  [u]=undo  [q]=stop",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.imshow("label_gt", display)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow("label_gt")

                if key == ord("q"):
                    cap.release()
                    return False
                elif key == ord("u"):
                    _handle_undo(gt)
                    continue   # tampilkan ulang frame yang sama
                elif key == ord("m"):
                    # box otomatis YOLO kurang/salah (mis. 1 box nutupin 2 orang) —
                    # gambar 1 box tambahan manual di atas box yang udah ada.
                    # selectROI (tunggal, bukan selectROIs) sengaja dipakai: drag
                    # kotak lalu ENTER langsung selesai. selectROIs (jamak) minta
                    # ESC buat keluar dari mode "gambar lagi?" — gampang bikin
                    # kejebak loop kalau nggak tau harus ESC bukan ENTER.
                    x, y, w, h = cv2.selectROI("label_gt", display, showCrosshair=True)
                    cv2.destroyWindow("label_gt")
                    if w > 0 and h > 0:
                        boxes.append((int(x), int(y), int(w), int(h)))
                    continue   # tampilkan ulang frame dengan box baru ikut kehitung
                elif key == ord("n") or not boxes:
                    break
                else:
                    i = 0
                    while i < len(boxes):
                        x, y, w, h = boxes[i]
                        _flush_stdin()
                        person = input(f"  box {i} ({x},{y},{w},{h}) -> person ID "
                                        f"(kosong=tolak, u=undo): ").strip()
                        if person.lower() == "u":
                            _handle_undo(gt)
                            continue   # ulang box yang sama
                        if person:
                            gt.write_row(clip_name, local_frame, camera, person, x, y, w, h)
                        i += 1
                    break

        local_frame += 1

    cap.release()
    return True


def main() -> None:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    done = already_labeled_clips(OUT_CSV)
    gt = _GTWriter(OUT_CSV)

    print(f"loading {YOLO_PATH}…")
    model = YOLO(YOLO_PATH)
    if done:
        print(f"{len(done)} klip udah punya label, di-skip: {sorted(done)}")

    clip_dirs = sorted(p for p in GT_DIR.iterdir() if p.is_dir())
    for cam_dir in clip_dirs:
        camera = camera_from_folder(cam_dir.name)
        for clip in sorted(cam_dir.glob("*.avi")):
            if clip.name in done:
                continue
            if not label_clip(clip, camera, model, gt):
                gt.close()
                print(f"\nDihentikan. Hasil tersimpan di {OUT_CSV}")
                return

    gt.close()
    cv2.destroyAllWindows()
    print(f"\nSelesai. Hasil tersimpan di {OUT_CSV}")


if __name__ == "__main__":
    main()
