"""Jalankan beberapa kombinasi lepas (tanpa nyentuh COMBOS/markdown utama di
run_comparison.py) DENGAN rekaman klip ber-bbox aktif (skip_recording=False,
tapi tetap skip_gallery_restore=True — terisolasi, gak nyentuh DB/gallery
asli). Tulis hasilnya ke SATU .md: tabel metrik utama (1 baris per kombinasi)
+ tabel akuntansi tracklet per kombinasi (lolos/gagal gate, lengkap deteksi &
sample tiap tracklet), parsing dari log server.

Jalankan dari ai-service/: python benchmark/run_single.py
Klip ber-bbox hasilnya ada di output/clips/ (CLIP_BBOX_OVERLAY=1 di .env).
"""
import re
from pathlib import Path

import run_comparison as rc

TRACKER_TYPE = "bytetrack"
TRACKER_LABEL = "ByteTrack"
REID_MODEL   = "transreid"
REID_LABEL   = "TransReID (ViT-B/16*)"

COMBOS = [
    (60, "yolo26n.pt", "YOLO26n"),
    (61, "yolo26x.pt", "YOLO26x"),
]

OUT_MD = Path("benchmark/yolo26n-vs-yolo26x-bytetrack-transreid.md")

METRIC_HEADER = ("| Detektor   | Tracker   | Re-ID                 | Total frame | "
                  "Cakupan deteksi | Identitas terbentuk | Precision | Recall | F1    | "
                  "Akurasi ketat | False merge | False split | "
                  "FPS efektif | CPU peak | RAM peak | Decode ms | Detect ms | Track ms | "
                  "ReID ms | Match ms | Total AI ms |")
METRIC_SEP = ("| ---------- | --------- | --------------------- | ----------- | "
              "--------------- | ------------------- | --------- | ------ | ----- | "
              "------------- | ----------- | ----------- | "
              "----------- | -------- | -------- | --------- | --------- | -------- | "
              "------- | -------- | ----------- |")

_LINE_RE = re.compile(
    r"\[reid\] [\d:.]+ (?P<cam>\w+)/t(?P<tid>\d+) tracklet ditutup "
    r"\((?P<det>\d+) det, (?P<sample>\d+) sample(?:, koh=[\d.]+)?\) → "
    r"(?P<outcome>NEW|MATCH|BUANG)(?: '(?P<name>[^']+)')? *\((?P<detail>[^)]*)\)"
)
_FOLD_RE = re.compile(
    r"\[reid\] fragmen (?P<cam>\w+)/t(?P<tid>\d+) \((?P<det>\d+) det\) "
    r"dilipat ke tracklet terbuka t(?P<target>\d+)"
)


def parse_tracklets(log_text: str) -> list[dict]:
    rows = []
    for m in _LINE_RE.finditer(log_text):
        rows.append({
            "cam": m.group("cam"), "tid": m.group("tid"),
            "det": m.group("det"), "sample": m.group("sample"),
            "outcome": m.group("outcome"), "name": m.group("name") or "",
            "detail": m.group("detail"),
        })
    for m in _FOLD_RE.finditer(log_text):
        rows.append({
            "cam": m.group("cam"), "tid": m.group("tid"),
            "det": m.group("det"), "sample": "",
            "outcome": "FOLD", "name": f"→ t{m.group('target')}",
            "detail": "fragmen dilipat ke tracklet lain",
        })
    return rows


def render_tracklet_table(rows: list[dict]) -> str:
    lolos  = [r for r in rows if r["outcome"] in ("NEW", "MATCH")]
    gagal  = [r for r in rows if r["outcome"] == "BUANG"]
    folded = [r for r in rows if r["outcome"] == "FOLD"]

    lines = [
        f"Total tracklet ditutup: {len(rows)} — lolos gate (NEW+MATCH): {len(lolos)}, "
        f"gagal gate (BUANG): {len(gagal)}, dilipat ke tracklet lain (FOLD): {len(folded)}.",
        "",
        "| Kamera | Track ID | Deteksi | Sample | Hasil | Identitas / keterangan |",
        "| ------ | -------- | ------- | ------ | ----- | ----------------------- |",
    ]
    for r in rows:
        ident = r["name"] if r["outcome"] in ("NEW", "MATCH", "FOLD") else r["detail"]
        lines.append(
            f"| {r['cam']} | t{r['tid']} | {r['det']} | {r['sample']} | "
            f"{r['outcome']} | {ident} |"
        )
    return "\n".join(lines)


def run_combo(idx: int, detector_model: str, det_label: str) -> tuple[str, str]:
    print(f"\n{'='*70}\n#{idx}: {det_label} + {TRACKER_LABEL} + {REID_LABEL} (record=True)\n{'='*70}")
    res = rc.run_one(idx, detector_model, TRACKER_TYPE, REID_MODEL, record=True)
    metrics = rc.parse_metrics(res["stdout"])
    for key in ("fps", "cpu_peak", "ram_peak", "decode_ms", "detection_ms",
                "tracking_ms", "reid_ms", "matching_ms", "total_ai_ms"):
        metrics[key] = res.get(key, "")
    print(f"  -> {metrics}")

    total_frame = ""
    pred_path = Path(res["pred_path"])
    if pred_path.exists():
        total_frame = str(sum(1 for _ in open(pred_path)) - 1)

    log_text = (rc.LOG_DIR / f"server_{idx}.log").read_text(errors="ignore")
    tracklet_rows = parse_tracklets(log_text)

    metric_row = ("| {det:<10} | {trk:<9} | {reid:<22} | {tf:<11} | "
                  "{cakupan:<15} | {ident:<19} | {prec:<9} | {rec:<6} | {f1:<5} | "
                  "{ak:<13} | {fm:<11} | {fs:<11} | "
                  "{fps:<11} | {cpu:<8} | {ram:<8} | {dec:<9} | {det_ms:<9} | {trk_ms:<8} | "
                  "{reid_ms:<7} | {match:<8} | {total:<11} |").format(
        det=det_label, trk=TRACKER_LABEL, reid=REID_LABEL,
        tf=total_frame, cakupan=metrics.get("cakupan", ""), ident=metrics.get("identitas", ""),
        prec=metrics.get("precision", ""), rec=metrics.get("recall", ""), f1=metrics.get("f1", ""),
        ak=metrics.get("akurasi", ""),
        fm=metrics.get("false_merge", ""), fs=metrics.get("false_split", ""),
        fps=metrics.get("fps", ""), cpu=metrics.get("cpu_peak", ""), ram=metrics.get("ram_peak", ""),
        dec=metrics.get("decode_ms", ""), det_ms=metrics.get("detection_ms", ""),
        trk_ms=metrics.get("tracking_ms", ""), reid_ms=metrics.get("reid_ms", ""),
        match=metrics.get("matching_ms", ""), total=metrics.get("total_ai_ms", ""),
    )

    section = (
        f"### {det_label} + {TRACKER_LABEL} + {REID_LABEL}\n\n"
        + render_tracklet_table(tracklet_rows) + "\n"
    )
    return metric_row, section


def main() -> None:
    rc.PRED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    rc.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rc.LATENCY_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows: list[str] = []
    sections: list[str] = []
    for idx, detector_model, det_label in COMBOS:
        row, section = run_combo(idx, detector_model, det_label)
        metric_rows.append(row)
        sections.append(section)

    clip_note = (
        "Klip rekaman (dengan overlay bbox, `CLIP_BBOX_OVERLAY=1`) ada di "
        "`output/clips/clip_*_sim_*.mp4` untuk kamera-kamera `sample/` (`c8_sim`..`c12_sim`) "
        "selama run ini — cek modified-time-nya biar tau file mana yang dari run ini."
    )

    OUT_MD.write_text(
        "# YOLO26n vs YOLO26x (ByteTrack + TransReID)\n\n"
        + METRIC_HEADER + "\n" + METRIC_SEP + "\n" + "\n".join(metric_rows) + "\n\n"
        + "## Akuntansi Tracklet\n\n"
        + "\n\n".join(sections) + "\n\n"
        + "## Klip\n\n" + clip_note + "\n"
    )
    print(f"\nDitulis ke {OUT_MD}")


if __name__ == "__main__":
    main()
