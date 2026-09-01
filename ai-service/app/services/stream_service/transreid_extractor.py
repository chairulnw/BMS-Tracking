"""Wrapper TransReID (ViT-B/16, checkpoint resmi fine-tuned Market1501) biar
bisa dipanggil persis seperti torchreid.utils.FeatureExtractor:
__call__(list[crop RGB uint8]) -> tensor embedding [N, 3840]. Dipakai
batch_processor.py saat REID_MODEL=transreid.

Model dibangun lewat make_model() TransReID/model/make_model.py dengan JPM +
SIE_CAMERA aktif (config configs/Market/vit_transreid_stride.yml — checkpoint
resmi "TransReID*(ViT)" Market1501 dari README repo, mAP 89.0/R1 95.1, BUKAN
bobot ImageNet polos). Output eval-mode = concat(global_feat, 4x local_feat/4)
per make_model.py build_transformer_local.forward(), 768*5=3840 dim.

SIE_CAMERA butuh index kamera Market1501 (0-5) per sample — surveillance feed
kita tidak punya pemetaan itu, jadi cam_label dipatok 0 buat semua crop
(ponytail: sedikit sub-optimal dibanding index kamera asli, tapi SIE cuma
nambah bias posisi kecil, bukan penentu utama; upgrade ke index per-kamera
kalau perlu akurasi lebih presisi).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_AI_SERVICE_DIR = Path(__file__).resolve().parents[3]
_REPO_DIR    = _AI_SERVICE_DIR / "checkpoints" / "TransReID"   # vendored kode arsitektur (config/model/loss)
_CONFIG_PATH = _REPO_DIR / "configs" / "Market" / "vit_transreid_stride.yml"
_CKPT_PATH   = _AI_SERVICE_DIR / "checkpoints" / "vit_transreid_market1501.pth"   # bobot, sejajar checkpoint Re-ID lain
_IMG_SIZE    = (256, 128)   # (H, W) — samain dengan INPUT.SIZE_TEST config
_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)   # PIXEL_MEAN/STD config (bukan ImageNet)
_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

# Nama package top-level TransReID/ (config, model, loss, ...) yang generik dan
# tabrakan sama modul lain di app/ (mis. app/par/clip/model.py juga
# `from config import ...`, lihat par_service.py). sys.modules cache import
# by name, bukan by path, jadi buka sys.path doang nggak cukup — begitu
# `import config` kepanggil sekali, cache-nya nempel biarpun path udah dicabut.
_TRANSREID_TOP_PKGS = {"config", "datasets", "loss", "model", "processor", "solver", "utils"}


class TransReIDExtractor:
    def __init__(self, device: str = "cpu") -> None:
        # Simpan sys.modules entries yang mungkin sudah ke-cache dari tempat lain
        # (mis. PAR sempat load duluan) buat nama-nama yang sama dipakai
        # TransReID, biar bisa dipulihkan persis — bukan cuma dihapus — setelah
        # import kita selesai. Load model sekali di awal proses (bukan tiap
        # request), jadi biaya save/restore ini diabaikan.
        _saved_mods = {k: v for k, v in sys.modules.items()
                       if k in _TRANSREID_TOP_PKGS or any(k.startswith(p + ".") for p in _TRANSREID_TOP_PKGS)}
        for k in _saved_mods:
            del sys.modules[k]

        sys.path.insert(0, str(_REPO_DIR))
        try:
            from config import cfg
            from model import make_model
        finally:
            sys.path.remove(str(_REPO_DIR))
            for k in list(sys.modules):
                if k in _TRANSREID_TOP_PKGS or any(k.startswith(p + ".") for p in _TRANSREID_TOP_PKGS):
                    del sys.modules[k]
            sys.modules.update(_saved_mods)

        cfg.merge_from_file(str(_CONFIG_PATH))
        cfg.merge_from_list(["MODEL.PRETRAIN_CHOICE", "no"])   # load checkpoint market1501 langsung, skip imagenet
        cfg.freeze()

        self.device = torch.device(device)
        # num_class tidak dipakai di eval forward (classifier tidak dipanggil),
        # camera_num=6/view_num=1 sesuai Market1501 (dataset asal checkpoint).
        self.model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
        if _CKPT_PATH.exists():
            sd = torch.load(str(_CKPT_PATH), map_location="cpu")
            for k, v in sd.items():
                self.model.state_dict()[k.replace("module.", "")].copy_(v)
        else:
            print(f"[transreid] checkpoint {_CKPT_PATH} tidak ada, pakai bobot random init "
                  f"(unduh dulu, lihat README TransReID/ — link Google Drive Market1501)")
        self.model.eval().to(self.device)

    def __call__(self, crops: list[np.ndarray]) -> torch.Tensor:
        batch = torch.stack([self._preprocess(c) for c in crops]).to(self.device)
        cam_label = torch.zeros(len(crops), dtype=torch.long, device=self.device)
        with torch.no_grad():
            feat = self.model(batch, cam_label=cam_label, view_label=None)
        return F.normalize(feat, dim=1)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = F.interpolate(t, size=_IMG_SIZE, mode="bilinear", align_corners=False)
        t = (t - _MEAN) / _STD
        return t[0]
