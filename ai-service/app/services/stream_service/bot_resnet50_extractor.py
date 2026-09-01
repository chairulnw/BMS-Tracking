"""Wrapper BoT (Bag of Tricks) ResNet50 — Luo et al., "Bag of Tricks and a
Strong Baseline for Deep Person Re-Identification", CVPRW 2019 (arXiv:1903.07071).
Checkpoint resmi Market1501 (Rank-1 94.5%), diambil dari mirror komunitas
(link resmi di README michuanhaohao/reid-strong-baseline sudah dead/500 —
lihat issue #151 repo itu) karena upload asli terpecah jadi banyak zip kecil.

__call__(list[crop RGB uint8]) -> tensor embedding [N, 2048], kompatibel
dipanggil kayak torchreid.utils.FeatureExtractor.

Arsitektur (modeling/baseline.py repo asli): ResNet50 standar (persis
torchvision, tanpa vendor kode) + last_stride=1 di layer4 + BNNeck
(BatchNorm1d) sebelum classifier. Output eval mode = fitur SETELAH BNNeck
(NECK_FEAT='after' di config resmi), bukan classifier (dibuang, gak
dibutuhkan buat ekstraksi embedding).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

_CKPT_PATH = Path(__file__).resolve().parents[3] / "checkpoints" / "bot_resnet50_market1501.pth"
_IMG_SIZE  = (256, 128)   # (H, W) — INPUT.SIZE_TEST config resmi
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # ImageNet, sesuai INPUT.PIXEL_MEAN/STD
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BotResNet50Extractor:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.backbone = torchvision.models.resnet50(weights=None)
        # last_stride=1 (bukan default 2) — trik BoT biar feature map akhir
        # lebih rapat spasial, standar buat person Re-ID resolusi rendah.
        self.backbone.layer4[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        self.bottleneck = torch.nn.BatchNorm1d(2048)

        if _CKPT_PATH.exists():
            sd = torch.load(str(_CKPT_PATH), map_location="cpu")
            base_sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
            self.backbone.load_state_dict(base_sd, strict=False)   # fc.* sengaja missing, gak dipakai
            bn_sd = {k[len("bottleneck."):]: v for k, v in sd.items() if k.startswith("bottleneck.")}
            self.bottleneck.load_state_dict(bn_sd)
        else:
            print(f"[bot-resnet50] checkpoint {_CKPT_PATH} tidak ada, pakai bobot random init")

        self.backbone.eval().to(self.device)
        self.bottleneck.eval().to(self.device)

    def __call__(self, crops: list[np.ndarray]) -> torch.Tensor:
        batch = torch.stack([self._preprocess(c) for c in crops]).to(self.device)
        with torch.no_grad():
            feat = self.backbone.conv1(batch)
            feat = self.backbone.bn1(feat)
            feat = self.backbone.relu(feat)
            feat = self.backbone.maxpool(feat)
            feat = self.backbone.layer1(feat)
            feat = self.backbone.layer2(feat)
            feat = self.backbone.layer3(feat)
            feat = self.backbone.layer4(feat)
            feat = self.backbone.avgpool(feat).flatten(1)   # global avg pool, (N, 2048)
            feat = self.bottleneck(feat)                    # BNNeck, NECK_FEAT='after'
        return F.normalize(feat, dim=1)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = F.interpolate(t, size=_IMG_SIZE, mode="bilinear", align_corners=False)
        t = (t - _MEAN) / _STD
        return t[0]
