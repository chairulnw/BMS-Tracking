"""Unduh sekali: bobot ImageNet1k ViT-B/16 dari timm, disimpan sebagai
checkpoint lokal yang bisa dibaca TransReID.load_param() (format jax-style
key names, sama seperti checkpoint asli jx_vit_base_p16_224).

Jalankan: python download_transreid_pretrained.py
"""

from pathlib import Path

import timm
import torch

OUT = Path("TransReID/pretrained/vit_base_p16_224_timm.pth")

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    torch.save(m.state_dict(), OUT)
    print(f"saved -> {OUT}")
