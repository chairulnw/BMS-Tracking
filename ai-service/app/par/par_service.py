"""
PARExtractor — PromptPAR inference wrapper for single person crop.

Model : CLIP ViT-L/14 + TransformerClassifier (MM-Former, 1 block)
Checkpoint : RAP1.pth  (51 RAP v1 attributes total)
Attributes scored : subset dipilih di SELECTED_ATTRS (lihat N_ATTRS untuk jumlah
persisnya) — nama atribut memakai string asli RAP1_ATTR_WORDS, tidak diterjemahkan.

Usage:
    par = PARExtractor("par_checkpoints/RAP1.pth", device="cpu")
    probs  = par.extract(bgr_crop)          # float32 (N_ATTRS,) probabilities
    binary = par.attribute_vector(bgr_crop) # float32 (N_ATTRS,) binary @ threshold 0.45
    score  = PARExtractor.attribute_similarity(a, b)  # [0,1] match ratio
"""

import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

# ── sys.path injection ─────────────────────────────────────────────────────────
# PromptPAR source files (clip/, models/) use plain `from config import ...`
# and `from models.vit import *` — they require their parent dir on sys.path.
_PAR_DIR = Path(__file__).parent
if str(_PAR_DIR) not in sys.path:
    sys.path.insert(0, str(_PAR_DIR))

# Import PromptPAR modules AFTER sys.path is set so they resolve config.py here
from clip.model import build_model          # noqa: E402
from models.base_block import TransformerClassifier  # noqa: E402

# ── RAP1 attribute manifest ────────────────────────────────────────────────────

# Exact strings used during training (from rap1_pad.py preprocess script)
RAP1_ATTR_WORDS: list[str] = [
    "female", "age less 16", "age 17 30", "age 31 45",
    "body fat", "body normal", "body thin", "customer", "clerk",
    "head bald head", "head long hair", "head black hair",
    "head hat", "head glasses", "head muffler",
    "upper shirt", "upper sweater", "upper vest", "upper t-shirt", "upper cotton",
    "upper jacket", "upper suit up", "upper tight", "upper short sleeve",
    "lower long trousers", "lower skirt", "lower short skirt", "lower dress",
    "lower jeans", "lower tight trousers",
    "shoes leather", "shoes sport", "shoes boots", "shoes cloth", "shoes casual",
    "attach backpack", "attach shoulder bag", "attach hand bag", "attach box",
    "attach plastic bag", "attach paper bag", "attach hand trunk", "attach other",
    "action calling", "action talking", "action gathering", "action holding",
    "action pushing", "action pulling", "action carry arm", "action carry hand",
]

# Atribut yang dipakai proyek ini → (nama, index RAP1). Nama PERSIS string asli
# RAP1_ATTR_WORDS (bukan nama ramah buatan) — supaya tetap tertelusur ke definisi
# aslinya. Dipangkas ke yang benar-benar dipakai UI (gender, topi, kacamata,
# tas) — usia dibuang (konteks kantor, semua dewasa), sisanya (bentuk tubuh,
# rambut, jenis pakaian detail, sepatu, aksi) dibuang karena daya beda rendah
# atau tidak dipakai sebagai kriteria pencarian orang di UI.
SELECTED_ATTRS: list[tuple[str, int]] = [
    ("female",              0),
    ("head hat",           12),
    ("head glasses",       13),
    ("attach backpack",     35),
    ("attach shoulder bag", 36),
    ("attach hand bag",     37),
]
ATTR_NAMES:      list[str] = [n for n, _ in SELECTED_ATTRS]
SELECTED_INDICES: list[int] = [i for _, i in SELECTED_ATTRS]
N_ATTRS = len(SELECTED_INDICES)   # 6

THRESHOLD = 0.6

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ── CLIP color detection ───────────────────────────────────────────────────────

CLIP_COLORS = ["black", "white", "gray", "red", "green", "blue", "brown", "yellow", "purple", "pink"]
_COLOR_PROMPTS = [f"a person wearing {c} clothes" for c in CLIP_COLORS]

_COLOR_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
])


def _load_openai_clip(device: str):
    """Load openai/CLIP ViT-B/32, bypassing PromptPAR's local clip/ package."""
    _saved_mods = {k: v for k, v in sys.modules.items()
                   if k == "clip" or k.startswith("clip.")}
    for k in _saved_mods:
        del sys.modules[k]
    _saved_path = sys.path[:]
    sys.path[:] = [p for p in sys.path if p != str(_PAR_DIR)]
    try:
        import clip as _oai
        model, _ = _oai.load("ViT-B/32", device=device)
        model = model.float().eval() if device != "cuda" else model.eval()
        tokens = _oai.tokenize(_COLOR_PROMPTS)
        dev = torch.device(device)
        with torch.no_grad():
            text_feat = model.encode_text(tokens.to(dev)).float()
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
        return model, text_feat
    finally:
        sys.path[:] = _saved_path
        for k, v in _saved_mods.items():
            sys.modules[k] = v


# ── PARExtractor ──────────────────────────────────────────────────────────────

class PARExtractor:
    """Thread-safe (read-only after init) PAR inference wrapper."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self._device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        clip_model = build_model(ckpt["ViT_model"])
        # fp16 → fp32 when not on CUDA (CPU / MPS don't support all fp16 ops)
        if self._device.type != "cuda":
            clip_model = clip_model.float()
        clip_model = clip_model.to(self._device).eval()
        self._clip = clip_model

        model = TransformerClassifier(clip_model, len(RAP1_ATTR_WORDS), RAP1_ATTR_WORDS)
        sd = {k.replace("vis_embed.", "visual_embed."): v
              for k, v in ckpt["model_state_dict"].items()}
        result = model.load_state_dict(sd, strict=False)
        # Expected missing: 'text' (recomputed from attr words), 'visual_embed.*' (identity init)
        unexpected = [k for k in result.missing_keys
                      if k != "text" and not k.startswith("visual_embed.")]
        if unexpected:
            print(f"[par] unexpected missing keys: {unexpected}")
        model = model.to(self._device).eval()
        self._model = model

        print(f"[par] RAP1.pth loaded → device={device}, attrs={N_ATTRS} selected")

        # ── CLIP color detector ────────────────────────────────────────────────
        try:
            self._clip_color, self._color_text_feat = _load_openai_clip(device)
            print(f"[par] CLIP ViT-B/32 color detector loaded → device={device}")
        except Exception as e:
            print(f"[par] CLIP color detection unavailable: {e}")
            self._clip_color = None
            self._color_text_feat = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def extract(self, crop_bgr: np.ndarray) -> "np.ndarray | None":
        """
        Returns float32 probability array of shape (16,) for the selected
        attributes, or None if the crop is too small to be meaningful.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 16:
            return None

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img_t = _TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits, _ = self._model(img_t, clip_model=self._clip)
            probs = torch.sigmoid(logits).cpu().float().numpy()[0]  # (51,)

        return probs[SELECTED_INDICES].astype(np.float32)  # (16,)

    def attribute_vector(self, crop_bgr: np.ndarray) -> "np.ndarray | None":
        """Binary attribute vector (16,) thresholded at 0.45."""
        probs = self.extract(crop_bgr)
        return None if probs is None else (probs >= THRESHOLD).astype(np.float32)

    def attr_jaccard(self, query_vec: np.ndarray, gal_vec: np.ndarray) -> "tuple[float, str]":
        """Jaccard similarity + two-line debug string for logging."""
        q = query_vec.astype(bool)
        g = gal_vec.astype(bool)
        both   = q & g
        q_only = q & ~g
        g_only = g & ~q
        n_both, n_q, n_g = int(both.sum()), int(q_only.sum()), int(g_only.sum())
        n_union = n_both + n_q + n_g
        jaccard = n_both / n_union if n_union > 0 else 0.0

        parts = []
        if n_q:
            parts.append(f"q+[{','.join(ATTR_NAMES[i] for i in range(N_ATTRS) if q_only[i])}]")
        if n_g:
            parts.append(f"g+[{','.join(ATTR_NAMES[i] for i in range(N_ATTRS) if g_only[i])}]")
        if n_both:
            parts.append(f"both+[{','.join(ATTR_NAMES[i] for i in range(N_ATTRS) if both[i])}]")
        diff_line = " ".join(parts) if parts else "no-active-attrs"
        stat_line = (f"match={n_both} mismatch={n_q + n_g} union={n_union}"
                     f" → {n_both}/{n_union}={jaccard:.3f}")
        return jaccard, f"{diff_line}\n{stat_line}"

    def detect_color_scored(self, crop_bgr: np.ndarray, body_part: str) -> "tuple[str, float] | None":
        """Detect dominant clothing color; returns (color_name, similarity_score) or None."""
        if self._clip_color is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if body_part == "upper":
            y1, y2 = int(h * 0.20), int(h * 0.50)
        elif body_part == "lower":
            if h / max(w, 1) < 2.0:
                return None
            y1, y2 = int(h * 0.55), int(h * 0.90)
        else:
            return None
        region = crop_bgr[y1:y2]
        if region.size == 0 or (y2 - y1) < 10:
            return None
        rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        img_t = _COLOR_TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            img_feat = self._clip_color.encode_image(img_t).float()
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
            sims = (img_feat @ self._color_text_feat.T).squeeze(0)
        best_idx = int(sims.argmax())
        return CLIP_COLORS[best_idx], float(sims[best_idx].cpu())

    def detect_color(self, crop_bgr: np.ndarray, body_part: str) -> "str | None":
        """Detect dominant clothing color; returns color name or None."""
        result = self.detect_color_scored(crop_bgr, body_part)
        return result[0] if result else None

    @staticmethod
    def attribute_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of matching binary attributes in [0, 1]."""
        return float(np.mean(a == b))
