import torch.nn as nn
import torch
from clip import clip
from models.vit import *
from config import argument_parser

_args = argument_parser().parse_args()


class TransformerClassifier(nn.Module):
    def __init__(self, clip_model, attr_num, attributes, dim=768,
                 pretrain_path=None):
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
        self.visual_embed = nn.Linear(clip_model.visual.output_dim, dim)
        # RAP1 checkpoint has no visual_embed weights — initialize as identity so
        # visual features are not corrupted when the checkpoint is loaded with strict=False.
        nn.init.eye_(self.visual_embed.weight)
        nn.init.zeros_(self.visual_embed.bias)
        vit = vit_base()
        if pretrain_path and __import__("pathlib").Path(pretrain_path).exists():
            vit.load_param(pretrain_path)
        self.norm = vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
        self.dim = dim
        # Register as buffer so model.to(device) moves it automatically
        self.register_buffer("text", clip.tokenize(attributes))
        self.bn = nn.BatchNorm1d(self.attr_num)
        # RAP1 checkpoint uses MM-Former (1 ViT block) — hardcoded from checkpoint keys
        self.blocks = vit.blocks[-_args.mm_layers:]

    def forward(self, imgs, clip_model):
        b_s = imgs.shape[0]
        clip_image_features, all_class, attenmap = clip_model.visual(imgs.type(clip_model.dtype))
        # text is a buffer — lives on the same device as the model parameters
        text_features = clip_model.encode_text(self.text).float()
        # forward_aggregate used for training loss only; skip at inference
        final_similarity = None
        textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
        x = torch.cat([textual_features, self.visual_embed(clip_image_features.float())], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)
        return bn_logits, final_similarity
