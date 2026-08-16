"""
Fake argument_parser for PromptPAR — hard-wired for RAP1 checkpoint inference.
This module is picked up by clip/model.py and models/base_block.py when
app/par/ is on sys.path.
"""
import argparse


def argument_parser():
    return _FixedParser()


class _FixedParser:
    def parse_args(self, args=None):
        return argparse.Namespace(
            # Dataset / architecture
            dataset="RAPV1",
            # Visual prompt: 24 ViT layers, 50 prompts per layer (from prompt_deep shape)
            vis_depth=24,
            vis_prompt=50,
            # Div (horizontal strip) CLS tokens: 4 strips (from part_class_embedding shape)
            use_div=True,
            div_num=4,
            overlap_row=2,
            # Text prompts: 3 per text-transformer layer (from prompt_text_deep shape)
            use_textprompt=True,
            text_prompt=3,
            # MM-Former fusion: 1 ViT block for cross-modal attention
            use_mm_former=True,
            mm_layers=1,
            # Vision mask (not used in this checkpoint)
            use_vismask=False,
            # Guided learning loss — only for training, irrelevant at inference
            use_GL=False,
            # Aggregation gate threshold
            ag_threshold=0.5,
            # Misc training params (unused at inference, but prevent AttributeError)
            batchsize=1, epoch=1, height=224, width=224,
            lr=8e-3, weight_decay=1e-4, smooth_param=0.1,
            clip_lr=4e-3, clip_weight_decay=1e-4,
            train_split="trainval", valid_split="test",
            redirector=False, save_freq=1, checkpoint=False,
            dir=None,
            mmformer_update_parameters=[
                "word_embed", "visual_embed", "weight_layer", "bn", "norm",
            ],
            clip_update_parameters=[
                "prompt_deep", "prompt_text_deep", "part_class_embedding",
                "agg_bn", "softmax_model",
            ],
        )
