"""
CLIP: Contrastive Language-Image Pre-Training
Paper: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)

Core idea:
- Train image encoder + text encoder jointly with contrastive loss
- At inference: compute cosine similarity between image and text embeddings
- Zero-shot classification: compare image to text descriptions of each class
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import VisionTransformer
from .loss import CLIPLoss
from .text_encoder import TextTransformer


@dataclass
class CLIPConfig:
    # Vision encoder (ViT)
    img_size: int = 224
    patch_size: int = 16
    vision_embed_dim: int = 768
    vision_depth: int = 12
    vision_num_heads: int = 12

    # Text encoder (Causal Transformer)
    vocab_size: int = 49408
    context_length: int = 77
    text_embed_dim: int = 512
    text_depth: int = 12
    text_num_heads: int = 8

    # Shared projection dimension
    output_dim: int = 512


# Configs matching official OpenAI CLIP checkpoints
CLIP_CONFIGS: dict[str, CLIPConfig] = {
    "ViT-B/32": CLIPConfig(
        patch_size=32,
        vision_embed_dim=768, vision_depth=12, vision_num_heads=12,
        text_embed_dim=512, text_depth=12, text_num_heads=8,
        output_dim=512,
    ),
    "ViT-B/16": CLIPConfig(
        patch_size=16,
        vision_embed_dim=768, vision_depth=12, vision_num_heads=12,
        text_embed_dim=512, text_depth=12, text_num_heads=8,
        output_dim=512,
    ),
    "ViT-L/14": CLIPConfig(
        patch_size=14,
        vision_embed_dim=1024, vision_depth=24, vision_num_heads=16,
        text_embed_dim=768, text_depth=12, text_num_heads=12,
        output_dim=768,
    ),
}


class CLIP(nn.Module):
    """
    CLIP model.

    Forward pass returns (logits_per_image, logits_per_text):
      - logits_per_image[i, j]: similarity of image i with text j
      - logits_per_text[j, i]: same matrix transposed

    Training: minimize CLIPLoss on these logits.
    Zero-shot: argmax over text logits for a set of candidate descriptions.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config

        self.image_encoder = VisionTransformer(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.vision_embed_dim,
            depth=config.vision_depth,
            num_heads=config.vision_num_heads,
            output_dim=config.output_dim,
        )
        self.text_encoder = TextTransformer(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            embed_dim=config.text_embed_dim,
            depth=config.text_depth,
            num_heads=config.text_num_heads,
            output_dim=config.output_dim,
        )

        # log(1/0.07) = ~2.659; clamped to [0, 4.6] during training per the paper
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.loss_fn = CLIPLoss()

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(image), dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(text), dim=-1)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_feat = self.encode_image(image)  # (B, output_dim)
        text_feat = self.encode_text(text)     # (B, output_dim)

        # Clamp temperature to prevent training instability
        logit_scale = self.logit_scale.exp().clamp(max=100)

        logits_per_image = logit_scale * image_feat @ text_feat.T  # (B, B)
        logits_per_text = logits_per_image.T                        # (B, B)

        return logits_per_image, logits_per_text

    def compute_loss(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        logits_per_image, logits_per_text = self(image, text)
        return self.loss_fn(logits_per_image, logits_per_text)

    @classmethod
    def from_config_name(cls, name: str) -> "CLIP":
        if name not in CLIP_CONFIGS:
            raise ValueError(f"Unknown config '{name}'. Available: {list(CLIP_CONFIGS)}")
        return cls(CLIP_CONFIGS[name])
