"""
BLIP: Bootstrapping Language-Image Pre-training
Paper: "BLIP: Bootstrapping Language-Image Pre-training for Unified
        Vision-Language Understanding and Generation" (Li et al., 2022)

핵심 기여:
1. 단일 모델이 이해(ITC, ITM)와 생성(LM) 모두 처리
2. 세 가지 학습 목적함수를 동시에 최적화
3. CapFilt: noisy web 캡션을 Captioner로 생성 + Filter로 정제 (부트스트래핑)

이 구현:
- CapFilt 부트스트래핑 제외 (학습 인프라 수준)
- 세 가지 목적함수 forward pass 구현에 집중
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import ITCLoss, ITMLoss, LMLoss
from .med import BertConfig, BertModel, BertPredictionHead
from .vit import VisionTransformer


@dataclass
class BLIPConfig:
    # Vision encoder
    img_size: int = 224
    patch_size: int = 16
    vision_embed_dim: int = 768
    vision_depth: int = 12
    vision_num_heads: int = 12

    # Text encoder/decoder (BERT-based)
    vocab_size: int = 30522
    text_hidden_size: int = 768
    text_num_layers: int = 12
    text_num_heads: int = 12
    text_intermediate_size: int = 3072
    max_position_embeddings: int = 512

    # ITC projection dimension
    embed_dim: int = 256


class BLIP(nn.Module):
    """
    BLIP 모델 — 세 가지 학습 목적함수.

    [ITC] Image-Text Contrastive
    - 각 인코더에서 CLS 특징 추출 → 저차원 공간에 투영 → InfoNCE
    - encoder_hidden_states=None (unimodal 텍스트 인코더)

    [ITM] Image-Text Matching
    - 이미지 feature를 cross-attention으로 주입한 multimodal 인코더
    - CLS 출력 → 2-class 분류 (matched / not-matched)

    [LM] Language Modeling
    - causal self-attention + cross-attention(이미지)으로 캡션 생성
    - 다음 토큰 예측 (teacher forcing)
    """

    def __init__(self, config: BLIPConfig):
        super().__init__()
        self.config = config

        bert_cfg = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.text_hidden_size,
            num_hidden_layers=config.text_num_layers,
            num_attention_heads=config.text_num_heads,
            intermediate_size=config.text_intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            encoder_width=config.vision_embed_dim,
        )

        # 이미지 인코더: 전체 패치 토큰 반환
        self.visual_encoder = VisionTransformer(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.vision_embed_dim,
            depth=config.vision_depth,
            num_heads=config.vision_num_heads,
        )

        # 텍스트 인코더/디코더 (ITC/ITM/LM 공유)
        self.text_encoder = BertModel(bert_cfg)

        # ITC 투영 헤드 (embed_dim << hidden_size)
        self.vision_proj = nn.Linear(config.vision_embed_dim, config.embed_dim)
        self.text_proj = nn.Linear(config.text_hidden_size, config.embed_dim)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # ITM 분류 헤드
        self.itm_head = nn.Linear(config.text_hidden_size, 2)

        # LM 예측 헤드
        self.lm_head = BertPredictionHead(bert_cfg)

        self.itc_loss_fn = ITCLoss()
        self.itm_loss_fn = ITMLoss()
        self.lm_loss_fn = LMLoss()

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """(B, N+1, vision_embed_dim) — 모든 패치 토큰 + CLS"""
        return self.visual_encoder(image)

    # ------------------------------------------------------------------
    # Per-objective forward passes
    # ------------------------------------------------------------------

    def forward_itc(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ITC: unimodal 인코더 CLS → 투영 → InfoNCE."""
        image_feat = self.encode_image(image)
        image_embed = F.normalize(self.vision_proj(image_feat[:, 0]), dim=-1)

        # is_decoder=False, encoder_hidden_states=None → self-attention only
        _, text_pooled = self.text_encoder(input_ids, attention_mask)
        text_embed = F.normalize(self.text_proj(text_pooled), dim=-1)

        return self.itc_loss_fn(image_embed, text_embed, self.temp)

    def forward_itm(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """ITM: multimodal 인코더 CLS → 이진 분류."""
        image_feat = self.encode_image(image)

        _, pooled = self.text_encoder(
            input_ids,
            attention_mask,
            encoder_hidden_states=image_feat,
            is_decoder=False,
        )
        return self.itm_loss_fn(pooled, self.itm_head, labels)

    def forward_lm(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """LM: causal decoder + cross-attention → 다음 토큰 예측."""
        image_feat = self.encode_image(image)

        hidden, _ = self.text_encoder(
            input_ids,
            attention_mask,
            encoder_hidden_states=image_feat,
            is_decoder=True,
        )
        logits = self.lm_head(hidden)
        return self.lm_loss_fn(logits, labels)

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        itm_labels: Optional[torch.Tensor] = None,
        lm_labels: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
            total_loss (Tensor)
            losses (dict): {'itc': ..., 'itm': ..., 'lm': ...}
        """
        losses = {}
        losses["itc"] = self.forward_itc(image, input_ids, attention_mask)

        if itm_labels is not None:
            losses["itm"] = self.forward_itm(image, input_ids, attention_mask, itm_labels)

        if lm_labels is not None:
            losses["lm"] = self.forward_lm(image, input_ids, attention_mask, lm_labels)

        return sum(losses.values()), losses
