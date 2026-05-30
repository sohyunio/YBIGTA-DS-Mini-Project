"""
BLIP 세 가지 학습 목적함수:

1. ITC (Image-Text Contrastive): CLIP과 동일한 InfoNCE
2. ITM (Image-Text Matching): 매칭/비매칭 이진 분류
3. LM  (Language Modeling): 다음 토큰 예측 (이미지 캡셔닝)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ITCLoss(nn.Module):
    """
    ITC: 정규화된 image/text CLS 특징 사이의 InfoNCE loss.
    momentum encoder + queue는 생략 (구조 이해 목적).
    """

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        sim_i2t = image_feat @ text_feat.T / temp  # (B, B)
        sim_t2i = text_feat @ image_feat.T / temp  # (B, B)

        B = image_feat.shape[0]
        labels = torch.arange(B, device=image_feat.device)

        return (F.cross_entropy(sim_i2t, labels) + F.cross_entropy(sim_t2i, labels)) / 2


class ITMLoss(nn.Module):
    """
    ITM: multimodal encoder의 CLS 출력 → 2-class 분류 (matched=1, unmatched=0).
    실제 논문에서는 hard negative mining을 사용하지만 여기서는 단순화.
    """

    def forward(self, pooled_output: torch.Tensor, itm_head: nn.Linear, labels: torch.Tensor) -> torch.Tensor:
        logits = itm_head(pooled_output)  # (B, 2)
        return F.cross_entropy(logits, labels)


class LMLoss(nn.Module):
    """
    LM: teacher-forcing 방식의 next-token prediction.
    ignore_index=-100 으로 패딩 위치 무시.
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: (B, L, vocab_size), labels: (B, L)
        # 위치 i에서 i+1 토큰을 예측
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
