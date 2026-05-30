"""
CLIP Contrastive Loss: InfoNCE (Noise Contrastive Estimation)

Given a batch of N (image, text) pairs, the N×N similarity matrix should have
high values on the diagonal (matched pairs) and low values off-diagonal.
Both image→text and text→image directions are averaged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def forward(self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
        B = logits_per_image.shape[0]
        labels = torch.arange(B, device=logits_per_image.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2
