"""
BLIP Multimodal Encoder-Decoder (MED)
BERT 구조 기반으로, 세 가지 모드를 지원:

1. Unimodal Text Encoder (ITC용):
   - encoder_hidden_states=None 으로 호출
   - 일반 self-attention만 사용

2. Multimodal Encoder (ITM용):
   - encoder_hidden_states=image_feat 으로 호출, is_decoder=False
   - self-attention + cross-attention (이미지 조건부)

3. Text Decoder (LM용):
   - encoder_hidden_states=image_feat 으로 호출, is_decoder=True
   - causal self-attention + cross-attention
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    max_position_embeddings: int = 512
    pad_token_id: int = 0
    encoder_width: int = 768  # ViT output dimension (for cross-attention key/value)


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        L = input_ids.shape[1]
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0)
        type_ids = torch.zeros_like(input_ids)

        emb = self.word_embeddings(input_ids) + self.position_embeddings(pos_ids) + self.token_type_embeddings(type_ids)
        return self.dropout(self.norm(emb))


class SelfAttention(nn.Module):
    def __init__(self, config: BertConfig, is_cross: bool = False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        kv_dim = config.encoder_width if is_cross else config.hidden_size
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(kv_dim, config.hidden_size)
        self.v = nn.Linear(kv_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if kv is None:
            kv = hidden
        B, N, _ = hidden.shape
        M = kv.shape[1]

        q = self.q(hidden).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(attn.softmax(dim=-1))

        return (attn @ v).transpose(1, 2).reshape(B, N, -1)


class AttentionWithResidual(nn.Module):
    def __init__(self, config: BertConfig, is_cross: bool = False):
        super().__init__()
        self.attn = SelfAttention(config, is_cross)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden, kv=None, mask=None):
        attn_out = self.dropout(self.out(self.attn(hidden, kv, mask)))
        return self.norm(hidden + attn_out)


class BertLayer(nn.Module):
    """
    단일 BERT 레이어.
    with_cross_attention=True이면 cross-attention 포함 (multimodal encoder/decoder용).
    """

    def __init__(self, config: BertConfig, with_cross_attention: bool = False):
        super().__init__()
        self.self_attn = AttentionWithResidual(config, is_cross=False)
        self.with_cross_attention = with_cross_attention
        if with_cross_attention:
            self.cross_attn = AttentionWithResidual(config, is_cross=True)

        mlp_dim = config.intermediate_size
        self.ffn = nn.Sequential(nn.Linear(config.hidden_size, mlp_dim), nn.GELU())
        self.ffn_out = nn.Linear(mlp_dim, config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_hidden: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.self_attn(hidden, mask=self_mask)

        if self.with_cross_attention and encoder_hidden is not None:
            hidden = self.cross_attn(hidden, kv=encoder_hidden, mask=cross_mask)

        ffn_out = self.ffn_dropout(self.ffn_out(self.ffn(hidden)))
        return self.ffn_norm(hidden + ffn_out)


class BertPredictionHead(nn.Module):
    """LM prediction head: hidden → vocab logits."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(hidden))))


class BertModel(nn.Module):
    """
    BERT-based multimodal encoder-decoder for BLIP.
    세 가지 모드:
      - ITC: encoder_hidden_states=None, is_decoder=False
      - ITM: encoder_hidden_states=image_feat, is_decoder=False
      - LM : encoder_hidden_states=image_feat, is_decoder=True
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList([BertLayer(config, with_cross_attention=True) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh())

    def _get_self_mask(self, attention_mask: torch.Tensor, is_decoder: bool) -> torch.Tensor:
        # attention_mask: (B, L), 1=attend, 0=pad
        ext = (1.0 - attention_mask.float())[:, None, None, :] * -10000.0  # (B, 1, 1, L)

        if is_decoder:
            L = attention_mask.shape[1]
            # causal mask: prevent attending to future positions
            causal = torch.triu(torch.ones(L, L, device=attention_mask.device), diagonal=1) * -10000.0
            ext = ext + causal[None, None]

        return ext  # broadcastable to (B, heads, L, L)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        is_decoder: bool = False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        self_mask = self._get_self_mask(attention_mask, is_decoder)

        hidden = self.embeddings(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, encoder_hidden_states, self_mask)

        pooled = self.pooler(hidden[:, 0])
        return hidden, pooled  # (B, L, hidden_size), (B, hidden_size)
