"""
CLIP Text Encoder: Causal Transformer
CLIP uses a GPT-style causal transformer for text encoding.
Features are extracted from the EOS token position.
"""
import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)

        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TextTransformer(nn.Module):
    """
    Causal Transformer text encoder for CLIP.

    Key design choices (from the paper):
    - Causal (autoregressive) attention mask — same direction as GPT
    - EOS token aggregates the full sequence representation
    - Projection from text embed_dim to shared output_dim
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        context_length: int = 77,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        output_dim: int = 512,
    ):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty(context_length, embed_dim))

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # proj maps embed_dim -> output_dim; stored as Parameter for weight tying flexibility
        self.proj = nn.Parameter(torch.empty(embed_dim, output_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)
        nn.init.normal_(self.proj, std=self.embed_dim ** -0.5)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask — prevents attending to future tokens."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text: (B, context_length) — integer token ids
        x = self.token_embed(text) + self.pos_embed  # (B, L, embed_dim)

        attn_mask = self._build_causal_mask(x.shape[1], x.device)
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.norm(x)
        # EOS token: position of the highest token id (49407 = <|endoftext|>)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]  # (B, embed_dim)
        x = x @ self.proj  # (B, output_dim)
        return x
