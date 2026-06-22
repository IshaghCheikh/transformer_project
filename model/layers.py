"""Transformer encoder and decoder blocks."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .modules import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        x = self.norm1(
            src + self.dropout(self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask))
        )
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.norm1(
            tgt
            + self.dropout(
                self.self_attn(
                    tgt,
                    tgt,
                    tgt,
                    key_padding_mask=tgt_key_padding_mask,
                    attn_mask=tgt_mask,
                )
            )
        )
        x = self.norm2(
            x
            + self.dropout(
                self.cross_attn(
                    x,
                    memory,
                    memory,
                    key_padding_mask=memory_key_padding_mask,
                )
            )
        )
        return self.norm3(x + self.dropout(self.ffn(x)))

