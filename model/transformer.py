"""Top-level Transformer model assembly."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .layers import DecoderLayer, EncoderLayer
from .modules import PositionalEncoding, TokenEmbedding


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        x = self.positional_encoding(self.src_embedding(src))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.positional_encoding(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_key_padding_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        decoded = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_projection(decoded)

