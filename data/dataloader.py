"""Batching utilities and sequence masking."""

from __future__ import annotations

import torch
from torch import Tensor


def create_padding_mask(tokens: Tensor, pad_token_id: int = 0) -> Tensor:
    """Return True where tokens are padding."""
    return tokens.eq(pad_token_id)


def create_causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Return upper-triangular causal mask with True for blocked positions."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

