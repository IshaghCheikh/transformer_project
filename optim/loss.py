"""Loss functions for training."""

from __future__ import annotations

import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = -100) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

    def forward(self, logits, targets):
        return self.loss(logits, targets)

