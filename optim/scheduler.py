"""Noam learning rate schedule."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def noam_scheduler(optimizer: Optimizer, d_model: int, warmup_steps: int = 4000) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

