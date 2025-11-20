"""Simplified GradNorm implementation for dynamic loss weighting.

Usage (in a custom training loop):
    from optional_modules.dynamic_loss.gradnorm import GradNorm
    gn = GradNorm(num_tasks=2, alpha=1.5)
    loss = gn.update_and_weight([seg_loss, cls_loss], model)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GradNorm(nn.Module):
    def __init__(self, num_tasks: int, alpha: float = 1.5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.alpha = alpha
        self.initial_losses = None

    def forward(self, losses: List[torch.Tensor], shared_parameters: torch.Tensor) -> torch.Tensor:
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in losses], device=losses[0].device)
        weighted = sum(w * l for w, l in zip(self.weights, losses))
        weighted.backward(retain_graph=True)

        grads = []
        for w, l in zip(self.weights, losses):
            grad_norm = torch.autograd.grad(l, shared_parameters, retain_graph=True, allow_unused=True)[0]
            grads.append(torch.norm(w * grad_norm))
        grads = torch.stack(grads)

        loss_ratios = torch.tensor([l.item() / i for l, i in zip(losses, self.initial_losses)], device=grads.device)
        target = grads.mean() * (loss_ratios ** self.alpha)
        grad_loss = (grads - target.detach()).abs().sum()
        grad_loss.backward()
        with torch.no_grad():
            self.weights.data = torch.relu(self.weights.data)
            self.weights.data /= self.weights.data.sum()
        return sum(self.weights.detach() * torch.stack(losses))

    def update_and_weight(self, losses: List[torch.Tensor], model: nn.Module) -> torch.Tensor:
        shared_params = next(model.parameters())
        return self.forward(losses, shared_params)
