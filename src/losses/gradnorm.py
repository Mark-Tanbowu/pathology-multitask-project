"""GradNorm dynamic weighting for multitask losses.

Reference: GradNorm (Chen et al., ICML 2018). This is a simplified implementation
for quick ablation experiments. Note that update_and_weight performs backward
passes internally, so the training loop should NOT call loss.backward() again.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class GradNorm(nn.Module):
    """GradNorm task weighting module.

    - weights are trainable parameters that balance task losses
    - update_and_weight updates weights and returns the weighted loss
    - weight_losses computes a weighted sum without updating weights (eval/log)
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5, eps: float = 1e-8) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError("GradNorm requires at least one task.")
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.register_buffer("initial_losses", torch.ones(num_tasks))
        self._initialized = False

    def _init_losses(self, losses: Iterable[torch.Tensor]) -> None:
        losses_tensor = torch.stack([loss.detach() for loss in losses])
        self.initial_losses.copy_(losses_tensor)
        self._initialized = True

    def weight_losses(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Compute a weighted sum without updating weights."""
        weights = self.weights.detach()
        return torch.sum(weights * torch.stack(losses))

    def update_and_weight(
        self, losses: list[torch.Tensor], shared_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Update weights with GradNorm rules, then return the weighted loss."""
        if len(losses) != self.weights.numel():
            raise ValueError("Number of losses must match GradNorm weights.")
        if not self._initialized:
            self._init_losses(losses)

        # Model update uses detached weights to avoid pushing weights up by loss magnitude.
        loss_stack = torch.stack(losses)
        weighted_loss = torch.sum(self.weights.detach() * loss_stack)
        weighted_loss.backward(retain_graph=True)

        grad_norms = []
        for weight, loss in zip(self.weights, losses):
            grad = torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                grad_norms.append(torch.zeros((), device=loss.device))
            else:
                grad_norms.append(torch.norm(grad.detach() * weight))
        grad_norms = torch.stack(grad_norms)

        loss_ratios = loss_stack.detach() / (self.initial_losses + self.eps)
        target = grad_norms.mean().detach() * (loss_ratios**self.alpha)
        gradnorm_loss = torch.nn.functional.l1_loss(grad_norms, target, reduction="sum")
        gradnorm_loss.backward()

        with torch.no_grad():
            self.weights.data.clamp_(min=self.eps)
            weight_sum = self.weights.data.sum().clamp_min(self.eps)
            self.weights.data.mul_(len(losses) / weight_sum)

        return self.weight_losses(losses)
