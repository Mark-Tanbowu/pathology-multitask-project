"""Dynamic Weight Averaging (DWA) for multitask loss balancing."""

from __future__ import annotations

from typing import List

import torch


class DynamicWeightAveraging:
    def __init__(self, num_tasks: int, temperature: float = 2.0):
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.history: List[torch.Tensor] = []

    def compute_weights(self, losses: List[torch.Tensor]) -> torch.Tensor:
        self.history.append(torch.tensor([l.item() for l in losses]))
        if len(self.history) < 2:
            return torch.ones(self.num_tasks) / self.num_tasks
        r_t = self.history[-1] / (self.history[-2] + 1e-8)
        weights = torch.exp(r_t / self.temperature)
        weights = weights / weights.sum() * self.num_tasks
        return weights

    def apply(self, losses: List[torch.Tensor]) -> torch.Tensor:
        weights = self.compute_weights(losses).to(losses[0].device)
        return sum(w * l for w, l in zip(weights, losses))
