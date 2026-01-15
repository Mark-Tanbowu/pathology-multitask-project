"""Custom samplers for balanced loading."""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Sequence

from torch.utils.data import Sampler


class StratifiedBatchSampler(Sampler[List[int]]):
    """Yield batches that roughly respect a target positive ratio.

    Each index is used at most once per epoch; when某一类耗尽时，剩余样本会直接填满批次。
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        positive_ratio: float = 0.33,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.labels = [int(l) for l in labels]
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        pos_indices = [i for i, l in enumerate(self.labels) if l == 1]
        neg_indices = [i for i, l in enumerate(self.labels) if l == 0]
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        pos_target = max(1, int(round(self.batch_size * self.positive_ratio)))

        batches: List[List[int]] = []
        pos_ptr = 0
        neg_ptr = 0
        total_len = len(self.labels)

        while pos_ptr < len(pos_indices) or neg_ptr < len(neg_indices):
            batch: List[int] = []
            # fill positives
            remaining_batch = self.batch_size - len(batch)
            pos_take = min(pos_target, len(pos_indices) - pos_ptr, remaining_batch)
            if pos_take > 0:
                batch.extend(pos_indices[pos_ptr : pos_ptr + pos_take])
                pos_ptr += pos_take
            # fill negatives
            remaining_batch = self.batch_size - len(batch)
            neg_take = min(remaining_batch, len(neg_indices) - neg_ptr)
            if neg_take > 0:
                batch.extend(neg_indices[neg_ptr : neg_ptr + neg_take])
                neg_ptr += neg_take

            # fill leftovers with whichever class still has samples
            remaining_batch = self.batch_size - len(batch)
            if remaining_batch > 0:
                # first positives
                extra_pos = min(remaining_batch, len(pos_indices) - pos_ptr)
                if extra_pos > 0:
                    batch.extend(pos_indices[pos_ptr : pos_ptr + extra_pos])
                    pos_ptr += extra_pos
                    remaining_batch -= extra_pos
                if remaining_batch > 0:
                    extra_neg = min(remaining_batch, len(neg_indices) - neg_ptr)
                    if extra_neg > 0:
                        batch.extend(neg_indices[neg_ptr : neg_ptr + extra_neg])
                        neg_ptr += extra_neg

            if len(batch) == self.batch_size or (not self.drop_last and batch):
                rng.shuffle(batch)
                batches.append(batch)

            if len(batches) * self.batch_size >= total_len:
                break

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return (len(self.labels) + self.batch_size - 1) // self.batch_size
