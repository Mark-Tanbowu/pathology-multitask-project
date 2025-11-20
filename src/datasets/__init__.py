"""Dataset utilities for pathology multitask training."""

from .camelyon_dataset import PathologyDataset
from .dummy_dataset import DummyPathologyDataset

__all__ = ["PathologyDataset", "DummyPathologyDataset"]
