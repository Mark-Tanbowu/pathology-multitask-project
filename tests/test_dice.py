from src.utils.metrics import dice_coefficient
import torch


def test_dice_trivial():
    a = torch.tensor([[1, 0], [0, 1]]).float()
    b = torch.tensor([[1, 0], [0, 1]]).float()
    assert abs(float(dice_coefficient(a, b)) - 1.0) < 1e-6
