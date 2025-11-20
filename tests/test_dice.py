import torch

from src.utils.metrics import dice_coefficient


def test_dice_trivial():
    a = torch.tensor([[[[1, 0], [0, 1]]]]).float()
    b = torch.tensor([[[[1, 0], [0, 1]]]]).float()
    assert torch.isclose(dice_coefficient(a, b), torch.tensor(1.0)).all()
