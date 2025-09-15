def test_import_and_forward():
    import torch

    from src.models.multitask_model import MultiTaskModel

    model = MultiTaskModel()
    x = torch.randn(2, 3, 256, 256)
    seg, cls = model(x)
    assert seg.shape[0] == 2 and seg.shape[1] == 1
    assert cls.shape[0] == 2 and cls.shape[1] == 1
