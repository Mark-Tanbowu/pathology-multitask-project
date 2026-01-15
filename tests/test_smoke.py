def test_import_and_forward():
    import torch

    from src.models.multitask_model import MultiTaskModel

    model = MultiTaskModel()
    x = torch.randn(2, 3, 256, 256)
    seg, cls = model(x)
    assert seg.shape[0] == 2 and seg.shape[1] == 1
    assert cls.shape[0] == 2 and cls.shape[1] == 1


def test_gradnorm_smoke():
    import torch

    from src.losses import GradNorm, MultiTaskLoss
    from src.models.multitask_model import MultiTaskModel

    torch.manual_seed(42)
    model = MultiTaskModel()
    loss_fn = MultiTaskLoss(weighting="fixed")
    weighter = GradNorm(num_tasks=2, alpha=1.5)

    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, 2, (2, 1, 64, 64)).float()
    labels = torch.randint(0, 2, (2,))
    seg_logits, cls_logits = model(images)
    task_losses, _, _ = loss_fn.compute_task_losses(seg_logits, masks, cls_logits, labels)

    model.zero_grad(set_to_none=True)
    weighter.zero_grad(set_to_none=True)
    loss = weighter.update_and_weight(task_losses, next(model.encoder.parameters()))
    assert loss.item() > 0
