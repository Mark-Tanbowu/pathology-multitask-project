import torch
from sklearn.metrics import roc_auc_score

def dice_coefficient(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float=1e-6) -> float:
    pred = (pred_mask > 0.5).float()
    num = 2 * (pred * true_mask).sum(dim=(1,2,3))
    den = pred.sum(dim=(1,2,3)) + true_mask.sum(dim=(1,2,3)) + eps
    dice = (num / den).mean().item()
    return float(dice)

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
    return float((preds == labels.long()).float().mean().item())

def auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y = labels.detach().cpu().numpy().ravel()
    try:
        return float(roc_auc_score(y, probs))
    except Exception:
        return float("nan")
