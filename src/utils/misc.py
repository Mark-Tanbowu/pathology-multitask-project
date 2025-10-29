"""
misc.py - 常用辅助函数
包括随机种子、设备管理、模型参数统计等。
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """设定随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] 已设置随机种子: {seed}")


def get_device(pref: str = "auto") -> torch.device:
    """根据可用性选择设备"""
    if pref == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"[Device] 使用设备: {device}")
    return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """统计模型参数数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
