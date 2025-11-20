"""用于无真实病理数据场景的合成数据集（快速冒烟测试）。

- 生成随机图像 + 简单规则的伪标签，确保训练/验证管线能端到端跑通；
- 接入真实数据时可替换为 `PathologyDataset` 或自定义 Dataset，接口保持一致；
- 维持轻量依赖，便于在 CPU/GPU 环境下快速验证代码逻辑。

设计意义：
    - 真实病理数据获取与预处理通常耗时且受隐私限制，合成数据集可以让新人快速熟悉
      训练流程，不必等待数据准备完成；
    - 通过保持 ``__getitem__`` 的输出格式与真实数据一致（image, mask, label），后续
      替换数据集时无需改动训练/推理引擎代码。
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class DummyPathologyDataset(Dataset):
    """生成随机病理 patch、伪分割掩码与分类标签。

    参数解释：
        length: 数据集中样本数量，决定每个 epoch 的迭代次数；
        image_size: 输出方形 patch 的边长；
        num_classes: 分类头类别数（1 代表二分类的单输出 logit）；
        seed: 固定随机种子，保证不同机器上的合成数据一致可复现。
    """

    def __init__(self, length: int = 32, image_size: int = 256, num_classes: int = 1, seed: int = 42):
        super().__init__()
        random.seed(seed)
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) 生成伪造的 RGB 病理 patch：用随机噪声代替颜色纹理
        h = w = self.image_size
        image = torch.rand(3, h, w)
        mask = torch.zeros(1, h, w)

        # 2) 构造简单的伪“肿瘤”区域：随机圆形，模拟局部病灶且确保分割标签非空
        center_x, center_y = random.randint(h // 4, 3 * h // 4), random.randint(w // 4, 3 * w // 4)
        radius = random.randint(h // 8, h // 6)
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        mask_circle = ((xs - center_x) ** 2 + (ys - center_y) ** 2) < radius**2
        mask[0] = mask_circle.float()

        if self.num_classes == 1:
            # 3) 二分类：根据伪掩码面积生成 0/1 标签，保证分类头与分割头对齐
            label = torch.tensor(float(mask_circle.float().mean() > 0.1)).unsqueeze(0)
        else:
            # 4) 多分类：随机类别，主要用于结构验证；真实场景替换为 CSV 读取即可
            label = torch.tensor(random.randint(0, self.num_classes - 1))
        return image, mask, label
