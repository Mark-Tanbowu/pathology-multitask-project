"""多任务模型：共享编码器 + U-Net 解码器 + 分类头。

- 编码器：ResNet18/34（可切换预训练），同时输出深层特征与 skip；
- 分割支路：U-Net 解码器，保持与输入相同的空间分辨率；
- 分类支路：全局池化 + 全连接，复用最高层特征；
- 前向同时返回分割与分类 logits，供多任务损失计算。

设计动机：
    - 通过共享编码器，避免分割/分类重复计算，提升效率；
    - 双头输出方便做多任务联合训练，缓解单任务数据量不足的过拟合痛点；
    - 结构清晰，后续替换编码器或解码器时无需重写训练逻辑。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import ResNetEncoder, get_backbone
from .classification import ClassificationHead
from .segmentation import UNetDecoder


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone_name: str = "resnet18",
        seg_upsample_to_input: bool = True,  # kept for config compatibility
        encoder_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder: ResNetEncoder = get_backbone(backbone_name, pretrained=encoder_pretrained)
        self.decoder = UNetDecoder(self.encoder.feature_dims)
        self.cls_head = ClassificationHead(self.encoder.out_channels, num_classes=num_classes)
        self.seg_upsample_to_input = seg_upsample_to_input

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_size = x.shape[2:]
        # 编码器：输出最深特征 + 四级 skip，兼顾语义与细节
        enc_out = self.encoder(x)

        # 分割支路：逐级上采样，最终输出与输入同尺度的 mask logits
        seg_logits = self.decoder(enc_out.features, enc_out.skips)
        if self.seg_upsample_to_input:
            seg_logits = torch.nn.functional.interpolate(seg_logits, size=input_size, mode="bilinear", align_corners=False)

        # 分类支路：直接基于最高层特征，全局池化后预测肿瘤/非肿瘤概率
        cls_logits = self.cls_head(enc_out.features)
        return seg_logits, cls_logits
