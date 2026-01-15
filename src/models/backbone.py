"""多任务病理模型的 ResNet 编码器封装。 

- 目标：在不修改 torchvision 官方实现的前提下，额外返回 U-Net 解码器所需的多尺度特征；
- 痛点：原生 ResNet 只输出最高层特征，分割解码器缺少 skip connection 输入；
- 意义：通过轻量包装，既能复用 ImageNet 预训练权重，也能在离线环境下默认不下载权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Union

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

ResNetName = Literal["resnet18", "resnet34", "mobilenet_v2"]
AttentionType = Literal["none", "se", "cbam"]


@dataclass
class EncoderOutput:
    """编码器前向输出的统一容器。

    features: 最深层特征图，供分类头与解码器入口使用；
    skips:     自顶向下的中间特征列表，供 U-Net 解码器做特征拼接。"""

    features: torch.Tensor
    skips: List[torch.Tensor]

    @property
    def channels(self) -> int:
        return self.features.shape[1]


class ResNetEncoder(nn.Module):
    """返回 skip connection 的 ResNet 编码器。

    参数：
        name: 选择 ResNet18/34，兼顾轻量与表现；
        pretrained: 是否加载 ImageNet 预训练，默认为 False 以便完全离线运行。
        attention: 编码器注意力类型（none/se/cbam），用于方案 A 的通道或通道+空间增强。
        attention_reduction: 注意力降维比例，默认 16。
    """

    def __init__(
        self,
        name: ResNetName = "resnet18",
        pretrained: bool = False,
        attention: AttentionType = "none",
        attention_reduction: int = 16,
    ):
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet34_Weights.DEFAULT if name == "resnet34" else ResNet18_Weights.DEFAULT

        if name == "resnet34":
            backbone = resnet34(weights=weights)
            self.feature_dims = [64, 64, 128, 256, 512]
        else:
            backbone = resnet18(weights=weights)
            self.feature_dims = [64, 64, 128, 256, 512]

        # Stem：保持原始输入的低层纹理，供最后一级 skip 拼接，减少信息损失
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool

        # Encoder blocks：逐层提取语义，channel 维度递增、空间分辨率递减
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # 编码器注意力（方案 A）：按 stage 输出通道初始化注意力模块
        self.attention = attention
        self.attn_x0 = self._build_attention(64, attention, attention_reduction)
        self.attn_x2 = self._build_attention(64, attention, attention_reduction)
        self.attn_x3 = self._build_attention(128, attention, attention_reduction)
        self.attn_x4 = self._build_attention(256, attention, attention_reduction)
        self.attn_x5 = self._build_attention(512, attention, attention_reduction)

    @staticmethod
    def _build_attention(channels: int, attention: AttentionType, reduction: int):
        """按配置创建注意力模块（None/SE/CBAM）。"""
        if attention == "se":
            from optional_modules.attention_modules.se_block import SEBlock

            return SEBlock(channels, reduction=reduction)
        if attention == "cbam":
            from optional_modules.attention_modules.cbam_block import CBAMBlock

            return CBAMBlock(channels, reduction=reduction)
        return None

    def _apply_attention(self, x: torch.Tensor, attn: nn.Module | None) -> torch.Tensor:
        """统一注意力前向，兼容 SE/CBAM 的不同输出形式。"""
        if attn is None:
            return x
        if self.attention == "se":
            return attn(x) * x
        return attn(x)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        # 逐层前向：保留每一级输出，兼顾分辨率与语义丰富度
        x0 = self.stem(x)  # [B, 64, H/2, W/2] 低层纹理，利于精细分割
        x0 = self._apply_attention(x0, self.attn_x0)
        x1 = self.pool(x0)  # [B, 64, H/4, W/4]
        x2 = self.layer1(x1)  # [B, 64, H/4, W/4]
        x2 = self._apply_attention(x2, self.attn_x2)
        x3 = self.layer2(x2)  # [B, 128, H/8, W/8]
        x3 = self._apply_attention(x3, self.attn_x3)
        x4 = self.layer3(x3)  # [B, 256, H/16, W/16]
        x4 = self._apply_attention(x4, self.attn_x4)
        x5 = self.layer4(x4)  # [B, 512, H/32, W/32] 高层语义，供分类与解码器入口
        x5 = self._apply_attention(x5, self.attn_x5)

        # U-Net 需要从深到浅的 skip 列表，这里按解码顺序组织
        skips: List[torch.Tensor] = [x4, x3, x2, x0]
        return EncoderOutput(features=x5, skips=skips)

    @property
    def out_channels(self) -> int:
        return self.feature_dims[-1]


def get_backbone(
    name: ResNetName = "resnet18",
    pretrained: bool = False,
    mobilenet_width_mult: float = 1.0,
    attention: AttentionType = "none",
    attention_reduction: int = 16,
) -> ResNetEncoder:
    """Factory function to align with legacy API."""

    if name in {"resnet18", "resnet34"}:
        return ResNetEncoder(
            name=name,
            pretrained=pretrained,
            attention=attention,
            attention_reduction=attention_reduction,
        )
    if name == "mobilenet_v2":
        # 轻量化骨干，通过适配器包装以复用 EncoderOutput 接口
        from optional_modules.lightweight_backbones.mobilenet_encoder import MobileNetEncoder

        class MobileNetAdapter(nn.Module):
            def __init__(
                self,
                pretrained: bool = False,
                width_mult: float = 0.4,
                attention: AttentionType = "none",
                attention_reduction: int = 16,
            ):
                super().__init__()
                self.encoder = MobileNetEncoder(pretrained=pretrained, width_mult=width_mult)
                self.feature_dims = self.encoder.feature_dims
                self.attention = attention
                # 对齐 feature_dims：[x0, x2, x3, x4, features]
                self.attn_x0 = ResNetEncoder._build_attention(
                    self.feature_dims[0], attention, attention_reduction
                )
                self.attn_x2 = ResNetEncoder._build_attention(
                    self.feature_dims[1], attention, attention_reduction
                )
                self.attn_x3 = ResNetEncoder._build_attention(
                    self.feature_dims[2], attention, attention_reduction
                )
                self.attn_x4 = ResNetEncoder._build_attention(
                    self.feature_dims[3], attention, attention_reduction
                )
                self.attn_x5 = ResNetEncoder._build_attention(
                    self.feature_dims[4], attention, attention_reduction
                )

            def forward(self, x: torch.Tensor) -> EncoderOutput:
                feats, skips = self.encoder(x)
                # 注意力位置：编码器输出与 skip
                if self.attention == "se":
                    skips = [
                        self.attn_x4(skips[0]) * skips[0],
                        self.attn_x3(skips[1]) * skips[1],
                        self.attn_x2(skips[2]) * skips[2],
                        self.attn_x0(skips[3]) * skips[3],
                    ]
                    feats = self.attn_x5(feats) * feats
                elif self.attention == "cbam":
                    skips = [
                        self.attn_x4(skips[0]),
                        self.attn_x3(skips[1]),
                        self.attn_x2(skips[2]),
                        self.attn_x0(skips[3]),
                    ]
                    feats = self.attn_x5(feats)
                return EncoderOutput(features=feats, skips=skips)

            @property
            def out_channels(self) -> int:
                return self.encoder.out_channels

        return MobileNetAdapter(
            pretrained=pretrained,
            width_mult=mobilenet_width_mult,
            attention=attention,
            attention_reduction=attention_reduction,
        )  # type: ignore[return-value]
    raise ValueError(f"Unsupported backbone: {name}")
