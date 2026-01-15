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

import time

import torch
import torch.nn as nn

from .backbone import ResNetEncoder, get_backbone
from .classification import ClassificationHead
from .segmentation import LightUNetDecoder, UNetDecoder


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        backbone_name: str = "resnet18",
        seg_upsample_to_input: bool = True,  # kept for config compatibility
        encoder_pretrained: bool = False,
        enable_seg: bool = True,
        enable_cls: bool = True,
        mobilenet_width_mult: float = 0.4,
        use_light_decoder: bool | None = None,
        attention: str = "none",
        attention_location: str = "decoder",
        encoder_attention: str | None = None,
        decoder_attention: str | None = None,
        attention_reduction: int = 16,
        decoder_attention_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        # 注意力位置控制：
        # 1) 若显式传入 encoder_attention/decoder_attention，则优先使用
        # 2) 否则回退到 attention + attention_location 的旧配置方式
        if encoder_attention is None or decoder_attention is None:
            attention_location = attention_location.lower()
            encoder_attention = attention if attention_location in {"encoder", "both"} else "none"
            decoder_attention = attention if attention_location in {"decoder", "both"} else "none"

        self.encoder: ResNetEncoder = get_backbone(
            backbone_name,
            pretrained=encoder_pretrained,
            mobilenet_width_mult=mobilenet_width_mult,
            attention=encoder_attention or "none",
            attention_reduction=attention_reduction,
        )
        if use_light_decoder is None:
            use_light_decoder = backbone_name == "mobilenet_v2"
        self.decoder = (
            LightUNetDecoder(
                self.encoder.feature_dims,
                attention=decoder_attention or "none",
                attention_reduction=attention_reduction,
                attention_layers=decoder_attention_layers,
            )
            if use_light_decoder
            else UNetDecoder(
                self.encoder.feature_dims,
                attention=decoder_attention or "none",
                attention_reduction=attention_reduction,
                attention_layers=decoder_attention_layers,
            )
        )
        self.cls_head = ClassificationHead(self.encoder.out_channels, num_classes=num_classes)
        self.seg_upsample_to_input = seg_upsample_to_input
        if not enable_seg and not enable_cls:
            raise ValueError("At least one task (segmentation or classification) must be enabled.")
        self.enable_seg = enable_seg
        self.enable_cls = enable_cls

    def forward(
        self, x: torch.Tensor, timings: dict[str, float] | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        input_size = x.shape[2:]
        # 编码器：输出最深特征 + 四级 skip，兼顾语义与细节
        t0 = time.perf_counter()
        enc_out = self.encoder(x)
        t1 = time.perf_counter()

        seg_logits: torch.Tensor | None = None
        cls_logits: torch.Tensor | None = None
        if self.enable_seg:
            # 分割支路：逐级上采样，最终输出与输入同尺度的 mask logits
            t_dec_start = time.perf_counter()
            seg_logits = self.decoder(enc_out.features, enc_out.skips)
            if self.seg_upsample_to_input:
                seg_logits = torch.nn.functional.interpolate(
                    seg_logits, size=input_size, mode="bilinear", align_corners=False
                )
            t_dec_end = time.perf_counter()
        else:
            t_dec_start = t1
            t_dec_end = t1
            seg_logits = None

        if self.enable_cls:
            # 分类支路：直接基于最高层特征，全局池化后预测肿瘤/非肿瘤概率
            t_cls_start = time.perf_counter()
            cls_logits = self.cls_head(enc_out.features)
            t_cls_end = time.perf_counter()
        else:
            t_cls_end = t_dec_end
            t_cls_start = t_dec_end
            cls_logits = None

        if timings is not None:
            timings["encoder"] = timings.get("encoder", 0.0) + (t1 - t0)
            timings["decoder"] = timings.get("decoder", 0.0) + (t_dec_end - t_dec_start)
            timings["cls_head"] = timings.get("cls_head", 0.0) + (t_cls_end - t_cls_start)
        return seg_logits, cls_logits
