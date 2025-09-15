import torch
import torch.nn as nn
from .backbone import get_backbone
from .segmentation import SimpleSegHead
from .classification import SimpleClsHead


class MultiTaskModel(nn.Module):
    """共享编码器 + 分割头 + 分类头。"""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 1,
        seg_upsample_to_input: bool = True,
    ):
        super().__init__()
        self.encoder = get_backbone(backbone_name)
        in_ch = self.encoder.out_channels
        self.seg_head = SimpleSegHead(
            in_channels=in_ch, upsample_to_input=seg_upsample_to_input
        )
        self.cls_head = SimpleClsHead(in_channels=in_ch, num_classes=num_classes)

    def forward(self, x):
        input_size = (x.shape[2], x.shape[3])
        feats = self.encoder(x)
        seg_logits = self.seg_head(feats, input_size=input_size)
        cls_logits = self.cls_head(feats)
        return seg_logits, cls_logits
