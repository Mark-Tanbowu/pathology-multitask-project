#构建多任务网络

import torch.nn as nn
#引入编码器构造 分类头 分割头
from .backbone import get_backbone
from .classification import SimpleClsHead
from .segmentation import SimpleSegHead


class MultiTaskModel(nn.Module):
    """共享编码器 + 分割头 + 分类头。"""

    def __init__(
        self,
        num_classes: int = 1,#分类头的输出维度 这个参数传给了classification
        seg_upsample_to_input: bool = True,#是否上采样 这个参数传给了segmentation
        backbone_name: str = "resnet18",
    ):
        super().__init__()
        self.encoder = get_backbone(backbone_name)#调用backbone构建共享编码器
        in_ch = self.encoder.out_channels#读取维度通道数 后面都用这个
        self.seg_head = SimpleSegHead(
            in_channels=in_ch, upsample_to_input=seg_upsample_to_input
        )
        self.cls_head = SimpleClsHead(in_channels=in_ch, num_classes=num_classes)
#选取参数传入分类与分割头
    def forward(self, x):
        input_size = (x.shape[2], x.shape[3])#记录空间尺寸 方便上采样还原
        feats = self.encoder(x)#前向传播通过共享编码器
        seg_logits = self.seg_head(feats, input_size=input_size)#通过分割头得到像素级logits
        cls_logits = self.cls_head(feats)#得到类别级logits
        return seg_logits, cls_logits#返回一个二元分数组
