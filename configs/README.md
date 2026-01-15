配置说明（Hydra）
================

目录文件
--------
- `defaults.yaml`：训练/推理默认配置，含数据路径、模型、损失、日志、数据加载等键。
- `model_unet.yaml`：可选模型配置示例，可用作替换/扩展 backbone、解码器参数。

用途与调用
---------
- 训练：`python -m src.engine.train`（Hydra 自动读取 `configs/`），可用 `python -m src.engine.train data.use_dummy=false model.backbone=resnet34` 覆写。
- 推理：`python -m src.engine.infer ...` 同样读取默认配置，可在命令行覆写路径或模型参数。
- 测试评估：`python -m src.engine.eval` 使用 `data.test_*` 路径与 `outputs/best.pt`；可覆写 `eval.ckpt=... data.test_images=...`。
- 脚本：`scripts/run_train.py`、`scripts/run_infer.py` 直接使用这些配置。
- 轻量骨干：`python -m src.engine.train model.backbone=mobilenet_v2 model.pretrained=true` 可启用 MobileNetV2 编码器。

注意力消融启动方式
-----------------
- 方案 A（SE 编码器）：`python -m src.engine.train model.encoder_attention=se model.decoder_attention=none`
- 方案 B（CBAM 解码器）：`python -m src.engine.train model.encoder_attention=none model.decoder_attention=cbam`
- 方案 C（SE 编码器 + CBAM 解码器）：`python -m src.engine.train model.encoder_attention=se model.decoder_attention=cbam`
- D: `python -m src.engine.train model.encoder_attention=se model.decoder_attention=cbam model.decoder_attention_layers=[up4]`

待补充
-----
- 针对轻量骨干、注意力、动态损失的独立配置文件。
- 推理/部署专用配置（滑窗尺寸、TTA、后处理）。
- 多任务 vs 单任务、共享 vs 独立编码器的对照配置集合。
