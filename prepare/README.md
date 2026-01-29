# prepare

用于搭建 CAMELYON 风格 patch 分类流水线的独立工具集。
已提供 `configs/prepare.yaml` 集中管理准备阶段的路径与参数。

文件说明：
- `prepare/wsi_reader.py`：WSI 读取封装（OpenSlide + PIL 回退）。
- `prepare/xml_annotations.py`：解析 ASAP XML 多边形标注，返回 level-0 坐标。
- `prepare/tissue_mask.py`：生成低分辨率 tissue mask 并计算覆盖率。
- `prepare/patch_labeling.py`：计算 patch 与肿瘤重叠比例并映射为标签。
- `prepare/manifest_builder.py`：扫描 WSI、采样 patch、写出 manifest CSV（支持 XML 或 mask 标注）。
- `prepare/patch_dataset.py`：根据 manifest 读取 patch 的 PyTorch Dataset。

示例（在仓库根目录运行）：
```bash
python -m prepare.manifest_builder --config configs/prepare.yaml
```

备注：
- manifest 的 `x`/`y` 使用 level-0 坐标，方便 OpenSlide 直接读取。
- `--groups` 默认是 `Tumor`，请按你的标注分组名称调整。
- `configs/prepare.yaml` 默认接入 `camelyon16/images` 与 `camelyon16/masks`。
- mask 命名规则默认使用 `*_mask.tif`，可通过 `mask_suffix` 调整。
- 若同时存在 XML 与 mask，默认优先使用 mask（`prefer_masks: true`）。
