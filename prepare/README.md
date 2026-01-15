# prepare

用于搭建 CAMELYON 风格 patch 分类流水线的独立工具集。
这些文件目前未接入 Hydra 配置，主要用于检查与迭代开发。

文件说明：
- `prepare/wsi_reader.py`：WSI 读取封装（OpenSlide + PIL 回退）。
- `prepare/xml_annotations.py`：解析 ASAP XML 多边形标注，返回 level-0 坐标。
- `prepare/tissue_mask.py`：生成低分辨率 tissue mask 并计算覆盖率。
- `prepare/patch_labeling.py`：计算 patch 与肿瘤重叠比例并映射为标签。
- `prepare/manifest_builder.py`：扫描 WSI、采样 patch、写出 manifest CSV。
- `prepare/patch_dataset.py`：根据 manifest 读取 patch 的 PyTorch Dataset。

示例（在仓库根目录运行）：
```bash
python -m prepare.manifest_builder ^
  --slides-dir data/raw/wsi ^
  --annotations-dir data/raw/annotations ^
  --output-csv data/processed/patch_manifest.csv ^
  --level 0 ^
  --patch-size 256 ^
  --stride 256 ^
  --pos-threshold 0.5 ^
  --neg-threshold 0.0
```

备注：
- manifest 的 `x`/`y` 使用 level-0 坐标，方便 OpenSlide 直接读取。
- `--groups` 默认是 `Tumor`，请按你的标注分组名称调整。
- 配置接入被刻意留空，准备好后再加入 Hydra wiring。
