from PIL import Image
import os

"""
============================================================
病理图像数据集尺寸修复与掩码同步处理工具 (fix_size.py)
------------------------------------------------------------
功能概述：
本脚本用于对数字病理数据集中的图像和对应掩码进行统一尺寸重采样，
以确保输入数据在深度学习训练阶段尺寸一致、格式规范。

主要功能：
1. 自动读取指定数据集（train/val）目录下的图像与掩码：
   - 原始图像目录: data/<split>/images/
   - 掩码目录: data/<split>/masks/
2. 按照给定目标尺寸 (target_size) 进行重采样处理：
   - 图像使用双线性插值 (BILINEAR)
   - 掩码使用最近邻插值 (NEAREST)，保持像素标签不被平滑
3. 自动创建修复后目录结构：
   - data/<split>/images_fixed/
   - data/<split>/masks_fixed/
4. 检查缺失掩码文件并输出统计报告。

使用说明：
------------------------------------------------------------
运行方式：
    python tools/fix_size.py

可选参数：
    - `target_size`：输出图像与掩码的目标分辨率（默认 (512, 512)）
    - `split_name`：数据集划分名称（默认处理 train 和 val）

输出内容：
    - 修复完成的文件数量统计
    - 缺失掩码数量提示
    - 各子集处理完成通知

注意事项：
------------------------------------------------------------
1. 掩码必须以 `_anno.bmp` 命名规则与原图一一对应。
2. 掩码图像必须为灰度图 (单通道)，否则需手动预处理。
3. 使用 NEAREST 插值可防止掩码边界标签混合，是语义分割任务的推荐方案。
4. 若目标尺寸与原始尺寸相同，可跳过此步骤以节省时间。

适用场景：
------------------------------------------------------------
适合数字病理图像项目在数据准备阶段使用，用于规范化输入数据，
保证后续训练脚本（train.py）能够一致读取 512×512 规模的图像与掩码。

============================================================
"""
def fix_dataset(split_name, target_size=(512, 512)):
    img_dir = f"data/{split_name}/images"
    mask_dir = f"data/{split_name}/masks"
    fixed_img_dir = f"data/{split_name}/images_fixed"
    fixed_mask_dir = f"data/{split_name}/masks_fixed"

    os.makedirs(fixed_img_dir, exist_ok=True)
    os.makedirs(fixed_mask_dir, exist_ok=True)

    count = 0
    missing_masks = 0

    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".bmp"):
            continue

        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace(".bmp", "_anno.bmp"))

        if not os.path.exists(mask_path):
            print(f"❌ 缺少 mask: {mask_path}")
            missing_masks += 1
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 重采样（mask 用 NEAREST 以保持标签）
        img = img.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)

        img.save(os.path.join(fixed_img_dir, fname))
        mask.save(os.path.join(fixed_mask_dir, fname.replace(".bmp", "_anno.bmp")))

        count += 1

    print(f"✅ {split_name} 集处理完成：{count} 张图像已修复，缺少 {missing_masks} 张 mask。")


# ========================
# 主执行部分
# ========================
if __name__ == "__main__":
    target_size = (512, 512)  # 可按需修改
    fix_dataset("train", target_size)
    fix_dataset("val", target_size)
    print("🎯 所有数据集均已修复完毕！")
