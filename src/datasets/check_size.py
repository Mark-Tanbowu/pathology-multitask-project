from PIL import Image
import os

"""
============================================================
数据集完整性与尺寸一致性检查工具 (check_size.py)
------------------------------------------------------------
功能概述：
本脚本用于对预处理后的病理图像数据集进行完整性验证，
检查图像与掩码文件在命名、数量与尺寸上的一致性，
以防止模型训练阶段出现维度不匹配或样本缺失问题。

主要功能：
1. 自动扫描指定数据集划分（train / val）的固定目录：
   - 图像目录: data/<split>/images_fixed/
   - 掩码目录: data/<split>/masks_fixed/
2. 检查内容包括：
   - 图像文件是否存在对应的掩码文件
   - 图像与掩码的尺寸是否一致
   - 输出缺失文件和异常尺寸的详细信息
3. 汇总检测结果：
   - 图像总数、缺失数量、不一致数量统计
   - 每张图像检查结果即时打印

使用说明：
------------------------------------------------------------
运行方式：
    python tools/dataset_checker.py

可选参数：
    - `split_name`：数据集划分名（默认检查 train 和 val）
      若需要检查 test，可自行调用 check_size("test")

输出内容：
    - 每张图像与掩码的匹配状态
    - 尺寸不一致或缺失文件提示
    - 每个数据集的汇总统计报告

注意事项：
------------------------------------------------------------
1. 本脚本假定数据已通过 fix_size.py 预处理，
   即图像与掩码存放在 *_fixed 目录中。
2. 掩码命名应遵循 `_anno.bmp` 规则。
3. 若检测出尺寸不一致，请重新运行重采样脚本进行修复。
4. 若目录不存在或为空，脚本会自动跳过该数据集并给出提示。

适用场景：
------------------------------------------------------------
本工具用于数字病理图像项目的数据质量控制阶段，
帮助研究者快速验证数据准备是否完整、规范，
以确保后续训练（train.py）阶段稳定运行。

============================================================
"""

def check_dataset(split_name):
    img_dir = f"data/{split_name}/images_fixed"
    mask_dir = f"data/{split_name}/masks_fixed"

    print(f"\n🔍 开始检查 {split_name} 集...")
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"❌ {split_name} 集目录不存在，跳过。")
        return

    total = 0
    inconsistent = 0
    missing = 0

    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(".bmp"):
            continue

        total += 1
        img_path = os.path.join(img_dir, fname)
        mask_name = fname.replace(".bmp", "_anno.bmp")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"❌ 缺少 mask 文件: {mask_name}")
            missing += 1
            continue

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        if img.size != mask.size:
            print(f"⚠️ 尺寸不一致: {fname}")
            print(f"   图像尺寸 = {img.size}, mask 尺寸 = {mask.size}")
            inconsistent += 1
        else:
            print(f"✅ 一致: {fname} ({img.size})")

    print(f"\n📊 {split_name} 集检测完成：共 {total} 张图像，"
          f"缺少 {missing}，尺寸不一致 {inconsistent}。\n")


if __name__ == "__main__":
    check_dataset("train")
    check_dataset("val")
    print("🎯 全部检测完成！")
