"""
============================================================
掩码完整性与归一化检查工具 (mask_checker.py)
------------------------------------------------------------
功能概述：
本脚本用于对病理图像项目中的掩码文件进行快速质量检查，
帮助研究者判断掩码图像是否已正确归一化、像素取值范围是否合理，
并验证 Hydra 配置文件中指定的数据路径是否有效。

主要功能：
1. 读取 Hydra YAML 配置文件 (configs/defaults.yaml)，自动获取训练/验证集路径。
2. 对掩码文件逐一分析：
   - 输出最小值、最大值、像素类型 (uint8/float32 等)
   - 判断掩码是否已归一化（0~1）
   - 检测潜在问题（如重复归一化或异常取值）
3. 批量扫描文件夹中多个掩码样本，打印检查摘要。
4. 提供针对 0~255 灰度图与 0~1 浮点掩码的自动化建议。

使用说明：
------------------------------------------------------------
运行方式：
    python tools/mask_checker.py

参数说明：
    - `config_path`：Hydra 配置文件路径 (默认: configs/defaults.yaml)
    - `limit`：每个文件夹中预览的掩码样本数量（默认 5）

输出内容：
    - 每个掩码的像素分布与唯一值
    - 掩码类型判断与建议
    - 路径有效性检测结果

适用场景：
------------------------------------------------------------
该工具适合在数字病理图像项目的数据准备阶段使用，
可在模型训练前确认掩码预处理（除255、归一化等）是否正确，
避免出现 Dice 值异常、Loss 不收敛等问题。

============================================================
"""
import os
import yaml
import numpy as np
from PIL import Image

def load_hydra_config(config_path="configs/defaults.yaml"):
    """读取 Hydra YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def analyze_mask(mask_path):
    """分析单个掩码文件像素分布"""
    mask = np.array(Image.open(mask_path))
    min_val = float(mask.min())
    max_val = float(mask.max())
    unique_vals = np.unique(mask)
    n_unique = len(unique_vals)

    # 判断掩码类型
    if max_val <= 1.0:
        mask_type = "已归一化 (0~1 浮点型)"
    elif max_val <= 255 and mask.dtype in [np.uint8, np.uint16]:
        mask_type = "灰度掩码 (0~255 整数型)"
    else:
        mask_type = "可能是类别索引或异常类型"

    # 判断归一化状态
    normalized_once = np.isclose(max_val, 1.0, atol=1e-2)
    normalized_twice = max_val < 0.01

    if normalized_twice:
        suggestion = "⚠️ 可能被重复归一化，请检查 /255 操作。"
    elif normalized_once:
        suggestion = "✅ 正常归一化，无需再除以 255。"
    elif max_val > 1 and max_val <= 255:
        suggestion = "💡 建议归一化：mask = mask / 255.0"
    else:
        suggestion = "⚠️ 取值异常，请人工确认。"

    print(f"\n🧩 文件: {os.path.basename(mask_path)}")
    print(f"   - 形状: {mask.shape}, 类型: {mask.dtype}")
    print(f"   - 最小值: {min_val:.4f}, 最大值: {max_val:.4f}")
    print(f"   - 唯一值数: {n_unique}")
    print(f"   - 类型判断: {mask_type}")
    print(f"   - 建议: {suggestion}")
    print(f"   - 前10个唯一值: {unique_vals[:10]}")

def analyze_folder(folder_path, limit=5):
    """批量检查文件夹中的掩码图像"""
    if not os.path.exists(folder_path):
        print(f"❌ 路径不存在: {folder_path}")
        return
    print(f"\n🔍 正在扫描文件夹: {folder_path}\n")
    count = 0
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.png', '.jpg', '.tif', '.tiff', '.bmp')):
            analyze_mask(os.path.join(folder_path, file))
            count += 1
            if count >= limit:
                print(f"\n📦 已检查 {limit} 个文件（可修改 limit 检查更多）")
                break

if __name__ == "__main__":
    # 读取 Hydra 配置
    cfg_path = "../../configs/defaults.yaml"
    config = load_hydra_config(cfg_path)

    # 获取训练与验证掩码路径
    train_masks = config["data"]["train_masks"]
    val_masks = config["data"]["val_masks"]

    print("🧠 掩码数据检查开始！")
    analyze_folder(train_masks, limit=5)
    analyze_folder(val_masks, limit=5)
    print("\n✅ 掩码检查完成。建议结合输出结果调整预处理逻辑。")

