import os
import yaml
import numpy as np
from PIL import Image

# 自动确定项目根目录（mask_checker 位于 src/datasets/ 下两层）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONFIG_FILE = os.path.join(ROOT, "configs/defaults.yaml")

def load_hydra_config(config_path=CONFIG_FILE):
    """读取 Hydra YAML 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def analyze_mask(mask_path):
    """分析单个掩码像素分布"""
    mask = np.array(Image.open(mask_path))
    min_val = float(mask.min())
    max_val = float(mask.max())
    unique_vals = np.unique(mask)
    n_unique = len(unique_vals)

    # 判断掩码类型
    if max_val <= 1.0:
        mask_type = "已归一化 (0~1)"
    elif max_val <= 255:
        mask_type = "0~255 灰度掩码"
    else:
        mask_type = "异常/类别索引"

    print(f"\n🧩 文件: {os.path.basename(mask_path)}")
    print(f"   - 形状: {mask.shape}, 类型: {mask.dtype}")
    print(f"   - 最小值: {min_val}, 最大值: {max_val}")
    print(f"   - 唯一值: {unique_vals[:10]} ...")

def analyze_folder(folder_path, limit=5):
    """批量检查掩码"""
    if not os.path.exists(folder_path):
        print(f"❌ 路径不存在: {folder_path}")
        return
    print(f"\n🔍 检查目录: {folder_path}")
    count = 0
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.png', '.jpg', '.tif', '.bmp')):
            analyze_mask(os.path.join(folder_path, file))
            count += 1
            if count >= limit:
                break

if __name__ == "__main__":
    config = load_hydra_config()

    train_masks = os.path.join(ROOT, config["data"]["train_masks"])
    val_masks = os.path.join(ROOT, config["data"]["val_masks"])

    print("🧠 掩码检测开始…")
    analyze_folder(train_masks, limit=5)
    analyze_folder(val_masks, limit=5)
    print("\n🎯 掩码检查完成！")
