from pathlib import Path

train_imgs = {p.name for p in Path("../../data/train/images_fixed").glob("*.bmp")}
val_imgs   = {p.name for p in Path("../../data/val/images_fixed").glob("*.bmp")}
dup = train_imgs & val_imgs
print(f"重复样本数量: {len(dup)}")
if dup:
    print("重复文件示例:", list(dup)[:5])
