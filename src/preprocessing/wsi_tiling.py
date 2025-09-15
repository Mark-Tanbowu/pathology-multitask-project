"""
WSI 切片工具（占位版）
- 真实项目可使用 OpenSlide/pyvips 读取 .svs/.tiff，并按网格裁切 tile。
- 这里提供一个普通图像的简化切片函数用于演示。
"""
from typing import Tuple, List
from PIL import Image

def tile_image(image_path: str, tile_size: Tuple[int, int]=(256,256), stride: Tuple[int,int]=(256,256)) -> List[Image.Image]:
    """将大图按 tile_size/stride 裁成小块（简单示例）。
    返回 PIL Image 列表。
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    tw, th = tile_size
    sw, sh = stride
    tiles = []
    for y in range(0, H - th + 1, sh):
        for x in range(0, W - tw + 1, sw):
            tiles.append(img.crop((x, y, x+tw, y+th)))
    return tiles
