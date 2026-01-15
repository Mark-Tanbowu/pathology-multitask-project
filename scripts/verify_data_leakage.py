# -*- coding: utf-8 -*-
"""数据泄露检查脚本。

功能：
- 检查 train/val 之间的样本名重合（基于 labels 或文件名）；
- 可选检查图像/掩膜文件的哈希重合（同内容不同文件名也能发现）；
- 输出简洁报告，帮助排查数据划分问题。
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from typing import List, Sequence, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_label_names(path: str) -> List[str]:
    """读取 labels 文件中的样本名（默认格式：name,label）。"""
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if parts:
                names.append(parts[0])
    return names


def list_basenames(directory: str) -> List[str]:
    """扫描目录，返回所有图片文件的 basename（不含扩展名）。"""
    names: List[str] = []
    for fname in os.listdir(directory):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            names.append(os.path.splitext(fname)[0])
    return names


def find_duplicate_names(names: Sequence[str]) -> List[str]:
    """找出重复的样本名。"""
    counter = Counter(names)
    return [name for name, count in counter.items() if count > 1]


def hash_file(path: str, block_size: int = 1024 * 1024) -> str:
    """计算文件 MD5，用于检测内容重复。"""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


def build_hash_set(directory: str) -> Tuple[set, List[str]]:
    """扫描目录所有图片文件，返回哈希集合与文件列表。"""
    hashes = set()
    files: List[str] = []
    for fname in os.listdir(directory):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            path = os.path.join(directory, fname)
            files.append(path)
            hashes.add(hash_file(path))
    return hashes, files


def summarize_overlap(name: str, overlap: Sequence[str], limit: int = 20) -> None:
    """输出重合样本的摘要。"""
    print(f"[{name}] 重合数量: {len(overlap)}")
    if overlap:
        preview = overlap[:limit]
        print(f"[{name}] 示例(最多{limit}条): {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查训练/验证数据是否泄露")
    parser.add_argument("--train-images", required=True, help="训练图像目录")
    parser.add_argument("--val-images", required=True, help="验证图像目录")
    parser.add_argument("--train-labels", default=None, help="训练 labels.csv（可选）")
    parser.add_argument("--val-labels", default=None, help="验证 labels.csv（可选）")
    parser.add_argument("--train-masks", default=None, help="训练 mask 目录（可选）")
    parser.add_argument("--val-masks", default=None, help="验证 mask 目录（可选）")
    parser.add_argument("--check-hash", action="store_true", help="是否做文件哈希重复检查（较慢）")
    args = parser.parse_args()

    # 1) 基于 labels 或目录名的样本名重合检查
    if args.train_labels and args.val_labels:
        train_names = read_label_names(args.train_labels)
        val_names = read_label_names(args.val_labels)
        name_source = "labels"
    else:
        train_names = list_basenames(args.train_images)
        val_names = list_basenames(args.val_images)
        name_source = "filenames"

    print("=" * 70)
    print(f"样本名来源: {name_source}")
    print(f"训练样本数: {len(train_names)} | 验证样本数: {len(val_names)}")

    dup_train = find_duplicate_names(train_names)
    dup_val = find_duplicate_names(val_names)
    if dup_train:
        summarize_overlap("train 内部重复", dup_train)
    if dup_val:
        summarize_overlap("val 内部重复", dup_val)

    overlap_names = sorted(set(train_names) & set(val_names))
    summarize_overlap("train/val 样本名重合", overlap_names)

    # 2) 可选哈希检查（图像）
    if args.check_hash:
        print("=" * 70)
        print("开始哈希检查：图像")
        train_hashes, _ = build_hash_set(args.train_images)
        val_hashes, _ = build_hash_set(args.val_images)
        hash_overlap = sorted(train_hashes & val_hashes)
        print(f"图像哈希重合数量: {len(hash_overlap)}")

    # 3) 可选哈希检查（掩膜）
    if args.check_hash and args.train_masks and args.val_masks:
        print("=" * 70)
        print("开始哈希检查：掩膜")
        train_mask_hashes, _ = build_hash_set(args.train_masks)
        val_mask_hashes, _ = build_hash_set(args.val_masks)
        mask_overlap = sorted(train_mask_hashes & val_mask_hashes)
        print(f"掩膜哈希重合数量: {len(mask_overlap)}")

    print("=" * 70)
    print("检查完成。若出现重合，请优先排查同源 WSI 切片或重复采样。")


if __name__ == "__main__":
    main()
