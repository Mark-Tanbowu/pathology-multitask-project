"""查看 best.pt 等多任务 checkpoint 里保存了哪些信息的辅助脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def summarize_checkpoint(path: Path) -> None:
    """加载 checkpoint 并打印最重要的字段，方便快速检查内容。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到 checkpoint 文件：{path}")

    payload = torch.load(path, map_location="cpu")
    print(f"=== Checkpoint 文件: {path} ===")
    print(f"可用字段: {list(payload.keys())}")

    epoch = payload.get("epoch")
    best_score = payload.get("best_score")
    metrics = payload.get("metrics", {})
    history = payload.get("history", {})
    config = payload.get("config", {})

    print(f"\n- 保存 epoch: {epoch}")
    print(f"- best_score: {best_score}")
    print(f"- metrics: {metrics}")
    if history:
        train_hist = history.get("train", [])
        val_hist = history.get("val", [])
        print(f"- history: train={len(train_hist)} 条记录, val={len(val_hist)} 条记录")
    if config:
        model_cfg = config.get("model", {})
        loss_cfg = config.get("loss", {})
        tasks_cfg = config.get("tasks", {})
        print(f"- model 配置: {model_cfg}")
        print(f"- loss 配置: {loss_cfg}")
        print(f"- tasks 配置: {tasks_cfg}")

    state_dict = payload.get("model_state_dict")
    if state_dict:
        print(f"\n模型参数 tensors: {len(state_dict)} 个")

    opt_state = payload.get("optimizer_state_dict")
    print(f"包含优化器状态: {opt_state is not None}")
    sched_state = payload.get("scheduler_state_dict")
    print(f"包含调度器状态: {sched_state is not None}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看 best.pt 内的关键信息")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/best.pt"),
        help="checkpoint 路径（默认 outputs/best.pt）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
