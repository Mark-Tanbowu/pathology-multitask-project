"""WSI 阈值标定脚本。

用途：
1. 读取 slide 级分数日志（JSONL），支持 test_slide_scores 与 wsi_scores 两种格式；
2. 在给定阈值区间内扫描，计算 ACC / Balanced-ACC / F1 等指标；
3. 输出最佳阈值，并可将全量扫描结果写入 CSV 或 JSON。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EPS = 1e-12


@dataclass
class SlideRow:
    """单个 slide 的标签与分数记录。"""

    slide_id: str
    slide_label: int
    score: float
    epoch: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate WSI threshold from slide-score JSONL logs."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="日志路径（例如 run/.../test_slide_scores_*.log 或 wsi_scores_*.log）。",
    )
    parser.add_argument(
        "--score-key",
        type=str,
        default="slide_score_topk",
        help="作为阈值判定依据的分数字段名。",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="slide_label",
        help="标签字段名（默认 slide_label）。",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="仅评估指定 epoch（仅对含 epoch 字段的日志有效）。",
    )
    parser.add_argument(
        "--auto-epoch",
        choices=["none", "max", "min"],
        default="max",
        help=(
            "若 --epoch 未指定且日志中含 epoch，自动选择哪个 epoch："
            "max（最新）/min（最早）/none（不按 epoch 过滤）。"
        ),
    )
    parser.add_argument("--start", type=float, default=0.10, help="阈值扫描起点（含）。")
    parser.add_argument("--end", type=float, default=0.99, help="阈值扫描终点（含）。")
    parser.add_argument("--step", type=float, default=0.01, help="阈值扫描步长。")
    parser.add_argument(
        "--objective",
        choices=["balanced_acc", "f1", "acc", "recall", "specificity", "precision"],
        default="balanced_acc",
        help="最佳阈值选择目标。",
    )
    parser.add_argument("--topn", type=int, default=5, help="控制台展示前 N 个候选阈值。")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="可选输出路径。后缀 .csv 写表格；其他后缀写 JSON。",
    )
    return parser.parse_args()


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(log_path: Path, score_key: str, label_key: str) -> tuple[list[SlideRow], int]:
    rows: list[SlideRow] = []
    skipped = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                item = json.loads(s)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if score_key not in item or label_key not in item:
                skipped += 1
                continue
            score = _to_float(item.get(score_key))
            label = _to_int(item.get(label_key))
            if score is None or label is None:
                skipped += 1
                continue

            epoch = _to_int(item.get("epoch"))
            slide_id = str(item.get("slide_id", f"line_{len(rows)}"))
            rows.append(SlideRow(slide_id=slide_id, slide_label=label, score=score, epoch=epoch))
    return rows, skipped


def choose_epoch(rows: list[SlideRow], explicit_epoch: int | None, auto_epoch: str) -> int | None:
    if explicit_epoch is not None:
        return explicit_epoch
    if auto_epoch == "none":
        return None
    epochs = sorted({r.epoch for r in rows if r.epoch is not None})
    if not epochs:
        return None
    return epochs[-1] if auto_epoch == "max" else epochs[0]


def filter_and_dedup(rows: list[SlideRow], epoch: int | None) -> tuple[list[SlideRow], int]:
    if epoch is not None:
        rows = [r for r in rows if r.epoch == epoch]
    # 同一 slide 若重复出现，保留最后一条（通常是最新写入）。
    by_slide: dict[str, SlideRow] = {}
    dup_count = 0
    for row in rows:
        if row.slide_id in by_slide:
            dup_count += 1
        by_slide[row.slide_id] = row
    return list(by_slide.values()), dup_count


def frange(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    cur = float(start)
    while cur <= end + EPS:
        values.append(round(cur, 10))
        cur += step
    return values


def calc_metrics(rows: list[SlideRow], threshold: float) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for row in rows:
        pred = 1 if row.score > threshold else 0
        if row.slide_label == 1 and pred == 1:
            tp += 1
        elif row.slide_label == 0 and pred == 0:
            tn += 1
        elif row.slide_label == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    youden_j = recall + specificity - 1.0
    return {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "youden_j": youden_j,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def rank_key(item: dict[str, float], objective: str) -> tuple[float, float, float, float, float]:
    # 主目标一致时按 bacc -> f1 -> acc -> recall -> -fp 做稳定 tie-break。
    return (
        float(item.get(objective, 0.0)),
        float(item.get("balanced_acc", 0.0)),
        float(item.get("f1", 0.0)),
        float(item.get("acc", 0.0)),
        -float(item.get("fp", 0.0)),
    )


def dump_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.log_path.exists():
        raise FileNotFoundError(f"Log not found: {args.log_path}")
    if args.step <= 0:
        raise ValueError("--step 必须 > 0")
    if args.start > args.end:
        raise ValueError("--start 必须 <= --end")

    raw_rows, skipped = load_rows(args.log_path, args.score_key, args.label_key)
    if not raw_rows:
        raise ValueError(
            "未读取到可用记录，请检查 --score-key/--label-key 是否与日志字段一致。"
        )

    target_epoch = choose_epoch(raw_rows, args.epoch, args.auto_epoch)
    rows, dup_count = filter_and_dedup(raw_rows, target_epoch)
    if not rows:
        raise ValueError(
            f"筛选后无数据：epoch={target_epoch}，请检查 --epoch 或 --auto-epoch 设置。"
        )

    label_set = {r.slide_label for r in rows}
    if len(label_set) < 2:
        raise ValueError("标签只包含单一类别，无法进行有效阈值标定。")

    thresholds = frange(args.start, args.end, args.step)
    scan_rows: list[dict[str, float]] = []
    for th in thresholds:
        m = calc_metrics(rows, th)
        record: dict[str, float] = {"threshold": float(th)}
        record.update(m)
        scan_rows.append(record)

    ranked = sorted(scan_rows, key=lambda x: rank_key(x, args.objective), reverse=True)
    best = ranked[0]
    topn = max(int(args.topn), 1)

    print("=== WSI Threshold Calibration ===")
    print(f"log_path={args.log_path}")
    print(f"score_key={args.score_key} label_key={args.label_key}")
    print(f"slides={len(rows)} skipped_rows={skipped} duplicate_slide_rows={dup_count}")
    print(f"epoch={target_epoch if target_epoch is not None else 'NA'}")
    print(
        f"range=[{args.start:.4f}, {args.end:.4f}] step={args.step:.4f} "
        f"objective={args.objective}"
    )

    print("\nTop candidates:")
    for i, row in enumerate(ranked[:topn], start=1):
        print(
            f"{i:>2d}. th={row['threshold']:.4f} "
            f"acc={row['acc']:.4f} bacc={row['balanced_acc']:.4f} "
            f"f1={row['f1']:.4f} pre={row['precision']:.4f} rec={row['recall']:.4f} "
            f"spec={row['specificity']:.4f} "
            f"tp={int(row['tp'])} tn={int(row['tn'])} fp={int(row['fp'])} fn={int(row['fn'])}"
        )

    print("\nBest threshold:")
    print(
        f"threshold={best['threshold']:.4f} "
        f"acc={best['acc']:.4f} bacc={best['balanced_acc']:.4f} "
        f"f1={best['f1']:.4f} pre={best['precision']:.4f} rec={best['recall']:.4f} "
        f"spec={best['specificity']:.4f}"
    )
    print(
        "Hydra override suggestion: "
        f"test_detect.wsi_acc_threshold={best['threshold']:.4f}"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix.lower() == ".csv":
            dump_csv(args.out, scan_rows)
        else:
            payload: dict[str, Any] = {
                "log_path": str(args.log_path),
                "score_key": args.score_key,
                "label_key": args.label_key,
                "epoch": target_epoch,
                "slides": len(rows),
                "skipped_rows": skipped,
                "duplicate_slide_rows": dup_count,
                "scan": scan_rows,
                "best": best,
                "objective": args.objective,
            }
            with args.out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved scan results to: {args.out}")


if __name__ == "__main__":
    main()
