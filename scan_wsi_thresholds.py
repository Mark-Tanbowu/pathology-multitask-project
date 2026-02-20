"""Scan WSI classification thresholds from per-slide score logs.

Default log path:
run/20260209_1439/wsi_scores_20260209_143935.log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan threshold metrics on WSI score logs.")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("run/20260209_1439/wsi_scores_20260209_143935.log"),
        help="Path to wsi_scores_*.log",
    )
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to evaluate.")
    parser.add_argument("--score-key", type=str, default="slide_score_topk", help="Score field key.")
    parser.add_argument("--start", type=float, default=0.90, help="Start threshold.")
    parser.add_argument("--end", type=float, default=0.99, help="End threshold.")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold step.")
    return parser.parse_args()


def load_epoch_rows(log_path: Path, epoch: int, score_key: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("epoch") != epoch:
                continue
            if "slide_id" not in item:
                continue
            if score_key not in item or "slide_label" not in item:
                continue
            rows.append((int(item["slide_label"]), float(item[score_key])))
    return rows


def calc_metrics(rows: list[tuple[int, float]], threshold: float) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for y_true, score in rows:
        y_pred = 1 if score > threshold else 0
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        else:
            fn += 1

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_acc = 0.5 * (recall + specificity)

    return {
        "acc": acc,
        "balanced_acc": balanced_acc,
        "recall": recall,
        "specificity": specificity,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def frange(start: float, end: float, step: float) -> list[float]:
    values = []
    cur = start
    # Include end boundary.
    while cur <= end + 1e-12:
        values.append(round(cur, 10))
        cur += step
    return values


def main() -> None:
    args = parse_args()
    if not args.log_path.exists():
        raise FileNotFoundError(f"Log not found: {args.log_path}")

    rows = load_epoch_rows(args.log_path, args.epoch, args.score_key)
    if not rows:
        raise ValueError(
            f"No usable rows found for epoch={args.epoch}, score_key={args.score_key} in {args.log_path}"
        )

    print(
        f"log={args.log_path} epoch={args.epoch} slides={len(rows)} score_key={args.score_key} "
        f"range=[{args.start}, {args.end}] step={args.step}"
    )

    best = None
    for th in frange(args.start, args.end, args.step):
        m = calc_metrics(rows, th)
        print(
            f"th={th:.4f} acc={m['acc']:.4f} bacc={m['balanced_acc']:.4f} "
            f"recall={m['recall']:.4f} spec={m['specificity']:.4f} "
            f"tp={int(m['tp'])} tn={int(m['tn'])} fp={int(m['fp'])} fn={int(m['fn'])}"
        )
        if best is None or m["balanced_acc"] > best[1]["balanced_acc"]:
            best = (th, m)

    assert best is not None
    th, m = best
    print("\nBEST (by balanced_acc)")
    print(
        f"th={th:.4f} acc={m['acc']:.4f} bacc={m['balanced_acc']:.4f} "
        f"recall={m['recall']:.4f} spec={m['specificity']:.4f} "
        f"tp={int(m['tp'])} tn={int(m['tn'])} fp={int(m['fp'])} fn={int(m['fn'])}"
    )


if __name__ == "__main__":
    main()
