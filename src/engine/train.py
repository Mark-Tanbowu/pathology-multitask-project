import os
import torch
import sys
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from src.utils.visualizer import LossVisualizer
from datetime import datetime
from src.datasets.camelyon_dataset import PathologyDataset
from src.datasets.transforms import Compose, HorizontalFlip
from src.losses.combined import MultiTaskLoss
from src.models.multitask_model import MultiTaskModel
from src.utils.metrics import accuracy_from_logits, dice_coefficient
from src.utils.misc import get_device, set_seed


# 保证在项目根目录运行（从 src/engine 跳两级）
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


@hydra.main(config_path="../../configs", config_name="defaults", version_base=None)
def main(cfg):
    # ------------------------
    # 初始化配置与环境
    # ------------------------
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.log.save_dir, exist_ok=True)

    # ------------------------
    # 将控制台输出重定向到日志文件
    # ------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = cfg.log.save_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{timestamp}.log")

    # 将 print 同时输出到控制台和文件
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()  # 实时写入文件

        def flush(self):
            pass  # 为兼容性保留

    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout  # 同时捕获错误信息

    print(f"[Logger] 日志已重定向到文件: {log_path}")
    print("=" * 60)
    print("[INFO] 启动训练进程")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # ------------------------
    # 数据加载与预处理
    # ------------------------
    transform = Compose([HorizontalFlip(p=0.5)])  # 示例：仅水平翻转
    train_ds = PathologyDataset(cfg.data.train_images, cfg.data.train_masks, cfg.data.train_labels, transform=transform)
    val_ds = PathologyDataset(cfg.data.val_images, cfg.data.val_masks, cfg.data.val_labels, transform=None)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # ------------------------
    # 模型、损失函数与优化器
    # ------------------------
    model = MultiTaskModel(
        backbone_name=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        seg_upsample_to_input=cfg.model.seg_upsample_to_input,
    ).to(device)

    criterion = MultiTaskLoss(
        seg_type=cfg.loss.seg,
        cls_type=cfg.loss.cls,
        seg_weight=cfg.loss.seg_weight,
        cls_weight=cfg.loss.cls_weight,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # ✅ 新增：初始化可视化对象
    visualizer = LossVisualizer(save_dir=cfg.log.save_dir)

    best_val = -1.0

    # ===============================================================
    # 主训练循环
    # ===============================================================
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n===== [Epoch {epoch:03d}/{cfg.num_epochs}] 训练阶段开始 =====")

        # ---------------- Train ----------------
        model.train()
        run_loss = 0.0
        seg_part, cls_part = 0.0, 0.0  # 用于记录子任务损失占比

        for step, (images, masks, labels) in enumerate(train_loader, start=1):
            # === 数据送入设备 ===
            images = images.to(device)
            masks = masks.to(device).float()  # ✅ Dataset 已是 {0,1}，仅保证 dtype
            labels = labels.to(device)

            optimizer.zero_grad()

            # === 模型前向 ===
            seg_logits, cls_logits = model(images)

            # === 计算多任务损失 ===
            loss, parts = criterion(seg_logits, masks, cls_logits, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * images.size(0)
            seg_part += parts.get("seg_loss", 0)
            cls_part += parts.get("cls_loss", 0)

            # === 日志打印 ===
            if step % 50 == 0 or step == 1:
                seg_mean = seg_logits.mean().item()
                cls_mean = cls_logits.mean().item()
                print(
                    f"[Train Step {step:04d}] "
                    f"loss={loss.item():.4f} "
                    f"(seg={parts.get('seg_loss', 0):.4f}, cls={parts.get('cls_loss', 0):.4f}) "
                    f"mask_range=({masks.min().item():.1f},{masks.max().item():.1f}) "
                    f"logit_mean(seg={seg_mean:.3f}, cls={cls_mean:.3f})"
                )

                # 异常检测
                if torch.isnan(loss):
                    print("❌ [错误] 检测到 NaN，请检查学习率或数据输入！")
                    return

        train_loss = run_loss / len(train_ds)
        avg_seg = seg_part / max(len(train_loader), 1)
        avg_cls = cls_part / max(len(train_loader), 1)
        print(f"[Train Summary] train_loss={train_loss:.4f} | seg_loss={avg_seg:.4f} | cls_loss={avg_cls:.4f}")

        # ---------------- Validate ----------------
        print(f"----- [Epoch {epoch:03d}] 验证阶段开始 -----")
        model.eval()
        val_loss, dice_sum, acc_sum, n_batches = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for step, (images, masks, labels) in enumerate(val_loader, start=1):
                images = images.to(device)
                masks = masks.to(device).float()  # ✅ 不再除255，保持一致
                labels = labels.to(device)

                seg_logits, cls_logits = model(images)
                loss, parts = criterion(seg_logits, masks, cls_logits, labels)

                val_loss += loss.item() * images.size(0)
                dice = dice_coefficient(torch.sigmoid(seg_logits), masks)
                acc = accuracy_from_logits(cls_logits, labels)

                dice_sum += dice.item()
                acc_sum += acc
                n_batches += 1

                print(f"[DEBUG] batch={step:02d}, dice={dice.item():.4f}, acc={acc:.4f}")

                if step == 1:
                    print(f"[DEBUG] seg_logits range=({seg_logits.min().item():.3f},{seg_logits.max().item():.3f}), "
                          f"sigmoid_mean={torch.sigmoid(seg_logits).mean().item():.3f}")
                pred = torch.sigmoid(seg_logits)
                print(f"pred mean={pred.mean().item():.3f}, pred>0.9比例={(pred > 0.9).float().mean().item():.3f}")

                # 每隔若干 batch 打印一次验证日志
                if step % 20 == 0 or step == 1:
                    print(
                        f"[Val Step {step:03d}] "
                        f"val_loss={loss.item():.4f} "
                        f"dice={dice:.4f}, acc={acc:.4f} "
                        f"mask_range=({masks.min().item():.1f},{masks.max().item():.1f}) "
                        f"logit_mean(seg={seg_logits.mean().item():.3f})"
                    )

        # === 验证指标汇总 ===
        val_loss /= len(val_ds)
        val_dice = dice_sum / max(n_batches, 1)
        val_acc = acc_sum / max(n_batches, 1)

        # === 异常检测 ===
        if val_dice > 1.0 or val_dice < 0:
            print(f"⚠️ [异常] val_dice={val_dice:.4f} 超出 [0,1] 范围，可能是 mask 未归一化。")
        if val_loss < 0:
            print(f"⚠️ [异常] val_loss={val_loss:.4f} 为负，请检查损失函数实现。")

        print(f"[Validate Summary] val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | val_acc={val_acc:.4f}")

        # ✅ 记录当前 epoch 损失曲线
        visualizer.update(
            epoch=epoch,
            train_losses={"seg": avg_seg, "cls": avg_cls, "total": train_loss},
            val_losses={"total": val_loss, "dice": val_dice, "acc": val_acc}
        )

        # === 保存最佳模型 ===
        score = (val_dice + val_acc) / 2.0
        if score > best_val and cfg.log.save_ckpt:
            best_val = score
            ckpt_path = os.path.join(cfg.log.save_dir, "best.pt")
            torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg)}, ckpt_path)
            print(f"✅ [INFO] 已保存最优模型到: {ckpt_path}")

    print(f"🎯 训练完成，最佳验证评分: {best_val:.4f}")

if __name__ == "__main__":
    main()
