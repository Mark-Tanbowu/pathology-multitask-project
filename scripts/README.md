脚本入口说明
==========

文件列表
--------
- `run_train.py`：封装 Hydra 调用，便于在脚本内追加默认覆写或批量实验。
- `run_infer.py`：推理示例脚本，可指定单张图片与输出 mask。
- `python -m src.engine.eval`：测试集评估入口，读取 `data.test_*` 与 `outputs/best.pt`。
- `run_ablation.py`：消融/批量实验脚本，需在文件内配置实验列表。
- `verify_data_leakage.py`：训练/验证数据泄露检查（样本名重合与可选哈希重复）。
- `train.sh` / `train.ps1`：命令行训练示例（Linux/Windows）。

使用建议
--------
- 默认读取 `configs/defaults.yaml`，可在命令行追加 `key=value` 覆写。
- 批量实验：在 `run_ablation.py` 中集中管理不同的 backbone/注意力/动态损失开关。
- 记录：建议在脚本内或运行日志里写明实验名称与 Hydra 覆写，便于复现。

待补充
-----
- 推理批量脚本（目录推理、滑窗推理）。
- 结果整理/绘图脚本（从 `outputs/` 汇总指标、生成表格/曲线）。
