# Repository Guidelines

## 项目结构 Project Structure
核心训练逻辑集中在 `src/`（datasets、models、losses、engine、utils），Hydra 配置放在 `configs/` 并由 `scripts/run_train.py`、`scripts/run_infer.py` 调用；新增模块时遵循相同子目录。`tests/` 保存 pytest 用例，研究笔记或实验草稿位于 `notebooks/`、`demo/`、`web_demo/`，而 attention/轻量骨干/动态 loss 原型放在 `optional_modules/`。训练日志、权重和 Hydra 快照写入 `outputs/`、`.hydra/`，这些目录已被 `.gitignore` 屏蔽。

## 构建与开发 Build/Test Commands
依赖通过 `pip install -r requirements.txt` 或 `poetry install` 安装。默认冒烟命令为 `python -m src.engine.train`（使用 dummy data），推理示例可运行 `python -m src.engine.infer --image demo/demo_patch.png --mask_out demo/pred_mask.png`。如需真实数据，追加 Hydra 覆写：`python -m src.engine.train data.use_dummy=false data.train_dir=data/train`。提交前执行 `python -m pytest tests` 与 `ruff check src tests`，保持与 `.github/workflows/ci.yml` 一致。

## 代码风格 Coding Style
遵循 Ruff 规则（line length 100、E/F/I 选择），函数/变量使用 snake_case，类名 PascalCase，模块名小写。导入顺序为 stdlib → third_party → local，并用空行分隔。新增 Hydra 配置键采用点式命名（如 `model.backbone`），公开 API 需附简洁 docstring。

## 测试规范 Testing Guidelines
所有单测由 pytest 驱动，文件命名 `tests/test_<feature>.py`，函数命名 `test_<behavior>`。修改训练/推理主流程时扩展 `tests/test_smoke.py`，涉及 metric 或 loss 的逻辑则在 `tests/test_dice.py` 附近加入 tensor-level 检验。若存在随机增强，请设置 `torch.manual_seed(42)`、`numpy.random.seed(42)` 以保证 CI 可复现。

## 提交流程 Commit & PR Workflow
Git 历史采用简短祈使句（如 “Update ci.yml”），因此 commit subject 也保持动词开头并在本地 squash 噪声提交。Pull Request 需说明改动动机、关键 Hydra overrides、数据来源与核心指标（loss、Dice、ROC 等），并在可视化改动时附 overlay 截图。提交前确保 pytest/ruff 全绿，PR 描述里链接相关 issues 并记录潜在风险或依赖升级。

## 安全与配置 Security Tips
真实病理数据与 checkpoints 保存在 `data/raw`、`outputs/` 等已忽略目录，不得上传 PHI。敏感凭据（MLflow、S3、数据库）通过环境变量或 Hydra 运行时参数注入，禁止写入 repo。分享日志或可视化前检查 `.hydra/`、overlay 图是否包含路径或患者信息，必要时脱敏或清理。
