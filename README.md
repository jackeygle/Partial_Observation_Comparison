# Partial Observation Comparison

在 ATC 行人轨迹数据集上，对比三种从**部分观测**重建人群密度/速度场的方法。

| 目录 | 方法 | 入口 |
|---|---|---|
| [`4dvarnet_enkf/`](4dvarnet_enkf/) | 4DVarNet 变分同化 + EnKF 基线 | `train_varnet.py`、`enkf_lab/`、`enkf_opt/` |
| [`dincae_crowd/`](dincae_crowd/) | DINCAE（卷积自编码器插补） | `train.py` |
| [`senseiver_crowd/`](senseiver_crowd/) | Senseiver（稀疏传感器 → 场重建） | `train.py` |

每个子项目结构一致：
- `checks/` — 验证脚本，`check_outputs/` — 其输出（图 + 指标）
- `sbatch/` — Slurm 作业脚本（Aalto Triton）
- `runs/` — 训练日志与指标（权重 `*.pt` 未入库）
- `README.md` — 该方法的说明

## 未入库的内容

仓库只含代码、配置、指标和图。以下生成数据体积约 30 GB，由 `.gitignore` 排除，需在 Triton 上重新生成：

- `*.pt` 模型权重（`runs/`，约 1 GB）
- `*.npz` EnKF 估计场（`4dvarnet_enkf/check_outputs/`，约 18 GB）
- `dincae_crowd/cache/` 网格缓存（约 13 GB）
- `4dvarnet_enkf/check_outputs/eval/seq_ppt_*/` 逐帧序列图
