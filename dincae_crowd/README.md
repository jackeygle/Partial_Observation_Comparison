# dincae_crowd — DINCAE 在 ATC 人群场上的复现

第二条技术路线。第一条（4DVarNet 复现 + EnKF 对比）已完成，在
[`../4dvarnet_enkf/`](../4dvarnet_enkf/)，本目录**不依赖它的结论**，只复用它的数据管道
（`observation_model.py` / `navigation.py` / `data_pipeline/`）和**完全相同的观测配置**。

论文：Barth et al. 2020 (GMD 13, 1609) = **DINCAE 1.0**；Barth et al. 2022 (GMD 15, 2183)
= **DINCAE 2.0**。参考实现：[`../../reference/DINCAE.jl`](../../reference/DINCAE.jl)（Julia v2.0.6）。

**目标是严格复现论文的方法**，不是先做改进。凡是偏离论文的地方都在下面列出并说明理由。

---

## 术语

论文是海洋遥感背景，它的一些词（climatology、cloud）在人群场里读起来是外来物。**本目录一律
用右列的说法**，论文原词只出现在这张表里，方便和论文/参考实现对照时查。

| 论文 / 参考实现 | 本目录 | 是什么 |
|---|---|---|
| climatology、`remove_mean`、`meandata` | **逐格均值场** | 每个格子在 32 个训练日上的平均值，一张 `(4,36,12)` 的表。存在 `artifacts/state_stats.npz` 的 `mean` 里 |
| anomaly | **残差** | 减掉逐格均值场之后剩下的部分，即"比平常多还是少"。网络就工作在这个空间 |
| cloud / non-cloud（SST 的云遮挡） | **缺测 / 有观测** | 我们这里是机器人视野外，不是云 |
| `obs_err_std`、σ²_obs | 观测误差方差 | 取常数 1，见"编码"一节 |

代码里的标识符跟着这张表走：`residual_mse()`、`resid_std`、`dev_resid_mse`。产物文件
`state_stats.npz` 的数组名（`mean` / `std` / `count` / `valid_mask`）是中性的，未改动。

---

## 目录结构

| path | 角色 |
|---|---|
| `state.py` | **状态定义**：四通道、逐通道有效性规则、逐通道变换（`var` 走 log1p）、以及**逐格均值场**（每个格子在 32 个训练日上的平均值）+ 残差标准差 |
| `encoding.py` | information-form 编码：`y/σ²` + `1/σ²`，缺测=两片皆 0 |
| `dataset.py` | 一天 → 训练样本；观测配置读 `4dvarnet_enkf/config.yaml`；磁盘缓存 |
| `model.py` | U-Net + SumSkip + 精化步 + σ̂ 参数化（Eq.6-7） |
| `losses.py` | 高斯 NLL（Eq.3），逐变量独立归一化后相加 |
| `train.py` | 训练循环（Adam / 裁值 5 / 输出平均用的 checkpoint） |
| `checks/` | 自检、评估、以及关于数据的测量脚本 |
| `sbatch/` | SLURM 提交脚本 |
| `artifacts/` | **产物**：`state_stats.npz`（训练要读）、`decay_tables.*`（仅诊断） |
| `cache/` | **产物**：编码缓存（约 23 GB，可随时删除重建） |
| `check_outputs/` | **产物**：`checks/` 脚本输出的指标 JSON |
| `runs/` | **产物**：checkpoint、`metrics.jsonl`、SLURM 日志 |

`checks/` 里的内容：

- `check_encoding.py` — `encoding.py` 的不变量自检（缺测处两片是否为 0、目标掩膜是否正确…）
- `evaluate.py` — 评估：多 epoch 输出平均、σ̂ 校准、变率保持、两套 MSE 口径
- `measure_coverage_revisit.py` / `measure_obs_age.py` / `measure_decorrelation.py` —
  关于**数据本身**的测量（覆盖率、观测年龄、时间自相关），不在训练路径上
- `measure_decay_tables.py` — 早期为一版 age-dependent σ² 做的标定。**不在论文实现里**，
  保留是因为它产出的 AC 表是有效测量。产物在 `artifacts/decay_tables.*`。

---

## 端到端

```bash
module load scicomp-pytorch-env/2026.1
cd /scratch/work/zhangx29/Thesis_Project/dincae_crowd

# 1. 逐格统计量（一次；纯 numpy/scipy，登录节点即可，约 8 分钟）
python3 state.py                       # -> artifacts/state_stats.npz

# 2. 编码自检（约 2 分钟，登录节点即可）
python3 checks/check_encoding.py             # 期望 PASS

# 3. 训练（GPU 节点）
sbatch sbatch/submit_train.sbatch            # 200 epoch，自链接 + --resume
#   -> runs/dincae_full/{last.pt, ckpt_*.pt, metrics.jsonl}
#   第一个 epoch 建 cache/（约 23 GB），之后每 epoch 只读盘

# 4. 评估（GPU 节点）
sbatch sbatch/submit_eval.sbatch --split test
#   -> check_outputs/eval/dincae_metrics_test.json
```

**torch 只在 GPU 节点跑**（`sbatch`，或 `srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00`），
不要在登录节点。`state.py` 和 `checks/measure_*.py` 是纯 numpy/scipy，登录节点可跑。

---

## 方法

### 编码（`encoding.py`）

论文不"填充"缺测。输入两片 `y/σ²` 和 `1/σ²`，缺测 = 两者皆 0 = 精度为零
（1.0 §3；2.0 §2.3；`data.jl:311-325`）。**σ²_obs 取常数 1**（论文/代码默认
`obs_err_std = 1`；1.0 §3 说这个常数的具体值不重要，它会被第一层权重吸收）。于是

```
观测到  ->  scaled = (fwd(y) − mean)/std ,  invvar = 1
缺测    ->  scaled = 0                   ,  invvar = 0
```

时间上只用相邻 3 帧（`ntime_win = 3`，1.0 §3 的"前一天/当天/后一天"）。一个格子若不在
{t−1, t, t+1} 里被观测到，就是缺测 —— **论文里没有"沿用更早的观测"这回事**。

**输入 30 通道**：row, col (2) + cos/sin 日周期与周周期 (4) + `dt ∈ {−1,0,+1}` ×
(残差[4], mask[4]) (24)。**目标 8 通道**：每通道一对 (残差·mask, mask)。

### 逐通道有效性（`state.channel_valid`）

四个通道"在哪里有定义"各不相同，所有统计与损失都只在有定义处进行。这不是对论文的改动，
而是论文"缺测 = 精度为零"这条原则在我们数据上的落实：

```
density : 处处有定义（density=0 是真实测量："这里没人"）
vx, vy  : density > 0    —— 空格子的速度是占位符
var     : vel_var > 0 ⟺ 格内至少 2 人（1 个点的方差无定义）
```

`var` 通道（格内**人群速度的离散程度**，一个真实物理场）**既是被重建的第 4 个通道**，
也是速度不确定度的来源（`vel_var / density` = 格均值的标准误）—— 两个角色不冲突。

### 网络（`model.py`）

照 2.0 的结构，不是 1.0 论文 Table 1：

- **无全连接瓶颈、无 dropout**（2.0 §4：FC 要求训练与推理输入尺寸完全一致）
- **SumSkip（加法）而非 concat**（2.0 §2.1 Eq.2；同一算例 0.3835 → **0.3604**）
- **MeanPool**（`model.jl:268` 硬编码）。⚠️ 2.0 正文 Table 1 写的是 max pooling ——
  **论文与代码不一致**，这条在 2.0 里没被重新验证过，`--pool max` 可 A/B
- **精化步**（2.0 §2.2 Eq.4，α=0.3/α'=0.7；Table 2：0.60 → **0.55**）
- 36×12 → 18×6 → 9×3 → 5×2，滤波器 32/64/96，参数约 32 万

### 损失（`losses.py`）

2.0 Eq.3 的高斯 NLL。每个输出变量**独立算、各自用自己的 N 归一化、然后相加**
（`model.jl:132-148`），所以通道间权重完全由学出来的 `1/σ̂²` 决定，不手工设。
`truth_uncertain` 的 KL 分支在代码里实现了但**默认不用**（两篇论文都没有）。

---

## 对论文的三处偏离

| # | 偏离 | 理由 |
|---|---|---|
| 1 | **目标用 ground truth**，而非有缺测的观测 | 论文没有 truth（这正是 DINCAE 存在的理由），我们有。用观测当目标等于主动放弃已有信息 |
| 2 | **各通道残差归一到单位方差** | 论文所有变量都用 `obs_err_std=1`，在 SST 上没问题（残差本就 O(1) °C）。我们四通道残差方差差三个数量级，不归一化时 `log σ̂²` 项可被小量级通道白赚负 loss —— 实测 train NLL 降 5 个单位而 density 的 dev MSE 反而从 0.0254 涨到 0.0419。参考代码的 `normalize2`（`data.jl:111-120`）做的就是这件事 |
| 3 | **`var` 通道走 log1p** | 1.0 结论段：这套方法"可以很容易推广到 log-normal 分布来处理浓度类变量"。`var` 非负、重尾。不变换时它的归一化 dev MSE 在 8~62 之间震荡（1.0 = 只输出逐格均值的水平），加 log1p 后降到 0.7~1.6 |

**没有**：age-dependent σ²、多尺度时间聚合、按 track 随机丢弃、`truth_uncertain`。前两项曾
实现过（现只剩 `checks/measure_decay_tables.py` 的测量），后两项论文/代码里有但不用 —— 都不是论文的东西。

---

## ATC 上的两个硬约束（`checks/check_encoding.py` 会打印）

```
单帧覆盖 walkable 格子:  55.5%
3 帧窗口内至少一次:      63.1%   -> 36.9% 的格子两片全 0，只能靠坐标+时钟+逐格均值
目标有定义比例  density: 67.1%    vx/vy: 13.2%    var: 12.8%
```

论文的数据是**每日一张快照**，"前一天/当天/后一天"覆盖率高得多。我们是 1 Hz 采样、场约
2 秒去相关（`checks/measure_decorrelation.py`），所以论文原版在这里天然拿不到多少观测信息。
这两个数字是解释性能的关键，不是 bug。

---

## 评估口径（`checks/evaluate.py`）

**两套 MSE 都报**，因为它们会给出不同结论（4dvarnet_enkf 项目里已经踩过这个坑）：

- `ours` — 只在该通道**有定义**的格子上、限制在 walkable 内。训练时的口径。
- `v4dvar` — 完全照 `4dvarnet_enkf/checks/eval_test_days.py`：原始场、**所有**格子（含非
  walkable）、四通道无权重、盲区 = `mask < 0.5`，并施加与 EnKF 相同的物理裁剪。
  这个口径会在我们从未训练过的格子上打分，对我们不利；报它是为了可比。

两套都同时给 `_noclip` 版本。**`noclip` 与 `clip` 的差距本身是个收敛诊断**：信息形式下
`m = x₁·σ̂²` 而 σ̂² 上限是 `1/µ = 1000`（Eq.6 钳位），未收敛的模型能输出 μ≈1000，
`var` 反变换后就是天文数字（实测冒烟时物理 MSE 达 10²²）。收敛后两者应当接近。

另外两项都来自论文：**σ̂ 校准**（2.0 §5.2，按预测 SD 分 10 档比实际 RMS；注意 2.0 用了
**全局调整因子**，即原始 σ̂ 的绝对尺度有偏，可信的是结构与排序）和**变率保持**
（1.0 Fig.8 / 2.0 Table 3，RMSE 偏爱平滑场，所以必须另外报变率）。

---

## Gotchas

- **模块名遮蔽**：`4dvarnet_enkf` 里也有 `losses.py`。所以本目录的脚本一律
  `sys.path.insert(0, 项目根)` 之后才 `sys.path.append(4dvarnet_enkf)` —— 顺序反了会导入错文件。
- **改了编码就要清 `cache/`**。缓存键含 `CACHE_VER` 与观测配置，改这些会自动失效；但改了
  `state_stats.npz` 的内容（同样天数）不会触发重建，需手动 `rm -rf cache`。
- **`/tmp` 是节点本地的** —— 计算节点读不到登录节点的 `/tmp`。产物写到项目内。
- 观测配置**不要在这里另设默认值**，一律读 `4dvarnet_enkf/config.yaml`（`dataset.obs_config`），
  否则两条路线的观测场景就不可比了。
