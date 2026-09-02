# 原始数据流程 / Raw-data pipeline (CSV → grid_cache)

> 目的：让工程从**原始 ATC CSV** 就能完整复现出我们加载的 `grid_cache`，不依赖任何
> 预处理好的中间产物。两段都用纯 numpy/h5py 干净重写，并与现有数据**对拍验证**。
> Goal: reproduce `grid_cache` from the **raw ATC CSV**, with no reliance on
> pre-made intermediates. Both stages are clean numpy/h5py and **validated** against
> the existing data.

```
raw CSV ──[csv_to_h5.py]──► 轨迹 H5 (position/velocity/index) ──[h5_to_grid.py]──► grid_cache (N,4,H,W)
```

---

## 阶段 1：`csv_to_h5.py` — CSV → 轨迹 H5

**原始 CSV（8 列，每行 = 某时刻某行人的一次观测）**

| 列 | 字段 | 单位 | 说明 |
|---|---|---|---|
| 0 | time | s | 时间戳 |
| 1 | **pid** | — | 行人 ID（**不写入 H5**，见下方说明） |
| 2 | pos_x_mm | mm | x 位置 |
| 3 | pos_y_mm | mm | y 位置 |
| 4 | pos_z_mm | mm | z 高度（未用） |
| 5 | spd_mm | mm/s | 速率 |
| 6 | ang | rad | 运动方向 |
| 7 | face | rad | 朝向（未用） |

**输出轨迹 H5**

| 字段 | 形状 | 含义 |
|---|---|---|
| `position` | (N,2) f32 | (x,y) 米 = mm/1000 |
| `velocity` | (N,2) f32 | (vx,vy) m/s = spd/1000·(cos ang, sin ang) |
| `index` | (M,) struct (time,start,stop,count) | 按时间戳分组：`position[start:stop]` 是同一时刻的所有行人；count=1 |

> **关于 ID（导师会问）**：CSV 里有 `pid`，但我们的目标是**宏观场**（每个格子的密度/速度统计量），
> 不追踪个体轨迹，所以 H5 **有意不存 pid**。个体身份在网格化时被聚合掉了。

---

## 阶段 2：`h5_to_grid.py` — 轨迹 H5 → grid_cache

把轨迹点核密度估计成 4 通道网格（三角核、走廊子集旋转矩形 36×12、1m/格、1s/帧）。

**输出 grid_cache**

| 字段 | 形状 | 含义 |
|---|---|---|
| `grid` | (N,4,H,W) f32 | 通道 [**density, vx, vy, vel_var**] |
| `time` | (N,) f64 | 每帧秒数 |
| attrs | — | subset, period, resolution, kernel, origin, theta, shape |

- **density 密度**：每格行人密度（核估计，按该秒累计时间步数 count 归一化）
- **vx, vy**：每格平均速度（已旋转进网格坐标系）
- **vel_var 速度方差**：每格速度偏差平方和的加权平均（带 nnz/(nnz-1) 无偏修正）

> 中间还有一步"按周期分窗"：原始 `index`（逐时间戳）→ `index_1.0s`（每 1 秒一帧，
> count = 该秒累计的时间步数）。现有轨迹 H5 已缓存 `index_1.0s`，直接读。

---

## 怎么运行 / How to run

```bash
module load scicomp-pytorch-env/2026.1

# 阶段1: CSV -> 轨迹 H5
python3 csv_to_h5.py --csv /scratch/work/zhangx29/ATC/atc-20121028.csv \
                     --out /tmp/atc-20121028.h5

# 阶段2: 轨迹 H5 -> grid_cache
python3 h5_to_grid.py --traj-h5 /tmp/atc-20121028.h5 \
                      --out /tmp/atc-20121028_corridor_1.0s.h5 --subset corridor
```

## 验证结果 / Validation (proves the clean reimpl is faithful)

```bash
# 阶段1: 转换结果 vs 现有轨迹 H5
python3 csv_to_h5.py --csv .../atc-20121028.csv --validate .../Sundays/atc-20121028.h5
#   position max|Δ| = 0.000e+00   velocity max|Δ| = 0.000e+00     ✓

# 阶段2: 网格化结果 vs 现有 grid_cache
python3 h5_to_grid.py --traj-h5 .../Sundays/atc-20121028.h5 \
                      --validate .../grid_cache/atc-20121028_corridor_1.0s.h5
#   density/vx/vy/var max|Δ| ~ 6e-6  (float32 舍入级别)            ✓
```

→ 两段都复现到浮点舍入级别，证明这套干净流程与原始 pipeline 等价。

**更全面的验证**（合成手算微例 + 全行/全 index 对拍 + CSV→H5→grid 端到端串联）：

```bash
python3 checks/check_data_pipeline.py                  # 三层全跑（默认 20121028 那天）
python3 checks/check_data_pipeline.py --synthetic-only # 不需要真实数据
```

> 验证时核实的两个**固有约定**（原 pipeline 如此，复现它才与现有数据等价）：
> ① 原始 `index` 记录的 `time` 是该组的"结束时刻"（= 下一组的时间戳；末条 = 本组自身时间戳）；
> ② `index_1.0s` 首窗的 `count` 把第一条原始记录重复计了一次（与原缓存逐字段一致）。
