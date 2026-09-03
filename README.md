# Partial Observation Comparison

在 ATC 行人轨迹数据集上，对比五种从**部分观测**重建人群密度/速度场的方法。

## 结构

```
crowdcore/          五方共用：与方法无关的那一半
  config.py+yaml      全部参数的唯一来源
  navigation.py       walkable 区域、A* 路径规划、真实 ATC 地图
  observation_model.py 多机器人传感器的移动与观测生成、Omega、X0 填充
  paths.py            仓库内路径的唯一真相
  data/               CSV -> h5 -> 4 通道网格

methods/            每个子包一个方法，互不 import
  varnet/             4DVarNet 变分同化（plain MSE 与 NLL/不确定性头两条臂）
  enkf/               局部化 EnKF —— Partial_observation 的只读 vendor 副本
  dincae/             DINCAE 卷积自编码器插补
  senseiver/          Senseiver 稀疏传感器重建

compare/            跨方法的评测与画图。只有这里可以同时 import 多个方法
slides/             汇报用的 deck
sbatch/_env.sh      唯一的环境入口
```

## 怎么跑

```bash
source sbatch/_env.sh                    # module load + PYTHONPATH + PYTHONSAFEPATH
python3 -m compare.compare4              # 五方对比
python3 -m methods.varnet.train --help
python3 -m methods.enkf.checks.verify_enkf_opt --frames 12
```

**必须用 `python3 -m`。**直接 `python3 methods/varnet/train.py` 会把 `methods/varnet/`
塞进 `sys.path[0]`，而三个方法各有一份**不同的** `losses.py`（`dataset.py`、`model.py`
也一样），谁先进搜索路径谁赢。`_env.sh` 里的 `PYTHONSAFEPATH=1` 挡的就是这个。

作业脚本在各方法的 `sbatch/` 下，它们自己 `cd` 到方法目录，所以 `runs/`、
`check_outputs/` 这类相对路径照常可用：

```bash
sbatch methods/varnet/sbatch/submit_varnet.sbatch --days 32 --dT 200
sbatch methods/dincae/sbatch/submit_eval.sbatch
```

## 五个方法

| 目录 | 方法 | 不确定性输出 |
|---|---|---|
| `methods/varnet/` | 4DVarNet（plain MSE，Eq.14） | 无 |
| `methods/varnet/` | 4DVarNet + 不确定性头（NLL，5 成员集成） | 学到的 σ̂ + epistemic |
| `methods/dincae/` | DINCAE（16 个 checkpoint 输出平均） | σ̂（信息形式） |
| `methods/senseiver/` | Senseiver | 无 |
| `methods/enkf/` | 局部化 EnKF（+ PedPred3 前向模型） | 集合离散度 |

`methods/enkf/enkf_lab/` 是 `/scratch/work/zhangx29/Partial_observation` 的**逐字节只读
副本**，文件刻意 chmod 444。要改就改 `enkf_opt/`，并且必须通过
`methods/enkf/checks/verify_enkf_opt.py` 的逐位比对（`np.array_equal`，不是 `isclose`）。

## 评测口径：一个必须读的坑

**名次会随口径翻转。**同一份预测，在「所有格子」口径下 Senseiver 第一、DINCAE 差 12 倍
垫底；换成「有定义格子」口径后 DINCAE 第一、Senseiver 第三。

原因是速度通道：空格子里没有人，也就没有速度，数据管线在那里存的 `0` 是占位符而不是
测量值。而盲区里 **88.4%** 的格子是空的。三个在完整场上训练的方法学会了"空格子输出 0"
白拿这部分分；DINCAE 只在有定义的格子上训练过，就被这一项压垮。

所以任何图表**只给一个数字而不说清算的是哪些格子，就是误导**。`compare/compare4.py`
把两套口径并排算出来，规则的唯一定义在 `methods/dincae/state.py:channel_valid()`。

另外 4DVarNet 的数字有 **~4e-4 的复现下限**（它推理时也要走 autograd 反传），
第 4 位小数是噪声 —— 见 `refactor_baseline/README.md`。

## 未入库的内容

仓库只含代码、配置、指标和图。以下生成数据约 30 GB，由 `.gitignore` 排除：

- `*.pt` 模型权重（`methods/*/runs/`，约 1 GB）
- `*.npz` EnKF 估计场（`methods/varnet/check_outputs/enkf*`，约 18 GB）
- `methods/dincae/cache/` 网格缓存（约 13 GB）
- `**/check_outputs/eval/seq_ppt_*/` 逐帧序列图
