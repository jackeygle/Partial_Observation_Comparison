# 演讲备注 / Speaker notes

## Slide 1: Progress: 50% map · full-data training · fair EnKF comparison

EN: three things done since last meeting, in order — (1) rebuilt the walkable map with the supervisor's 50% occupancy rule; (2) trained 4DVarNet to convergence on the FULL 32-day dataset at the paper window dT=200; (3) a FAIR reconstruction comparison against the EnKF (apt-ibex) baseline on held-out test days. Framing: reproducing the paper + an honest baseline comparison, not chasing a single number.
中文: 上次会后做了三件事 — ①按老师 50% 占用规则重建可行走地图;②在全量 32 天数据、论文窗口 dT=200 下把 4DVarNet 训到收敛;③在留出测试日上和 EnKF(apt-ibex)baseline 做了公平的重建对比。定调: 复现论文 + 诚实 baseline 对比,不刷单一指标。

## Slide 2: 1 — Walkable map: supervisor's 50% occupancy rule

EN: the map now follows the supervisor's literal 50% rule — a 1 m cell is wall iff ≥50% of it is obstacle (config obstacle_rule=per_cell, occupancy_thresh=0.5). At 50% this equals the connected-footprint variant (verified same mask). Obstacles = ROS OCCUPIED ∪ enclosed-UNKNOWN. Hybrid with the training-day 'ever walked' criterion to kill kernel spill across walls; largest component so nobody is trapped. The walkable count is printed in the figure (computed live from the mask, never hard-coded here).
中文: 地图现在照老师字面 50% 规则 — 1m 格 ≥50% 是障碍就整格标墙(config per_cell/0.5)。50% 时与连通 footprint 版一致(已验证同一 mask)。障碍=ROS OCCUPIED∪封闭 UNKNOWN。再与'训练日走过'混合,去掉跨墙核溢出;保留最大连通域防困住。可行格数打印在图里(由 mask 实时算,不写死)。

## Slide 3: 2 — The framework (recap): learned variational reconstruction

EN: one-slide recap of how it works and how it trains, since it drives everything after. Two-term cost (fit observations + plausibility), 20-step learned gradient descent from X₀, uses only observations + Φ at inference. Trained end-to-end: loss back-propagates through the whole unrolled solver (paper Eq.14).
中文: 一页回顾框架怎么跑、怎么训。两项代价(拟合观测+合理性),从 X₀ 做 20 步学习型梯度下降,推理只用观测+Φ。端到端训练: 损失沿整条展开 solver 反传(论文 Eq.14)。

## Slide 4: 3 — Trained to convergence on the full 32-day dataset (dT=200)

EN: report as a REPRODUCTION of the paper's dT=200 setting, not a metric contest. Full 32-day dataset, 100 epochs, 20 solver iters, one H200. The table numbers are read back from the checkpoint's saved args (never hand-typed); the curve is this run's own metrics.jsonl — converges below the naive X₀ fill.
中文: 以'复现论文 dT=200 设置'讲,不搞指标比赛。全量 32 天、100 epoch、20 solver 迭代、单张 H200。表里数字从 checkpoint 的 args 读回(不手打);曲线是本次 metrics.jsonl,收敛到朴素 X₀ 填充以下。

## Slide 5: 4 — Fair comparison: identical everything, no truth at init

EN: this is the slide the supervisors will scrutinise. Stress fairness: same test days / observations / noise / frames / metric / clip; and crucially NEITHER method is initialised from truth — 4DVarNet uses the obs-based X₀ (paper-faithful), EnKF uses its original random init (we did NOT modify the EnKF project). We only compare reconstruction accuracy; EnKF's spread is its own strength, shown separately.
中文: 老师最会盯这页。强调公平: 同测试日/观测/噪声/帧/指标/裁剪;且两者都不从真值初始化 — 4DVarNet 用基于观测的 X₀(忠于论文),EnKF 用它原始随机初始化(没改 EnKF 项目)。只比重建精度;EnKF 的 spread 是它的长处,单独展示。

## Slide 6: 5 — Reconstruction: density and velocity (EnKF-baseline style)

EN: two matched frames in the EnKF project's own visual style so they sit next to the baseline with no mismatch. Top = density (Blues) + heading arrows; bottom = speed magnitude (Blues) + heading arrows, so velocity accuracy is visible. Point out: true has a few high-speed blocks; 4DVarNet recovers more of them than the EnKF, which under-estimates speed. Last panel is the EnKF ensemble spread (its uncertainty).
中文: 两张同帧、用 EnKF 项目的可视化风格,和 baseline 并排无违和。上=密度(蓝)+朝向箭头;下=速率大小(蓝)+朝向箭头,能看出速度精度。指出: true 有几处高速块,4DVarNet 比 EnKF 恢复得多,EnKF 普遍低估速度。最后一栏是 EnKF ensemble spread(它的不确定性)。

## Slide 7: 6 — Results: 4DVarNet vs EnKF (held-out test days)

EN: headline + honesty in one slide. Left = overall blind-zone / full-state MSE (4DVarNet lower overall). Right = per-channel by region (observed | blind): 4DVarNet wins var and vx decisively; but on the blind zone EnKF is slightly better on density and vy (density is sparse, low-SNR — 4DVarNet's weak spot). Say this openly — the win is real but not uniform. All numbers here are read from the eval JSONs.
中文: 头条+诚实一页讲完。左=总体盲区/全场 MSE(4DVarNet 总体更低)。右=分通道分区域(观测|盲区): 4DVarNet 在 var、vx 明显赢;但盲区里 density、vy 是 EnKF 略好(密度稀疏、低 SNR,是弱项)。要坦白说 — 赢是真的但不均匀。数字都从 eval JSON 读。

## Slide 8: Appendix — Architecture in detail (backup)

EN: BACKUP slide — do not present by default; pull up only if the supervisor asks for architecture detail. Three bands: (A) end-to-end flow X₀→GradSolver(×20)→x_rec→loss, back-prop trains Φ and the solver jointly; (B) inside one iteration — build the two-term cost J, take its gradient by AUTODIFF (no hand-derived ∇), a ConvLSTM maps the gradient to an update, x ← x − u/n_iter; (C) inside Φ=GENN — two-scale (coarse avg-pool branch + fine residual branch, summed), each branch = ψ (zero-centre 3×3×3 conv, centre tap=0) → ReLU → φ (two 1×1×1 pointwise convs). Zero-centre is why the prior can't collapse to the identity. All kernel/width/iter numbers are read from config + the checkpoint.
中文: 备用页 — 默认不讲,老师问架构细节才翻。三段: (A) 端到端流程,反传联合训 Φ+solver; (B) 单次迭代内部: 组两项代价 J → autodiff 求梯度(不手推)→ ConvLSTM 把梯度变成更新 → x←x−u/n_iter; (C) Φ=GENN 内部: two-scale(粗 avg-pool 分支 + 细残差分支相加),每个分支 = ψ(zero-centre 3×3×3,中心抽头=0)→ReLU→φ(两层 1×1×1 pointwise)。zero-centre 是先验不塌缩成恒等的关键。所有核/宽度/迭代数从 config+checkpoint 读。
