# Detailed explanation of the meeting deck / 汇报 deck 详细解释文档

A full technical companion to `slides/meeting_deck.pptx`. Organised by topic (each topic
maps to one or more slides). For every topic: **the question → the method → the full math →
where the numbers come from → the physical meaning → honest caveats → link to the paper.**

本文件是 `slides/meeting_deck.pptx` 的完整技术解释,按主题组织(每个主题对应一到几页)。
每个主题都讲:**问题是什么 → 方法 → 完整公式 → 数字从哪来 → 物理意义 → 诚实点 → 与论文的关系。**

> Reference paper / 参考论文: R. Fablet et al., *End-to-End Learning for Variational Data
> Assimilation Models and Solvers* (arXiv:2007.12941, 2020) — "4DVarNet".
> Our contribution / 我们的工作: reproduce that framework on a THIRD example — ATC pedestrian
> crowd fields — alongside the paper's Lorenz-63/96. 在论文的 Lorenz-63/96 之外,把这套框架
> 复现到第三个例子:ATC 商场人群密度场。

---

# 0. The big picture / 总览

**EN.** We have a macroscopic crowd field `x` on a 36×12 grid, 4 channels per cell
(density, vx, vy, velocity-variance), one frame per second. A few robots move through the
corridor and only observe the cells near them. **The task: from the robots' sparse, noisy
observations, reconstruct the full field over a time window.** This is a data-assimilation
(inverse) problem. The paper's method (4DVarNet) solves it by (a) writing a variational cost
whose minimiser is the reconstruction, and (b) *learning* both the dynamical prior in that
cost and the solver that minimises it, end-to-end. We reproduce exactly this on our data.

**中文.** 我们有一个宏观人群场 `x`,网格 36×12,每格 4 通道(密度、vx、vy、速度方差),每秒一帧。
几个机器人在走廊里移动,只能观测到自己附近的格子。**任务:从机器人稀疏、带噪的观测,重建
整个时间窗内的完整场。** 这是数据同化(反问题)。论文方法(4DVarNet)的做法是:(a) 写一个
变分代价,它的极小点就是重建;(b) 把代价里的动态先验、以及最小化它的求解器,**都用学习的方式、
端到端**训练出来。我们在自己的数据上严格复现这一套。

---

# 1. The variational framework — the backbone (slides 7–9) / 变分框架(核心)

## 1.1 The cost function / 代价函数

**EN.** The reconstruction `x` (a whole window, shape C×T×H×W) is defined as the minimiser of

    J(x) = λ₁ · ‖ (x − y) ⊙ Ω ‖²   +   λ₂ · ‖ x − Φ(x) ‖²
           └──── observation term ────┘     └── prior term ──┘

- **Observation term** `‖(x−y)⊙Ω‖²`: on the cells a robot actually observed (mask Ω=1), the
  reconstruction should match the measurement `y`. `⊙` is element-wise; unobserved cells
  contribute nothing. This term pulls `x` toward the data *where we have data*.
- **Prior term** `‖x−Φ(x)‖²`: `Φ` is a learned operator that, given a candidate sequence `x`,
  returns "what this sequence *should* look like if it were dynamically consistent". If the
  sequence is plausible, `x ≈ Φ(x)` and this term is small. This term fills the *blind zone*
  (unobserved cells) with a dynamically sensible guess.
- `λ₁, λ₂` weight the two terms; in 4DVarNet they are effectively learned.

The whole point: the observation term alone is under-determined (most cells unobserved); the
prior term regularises it so a unique, plausible full field comes out.

**中文.** 重建 `x`(整段窗口,形状 C×T×H×W)定义为下面这个代价的极小点:

    J(x) = λ₁·‖(x−y)⊙Ω‖²  +  λ₂·‖x−Φ(x)‖²
           观测项                先验项

- **观测项**:在机器人真正观测到的格子上(掩码 Ω=1),重建要贴近测量 `y`;`⊙` 是逐元素相乘,
  没观测的格子不贡献。它把 `x` 往"有数据的地方"的数据上拉。
- **先验项**:`Φ` 是一个学习到的算子——给它一段候选序列 `x`,它返回"若这段序列动态自洽,应该
  长什么样"。序列合理时 `x≈Φ(x)`,这项就小。它负责把**盲区**(没观测的格子)用动态上合理的
  猜测补上。
- `λ₁,λ₂` 是两项权重,4DVarNet 里等价于学出来的。

核心:光有观测项是欠定的(大多数格子没观测);先验项做正则,才能解出唯一、合理的完整场。

## 1.2 The three parts / 三个部分

**EN.**
1. **Observation operator Ω** — decides which (cell, channel) each robot measures each second.
   In our case: a robot sees a disk of radius 7 cells, INTERSECTED with line-of-sight (walls
   block sight), and observations are taken every 4th frame (temporal sparsity). This is the
   "forward model" of the sensor.
2. **Dynamical prior Φ (a GENN)** — a neural operator acting as a *regulariser*, NOT a
   forecaster. It scores dynamical plausibility of a whole space-time sequence. (Its key
   defining property is stated in §1.4, but we deliberately do NOT show its internal conv
   layers in the deck.)
3. **Solver** — a *learned* iterative minimiser of `J`. Instead of hand-tuned gradient
   descent (which needs 10⁵ steps), a small recurrent network learns how to update `x` from
   the gradient of `J`, reaching a good minimum in ~20 steps.

**中文.**
1. **观测算子 Ω** —— 决定每秒每个机器人测哪些(格子,通道)。我们的设定:半径 7 格的圆盘 ∩
   视线(墙挡视线),且每 4 帧才观测一次(时间稀疏)。这是传感器的"前向模型"。
2. **动态先验 Φ(一个 GENN)** —— 作**正则器**的神经算子,不是预报器;它给整段时空序列的
   "动态合理性"打分。(它的关键性质见 §1.4;deck 里刻意不展示其内部卷积层。)
3. **求解器 Solver** —— **学习型**的 `J` 迭代最小化器。不用手调的梯度下降(要 10⁵ 步),而是
   一个小循环网络学会"如何用 `J` 的梯度去更新 `x`",约 20 步就到好极小点。

## 1.3 The solver in detail / 求解器细节 (code: `variational_solver.py`)

**EN.** Starting from a rough initial fill `x₀` (missing cells filled by the previous frame),
the solver repeats, for k = 1…20:

    g_k   = ∂J/∂x           (gradient of the variational cost, by AUTOMATIC DIFFERENTIATION)
    u_k   = LSTM(g_k, h)    (a convolutional-LSTM turns the gradient into an update)
    x     = x − u_k / 20    (apply the update)

Key points:
- the gradient `∂J/∂x` is obtained by autodiff — no hand-derived formula; this matches the
  paper ("automatic differentiation to compute the gradient of assimilation cost").
- the LSTM is the *learned* part: it learns a better-than-plain-gradient update rule. This is
  the "learning to solve" idea.
- 20 iterations is the paper's setting; the paper shows a hand-tuned gradient descent needs
  >150000 steps for comparable quality.

**中文.** 从粗初值 `x₀`(缺失格用前一帧填)出发,对 k=1…20 重复:

    g_k = ∂J/∂x        (变分代价的梯度,自动微分求)
    u_k = LSTM(g_k,h)  (卷积 LSTM 把梯度变成更新量)
    x   = x − u_k/20   (施加更新)

要点:梯度用自动微分(不手推,和论文一致);LSTM 是"学习"的部分,学出比朴素梯度更好的更新规则
("学习去求解");20 步是论文设定,论文里手调梯度下降要 >15 万步才有相当质量。

## 1.4 The prior Φ (concept only) / 先验 Φ(只讲概念)

**EN.** Φ is a Gibbs-Energy Neural Network (GENN). Its ONE essential property (worth saying):
it is **zero-centre** — the value `Φ(x)` at a space-time point (t,i,j) does **not** read the
input `x(t,i,j)` at that same point; it is rebuilt only from the *neighbours* (frames t±1, a
3×3 spatial neighbourhood). Why this matters: if Φ could read x(t,i,j), the trivial solution
Φ(x)=x would make the prior term ‖x−Φ(x)‖²=0 for ANY x — useless. Zero-centre forbids that,
so the prior genuinely constrains the sequence. (Per your instruction we do NOT present the
conv-kernel internals in the deck — this concept is enough.)

**中文.** Φ 是一个 Gibbs 能量神经网络(GENN)。它唯一要点是**零中心**:`Φ(x)` 在时空点 (t,i,j)
的值**不读**该点自身的输入 `x(t,i,j)`,只由**邻居**(帧 t±1、3×3 空间邻域)重建。为什么重要:
若 Φ 能读 x(t,i,j),平凡解 Φ(x)=x 会让先验项对任意 x 都为 0,毫无用处;零中心禁止这一点,先验
才真正约束序列。(按你的要求,deck 里不展示卷积内部,这个概念就够。)

## 1.5 End-to-end training (slide 8) / 端到端训练

**EN.** This is the paper's central idea and the point the supervisor cares about most.
- **Loss** (supervised, paper Eq.14): `L = ‖x_rec − X_true‖²`, where `x_rec` is the solver's
  20-step output and `X_true` is the ground-truth field (used ONLY in the loss, never inside
  the solver).
- **What is trained**: BOTH the prior Φ *and* the solver's LSTM, **together**. The loss is
  back-propagated through the *entire unrolled 20-step solver*. We do NOT train Φ on its own
  and then plug it in — Φ is shaped by how well the *whole* reconstruction pipeline performs.
- **Subtlety** (paper): during supervised training the gradient FED INTO the solver is the
  gradient of the *variational cost* J (which uses only observations), not the gradient of
  the training loss. The training loss only supervises the final output.
- Settings (slide-8 table): Adam, lr 1e-3, 15 epochs, batch 8, window dT=200 (paper window),
  20 solver iterations, ~2.05 M trainable parameters, 8 days of data = 1570 windows, on one
  NVIDIA H200.

**中文.** 这是论文的核心思想,也是老师最在意的点。
- **损失**(监督,论文 Eq.14):`L=‖x_rec−X_true‖²`,`x_rec` 是 solver 20 步的输出,`X_true` 是
  真值场(**只在损失里用,solver 内部从不碰**)。
- **训什么**:先验 Φ **和** solver 的 LSTM,**一起训**。损失沿**整条展开的 20 步 solver** 反传。
  不是先单独把 Φ 训好再插进去——Φ 是被"整条重建管线好不好"塑形的。
- **微妙点**(论文):监督训练时,**喂给 solver 的梯度是变分代价 J 的梯度**(只用观测),不是训练
  损失的梯度;训练损失只监督最终输出。
- 设置(第 8 页表):Adam、lr 1e-3、15 epoch、batch 8、窗口 dT=200(论文窗口)、solver 20 步、
  约 2.05M 可训练参数、8 天数据=1570 个窗口、单张 NVIDIA H200。

### Inputs & outputs / 输入与输出

**EN.** One training example = one dT=200 window; every tensor is shape (4, 200, 36, 12) = 4
channels × 200 frames × 36×12 grid.
- **Inputs to the solver (no ground truth):**  Y = the robots' partial noisy observations (0 where
  unobserved) · Ω = the observation mask (1 where observed) · X₀ = a rough initial fill (missing
  cells copied from the previous frame).
- **Output:** x_rec = the full reconstructed field over the whole window (all 4 channels, all cells,
  all 200 frames), from the solver's 20 iterations.
- **Loss / target (supervised, Eq.14):** the ground-truth field X, used ONLY to compute
  L = ‖x_rec − X‖² (MSE over the whole state). Back-prop through the unrolled solver updates Φ + the
  solver's LSTM jointly.
- Note: Y and Ω are SIMULATED from X by the observation model (disk ∩ line-of-sight + Gaussian
  noise). So X is used only to (a) generate the observations and (b) grade the loss; at inference
  the solver sees only Y, Ω, X₀.

**中文.** 一个训练样本 = 一个 dT=200 窗口;每个张量形状都是 (4, 200, 36, 12)= 4 通道 × 200 帧 × 36×12 网格。
- **喂给 solver 的输入(不含真值):** Y = 机器人的部分带噪观测(未观测格=0) · Ω = 观测掩码(1=被观测) ·
  X₀ = 粗初值(缺失格用前一帧填)。
- **输出:** x_rec = solver 20 步后的整段重建场(全 4 通道、全部格子、200 帧)。
- **损失/标签(监督,Eq.14):** 真值场 X,**只用来算** L = ‖x_rec − X‖²(全状态 MSE);损失沿展开的
  solver 反传,联合更新 Φ + solver 的 LSTM。
- 注:Y、Ω 是**从 X 用观测模型模拟**出来的(圆盘∩视线 + 高斯噪声)。所以 X 只在"生成观测"和"算损失"
  两处用;推理时 solver 只看 Y、Ω、X₀。

## 1.6 Inference & results (slide 9) / 推理与结果

**EN.** Inference = run the trained solver for 20 steps on a held-out test window, using only
the robots' observations. The output is the **whole reconstructed sequence** (all 200 frames),
not a single frame — the video/strip on slide 9 shows sampled timepoints of that sequence.
Metrics over 15 epochs:
- **blind-zone MSE** (reconstruction error on UNobserved cells) and **R-score** (paper's
  metric = MSE over the whole state) both fall from ~0.079 to ~0.038.
- the **baseline** (the rough initial fill X₀) is 0.083 — so the reconstruction is ~0.45× the
  baseline error, i.e. it clearly recovers structure the robots never saw.
Report this as a **reproduction of the paper's dT=200 setting**, not a leaderboard number.
Train-loss oscillates a little (normal for a learned LSTM solver); the eval metrics are smooth.

**中文.** 推理 = 在留出的测试窗口上跑训练好的 solver 20 步,只用机器人观测。输出是**整段重建
序列**(全部 200 帧),不是单帧——第 9 页的视频/长条图是这段序列的若干时间点采样。15 轮的指标:
- **盲区 MSE**(未观测格子的重建误差)和 **R-score**(论文指标=全状态 MSE)都从 ~0.079 降到 ~0.038;
- **baseline**(粗初值 X₀)是 0.083——所以重建误差约为 baseline 的 0.45×,明显恢复出了机器人
  没看到的结构。
以"复现论文 dT=200 设置"来讲,不是刷榜数字。train loss 有点震荡(学习型 LSTM solver 正常),
评估指标平滑。

---

# 2. Observation noise — the 4 σ values (slide 2) / 观测噪声的 4 个 σ

## 2.1 The sensor model / 传感器模型 (code: `observation_model.py`)

**EN.** An observed value is the true value plus independent Gaussian noise, ONLY on observed
cells:

    y(s) = ( x(s) + ε(s) ) · Ω(s),      ε_c(s) ~ Normal( mean = 0, variance = σ_c² )

The full Gaussian probability density of the noise:

    p(ε_c) = 1 / (√(2π) · σ_c) · exp( − ε_c² / (2 σ_c²) )

There is one σ per channel c ∈ {density, vx, vy, var} — that's the "4 numbers" the supervisor
asked about.

**中文.** 观测值 = 真值 + 独立高斯噪声,**只加在被观测的格子上**:

    y(s) = (x(s)+ε(s))·Ω(s),   ε_c(s) ~ 正态(均值=0, 方差=σ_c²)

噪声的完整高斯概率密度:

    p(ε_c) = 1/(√(2π)·σ_c) · exp(−ε_c²/(2σ_c²))

每个通道 c∈{density,vx,vy,var} 一个 σ —— 这就是老师问的"4 个数"。

## 2.2 Where the 4 numbers come from — full derivation / 4 个数怎么来(完整推导)

**EN.** For each channel, over the ACTIVE cells `A = { s : x_c(s) ≠ 0 }` (cells that actually
carry signal):

    (1) centre value:      m_c = median over A of x_c(s)
    (2) robust spread:     MedAbsDev_c = median over A of | x_c(s) − m_c |
    (3) robust std:        σ_robust,c = 1.4826 × MedAbsDev_c
    (4) sensor noise std:  σ_c = 0.25 × σ_robust,c

Fully substituted:

    σ_c = 0.25 × 1.4826 × median_{s∈A} | x_c(s) − median_{s'∈A} x_c(s') |

- **Why the median / median-absolute-deviation and not the plain mean/std?** Because the
  active-cell distributions have heavy tails (a few very crowded cells). The median and MAD
  are outlier-resistant, so a handful of crowd spikes don't inflate the estimate.
- **Why 1.4826?** For Gaussian data, MAD = 0.6745·σ, so σ = MAD/0.6745 = 1.4826·MAD. It is the
  *fixed* constant that converts a robust spread into a standard deviation — this is exactly
  where the "Gaussian" assumption enters, and it is not something we tuned.
- **Why 0.25?** This is the ONE assumption: we assume the sensor is about 4× more precise than
  the field's own natural fluctuation. It is NOT a measured sensor spec.

Reproduced numbers (script `checks/rederive_obs_std.py`, matches config within 3–5%):

    channel   median   MedAbsDev   1.4826·MAD   ×0.25 = σ_c   config
    density    0.175     0.159        0.235        0.0588      0.0569
    vx         0.035     0.843        1.249        0.312       0.315
    vy        -0.004     0.221        0.327        0.082       0.086
    var        0.020     0.017        0.025        0.0062      0.0064

**中文.** 每个通道,在**活跃格** `A={s:x_c(s)≠0}`(真正有信号的格子)上:

    (1) 中心值:   m_c = A 上 x_c(s) 的中位数
    (2) 稳健离散度:MedAbsDev_c = A 上 |x_c(s)−m_c| 的中位数(中位绝对偏差)
    (3) 稳健标准差:σ_robust,c = 1.4826 × MedAbsDev_c
    (4) 噪声标准差:σ_c = 0.25 × σ_robust,c

完整代入:σ_c = 0.25×1.4826×median_{s∈A}|x_c(s)−median_{s'∈A}x_c(s')|

- **为什么用中位数/中位绝对偏差而不是均值/标准差?** 活跃格分布重尾(少数极挤格子);中位数和 MAD
  抗离群,几个人群尖峰不会把估计撑大。
- **1.4826 哪来的?** 高斯下 MAD=0.6745·σ,反解 σ=MAD/0.6745=1.4826·MAD;它是把稳健离散度换成
  标准差的**固定常数**——"高斯"假设就在这里进来,不是我们调的。
- **0.25 哪来的?** 这是**唯一的假设**:设传感器比场的自然涨落准约 4 倍;不是测出来的规格。

复现数字(脚本 `checks/rederive_obs_std.py`,与 config 误差 3–5%)见上表。

## 2.3 How the paper does it, and why we don't copy variance = 2 / 论文怎么做、为什么不照搬方差 2

**How the paper computes its noise / 论文的噪声怎么来的.**
EN — it does NOT compute it from data: for the (synthetic) Lorenz-63 experiment the paper simply
PRESCRIBES a fixed additive Gaussian noise of **variance = 2** (σ=√2≈1.414) on the observed
component, sampled every 8 steps (Lorenz-96: every 4 steps, likewise a fixed variance). Because
Lorenz data is simulated, the noise level is a free design choice — a round value of 2 is fine.
中文 — 论文**不是从数据算的**:合成的 Lorenz-63 实验里,它**直接规定**观测加"方差=2 的加性高斯噪声"
(σ=√2),每 8 步采一次(Lorenz-96 每 4 步,同样固定方差)。合成数据噪声想加多少加多少,取整齐的 2 即可。
**Contrast / 对比:** the paper *prescribes* one fixed variance; we *calibrate* a per-channel σ from
the real data (§2.2). 论文=拍一个固定方差;我们=从真实数据逐通道标定。

**EN.** The paper adds variance=2 (i.e. σ=√2≈1.414) on the **Lorenz** state, whose signal std
is ~8–10, giving a noise/signal ratio of 0.16–0.39. Our channels have signal std 0.2–0.9, so
the same absolute √2 would be **1.6–6.9× the signal** — it would bury the data. What transfers
across systems is the **noise/signal RATIO**, not the absolute variance. Our per-channel ratios
(density 0.20, vx 0.35, vy 0.25) already sit inside the paper's 0.16–0.39 band; the same values
are also the R matrix of the EnKF/PF baseline on the same data. So we are aligned with the paper
in the *meaningful* (scale-free) sense. (Verbal answer if asked; no dedicated slide.)

**中文.** 论文的 variance=2(σ=√2≈1.414)加在 **Lorenz** 状态上(信号 std ~8–10),换算成噪信比
0.16–0.39。我们的通道信号 std 只有 0.2–0.9,同一个绝对 √2 就是**信号的 1.6–6.9 倍**,会淹没数据。
可迁移的是**噪信比**不是绝对方差;我们的比值(density 0.20、vx 0.35、vy 0.25)已落在论文 0.16–0.39
带内,且=同数据 EnKF/PF baseline 的 R 阵。所以我们在**有意义的(尺度无关)**层面与论文对齐。
(被问到口头回答,不单独做页。)

---

# 3. The map: real map, walkable region, coordinates (slides 3, 4) / 地图

## 3.1 Real map + "any obstacle → wall" (slide 3) / 真实图 + "有障碍就标墙"

**EN.** Earlier we tried a data-driven walkable map (infer where people CAN go from where they
DID go); it caused conflicts (density landing in obstacles). Per the supervisor, we dropped it
and use the REAL ATC localization map directly: a ROS PGM at 0.05 m/pixel where black pixels are
obstacles (~1.4%). Rule (taken literally): **a 1 m grid cell is a WALL if it contains ANY
obstacle pixel** — no threshold, no robot-radius inflation. Then keep only the largest connected
walkable component (so no robot is trapped on an island). Result: **138 of 432 cells walkable.**
Note there are two different masks in play (see §4): "where a robot can DRIVE" (this walkable
mask) and "what a robot can SEE" — they are NOT the same thing.

**中文.** 之前用数据驱动可行图(从"人去过哪"反推"人能去哪"),会冲突(密度落进障碍)。按老师意见
丢掉,直接用真实 ATC 定位地图:ROS PGM,0.05m/像素,黑像素=障碍(~1.4%)。规则(照字面):
**1m 格子含任一障碍像素就判为墙**——无阈值、无机器人半径膨胀;再保留最大连通可行域(防机器人被困)。
结果:**432 格里 138 格可行。** 注意有两套不同的掩码(见 §4):"能开进去"(这个可行掩码)和
"能看见"——两者不是一回事。

## 3.2 Coordinates & the rotation question (slide 4) / 坐标与旋转

**EN.** The grid is a 36×12 box placed on the mall map at a world origin `O=(38.28,−15.81) m`,
rotated by `θ=147°` (the corridor runs diagonally). Two coordinate frames:
- **world (x,y)** — position in metres; the mall's own axes.
- **grid (i,j)** — integer cell indices (i = row, j = column); i runs along the corridor,
  j across it.
The transform (both ways):

    grid → world:   world = local · Rᵀ + O
    world → grid:   local = (world − O) · R ,   R = [[cosθ,−sinθ],[sinθ,cosθ]]

i.e. **translate by −O, then rotate by θ.** (Slide-4 figure walks a real point through all four
steps: locate → subtract O → rotate → read the cell.)

**Was the map flipped?** No. Evidence: the all-day accumulated density, overlaid on the real map
via this transform, lands on the FREE corridor, not on walls. The Pearson correlation
`corr(density, obstacle-fraction) = −0.24` over the 432 cells is **negative**, meaning density
and obstacle move in OPPOSITE directions (a seesaw): where there's wall there are no people, and
vice-versa — i.e. **people are in free space, exactly as physics demands.** A positive correlation
would have meant "people inside walls" = a real flip/rotation bug. The magnitude is modest (−0.24)
only because most cells have near-zero density AND near-zero obstacle; the *sign* is what matters.
So the transform was always correct — what the supervisor saw as a "flip" was a matplotlib DISPLAY
convention (image row 0 at top), now fixed by drawing everything ON the real map with an explicit
coordinate system.

**中文.** 网格是 36×12 的框,放在商场地图上,世界原点 `O=(38.28,−15.81) m`,旋转 `θ=147°`
(走廊斜着)。两套坐标系:
- **世界 (x,y)** —— 米制位置,商场自身的轴。
- **网格 (i,j)** —— 整数格子编号(i=行、j=列);i 沿走廊、j 横向。
变换(双向):

    网格→世界:world = local·Rᵀ + O
    世界→网格:local = (world−O)·R ,  R=[[cosθ,−sinθ],[sinθ,cosθ]]

即**先减原点 O(平移)、再旋转 θ**。(第 4 页的图把一个真实点走完四步:定位→减 O→旋转→读格子。)

**地图翻了吗?** 没有。证据:全天累积密度按此变换叠到真实图上,落在**自由走廊**、不在墙上;432 格上
`corr(密度,障碍占比)=−0.24` 是**负的**,意味着密度与障碍此消彼长(跷跷板):有墙处没人、反之亦然
——**人在空处,完全符合物理。** 若是正相关才说明"人在墙里"=真翻转 bug。幅度小(−0.24)只是因为大多数
格子密度≈0 且障碍≈0;关键看**符号**。所以变换本就对——老师看到的"翻转"是 matplotlib 的显示约定
(图像第 0 行在顶部),已通过"把图画到真实地图上 + 显式坐标系"修正。

---

# 4. Sensing: line-of-sight & the "see-through walls" fix (slide 5) / 感知:视线与穿墙

## 4.1 The two original bugs / 原本两个 bug

**EN.**
1. **See-through walls**: the sensing model was a plain disk (radius 7) — any cell within range
   counted as observed, even cells BEHIND a wall. A camera/lidar can't see through a stall.
2. **Observation gaps**: the observed mask had been intersected with the WALKABLE mask,
   conflating "can drive" with "can see" — so open, populated cells right next to a pillar were
   wrongly marked unobserved.

Fix: `observed = disk ∩ line-of-sight`, and NOT intersected with walkable. Line-of-sight is
precomputed by ray-casting between cell centres against the map's obstacles.

**中文.**
1. **穿墙**:感知是纯圆盘(半径 7)——范围内的格子都算观测,连墙**背后**的也算;摄像头/激光穿不过货摊。
2. **观测缺口**:观测掩码曾与**可行域**相交,把"能开"和"能看"混为一谈——紧挨柱子、开阔且有人的格子
   被错误标为未观测。
修法:`观测 = 圆盘 ∩ 视线`,且**不**与可行域相交。视线由格心到格心对着地图障碍做射线预计算。

## 4.2 Why sight still "leaked", and the final fix / 为什么仍"漏"、最终修法

**EN.** After adding line-of-sight it STILL looked like the robot saw behind some walls. Root
cause = a **resolution mismatch**, not an LOS bug:
- the walkable/wall COLOURING is at 1 m and very aggressive ("any obstacle pixel → wall"), so
  **56% of the 270 "wall" cells actually hold <10% obstacle** (a thin laser line marked the
  whole cell as wall);
- but line-of-sight was tested at the fine 0.05 m pixel level, so a sight line legitimately
  passed through the ~99%-empty part of such a "wall" cell.

So the sight was physically correct (through a real gap), but the coarse 1 m colouring made it
*look* like see-through. **Final fix (your choice A):** make sight consistent with the wall
colouring — a sight line is blocked by ANY 1 m wall cell it crosses. Result: through-wall
observation pairs `3720 → 0`, and single-frame coverage drops `70% → 51%` (the honest effect of
proper occlusion, not a regression).

**中文.** 加了视线后**仍**像能看到墙后一些格子。根因是**分辨率错配**,不是 LOS 写错:
- 墙的**上色**在 1m 上、很激进("有障碍像素就标墙"),所以 **270 个"墙格"里 56% 其实障碍 <10%**
  (一条细线就把整格标成墙);
- 但视线是在 0.05m 细像素上判的,视线合法地从这种"墙格"里 ~99% 的空处穿过。

所以视线在物理上是对的(穿真实空隙),只是 1m 粗上色让它**看着像**穿墙。**最终修法(你选的 A):**
让视线与墙上色一致——视线穿过任何 1m 墙格即被遮挡。结果:穿墙观测对 `3720→0`,单帧覆盖 `70%→51%`
(正确遮挡的结果,不是退步)。

---

# 5. Visualisation conventions (slide 6) / 可视化约定

**EN.**
- **The 4 channels** are each a SCALAR field (one number per cell), shown as heatmaps with NO
  arrows: density (people/cell), vx, vy (velocity components), vel_var (velocity variance =
  crowd disorder).
- **vx, vy are SIGNED velocity components, NOT speed.** The sign is DIRECTION. In the corridor
  there are two opposing pedestrian flows, so vx is strongly bipolar (red = one way, blue = the
  other); vy (across the corridor) is small. Speed = √(vx²+vy²) is always ≥ 0; a negative vx is
  NOT a "negative speed", it is motion in the −x direction.
- **Axes**: x = along the corridor, y = across it (matching vx, vy). The channel figure now draws
  a big x/y coordinate arrow so this is unambiguous.
- **If a velocity-arrow plot is shown**: convention is background colour = speed magnitude,
  arrows = heading (unit length). In our reconstruction figures the background is DENSITY and the
  arrows are heading only — say the colour is density, arrows carry direction not speed.

**中文.**
- **4 个通道**各是一个**标量**场(每格一个数),用热图、**无箭头**:密度(人/格)、vx、vy(速度分量)、
  vel_var(速度方差=人群乱度)。
- **vx、vy 是带符号的速度分量,不是速率。** 符号=方向。走廊里两股对向人流,所以 vx 明显正负两极
  (红=一个方向、蓝=另一个);vy(横向)很小。速率=√(vx²+vy²) 恒 ≥0;vx 为负不是"负速率",是朝
  −x 方向运动。
- **坐标轴**:x=沿走廊、y=横向(与 vx、vy 对应)。通道图现在画了大号 x/y 箭头,不再含糊。
- **若展示速度箭头图**:约定是背景色=速率大小、箭头=朝向(单位长)。我们的重建图背景是**密度**、箭头
  只表朝向——要说清颜色是密度、箭头是方向不是速率。

## 5.1 Known artifact — "ghost density" from additive noise / 已知现象:加性噪声的"幽灵密度"

**EN.** In the partial-observation panels you see faint blue in cells where the true state has
no people. That is the additive Gaussian noise on truly-empty observed cells: **96% of observed
cells are truly empty** (the crowd is sparse), and on them the observed density is `0 + noise`
with σ≈0.057, ~31% of which is positive → light blue. It is honest sensor noise, NOT phantom
people; turning noise off makes those cells exactly 0. A count-appropriate (Poisson /
multiplicative) noise would leave empty cells at 0, but the paper uses additive Gaussian noise —
so this is a faithful consequence of the paper's noise model on a sparse field. Mention only if
asked.

**中文.** 部分观测图里,真值没人的格子有浅蓝——那是"空格子被观测到"时加的高斯噪声:**被观测的格子里
96% 真的是空的**(人群稀疏),它们的观测密度=`0+噪声`(σ≈0.057),约 31% 是正的 → 浅蓝。这是诚实的
传感器噪声,不是幽灵人;关掉噪声这些格子精确为 0。换成计数型(泊松/乘性)噪声空格子会保持 0,但论文
用加性高斯噪声——所以这是"论文噪声模型用在稀疏场"的忠实结果。被问到再提。

---

# 6. One-line summaries per slide / 每页一句话总结

| slide | one-liner |
|---|---|
| 1 title | 上次 4 项都处理了;端到端训练用论文窗口 dT=200 已完成。 |
| 2 noise | 4 个 σ = 0.25 × 每通道稳健标准差;前四步有据,只有 0.25 是假设。 |
| 3 real map | 丢数据驱动图,用真实 0.05m 图,有障碍就标墙 → 138 可行格。 |
| 4 rotation | 变换本就对(corr=−0.24,人避开墙),是显示问题不是旋转角错。 |
| 5 see-through | 根因是分辨率错配;墙也挡视线后穿墙对 3720→0、覆盖 70%→51%。 |
| 6 visualisation | vx/vy 是带符号分量(红蓝=两个方向)不是速率;x 沿走廊、y 横向。 |
| 7 framework | 最小化两项代价(拟合观测+先验);三部分 Ω/Φ/solver;跑 20 步。 |
| 8 training | 端到端:Φ 和 solver 一起训(非单独);监督 Eq.14、dT=200。 |
| 9 results | 盲区 MSE 0.079→0.038(baseline 0.083);重建整段序列,复现论文。 |
