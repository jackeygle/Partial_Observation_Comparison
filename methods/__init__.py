"""methods — 每个子包是一个方法，互不依赖。

  varnet      4DVarNet 变分同化（plain MSE 与 NLL/不确定性头两条臂）
  enkf        局部化 EnKF —— 是 Partial_observation 的 vendor 副本，见 enkf/README
  dincae      DINCAE 卷积自编码器插补
  senseiver   Senseiver 稀疏传感器重建

方法之间**不允许互相 import**。跨方法的评测与画图在顶层 compare/ 里。
共用的观测模型/导航/配置在顶层 crowdcore/ 里。
"""
