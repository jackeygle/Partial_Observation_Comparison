"""4DVarNet:用一个学到的优化器最小化变分代价 J。

  prior_model         GENN 先验 Phi
  variational_solver  GradSolver —— 展开的梯度下降 + ConvLSTM
  losses              Eq.14 的 plain MSE 与高斯 NLL
  train               训练入口
  checks/             本方法自己的诊断（跨方法的在顶层 compare/）
"""
