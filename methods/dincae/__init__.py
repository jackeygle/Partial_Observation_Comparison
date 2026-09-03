"""DINCAE:卷积自编码器插补，逐通道输出均值与 sigma^2（信息形式）。

  state / encoding    通道定义、变换、逐格统计、输入编码
  dataset / model     数据与网络
  losses / train      损失与训练入口

注意它只在**该通道有定义**的格子上训练（空格子没有速度），这决定了它必须用哪套
评测口径 —— 见顶层 compare/compare4.py 的文件头。
"""
