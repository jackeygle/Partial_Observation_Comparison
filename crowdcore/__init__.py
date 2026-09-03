"""crowdcore — 五个方法共用的核心。

这里放的是**与方法无关**的东西:数据从哪来、机器人怎么走、观测怎么生成、哪些格子
可行走、哪个通道在哪些格子上有定义。任何一个方法的模型、损失、训练循环都不属于这里。

  config              config.yaml 的读取（网格、观测、导航、先验的全部参数）
  navigation          walkable 区域、A* 路径规划、真实 ATC 地图
  observation_model   多机器人传感器的移动与观测生成、Omega/Omega_c、X0 填充
  data                CSV -> h5 -> 网格 的数据管线

在这之前这四样住在 4dvarnet_enkf/ 里，另外两个方法靠硬编码
`/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf` 的绝对路径去 import 它们
（全仓库 72 处）。搬到这里之后，共享关系是显式的。
"""
