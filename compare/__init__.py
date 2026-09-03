"""compare — 跨方法的评测与画图，只有这里可以同时 import 多个方法。

存在的理由:在这之前跨方法脚本散在两个地方 —— compare3/compare4 住在
senseiver_crowd/checks/（却要读全部四方），eval_threeway_accuracy 住在
4dvarnet_enkf/checks/。结果是同一个量有两个互相矛盾的脚本，4DVarNet 的
盲区 RMSE 一个报 0.1912、一个报 0.2198，差 15% 至今没定位。

一个量只应该有一处实现。新的跨方法评测放这里，不要放回某个方法的 checks/。
"""
