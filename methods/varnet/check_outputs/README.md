# check_outputs/ — 验证脚本的输出（图/报告，全部可重新生成）

`checks/` 里的脚本负责**验证和画图**，输出统一放在这里，按模块分子目录：

| 子目录 | 内容 | 生成脚本 |
|---|---|---|
| `data_pipeline/` | pipeline_1~4_*.png：CSV→H5→grid 流程解释图（坐标变换/三角核/4通道/分窗） | `checks/plot_data_pipeline.py` |
| `navigation/` | astar_paths.png：A* 路径在真实走廊掩码上的可视化 | `checks/check_navigation.py` |
| `navigation/` | nav_mask_on_map.png / map_orientation_check.png：真实地图导航掩码 + 朝向诊断 | `checks/check_map_orientation.py` |
| `observation_model/` | frame_*.png（掩码生成四联图）、coverage_vs_range.png、io_report.txt、stats.json | `checks/check_observation_model.py` |
| `training/` | varnet_training.png：盲区 MSE 收敛曲线（数据来自 runs/varnet/metrics.jsonl） | `slides/build_slides.py` |

全部重新生成：

```bash
module load scicomp-pytorch-env/2026.1
python3 checks/check_navigation.py
python3 checks/plot_data_pipeline.py
python3 checks/check_observation_model.py --file first --frames 60 --init prev
python3 slides/build_slides.py          # 训练曲线 + PPT/PDF/讲稿
```

注意：`checks/check_observation_model.py` 每次运行会**清空**自己的输出子目录再写入。
