"""EnKF 基线 —— 不是我们写的，是 vendor 进来的。

  enkf_lab/   /scratch/work/zhangx29/Partial_observation 的逐字节只读副本。
              **不要编辑**，文件是刻意 chmod 444 的。
  enkf_opt/   允许修改的副本；任何改动都必须通过 checks/verify_enkf_opt.py 的
              逐位比对（np.array_equal，不是 isclose）。

注意 enkf_lab/pedpred/ 与 enkf_opt/pedpred/ **刻意不是**本包的子包:它们里面有
`pedpred -> .` 自链接，靠把那一层放进 sys.path 让包内的 `from pedpred.X import Y`
解析 —— 这是原项目的布局，动它就等于改 vendor 副本。所以这两个目录没有从
methods.enkf 继续往下的 __init__.py，按路径加载即可。
"""
