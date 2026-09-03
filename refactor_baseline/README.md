# `refactor_baseline/` — 重构的验收基线

冻结于 **2026-09-03**,提交 `2b2d121` 之后、目录重构之前。

重构只允许改变**代码在哪里**,不允许改变**代码算出什么**。这个目录就是那句话的可执行版本:
重构完成后,同样的命令必须重新产出这里的数字,**逐位相同**。

## 内容

| 文件 | 来源 | 它保证什么 |
|---|---|---|
| `compare4.json` | `senseiver_crowd/checks/compare4.py`(作业 20041056) | 5 个模型 × 6 个口径 × 4 通道的池化/按日平均误差。这是主结果,也是覆盖面最广的一个测试 —— 它同时经过 `config`、`navigation`、`observation_model`、Senseiver、4DVarNet 和 EnKF 的导出 |
| `verify_enkf_opt.out` | `4dvarnet_enkf/checks/verify_enkf_opt.py --frames 12 --ensemble 100` | `enkf_opt` 与 `enkf_lab` 逐位相同。`enkf_lab` 是 `Partial_observation` 的只读 vendor 副本,这条是它存在的全部意义 |
| `threeway_accuracy.json` | `4dvarnet_enkf/checks/eval_threeway_accuracy.py` | 留作记录。**注意它的 4DVarNet 绝对值与 `compare4.json` 差约 15%,原因未定位**,所以它不是可信基线,只用来确认重构没有让它变得*更*不一样 |
| `dincae_metrics_test.json` | `dincae_crowd/checks/evaluate.py` | DINCAE 的四套口径 |
| `senseiver_metrics_test.json` | `senseiver_crowd/checks/evaluate.py` | Senseiver 的逐日/逐通道 |

## 怎么验收

    # 4DVarNet 那几行有 ~4e-4 的固有抖动，见下面"复现下限"，所以容差按实测取
    python3 refactor_baseline/check_against_baseline.py <新产出的 json> --tol 3e-4

EnKF 那条单独跑:

    # 重构后
    python3 -m methods.enkf.checks.verify_enkf_opt --frames 12 --ensemble 100
    diff <(...) refactor_baseline/verify_enkf_opt.out

## 为什么要有这个目录

三个子项目不是 Python 包,靠 `sys.path.insert/append` 拼起来,`losses.py` 有三份同名的,
72 处硬编码绝对路径散在 21 个文件里。重构要动的正是这些,而它们**全是**"改错了不会报错、
只会静默算出别的数字"的那类东西。没有数值基线的话,重构完根本无法判断有没有改坏。

重构完成、验收通过后,这个目录可以删,或者留着当回归测试。


## 复现下限：4DVarNet 的数字只在前 3 位有效

重构后 compare4 的 957 个数值里有 521 处与基线不同，全部集中在 4DVarNet 那几行，
最大绝对差 2.39e-04（相对 4.2e-04）。Senseiver、EnKF、DINCAE 三方**逐位相同**。

这不是重构造成的。同一台机器、同一份代码、连跑两次（作业 20051734）：

| 方法 | 两次跑的最大相对差 |
|---|---|
| Senseiver | 0.00e+00 |
| EnKF k1 | 0.00e+00 |
| DINCAE | 0.00e+00 |
| **4DVarNet b0_k1** | **4.18e-04** |

和重构前后的 4.20e-04 是同一个量级 —— 也就是说 4DVarNet 的这个指标本身就没法逐位复现。

原因：`GradSolver` 在**推理时**也要走 `torch.enable_grad()` 做 15~20 次反传（那是它的
求解方式，不是训练），而 conv backward 的归约用 atomicAdd，累加顺序每次不同。
Senseiver 在 `torch.no_grad()` 下只做前向，所以逐位可复现；EnKF 从 npz 读，纯 numpy。

**对结果解读的影响**：4DVarNet 的任何数字，第 4 位小数是噪声。
"a4 比 b0 好 15%" 这种结论远在下限之上（0.1109 vs 0.1284），安全；
但不要报 "0.1782 vs 0.1781" 这种量级的差别，那在噪声里。
