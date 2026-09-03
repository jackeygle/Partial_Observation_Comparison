"""
paths.py — 仓库内路径的唯一真相

重构前，`/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf` 这个字符串在 22 个地方被写死，
另有若干处写死 `dincae_crowd`。2026-09-03 的目录重构让它们全部失效 —— 而这类失效不会
报错，只会让脚本读不到文件或者读到别人的文件。所以路径集中在这里，按**本文件的位置**
反推仓库根，跟着仓库走。

数据本身（ATC 的 h5、grid_cache、真实地图）不在这里 —— 它们在仓库之外，路径在
`config.yaml` 的 `data.root` / `navigation.map_dir` 里，那是给用户改的配置，不是代码常量。

用法:
    from crowdcore import paths
    paths.method("varnet")                  -> <root>/methods/varnet
    paths.runs("varnet")                    -> <root>/methods/varnet/runs
    paths.check_outputs("dincae")           -> <root>/methods/dincae/check_outputs
    paths.enkf_export("enkf_k1_full")       -> <root>/methods/varnet/check_outputs/enkf_k1_full
    paths.REFERENCE_IMPL                    -> /scratch/work/zhangx29/Partial_observation
"""
from __future__ import annotations

import os

#: 仓库根。本文件在 <root>/crowdcore/paths.py，所以上两级就是根。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CROWDCORE = os.path.join(ROOT, "crowdcore")
METHODS = os.path.join(ROOT, "methods")
COMPARE = os.path.join(ROOT, "compare")
SLIDES = os.path.join(ROOT, "slides")
BASELINE = os.path.join(ROOT, "refactor_baseline")

#: 五个方法的目录名。写在这里而不是各处字符串，改名时只有一处要动。
VARNET, ENKF, DINCAE, SENSEIVER = "varnet", "enkf", "dincae", "senseiver"

#: 参考实现（Kazemi Eskeri et al., IROS 2025 workshop）。**在本仓库之外**，
#: 我们只读它，从不写它 —— methods/enkf/enkf_lab/ 是它的只读 vendor 副本。
REFERENCE_IMPL = "/scratch/work/zhangx29/Partial_observation"


def method(name: str) -> str:
    """<root>/methods/<name>"""
    return os.path.join(METHODS, name)


def runs(name: str) -> str:
    """训练产出:checkpoint、metrics.jsonl、slurm 日志。"""
    return os.path.join(METHODS, name, "runs")


def run_dir(name: str, tag: str) -> str:
    """单个训练 run,例如 run_dir("varnet", "b0_k1") -> methods/varnet/runs/varnet_b0_k1。

    注意 varnet 的目录名带 `varnet_` 前缀（历史原因），这里不替调用方拼前缀，
    调用方传完整的目录名。
    """
    return os.path.join(METHODS, name, "runs", tag)


def check_outputs(name: str) -> str:
    """验证脚本的产出:指标 json、图、slurm 日志。"""
    return os.path.join(METHODS, name, "check_outputs")


def eval_out(name: str) -> str:
    """check_outputs/eval —— 各方法放最终指标与图的地方。"""
    return os.path.join(METHODS, name, "check_outputs", "eval")


def enkf_export(which: str = "enkf_k1_full") -> str:
    """EnKF 导出的估计场（est_*.npz / obs_*.npz），约 18 GB。

    这些文件**物理上留在 methods/varnet/check_outputs/ 下**，没有随 EnKF 代码搬到
    methods/enkf/。理由：重新生成要几十个 GPU 小时，而 /scratch 上搬动它们虽然是
    瞬间重命名，却会让所有引用它们的已有产出（compare4.json 等）对不上。
    需要挪的话改这一个函数。
    """
    return os.path.join(METHODS, VARNET, "check_outputs", which)


def enkf_vendor(which: str = "enkf_lab") -> str:
    """methods/enkf/enkf_lab 或 enkf_opt。

    这两个目录**刻意不是 Python 包的一部分**：它们内部有 `pedpred -> .` 自链接，
    靠把 `<dir>` 放进 sys.path 让包内的 `from pedpred.X import Y` 解析。
    见 methods/enkf/__init__.py。
    """
    assert which in ("enkf_lab", "enkf_opt"), which
    return os.path.join(METHODS, ENKF, which)
