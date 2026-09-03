"""
check_encoding.py — information-form 编码的自检（论文版）
=========================================================

这是 `encoding.py` 的模块自测，不是"实验"。检查的都是**不变量**，任何一条为假都说明
编码写错了，而这类错误的特点是**不报错、只让结果变差**，等训练不好的时候分不清是模型
问题还是编码 bug。

检查项：
  1. 形状、无非有限值
  2. 缺测处两片皆 0（论文的核心语义：缺测 = 精度为零）
  3. 观测到处第二片 = 1（σ²_obs 取常数 1，见 encoding.py）
  4. 观测到处第一片 = 观测值 − 逐格均值场（残差）
  5. 目标：每通道在"无定义"处精度为 0（density 处处有定义；vx/vy 需 density>0；
     var 需 vel_var>0），非 walkable 处精度为 0
  6. 参考量：观测覆盖率、以及"3 帧窗口内至少被观测到一次"的比例
     —— 后者直接说明论文原版在 ATC 上能拿到多少观测信息

用法（纯 numpy/scipy，登录节点可跑）:
    python3 check_encoding.py
"""
from __future__ import annotations

import os
import sys

import h5py
import numpy as np

# checks/ 里的脚本从项目根导入源码模块；根要插在最前(本目录的 losses.py 优先)，
# 4dvarnet_enkf 只能 append(它也有 losses.py，插到最前会把本目录的顶掉)
from crowdcore import observation_model as om                                    # noqa: E402

from methods.dincae.state import (CHANNEL_TRANSFORM, CHANNELS, StateStats,  # noqa: E402
                         NCH, channel_valid, fwd_channel)
from methods.dincae.dataset import encode_day, obs_config                         # noqa: E402
from methods.dincae.encoding import (FRESH_OFFSETS, N_IN, N_STATIC, N_TGT,        # noqa: E402
                      encode_target, observed_pair)


def main():
    stats = StateStats()
    print(f"逐格均值场: {stats.n_train_days} 天, walkable {stats.valid.sum()}/{stats.valid.size}")
    print(f"通道 {CHANNELS};  N_IN={N_IN}  N_TGT={N_TGT}  ntime_win={len(FRESH_OFFSETS)}")
    print(f"逐通道变换 {CHANNEL_TRANSFORM}")

    fp = om.split_files("valid")[0]                    # 留出日
    with h5py.File(fp, "r") as f:
        X, t_unix = f["grid"][:6000], f["time"][:6000]
    print(f"dev day {os.path.basename(fp)}  T={X.shape[0]}")

    oc = obs_config()
    obs = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"], seed=oc["seed"], valid_mask=stats.valid,
        obs_std=oc["obs_std"], obs_every_k=oc["obs_every_k"])
    Y, M = obs["Y"][:, :NCH], obs["Omega_c"][:, :NCH]
    scaled, invvar = observed_pair(Y, M, stats.mean, stats.std)
    tgt = encode_target(X, stats.mean, stats.std, stats.valid)

    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'OK  ' if cond else 'FAIL'} {name}")

    print("\n=== 不变量 ===")
    for nm, arr in (("scaled", scaled), ("invvar", invvar), ("target", tgt)):
        chk(f"{nm} {arr.shape} 全为有限值", np.isfinite(arr).all())

    chk("缺测处 scaled==0", (scaled[~M] == 0).all())
    chk("缺测处 invvar==0", (invvar[~M] == 0).all())
    chk("观测处 invvar==1", (invvar[M] == 1).all())
    resid = np.stack([(fwd_channel(Y[:, c].astype(np.float64), c) - stats.mean[c][None])
                     / stats.std[c] for c in range(NCH)], axis=1)
    chk("观测处 scaled == 归一化(fwd(观测值)−逐格均值场)",
        np.allclose(scaled[M], resid[M], atol=1e-5))

    cv = channel_valid(X)
    chk("目标: 每通道无定义处精度==0",
        all((tgt[:, 2 * c + 1][~cv[c]] == 0).all() for c in range(NCH)))
    nw = ~np.broadcast_to(stats.valid[None], cv[0].shape)
    chk("目标: 非 walkable 处精度==0",
        all((tgt[:, 2 * c + 1][nw] == 0).all() for c in range(NCH)))
    chk("目标: 有定义处第一片 == 归一化(fwd(truth)−逐格均值场)",
        all(np.allclose(tgt[:, 2 * c][cv[c] & ~nw],
                        ((fwd_channel(X[:, c].astype(np.float64), c) - stats.mean[c][None])
                         / stats.std[c])[cv[c] & ~nw], atol=1e-5)
            for c in range(NCH)))

    # 端到端：encode_day 的通道拼装
    xin, ytg = encode_day(fp, stats, stride=97, max_frames=6000)
    chk(f"encode_day 形状 {xin.shape} / {ytg.shape}",
        xin.shape[1] == N_IN and ytg.shape[1] == N_TGT)
    chk("encode_day 无非有限值", np.isfinite(xin).all() and np.isfinite(ytg).all())
    o = N_STATIC + NCH                                  # 第一个 dt 的 invvar 块
    chk("拼装后的 invvar 块只含 0/1",
        np.isin(xin[:, o:o + NCH], (0.0, 1.0)).all())

    print("\n=== 参考量（论文原版在 ATC 上能拿到多少观测）===")
    Om = obs["Omega"]
    v = stats.valid
    print(f"  单帧覆盖 walkable 格子: {100 * Om[:, v].mean():.1f}%")
    win = np.zeros_like(Om[0], dtype=bool)
    seen = []
    for t in range(1, Om.shape[0] - 1):
        win = Om[t - 1] | Om[t] | Om[t + 1]
        seen.append(win[v].mean())
    print(f"  3 帧窗口内至少一次: {100 * np.mean(seen):.1f}%"
          f"   -> 其余 {100 - 100 * np.mean(seen):.1f}% 的格子两片全 0，"
          "只能靠坐标+时钟+逐格均值场")
    for c in range(NCH):
        print(f"  目标有定义比例 {CHANNELS[c]:8s}: "
              f"{100 * (cv[c] & ~nw).mean():.1f}% of (帧×格)")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
