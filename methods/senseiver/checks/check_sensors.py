"""
check_sensors.py — `sensors.build_batch` 的不变量自检
=====================================================

这是移植后的第一道关。参考实现的传感器集是定长的，padding 与 pad_mask 这条路
它从未走过；我们必须自己证明它是对的，否则后面所有数字都不可信。

检查项：
  1. 每帧的 token 数 == 该帧 Omega 的观测格数
  2. pad_mask 与 padding 位逐位对齐（True 恰好落在无效位）
  3. token 的前 C 维 == 该格 Y 值经逐通道标准化后的结果
  4. token 的后 P 维 == 该格在 pos_enc 里的那一行
  5. 空观测帧不产生 NaN，且被标出一个有效的哑 token
  6. **机器人位置**落在 walkable 内，但**观测格允许越界**——4dvarnet_enkf 的 README
     明写机器人 observe `disk ∩ line-of-sight`，*not* intersected with the walkable
     mask（"a robot can see a pillar it cannot drive into"）。这一条同时是我们
     "查询全部 432 格、取消参考实现 pix_avail" 这个决定的独立佐证：非 walkable 的
     格子既被观测也被评分，不能当成"没有值可重建"的区域。
     （注意 observation_model.generate_observations 的 docstring 说的是
     "only observe walkable cells"，与 README 和实际数据不符；以数据为准。）

用法（纯 numpy/torch，无需训练；仍按项目规矩放 GPU 节点跑）:
    python3 checks/check_sensors.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.positional import PositionalEncoder

FAIL = []


def ck(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAIL.append(name)


def main():
    C, H, W = ds.state_shape()
    HW = H * W
    day = ds.om.split_files("test")[0]
    print(f"[data] {os.path.basename(day)}  前 300 帧")
    X, Y, Om = ds.load_day(day, stride=1, seed=0, frames=300)
    pe = PositionalEncoder((H, W, C), 16).numpy().astype(np.float32)
    mean = np.array([0.06, 0.03, 0.0, 0.02], np.float32)
    std = np.array([0.17, 0.46, 0.16, 0.11], np.float32)

    B = 16
    idx = np.arange(B)
    tok, pad, n = sensors.build_batch(Y[idx], Om[idx], pe, mean, std)
    tok, pad, n = tok.numpy(), pad.numpy(), n.numpy()
    print(f"[shape] tokens {tok.shape}  pad_mask {pad.shape}  "
          f"传感器数 min {n.min()} max {n.max()}")

    ck("1. token 数 == Omega 观测格数", bool((n == Om[idx].sum(1)).all()))
    ck("2. pad_mask 与有效位对齐",
       bool(all((~pad[b]).sum() == max(n[b], 1) for b in range(B))))

    b = int(np.argmax(n))                       # 取观测最多的一帧逐值核对
    ii = np.flatnonzero(Om[idx][b])
    want_v = ((Y[idx][b][:, ii].T - mean) / std).astype(np.float32)
    ck("3. 前 C 维 == 标准化后的 Y", np.allclose(tok[b, :len(ii), :C], want_v, atol=1e-6))
    ck("4. 后 P 维 == pos_enc 对应行", np.allclose(tok[b, :len(ii), C:], pe[ii], atol=1e-6))
    ck("   padding 位全零", bool(np.abs(tok[b, len(ii):]).max() == 0.0) if len(ii) < tok.shape[1] else True)

    # 5. 空观测帧
    Om0 = Om[idx].copy(); Om0[0] = False
    t0, p0, n0 = sensors.build_batch(Y[idx], Om0, pe, mean, std)
    ck("5. 空观测帧无 NaN 且留一个有效哑 token",
       bool(np.isfinite(t0.numpy()).all() and (~p0.numpy()[0]).sum() == 1 and n0.numpy()[0] == 0))

    # 6. 机器人位置 vs 观测足迹
    Xd, _ = ds.om.load_state(day)
    Xd = np.asarray(Xd)[:300]
    valid2 = ds.nav.build_valid_mask_from_config(Xd)
    valid = valid2.reshape(-1)
    out = ds.om.generate_observations(Xd, valid_mask=valid2, seed=0)
    pos = out["positions"].reshape(-1, 2)
    ck("6. 机器人位置全在 walkable 内", bool(valid2[pos[:, 0], pos[:, 1]].all()),
       f"walkable {int(valid.sum())}/{HW}")
    seen = np.flatnonzero(Om.any(0))
    frac = 1.0 - valid[seen].mean()
    print(f"  INFO  观测格中落在非 walkable 的比例 {frac*100:.1f}% "
          f"(机器人能看见开不进去的格子 -> 必须查询全部 {HW} 格)")

    print("\n" + ("ALL PASS" if not FAIL else f"FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
