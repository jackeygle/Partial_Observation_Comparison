"""
compare5.py — 五方逐通道对比，四套格子口径并列
==============================================

五个方法（都在 7 个留出日的整天上、obs_every_k=1、seed=0、同一物理裁剪）:

    Senseiver                    稀疏传感器 -> 场
    4DVarNet MSE   单成员/集成    Eq.14 的 plain 平方损失
    4DVarNet NLL   单成员/集成    高斯 NLL + σ̂ 读出头（"不确定性头"）
    DINCAE                       卷积自编码器插补（16 个 checkpoint 输出平均）
    EnKF k1                      局部化 EnKF，Partial_observation 的 vendor 副本

MSE 臂与 NLL 臂的先验/求解器架构**逐权重相同**（hidden 32、kt 3、lstm_hidden 64、
GENN 9,474），唯一差别是 NLL 那边多了 `grad_net.out_var` —— 692,000 参数的 σ̂ 读出头。
所以损失函数的影响在单成员和集成两个层级上都是干净对比。

为什么要四套口径
----------------
**名次会随口径翻转。**同一份预测，在「所有格子」口径下 Senseiver 第一、DINCAE 差 12 倍
垫底；换成「有定义格子」后 DINCAE 第一、Senseiver 第三。

原因在速度通道：空格子里没有人，也就没有速度，数据管线在那里存的 `0` 是占位符而不是
测量值，而盲区里 88.4% 的格子是空的。三个在完整场上训练的方法学会了"空格子输出 0"
白拿这部分分；DINCAE 只在有定义的格子上训练过，就被这一项压垮 —— 它的 `vy` 在自己
口径下是 0.1026，换成所有格子口径跳到 1.6489（16 倍），而 `vx` 几乎不动（0.546→0.571）。
这种不对称是口径伪影，不是模型性质。

所以四套并列，让差别在同一份输出里看得见，而不是要读者自己去比几个 json：

    ① defined        盲区 ∩ 有定义 ∩ walkable      主结果，五方可比
    ② defined_full   全场 ∩ 有定义 ∩ walkable      含观测格子
    ③ allcells       盲区，所有格子                compare3 的旧口径
    ④ full           全场，所有格子                eval_test_days 的 full_mse

每套再乘「裁剪/不裁剪」和「池化/按日平均」，因为这两组也各自被混用过：
`test_metrics_*.json` 报按日平均，`eval_threeway_accuracy.py` 报池化，两者不是同一个量。

口径细节
--------
  * 有定义格子：density 处处有定义；vx/vy 要求 density>0；var 要求 var>0。唯一的定义在
    `methods/dincae/state.py:channel_valid()`，本脚本直接 import 它，**不复制规则**。
  * ∩ walkable。已逐格核对 `methods/dincae/artifacts/state_stats.npz["valid_mask"]` 与
    `nav.build_valid_mask_from_config()` 完全相同（290/432 格），所以两边喂给
    `generate_observations` 的 `valid_mask` 是同一个，盲区集合本来就一致（脚本里有断言）。
  * 帧范围 `[1, T-1)`：DINCAE 的 `FRESH_OFFSETS=(-1,0,1)` 逼它丢掉首尾各一帧，其余方法
    跟着丢，否则分母不同。
  * 池化：先按通道累加平方误差与格数，最后一次相除。

DINCAE 那一行
-------------
从 `methods/dincae/check_outputs/eval/dincae_metrics_test.json` 读，不重跑它的 16 个
checkpoint 输出平均。它自己的 evaluate.py 恰好就报四套口径，与这里一一对应
（`ours_blind` / `ours_all` / `v4dvar_blind` / `v4dvar_all`）。合并成"合计"时用的是
**本脚本自己算出的逐通道格数**，而不是它 json 里的 all_channels 值 —— 后者的分母是
它自己的帧范围。

公平性提醒：DINCAE 这一行**本身就是 16 个 checkpoint 的输出平均**，天然带集成优势，
该和"集成"那几行比，不该和单模型比。

复现下限
--------
4DVarNet 的数字有 ~4e-4 的相对抖动：`GradSolver` 在**推理时**也要走 autograd 反传，
conv backward 的 atomicAdd 归约顺序每次不同。Senseiver（no_grad）、EnKF（numpy 读 npz）、
DINCAE（读 json）都逐位可复现。见 `refactor_baseline/README.md`。
**4DVarNet 的第 4 位小数是噪声，不要报那个量级的差别。**

用法（GPU 节点，从仓库根）
--------------------------
    source sbatch/_env.sh
    python3 -m compare.compare5
    python3 -m compare.compare5 --days 1 --ensembles ""     # 冒烟，跳过集成
"""
from __future__ import annotations

import argparse
import json
import os
from crowdcore import paths
import sys

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver

from methods.varnet.checks.model_io import load_solver
from crowdcore import navigation as nav
# 「有定义格子」规则的唯一定义。2026-09-03 重构前这里是一段 importlib.spec_from_file_location
# 的 hack —— 因为 dincae 和 senseiver 都有 dataset.py/losses.py/model.py，把 dincae 的根塞进
# sys.path 会把本项目的模块顶掉。包化之后直接 import 就行。
from methods.dincae.state import StateStats, channel_valid

# DINCAE 的 FRESH_OFFSETS = (-1, 0, 1)，见 dincae_crowd/encoding.py:30。写死在这里并断言，
# 免得为了一个常量去 import 那边会撞名的 encoding.py。
FRESH_OFFSETS = (-1, 0, 1)
FRAME_LO = -min(FRESH_OFFSETS)
FRAME_HI_PAD = max(FRESH_OFFSETS)

LO = np.array([0.0, -5.0, -5.0, 0.0], np.float32)
HI = np.array([5.0, 5.0, 5.0, 2.0], np.float32)

# 三套格子选择 × 裁剪/不裁剪。`full` 是「全场」——观测 + 盲区所有格子，即
# eval_test_days.py 的 full_mse 和 eval_threeway_accuracy.py 的 rmse_all 那个量；
# 放进来是为了在同一条代码路径上核对那两个脚本对 4DVarNet 报的 0.1702 与 0.2104。
# `_noclip` 存在的理由：threeway 不裁剪，test_metrics 裁剪，这是它们唯一已知的口径差。
BASE_CONVENTIONS = ("defined", "defined_full", "allcells", "full")
CONVENTIONS = BASE_CONVENTIONS + tuple(f"{k}_noclip" for k in BASE_CONVENTIONS)


def clip_np(x):
    return np.clip(x, LO[None, :, None, None], HI[None, :, None, None])


class Acc:
    """逐通道、逐口径累计平方误差与格数（池化用）。"""

    def __init__(self, C):
        self.se = {k: np.zeros(C) for k in CONVENTIONS}
        self.n = {k: np.zeros(C, dtype=np.int64) for k in CONVENTIONS}

    def add(self, pred_clip, pred_raw, true, sel):
        """pred/true (T,C,H,W)；sel 是 {口径: (T,C,H,W) bool} 的掩码字典。

        `*_noclip` 的口径用未裁剪的 pred_raw 打分，其余用 pred_clip。
        """
        d2c = (pred_clip - true) ** 2
        d2r = (pred_raw - true) ** 2
        for k, m in sel.items():
            d2 = d2r if k.endswith("_noclip") else d2c
            for c in range(d2.shape[1]):
                self.se[k][c] += float(d2[:, c][m[:, c]].sum())
                self.n[k][c] += int(m[:, c].sum())

    def mse(self, k):
        return self.se[k] / np.maximum(self.n[k], 1)

    def overall(self, k):
        return self.se[k].sum() / max(self.n[k].sum(), 1)

    def merge(self, other):
        for k in CONVENTIONS:
            self.se[k] += other.se[k]
            self.n[k] += other.n[k]


def run_senseiver(model, Y, Om, dev, batch):
    pe = model.pos_enc.cpu().numpy()
    mu = model.in_mean.cpu().numpy()
    sd = model.in_std.cpu().numpy()
    outs = []
    with torch.no_grad():
        for i in range(0, Y.shape[0], batch):
            sl = slice(i, i + batch)
            tok, pad, _ = sensors.build_batch(Y[sl], Om[sl], pe, mu, sd)
            outs.append(model.reconstruct(tok.to(dev), pad.to(dev)).cpu().numpy())
    return np.concatenate(outs, 0)


def run_varnet(solver, Y, Omc, X0, dT, dev, batch):
    """返回 (重建 (nw*dT, C, H, W), 覆盖的帧数)。与 compare3.run_varnet 同一路径。"""
    from crowdcore import observation_model as om
    win = lambda a: om.to_windows(a, dT)
    Yw, Mw, X0w = win(Y), win(Omc.astype(np.float32)), win(X0)
    outs = []
    with torch.enable_grad():
        for i in range(0, Yw.shape[0], batch):
            yb = torch.from_numpy(Yw[i:i + batch]).float().to(dev)
            mb = torch.from_numpy(Mw[i:i + batch]).float().to(dev)
            xb = torch.from_numpy(X0w[i:i + batch]).float().to(dev)
            outs.append(solver(xb, yb, mb).detach().cpu().numpy())
    r = np.concatenate(outs, 0)                          # (nw, C, dT, H, W)
    nw, C, dt, H, W = r.shape
    return r.transpose(0, 2, 1, 3, 4).reshape(nw * dt, C, H, W), nw * dt


def build_masks(X, Omf, walk, C):
    """两套口径的盲区掩码，(T,C,H,W) bool。

    X    (T,C,H,W) 真值（原始物理值 —— channel_valid 在原始值上判定）
    Omf  (T,H,W)   观测掩膜（四通道同时被观测，所以是逐格而非逐通道）
    walk (H,W)     walkable
    """
    blind = ~np.repeat(Omf[:, None], C, axis=1)                  # (T,C,H,W)
    cv = np.moveaxis(channel_valid(X), 0, 1)                     # (NCH,T,H,W) -> (T,C,H,W)
    w = walk[None, None]
    base = {"defined": blind & cv & w,          # 盲区 ∩ 有定义 ∩ walkable
            "defined_full": cv & w,                # 有定义 ∩ walkable，观测格子也算
            "allcells": blind,                     # 盲区，所有格子
            "full": np.ones_like(blind)}           # 全场 = 观测 + 盲区，所有格子
    base.update({f"{k}_noclip": v for k, v in base.items()})      # 同一批格子，不裁剪打分
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senseiver",
                    default=os.path.join(paths.runs(paths.SENSEIVER),
                                         "senseiver_A", "best.pt"))
    ap.add_argument("--varnet", default="a4_k1,b0_k1",
                    help="逗号分隔的 4dvarnet run 名（runs/varnet_<名>/varnet_best.pt）")
    ap.add_argument("--ensembles", default="MSE=mse5_s{},NLL=vsb0_s{}",
                    help="逗号分隔的 `标签=run名模板`。每个集成会出 N 个单成员行加一个"
                         "集成行。留空则不评任何集成。")
    ap.add_argument("--ensemble-members", default="0,1,2,3,4")
    ap.add_argument("--enkf-dir", default=paths.enkf_export("enkf_k1_full"))
    ap.add_argument("--dincae-json",
                    default=os.path.join(paths.eval_out(paths.DINCAE),
                                         "dincae_metrics_test.json"))
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--varnet-batch", type=int, default=16)
    # 输出落在 compare/ 自己的目录：这个量是跨方法的，不属于任何一个方法的
    # check_outputs。重构前它写在 senseiver_crowd/check_outputs/eval/ 下，
    # 让人以为那是 Senseiver 的指标。
    ap.add_argument("--out", default=os.path.join(paths.COMPARE, "results",
                                                  "compare5.json"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = ds.state_shape()
    chans = ds.channels()

    # walkable：两个项目的掩码已核对相同，这里断言一次，别让它悄悄漂掉
    walk = StateStats(os.path.join(paths.method(paths.DINCAE),
                                   "artifacts", "state_stats.npz")).valid
    walk_4d = nav.build_valid_mask_from_config()
    assert walk.shape == walk_4d.shape and (walk == walk_4d).all(), \
        "dincae 的 walkable 与 4dvarnet 的不再一致，四方口径的前提被破坏了"

    sck = torch.load(args.senseiver, map_location=dev, weights_only=False)
    sm = Senseiver(**sck["hparams"]).to(dev)
    sm.load_state_dict(sck["model"])
    sm.eval()

    vnames = [z.strip() for z in args.varnet.split(",") if z.strip()]
    vsolvers = {}
    for vn in vnames:
        sol, va, _ = load_solver(os.path.join(paths.runs(paths.VARNET), f"varnet_{vn}", "varnet_best.pt"), dev)
        vsolvers[vn] = (sol, va)

    # 深度集成。点估计是各成员重建的均值（论文 Sec 2.4），所以"集成"和"单成员均值±std"
    # 是两个不同的量，两个都报 —— 只报集成会把"平均带来的好处"和"损失函数的影响"混在一起。
    #
    # 两个集成并列的意义：mse5 与 vsb0 的先验/求解器架构**逐权重相同**（hidden 32、
    # kt 3、lstm_hidden 64、GENN 9,474），唯一差别是 vsb0 多了 grad_net.out_var 这个
    # 692,000 参数的 σ̂ 读出头。所以 MSE-vs-NLL 在单成员和集成两个层级上都是干净对比。
    # （曾经以为要拿 b0_k1 和 vsb0 比会混进架构差异 —— 那是误记，b0_k1 本来就同架构。）
    ens_members = [z.strip() for z in args.ensemble_members.split(",") if z.strip()]
    ensembles = {}                                   # 标签 -> [solver, ...]
    for spec in (z.strip() for z in args.ensembles.split(",") if z.strip()):
        label, _, fmt = spec.partition("=")
        assert fmt, f"--ensembles 的每一项要写成 标签=run名模板，收到 {spec!r}"
        sols, edT = [], None
        for m in ens_members:
            sol, va, _ = load_solver(
                os.path.join(paths.runs(paths.VARNET), f"varnet_{fmt.format(m)}",
                             "varnet_best.pt"), dev)
            sols.append(sol)
            edT = va["dT"]
        ensembles[label] = (sols, edT)
    print(f"[model] Senseiver {sm.num_params:,} 参数 | "
          + " | ".join(f"4DVarNet {vn} dT={va['dT']} n_iter={sol.n_iter}"
                       for vn, (sol, va) in vsolvers.items())
          + f" | walkable {int(walk.sum())}/{walk.size} | device={dev}", flush=True)

    days = ds.om.split_files("test")
    if args.days:
        days = days[:args.days]

    names = ["Senseiver"] + [f"4DVarNet {vn}" for vn in vnames]
    for label, (sols, _) in ensembles.items():
        names.append(f"4DVarNet {label} ens{len(sols)}")
        names += [f"4DVarNet {label} s{m}" for m in ens_members]
    names.append("EnKF k1")
    accs = {k: Acc(C) for k in names}
    # 按日的 overall，用来算"按日平均"那个口径。test_metrics_*.json / eval_test_days.py 报的是
    # 它，eval_threeway_accuracy.py 报的是池化 —— 两者不是同一个量，这里同时给出。
    day_overall = {k: {c: [] for c in CONVENTIONS} for k in names}
    per_day = []

    for d in days:
        stem = os.path.basename(d).split("_")[0]
        X, Y, Om = ds.load_day(d, stride=1, seed=0)
        n = X.shape[0]
        Xf = X.reshape(n, C, H, W)
        Yf = Y.reshape(n, C, H, W)
        Omf = Om.reshape(n, H, W)

        lo, hi = FRAME_LO, n - FRAME_HI_PAD                  # DINCAE 能评的帧范围
        sel_all = build_masks(Xf, Omf, walk, C)
        cut = lambda m, a, b: {k: v[a:b] for k, v in m.items()}
        dacc = {k: Acc(C) for k in names}                    # 这一天单独攒，才能算按日平均

        p = run_senseiver(sm, Y, Om, dev, args.batch)
        dacc["Senseiver"].add(clip_np(p[lo:hi]), p[lo:hi], Xf[lo:hi], cut(sel_all, lo, hi))
        del p

        Omc = np.repeat(Omf[:, None], C, axis=1)
        X0 = ds.om.fill_missing_state(Yf, Omc, method=ds.obs_config()["init_method"])
        for vn in vnames:
            sol, va = vsolvers[vn]
            pv, nkeep = run_varnet(sol, Yf, Omc, X0, va["dT"], dev, args.varnet_batch)
            b = min(hi, nkeep)                               # dT 对不齐时丢的尾巴
            dacc[f"4DVarNet {vn}"].add(clip_np(pv[lo:b]), pv[lo:b], Xf[lo:b],
                                       cut(sel_all, lo, b))
            del pv

        for label, (sols, edT) in ensembles.items():
            ens = None
            for m, sol in zip(ens_members, sols):
                pv, nkeep = run_varnet(sol, Yf, Omc, X0, edT, dev, args.varnet_batch)
                b = min(hi, nkeep)
                dacc[f"4DVarNet {label} s{m}"].add(clip_np(pv[lo:b]), pv[lo:b], Xf[lo:b],
                                                   cut(sel_all, lo, b))
                ens = pv if ens is None else ens + pv      # 累加，别同时留 5 份整天数组
                del pv
            ens /= len(sols)
            b = min(hi, ens.shape[0])
            dacc[f"4DVarNet {label} ens{len(sols)}"].add(
                clip_np(ens[lo:b]), ens[lo:b], Xf[lo:b], cut(sel_all, lo, b))
            del ens

        ep = os.path.join(args.enkf_dir, f"est_{stem}.npz")
        if os.path.exists(ep):
            est = np.load(ep)["Est"].astype(np.float32)
            b = min(hi, est.shape[0])
            dacc["EnKF k1"].add(clip_np(est[lo:b]), est[lo:b], Xf[lo:b], cut(sel_all, lo, b))
            del est
        else:
            print(f"  [warn] {stem} 没有 EnKF 导出", flush=True)

        rec = {"day": stem, "frames_total": int(n), "frames_scored": int(hi - lo)}
        for k, a in dacc.items():
            if a.n["full"].sum() == 0:
                continue
            accs[k].merge(a)
            for c in CONVENTIONS:
                day_overall[k][c].append(float(a.overall(c)))
            rec[k] = {"full_mse": float(a.overall("full")),
                      "defined_blind_mse": float(a.overall("defined"))}
        per_day.append(rec)
        print(f"  {stem}: {n} 帧, 评 [{lo},{hi})  "
              + "  ".join(f"{k} full={rec[k]['full_mse']:.4f}" for k in dacc if k in rec),
              flush=True)

    # --- DINCAE：读它已经算好的「有定义格子」逐通道 MSE ---------------------------
    dincae_pc = None
    if os.path.exists(args.dincae_json):
        with open(args.dincae_json) as f:
            dj = json.load(f)
        # DINCAE 自己的 evaluate.py 恰好就报这四套，与本脚本的四个口径一一对应
        dincae_by_conv = {
            "defined":      dj.get("ours_blind_mse"),      # 有定义 ∩ walkable ∩ 盲区
            "defined_full": dj.get("ours_all_mse"),        # 有定义 ∩ walkable ∩ 全场
            "allcells":     dj.get("v4dvar_blind_mse"),    # 所有格子 ∩ 盲区
            "full":         dj.get("v4dvar_all_mse"),      # 所有格子 ∩ 全场
        }
        dincae_pc = dincae_by_conv["defined"]
        if int(dj.get("n_days", 0)) != len(days):
            print(f"\n  [warn] DINCAE 那一行是 {dj.get('n_days')} 天池化的，本次只跑了 "
                  f"{len(days)} 天 —— 两者不可比，这张表只能用来看代码通不通。", flush=True)
    else:
        print(f"  [warn] 找不到 {args.dincae_json}，DINCAE 行留空", flush=True)
        dincae_by_conv = {}

    # --- 输出 -------------------------------------------------------------------
    res = {"protocol": {
        "cells": "blind ∩ channel-defined ∩ walkable（'defined'）与 blind（'allcells'）并列",
        "channel_defined": "density: 全部; vx/vy: density>0; var: var>0"
                           "（dincae_crowd/state.py:channel_valid）",
        "walkable_cells": int(walk.sum()), "grid_cells": int(walk.size),
        "frame_range": f"[{FRAME_LO}, T-{FRAME_HI_PAD})  （对齐 DINCAE 的 FRESH_OFFSETS）",
        "obs_every_k": 1, "seed": 0, "clip": "EnKF 物理界",
        "pooling": "逐通道累加平方误差与格数，最后一次相除",
        "n_days": len(days)},
        "per_day": per_day, "channels": chans, "results": {}}

    for k, a in accs.items():
        if a.n["defined"].sum() == 0:
            continue
        res["results"][k] = {
            conv: {"per_channel": {chans[i]: float(v) for i, v in enumerate(a.mse(conv))},
                   "n_per_channel": {chans[i]: int(v) for i, v in enumerate(a.n[conv])},
                   "overall": float(a.overall(conv)),
                   # 同一批数字的两种汇总方式，摆在一起就不会再被混用
                   "overall_pooled": float(a.overall(conv)),
                   "overall_mean_of_days": float(np.mean(day_overall[k][conv])),
                   "rmse_pooled": float(np.sqrt(a.overall(conv))),
                   "rmse_mean_of_days": float(np.mean(
                       [np.sqrt(v) for v in day_overall[k][conv]]))}
            for conv in CONVENTIONS}

    # DINCAE 的合计用本脚本的格数重算，理由见文件头
    if dincae_pc:
        ref = next(iter(accs.values()))
        entry = {}
        for conv, pc in dincae_by_conv.items():
            if not pc:
                continue
            nn = ref.n[conv]
            se = sum(pc[c] * nn[i] for i, c in enumerate(chans))
            ov = float(se / max(nn.sum(), 1))
            # 字段与其他方法对齐，方便下游画图代码一视同仁。**没有** rmse_mean_of_days：
            # DINCAE 那一行是从它自己的 json 读的池化值，我们手里没有它的逐日数据。
            entry[conv] = {"per_channel": {c: float(pc[c]) for c in chans},
                           "n_per_channel": {chans[i]: int(v) for i, v in enumerate(nn)},
                           "overall": ov, "overall_pooled": ov,
                           "rmse_pooled": float(np.sqrt(ov)),
                           "source": os.path.relpath(args.dincae_json, paths.ROOT)}
        res["results"]["DINCAE"] = entry

    print("\n\n全场 RMSE（观测 + 盲区所有格子）—— 池化 vs 按日平均 vs 不裁剪\n")
    print(f"{'方法':<24}{'池化':>10}{'按日平均':>12}{'池化,不裁':>12}{'按日平均,不裁':>16}")
    print("-" * 74)
    for k, q in res["results"].items():
        if "full" not in q:
            continue
        cell = lambda c, f: (f"{q[c][f]:.4f}" if c in q and f in q[c] else "—")
        print(f"{k:<24}{cell('full','rmse_pooled'):>10}"
              f"{cell('full','rmse_mean_of_days'):>12}"
              f"{cell('full_noclip','rmse_pooled'):>12}"
              f"{cell('full_noclip','rmse_mean_of_days'):>16}")

    for conv, title in (
            ("defined", "① 有定义格子 ∩ walkable ∩ 盲区  —— 五方可比，主结果"),
            ("defined_full", "② 有定义格子 ∩ walkable ∩ 全场（含观测格子）"),
            ("allcells", "③ 所有格子 ∩ 盲区  —— compare3 的旧口径，供对照"),
            ("full", "④ 所有格子 ∩ 全场  —— eval_test_days 的 full_mse 那个量")):
        print(f"\n{title}\n")
        print(f"{'方法':<24}" + "".join(f"{c:>11}" for c in chans)
              + f"{'合计':>11}{'合计RMSE':>11}")
        print("-" * (24 + 11 * (len(chans) + 2)))
        for k, q in res["results"].items():
            if conv not in q:
                continue
            pc, ov = q[conv]["per_channel"], q[conv]["overall"]
            print(f"{k:<24}" + "".join(f"{pc[c]:>11.4f}" for c in chans)
                  + f"{ov:>11.4f}{np.sqrt(ov):>11.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
