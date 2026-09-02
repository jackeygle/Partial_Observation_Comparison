"""
compare4.py — 四方逐通道对比，「有定义格子」口径（Senseiver / 4DVarNet / EnKF / DINCAE）
========================================================================================

为什么要这个脚本。`compare3.py` 用的是 4dvarnet_enkf 的「所有格子」口径 —— 原始场、含非
walkable、四通道无权重。DINCAE 在那个口径下的盲区 MSE 是 0.5644，但那不是它的真实水平：
它从未在空格子的速度占位符 0 上训练过（`dincae_crowd/checks/evaluate.py` 的文件头写了这
件事，并且已经预告"报它是为了可比，不是因为它更对"）。证据是 `vy`：DINCAE 自己的口径下
0.1026，换成所有格子口径跳到 1.6489（16 倍），而 `vx` 几乎不动（0.546 → 0.571）。这种不
对称是口径伪影，不是模型性质。

反过来，DINCAE 自己那个 0.1047 也不能和 compare3 的 0.028~0.046 并排放 —— 分母根本不是
同一批格子。

所以这里把**另外三方也搬到「有定义格子」口径**上，让四方第一次落在同一张表里。

口径（四方完全一致）
--------------------
  * 有定义格子：density 处处有定义；vx/vy 要求 density>0；var 要求 var>0。
    唯一的定义在 `dincae_crowd/state.py:channel_valid()`，本脚本按路径 import 它，
    **不复制规则**（复制过来就会有第二个真相）。
  * ∩ walkable。已逐格核对 `dincae_crowd/artifacts/state_stats.npz["valid_mask"]` 与
    `nav.build_valid_mask_from_config()` 完全相同（290/432 格），所以两个项目喂给
    `generate_observations` 的 `valid_mask` 是同一个，盲区集合本来就一致。
  * ∩ 盲区（`~Omega`）。
  * 同一物理裁剪（EnKF 的界，density[0,5] / vx,vy[-5,5] / var[0,2]）。
  * 7 个留出日、整天、`obs_every_k=1`、`seed=0`、`obs_std` 同为 config 的值。
  * 帧范围 `[1, T-1)`：DINCAE 的 `FRESH_OFFSETS=(-1,0,1)` 逼它丢掉首尾各一帧，另外三方
    跟着丢，否则分母不同（这也是它 per_day 帧数比 Senseiver 少 2 的原因）。
  * 池化：先按通道累加平方误差与格数，最后一次相除。不是"按日平均再平均" ——
    `eval_threeway_accuracy.py` 的文件头解释过这两者不是同一个量。

「所有格子」口径同时并排算出来，这样两套口径的差别在同一张表里看得见，而不是要读者
自己去比两个 json。

DINCAE 那一行
-------------
从 `dincae_crowd/check_outputs/eval/dincae_metrics_test.json` 的 `ours_blind_mse` 读，不重跑
它的 16 个 checkpoint 输出平均。合并成"合计"时用的是**本脚本自己算出的逐通道格数**
（掩码相同，格数就相同），而不是它的 `ours_blind_mse_all_channels` —— 后者是它自己帧范围
下的分母，和这里对齐后的帧范围差 2 帧。

本脚本只读 4dvarnet_enkf 和 dincae_crowd，不写入它们的任何目录。

用法（GPU 节点）
----------------
    sbatch sbatch/submit_compare4.sbatch
    sbatch sbatch/submit_compare4.sbatch --days 1        # 冒烟，只跑第一天
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import dataset as ds
import sensors
from network import Senseiver

V4D = "/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf"
DINCAE = "/scratch/work/zhangx29/Thesis_Project/dincae_crowd"
for p in (V4D, os.path.join(V4D, "checks")):
    if p not in sys.path:
        sys.path.append(p)
from model_io import load_solver                                    # noqa: E402
import navigation as nav                                            # noqa: E402


def _load_isolated(name, path):
    """按路径加载一个模块，给它一个不会和本项目撞名的名字。

    dincae_crowd 和 senseiver_crowd 都有 dataset.py / losses.py / model.py，把 dincae 的根
    塞进 sys.path 会把本项目的模块顶掉。dincae 的 state.py 只依赖 numpy/h5py 和
    4dvarnet_enkf 的 navigation/observation_model（都已在 sys.path 上），所以可以单独加载。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dstate = _load_isolated("dincae_state", os.path.join(DINCAE, "state.py"))
channel_valid = _dstate.channel_valid
StateStats = _dstate.StateStats

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
BASE_CONVENTIONS = ("defined", "allcells", "full")
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
    import observation_model as om
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
    base = {"defined": blind & cv & w,
            "allcells": blind,
            "full": np.ones_like(blind)}                          # 全场 = 观测 + 盲区
    base.update({f"{k}_noclip": v for k, v in base.items()})      # 同一批格子，不裁剪打分
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senseiver", default="runs/senseiver_A/best.pt")
    ap.add_argument("--varnet", default="a4_k1,b0_k1",
                    help="逗号分隔的 4dvarnet run 名（runs/varnet_<名>/varnet_best.pt）")
    ap.add_argument("--de-fmt", default="vsb0_s{}",
                    help="不确定性头（NLL）深度集成的 run 名模板；留空则不评它")
    ap.add_argument("--de-members", default="0,1,2,3,4")
    ap.add_argument("--enkf-dir", default=os.path.join(V4D, "check_outputs", "enkf_k1_full"))
    ap.add_argument("--dincae-json",
                    default=os.path.join(DINCAE, "check_outputs", "eval",
                                         "dincae_metrics_test.json"))
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--varnet-batch", type=int, default=16)
    ap.add_argument("--out", default="check_outputs/eval/compare4.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C, H, W = ds.state_shape()
    chans = ds.channels()

    # walkable：两个项目的掩码已核对相同，这里断言一次，别让它悄悄漂掉
    walk = StateStats(os.path.join(DINCAE, "artifacts", "state_stats.npz")).valid
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
        sol, va, _ = load_solver(os.path.join(V4D, f"runs/varnet_{vn}/varnet_best.pt"), dev)
        vsolvers[vn] = (sol, va)

    # 不确定性头（NLL 损失）的深度集成。点估计是各成员重建的均值（论文 Sec 2.4），
    # 所以"集成"和"单成员均值±std"是两个不同的量，两个都报 —— 只报集成会把
    # "平均带来的好处"和"损失函数的影响"混在一起。
    de_members = [z.strip() for z in args.de_members.split(",") if z.strip()] \
        if args.de_fmt else []
    de_solvers, de_dT = [], None
    for m in de_members:
        sol, va, _ = load_solver(
            os.path.join(V4D, f"runs/varnet_{args.de_fmt.format(m)}/varnet_best.pt"), dev)
        de_solvers.append(sol)
        de_dT = va["dT"]
    print(f"[model] Senseiver {sm.num_params:,} 参数 | "
          + " | ".join(f"4DVarNet {vn} dT={va['dT']} n_iter={sol.n_iter}"
                       for vn, (sol, va) in vsolvers.items())
          + f" | walkable {int(walk.sum())}/{walk.size} | device={dev}", flush=True)

    days = ds.om.split_files("test")
    if args.days:
        days = days[:args.days]

    de_names = [f"4DVarNet nll s{m}" for m in de_members]
    names = ["Senseiver"] + [f"4DVarNet {vn}" for vn in vnames] \
        + (["4DVarNet nll ens%d" % len(de_members)] if de_solvers else []) \
        + de_names + ["EnKF k1"]
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

        if de_solvers:
            ens = None
            for m, sol in zip(de_members, de_solvers):
                pv, nkeep = run_varnet(sol, Yf, Omc, X0, de_dT, dev, args.varnet_batch)
                b = min(hi, nkeep)
                dacc[f"4DVarNet nll s{m}"].add(clip_np(pv[lo:b]), pv[lo:b], Xf[lo:b],
                                               cut(sel_all, lo, b))
                ens = pv if ens is None else ens + pv      # 累加，别同时留 5 份整天数组
                del pv
            ens /= len(de_solvers)
            b = min(hi, ens.shape[0])
            dacc[f"4DVarNet nll ens{len(de_solvers)}"].add(
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
        dincae_pc = dj.get("ours_blind_mse")
        dincae_all = dj.get("v4dvar_blind_mse")
        if int(dj.get("n_days", 0)) != len(days):
            print(f"\n  [warn] DINCAE 那一行是 {dj.get('n_days')} 天池化的，本次只跑了 "
                  f"{len(days)} 天 —— 两者不可比，这张表只能用来看代码通不通。", flush=True)
    else:
        print(f"  [warn] 找不到 {args.dincae_json}，DINCAE 行留空", flush=True)
        dincae_all = None

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
        for conv, pc in (("defined", dincae_pc), ("allcells", dincae_all)):
            if not pc:
                continue
            nn = ref.n[conv]
            se = sum(pc[c] * nn[i] for i, c in enumerate(chans))
            entry[conv] = {"per_channel": {c: float(pc[c]) for c in chans},
                           "n_per_channel": {chans[i]: int(v) for i, v in enumerate(nn)},
                           "overall": float(se / max(nn.sum(), 1)),
                           "source": os.path.relpath(args.dincae_json, DINCAE)}
        res["results"]["DINCAE"] = entry

    print("\n\n全场 RMSE（观测 + 盲区所有格子）—— 池化 vs 按日平均 vs 不裁剪\n")
    print(f"{'方法':<20}{'池化':>10}{'按日平均':>12}{'池化,不裁':>12}{'按日平均,不裁':>16}")
    print("-" * 70)
    for k, q in res["results"].items():
        if "full" not in q:
            continue
        print(f"{k:<20}{q['full']['rmse_pooled']:>10.4f}"
              f"{q['full']['rmse_mean_of_days']:>12.4f}"
              f"{q['full_noclip']['rmse_pooled']:>12.4f}"
              f"{q['full_noclip']['rmse_mean_of_days']:>16.4f}")

    for conv, title in (("defined", "有定义格子 ∩ walkable ∩ 盲区（四方可比的口径）"),
                        ("allcells", "所有格子 ∩ 盲区（compare3 的旧口径，供对照）")):
        print(f"\n{title}\n")
        print(f"{'方法':<20}" + "".join(f"{c:>11}" for c in chans)
              + f"{'合计':>11}{'合计RMSE':>11}")
        print("-" * (20 + 11 * (len(chans) + 2)))
        for k, q in res["results"].items():
            if conv not in q:
                continue
            pc, ov = q[conv]["per_channel"], q[conv]["overall"]
            print(f"{k:<20}" + "".join(f"{pc[c]:>11.4f}" for c in chans)
                  + f"{ov:>11.4f}{np.sqrt(ov):>11.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
