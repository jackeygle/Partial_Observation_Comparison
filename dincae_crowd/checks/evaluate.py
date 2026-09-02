"""
evaluate.py — 在留出日上评估 DINCAE，并做论文的三项检验
========================================================

三件事，都来自论文：

1. **多 epoch 输出平均**（1.0 Fig.3 / 参考实现 `save_epochs = 200:10:epochs` 是默认行为）
   把后段每 10 个 epoch 存下的 checkpoint 的**输出**平均（不是权重平均），比任何单个 epoch
   都好。σ̂² 同样平均。1.0 也提醒：忽略不同 epoch 之间误差的相关性会**高估** σ̂。

2. **σ̂ 校准**（2.0 §5.2 Fig.9b/10b）
   按预测 SD 把格子分 10 档（在预测 SD 的 10%~90% 分位之间均匀切），每档算实际 RMS，
   画"实际 SD vs 预测 SD"。理想情况落在对角线上。
   2.0 还对 σ̂ 施加了一个**全局调整因子**，让平均 RMS 对上平均预测 SD —— 也就是说原始 σ̂
   的绝对尺度是有偏的，可信的是它的结构与排序。这里把调整前后都报出来。

3. **变率保持**（1.0 Fig.8 / 2.0 Table 3）
   重建场的标准差 vs 真值的标准差。RMSE 类指标偏爱平滑场（double penalty，1.0 引
   Gilleland 2009 / Ebert 2013），所以必须另外报变率，否则"更平滑"会被误读成"更好"。

**两套 MSE 口径都报**，因为它们会给出不同结论（这是 4dvarnet_enkf 项目里已经踩过的坑）：

  * `ours`  : 只在**该通道有定义**的格子上算（速度要求 density>0，var 要求 vel_var>0），
              且限制在 walkable 内。这是我们训练时的口径。
  * `v4dvar`: 完全照 `4dvarnet_enkf/checks/eval_test_days.py` 的口径 —— 原始场、所有格子
              （含非 walkable）、四通道无权重、盲区 = `mask < 0.5`，并施加与 EnKF 相同的
              物理裁剪 (density[0,5], vx/vy[-5,5], var[0,2])。
              注意这个口径会在**我们从未训练过的格子**（非 walkable、空格子的速度占位符 0）
              上打分，所以对我们不利；报它是为了可比，不是因为它更对。

用法（**GPU 节点**）:
    sbatch sbatch/submit_eval.sbatch                 # 默认 test split
    # 或
    srun -p gpu-debug --gres=gpu:1 -t 00:14:00 bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u evaluate.py --split valid --days 1 --frames 4000'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

# checks/ 里的脚本从项目根导入源码模块；根要插在最前(本目录的 losses.py 优先)，
# 4dvarnet_enkf 只能 append(它也有 losses.py，插到最前会把本目录的顶掉)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
import observation_model as om                                    # noqa: E402

from state import (CHANNELS, StateStats, NCH, channel_valid,  # noqa: E402
                         fwd_channel, inv_channel)
from dataset import obs_config                                     # noqa: E402
from encoding import (FRESH_OFFSETS, N_IN, N_STATIC, observed_pair,  # noqa: E402
                      static_channels)
from model import DINCAE                                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 与 4dvarnet_enkf/checks/eval_test_days.py:clip_bounds 完全一致（EnKF 的物理界）
CLIP = ((0.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (0.0, 2.0))


def load_models(run_dir, ckpt_glob, dev):
    """载入要做输出平均的 checkpoint 列表。"""
    paths = sorted(glob.glob(ckpt_glob or os.path.join(run_dir, "ckpt_*.pt")))
    if not paths:                                    # 没有中间 checkpoint 就用 last.pt
        paths = [os.path.join(run_dir, "last.pt")]
    models, epochs = [], []
    for p in paths:
        st = torch.load(p, map_location=dev)
        a = st["args"]
        m = DINCAE(N_IN, NCH, enc_internal=tuple(a["enc"]),
                   loss_weights=tuple(a["loss_weights"]), pool=a["pool"]).to(dev)
        m.load_state_dict(st["model"]); m.eval()
        models.append(m); epochs.append(int(st["epoch"]))
    return models, epochs, paths


def build_inputs(scaled, invvar, t_unix, idx, H, W):
    """按 dataset.encode_day 的同一布局拼输入（这里是评估路径，逐块处理以省内存）。"""
    out = np.empty((len(idx), N_IN, H, W), dtype=np.float32)
    out[:, :N_STATIC] = static_channels(t_unix[idx], H, W)
    o = N_STATIC
    for dt in FRESH_OFFSETS:
        j = idx + dt
        out[:, o:o + NCH] = scaled[j]
        out[:, o + NCH:o + 2 * NCH] = invvar[j]
        o += 2 * NCH
    assert o == N_IN, (o, N_IN)
    return out


@torch.no_grad()
def predict_day(models, stats, fp, dev, frames=0, batch=256):
    """一天的重建（多 checkpoint 输出平均）。

    返回 (Xt, rec, mu_n, sd_n, M)：
      Xt   (n,NCH,H,W) 真值，**原始物理单位**
      rec  (n,NCH,H,W) 重建，**原始物理单位**（已反归一化、加回逐格均值、并做逆变换；
           log1p 通道取**中位数** expm1(μ)，不做对数正态均值修正 —— 理由见
           `state.inv_channel`）
      mu_n (n,NCH,H,W) 归一化残差空间的均值预测
      sd_n (n,NCH,H,W) 归一化残差空间的预测标准差
      M    (n,NCH,H,W) bool 观测掩膜

    σ̂ 校准在**归一化空间**做：高斯假设活在那里，而校准看的是 actual/pred 的比值，
    本身无量纲，所以在哪个空间做都一样，在模型自己的空间做最干净（log1p 通道尤其）。
    """
    oc = obs_config()
    with h5py.File(fp, "r") as f:
        T_all = f["grid"].shape[0]
        T = min(T_all, frames) if frames else T_all
        X = f["grid"][:T]
        t_unix = f["time"][:T]
    obs = om.generate_observations(
        X, sensing_range=oc["sensing_range"], num_agents=oc["num_agents"],
        add_noise=oc["add_noise"], seed=oc["seed"], valid_mask=stats.valid,
        obs_std=oc["obs_std"], obs_every_k=oc["obs_every_k"])
    Y, M = obs["Y"][:, :NCH], obs["Omega_c"][:, :NCH]
    scaled, invvar = observed_pair(Y, M, stats.mean, stats.std)

    lo, hi = -min(FRESH_OFFSETS), T - max(FRESH_OFFSETS)
    idx_all = np.arange(lo, hi, dtype=np.int64)
    H, W = X.shape[2], X.shape[3]
    std = stats.std.reshape(1, NCH, 1, 1)
    mean = stats.mean[None]

    mu_n = np.empty((len(idx_all), NCH, H, W), dtype=np.float32)
    s2_n = np.empty_like(mu_n)
    for b in range(0, len(idx_all), batch):
        idx = idx_all[b:b + batch]
        xb = torch.from_numpy(build_inputs(scaled, invvar, t_unix, idx, H, W)).to(dev)
        m_sum = torch.zeros(len(idx), NCH, H, W, device=dev)
        s2_sum = torch.zeros_like(m_sum)
        for m in models:                             # 输出平均（1.0 Fig.3）
            mo, s2 = m(xb)[-1]                       # 取最后一级（精化后的输出）
            m_sum += mo; s2_sum += s2
        k = len(models)
        mu_n[b:b + len(idx)] = (m_sum / k).cpu().numpy()
        s2_n[b:b + len(idx)] = (s2_sum / k).cpu().numpy()

    # 归一化残差 -> 变换空间的绝对值 -> 原始物理值
    x_sp = mu_n * std + mean                                 # 变换空间（log1p 通道仍是 log）
    rec = np.empty_like(x_sp)
    for c in range(NCH):
        # 用**中位数** expm1(μ)，不做对数正态均值修正 —— 后者在 σ̂ 未校准时会炸
        # （实测 var 的物理 MSE 达 10²²）。理由见 `state.inv_channel`。
        rec[:, c] = inv_channel(x_sp[:, c], c)
    return X[idx_all], rec, mu_n, np.sqrt(np.maximum(s2_n, 0.0)), M[idx_all]


def clip_bounds(x):
    x = x.copy()
    for c, (lo, hi) in enumerate(CLIP):
        np.clip(x[:, c], lo, hi, out=x[:, c])
    return x


def accumulate(acc, Xt, rec, mu_n, sd_n, M, stats):
    """把一天的统计量累加进 acc（逐通道；两套 MSE 口径 + 校准 + 变率）。

    MSE 与变率在**物理空间**；校准在**归一化空间**（高斯假设所在，见 predict_day）。
    """
    cv = channel_valid(Xt)                                    # (NCH,n,H,W)
    walk = stats.valid[None]
    blind = ~M                                                # 未被观测
    rec_clip = clip_bounds(rec)

    for c in range(NCH):
        d2 = (rec[:, c] - Xt[:, c]) ** 2
        d2c = (rec_clip[:, c] - Xt[:, c]) ** 2
        # --- ours: 该通道有定义 ∩ walkable。**也施加物理裁剪** ---
        # 物理界（density≥0、var∈[0,2]…）是先验已知的，两套口径都可以用。必须裁的原因：
        # 信息形式下 `m = x₁·σ̂²` 而 σ̂² 上限是 1/µ = 1000（Eq.6 钳位），未收敛的模型能输出
        # μ≈1000，log1p 通道再一反变换就是天文数字，几个格子就能统治整条 MSE。
        # 同时保留 `*_noclip` 让这种病态可见，而不是被裁剪悄悄藏起来。
        ours = cv[c] & walk
        for tag, sel in (("ours_blind", ours & blind[:, c]), ("ours_all", ours)):
            a = acc[tag][c]
            a["se"] += float(d2c[sel].sum()); a["n"] += int(sel.sum())
        for tag, sel in (("ours_blind_noclip", ours & blind[:, c]),
                         ("ours_all_noclip", ours)):
            a = acc[tag][c]
            a["se"] += float(d2[sel].sum()); a["n"] += int(sel.sum())
        # --- v4dvar: 所有格子、裁剪后 ---
        for tag, sel in (("v4dvar_blind", blind[:, c]),
                         ("v4dvar_all", np.ones_like(blind[:, c]))):
            a = acc[tag][c]
            a["se"] += float(d2c[sel].sum()); a["n"] += int(sel.sum())
        # --- 变率保持（1.0 Fig.8）：在 ours 口径的格子上比标准差 ---
        a = acc["var_retention"][c]
        a["st"] += float(Xt[:, c][ours].sum()); a["st2"] += float((Xt[:, c][ours] ** 2).sum())
        a["sr"] += float(rec_clip[:, c][ours].sum())          # 裁剪后，同上
        a["sr2"] += float((rec_clip[:, c][ours] ** 2).sum())
        a["n"] += int(ours.sum())
        # --- 校准：盲区 ∩ 有定义，在归一化空间收集 (预测 SD, 平方误差) ---
        sel = ours & blind[:, c]
        tgt_n = (fwd_channel(Xt[:, c].astype(np.float64), c)
                 - stats.mean[c][None]) / stats.std[c]
        acc["calib"][c]["sd"].append(sd_n[:, c][sel].astype(np.float32))
        acc["calib"][c]["se"].append(((mu_n[:, c] - tgt_n) ** 2)[sel].astype(np.float32))


def calibration_table(sd, se, nbin=10):
    """2.0 §5.2 的做法：按预测 SD 在 p10~p90 之间均匀分 nbin 档，每档算实际 RMS。"""
    if len(sd) == 0:
        return []
    lo, hi = np.percentile(sd, [10, 90])
    edges = np.linspace(lo, hi, nbin + 1)
    rows = []
    for i in range(nbin):
        m = (sd >= edges[i]) & (sd < edges[i + 1] if i < nbin - 1 else sd <= edges[i + 1])
        if m.sum() < 100:
            continue
        rows.append({"pred_sd": float(sd[m].mean()),
                     "actual_sd": float(np.sqrt(se[m].mean())),
                     "n": int(m.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "runs", "dincae_full"))
    ap.add_argument("--ckpt-glob", default="")
    ap.add_argument("--split", default="test", choices=["test", "valid"])
    ap.add_argument("--days", type=int, default=0, help="只用前 N 天(调试)")
    ap.add_argument("--frames", type=int, default=0, help="每天只用前 N 帧(调试)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--calib-sample", type=int, default=4_000_000,
                    help="校准直方图最多保留多少个点(随机下采样)")
    ap.add_argument("--out", default=os.path.join(ROOT, "check_outputs", "eval"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("!! 没有 GPU —— torch 不要在登录节点跑（见 README）", flush=True)

    stats = StateStats()
    models, epochs, paths = load_models(args.run_dir, args.ckpt_glob, dev)
    print(f"输出平均使用 {len(models)} 个 checkpoint: epochs {epochs}")

    files = om.split_files(args.split)
    if args.days:
        files = files[: args.days]
    print(f"{args.split} split: {len(files)} 天")

    mk = lambda: [{"se": 0.0, "n": 0} for _ in range(NCH)]
    TAGS = ("ours_blind", "ours_all", "ours_blind_noclip", "ours_all_noclip",
            "v4dvar_blind", "v4dvar_all")
    acc = {t: mk() for t in TAGS}
    acc["var_retention"] = [{"st": 0.0, "st2": 0.0, "sr": 0.0, "sr2": 0.0, "n": 0}
                            for _ in range(NCH)]
    acc["calib"] = [{"sd": [], "se": []} for _ in range(NCH)]
    per_day = []

    for fp in files:
        Xt, rec, mu_n, sd_n, M = predict_day(models, stats, fp, dev, args.frames, args.batch)
        # 当日的 ours_blind MSE（逐天记录，便于看稳定性）
        cv = channel_valid(Xt); walk = stats.valid[None]
        day_mse = {}
        for c in range(NCH):
            sel = cv[c] & walk & (~M[:, c])
            day_mse[CHANNELS[c]] = (float(((clip_bounds(rec)[:, c] - Xt[:, c]) ** 2)[sel].mean())
                                    if sel.any() else None)
        per_day.append({"day": os.path.splitext(os.path.basename(fp))[0],
                        "frames": int(Xt.shape[0]), "ours_blind_mse": day_mse})
        print(f"  {per_day[-1]['day']}  " +
              "  ".join(f"{k} {v:.5f}" for k, v in day_mse.items() if v is not None),
              flush=True)
        accumulate(acc, Xt, rec, mu_n, sd_n, M, stats)
        del Xt, rec, mu_n, sd_n, M

    rng = np.random.default_rng(0)
    result = {"run_dir": args.run_dir, "split": args.split,
              "checkpoints": [os.path.basename(p) for p in paths], "epochs": epochs,
              "n_days": len(files), "per_day": per_day, "channels": list(CHANNELS)}

    for tag in TAGS:
        result[tag + "_mse"] = {CHANNELS[c]: (acc[tag][c]["se"] / acc[tag][c]["n"]
                                             if acc[tag][c]["n"] else None)
                                for c in range(NCH)}
        # 四通道无权重合并（4dvarnet_enkf 报的就是这个合并数）
        se = sum(acc[tag][c]["se"] for c in range(NCH))
        n = sum(acc[tag][c]["n"] for c in range(NCH))
        result[tag + "_mse_all_channels"] = se / n if n else None

    result["var_retention"] = {}
    for c in range(NCH):
        a = acc["var_retention"][c]
        n = max(a["n"], 1)
        st = np.sqrt(max(a["st2"] / n - (a["st"] / n) ** 2, 0.0))
        sr = np.sqrt(max(a["sr2"] / n - (a["sr"] / n) ** 2, 0.0))
        result["var_retention"][CHANNELS[c]] = {
            "truth_sd": float(st), "rec_sd": float(sr),
            "ratio": float(sr / st) if st > 0 else None}

    result["calibration"] = {}
    for c in range(NCH):
        sd = np.concatenate(acc["calib"][c]["sd"]) if acc["calib"][c]["sd"] else np.array([])
        se = np.concatenate(acc["calib"][c]["se"]) if acc["calib"][c]["se"] else np.array([])
        if len(sd) > args.calib_sample:
            j = rng.choice(len(sd), args.calib_sample, replace=False)
            sd, se = sd[j], se[j]
        rows = calibration_table(sd, se)
        # 全局调整因子（2.0 §5.2）：让平均预测 SD 对上实际 RMS
        adj = (float(np.sqrt(se.mean()) / sd.mean()) if len(sd) and sd.mean() > 0 else None)
        result["calibration"][CHANNELS[c]] = {
            "bins": rows, "global_adjust_factor": adj,
            "mean_pred_sd": float(sd.mean()) if len(sd) else None,
            "actual_rms": float(np.sqrt(se.mean())) if len(se) else None,
            "n": int(len(sd))}

    path = os.path.join(args.out, f"dincae_metrics_{args.split}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== MSE（两套口径）===")
    print(f"{'口径':16s} " + "  ".join(f"{c:>9s}" for c in CHANNELS) + "   四通道合并")
    for tag in TAGS:
        vals = "  ".join(f"{result[tag + '_mse'][c]:9.5f}"
                         if result[tag + "_mse"][c] is not None else "        -"
                         for c in CHANNELS)
        print(f"{tag:16s} {vals}   {result[tag + '_mse_all_channels']:.5f}")

    print("\n=== 变率保持（重建 SD / 真值 SD，1 最好；<1 = 被抹平）===")
    for c in CHANNELS:
        v = result["var_retention"][c]
        print(f"  {c:8s} truth {v['truth_sd']:.4f}  rec {v['rec_sd']:.4f}  "
              f"ratio {v['ratio']:.3f}" if v["ratio"] else f"  {c}: -")

    print("\n=== σ̂ 校准（盲区、归一化空间；理想是 actual ≈ pred）===")
    for c in CHANNELS:
        v = result["calibration"][c]
        if not v["bins"]:
            print(f"  {c}: 样本不足"); continue
        print(f"  {c:8s} 平均预测SD {v['mean_pred_sd']:.4f}  实际RMS {v['actual_rms']:.4f}  "
              f"全局调整因子 {v['global_adjust_factor']:.3f}  (n={v['n']:,})")
        for r in v["bins"]:
            print(f"      pred {r['pred_sd']:.4f} -> actual {r['actual_sd']:.4f}  "
                  f"(n={r['n']:,})")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
