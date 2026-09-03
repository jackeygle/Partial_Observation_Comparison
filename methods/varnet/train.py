"""
train_varnet.py  —  train the variational reconstruction (GENN prior + solver)
===============================================================================

Real training entry point (GPU, on Triton). Wires the three components:
    Step-1 data (X, Y, Ω, X0)  ->  GENN prior Φ  ->  GradSolver  ->  x_rec

and trains Φ + solver jointly to reconstruct the full macroscopic field from the
robots' partial observations.

Loss (paper):
    --loss supervised   :  ‖x_rec − X_true‖²              (Eq. 14, uses ground truth)
    --loss unsupervised :  ‖(x_rec−y)⊙Ω‖² + ‖x_rec−Φ(x_rec)‖²   (Eq. 13, no GT)

Run (GPU):
    sbatch submit_varnet.sbatch --days 8 --dT 7 --n-iter 15 --epochs 30
or a quick debug:
    srun --partition=gpu-debug --gres=gpu:1 --time=00:15:00 \
         bash -c 'module load scicomp-pytorch-env/2026.1; python3 -u train_varnet.py --days 1 --steps 50'
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext

import numpy as np
import torch

from crowdcore import config
from crowdcore import observation_model as d
from crowdcore import navigation as nav
from methods.varnet.prior_model import GENN
from methods.varnet.variational_solver import GradSolver
from methods.varnet.losses import compute_loss

_T = config.CFG["training"]                              # training defaults (config.yaml)


def _parse_sched(spec, default_lr):
    """Parse --iter-schedule "epoch:n_iter[:lr],..." into a sorted [(epoch, n_iter, lr)] list.

    Implements the paper's §3.4 curriculum ("increase incrementally ... from 5 iterations to
    20 ones"). lr is optional per stage and falls back to --lr; the official code steps both
    together (lit_model.py's iter_update / nb_grad_update / lr_update triple).
    """
    stages = []
    for part in spec.split(","):
        f = part.strip().split(":")
        if len(f) not in (2, 3):
            raise ValueError(f"bad --iter-schedule stage {part!r}; want epoch:n_iter[:lr]")
        stages.append((int(f[0]), int(f[1]), float(f[2]) if len(f) == 3 else default_lr))
    stages.sort()
    if stages[0][0] != 0:
        raise ValueError("--iter-schedule must define a stage at epoch 0")
    return stages


def _sched_at(stages, epoch):
    """(n_iter, lr) in force at `epoch` — the last stage whose start epoch has been reached."""
    n_it, lr = stages[0][1], stages[0][2]
    for e0, n, l in stages:
        if epoch >= e0:
            n_it, lr = n, l
    return n_it, lr


def build_windows(split_files, dT, sensing_range, num_agents, seed, max_days,
                  add_noise=None, init_method=None, obs_every_k=None):
    """Build the training tensors by cutting each day into dT-frame windows.

    For every day we produce four aligned tensors, each shaped (N, C, dT, H, W):
        X   ground-truth full field  (the target the model must reconstruct)
        Y   partial noisy observation (0 where a cell was not observed)
        M   observation mask Omega_c  (1.0 = that (channel,cell,frame) was observed)
        X0  initial guess fed to the solver (observations filled into the blind cells)
    where N = number of windows, C = 4 channels, dT = frames per window, H×W = 36×12.
    A "window" is dT CONSECUTIVE frames; windows do not overlap.

    add_noise / init_method / obs_every_k default to config.yaml when not given (single
    source). obs_every_k is overridable per run so a k=1 and a k=4 experiment can train
    side by side without fighting over the one global config value.
    """
    if add_noise is None:
        add_noise = config.get("observation", "add_noise")
    if init_method is None:
        init_method = config.get("observation", "init_method")
    Xs, Ys, Ms, X0s = [], [], [], []
    for fp in split_files[:max_days]:
        # X_day: the whole day's ground-truth field, shape (n_frames, C, H, W).
        X_day, _ = d.load_state(fp)
        # walkable mask: hybrid of "walked in training data" AND the real ATC map (config rule).
        valid = nav.build_valid_mask_from_config(X_day)
        # drop the tail frames so the length is an exact multiple of dT (clean windows).
        n = (X_day.shape[0] // dT) * dT
        X_day = X_day[:n]
        # Component 1: simulate the moving robots -> partial noisy observation Y and its mask.
        out = d.generate_observations(X_day, sensing_range, num_agents,
                                      add_noise=add_noise, seed=seed, valid_mask=valid,
                                      obs_every_k=obs_every_k)
        # solver's starting point: fill the unobserved cells (e.g. carry the previous frame).
        X0 = d.fill_missing_state(out["Y"], out["Omega_c"], method=init_method)

        win = lambda a: d.to_windows(a, dT)             # -> (n_windows, C, dT, H, W)
        Xs.append(win(X_day)); Ys.append(win(out["Y"]))
        Ms.append(win(out["Omega_c"].astype(np.float32))); X0s.append(win(X0))
    # concatenate all days along the window axis and hand back float32 torch tensors.
    cat = lambda L: torch.from_numpy(np.concatenate(L, 0)).float()
    return cat(Xs), cat(Ys), cat(Ms), cat(X0s)


def evaluate(solver, X, Y, M, X0, unobs, batch, dev, channels):
    """Score the reconstruction on a fixed subset. Returns (blind MSE, full-state MSE, per-channel).

    torch.enable_grad() even though nothing is being trained: the solver computes dJ/dx by
    autograd INSIDE each of its iterations, so it cannot run at all with grad disabled. The
    output is detached because we are only measuring.
    """
    solver.eval()
    with torch.enable_grad():
        recs = [solver(*(t[i:i + batch].to(dev) for t in (X0, Y, M))).detach().cpu()
                for i in range(0, X.shape[0], batch)]
    xr = torch.cat(recs, 0)
    se = (xr - X) ** 2
    return (float(se[unobs].mean()),                 # blind zone — the actual task
            float(se.mean()),                        # paper's R-score, whole state
            {c: float(se[:, i][unobs[:, i]].mean()) for i, c in enumerate(channels)})


def main():
    # All defaults come from config.yaml (single source of parameters);
    # the command line is only for per-experiment overrides
    ap = argparse.ArgumentParser(description="Train GENN-prior variational reconstruction "
                                             "(defaults from config.yaml)")
    ap.add_argument("--split", default=_T["split"], choices=["train", "valid", "test"])
    ap.add_argument("--days", type=int, default=_T["days"], help="最多用多少天")
    ap.add_argument("--dT", type=int, default=_T["dT"], help="时间窗长度(秒)")
    ap.add_argument("--sensing-range", type=int, default=d.SENSING_RANGE)
    ap.add_argument("--num-agents", type=int, default=d.NUM_AGENTS)
    ap.add_argument("--n-iter", type=int, default=config.get("solver", "n_iter"),
                    help="solver 迭代次数(论文 5~20)")
    ap.add_argument("--hidden", type=int, default=config.get("prior", "hidden"),
                    # %% not % — argparse runs every help string through `help % params`,
                    # so a literal percent sign makes --help raise ValueError.
                    help="GENN 隐藏通道(先验的主要容量旋钮; GENN 只占全模型 0.46%% 参数)")
    ap.add_argument("--n-phi-layers", type=int, default=config.get("prior", "n_phi_layers"),
                    help="GENN 里 phi 的 pointwise(1x1x1) 层数")
    ap.add_argument("--kt", type=int, default=config.get("prior", "kt"),
                    help="GENN psi 的时间核(必须奇数; 越大 Phi 能看到越远的帧)")
    ap.add_argument("--kh", type=int, default=config.get("prior", "kh"),
                    help="GENN psi 的空间核-高(奇数)。加大核是给 GENN 加容量的低显存路径: "
                         "激活只由 --hidden 决定,核只影响权重和算力")
    ap.add_argument("--kw", type=int, default=config.get("prior", "kw"),
                    help="GENN psi 的空间核-宽(奇数)")
    ap.add_argument("--lstm-hidden", type=int, default=config.get("solver", "lstm_hidden"),
                    help="solver LSTM 隐藏通道")
    ap.add_argument("--loss", default=_T["loss"],
                    choices=["supervised", "unsupervised", "nll"],
                    help="supervised=Eq.14 (4DVarNet), unsupervised=Eq.13, "
                         "nll=Gaussian NLL (deep-ensembles Eq.1; adds the sigma^2 read-out)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit training without CUDA. Off by default: a silent CPU fallback "
                         "under a --gres=gpu allocation costs ~30x the wall-clock and produces "
                         "nothing, so the run aborts instead")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="plain Bernoulli dropout in the solver's update net, on grad_2d and on "
                         "h, as a regulariser. No current run uses it; every reported model is "
                         "trained at 0")
    ap.add_argument("--batch", type=int, default=_T["batch"])
    ap.add_argument("--epochs", type=int, default=_T["epochs"])
    ap.add_argument("--max-windows", type=int, default=_T["max_windows"],
                    help=">0 时随机下采样到这么多训练窗口(提速)")
    ap.add_argument("--n-eval", type=int, default=_T["n_eval"], help="每轮评估用的固定窗口数")
    ap.add_argument("--steps", type=int, default=0, help=">0 时只跑这么多步(调试用)")
    ap.add_argument("--lr", type=float, default=_T["lr"])
    ap.add_argument("--seed", type=int, default=_T["seed"],
                    help="旧的单一 seed;同时设定初始权重和观测实现。保留以复现既有 run")
    # Deep-ensembles (arXiv:1612.01474 Sec. 2.4) varies ONLY the initialisation and the data
    # ordering — every member sees the same training set ("We used the entire training
    # dataset to train each network"; bagging is reported to hurt). Our single --seed drove
    # BOTH the weights and the robot routes + observation noise, so five members would each
    # have seen a different observation realisation. That also double-counts observation
    # noise, and observation noise is already in the data itself. Split them:
    # --init-seed is what an ensemble member varies, --data-seed stays fixed across members.
    ap.add_argument("--init-seed", type=int, default=None,
                    help="只影响权重初始化(集成成员之间改这个);默认沿用 --seed")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="只影响机器人路线与观测噪声(集成成员之间保持相同);默认沿用 --seed")
    ap.add_argument("--var-hidden", type=int, default=32,
                    help="--loss nll 的方差头宽度")
    ap.add_argument("--var-layers", type=int, default=2,
                    help="--loss nll 的方差头层数(逐点 1x1x1)")
    ap.add_argument("--augmented-var", action="store_true",
                    help="iterate log sigma^2 as part of the state instead of reading it off "
                         "the last hidden layer. The prior term of J becomes a Gaussian "
                         "log-likelihood of the prior residual, which is what gives sigma^2 a "
                         "gradient from the cost; the LSTM then descends [x, log sigma^2] "
                         "together. Doubles the optimiser's channel axis (C*dT -> 2*C*dT). "
                         "Requires --loss nll")
    ap.add_argument("--var-h-only", action="store_true",
                    help="build the sigma^2 read-out on the LSTM hidden state ALONE, without "
                         "the detached x_hat. This is what runs/varnet_vrb{0,1}_s0 did, and it "
                         "made sigma^2 spatially flat (0.96-1.21x empty/occupied against a "
                         "19-30x true error spread). Kept only to reproduce those baselines")
    ap.add_argument("--var-eps", type=float, default=1e-6,
                    help="minimum variance added after the softplus. 1e-6 is the value "
                         "Lakshminarayanan et al. give in Sec. 2.2.1 footnote 2; it is there "
                         "for numerical stability, not as a physical floor")
    ap.add_argument("--nll-beta", type=float, default=0.0,
                    # %% not % — argparse runs every help string through `help % params`.
                    help="每点 NLL 上的 detach(sigma^2)^beta 权重。这不在 Lakshminarayanan "
                         "et al. 里,所以默认 0,此时权重恒为 1,损失就是论文 Eq.1 原样。"
                         "非零只用于排查:beta=0 曾在旧设计(独立 head + 0.25xMAD 下限)下崩过,"
                         "RMSE +52%%,那两样现在都没了")
    ap.add_argument("--outdir", default=_T["outdir"])
    ap.add_argument("--no-noise", action="store_true",
                    help="观测不加噪声(对照实验; 默认由 config observation.add_noise 决定)")
    ap.add_argument("--resume", action="store_true",
                    help="从 outdir/varnet_last.pt 断点续训(载入 solver+optimizer+epoch)")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 混合精度(H200 提速,精度几乎无损)")
    ap.add_argument("--obs-every-k", type=int, default=None,
                    help="覆盖 config 的 observation.obs_every_k(每几帧观测一次)。"
                         "k=1 对齐 EnKF 项目(每帧观测); k=4 对齐论文 Lorenz-96。"
                         "留空 = 用 config 的值。设了此项才能让两个 k 的实验同时训练")
    ap.add_argument("--iter-schedule", default="",
                    help="论文 §3.4 的迭代递增课程 'epoch:n_iter[:lr],...',例如 "
                         "'0:5:1e-3,25:10:7e-4,50:15:5e-4,75:20:3e-4'。空 = 全程固定 --n-iter。"
                         "官方实现同时切 lr(步数变多→梯度路径变长→lr 要降)")
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help=">0 时按该范数裁剪梯度。大先验(--hidden 128)在 15 epoch 里出现过 "
                         "loss spike(盲区MSE 0.0516 -> 0.0992),裁剪是标准的稳定手段")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda" and not args.allow_cpu:
        raise SystemExit(
            "ABORT: CUDA is not available and --allow-cpu was not given.\n"
            "  Refusing to fall back to CPU silently. On 2026-08-26 six chained jobs were placed\n"
            "  on a draining node whose driver returned CUDA error 802 ('system not yet\n"
            "  initialized'); every one of them ran on CPU for 1.5 h and finished zero epochs,\n"
            "  while the chain kept submitting successors that would have done the same.\n"
            f"  SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID','-')} "
            f"node={os.environ.get('SLURMD_NODENAME','-')} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','-')}\n"
            "  If the node is at fault, resubmit with --exclude=<node>. To train on CPU on\n"
            "  purpose, pass --allow-cpu.")
    init_seed = args.seed if args.init_seed is None else args.init_seed
    data_seed = args.seed if args.data_seed is None else args.data_seed
    torch.manual_seed(init_seed)                            # weights only

    # ---- data ----
    k_eff = args.obs_every_k or config.get("observation", "obs_every_k")
    print(f"[obs] obs_every_k = {k_eff}"
          + ("  (每帧观测,对齐 EnKF 项目)" if k_eff == 1 else f"  (每 {k_eff} 帧观测)"), flush=True)
    files = d.split_files(args.split)
    X, Y, M, X0 = build_windows(files, args.dT, args.sensing_range, args.num_agents,
                                data_seed, args.days,
                                add_noise=False if args.no_noise else None,
                                obs_every_k=args.obs_every_k)
    if args.max_windows and X.shape[0] > args.max_windows:  # randomly subsample training windows (speed-up)
        sub = torch.randperm(X.shape[0])[:args.max_windows]
        X, Y, M, X0 = X[sub], Y[sub], M[sub], X0[sub]
    N, C, T, H, W = X.shape
    print(f"[data] {N} windows of {tuple((C,T,H,W))}  device={dev}", flush=True)


    # ---- model ----
    P = config.CFG["prior"]
    phi = GENN(n_channels=C, hidden=args.hidden, kt=args.kt, kh=args.kh, kw=args.kw,
               n_phi_layers=args.n_phi_layers, two_scale=P["two_scale"], scale=P["scale"])
    n_phi_param = sum(p.numel() for p in phi.parameters())
    solver = GradSolver(phi, n_channels=C, dT=T, n_iter=args.n_iter,
                        hidden_ch=args.lstm_hidden,
                        dropout=args.dropout,
                        predict_var=args.loss == "nll", var_eps=args.var_eps,
                        var_sees_state=not args.var_h_only,
                        augmented_var=args.augmented_var).to(dev)
    params = list(solver.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    n_param = sum(p.numel() for p in solver.parameters())
    sched = _parse_sched(args.iter_schedule, args.lr) if args.iter_schedule else None
    print(f"[model] params={n_param:,} (GENN prior {n_phi_param:,} = "
          f"{n_phi_param/n_param*100:.2f}%; the rest is the solver's ConvLSTM)  "
          f"loss={args.loss}  n_iter={args.n_iter}  amp={args.amp}", flush=True)
    print(f"[seed] init={init_seed}  data={data_seed}", flush=True)
    if args.loss == "nll":
        # augmented_var has NO read-out: log sigma^2 is a state channel the solver iterates,
        # so out_var is deliberately None there and this banner must not assume it exists.
        if solver.augmented_var:
            print(f"[var] AUGMENTED STATE: log sigma^2 iterated with x over {args.n_iter} "
                  f"steps; prior term of J is a Gaussian log-likelihood of the prior residual"
                  f"   eps={args.var_eps:g}", flush=True)
        else:
            _nv = sum(p.numel() for p in solver.grad_net.out_var.parameters())
            print(f"[var read-out] {_nv:,} params on the last hidden state   "
                  f"sigma^2 = softplus(out_var(h)) + {args.var_eps:g}   beta={args.nll_beta}",
                  flush=True)
    if sched:
        print("[schedule] " + "  ".join(f"ep{e}→{n}it@lr{l:g}" for e, n, l in sched), flush=True)

    # ---- resume from checkpoint (used by the self-chaining SLURM job) ----
    # A training run may be killed at the SLURM time limit and relaunched with --resume;
    # we reload the solver weights, the optimizer state (Adam's momentum buffers), and the
    # epoch/step counters so training continues seamlessly from where it stopped.
    start_epoch, step = 0, 0
    ckpt_path = os.path.join(args.outdir, "varnet_last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        solver.load_state_dict(ck["solver"])

        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", -1)) + 1          # continue AFTER the last saved epoch
        step = int(ck.get("step", 0))
        print(f"[resume] loaded {ckpt_path}: continuing from epoch {start_epoch} (step {step})", flush=True)
        if start_epoch >= args.epochs:                      # target already reached -> stop the chain
            print(f"[resume] already at target {args.epochs} epochs — nothing to do.", flush=True)
            return

    # Fixed evaluation subset: score the SAME first n_eval windows every epoch, so the
    # per-epoch metrics are directly comparable across epochs (evaluating the full set each
    # epoch would be too slow). Me < 0.5 marks the UNOBSERVED cells (mask is 1.0/0.0 float).
    n_eval = min(N, args.n_eval)
    Xe, Ye, Me, X0e = X[:n_eval], Y[:n_eval], M[:n_eval], X0[:n_eval]
    unobs_e = (Me < 0.5)                                    # boolean: True where a cell was NOT observed

    # ---- training ----
    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    # Track the best blind-zone MSE so a loss spike cannot destroy the best model: the rolling
    # varnet_last.pt is overwritten every epoch, and a spike (seen with the larger prior: 0.0367
    # -> 0.0925 in one epoch, even with --clip-grad) would otherwise lose it. On resume the
    # previous best is recovered from metrics.jsonl, which already holds every past epoch.
    best_mse = float("inf")
    if os.path.exists(metrics_path):
        past = [json.loads(l)["rec_unobs_mse"] for l in open(metrics_path) if l.strip()]
        if past:
            best_mse = min(past)
            print(f"[best] previous best blind MSE = {best_mse:.4f}", flush=True)
    amp_dtype = torch.bfloat16 if args.amp else None       # bf16 mixed precision, or None = full fp32
    for epoch in range(start_epoch, args.epochs):
        # Paper §3.4: "We typically increase incrementally the number of iterations of the
        # gradient-based NN Solver (typically from 5 iterations to 20 ones)". Back-propagating
        # through a 20-step unrolled solver from scratch is a hard optimisation problem; the
        # curriculum starts short and lengthens it. The official code (lit_model.py:94-100)
        # steps the LEARNING RATE at the same time, which matters: more steps means a longer
        # gradient path, so the lr has to come down with it.
        if sched:
            n_it, lr_now = _sched_at(sched, epoch)
            if solver.n_iter != n_it or opt.param_groups[0]["lr"] != lr_now:
                solver.n_iter = n_it
                for g in opt.param_groups:
                    g["lr"] = lr_now
                print(f"[schedule] epoch {epoch}: n_iter -> {n_it}, lr -> {lr_now:g}", flush=True)
        perm = torch.randperm(N)                            # shuffle window order each epoch
        solver.train()
        for i in range(0, N, args.batch):                   # iterate over mini-batches of windows
            idx = perm[i:i + args.batch]
            # move this batch's four tensors to the GPU: xb=truth, yb=obs, mb=mask, x0b=init guess
            xb, yb, mb, x0b = (t[idx].to(dev) for t in (X, Y, M, X0))
            opt.zero_grad()
            # under bf16 autocast when --amp, otherwise a no-op context (plain fp32).
            ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
            with ctx:
                # forward pass: the solver runs its n_iter learned-gradient-descent steps
                # from x0b using only yb and mb, and returns the reconstruction x_rec.
                # sigma^2, when asked for, comes from the SECOND output layer on the last
                # hidden state -- Lakshminarayanan Sec. 2.2.1's "two values in the final
                # layer", not a separate module reading x_rec.
                if args.loss == "nll":
                    xr, var = solver(x0b, yb, mb, return_var=True)
                else:
                    xr, var = solver(x0b, yb, mb), None
                # all loss logic lives in losses.py; dispatch on --loss
                # (y/mask/phi are only used by the unsupervised loss).
                loss = compute_loss(args.loss, xr, xb, yb, mb, phi, var=var,
                                    nll_beta=args.nll_beta)
            loss.backward()                                 # backprop through the WHOLE unrolled solver
            if args.clip_grad > 0:                          # stabilise the larger-prior runs
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            opt.step()                                      # Adam updates Phi + solver + cost weights jointly
            step += 1
            stop = bool(args.steps) and step >= args.steps   # --steps: stop early (debugging)
            if stop:
                break
        # ---- per epoch: score on the fixed eval subset ----
        rec_unobs, r_score, per_ch = evaluate(solver, Xe, Ye, Me, X0e, unobs_e,
                                              args.batch, dev, config.get("grid", "channels"))
        train_loss = float(loss.detach())
        rec = {"epoch": epoch, "step": step, "train_loss": train_loss,
               "rec_unobs_mse": rec_unobs, "r_score": r_score, "per_channel": per_ch}
        print(f"[epoch {epoch:3d}] loss={train_loss:.4f}  盲区MSE={rec_unobs:.4f}  "
              + " ".join(f"{k}={v:.4f}" for k, v in per_ch.items()), flush=True)
        with open(metrics_path, "a") as f:                  # append this epoch's metrics (one JSON per line)
            f.write(json.dumps(rec) + "\n")
        # overwrite the single rolling checkpoint. `args` is saved too so eval/plotting scripts
        # can rebuild the exact model (hidden, dT, n_iter, ...) without hard-coding anything.
        # n_iter_eff = the count actually in force this epoch. With --iter-schedule this
        # differs from args.n_iter, and every downstream script must solve with the same
        # number of iterations the weights were trained for.
        ck = {"solver": solver.state_dict(), "optimizer": opt.state_dict(),
              "args": vars(args), "epoch": epoch, "step": step,
              "n_iter_eff": int(solver.n_iter)}
        torch.save(ck, os.path.join(args.outdir, "varnet_last.pt"))
        if rec_unobs < best_mse:                            # keep the best model separately
            best_mse = rec_unobs
            torch.save(ck, os.path.join(args.outdir, "varnet_best.pt"))
            print(f"           ↑ new best ({best_mse:.4f}) -> varnet_best.pt", flush=True)
        if stop:
            break
    print(f"[done] checkpoints + metrics in {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
