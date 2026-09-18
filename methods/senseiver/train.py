"""
train.py — training Senseiver on ATC
=====================================

Corresponds to the reference implementation's `train.py` +
`network_light.py::training_step`, but with a plain PyTorch loop (see
`network.py`'s file header for why). The training objective exactly matches the
reference implementation: unweighted 4-channel MSE on the raw field
(`losses.senseiver_loss`).

Hyperparameter defaults are taken from the reference implementation's
`s_parser.py`, with two adjusted for this problem's scale, explained here:

  * `--space-bands 16` (reference default 32). The frequencies are
    `linspace(1, dim/2, bands)`; with W=12 the highest frequency is only 6, so
    32 bands is pure redundancy.
  * `--lr 1e-3` (the reference argparse default is 1e-4). The README's example
    with tens of thousands of frames (pipe) also uses 1e-3, and we are in the
    1.28-million-frame range, the same regime.

Usage (**GPU node**):
    sbatch sbatch/submit_train.sbatch
    # smoke test
    srun -p gpu-debug --gres=gpu:1 -t 00:15:00 bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u train.py --days 1 --steps 50'
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.losses import diagnostics, history_aware_loss, senseiver_loss
from methods.senseiver.network import Senseiver


def make_batch(bank, idx, pos_enc_np, mean, std, dev):
    tok, pad, n = sensors.build_batch(bank.Y[idx], bank.Omega[idx], pos_enc_np, mean, std)
    C, H, W = ds.state_shape()
    x = torch.from_numpy(bank.X[idx]).reshape(len(idx), C, H, W)
    om = torch.from_numpy(bank.Omega[idx]).reshape(len(idx), H, W)
    return tok.to(dev), pad.to(dev), x.to(dev), om.to(dev), n


def make_batch_temporal(bank, idx, pos_enc_np, mean, std, dev):
    tok, pad, dt, n, cell_idx = sensors.build_batch_temporal(
        bank.windows(idx), pos_enc_np, mean, std, return_cell_idx=True)
    C, H, W = ds.state_shape()
    x = torch.from_numpy(bank.X[idx]).reshape(len(idx), C, H, W)
    om = torch.from_numpy(bank.target_omega(idx)).reshape(len(idx), H, W)
    return (tok.to(dev), pad.to(dev), dt.to(dev), cell_idx.to(dev),
            x.to(dev), om.to(dev), n)


@torch.no_grad()
def validate(model, bank, pos_enc_np, mean, std, dev, batch, chans, max_batches=40):
    model.eval()
    agg, nb = {}, 0
    rng = np.random.default_rng(0)
    for i, idx in enumerate(bank.batches(batch, rng)):
        if i >= max_batches:
            break
        if getattr(bank, "window", 1) > 1:
            tok, pad, dt, cell_idx, x, om, _ = make_batch_temporal(
                bank, idx, pos_enc_np, mean, std, dev)
        else:
            tok, pad, x, om, _ = make_batch(bank, idx, pos_enc_np, mean, std, dev)
            dt, cell_idx = None, None
        pred = model.reconstruct(tok, pad, dt, cell_idx)
        d = diagnostics(pred.float(), x, om, chans)
        for k, v in d.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    model.train()
    return {k: v / max(nb, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser(description="Senseiver on ATC")
    # Data
    ap.add_argument("--split", default="train")
    ap.add_argument("--days", type=int, default=0, help="max number of days to use (0 = all)")
    ap.add_argument("--stride", type=int, default=4, help="frame-subsampling stride (adjacent seconds are highly redundant)")
    ap.add_argument("--valid-days", type=int, default=3)
    ap.add_argument("--valid-stride", type=int, default=20)
    ap.add_argument("--frames", type=int, default=0,
                    help="when >0, only take the first N frames per day (for smoke tests; simulating observations for a whole day is slow)")
    ap.add_argument("--obs-every-k", type=int, default=None, help="overrides config's observation.obs_every_k")
    ap.add_argument("--obs-seed", type=int, default=None,
                    help="base seed for training robot paths/noise (default: --seed)")
    ap.add_argument("--valid-obs-seed", type=int, default=None,
                    help="base seed for validation paths/noise (default: training obs seed)")
    ap.add_argument("--trajectory-mode", choices=["fixed", "per_day"], default="fixed",
                    help="fixed replays one robot path on every training day; per_day uses base_seed+day_index")
    ap.add_argument("--valid-trajectory-mode", choices=["fixed", "per_day"], default=None,
                    help="validation trajectory mode (default: --trajectory-mode)")
    # Model (defaults taken from reference/Senseiver/s_parser.py)
    ap.add_argument("--space-bands", type=int, default=16)
    ap.add_argument("--enc-preproc-ch", type=int, default=32)
    ap.add_argument("--num-latents", type=int, default=64)
    ap.add_argument("--enc-num-latent-channels", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-cross-attention-heads", type=int, default=2)
    ap.add_argument("--enc-num-self-attention-heads", type=int, default=2)
    ap.add_argument("--num-self-attention-layers-per-block", type=int, default=3)
    ap.add_argument("--no-share-encoder-blocks", dest="share_encoder_blocks",
                    action="store_false",
                    help="give every encoder block independent parameters instead of "
                         "reusing the final block (capacity extension)")
    ap.add_argument("--dec-preproc-ch", type=int, default=32)
    ap.add_argument("--dec-num-latent-channels", type=int, default=32)
    ap.add_argument("--dec-num-cross-attention-heads", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    # Variant G (grid latent) -- an extension, not part of the reference implementation
    ap.add_argument("--latent-mode", choices=["abstract", "grid", "hierarchical"],
                    default="abstract",
                    help="abstract = the reference's 64 learnable latents (variant A); "
                         "grid = one latent token per grid cell (variant G); "
                         "hierarchical = 9x3 -> 18x6 -> 36x12 grid latents (H3)")
    ap.add_argument("--readout", choices=["decoder", "direct"], default="decoder",
                    help="decoder = the reference's query decoder; direct = each cell's "
                         "token -> Linear -> its 4 values (needs --latent-mode grid)")
    # Temporal extension -- also an extension, not part of the reference implementation
    ap.add_argument("--time-window", type=int, default=1,
                    help="k: each sample sees the observations of its last k frames "
                         "(1 = the current frame only, i.e. the non-temporal model)")
    ap.add_argument("--time-dim", type=int, default=8, help="size of the Δ embedding")
    ap.add_argument("--no-time-scalar", dest="time_scalar", action="store_false",
                    help="drop the Δ/k scalar that gives the embedding its order")
    ap.add_argument("--temporal-mixer", choices=["token_set", "advected_tokens", "ssm", "framewise"],
                    default="token_set",
                    help="token_set = baseline flattened temporal tokens; "
                         "advected_tokens = move historical token coordinates by Δt*v; "
                         "ssm = per-cell state-space scan over the time window; "
                         "framewise = shared per-frame spatial encoding then learned "
                         "residual temporal fusion")
    ap.add_argument("--spatial-attention", choices=["global", "geodesic"],
                    default="global",
                    help="global = baseline attention; geodesic = add a fixed-strength "
                         "obstacle-aware shortest-path bias")
    ap.add_argument("--ssm-hidden-ch", type=int, default=41,
                    help="41 exactly parameter-matches the k=16 token-set baseline")
    ap.add_argument("--geodesic-scale", type=float, default=1.0,
                    help="fixed multiplier on max-normalised shortest-path distance")
    ap.add_argument("--advection-tau", type=float, default=3.0,
                    help="seconds; velocity displacement is dt*exp(-dt/tau)*v")
    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--history-loss-weight", type=float, default=0.0,
                    help="extra per-element weight for cells blind now but seen earlier "
                         "in the temporal window (0 keeps the baseline objective)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=0, help="when >0, only run this many steps per epoch (debug)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="runs/senseiver_A")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()

    if args.history_loss_weight < 0:
        ap.error("--history-loss-weight must be non-negative")
    if args.history_loss_weight > 0 and args.time_window == 1:
        ap.error("--history-loss-weight needs --time-window > 1")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No GPU. This project's rule is that torch only runs "
                         "on GPU nodes; if you really need CPU, add --allow-cpu.")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    obs_seed = args.seed if args.obs_seed is None else args.obs_seed
    valid_obs_seed = obs_seed if args.valid_obs_seed is None else args.valid_obs_seed
    valid_trajectory_mode = args.valid_trajectory_mode or args.trajectory_mode

    C, H, W = ds.state_shape()
    chans = ds.channels()
    print(f"[config] observation config = {ds.obs_config()}", flush=True)

    if args.time_window > 1:
        Bank = lambda *a, **kw: ds.TemporalDayBank(*a, window=args.time_window, **kw)
    else:
        Bank = ds.DayBank
    print(f"[data] loading {args.split} split ...", flush=True)
    train_bank = Bank(ds.om.split_files(args.split), stride=args.stride,
                            seed=obs_seed, max_days=args.days, frames=args.frames,
                            obs_every_k=args.obs_every_k,
                            trajectory_mode=args.trajectory_mode)
    print(f"[data] {train_bank.n} training frames", flush=True)
    valid_bank = Bank(ds.om.split_files("valid"), stride=args.valid_stride,
                            seed=valid_obs_seed, max_days=args.valid_days, frames=args.frames,
                            obs_every_k=args.obs_every_k,
                            trajectory_mode=valid_trajectory_mode)
    print(f"[data] {valid_bank.n} validation frames", flush=True)
    print(f"[data] model_seed={args.seed}  train_obs_seed={obs_seed} "
          f"train_trajectory_mode={args.trajectory_mode}  "
          f"valid_obs_seed={valid_obs_seed} "
          f"valid_trajectory_mode={valid_trajectory_mode}", flush=True)

    mean, std = train_bank.input_stats()
    print(f"[stats] encoder input standardisation mean={mean} std={std}", flush=True)

    model = Senseiver(
        im_ch=C, grid=(H, W), space_bands=args.space_bands,
        enc_preproc_ch=args.enc_preproc_ch, num_latents=args.num_latents,
        enc_num_latent_channels=args.enc_num_latent_channels,
        num_layers=args.num_layers,
        num_cross_attention_heads=args.num_cross_attention_heads,
        enc_num_self_attention_heads=args.enc_num_self_attention_heads,
        num_self_attention_layers_per_block=args.num_self_attention_layers_per_block,
        dec_preproc_ch=args.dec_preproc_ch,
        dec_num_latent_channels=args.dec_num_latent_channels,
        dec_num_cross_attention_heads=args.dec_num_cross_attention_heads,
        dropout=args.dropout, latent_mode=args.latent_mode, readout=args.readout,
        time_window=args.time_window, time_dim=args.time_dim, time_scalar=args.time_scalar,
        temporal_mixer=args.temporal_mixer, spatial_attention=args.spatial_attention,
        ssm_hidden_ch=args.ssm_hidden_ch, geodesic_scale=args.geodesic_scale,
        advection_tau=args.advection_tau,
        share_encoder_blocks=args.share_encoder_blocks,
        in_mean=mean, in_std=std).to(dev)
    print(f"[model] {model.num_params:,} parameters  latent={args.latent_mode}  "
          f"readout={args.readout}  time_window={args.time_window}  "
          f"temporal={args.temporal_mixer}  spatial={args.spatial_attention}  "
          f"share_blocks={args.share_encoder_blocks}  seed={args.seed}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev.type == "cuda")
    pe_np = model.pos_enc.detach().cpu().numpy()

    start_epoch, best = 0, float("inf")
    last_p = os.path.join(args.out, "last.pt")
    mpath = os.path.join(args.out, "metrics.jsonl")
    if args.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1; best = ck.get("best", best)
        # Older segments wrote last.pt before updating `best`, so its value can
        # lag the final epoch of the preceding segment.  The append-only metric
        # log is authoritative and prevents a successor's first epoch from
        # accidentally replacing a genuinely better best.pt.
        if os.path.exists(mpath):
            with open(mpath) as f:
                for line in f:
                    try:
                        logged = json.loads(line).get("valid_mse_blind")
                        if logged is not None:
                            best = min(best, float(logged))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        print(f"[resume] continuing from epoch {start_epoch} (best={best:.5f})", flush=True)

    rng = np.random.default_rng(args.seed + start_epoch)
    for ep in range(start_epoch, args.epochs):
        t0, tot, nel, nstep = time.perf_counter(), 0.0, 0, 0
        for i, idx in enumerate(train_bank.batches(args.batch, rng)):
            if args.steps and i >= args.steps:
                break
            if args.time_window > 1:
                tok, pad, dt, cell_idx, x, _om, _n = make_batch_temporal(
                    train_bank, idx, pe_np, mean, std, dev)
            else:
                tok, pad, x, _om, _n = make_batch(train_bank, idx, pe_np, mean, std, dev)
                dt, cell_idx = None, None
            with torch.amp.autocast("cuda", enabled=args.amp and dev.type == "cuda"):
                pred = model.reconstruct(tok, pad, dt, cell_idx)
                if args.history_loss_weight > 0:
                    hist = torch.from_numpy(train_bank.history_reachable(idx)).reshape(
                        len(idx), H, W).to(dev)
                    loss = history_aware_loss(
                        pred, x, hist, args.history_loss_weight)
                else:
                    loss = senseiver_loss(pred, x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nel += x.numel(); nstep += 1

        v = validate(model, valid_bank, pe_np, mean, std, dev, args.batch, chans)
        rec = {"epoch": ep, "train_mse": tot / max(nel, 1), "steps": nstep,
               "sec": time.perf_counter() - t0, "lr": args.lr,
               **{f"valid_{k}": val for k, val in v.items()}}
        with open(mpath, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[ep {ep:3d}] train {rec['train_mse']:.5f}  valid mse {v['mse']:.5f}  "
              f"blind {v['mse_blind']:.5f}  density_occ {v['mse_density_occ']:.5f}  "
              f"({rec['sec']:.0f}s)", flush=True)

        improved = v["mse_blind"] < best
        if improved:
            best = v["mse_blind"]
        ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": ep,
              "hparams": model.hparams, "args": vars(args), "best": best,
              "in_mean": mean.tolist(), "in_std": std.tolist()}
        torch.save(ck, last_p)
        if improved:
            torch.save(ck, os.path.join(args.out, "best.pt"))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
