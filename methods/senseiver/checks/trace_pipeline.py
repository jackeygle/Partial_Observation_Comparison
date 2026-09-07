"""
trace_pipeline.py — prints the shape at every step of one frame of data, from the raw grid to the loss
================================================================

Not a test, an **executable version of the documentation**: every shape in the
README's "Pipeline" section was produced by this script. Rerun it whenever the
data pipeline changes, so the README and the code never disagree.

Usage (GPU node):
    srun -p gpu-debug --gres=gpu:1 -t 00:12:00 --mem=24G bash -c \
      'module load scicomp-pytorch-env/2026.1; python3 -u checks/trace_pipeline.py'
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from methods.senseiver import dataset as ds
from methods.senseiver import sensors
from methods.senseiver.network import Senseiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/senseiver_A/best.pt",
                    help="use its hyperparameters if a checkpoint exists; otherwise build a new model with the defaults")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    B = args.batch

    print("=== 1. Raw one day ===")
    day = ds.om.split_files("test")[0]
    X, _ = ds.om.load_state(day)
    X = np.asarray(X)[:args.frames]
    print(f"  grid_cache, one day: {X.shape} {X.dtype}  (T, C, H, W)")

    print("=== 2. Observation model (4dvarnet_enkf) ===")
    valid = ds.nav.build_valid_mask_from_config(X)
    out = ds.om.generate_observations(X, valid_mask=valid, seed=0)
    n_obs = out["Omega"].sum(axis=(1, 2))
    print(f"  Y      : {out['Y'].shape}  noisy partial observation, 0 where unobserved")
    print(f"  Omega  : {out['Omega'].shape}  bool, which cells were seen that second "
          f"-> {n_obs[:5].tolist()} ...  (min {n_obs.min()} max {n_obs.max()})")
    print(f"  walkable: {int(valid.sum())} / {valid.size}")

    print("=== 3. dataset.load_day flattening ===")
    Xs, Ys, Om = ds.load_day(day, stride=1, seed=0, frames=args.frames)
    print(f"  X : {Xs.shape}  (N, C, HW)   <- target")
    print(f"  Y : {Ys.shape}  (N, C, HW)   <- observation")
    print(f"  Om: {Om.shape}  (N, HW) bool")

    print("=== 4. sensors.build_batch, variable-length tokens ===")
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        m = Senseiver(**ck["hparams"])
        m.load_state_dict(ck["model"])
        print(f"  (hyperparameters from {args.ckpt}, epoch {ck.get('epoch')})")
    else:
        C, H, W = ds.state_shape()
        m = Senseiver(im_ch=C, grid=(H, W), space_bands=16, enc_preproc_ch=32,
                      num_latents=64, enc_num_latent_channels=32, num_layers=3,
                      num_cross_attention_heads=2, enc_num_self_attention_heads=2,
                      num_self_attention_layers_per_block=3, dec_preproc_ch=32,
                      dec_num_latent_channels=32, dec_num_cross_attention_heads=1)
        print("  (no checkpoint found, building a new one with the default hyperparameters)")
    m.eval()
    pe = m.pos_enc.numpy(); mu = m.in_mean.numpy(); sd = m.in_std.numpy()
    idx = np.arange(B)
    tok, pad, n = sensors.build_batch(Ys[idx], Om[idx], pe, mu, sd)
    C = m.im_ch
    print(f"  pos_enc : {pe.shape}  (HW, P)   P = 2 * 2 dims * bands")
    print(f"  tokens  : {tuple(tok.shape)}  (B, Nmax, C+P) = ({B}, {tok.shape[1]}, {C}+{pe.shape[1]})")
    print(f"  pad_mask: {tuple(pad.shape)}  True=padding slot")
    print(f"  actual sensor count per frame: {n.tolist()}")

    print("=== 5. Model forward pass ===")
    with torch.no_grad():
        z = m.encoder(tok, pad)
        print(f"  encoder ->  z: {tuple(z.shape)}  (B, num_latents, ch)  <-- independent of the sensor count!")
        coords = m.pos_enc[None].expand(B, -1, -1)
        print(f"  query coordinates coords: {tuple(coords.shape)}  (B, HW, P)  the whole grid")
        o = m.decoder(z, coords)
        print(f"  decoder ->   : {tuple(o.shape)}  (B, HW, C)")
        rec = m.reconstruct(tok, pad)
        print(f"  reshape ->   : {tuple(rec.shape)}  (B, C, H, W)")

    print("=== 6. Loss / metric alignment ===")
    Cc, H, W = ds.state_shape()
    xb = torch.from_numpy(Xs[idx]).reshape(B, Cc, H, W)
    obs = torch.from_numpy(Om[idx]).reshape(B, 1, H, W).expand(-1, Cc, -1, -1)
    d2 = (rec - xb) ** 2
    nb, nf = int((~obs).sum()), d2.numel()
    print(f"  blind cell count {nb} / total {nf}  ({nb/nf*100:.1f}%)")
    print(f"  the output channel count is decided by im_ch = {C}  (postproc.weight "
          f"{tuple(m.decoder.postproc.weight.shape)}), independent of latent_size="
          f"{m.hparams['latent_size']}")


if __name__ == "__main__":
    main()
