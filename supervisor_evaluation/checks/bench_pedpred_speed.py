"""PedPred3 training-step speed test: batch 64 vs 256, eager vs one CUDA graph per step.

    DATA_VARIANT=all92 python3 -m supervisor_evaluation.checks.bench_pedpred_speed --days 4

Each variant starts from the same weights (--init) with a fresh Adam(amsgrad), clips the gradient
norm at 1.0 like the restarted run, and times `--steps` steps after a warm-up. The graph variant
captures forward, backward, clipping and the optimiser step (Adam capturable=True) once and then
only replays it; it has no per-step finiteness check (a graph cannot branch). An equality check
compares its losses with eager steps from the same weights on the same batches. The trainer
itself is not modified.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import load_pedpred3, mean_forecast, original_loss

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=4)
ap.add_argument("--init", default="runs/pedpred3_5to5_s0_all92/epoch_14.pt")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--clip", type=float, default=1.0)
a = ap.parse_args()
dev = torch.device("cuda")
train = PairFrames("train", dev, a.days, 0, input_frames=5, output_frames=5)
print(f"[data] {train.n_pairs:,} pairs from {train.n_days} days", flush=True)
perm = torch.randperm(train.n_pairs, generator=torch.Generator().manual_seed(0)).to(dev)


def fresh():
    torch.manual_seed(0)
    m = load_pedpred3(device=dev)
    m.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    m.train()
    return m


def lossfn(m, x, y):
    return original_loss(mean_forecast(m, x, horizon=5), y)


def batches(b, n):
    for k in range(n):
        i = (k * b) % (train.n_pairs - b)
        yield train.batch(perm[i:i + b])


def eager_steps(m, opt, b, n, check=True, record=False):
    out = []
    for x, y in batches(b, n):
        loss = lossfn(m, x, y)
        opt.zero_grad(set_to_none=True)
        if check and not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
        opt.step()
        if record:
            out.append(float(loss))
    return out


def graphed(m, b):
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.9, 0.999), amsgrad=True, capturable=True)
    x0, y0 = next(batches(b, 1))
    sx, sy = x0.clone(), y0.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):                       # warm-up on a side stream, as required
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss = lossfn(m, sx, sy); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip); opt.step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        sloss = lossfn(m, sx, sy); sloss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip); opt.step()

    def step(x, y):
        sx.copy_(x); sy.copy_(y); g.replay()
        return sloss
    return step


res = {}
for b, n in ((64, 400), (256, 120)):
    m = fresh(); opt = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.9, 0.999), amsgrad=True)
    eager_steps(m, opt, b, 10)
    torch.cuda.synchronize(); t = time.time(); eager_steps(m, opt, b, n); torch.cuda.synchronize()
    res[f"eager_b{b}"] = n * b / (time.time() - t)
    try:
        m = fresh(); step = graphed(m, b)
        for x, y in batches(b, 10):
            step(x, y)
        torch.cuda.synchronize(); t = time.time()
        for x, y in batches(b, n):
            step(x, y)
        torch.cuda.synchronize()
        res[f"graph_b{b}"] = n * b / (time.time() - t)
    except Exception as e:                           # report, do not hide, a capture failure
        res[f"graph_b{b}"] = f"capture failed: {type(e).__name__}: {str(e)[:200]}"
    print(f"[speed] b{b}: " + json.dumps({k: v for k, v in res.items() if k.endswith(f'b{b}')}), flush=True)

# equality: 3 warm-up steps on batch 0 (as the graph does), then 20 recorded steps, batch 64
m = fresh(); opt = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.9, 0.999), amsgrad=True, capturable=True)
x0, y0 = next(batches(64, 1))
for _ in range(3):
    opt.zero_grad(set_to_none=True); loss = lossfn(m, x0, y0); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip); opt.step()
ref = []
for x, y in batches(64, 20):
    loss = lossfn(m, x, y); opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip); opt.step(); ref.append(float(loss))
try:
    m = fresh(); step = graphed(m, 64)
    got = [float(step(x, y)) for x, y in batches(64, 20)]
    diff = max(abs(p - q) / max(abs(p), 1e-12) for p, q in zip(ref, got))
    print(f"[equality] 20 steps, max relative loss difference graph vs eager: {diff:.3g}", flush=True)
    print(f"[equality] eager first/last {ref[0]:.6f} {ref[-1]:.6f}  graph {got[0]:.6f} {got[-1]:.6f}", flush=True)
except Exception as e:
    print(f"[equality] capture failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
print("[result] samples/s " + json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
