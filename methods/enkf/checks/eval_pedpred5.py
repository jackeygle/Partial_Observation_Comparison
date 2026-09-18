"""Fair PedPred3 comparison on identical validation sequences.

Reports direct five-step (teacher-forced history) forecasts and open-loop rolling
one-step forecasts.  The projected state is exactly what the EnKF propagates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from crowdcore import navigation
from methods.enkf.surrogate.data import PairFrames
from methods.enkf.surrogate.model import (CH, load_pedpred3, mean_forecast,
                                          original_loss, surrogate_mean)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--new-ckpt", required=True)
    p.add_argument("--old-ckpt", default=str(
        Path(__file__).resolve().parents[1] / "enkf_lab/apt-ibex_train_model_28D.pth"))
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--rolling-steps", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def add_errors(store, prediction, truth, walkable):
    """Accumulate channel-wise squared errors for three scientifically useful scopes."""
    occupied = truth[:, :, 0:1] > 0
    defined = torch.cat((torch.ones_like(occupied), occupied, occupied,
                         truth[:, :, 3:4] > 0), dim=2)
    spatial = walkable.view(1, 1, 1, *walkable.shape)
    masks = {
        "all": torch.ones_like(truth, dtype=torch.bool),
        "walkable": spatial.expand_as(truth),
        "defined_walkable": defined & spatial,
    }
    square = (prediction - truth).double().square()
    target_square = truth.double().square()
    for region, mask in masks.items():
        values = store.setdefault(region, {
            "se": torch.zeros(4, dtype=torch.float64, device=truth.device),
            "target_sq": torch.zeros(4, dtype=torch.float64, device=truth.device),
            "count": torch.zeros(4, dtype=torch.float64, device=truth.device),
        })
        for channel in range(4):
            selected = mask[:, :, channel]
            values["se"][channel] += square[:, :, channel][selected].sum()
            values["target_sq"][channel] += target_square[:, :, channel][selected].sum()
            values["count"][channel] += selected.sum()


def finish_errors(store):
    result = {}
    tiny = torch.finfo(torch.float64).tiny
    for region, values in store.items():
        count = values["count"].clamp_min(1)
        rmse = torch.sqrt(values["se"] / count)
        nmse = values["se"] / values["target_sq"].clamp_min(tiny)
        result[region] = {
            "rmse": {name: float(value) for name, value in zip(CH, rmse)},
            "normalized_mse": {name: float(value) for name, value in zip(CH, nmse)},
            "overall_rmse": float(torch.sqrt(values["se"].sum() / count.sum())),
            "overall_normalized_mse": float(
                values["se"].sum() / values["target_sq"].sum().clamp_min(tiny)),
            "count": {name: int(value) for name, value in zip(CH, values["count"])},
        }
    return result


def rolling_forecast(net, history, steps):
    predictions = []
    for _ in range(steps):
        next_state = surrogate_mean(net, history, horizon=1)
        predictions.append(next_state)
        history = torch.cat((history[:, 1:], next_state), dim=1)
    return torch.cat(predictions, dim=1)


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("GPU required unless --allow-cpu is supplied")
    if min(args.pairs, args.batch, args.rolling_steps) < 1:
        raise SystemExit("pairs, batch, and rolling-steps must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PairFrames("valid", device, input_frames=5, output_frames=args.rolling_steps)
    count = min(args.pairs, data.n_pairs)
    rows = torch.linspace(0, data.n_pairs - 1, count, device=device).round().long().unique()
    walkable = torch.from_numpy(navigation.build_valid_mask_from_config()).to(device).bool()
    models = {
        "new_5to5": load_pedpred3(args.new_ckpt, device).eval(),
        "old_apt_ibex": load_pedpred3(args.old_ckpt, device).eval(),
    }
    rolling_checkpoints = sorted({step for step in (1, 5, 10, args.rolling_steps)
                                  if step <= args.rolling_steps})
    detail_modes = ([f"direct_lead_{lead}" for lead in range(1, 6)]
                    + [f"rolling_{step}" for step in rolling_checkpoints])
    accumulators = {
        name: {mode: {} for mode in ("direct_5", "rolling", *detail_modes)}
        for name in models
    }
    direct_loss = {name: [0.0, 0] for name in models}
    persistence = {"direct_5": {}, "rolling": {}}

    with torch.inference_mode():
        for begin in range(0, len(rows), args.batch):
            history, future = data.batch(rows[begin:begin + args.batch])
            direct_truth = future[:, :5]
            last = history[:, -1:]
            add_errors(persistence["direct_5"], last.expand(-1, 5, -1, -1, -1),
                       direct_truth, walkable)
            add_errors(persistence["rolling"],
                       last.expand(-1, args.rolling_steps, -1, -1, -1), future, walkable)
            for name, net in models.items():
                raw = mean_forecast(net, history, horizon=5)
                direct = surrogate_mean(net, history, horizon=5)
                rolling = rolling_forecast(net, history, args.rolling_steps)
                add_errors(accumulators[name]["direct_5"], direct, direct_truth, walkable)
                add_errors(accumulators[name]["rolling"], rolling, future, walkable)
                for lead in range(1, 6):
                    add_errors(accumulators[name][f"direct_lead_{lead}"],
                               direct[:, lead - 1:lead], direct_truth[:, lead - 1:lead],
                               walkable)
                for step in rolling_checkpoints:
                    add_errors(accumulators[name][f"rolling_{step}"],
                               rolling[:, :step], future[:, :step], walkable)
                loss = original_loss(raw, direct_truth)
                direct_loss[name][0] += float(loss) * len(history)
                direct_loss[name][1] += len(history)

    new_meta = torch.load(args.new_ckpt, map_location="cpu")
    result = {
        "config": vars(args),
        "device": str(device),
        "pairs": len(rows),
        "new_checkpoint": {
            "epoch": new_meta.get("epoch"), "valid_loss": new_meta.get("valid_loss"),
            "init": new_meta.get("init"), "args": new_meta.get("args"),
        },
        "models": {},
        "persistence": {mode: finish_errors(values) for mode, values in persistence.items()},
    }
    for name, modes in accumulators.items():
        result["models"][name] = {
            mode: finish_errors(values) for mode, values in modes.items()
        }
        total, n = direct_loss[name]
        result["models"][name]["direct_5_original_weighted_nlll"] = total / n
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    for name, values in result["models"].items():
        direct = values["direct_5"]["defined_walkable"]
        rolling = values["rolling"]["defined_walkable"]
        print(json.dumps({
            "model": name,
            "weighted_nlll": values["direct_5_original_weighted_nlll"],
            "direct5_rmse": direct["overall_rmse"],
            "direct5_nmse": direct["overall_normalized_mse"],
            "rolling_rmse": rolling["overall_rmse"],
            "rolling_nmse": rolling["overall_normalized_mse"],
        }), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
