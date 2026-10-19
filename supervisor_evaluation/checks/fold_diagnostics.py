"""Audit saved fold-axis runs and probe their solver on real validation windows.

Diagnostic only: does not select additional candidates, train, or edit final models.
The real-data probe does not score any reference predictions. It records internal
gradient/update magnitudes and model error on three fixed validation batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from crowdcore import navigation as nav
from crowdcore import observation_model as om
from methods.varnet.checks.eval_comprehensive import clip_bounds
from methods.varnet.checks.model_io import load_solver
from methods.varnet.checks.select_checkpoint import score_day
from methods.varnet.train import build_windows

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "methods/varnet/runs"
ARMS = {
    "plain_legacy": "varnet_mse5_h96_pd_s3",
    "plain_ct": "varnet_mse5_h96_pdfold_s3",
    "aug_legacy": "varnet_aughead_obs_h96_s0",
    "aug_ct": "varnet_aughead_obs_h96_fold_s0",
}
CHANNELS = ["density", "vx", "vy", "var"]


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rms(tensor):
    return float(tensor.detach().double().square().mean().sqrt())


def optimizer_audit(solver, ckpt):
    """Map optimizer IDs using named_parameters, not state_dict/buffer ordering."""
    opt = ckpt.get("optimizer")
    if opt is None:
        return {"available": False}
    parameters = list(solver.named_parameters())
    ids = [pid for group in opt["param_groups"] for pid in group["params"]]
    if len(parameters) != len(ids):
        raise ValueError("optimizer parameter count differs from solver parameter count")
    group_for = {pid: g for g in opt["param_groups"] for pid in g["params"]}
    wanted = {"grad_net.lstm.gates.weight", "grad_net.out.weight",
              "phi.branch_fine.psi.conv.weight", "grad_net.out_var.weight"}
    out = {"available": True, "groups": [
        {k: g.get(k) for k in ["lr", "eps", "betas", "weight_decay", "amsgrad"]}
        for g in opt["param_groups"]], "parameters": {}}
    for (name, parameter), pid in zip(parameters, ids):
        if name not in wanted or pid not in opt["state"]:
            continue
        state, group = opt["state"][pid], group_for[pid]
        step = float(state["step"])
        b1, b2 = group["betas"]
        m, v = state["exp_avg"].double(), state["exp_avg_sq"].double()
        mh, vh = m / (1 - b1 ** step), v / (1 - b2 ** step)
        delta = group["lr"] * mh / (vh.sqrt() + group["eps"])
        w = parameter.detach().cpu()
        rounded = (w.double() - delta).to(w.dtype)
        out["parameters"][name] = {
            "step": step, "dtype": str(w.dtype),
            "exp_avg_abs_median": float(m.abs().median()),
            "sqrt_bias_corrected_v_median": float(vh.sqrt().median()),
            "estimated_update_abs_median": float(delta.abs().median()),
            "fraction_sqrt_v_below_eps": float((vh.sqrt() < group["eps"]).double().mean()),
            "fraction_estimated_update_rounds_to_same_weight": float((rounded == w).double().mean()),
            "note": "Estimate from stored moments; not a recorded historical step or a root-cause proof.",
        }
    return out


def inspect_checkpoint(path):
    # Current project checkpoints contain tensor state dicts and primitive metadata.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    solver, args, _ = load_solver(path, "cpu")
    bad = [k for k, v in ckpt["solver"].items()
           if v.dtype.is_floating_point and not bool(torch.isfinite(v).all())]
    return {
        "path": str(path.relative_to(ROOT)), "sha256": fingerprint(path),
        "epoch": ckpt.get("epoch"), "n_iter_eff": ckpt.get("n_iter_eff"),
        "fold_axes": args.get("fold_axes", "legacy"), "args": args,
        "nonfinite_weight_tensors": bad,
        "cost_parameters": {k: v.detach().tolist() for k, v in solver.var_cost.named_parameters()},
        "optimizer": optimizer_audit(solver, ckpt),
    }


def trace(solver, tensors, gate_trace=False):
    records = []
    original = solver._cost_and_grad

    def cost_and_grad(*args, **kwargs):
        j, gx, gs = original(*args, **kwargs)
        records.append({"iteration": len(records), "cost": float(j.detach()),
                        "state_gradient_rms": rms(gx),
                        "variance_gradient_rms": None if gs is None else rms(gs)})
        return j, gx, gs

    def before(_module, inputs):
        records[-1]["normalized_lstm_input_rms"] = rms(inputs[0])

    def after(_module, _inputs, outputs):
        records[-1]["state_update_rms"] = rms(outputs[0]) / solver.n_iter
        records[-1]["hidden_rms"] = rms(outputs[2])

    def gate_activation(_module, _inputs, output):
        i, f, o, g = output.detach().chunk(4, dim=1)
        stats = {}
        for name, z in [("input", i), ("forget", f), ("output", o)]:
            activation = z.sigmoid()
            stats[name] = {
                "activation_mean": float(activation.double().mean()),
                "fraction_below_1e-4": float((activation < 1e-4).double().mean()),
                "fraction_above_1_minus_1e-4": float((activation > 1 - 1e-4).double().mean()),
                "preactivation_min": float(z.min()), "preactivation_max": float(z.max()),
            }
        stats["candidate_rms"] = rms(g.tanh())
        stats["input_times_candidate_rms"] = rms(i.sigmoid() * g.tanh())
        records[-1]["gates"] = stats

    solver._cost_and_grad = cost_and_grad
    hooks = [solver.grad_net.register_forward_pre_hook(before),
             solver.grad_net.register_forward_hook(after)]
    if gate_trace:
        hooks.append(solver.grad_net.lstm.gates.register_forward_hook(gate_activation))
    try:
        with torch.enable_grad():
            pred = solver(*(t.clone() for t in tensors)).detach().cpu().numpy()
    finally:
        solver._cost_and_grad = original
        for hook in hooks:
            hook.remove()
    first = records[0]
    # The augmented RMS averages equally over the state/variance halves.
    g2 = first["state_gradient_rms"] ** 2
    if first["variance_gradient_rms"] is not None:
        g2 = (g2 + first["variance_gradient_rms"] ** 2) / 2
    return pred, {"initial_gradient_normalizer": math.sqrt(g2 + 1e-12),
                  "iterations": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--probe", action="store_true", help="requires a CUDA allocation")
    ap.add_argument("--batch", type=int, default=8, help="same grouping as checkpoint selection")
    ap.add_argument("--gate-trace", action="store_true")
    ap.add_argument("--probe-models", nargs="+", default=None)
    ap.add_argument("--full-precrash-score", action="store_true",
                    help="score training-selected snapshot on all validation days, diagnostic only")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, selections = {}, {}
    for label, run in ARMS.items():
        selection = json.loads((RUNS / run / "select_valid.json").read_text())
        if selection["split"] != "valid":
            raise ValueError("all selections must use the validation split")
        selections[label] = selection
        paths[label] = RUNS / run / selection["selected_ckpt"]
    paths["plain_ct_epoch80"] = RUNS / ARMS["plain_ct"] / "ckpt_00080.pt"
    # This snapshot is diagnostic only; it was preselected by a training-set metric.
    paths["plain_ct_train_selected_diagnostic_only"] = RUNS / ARMS["plain_ct"] / "varnet_best.pt"
    paths["aug_ct_last_diagnostic_only"] = RUNS / ARMS["aug_ct"] / "varnet_last.pt"
    if args.probe_models and set(args.probe_models) - set(paths):
        raise ValueError("unknown probe model label")
    report = {"purpose": "audit of Claude's fold-axis training diagnosis; no new baselines",
              "selections": selections, "checkpoints": {}, "probes": {}}
    for label, path in paths.items():
        report["checkpoints"][label] = inspect_checkpoint(path)
        print(f"[audit] {label}: epoch {report['checkpoints'][label]['epoch']}", flush=True)
    (args.output_dir / "checkpoint_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.probe:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("real-data probes require CUDA; refusing a slow CPU fallback")
    report["gpu"] = torch.cuda.get_device_name(0)
    fp = om.split_files("valid")[0]
    walk = nav.build_valid_mask_from_config()
    model_args = report["checkpoints"]["plain_ct"]["args"]
    observation_keys = ["dT", "sensing_range", "num_agents", "data_seed", "no_noise",
                        "obs_every_k", "trajectory_mode"]
    reference = {k: model_args.get(k) for k in observation_keys}
    for metadata in report["checkpoints"].values():
        if {k: metadata["args"].get(k) for k in observation_keys} != reference:
            raise ValueError("checkpoints differ in observation configuration; cannot share inputs")
    data_seed = model_args["seed"] if model_args.get("data_seed") is None else model_args["data_seed"]
    x, y, mask, initial = build_windows(
        [fp], model_args["dT"], model_args["sensing_range"], model_args["num_agents"],
        data_seed, 1, add_noise=False if model_args.get("no_noise") else None,
        obs_every_k=model_args.get("obs_every_k"),
        trajectory_mode=model_args.get("trajectory_mode", "fixed"))
    if x.shape[0] < args.batch:
        raise ValueError("not enough complete validation windows")
    n_groups = (x.shape[0] + args.batch - 1) // args.batch
    starts = sorted({int(i) * args.batch for i in np.linspace(0, n_groups - 1, 3, dtype=int)})
    report["probe_protocol"] = {
        "split": "valid", "day": Path(fp).name, "batch": args.batch,
        "window_starts": starts, "scope": "blind walkable, complete windows, clipped",
        "inference_dtype": "float32, as in checkpoint selection",
        "note": "Three fixed batches only; diagnostic scores are not full validation or test results.",
    }
    output_name = "gate_validation_probe.json" if args.gate_trace else "real_validation_probe.json"
    for label, path in paths.items():
        if args.probe_models and label not in args.probe_models:
            continue
        solver, _, _ = load_solver(path, "cuda")
        reports = []
        for start in starts:
            end = min(start + args.batch, x.shape[0])
            tensors = [t[start:end].to("cuda") for t in (initial, y, mask)]
            pred, detail = trace(solver, tensors, gate_trace=args.gate_trace)
            sel = (mask[start:end].numpy() < 0.5) & walk[None, None, None]
            truth = x[start:end].numpy()
            valid_prediction = bool(np.isfinite(pred).all())
            clipped = clip_bounds(pred.copy())
            squared = (clipped.astype(np.float64) - truth) ** 2
            channel_rmse = [float(np.sqrt(squared[:, c][sel[:, c]].mean())) for c in range(4)]
            detail.update(window_start=start, window_end=end,
                          mean_walkable_density=float(truth[:, 0, :, walk].mean()),
                          finite_prediction=valid_prediction,
                          pooled_rmse=float(np.sqrt(squared[sel].mean())),
                          channel_rmse=dict(zip(CHANNELS, channel_rmse)))
            reports.append(detail)
            print(f"[probe] {label} windows {start}:{end}: "
                  f"RMSE={detail['pooled_rmse']:.6f}, "
                  f"normg={detail['initial_gradient_normalizer']:.3e}, "
                  f"first update RMS={detail['iterations'][0]['state_update_rms']:.3e}", flush=True)
        report["probes"][label] = reports
        (args.output_dir / output_name).write_text(json.dumps(report, indent=2) + "\n")
        del solver
        torch.cuda.empty_cache()
    if args.full_precrash_score:
        label = "plain_ct_train_selected_diagnostic_only"
        path = paths[label]
        solver, model_args, ckpt = load_solver(path, "cuda")
        if ckpt["n_iter_eff"] != 20:
            raise ValueError("precrash diagnostic must use the final 20 iterations")
        total_se, total_n, days = 0.0, 0, []
        epoch = int(ckpt["epoch"])
        for fp in om.split_files("valid"):
            se, n = score_day({epoch: solver}, fp, {str(path): model_args},
                              walk, "cuda", args.batch)[epoch]
            total_se += se
            total_n += n
            days.append({"day": Path(fp).name, "squared_error_sum": se,
                         "scored_count": n, "mse": se / n})
            print(f"[precrash diagnostic] epoch {epoch} {Path(fp).name}: MSE={se / n:.6f}", flush=True)
        diagnostic = {"split": "valid", "checkpoint": str(path.relative_to(ROOT)),
                      "sha256": fingerprint(path), "epoch": epoch, "n_iter": 20,
                      "batch": args.batch, "scope": "blind walkable, complete windows, clipped",
                      "mse": total_se / total_n, "rmse": math.sqrt(total_se / total_n),
                      "per_day": days,
                      "note": "Diagnostic only. This training-selected file is not in the fixed checkpoint candidate grid.",
                      "old_plain_selected_mse": selections["plain_legacy"]["selected_mse"]}
        (args.output_dir / "precrash_full_validation_diagnostic.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n")
        print(f"[precrash diagnostic] pooled RMSE={diagnostic['rmse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
