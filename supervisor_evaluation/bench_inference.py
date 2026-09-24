"""Controlled GPU compute-throughput benchmark for the packaged final methods.

All methods run on one GPU and the first 600 frames of one held-out day. Input
arrays/tokens/windows and model loading are prepared before timing. The timed
region includes each predictor's complete forward computation; EnKF includes
its sequential forecast and analysis plus posterior output transfer. Results
are throughput per output frame, not response latency for windowed methods.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np


def run(output_dir: Path, data_root: Path, repeats: int = 3, frames: int = 600) -> None:
    import torch

    from crowdcore import config
    config.CFG["data"]["root"] = str(data_root.resolve())
    # observation_model resolves its dataset cache at import time.
    from crowdcore import observation_model as om
    from compare import bench_speed as bench
    from methods.senseiver import dataset as senseiver_ds, sensors
    from methods.senseiver.network import Senseiver
    from methods.varnet.checks.model_io import load_solver
    from supervisor_evaluation.evaluate import FINAL_CONFIG, MODELS, ROOT, _python_module

    if not torch.cuda.is_available():
        raise RuntimeError("controlled GPU benchmark requires a CUDA allocation")
    if frames < 400 or frames % 200:
        raise ValueError("frames must be a multiple of 200 and at least 400")
    output_dir = output_dir.resolve()
    day = str(om.split_files("test")[0])
    obs_dir = output_dir / "raw" / "enkf_observations"
    obs_file = obs_dir / "obs_atc-20130811.npz"
    if not obs_file.exists():
        raise FileNotFoundError(f"run the full EnKF evaluation first: {obs_file}")
    device = torch.device("cuda")
    prepared: dict[str, tuple[object, int, int]] = {}

    # The legacy benchmark helpers already prepare identical held-out inputs
    # outside their timing closures. They use the packaged checkpoints here.
    a_ckpt = MODELS / FINAL_CONFIG["senseiver_a"]["file"]
    a_once, a_warm, a_frames, _ = bench.bench_senseiver(
        str(a_ckpt), day, frames, "cuda", batch=64)
    a_warm()
    prepared[FINAL_CONFIG["senseiver_a"]["label"]] = (a_once, a_frames, 1)

    # Temporal Senseiver-G needs its own 16-frame tokenizer.
    g_ckpt = torch.load(MODELS / FINAL_CONFIG["senseiver_g"]["file"],
                        map_location=device, weights_only=False)
    g = Senseiver(**g_ckpt["hparams"]).to(device)
    g.load_state_dict(g_ckpt["model"]); g.eval()
    _, gy, gm = senseiver_ds.load_day(day, stride=1, seed=om.day_seed(day),
                                      frames=frames, obs_every_k=1)
    pe = g.pos_enc.detach().cpu().numpy()
    mean = g.in_mean.detach().cpu().numpy()
    std = g.in_std.detach().cpu().numpy()
    g_inputs = []
    for start in range(0, frames, 64):
        windows = [(gy[max(0, t - g.time_window + 1):t + 1],
                    gm[max(0, t - g.time_window + 1):t + 1])
                   for t in range(start, min(start + 64, frames))]
        tok, pad, dt, _, cell = sensors.build_batch_temporal(
            windows, pe, mean, std, return_cell_idx=True)
        g_inputs.append((tok.to(device), pad.to(device),
                         dt.to(device), cell.to(device)))

    def g_once() -> None:
        with torch.no_grad():
            for tok, pad, dt, cell in g_inputs:
                g.reconstruct(tok, pad, dt, cell)
        torch.cuda.synchronize()

    g_once()
    prepared[FINAL_CONFIG["senseiver_g"]["label"]] = (g_once, frames, 1)

    d_ckpt = MODELS / FINAL_CONFIG["dincae"]["file"]
    d_once, d_warm, d_frames, _ = bench.bench_dincae(
        str(MODELS), str(d_ckpt), day, frames, "cuda", batch=64)
    d_warm()
    prepared[FINAL_CONFIG["dincae"]["label"]] = (d_once, d_frames, 1)

    with np.load(obs_file) as z:
        y = z["Y"][:frames]
        x0 = z["X0"][:frames]
        mask = np.repeat(z["Omega"][:frames, None], 4, axis=1).astype(np.float32)

    for key in ("varnet_mse", "varnet_aughead"):
        solver, args, _ = load_solver(MODELS / FINAL_CONFIG[key]["file"], device)
        dt = int(args["dT"])
        if dt != 200:
            raise RuntimeError(f"unexpected {key} window length {dt}")
        windows = []
        for start in range(0, frames, dt):
            # Solver input is (batch, channels, time, height, width).
            arrays = (y[start:start + dt], mask[start:start + dt],
                      x0[start:start + dt])
            windows.append(tuple(torch.from_numpy(a[None].transpose(0, 2, 1, 3, 4))
                                 .float().to(device) for a in arrays))

        def make_once(solver=solver, windows=windows, with_var=key == "varnet_aughead"):
            def once() -> None:
                with torch.enable_grad():
                    for yp, mp, xp in windows:
                        if with_var:
                            mean_t, var_t = solver(xp, yp, mp, return_var=True)
                            del mean_t, var_t
                        else:
                            mean_t = solver(xp, yp, mp)
                            del mean_t
                torch.cuda.synchronize()
            return once

        fn = make_once()
        fn()
        prepared[FINAL_CONFIG[key]["label"]] = (fn, frames, dt)

    # The current GPU EnKF is run by its actual evaluator, with a separate
    # inference_seconds field ending before scoring. The same observation NPZ,
    # model and filter parameters as the final evaluation are supplied.
    cfg = FINAL_CONFIG["enkf"]
    enkf_dir = output_dir / "raw" / "latency_enkf"
    enkf_dir.mkdir(parents=True, exist_ok=True)

    def enkf_once(rep: int) -> float:
        result_path = enkf_dir / f"round_{rep}.json"
        args = ["--noise-kind", "residual", "--density-noise-space", "log1p",
                "--log-density-mean-correction", "ensemble",
                "--model-checkpoint", str(MODELS / cfg["file"]),
                "--input-frames", "5", "--bank", str(MODELS / cfg["bank_file"]),
                "--scale", str(cfg["q_scale"]), "--temporal-rho", str(cfg["temporal_rho"]),
                "--channel-scales", *map(str, cfg["channel_scales"]),
                "--blind-channel-scales", *map(str, cfg["blind_channel_scales"]),
                "--cross-channel-matrix", *map(str, cfg["cross_channel_matrix"]),
                "--ensemble", str(cfg["ensemble"]),
                "--radius", str(cfg["localization_radius"]),
                "--frames", str(frames), "--warmup", "100",
                "--day", "atc-20130811", "--obs-dir", str(obs_dir),
                "--out", str(result_path)]
        _python_module("methods.enkf.enkf_opt.experiments.eval_structured_q_gpu",
                       args, ROOT, data_root)
        result = json.loads(result_path.read_text())
        if result["frames_used"] != frames or result["n_analysis_failures"]:
            raise RuntimeError("EnKF timing run did not produce the expected frames")
        return 1000 * float(result["inference_seconds"]) / frames

    # One unreported EnKF warm-up run, then interleaved rounds on the same GPU.
    enkf_once(-1)
    measurements = {name: [] for name in prepared}
    enkf_name = cfg["label"]
    measurements[enkf_name] = []
    for rep in range(repeats):
        for name, (fn, n_frames, _) in prepared.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            measurements[name].append(1000 * (time.perf_counter() - start) / n_frames)
        measurements[enkf_name].append(enkf_once(rep))
        print(f"[benchmark] round {rep + 1}/{repeats}: "
              + ", ".join(f"{n}={v[-1]:.3f} ms/frame" for n, v in measurements.items()),
              flush=True)

    gpu = torch.cuda.get_device_name(0)
    output = output_dir / "inference_time_controlled.csv"
    rows = []
    for name, values in measurements.items():
        block = prepared[name][2] if name in prepared else 1
        rows.append({"method": name,
                     "median_ms_per_frame": float(np.median(values)),
                     "p95_ms_per_frame": float(np.percentile(values, 95)),
                     "gpu": gpu,
                     "scope": "predictor compute, prebuilt inputs; same GPU and frames",
                     "min_output_block_frames": block,
                     "repeats": repeats})
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "inference_time_controlled_detail.json").write_text(json.dumps({
        "gpu": gpu, "day": "atc-20130811", "frames": frames,
        "repeats": repeats, "individual_ms_per_frame": measurements,
        "note": "compute throughput per output frame; windowed models require a 200-frame block"},
        indent=2) + "\n")
    print(f"[benchmark] controlled timing written to {output}", flush=True)
