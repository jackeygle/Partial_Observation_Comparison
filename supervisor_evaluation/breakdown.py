"""Breakdown of the reconstruction error of the six final methods, for the thesis.

    python3 supervisor_evaluation/breakdown.py --data-root /scratch/work/zhangx29/data \
        --output-dir supervisor_evaluation/outputs/full

Runs every method on every frame of the seven test days with exactly the code paths of
`evaluate.py images` (same checkpoints, observations, clipping; the EnKF is read from the
evaluation's saved exports) and accumulates the squared error on blind walkable cells:

  * per frame (pooled over the four channels, and density alone), with the frame's people
    count and hour of day, so any binning can be done afterwards;
  * per "age" of the blind cell -- seconds since it was last observed that day;
  * split into cells that are empty or occupied in the ground truth.

All breakdowns use one common frame set for every method: from the end of the EnKF
warm-up to the last complete 4DVarNet window (the frames of the uncertainty protocol), so
that the methods are compared on the same frames in every bin. As a check, each method is
also scored on its own frame range, which must reproduce accuracy.csv.

Writes <output-dir>/breakdown/{per_frame.npz, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

METHODS = ["Senseiver-A", "Senseiver-G", "DINCAE", "4DVarNet", "4DVarNet aug. var.", "EnKF"]
if "varnet_aughead" not in ev.FINAL_CONFIG:              # EVAL_VARIANT without the aug. var. model
    METHODS.remove("4DVarNet aug. var.")
AGE_EDGES = [1, 2, 6, 16, 61, 201]           # bins: 1 | 2-5 | 6-15 | 16-60 | 61-200 | >200 | never
AGE_LABELS = ["1", "2-5", "6-15", "16-60", "61-200", ">200", "never"]
LO = np.asarray([0, -5, -5, 0], np.float32)[None, :, None, None]
HI = np.asarray([5, 5, 5, 2], np.float32)[None, :, None, None]


def age_since_observed(omask: np.ndarray) -> np.ndarray:
    """(T, H, W) seconds since each cell was last observed (0 if observed now, -1 never)."""
    T = omask.shape[0]
    age = np.empty(omask.shape, dtype=np.int32)
    last = np.full(omask.shape[1:], -1, dtype=np.int64)
    for t in range(T):
        last = np.where(omask[t], t, last)
        age[t] = np.where(last < 0, -1, t - last)
    return age


def age_bin(age: np.ndarray) -> np.ndarray:
    """Index into AGE_LABELS; age -1 (never observed) -> last bin."""
    b = np.searchsorted(AGE_EDGES, age, side="right") - 1
    return np.where(age < 0, len(AGE_LABELS) - 1, b)


def chunked(fn, frames, size=512):
    out = [fn(frames[i:i + size]) for i in range(0, len(frames), size)]
    return np.concatenate(out, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--days", type=int, default=0, help="limit number of days (smoke test)")
    ap.add_argument("--shard", default="",
                    help="k/n: only every n-th test day from the k-th; writes breakdown/partial_k_of_n.npz "
                         "(for jobs under a short time limit), then run --merge n")
    ap.add_argument("--merge", type=int, default=0,
                    help="n: combine the n partial files and write the outputs (no GPU needed)")
    a = ap.parse_args()
    ACC = ("se_age", "n_age", "se_age_rho", "se_occ", "n_occ", "se_common_channel")

    import torch
    from crowdcore import config
    config.CFG["data"]["root"] = str(a.data_root.resolve())
    from crowdcore import frame_reliability as fr, navigation, observation_model as om
    from methods.senseiver import dataset as senseiver_ds
    from methods.senseiver.network import Senseiver
    from methods.varnet.checks.model_io import load_solver
    from methods.dincae.checks import evaluate as dincae_eval
    from compare.compare5 import run_varnet

    MODELS, FC = ev.MODELS, ev.FINAL_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    walk = navigation.build_valid_mask_from_config().astype(bool)
    files = [Path(p) for p in om.split_files("test")]
    if a.days:
        files = files[:a.days]
    all_days = [fp.name.split("_")[0] for fp in files]
    if a.merge:
        # Sums are order-free; the per-frame rows are put back in day, then frame order, i.e.
        # the order of one unsharded run.
        parts = [np.load(a.output_dir / "breakdown" / f"partial_{k}_of_{a.merge}.npz", allow_pickle=False)
                 for k in range(a.merge)]
        done = sorted(d for p in parts for d in p["days"].tolist())
        if done != sorted(all_days):
            raise SystemExit(f"partials cover {done}, expected {sorted(all_days)}")
        acc = {k: sum(p[k] for p in parts) for k in ACC}
        se_age, n_age, se_age_rho = acc["se_age"], acc["n_age"], acc["se_age_rho"]
        se_occ, n_occ, se_common_channel = acc["se_occ"], acc["n_occ"], acc["se_common_channel"]
        common_cells = int(sum(int(p["common_cells"]) for p in parts))
        native = {m: [float(sum(p["native"][i, 0] for p in parts)), int(sum(p["native"][i, 1] for p in parts))]
                  for i, m in enumerate(METHODS)}
        keys = [k[4:] for k in parts[0].files if k.startswith("row_")]
        cols = {k: np.concatenate([p["row_" + k] for p in parts]) for k in keys}
        order = np.lexsort((cols["t"], np.array([all_days.index(d) for d in cols["day"]])))
        rows = [{k: (cols[k][i].item() if k != "day" else str(cols[k][i])) for k in keys} for i in order]
        files = []
    elif a.shard:
        k_sh, n_sh = (int(x) for x in a.shard.split("/"))
        files = files[k_sh::n_sh]

    def senseiver(key):
        ck = torch.load(MODELS / FC[key]["file"], map_location=device, weights_only=False)
        m = Senseiver(**ck["hparams"]).to(device); m.load_state_dict(ck["model"]); m.eval()
        return m
    if files:
        sense_a, sense_g = senseiver("senseiver_a"), senseiver("senseiver_g")
        var_solver, var_args, _ = load_solver(MODELS / FC["varnet_mse"]["file"], device)
        aug_solver = (load_solver(MODELS / FC["varnet_aughead"]["file"], device)[0]
                      if "varnet_aughead" in FC else None)
        dT = int(var_args["dT"])
        dincae_models, _, _ = dincae_eval.load_models(str(MODELS), str(MODELS / FC["dincae"]["file"]),
                                                      device)
        dincae_stats = dincae_eval.StateStats()
        obs_cfg = senseiver_ds.obs_config()
    nM, nA = len(METHODS), len(AGE_LABELS)

    # accumulators ---------------------------------------------------------------------------
    if not a.merge:
        rows = []                                          # per-frame records (common frames)
        se_age = np.zeros((nM, nA)); n_age = np.zeros(nA)  # pooled over channels
        se_age_rho = np.zeros((nM, nA))
        se_occ = np.zeros((nM, 2)); n_occ = np.zeros(2)    # density: truth empty / occupied
        se_common_channel = np.zeros((nM, 4))
        common_cells = 0
        native = {m: [0.0, 0] for m in METHODS}            # check against accuracy.csv

    for fp in files:
        day = fp.name.split("_")[0]
        X, times = om.load_state(str(fp))
        X = np.asarray(X, dtype=np.float32)
        T = len(X)
        obs = om.generate_observations(
            X, sensing_range=obs_cfg["sensing_range"], num_agents=obs_cfg["num_agents"],
            add_noise=obs_cfg["add_noise"], seed=om.day_seed(fp),
            valid_mask=navigation.build_valid_mask_from_config(X),
            obs_every_k=obs_cfg["obs_every_k"])
        Y, Om = obs["Y"].astype(np.float32), obs["Omega"].astype(bool)
        Yf, Of = Y.reshape(T, 4, -1), Om.reshape(T, -1)
        n200 = (T // dT) * dT
        frames = list(range(1, T - 1))

        P = {}
        P["Senseiver-A"] = np.clip(chunked(lambda f: ev._predict_senseiver_selected(
            sense_a, Yf, Of, f, device), frames), LO, HI)
        P["Senseiver-G"] = np.clip(chunked(lambda f: ev._predict_senseiver_selected(
            sense_g, Yf, Of, f, device), frames), LO, HI)
        Omc = np.repeat(Om[:, None], 4, axis=1)
        X0 = om.fill_missing_state(Y, Omc, method=obs_cfg["init_method"])
        # compare5's own path, in batches of 16 consecutive windows: the solver normalises the
        # gradient by its RMS over the BATCH, so the batch composition changes the result.
        # accuracy.csv was produced with batch 16, and so is this.
        for key, sol in (("4DVarNet", var_solver), ("4DVarNet aug. var.", aug_solver)):
            if sol is None:
                continue
            rec_v, nkeep = run_varnet(sol, Y, Omc, X0, dT, device, 16)
            assert nkeep == n200
            P[key] = np.clip(rec_v[1:], LO, HI)                 # frames 1 .. n200-1
            del rec_v
        Xt, rec, _, _, dmask = dincae_eval.predict_day(dincae_models, dincae_stats, str(fp), device,
                                                       batch=256)
        if not np.array_equal(Xt[0], X[1]) or not np.array_equal(dmask[0], obs["Omega_c"][1]):
            raise RuntimeError(f"DINCAE alignment check failed on {day}")
        P["DINCAE"] = dincae_eval.clip_bounds(rec)          # frames 1..T-2
        with np.load(a.output_dir / "raw" / "enkf_exports" / f"{day}.npz") as en:
            emean, eobs = en["mean"], en["observed"]
        warm = int(json.loads((a.output_dir / "raw" / "enkf" / f"{day}.json").read_text())
                   ["config"]["warmup"])
        # frame index -> row in each prediction array
        first = {"Senseiver-A": 1, "Senseiver-G": 1, "DINCAE": 1, "4DVarNet": 1,
                 "4DVarNet aug. var.": 1, "EnKF": warm}
        P["EnKF"] = emean
        if not np.array_equal(eobs[0], Om[warm]):
            raise RuntimeError(f"EnKF mask alignment failed on {day}")

        blind = walk[None] & ~Om                               # (T, H, W)
        keep = fr.scorable_frames(fp, T)                       # tracker-gap blocks left out
        if not keep.all():
            blind = blind & keep[:, None, None]
        age = age_since_observed(Om)
        abin = age_bin(age)
        hours = ((np.asarray(times, dtype=np.float64) + 9 * 3600) % 86400) / 3600

        # native frame ranges (the accuracy protocol of each method) ---------------------------
        for m in METHODS:
            p = P[m]
            ts = np.arange(first[m], first[m] + len(p))
            ts = ts[ts < T - 1] if m != "EnKF" else ts
            d = (p[: len(ts)] - X[ts]) ** 2                      # (n, 4, H, W)
            b = blind[ts][:, None].repeat(4, 1)
            native[m][0] += float(d[b].sum()); native[m][1] += int(b.sum())

        # common frames -----------------------------------------------------------------------
        common = np.arange(warm, min(n200, T - 1))
        for t in common:
            bt = blind[t]
            if not bt.any():
                continue
            rec_row = {"day": day, "t": int(t), "hour": float(hours[t]),
                       "people": float(X[t, 0][walk].sum()), "n_blind": int(bt.sum())}
            occ = X[t, 0] > 0
            ab = abin[t][bt]
            for i, m in enumerate(METHODS):
                e = P[m][t - first[m]] - X[t]                    # (4, H, W)
                e2 = (e ** 2)[:, bt]                             # (4, nb)
                se_common_channel[i] += e2.sum(axis=1, dtype=np.float64)
                rec_row[f"se_{i}"] = float(e2.sum())
                rec_row[f"se_rho_{i}"] = float(e2[0].sum())
                np.add.at(se_age[i], ab, e2.sum(0))
                np.add.at(se_age_rho[i], ab, e2[0])
                se_occ[i, 0] += float((e[0] ** 2)[bt & ~occ].sum())
                se_occ[i, 1] += float((e[0] ** 2)[bt & occ].sum())
            np.add.at(n_age, ab, 1)
            n_occ[0] += int((bt & ~occ).sum()); n_occ[1] += int((bt & occ).sum())
            common_cells += int(bt.sum())
            rows.append(rec_row)
        print(f"[breakdown] {day}: {len(common)} common frames", flush=True)
        del P, X, Y, obs, X0, Xt, rec, emean

    out = a.output_dir / "breakdown"
    out.mkdir(parents=True, exist_ok=True)
    if a.shard:
        keys = rows[0].keys() if rows else []
        np.savez(out / f"partial_{k_sh}_of_{n_sh}.npz", days=np.array([fp.name.split("_")[0] for fp in files]),
                 se_age=se_age, n_age=n_age, se_age_rho=se_age_rho, se_occ=se_occ, n_occ=n_occ,
                 se_common_channel=se_common_channel, common_cells=common_cells,
                 native=np.array([native[m] for m in METHODS], dtype=np.float64),
                 **{"row_" + k: np.array([r[k] for r in rows]) for k in keys})
        print(f"[breakdown] shard {a.shard}: {len(files)} days -> {out}/partial_{k_sh}_of_{n_sh}.npz", flush=True)
        return
    keys = rows[0].keys()
    np.savez(out / "per_frame.npz", methods=np.array(METHODS),
             **{k: np.array([r[k] for r in rows]) for k in keys})
    summary = {
        "methods": METHODS,
        "common_scored_cells_per_channel": common_cells,
        "common_rmse": {m: float(np.sqrt(se_common_channel[i].sum() / (4 * common_cells)))
                        for i, m in enumerate(METHODS)},
        "common_channel_rmse": {m: dict(zip(["density", "vx", "vy", "variance"],
                                           np.sqrt(se_common_channel[i] / common_cells).tolist()))
                                for i, m in enumerate(METHODS)},
        "native_rmse": {m: float(np.sqrt(v[0] / max(v[1], 1))) for m, v in native.items()},
        "age_labels": AGE_LABELS,
        "age_cells": n_age.tolist(),
        "age_rmse": {m: np.sqrt(se_age[i] / np.maximum(4 * n_age, 1)).tolist()
                     for i, m in enumerate(METHODS)},
        "age_rmse_density": {m: np.sqrt(se_age_rho[i] / np.maximum(n_age, 1)).tolist()
                             for i, m in enumerate(METHODS)},
        "occupancy_cells": {"empty": n_occ[0], "occupied": n_occ[1]},
        "density_rmse_empty_occupied": {m: np.sqrt(se_occ[i] / np.maximum(n_occ, 1)).tolist()
                                        for i, m in enumerate(METHODS)},
        "density_se_share_occupied": {m: float(se_occ[i, 1] / se_occ[i].sum())
                                      for i, m in enumerate(METHODS)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["native_rmse"], indent=1))
    import csv
    accuracy = {row["method"]: float(row["blind_rmse"])
                for row in csv.DictReader((a.output_dir / "accuracy.csv").open())}
    key_for = dict(zip(METHODS, ev.ACCURACY_METHODS))
    differences = {m: summary["native_rmse"][m] - accuracy[ev.FINAL_CONFIG[key_for[m]]["label"]]
                   for m in METHODS}
    (out / "accuracy_consistency.json").write_text(json.dumps(differences, indent=2) + "\n")
    for m, difference in differences.items():
        reference = accuracy[ev.FINAL_CONFIG[key_for[m]]["label"]]
        if abs(difference) > max(5e-4, 1e-3 * reference):
            raise RuntimeError(f"{m}: breakdown native RMSE does not reproduce accuracy.csv ({difference:+.6g})")
    print("[breakdown] native accuracy consistency passed", flush=True)


if __name__ == "__main__":
    main()
