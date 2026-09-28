"""Cache the one test frame the schematics show, so drawing never touches the full data.

    python3 -m thesis_figures.prepare_frame        # -> thesis_figures/data/example_frame.npz

Truth, noisy observation, observation mask and DINCAE's reconstruction and sigma come
from the evaluation's saved selection (supervisor_evaluation/.../selected_predictions.npz).
The robot positions are not saved there, so the day's observations are regenerated with
the same seed; the regenerated mask must equal the saved one, or the script stops.
"""
from __future__ import annotations

import os

import numpy as np

DAY, FRAME = "atc-20130811", 32778
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(ROOT, "supervisor_evaluation/outputs/full/images/selected_predictions.npz")
OUT = os.path.join(ROOT, "thesis_figures/data/example_frame.npz")


def main() -> None:
    from crowdcore import config
    config.CFG["data"]["root"] = "/scratch/work/zhangx29/data"
    from crowdcore import navigation, observation_model as om
    from methods.senseiver import dataset as senseiver_ds

    with np.load(SEL) as z:
        i = int(np.flatnonzero((z["day"] == DAY) & (z["frame"] == FRAME))[0])
        names = list(z["method_names"]); snames = list(z["spread_method_names"])
        rec = {"truth": z["truth"][i], "observation": z["observation"][i],
               "observed": z["observed"][i], "walkable": z["walkable"],
               "dincae_mean": z["predictions"][i, names.index("DINCAE")],
               "predictions": z["predictions"][i], "method_names": np.array(names),
               "spreads": z["density_spread"][i], "spread_method_names": np.array(snames),
               "dincae_density_sigma": z["density_spread"][i, snames.index("DINCAE")]}

    fp = next(p for p in om.split_files("test") if DAY in p)
    X = np.asarray(om.load_state(fp)[0][:FRAME + 1], dtype=np.float32)
    cfg = senseiver_ds.obs_config()
    obs = om.generate_observations(
        X, sensing_range=cfg["sensing_range"], num_agents=cfg["num_agents"],
        add_noise=cfg["add_noise"], seed=om.day_seed(fp),
        valid_mask=navigation.build_valid_mask_from_config(X), obs_every_k=cfg["obs_every_k"])
    if not np.array_equal(obs["Omega"][FRAME], rec["observed"]) or \
            not np.allclose(X[FRAME], rec["truth"], atol=1e-6):
        raise RuntimeError("regenerated observations do not match the saved frame")
    rec["robots"] = obs["positions"][FRAME]                      # (3, 2) grid (row, col)
    rec["robot_tracks"] = obs["positions"][FRAME - 20:FRAME + 1]  # last 20 s of each route
    rec["sensing_range"] = np.int64(cfg["sensing_range"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, day=DAY, frame=FRAME, **rec)
    print(f"[frame] {OUT}  robots {rec['robots'].tolist()}  observed "
          f"{rec['observed'][rec['walkable']].mean():.2f} of walkable cells")


if __name__ == "__main__":
    main()
