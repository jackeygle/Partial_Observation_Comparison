"""Reconstruct one contiguous block of a test day with the five final methods and draw it.

    python3 -m supervisor_evaluation.frames_block [--day atc-20130811] [--start 30000] [--frames 5000]

For looking at the reconstructions, not for scoring: the reported numbers stay those of
`evaluate.py`. The prediction path is the one of `generate_comparison_images` (same
packaged checkpoints, observations, 4DVarNet initial field and window split, DINCAE
helper, saved EnKF export), so frame 32778 must reproduce `images/selected_predictions.npz`;
that is checked. The aug. var. 4DVarNet is left out (five-method thesis).

Writes to supervisor_evaluation/outputs/dev/frames_<n>/:
    predictions.npz        truth, observation, mask, the five predictions (4 channels),
                           DINCAE and EnKF density sigma, per-frame scores
    boards/board_*.png     every 25th frame: observation, truth, five reconstructions,
                           absolute error; one colour scale for the whole block
    animation.gif          every 5th frame, density only, the same scales
    rmse_over_time.png     per-frame blind RMSE of each method and the number of people
    summary.json           block RMSE per method, colour scales, consistency check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from supervisor_evaluation.evaluate import (FINAL_CONFIG, HERE, MODELS, STATE_SHAPE, _predict_senseiver_selected,
                                            _predict_varnet_selected)

METHODS = ("Senseiver-A", "Senseiver-G (ours)", "DINCAE", "4DVarNet", "EnKF")
COLORS = ("#e57a9f", "#4b3aa6", "#2a7ad5", "#eda100", "#1aae7a")
LO = np.asarray([0, -5, -5, 0], np.float32)[None, :, None, None]
HI = np.asarray([5, 5, 5, 2], np.float32)[None, :, None, None]


def predict(day: str, start: int, n: int, data_root: Path, full_dir: Path) -> dict:
    import torch
    from crowdcore import config
    config.CFG["data"]["root"] = str(data_root.resolve())
    from crowdcore import navigation, observation_model as om
    from methods.senseiver import dataset as senseiver_ds
    from methods.senseiver.network import Senseiver
    from methods.varnet.checks.model_io import load_solver
    from methods.dincae.checks import evaluate as dincae_eval

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("run on a GPU node")
    fp = next(Path(p) for p in om.split_files("test") if Path(p).name.startswith(day))
    walk = navigation.build_valid_mask_from_config().astype(bool)
    c, h, w = STATE_SHAPE
    frames = list(range(start, start + n))

    X, _ = om.load_state(str(fp))
    X = np.asarray(X, dtype=np.float32)
    if start < 1 or start + n > len(X) - 1:
        raise ValueError(f"block [{start}, {start + n}) outside frames 1..{len(X) - 2}")
    obs_cfg = senseiver_ds.obs_config()
    obs = om.generate_observations(
        X, sensing_range=obs_cfg["sensing_range"], num_agents=obs_cfg["num_agents"],
        add_noise=obs_cfg["add_noise"], seed=om.day_seed(fp),
        valid_mask=navigation.build_valid_mask_from_config(X),
        obs_every_k=obs_cfg["obs_every_k"])
    Y, Omask = obs["Y"].astype(np.float32), obs["Omega"].astype(bool)
    Yflat, Oflat = Y.reshape(len(X), c, h * w), Omask.reshape(len(X), h * w)

    pred = np.empty((n, len(METHODS), c, h, w), np.float32)
    for k, key in enumerate(("senseiver_a", "senseiver_g")):
        ck = torch.load(MODELS / FINAL_CONFIG[key]["file"], map_location=device, weights_only=False)
        m = Senseiver(**ck["hparams"]).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        for b in range(0, n, 64):                       # batches of 64 frames, as evaluated
            p = _predict_senseiver_selected(m, Yflat, Oflat, frames[b:b + 64], device)
            pred[b:b + 64, k] = np.clip(p, LO, HI)
        del m
        print(f"[block] {key} done", flush=True)

    dincae_models, _, _ = dincae_eval.load_models(
        str(MODELS), str(MODELS / FINAL_CONFIG["dincae"]["file"]), device)
    stats = dincae_eval.StateStats()
    Xt, rec, _, dsd, dmask = dincae_eval.predict_day(dincae_models, stats, str(fp), device, batch=256)
    if not np.array_equal(Xt[0], X[1]) or not np.array_equal(dmask[start - 1], obs["Omega_c"][start]):
        raise RuntimeError("DINCAE frame/observation alignment check failed")
    rec = dincae_eval.clip_bounds(rec)
    pred[:, 2] = rec[start - 1:start - 1 + n]          # DINCAE row i is frame i + 1
    sigma = np.empty((n, 2, h, w), np.float32)
    sigma[:, 0] = dsd[start - 1:start - 1 + n, 0] * stats.std[0]
    del Xt, rec, dsd, dmask
    print("[block] dincae done", flush=True)

    solver, args, _ = load_solver(MODELS / FINAL_CONFIG["varnet_mse"]["file"], device)
    dT = int(args["dT"])
    if start % dT or n % dT:
        raise ValueError(f"start and frames must be multiples of the 4DVarNet window {dT}")
    Omc = np.repeat(Omask[:, None], c, axis=1)
    X0 = om.fill_missing_state(Y, Omc, method=obs_cfg["init_method"])   # whole day, as evaluated
    pv = _predict_varnet_selected(solver, dT, Y, Omc, X0, frames, device)
    pred[:, 3] = np.stack([pv[t] for t in frames])
    del pv, X0
    print("[block] 4dvarnet done", flush=True)

    enkf_json = json.loads((full_dir / "raw" / "enkf" / f"{day}.json").read_text())
    warm = int(enkf_json["config"]["warmup"])
    with np.load(full_dir / "raw" / "enkf_exports" / f"{day}.npz") as en:
        sl = slice(start - warm, start - warm + n)
        if start < warm or not np.array_equal(en["observed"][sl], Omask[start:start + n]) \
                or not np.allclose(en["truth"][sl], X[start:start + n], rtol=0, atol=1e-6):
            raise RuntimeError("EnKF export alignment check failed")
        pred[:, 4] = en["mean"][sl]
        sigma[:, 1] = en["spread"][sl, 0]
    print("[block] enkf (saved export) done", flush=True)

    return dict(truth=X[start:start + n], observation=Y[start:start + n],
                observed=Omask[start:start + n], walkable=walk, predictions=pred,
                density_sigma=sigma, frame=np.asarray(frames, np.int32))


def check_against_selection(a: dict, day: str, full_dir: Path) -> dict:
    """The block's prediction of a frame drawn in the thesis must match the saved selection."""
    z = np.load(full_dir / "images" / "selected_predictions.npz")
    names = list(z["method_names"])
    src = ("Senseiver-A", "Senseiver-G", "DINCAE", "4DVarNet MSE", "EnKF")
    out = {}
    for i in np.flatnonzero(z["day"] == day):
        f = int(z["frame"][i])
        if not a["frame"][0] <= f <= a["frame"][-1]:
            continue
        j = f - int(a["frame"][0])
        out[str(f)] = {m: float(np.abs(a["predictions"][j, k] - z["predictions"][i, names.index(s)]).max())
                       for k, (m, s) in enumerate(zip(METHODS, src))}
    return out


def scores(a: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    walk, obs = a["walkable"], a["observed"]
    blind = walk[None] & ~obs                                          # (n, h, w)
    err2 = (a["predictions"] - a["truth"][:, None]) ** 2               # (n, M, c, h, w)
    nb = blind.sum((1, 2)).astype(np.float64)
    pooled = np.sqrt((err2 * blind[:, None, None]).sum((2, 3, 4)) / (4 * nb[:, None]))
    dens = np.sqrt((err2[:, :, 0] * blind[:, None]).sum((2, 3)) / nb[:, None])
    tot = {m: {"pooled_rmse": float(np.sqrt((err2[:, k] * blind[:, None]).sum() / (4 * nb.sum()))),
               "density_rmse": float(np.sqrt((err2[:, k, 0] * blind).sum() / nb.sum()))}
           for k, m in enumerate(METHODS)}
    return pooled, dens, tot


def _board_figure(a: dict, day: str):
    """The board figure and an update(i) that draws frame i; scales span the whole block."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    walk, truth, pred = a["walkable"], a["truth"], a["predictions"]
    n = len(truth)
    vmax = float(np.percentile(truth[:, 0][:, walk], 99.5))
    errs = np.abs(pred[:, :, 0] - truth[:, None, 0])
    emax = float(np.percentile(errs[:, :, walk], 99.5))
    pooled, dens, _ = scores(a)
    people = truth[:, 0].sum((1, 2))

    def masked(v, keep):
        return np.ma.masked_where(~keep, v)

    fig, axes = plt.subplots(2, 7, figsize=(12, 6.4), constrained_layout=True)
    cm_d = plt.get_cmap("Blues").copy(); cm_d.set_bad("#bdbdbd")
    cm_e = plt.get_cmap("Reds").copy(); cm_e.set_bad("#bdbdbd")
    titles = ["observed", "truth"] + list(METHODS)
    # 'observed' panel: walls grey, walkable but unobserved cells white (an underlay
    # shows through the transparent masked cells)
    under = np.where(walk[..., None], 1.0, 0.741) * np.ones((36, 12, 3))
    axes[0, 0].imshow(under)
    cm_o = plt.get_cmap("Blues").copy(); cm_o.set_bad((1, 1, 1, 0))
    ims = []
    for col in range(7):
        ims.append(axes[0, col].imshow(np.zeros((36, 12)), cmap=cm_o if col == 0 else cm_d,
                                       vmin=0, vmax=vmax))
        axes[0, col].set_title(titles[col].replace(" (ours)", "\n(ours)"), fontsize=9)
    eims, etxt = [], []
    for col in range(7):
        if col < 2:
            axes[1, col].axis("off")
            continue
        eims.append(axes[1, col].imshow(np.zeros((36, 12)), cmap=cm_e, vmin=0, vmax=emax))
        etxt.append(axes[1, col].set_title("", fontsize=8))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(ims[1], ax=axes[0, :].tolist(), shrink=0.8, label="density (people/m$^2$)")
    fig.colorbar(eims[0], ax=axes[1, 2:].tolist(), shrink=0.8, label="|error| (people/m$^2$)")
    axes[1, 0].text(0, 0.5, "row 2: absolute density\nerror; above each panel\nthe frame's blind RMSE\n"
                    "(4 channels pooled)\n\ngrey: not walkable\nwhite in 'observed':\nnot observed",
                    fontsize=8, va="center", transform=axes[1, 0].transAxes)
    sup = fig.suptitle("", fontsize=10)

    def update(i):
        o = a["observed"][i]
        ims[0].set_data(masked(a["observation"][i, 0], walk & o))
        ims[1].set_data(masked(truth[i, 0], walk))
        for k in range(5):
            ims[2 + k].set_data(masked(pred[i, k, 0], walk))
            eims[k].set_data(masked(errs[i, k], walk))
            etxt[k].set_text(f"RMSE {pooled[i, k]:.3f}")
        sup.set_text(f"{day}   frame {int(a['frame'][i])}   people {people[i]:.0f}   "
                     f"observed {100 * o[walk].mean():.0f}% of walkable cells")

    return fig, update, plt, {"density_vmax": vmax, "error_vmax": emax}, pooled, dens, people


def _render_chunk(args) -> int:
    path, day, out, idx = args
    with np.load(path) as z:
        a = {k: z[k] for k in ("truth", "observation", "observed", "walkable", "predictions", "frame")}
    fig, update, plt, _, _, _, _ = _board_figure(a, day)
    for i in idx:
        update(i)
        fig.savefig(Path(out) / "boards" / f"board_{int(a['frame'][i]):05d}.png", dpi=110)
    plt.close(fig)
    return len(idx)


def redraw_boards(out: Path, day: str, every: int, workers: int) -> None:
    """Draw boards from the saved predictions.npz, in parallel; no inference."""
    from multiprocessing import Pool
    path = out / "predictions.npz"
    with np.load(path) as z:
        n = len(z["frame"])
    (out / "boards").mkdir(parents=True, exist_ok=True)
    idx = list(range(0, n, every))
    chunks = [(str(path), day, str(out), idx[k::workers]) for k in range(workers)]
    with Pool(workers) as pool:
        done = 0
        for m in pool.imap_unordered(_render_chunk, chunks):
            done += m
            print(f"[boards] {done}/{len(idx)}", flush=True)


def draw(a: dict, out: Path, day: str, board_every: int, gif_every: int) -> dict:
    from PIL import Image
    fig, update, plt, scales, pooled, dens, people = _board_figure(a, day)
    n = len(a["truth"])
    boards = out / "boards"; boards.mkdir(parents=True, exist_ok=True)
    for i in range(0, n, board_every):
        update(i)
        fig.savefig(boards / f"board_{int(a['frame'][i]):05d}.png", dpi=110)
    frames = []
    for i in range(0, n, gif_every):
        update(i)
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        frames.append(img.resize((img.width // 2, img.height // 2)).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out / "animation.gif", save_all=True, append_images=frames[1:],
                   duration=50, loop=0, optimize=True)
    plt.close(fig)

    # per-frame RMSE over the block, with a one-minute moving average
    t = (a["frame"] - a["frame"][0]) / 60.0
    k60 = np.ones(60) / 60
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    for k, m in enumerate(METHODS):
        ax1.plot(t, pooled[:, k], color=COLORS[k], lw=0.3, alpha=0.25)
        ax1.plot(t[59:], np.convolve(pooled[:, k], k60, "valid"), color=COLORS[k], lw=1.2, label=m)
        ax2.plot(t[59:], np.convolve(dens[:, k], k60, "valid"), color=COLORS[k], lw=1.2)
    ax3.plot(t, people, color="k", lw=0.8)
    ax1.set_ylabel("blind RMSE\n(4 channels pooled)"); ax2.set_ylabel("blind RMSE\n(density)")
    ax3.set_ylabel("people in\nthe corridor"); ax3.set_xlabel(f"minutes from frame {int(a['frame'][0])}")
    ax1.legend(ncol=5, fontsize=8, loc="upper left")
    ax1.set_title("thin: every frame; thick: one-minute moving average", fontsize=9)
    fig.savefig(out / "rmse_over_time.png", dpi=130)
    plt.close(fig)
    return scales


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", default="atc-20130811")
    ap.add_argument("--start", type=int, default=30000)
    ap.add_argument("--frames", type=int, default=5000)
    ap.add_argument("--board-every", type=int, default=25)
    ap.add_argument("--gif-every", type=int, default=5)
    ap.add_argument("--data-root", type=Path, default=Path("/scratch/work/zhangx29/data"))
    ap.add_argument("--full-dir", type=Path, default=HERE / "outputs" / "full")
    ap.add_argument("--redraw-boards", action="store_true",
                    help="only draw boards (every --board-every frames) from the saved predictions.npz")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = a.full_dir.parent / "dev" / f"frames_{a.frames}"
    out.mkdir(parents=True, exist_ok=True)
    if a.redraw_boards:
        redraw_boards(out, a.day, a.board_every, a.workers)
        return
    arrays = predict(a.day, a.start, a.frames, a.data_root, a.full_dir)
    check = check_against_selection(arrays, a.day, a.full_dir)
    pooled, dens, tot = scores(arrays)
    np.savez_compressed(out / "predictions.npz", **arrays, pooled_rmse=pooled, density_rmse=dens,
                        method_names=np.asarray(METHODS),
                        sigma_method_names=np.asarray(("DINCAE", "EnKF")))
    scales = draw(arrays, out, a.day, a.board_every, a.gif_every)
    summary = {"day": a.day, "start": a.start, "frames": a.frames, "block_rmse": tot,
               "colour_scales": scales,
               "max_abs_diff_vs_selected_predictions": check,
               "note": "visual inspection only; reported scores are those of evaluate.py"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
