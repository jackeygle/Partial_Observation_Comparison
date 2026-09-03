"""
compare_channels.py  —  per-channel MSE: EnKF vs 4DVarNet
========================================================================

The headline single-number MSE is dominated by the largest-scale channel (vx) and
hides that density is reconstructed poorly. This produces the HONEST per-channel
comparison: blind-zone MSE for each of the 4 channels (density, vx, vy, var) for the
the EnKF and 4DVarNet — both under EnKF's physical clip bounds
(fair). Writes check_outputs/eval/channel_metrics.json and a grouped bar chart.

Run on a GPU node:  python3 checks/compare_channels.py
"""
from __future__ import annotations
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
import config, navigation as nav, observation_model as om
from model_io import load_solver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENKFDIR = os.path.join(ROOT, "check_outputs", "enkf")
CH = config.get("grid", "channels")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.environ.get("VARNET_CKPT", "runs/varnet_b0_k1/varnet_best.pt")
solver, a, ck = load_solver(CKPT, dev)


def clip_np(x):                                          # EnKF _clip_bounds, on (n,4,H,W)
    x = x.copy()
    x[:, 0] = np.clip(x[:, 0], 0, 5); x[:, 1] = np.clip(x[:, 1], -5, 5)
    x[:, 2] = np.clip(x[:, 2], -5, 5); x[:, 3] = np.clip(x[:, 3], 0, 2)
    return x


def recon(Xf, out):
    x0 = om.fill_missing_state(out["Y"], out["Omega_c"], method=config.get("observation","init_method"))
    win = lambda x: om.to_windows(x, a["dT"], as_torch=True)
    yb=win(out["Y"]); mb=win(out["Omega_c"].astype(np.float32)); x0b=win(x0)
    recs=[]
    with torch.enable_grad():
        for i in range(0,yb.shape[0],8): recs.append(solver(x0b[i:i+8].to(dev),yb[i:i+8].to(dev),mb[i:i+8].to(dev)).detach().cpu())
    xr=torch.cat(recs,0).numpy(); nw=xr.shape[0]
    return clip_np(om.from_windows(xr)), np.asarray(x0)[:nw*a["dT"]]

# two regions: observed cells (Omega_c) and the unobserved blind zone (~Omega_c) —
# same split as eval_test_days.py (unobserved = mb < 0.5). Accumulate SE + count per
# (region, channel) for the EnKF and 4DVarNet.
REG = ["obs", "blind"]
s = {r: {m: np.zeros(4) for m in ("e", "v")} for r in REG}
cnt = {r: np.zeros(4) for r in REG}
for npz in sorted(glob.glob(os.path.join(ENKFDIR,"obs_*.npz"))):
    stem=os.path.basename(npz)[4:-4]; z=np.load(npz)
    day=[d for d in om.split_files("test") if stem in d][0]
    Xf,_=om.load_state(day); Xf=np.asarray(Xf)[:z["X_true"].shape[0]]
    out=om.generate_observations(Xf, add_noise=True, valid_mask=nav.build_valid_mask_from_config(Xf))
    vr,x0=recon(Xf,out); x0=clip_np(x0)
    er=np.load(os.path.join(ENKFDIR,f"est_{stem}.npz"))["Est"]
    n=min(vr.shape[0],er.shape[0]); Xf=Xf[:n]; vr=vr[:n]; x0=x0[:n]; er=er[:n]
    obsm=out["Omega_c"][:n].astype(bool)
    mask={"obs": obsm, "blind": ~obsm}
    for r in REG:
        for c in range(4):
            u=mask[r][:,c]
            s[r]["e"][c]+=((er[:,c]-Xf[:,c])**2)[u].sum()
            s[r]["v"][c]+=((vr[:,c]-Xf[:,c])**2)[u].sum()
            cnt[r][c]+=u.sum()

def per_ch(r, m): return {CH[c]: s[r][m][c]/cnt[r][c] for c in range(4)}
def allmse(r, m): return float(s[r][m].sum()/cnt[r].sum())
metrics={}
for r,rn in [("obs","observed region"),("blind","unobserved (blind) region")]:
    enkf, var = per_ch(r,"e"), per_ch(r,"v")
    ae, av = allmse(r,"e"), allmse(r,"v")
    print(f"\n== {rn} ==")
    print(f"{'channel':8s} {'EnKF':>9} {'4DVarNet':>9}")
    for c in range(4): print(f"{CH[c]:8s} {enkf[CH[c]]:>9.4f} {var[CH[c]]:>9.4f}")
    print(f"{'ALL':8s} {ae:>9.4f} {av:>9.4f}")
    metrics[r]={"enkf":enkf|{"ALL":ae}, "varnet":var|{"ALL":av}}
json.dump({"note":"per-channel MSE by region, EnKF-consistent clip, 7 test days x 400 frames; "
           "obs=observed cells (Omega_c), blind=unobserved cells", **metrics},
          open(os.path.join(ROOT,"check_outputs","eval","channel_metrics.json"),"w"), indent=2, default=float)

# two panels: observed region | unobserved (blind) region — EnKF vs 4DVarNet
labels=CH+["ALL"]; x=np.arange(len(labels)); w=0.34
fig,axes=plt.subplots(1,2,figsize=(16,5.5),sharey=False)
for ax,(r,rn) in zip(axes,[("obs","observed region  (Ω)"),("blind","unobserved / blind zone  (¬Ω)")]):
    enkf, var = per_ch(r,"e"), per_ch(r,"v"); ae, av = allmse(r,"e"), allmse(r,"v")
    # plot RMSE (sqrt of the MSE kept in the JSON): it is in the state's own units, so a
    # velocity bar reads directly in m/s, and it matches how the EnKF project scores its
    # filter (ENKF.py evaluate_enkf uses sqrt(mean(se))).
    ev=np.sqrt([enkf[c] for c in CH]+[ae]); vv=np.sqrt([var[c] for c in CH]+[av])
    ax.bar(x-w/2, ev, w, label="EnKF", color="#e08a3c")
    ax.bar(x+w/2, vv, w, label="4DVarNet", color="#0e6b8a")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("RMSE"); ax.legend()
    ax.set_title(rn, fontsize=12); ax.grid(axis="y", alpha=0.25)
    for i in range(len(labels)):
        ax.annotate("4DVarNet" if vv[i]<ev[i] else "EnKF", (x[i], max(vv[i],ev[i])),
                    ha="center", va="bottom", fontsize=7, color="#555")
fig.suptitle("Per-channel reconstruction RMSE by region (held-out test, same clip bounds) — lower is better", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
out=os.path.join(ROOT,"check_outputs","eval","compare_channels.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); print(f"\n[figure] {out}")
