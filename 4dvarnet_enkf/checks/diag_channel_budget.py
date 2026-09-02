"""
diag_channel_budget.py  —  how the 4 physical channels are actually handled, with real numbers

Answers two questions that the framework figures state but never quantify:

  1. the 4 channels live on different scales, so WHERE do they mix and where are they kept
     apart, and which one actually dominates the variational cost?
  2. what does one convolution output really compute? A single (t,h,w) voxel is taken from real
     data, run through the trained psi by hand, and every partial sum is recorded.

Output: check_outputs/eval/channel_budget.json, which checks/plot_math_detail.py reads. Nothing
is typed into a figure.

Needs a GPU: the cost is decomposed both at x0 and at the finished reconstruction, and the
second one means running the solver's 20 unrolled iterations over a 200-frame window. On CPU
that takes minutes, so submit it rather than running it on the login node.

    sbatch sbatch/submit_channel_budget.sbatch
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))
import config                                                        # noqa: E402
import observation_model as om                                       # noqa: E402
import navigation as nav                                             # noqa: E402
from model_io import load_solver                                     # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval", "channel_budget.json")
CHAN = list(config.get("grid", "channels"))
OBS_STD = list(config.get("observation", "obs_std"))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S, A, _ = load_solver(os.path.join(ROOT, "runs/varnet_b0_k1/varnet_best.pt"), DEV)
S.eval()
print(f"[device] {DEV}"
      + (f"  {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else "  (SLOW)"))
DT = A["dT"]
C = len(CHAN)

# ── one real window: truth, observation, mask, initial guess ────────────────────────────
day = sorted(config.split_files("test"))[0] if hasattr(config, "split_files") else None
import observation_model as _om                                      # noqa: E402
_files = _om.split_files("test")
X_day, _ = _om.load_state(_files[0])
valid = nav.build_valid_mask_from_config(X_day)
n = (X_day.shape[0] // DT) * DT
X_day = X_day[:n]
out = _om.generate_observations(X_day, A["sensing_range"], A["num_agents"],
                               add_noise=True, seed=A.get("data_seed", 0) or 0,
                               valid_mask=valid, obs_every_k=1)
X0_day = _om.fill_missing_state(out["Y"], out["Omega_c"],
                               method=config.get("observation", "init_method"))
w = lambda a: torch.from_numpy(_om.to_windows(a, DT)).float()
# pick the window with the most people in it, so the numbers are not all zero
_pop = X_day[:, 0].reshape(X_day.shape[0], -1).sum(1)
_wi = int(np.argmax(_pop.reshape(-1, DT).sum(1)))
X = w(X_day)[_wi:_wi + 1].to(DEV)
Y = w(out["Y"])[_wi:_wi + 1].to(DEV)
M = w(out["Omega_c"].astype(np.float32))[_wi:_wi + 1].to(DEV)
X0 = w(X0_day)[_wi:_wi + 1].to(DEV)
print(f"[window] day={os.path.basename(_files[0])}  window {_wi}  shape {tuple(X.shape)}")

# ── 1. the scale of each channel, from the truth itself ─────────────────────────────────
scale = {}
for i, c in enumerate(CHAN):
    v = X[:, i].cpu().numpy().ravel()
    nz = v[v != 0]
    scale[c] = dict(std=float(v.std()), absmean=float(np.abs(v).mean()),
                    vmin=float(v.min()), vmax=float(v.max()),
                    nonzero_frac=float((v != 0).mean()),
                    std_nonzero=float(nz.std()) if nz.size else 0.0,
                    obs_std=float(OBS_STD[i]))

# ── 2. the cost, decomposed per channel, at x0 ──────────────────────────────────────────
import time as _time
_t0 = _time.time()
with torch.enable_grad():                       # the solver needs autograd even in eval
    Xhat = S(X0, Y, M).detach()
print(f"[solve] one window, {S.n_iter} iterations, CPU: {_time.time() - _t0:.1f}s")
a_obs = float(S.var_cost.alpha_obs.detach()) ** 2
a_reg = float(S.var_cost.alpha_reg.detach()) ** 2
w_obs = (S.var_cost.w_obs.detach() ** 2).cpu().numpy()
w_reg = (S.var_cost.w_reg.detach() ** 2).cpu().numpy()
n_per = X.numel() / X.shape[1]


def decompose(x):
    with torch.no_grad():
        dy = (x - Y) * M
        dx = x - S.phi(x)
    sq_obs = (dy ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
    sq_reg = (dx ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
    t_obs = a_obs * sq_obs * w_obs / n_per
    t_reg = a_reg * sq_reg * w_reg / n_per
    Jt = float(t_obs.sum() + t_reg.sum())
    return dict(J=Jt, obs_total=float(t_obs.sum()), reg_total=float(t_reg.sum()),
                per_channel={c: dict(raw_obs=float(sq_obs[i] / n_per),
                                     raw_reg=float(sq_reg[i] / n_per),
                                     term_obs=float(t_obs[i]), term_reg=float(t_reg[i]),
                                     share=float((t_obs[i] + t_reg[i]) / Jt))
                             for i, c in enumerate(CHAN)})


cost = dict(alpha_obs=float(S.var_cost.alpha_obs.detach()),
            alpha_reg=float(S.var_cost.alpha_reg.detach()),
            w_obs=S.var_cost.w_obs.detach().cpu().tolist(),
            w_reg=S.var_cost.w_reg.detach().cpu().tolist(),
            n_per_channel=int(n_per),
            at_x0=decompose(X0), at_xhat=decompose(Xhat),
            mse_per_channel={c: float(((Xhat - X) ** 2)[:, i].mean()) for i, c in
                             enumerate(CHAN)})

# ── 3. one convolution output, by hand, on real numbers ─────────────────────────────────
psi = S.phi.branch_fine.psi
Wt = (psi.conv.weight * psi.centre_tap_mask).detach()          # the EFFECTIVE weight
Wraw = psi.conv.weight.detach()
b = psi.conv.bias.detach()
kt, kh, kw = psi.conv.kernel_size

# pick a voxel with people in it and a full interior neighbourhood
dens = X[0, 0].cpu().numpy()
inner = dens[1:-1, 1:-1, 1:-1]
_t, _h, _w = np.unravel_index(int(np.argmax(inner)), inner.shape)
t, h, ww = _t + 1, _h + 1, _w + 1
patch = X[0, :, t - 1:t + 2, h - 1:h + 2, ww - 1:ww + 2]        # (C,3,3,3)
with torch.no_grad():                                           # every feature at this voxel
    _all = torch.nn.functional.conv3d(X, Wt, b, padding=psi.conv.padding)[0, :, t, h, ww]
o = int(torch.argmax(_all))                                     # the largest POSITIVE one, so
                                                                # ReLU passes it and the example
                                                                # can be followed into phi

prod = (Wt[o] * patch)                                          # (C,3,3,3)
partial = prod.sum(dim=(1, 2, 3)).cpu().numpy()                       # per input channel
per_tau = prod.sum(dim=(2, 3)).cpu().numpy()                          # (C,3) per channel per tau
psi_val = float(prod.sum() + b[o])
# verify against the real conv3d, so the hand arithmetic cannot be wrong
with torch.no_grad():
    ref = torch.nn.functional.conv3d(X, Wt, b, padding=psi.conv.padding)[0, o, t, h, ww]
assert abs(psi_val - float(ref)) < 1e-3, f"hand sum {psi_val} != conv3d {float(ref)}"

# what the mask threw away, and the full prior output at the same voxel
dropped = [float(Wraw[o, c, kt // 2, kh // 2, kw // 2] * patch[c, 1, 1, 1]) for c in range(C)]
with torch.no_grad():
    phi_here = S.phi(X)[0, :, t, h, ww].cpu().numpy()
truth_here = X[0, :, t, h, ww].cpu().numpy()

conv = dict(window=_wi, day=os.path.basename(_files[0]), t=int(t), h=int(h), w=int(ww),
            out_feature=o, kernel=[int(kt), int(kh), int(kw)],
            patch=patch.cpu().tolist(), weight=Wt[o].cpu().tolist(),
            bias=float(b[o]), partial_per_channel=partial.tolist(),
            per_tau=per_tau.tolist(), psi=psi_val, relu=max(psi_val, 0.0),
            centre_values=patch[:, 1, 1, 1].cpu().tolist(),
            dropped_if_unmasked=dropped,
            phi_at_voxel=phi_here.tolist(), truth_at_voxel=truth_here.tolist(),
            residual_at_voxel=(truth_here - phi_here).tolist())

# ── 3b. J, as an arithmetic chain: one voxel -> one frame -> the window ─────────────────
# The point is that every step is checkable. The last line asserts the hand-assembled total
# against VarCost's own forward, so the figure cannot disagree with the module.
with torch.no_grad():
    phi_xh = S.phi(Xhat)
R_obs = ((Xhat - Y) * M)[0]                                    # (C,T,H,W)
R_reg = (Xhat - phi_xh)[0]

# two voxels in the same frame: one the robots saw, one they did not. Pick the busiest of each
# so neither is a row of zeros.
_dens = X[0, 0]
_seen = M[0, 0] > 0.5
_tf = int(torch.argmax((_dens * _seen).reshape(DT, -1).sum(1)))  # a frame with observations
_fl = _dens[_tf].reshape(-1)
_ob = (_seen[_tf].reshape(-1))
_iv_obs = int(torch.argmax(torch.where(_ob, _fl, torch.full_like(_fl, -1.0))))
_iv_bl = int(torch.argmax(torch.where(~_ob, _fl, torch.full_like(_fl, -1.0))))


def voxel(idx):
    _h, _w = divmod(idx, W_)
    return dict(t=_tf, h=int(_h), w=int(_w),
                observed=bool(M[0, 0, _tf, _h, _w] > 0.5),
                x=Xhat[0, :, _tf, _h, _w].cpu().tolist(),
                truth=X[0, :, _tf, _h, _w].cpu().tolist(),
                y=Y[0, :, _tf, _h, _w].cpu().tolist(),
                omega=M[0, :, _tf, _h, _w].cpu().tolist(),
                phi=phi_xh[0, :, _tf, _h, _w].cpu().tolist(),
                r_obs=R_obs[:, _tf, _h, _w].cpu().tolist(),
                r_reg=R_reg[:, _tf, _h, _w].cpu().tolist())


W_ = X.shape[-1]
j_ex = dict(
    voxel_observed=voxel(_iv_obs), voxel_blind=voxel(_iv_bl),
    frame=dict(t=_tf, obs_cells=int(_ob.sum()), cells=int(_ob.numel()),
               sq_obs=(R_obs[:, _tf] ** 2).sum(dim=(1, 2)).cpu().tolist(),
               sq_reg=(R_reg[:, _tf] ** 2).sum(dim=(1, 2)).cpu().tolist()),
    window=dict(frames=DT, n_per_channel=int(n_per),
                sq_obs=(R_obs ** 2).sum(dim=(1, 2, 3)).cpu().tolist(),
                sq_reg=(R_reg ** 2).sum(dim=(1, 2, 3)).cpu().tolist()),
)
# assemble J by hand from those sums, then check it against the module
_sqo = np.array(j_ex["window"]["sq_obs"])
_sqr = np.array(j_ex["window"]["sq_reg"])
_to = a_obs * w_obs * _sqo / n_per
_tr = a_reg * w_reg * _sqr / n_per
j_ex["term_obs"] = _to.tolist()
j_ex["term_reg"] = _tr.tolist()
j_ex["J_by_hand"] = float(_to.sum() + _tr.sum())
with torch.no_grad():
    j_ex["J_from_varcost"] = float(S.var_cost(R_reg[None], R_obs[None]))
_err = abs(j_ex["J_by_hand"] - j_ex["J_from_varcost"])
assert _err < 1e-4, f"hand-assembled J {j_ex['J_by_hand']} != VarCost {j_ex['J_from_varcost']}"
print(f"[J] by hand {j_ex['J_by_hand']:.6f}  vs VarCost {j_ex['J_from_varcost']:.6f}  "
      f"(diff {_err:.2e})")


# ── 3c. the WHOLE prior chain at that one voxel ─────────────────────────────────────────
# The worked psi example above covers one of 32 features of the first of three weight layers
# of one of two branches. This records every stage, so the figure can follow x -> Phi(x) end
# to end at a single point. The 1x1 convs are plain matrix multiplies at a fixed voxel, so
# they are evaluated directly and then checked against the module's own forward.
import torch.nn.functional as _F                                # noqa: E402

_bf = S.phi.branch_fine
_W1, _b1 = _bf.phi[0].weight.detach()[:, :, 0, 0, 0], _bf.phi[0].bias.detach()
_W2, _b2 = _bf.phi[2].weight.detach()[:, :, 0, 0, 0], _bf.phi[2].bias.detach()

with torch.no_grad():
    _psi_all = _F.conv3d(X, Wt, b, padding=psi.conv.padding)[0, :, t, h, ww]   # (32,)
    _a1 = _F.relu(_psi_all)                                                    # after ReLU
    _z1 = _W1 @ _a1 + _b1                                                      # phi.0
    _a2 = _F.relu(_z1)
    _fine = _W2 @ _a2 + _b2                                                    # phi.2 -> (4,)
    # the module's own fine branch, for the cross-check
    _fine_ref = _bf(X)[0, :, t, h, ww]
    # the coarse path, following GENN.forward exactly
    _B, _C, _T, _H, _Wd = X.shape
    _xr = X.reshape(_B * _T, _C, _H, _Wd)
    _co = _F.avg_pool2d(_xr, kernel_size=S.phi.scale, ceil_mode=True)
    _Hc, _Wc = _co.shape[-2:]
    _co = S.phi.branch_coarse(_co.reshape(_B, _C, _T, _Hc, _Wc))
    _up = S.phi.up(_co.reshape(_B * _T, _C, _Hc, _Wc))[..., :_H, :_Wd]
    _up = _up.reshape(_B, _C, _T, _H, _Wd)[0, :, t, h, ww]
    _phi_ref = S.phi(X)[0, :, t, h, ww]

assert torch.allclose(_fine, _fine_ref, atol=1e-3), "hand-evaluated fine branch mismatch"
assert torch.allclose(_fine + _up, _phi_ref, atol=1e-3), "fine + coarse != Phi"

_NP = lambda m: sum(q.numel() for q in m.parameters())
chain = dict(
    voxel=[int(t), int(h), int(ww)],
    x=X[0, :, t, h, ww].cpu().tolist(),
    psi=_psi_all.cpu().tolist(), psi_relu=_a1.cpu().tolist(),
    phi0=_z1.cpu().tolist(), phi0_relu=_a2.cpu().tolist(),
    fine=_fine.cpu().tolist(), coarse_up=_up.cpu().tolist(),
    phi=_phi_ref.cpu().tolist(),
    residual=(X[0, :, t, h, ww] - _phi_ref).cpu().tolist(),
    n_dead_after_psi=int((_a1 == 0).sum()), n_dead_after_phi0=int((_a2 == 0).sum()),
    coarse_grid=[int(_Hc), int(_Wc)], scale=int(S.phi.scale),
    params=dict(psi=_NP(_bf.psi), phi0=_NP(_bf.phi[0]), phi2=_NP(_bf.phi[2]),
                branch=_NP(_bf), up=_NP(S.phi.up), total=_NP(S.phi)),
)
print(f"[chain] psi {len(chain['psi'])} feats, {chain['n_dead_after_psi']} killed by ReLU; "
      f"phi0 {chain['n_dead_after_phi0']} killed; "
      f"fine {[round(v, 4) for v in chain['fine']]}  "
      f"coarse {[round(v, 4) for v in chain['coarse_up']]}")


# ── 4. where the channels mix and where they do not ─────────────────────────────────────
# read off the modules rather than asserted in prose
mixing = [
    ("observation mask $\\Omega$", "kept apart",
     f"built per (channel, cell); all {C} are sensed together, so it is the same "
     f"spatial mask broadcast {C}x"),
    ("observation term of $J$", "kept apart",
     f"squares summed within each channel, then weighted by $w_{{obs,c}}^2$"),
    ("$\\psi$  (prior, first conv)", "MIXED",
     f"in_channels={psi.conv.in_channels} $\\rightarrow$ out_channels="
     f"{psi.conv.out_channels}: every feature reads all {C}"),
    ("$\\varphi$  (prior, pointwise)", "MIXED",
     f"{psi.conv.out_channels}$\\rightarrow${psi.conv.out_channels}$\\rightarrow${C} "
     f"at each voxel"),
    ("prior term of $J$", "kept apart",
     f"same per-channel weighting, with $w_{{reg,c}}^2$"),
    ("ConvLSTM solver", "MIXED",
     f"channels and frames are one flat axis of "
     f"{S.grad_net.lstm.gates.in_channels - S.grad_net.lstm.hidden_ch}"),
    ("variance head", "kept apart at the output",
     f"reads {3 * C} mixed channels, emits {C} separate $\\sigma^2$, each floored at its "
     f"own sensor variance"),
    ("reported metric", "kept apart, then averaged",
     f"per-channel MSE, then the mean over the {C}"),
]

json.dump(dict(channels=CHAN, day=os.path.basename(_files[0]), window=_wi,
               dT=DT, scale=scale, cost=cost, conv=conv, chain=chain, j_example=j_ex, mixing=mixing),
          open(OUT, "w"), indent=1)
print(f"[json] {OUT}")
for _k in ("at_x0", "at_xhat"):
    _d = cost[_k]
    print(f"\n  J({_k[3:]}) = {_d['J']:.4f}   obs {_d['obs_total']:.4f}  "
          f"reg {_d['reg_total']:.4f}")
    print("     share: " + "  ".join(
        f"{c} {_d['per_channel'][c]['share'] * 100:.1f}%" for c in CHAN))
print(f"  psi_{o}({t},{h},{ww}) = {psi_val:+.4f}   partials "
      + " ".join(f"{c}:{p:+.4f}" for c, p in zip(CHAN, partial)))
print(f"  dropped by the mask: " + " ".join(f"{c}:{d:+.4f}" for c, d in zip(CHAN, dropped)))
