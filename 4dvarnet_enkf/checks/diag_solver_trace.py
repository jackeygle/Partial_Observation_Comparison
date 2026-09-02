"""
diag_solver_trace.py  —  what the solver's 20 unrolled steps actually do

The deck asserts "20 steps of gradient descent on J" and "the LSTM state is carried across
steps". Neither has ever been measured. This runs one window and records, per step:

    J and its two terms · the update size · how far x moved · the four gate means
    · the blind-zone and full-state MSE against the truth

and, for one voxel and one hidden channel, the gate arithmetic in full, so the ConvLSTM step
can be shown on real numbers the way psi and phi already are.

The curves are averaged over a batch of windows strided across the day, NOT one window. A first
version used the single busiest window and appeared to show the reconstruction degrading after
about step 9; the test-set sweep (sbatch/submit_niter_sweep.sbatch) showed the opposite —
error falls monotonically all the way to 20 and beyond — so the single window was not
representative and the batch is what the figure reports.

Everything is read out of the REAL forward pass with hooks, not from a re-implementation. A
hand-written copy of solve() was tried first and rejected: it agreed to 1e-5 at step 1 but drifted
to 0.9 by step 20, because F.conv2d and nn.Conv2d pick different cudnn kernels and 20 unrolled
iterations amplify the rounding. Hooks leave exactly one code path, so the numbers here are by
construction the ones the model computes.

  var_cost      -> J per step
  lstm.gates    -> the pre-activations, from which i, f, o, g follow by sigmoid/tanh
  lstm          -> h and c per step
  grad_net.out  -> the update, from which x at every step is rebuilt and checked against the
                   solver's own return value

What the curves are for: J itself is NOT what the solver descends. The solver is trained
end-to-end on the reconstruction loss, so J is free to rise while the reconstruction keeps
improving, and measuring both is the only way to say so honestly. The gate means say whether
the LSTM state carries anything at all.

Needs a GPU (the solve is 20 unrolled iterations over a 200-frame window).

    sbatch sbatch/submit_solver_trace.sbatch
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "checks"))
import config                                                        # noqa: E402
import observation_model as om                                       # noqa: E402
import navigation as nav                                             # noqa: E402
from model_io import load_solver                                     # noqa: E402

OUT = os.path.join(ROOT, "check_outputs", "eval", "solver_trace.json")
CHAN = list(config.get("grid", "channels"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S, A, _ = load_solver(os.path.join(ROOT, "runs/varnet_b0_k1/varnet_best.pt"), DEV)
S.eval()
DT, NIT = A["dT"], S.n_iter
print(f"[device] {DEV}" + (f"  {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

# ── the same window the other diagnostics use, so the numbers line up across slides ─────
_files = om.split_files("test")
X_day, _ = om.load_state(_files[0])
valid = nav.build_valid_mask_from_config(X_day)
X_day = X_day[:(X_day.shape[0] // DT) * DT]
out = om.generate_observations(X_day, A["sensing_range"], A["num_agents"], add_noise=True,
                               seed=A.get("data_seed", 0) or 0, valid_mask=valid,
                               obs_every_k=1)
X0_day = om.fill_missing_state(out["Y"], out["Omega_c"],
                              method=config.get("observation", "init_method"))
w = lambda a: torch.from_numpy(om.to_windows(a, DT)).float()
_XW, _YW, _MW, _X0W = w(X_day), w(out["Y"]), w(out["Omega_c"].astype(np.float32)), w(X0_day)
NW = int(os.environ.get("TRACE_WINDOWS", 16))
IDX = np.linspace(0, _XW.shape[0] - 1, NW).round().astype(int)      # strided across the day
X = _XW[IDX].to(DEV)
Y = _YW[IDX].to(DEV)
M = _MW[IDX].to(DEV)
X0 = _X0W[IDX].to(DEV)
WI = IDX.tolist()
UNOBS = M < 0.5
print(f"[data] {os.path.basename(_files[0])}  {len(IDX)} windows strided across the day  {tuple(X.shape)}")

# the voxel and hidden channel the single-step walk uses; the voxel matches the psi/phi slides
TV, HV, WV = 41, 10, 6
PC = 0                                          # physical channel for the read-out row
OROW = PC * DT + TV                             # out.weight index: the 800 axis is c*T + t
B, C, T, H, Wd = X.shape

lstm, outc = S.grad_net.lstm, S.grad_net.out
GW, GB = lstm.gates.weight, lstm.gates.bias
HID = lstm.hidden_ch


def cost_terms(x):
    """J split into its observation and prior halves, exactly as VarCost adds them."""
    with torch.no_grad():
        dy = (x - Y) * M
        dx = x - S.phi(x)
        vc = S.var_cost
        a_o, a_r = vc.alpha_obs ** 2, vc.alpha_reg ** 2
        t_o = float(a_o * vc._weighted_l2(dy, vc.w_obs))
        t_r = float(a_r * vc._weighted_l2(dx, vc.w_reg))
    return t_o, t_r


# ── run the real solver once, with hooks on the four places worth watching ──────────────
CAP = dict(J=[], gates=[], h=[], c=[], upd=[])


def _h_cost(_m, _i, out):
    CAP["J"].append(float(out.detach()))


def _h_gates(_m, _i, out):
    with torch.no_grad():
        if len(CAP["gates"]) == NIT // 2:            # the step the walk uses
            CAP["gate_in"] = _i[0].detach()[0].cpu() # (in_ch, H, W) = concat([ghat, h_prev])
            CAP["gate_pre"] = out.detach()[0].cpu()  # (4*hidden, H, W) pre-activation
        i_, f_, o_, gt = out.detach().chunk(4, 1)
        i_, f_, o_, gt = i_.sigmoid(), f_.sigmoid(), o_.sigmoid(), gt.tanh()
        _rail = lambda v: float(((v < 0.05) | (v > 0.95)).float().mean())
        CAP["gates"].append(dict(
            i=float(i_.mean()), f=float(f_.mean()), o=float(o_.mean()),
            g_absmean=float(gt.abs().mean()),
            rail_i=_rail(i_), rail_f=_rail(f_), rail_o=_rail(o_),
            rail_g=float((gt.abs() > 0.95).float().mean()),
            at=[float(i_[0, :, HV, WV].mean()), float(f_[0, :, HV, WV].mean())],
            vox=dict(i=i_[0, :, HV, WV].cpu(), f=f_[0, :, HV, WV].cpu(),
                     o=o_[0, :, HV, WV].cpu(), g=gt[0, :, HV, WV].cpu())))


def _h_lstm(_m, _i, out):
    with torch.no_grad():
        _hh, (_, _cc) = out
        CAP["h"].append(_hh.detach()[0, :, HV, WV].cpu())
        CAP["c"].append(_cc.detach()[0, :, HV, WV].cpu())
        CAP.setdefault("h_absmean", []).append(float(_hh.detach().abs().mean()))
        CAP.setdefault("c_absmean", []).append(float(_cc.detach().abs().mean()))


def _h_out(_m, _i, out):
    CAP["upd"].append(out.detach().clone())


_hs = [S.var_cost.register_forward_hook(_h_cost),
       lstm.gates.register_forward_hook(_h_gates),
       lstm.register_forward_hook(_h_lstm),
       outc.register_forward_hook(_h_out)]
with torch.enable_grad():
    xhat = S(X0.clone(), Y, M).detach()
for _hk in _hs:
    _hk.remove()
assert len(CAP["J"]) == NIT, f"expected {NIT} cost evaluations, captured {len(CAP['J'])}"

# rebuild x at every step from the captured updates, and check the rebuild is exact
xs = [X0]
for _u in CAP["upd"]:
    xs.append(xs[-1] - _u.reshape(B, C, T, H, Wd) / NIT)
_err = float((xs[-1] - xhat).abs().max())
assert _err < 1e-4, f"rebuilt x differs from the solver's return: max |diff| {_err}"
print(f"[check] x rebuilt from the captured updates vs solver output: {_err:.2e}")

normg = None                                    # the RMS the solver froze at step 0
with torch.enable_grad():
    _xg = X0.clone().requires_grad_(True)
    _J = S.var_cost(_xg - S.phi(_xg), (_xg - Y) * M)
    _g = torch.autograd.grad(_J, _xg)[0]
    normg = float(torch.sqrt((_g ** 2).mean() + 1e-12))

steps = []
for k in range(NIT + 1):
    t_o, t_r = cost_terms(xs[k])
    se = (xs[k] - X) ** 2
    row = dict(k=k, J=t_o + t_r, obs=t_o, reg=t_r,
               blind_mse=float(se[UNOBS].mean()), full_mse=float(se.mean()))
    if k < NIT:
        _u = CAP["upd"][k].reshape(B, C, T, H, Wd) / NIT
        g = CAP["gates"][k]
        row.update(J_captured=CAP["J"][k], upd_norm=float(_u.norm()),
                   upd_max=float(_u.abs().max()), i=g["i"], f=g["f"], o=g["o"],
                   g_absmean=g["g_absmean"], h_absmean=CAP["h_absmean"][k],
                   c_absmean=CAP["c_absmean"][k], rail_i=g["rail_i"], rail_f=g["rail_f"],
                   rail_o=g["rail_o"], rail_g=g["rail_g"])
        # the hook's J and the recomputed one must agree, or the rebuild is wrong somewhere
        assert abs(row["J"] - row["J_captured"]) < 2e-4, \
            f"step {k}: recomputed J {row['J']} vs captured {row['J_captured']}"
    steps.append(row)
print(f"[check] recomputed J matches the hooked J at all {NIT} steps")

# ── the single-step walk at one voxel: the gate arithmetic, then the read-out ────────────
# Step 0 is useless for this: the state starts at zero, so f*c_prev vanishes and the forget
# gate does nothing visible. Use a middle step, and pick a hidden channel whose gates are NOT
# saturated — the loudest channel is always one that has railed to 0 or 1, where the arithmetic
# is trivially f*0 + 1*(-1).
_k0 = NIT // 2
_hv = CAP["h"][_k0]
_cp = CAP["c"][_k0 - 1]                          # the cell state this step inherits
_g0 = CAP["gates"][_k0]["vox"]
_wout = outc.weight[OROW, :, 0, 0].detach().cpu()
_contrib = (_wout * _hv).abs()
_sat = ((_g0["i"] < 0.05) | (_g0["i"] > 0.95) | (_g0["f"] < 0.05) | (_g0["f"] > 0.95)
        | (_g0["o"] < 0.05) | (_g0["o"] > 0.95) | (_g0["g"].abs() > 0.95))
_n_sat = int(_sat.sum())
# the loudest contributor to the read-out, saturated or not. At this voxel every channel has
# at least one railed gate, so demanding an unsaturated one is not possible — the gates of the
# trained solver behave like switches, which is worth saying rather than working around.
_kk = int(torch.argmax(_contrib))
_u2d = float(CAP["upd"][_k0][0, OROW, HV, WV])
walk = dict(step=_k0, voxel=[TV, HV, WV], hidden=_kk, out_row=OROW, phys_channel=CHAN[PC],
            n_saturated=_n_sat, n_hidden=HID,
            c_prev=float(_cp[_kk]),
            i=float(_g0["i"][_kk]), f=float(_g0["f"][_kk]), o=float(_g0["o"][_kk]),
            g=float(_g0["g"][_kk]), c_new=float(CAP["c"][_k0][_kk]), h=float(_hv[_kk]),
            w_out_k=float(_wout[_kk]), contrib_k=float((_wout * _hv)[_kk]),
            h_vec=_hv.tolist(), w_out=_wout.tolist(), u_2d=_u2d,
            u_scaled=_u2d / NIT,
            x_before=float(xs[_k0][0, PC, TV, HV, WV]),
            x_after=float(xs[_k0 + 1][0, PC, TV, HV, WV]))
# the cell arithmetic, and the read-out sum, both checked
_cc = walk["f"] * walk["c_prev"] + walk["i"] * walk["g"]
assert abs(_cc - walk["c_new"]) < 1e-4, f"f*c + i*g = {_cc} != c_new {walk['c_new']}"
assert abs(walk["o"] * np.tanh(walk["c_new"]) - walk["h"]) < 1e-4, "o*tanh(c) != h"
_ws = float(np.dot(np.array(walk["w_out"]), np.array(walk["h_vec"])))
assert abs(_ws - walk["u_2d"]) < 1e-3, f"W_out . h = {_ws} != u {walk['u_2d']}"
print(f"[check] step {_k0}, hidden {_kk} (of {HID}; {_n_sat} have a railed gate): "
      f"f*c+i*g and o*tanh(c) both reproduce the module; "
      f"W_out.h = {_ws:+.5f} == u {walk['u_2d']:+.5f}")

# ── the gate CONVOLUTION at one output value, decomposed ────────────────────────────────
# This is the model's big convolution: 864 input channels x 3x3 = 7,776 terms per output
# value, and 97.5% of the solver's parameters. The forget gate of the hidden channel the walk
# uses is chosen, so sigmoid(total) must equal the f already printed above.
_GQ = 1 * HID + _kk                                  # chunk order is i, f, o, g
_gin = CAP["gate_in"]
_gpre = CAP["gate_pre"]
_wq = lstm.gates.weight[_GQ].detach().cpu()          # (864, 3, 3)
_patch = _gin[:, HV - 1:HV + 2, WV - 1:WV + 2]       # (864, 3, 3)
_prod = _wq * _patch
_bq = float(lstm.gates.bias[_GQ].detach())
_tot = float(_prod.sum()) + _bq
assert abs(_tot - float(_gpre[_GQ, HV, WV])) < 1e-3, \
    f"hand sum {_tot} != captured pre-activation {float(_gpre[_GQ, HV, WV])}"
assert abs(float(torch.sigmoid(torch.tensor(_tot))) - walk["f"]) < 1e-4, \
    "sigmoid(pre-activation) does not match the forget gate used in the walk"

_gp = _prod[:C * DT].reshape(C, DT, 3, 3)            # the gradient half, unfolded
_hp = _prod[C * DT:]                                 # the hidden-state half
gconv = dict(step=NIT // 2, q=_GQ, gate="f", hidden=_kk, voxel=[TV, HV, WV],
             in_ch=int(_gin.shape[0]), taps=9,
             n_terms=int(_gin.shape[0]) * 9,
             bias=_bq, total=_tot, sigmoid=float(torch.sigmoid(torch.tensor(_tot))),
             from_grad=float(_gp.sum()), from_hidden=float(_hp.sum()),
             per_channel=_gp.sum(dim=(1, 2, 3)).tolist(),
             frame_t=float(_gp[:, TV].sum()),
             other_frames=float(_gp.sum() - _gp[:, TV].sum()),
             per_tap=_prod.sum(dim=0).tolist(),
             n_grad_terms=C * DT * 9, n_hidden_terms=HID * 9,
             absmass_grad=float(_prod[:C * DT].abs().sum()),
             absmass_hidden=float(_hp.abs().sum()))
print(f"[gconv] gate f, hidden {_kk}, {gconv['n_terms']:,} terms: "
      f"grad {gconv['from_grad']:+.4f} + hidden {gconv['from_hidden']:+.4f} "
      f"+ bias {_bq:+.4f} = {_tot:+.4f}  ->  sigmoid {gconv['sigmoid']:.6f}")
print(f"        frame t={TV} alone {gconv['frame_t']:+.4f}  vs the other {DT - 1} frames "
      f"{gconv['other_frames']:+.4f}")

# the gate trace at that one voxel, averaged over the 64 hidden channels, for the figure
gate_trace = [dict(k=_k, i=float(CAP["gates"][_k]["vox"]["i"].mean()),
                   f=float(CAP["gates"][_k]["vox"]["f"].mean()),
                   o=float(CAP["gates"][_k]["vox"]["o"].mean()),
                   c=float(CAP["c"][_k].abs().mean()))
              for _k in range(NIT)]

json.dump(dict(day=os.path.basename(_files[0]), windows=WI, n_windows=len(IDX), n_iter=NIT, dT=DT,
               channels=CHAN, hidden=HID, normg=float(normg),
               steps=steps, walk=walk, gconv=gconv, gate_trace=gate_trace,
               params=dict(gates=GW.numel() + GB.numel(), out=outc.weight.numel(),
                           total=sum(p.numel() for p in S.grad_net.parameters()))),
          open(OUT, "w"), indent=1)
print(f"[json] {OUT}")

_f = [s["f"] for s in steps if "f" in s]
_J = [s["J"] for s in steps]
print(f"\n  J: {_J[0]:.4f} -> {_J[-1]:.4f}   ({(1 - _J[-1] / _J[0]) * 100:.1f}% down)")
print(f"  blind MSE: {steps[0]['blind_mse']:.5f} -> {steps[-1]['blind_mse']:.5f}")
print(f"  forget gate mean: first {_f[0]:.3f}  last {_f[-1]:.3f}  "
      f"range [{min(_f):.3f}, {max(_f):.3f}]")
_r = [s2["rail_f"] for s2 in steps if "rail_f" in s2]
print(f"  fraction of gate values railed (<0.05 or >0.95): "
      f"i {np.mean([s2['rail_i'] for s2 in steps if 'rail_i' in s2]):.1%}  "
      f"f {np.mean(_r):.1%}  "
      f"o {np.mean([s2['rail_o'] for s2 in steps if 'rail_o' in s2]):.1%}  "
      f"|g|>0.95 {np.mean([s2['rail_g'] for s2 in steps if 'rail_g' in s2]):.1%}")
print(f"  update norm: first {steps[0]['upd_norm']:.4f}  "
      f"last {steps[NIT - 1]['upd_norm']:.4f}")
_Jmin = min(range(NIT + 1), key=lambda q: steps[q]["J"])
_Bmin = min(range(NIT + 1), key=lambda q: steps[q]["blind_mse"])
print(f"  J is lowest at step {_Jmin} ({steps[_Jmin]['J']:.4f}), "
      f"blind MSE lowest at step {_Bmin} ({steps[_Bmin]['blind_mse']:.5f})")
print(f"  grad RMS frozen at step 0 and reused for all {NIT} steps: {normg:.6f}")
