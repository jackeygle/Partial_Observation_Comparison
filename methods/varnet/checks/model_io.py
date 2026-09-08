"""
model_io.py  —  one place that rebuilds a trained solver from a checkpoint
==========================================================================

Every eval / figure / benchmark script needs the same three lines: read the checkpoint,
reconstruct GENN + GradSolver with the architecture that was actually trained, load the
weights. Duplicating that in a dozen scripts went wrong twice:

  * `--iter-schedule` (paper §3.4's 5->20 curriculum) means the solver finishes training at
    a DIFFERENT n_iter than the `--n-iter` recorded in argparse. Reading args["n_iter"]
    silently evaluates a 20-iteration model with 15 iterations. The checkpoint therefore
    records `n_iter_eff`, and that is what is used here.
  * Architecture flags (two-scale, Eq.10 upsampling) are detected from the STATE DICT, not from
    args, so a checkpoint trained before a flag existed still loads correctly.

`strict=True` by default on purpose: with strict=False a genuine architecture mismatch
loads silently and produces a plausible-looking but meaningless model.
"""
from __future__ import annotations

import torch

from crowdcore import config
from crowdcore import paths
from methods.varnet.prior_model import GENN
from methods.varnet.variational_solver import GradSolver


def _HAS_VAR(sd):
    """Does this checkpoint carry a variance read-out? Decided by the weights, never by args."""
    return any("grad_net.out_var" in k for k in sd)


def _VAR_SEES_STATE(sd, a):
    """Does the variance read-out take [h, x_hat] or h alone?

    Read off the stored weight's input width, not off a flag: runs/varnet_vrb{0,1}_s0 predate
    var_sees_state and their out_var is (C*T, hidden_ch, 1, 1), while later runs are
    (C*T, hidden_ch + C*T, 1, 1). Constructing the wrong one is a shape error on load_state_dict,
    which would silently cost us those baselines.
    """
    w = sd.get("grad_net.out_var.weight")
    if w is None:
        return False
    return int(w.shape[1]) > int(a["lstm_hidden"])


def _AUGMENTED(sd, a):
    """Was this trained with log sigma^2 as part of the iterated state?

    Read off the LSTM's input width, not off a stored flag: the augmented solver feeds
    [x, log sigma^2] to the optimiser, so gates.weight has 2*C*dT + hidden input channels
    against C*dT + hidden for every earlier run. Building the wrong one is a shape error on
    load_state_dict, which would cost us those checkpoints.
    """
    w = sd.get("grad_net.lstm.gates.weight")
    if w is None:
        return False
    return int(w.shape[1]) > 4 * int(a["dT"]) + int(a["lstm_hidden"])


def load_solver(ckpt_path, device="cpu", strict=True, n_iter=None):
    """Rebuild the trained solver. Returns (solver, args, ckpt).

    n_iter : override the iteration count (e.g. to time the cost of a shorter solve).
             Default = the count the model actually finished training with.
    """
    ck = torch.load(paths.require_ckpt(ckpt_path, "4DVarNet checkpoint"), map_location="cpu")
    a, sd = ck["args"], ck["solver"]
    P = config.CFG["prior"]
    g = lambda k: a.get(k, P[k])                      # fall back to config for older ckpts

    phi = GENN(n_channels=4, hidden=a["hidden"], kt=g("kt"), kh=g("kh"), kw=g("kw"),
               n_phi_layers=g("n_phi_layers"),
               # architecture read off the weights, so it always matches what is in the file
               two_scale=any("branch_coarse" in k for k in sd),
               scale=P["scale"])
    n_it = n_iter if n_iter is not None else ck.get("n_iter_eff", a["n_iter"])
    # whether this run has the variance read-out is decided by the weights, never by args:
    # pre-NLL checkpoints simply have no out_var and must keep loading unchanged.
    solver = GradSolver(phi, n_channels=4, dT=a["dT"], n_iter=n_it,
                        hidden_ch=a["lstm_hidden"],
                        dropout=a.get("dropout", 0.0),
                        predict_var=_HAS_VAR(sd),
                        var_eps=a.get("var_eps", 1e-6),
                        var_sees_state=_VAR_SEES_STATE(sd, a),
                        augmented_var=_AUGMENTED(sd, a)).to(device)
    solver.load_state_dict(sd, strict=strict)
    solver.eval()
    return solver, a, ck


