# Run from the repo root: source sbatch/_env.sh && python3 -m methods.varnet.scratch.smoke_compat
import sys, os, torch
from methods.varnet.checks.model_io import load_solver
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for ck in ["runs/varnet_vrb0_s0/varnet_best.pt", "runs/varnet_vrb1_s0/varnet_best.pt",
           "runs/varnet_b0/varnet_best.pt", "runs/varnet_ml5_s0/varnet_best.pt"]:
    p = os.path.join(R, ck)
    if not os.path.exists(p): print(f"  -- absent: {ck}"); continue
    try:
        S, a, _ = load_solver(p, d)
        ov = S.grad_net.out_var
        print(f"OK  {ck:42s} out_var={'None' if ov is None else tuple(ov.weight.shape)}"
              f"  sees_state={S.grad_net.var_sees_state}")
    except Exception as e:
        print(f"FAIL {ck}: {type(e).__name__}: {str(e)[:130]}")
