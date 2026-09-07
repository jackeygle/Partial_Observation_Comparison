# Run from the repo root: source sbatch/_env.sh && python3 -m methods.varnet.scratch.smoke_postdel
import sys, os, torch
from methods.varnet.checks.model_io import load_solver
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for ck in ["runs/varnet_vsb0_s0/varnet_best.pt","runs/varnet_vrb0_s0/varnet_best.pt",
           "runs/varnet_ml5_s0/varnet_best.pt"]:
    S,a,_ = load_solver(os.path.join(R,ck), d)
    ov = S.grad_net.out_var
    print(f"OK {ck:40s} out_var={'None' if ov is None else tuple(ov.weight.shape)} "
          f"dropout={S.grad_net.dropout.p}")
