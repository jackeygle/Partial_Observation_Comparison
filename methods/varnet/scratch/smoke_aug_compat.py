# 从仓库根跑: source sbatch/_env.sh && python3 -m methods.varnet.scratch.smoke_aug_compat
import sys, os, torch
from methods.varnet.checks.model_io import load_solver
d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
for ck in ("runs/varnet_vsb0_s0","runs/varnet_vrb0_s0","runs/varnet_ml5_s0","runs/varnet_b0_k1"):
    p=os.path.join(R,ck,"varnet_best.pt")
    if not os.path.exists(p): print("  absent:",ck); continue
    S,a,_=load_solver(p,d)
    print(f"OK {ck:24s} gates_in={S.grad_net.lstm.gates.weight.shape[1]:>5d}  "
          f"augmented={S.augmented_var}  out_var={'None' if S.grad_net.out_var is None else 'yes'}")
