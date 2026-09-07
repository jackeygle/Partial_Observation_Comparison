# Run from the repo root: source sbatch/_env.sh && python3 -m methods.varnet.scratch.smoke_aug
import sys, os, torch
from methods.varnet.variational_solver import GradSolver
from methods.varnet.prior_model import GENN
d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
C,T,H,W=4,8,36,12
x0=torch.randn(1,C,T,H,W,device=d)*0.1
y =torch.randn(1,C,T,H,W,device=d)*0.1
m =(torch.rand(1,C,T,H,W,device=d)>0.5).float()
for aug in (False,True):
    torch.manual_seed(0)
    S=GradSolver(GENN(n_channels=C).to(d), n_channels=C, dT=T, n_iter=4, hidden_ch=64,
                 predict_var=True, augmented_var=aug).to(d)
    n=sum(p.numel() for p in S.parameters())
    with torch.enable_grad():
        xr,v = S(x0.clone(), y, m, return_var=True)
    print(f"augmented={aug!s:5s} params={n:>9,}  gates_in={S.grad_net.lstm.gates.weight.shape[1]:>4d}"
          f"  sigma^2 mean={v.mean().item():.5f}  min={v.min().item():.2e}  max={v.max().item():.3f}"
          f"  x finite={torch.isfinite(xr).all().item()}")
    # gradient must reach the cost weights through the variance path when augmented
    S.zero_grad(); (v.mean()+xr.mean()).backward()
    g=S.var_cost.alpha_reg.grad
    print(f"              grad on var_cost.alpha_reg = {0.0 if g is None else float(g.abs()):.3e}")
