# 从仓库根跑: source sbatch/_env.sh && python3 -m methods.varnet.scratch.smoke_grad_path
import sys, os, torch
from methods.varnet.variational_solver import GradSolver
from methods.varnet.prior_model import GENN
d = torch.device("cuda"); C,T,H,W = 4,8,36,12
torch.manual_seed(0)
S = GradSolver(GENN(n_channels=C).to(d), n_channels=C, dT=T, n_iter=4, hidden_ch=64,
               predict_var=True, var_sees_state=True).to(d)
x0 = torch.randn(1,C,T,H,W,device=d)*0.1
y  = torch.randn(1,C,T,H,W,device=d)*0.1
m  = (torch.rand(1,C,T,H,W,device=d) > 0.5).float()
# out_var.weight is zero-initialised, which makes d(sigma^2)/d(h) identically zero and would
# make this whole test read "nothing upstream is touched" no matter what the wiring is.
with torch.no_grad():
    S.grad_net.out_var.weight.normal_(0, 0.01)
xr, v = S(x0, y, m, return_var=True)
S.zero_grad(); v.sum().backward()                 # gradient ONLY from the variance branch
print("从 sigma^2 分支回传,各处梯度:")
for n in ("grad_net.out.weight", "grad_net.out_var.weight",
          "grad_net.lstm.gates.weight", "phi.branch_fine.psi.conv.weight"):
    g = dict(S.named_parameters()).get(n)
    if g is None:
        cand=[k for k,_ in S.named_parameters() if "psi" in k or "gates" in k]; print("  ?",n,cand[:3]); continue
    val = 0.0 if g.grad is None else g.grad.abs().sum().item()
    print(f"  {n:32s} {val:.4e}   {'不受影响' if val==0 else '← 受影响'}")
