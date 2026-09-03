import sys, torch
sys.path.insert(0, "/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
sys.path.insert(0, "/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf/checks")
from variational_solver import GradSolver, GradUpdateLSTM
from prior_model import GENN
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C, T, H, W = 4, 8, 36, 12
for sees in (True, False):
    phi = GENN(n_channels=C).to(d)
    S = GradSolver(phi, n_channels=C, dT=T, n_iter=3, hidden_ch=64,
                   predict_var=True, var_sees_state=sees).to(d)
    x0 = torch.randn(1, C, T, H, W, device=d) * 0.1
    y = torch.randn(1, C, T, H, W, device=d) * 0.1
    m = (torch.rand(1, C, T, H, W, device=d) > 0.5).float()
    xr, v = S(x0, y, m, return_var=True)
    n = S.grad_net.out_var.weight.numel()
    print(f"sees_state={sees}  out_var.weight {tuple(S.grad_net.out_var.weight.shape)} "
          f"= {n:,} params   sigma^2 mean {v.mean().item():.5f}   x_hat finite {torch.isfinite(xr).all().item()}")
    # gradient must NOT reach x through the var branch
    v.sum().backward()
    print(f"   grad on out.weight (the mean read-out) = {S.grad_net.out.weight.grad.abs().sum().item():.3e}  (must be 0)")
