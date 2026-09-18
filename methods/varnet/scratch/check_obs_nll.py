# Run from the repo root: source sbatch/_env.sh && python3 -m methods.varnet.scratch.check_obs_nll
import os, tempfile, torch
from methods.varnet.variational_solver import GradSolver, VarCost
from methods.varnet.prior_model import GENN
from methods.varnet.checks import model_io

torch.manual_seed(0)
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.float64
B, C, T, H, W = 2, 4, 6, 36, 12
dx = torch.randn(B, C, T, H, W, dtype=dt, device=d) * 0.3
dyr = torch.randn(B, C, T, H, W, dtype=dt, device=d) * 0.1
m = (torch.rand(B, C, T, H, W, device=d) > 0.6).to(dt)
dy = dyr * m
s = (torch.randn(B, C, T, H, W, dtype=dt, device=d) * 0.5 - 2).requires_grad_(True)

def cost(obs_nll):
    vc = VarCost(C, obs_nll=obs_nll).to(d).to(dt)
    with torch.no_grad():
        vc.alpha_obs.fill_(1.3); vc.alpha_reg.fill_(0.7)
        vc.w_obs.copy_(torch.tensor([1.0, 0.5, 2.0, 1.5])); vc.w_reg.copy_(torch.tensor([0.8, 1.1, 0.9, 1.2]))
    return vc

ok = True
# 1. no observed cells -> identical to aug0 (value and both gradients)
z = torch.zeros_like(m)
xg = dx.clone().requires_grad_(True)
J0 = cost(False)(xg, dy * 0, s, z); g0 = torch.autograd.grad(J0, (xg, s))
J1 = cost(True)(xg, dy * 0, s, z);  g1 = torch.autograd.grad(J1, (xg, s))
r = abs(J0 - J1).item() + max((a - b).abs().max().item() for a, b in zip(g0, g1))
print(f"[1] Omega=0 equals aug0: max diff {r:.2e}"); ok &= r < 1e-12

# 2. value matches the hand-written formula
vc = cost(True)
N = dx.numel() / C
w_o = vc.w_obs.view(1, C, 1, 1, 1) ** 2; w_r = vc.w_reg.view(1, C, 1, 1, 1) ** 2
ref = (vc.alpha_obs ** 2 * (w_o * (dy ** 2 * torch.exp(-s) + s) * m).sum()
       + vc.alpha_reg ** 2 * (w_r * (dx ** 2 * torch.exp(-s) + s)).sum()) / N
r = abs(vc(dx, dy, s, m) - ref).item()
print(f"[2] formula: diff {r:.2e}"); ok &= r < 1e-10

# 3. dJ/ds = 0 at sigma^2 = (a_r W_r dx^2 + a_o W_o dy^2 Omega) / (a_r W_r + a_o W_o Omega)
a_o = vc.alpha_obs ** 2 * w_o * m; a_r = vc.alpha_reg ** 2 * w_r
s_star = torch.log((a_r * dx ** 2 + a_o * dy ** 2) / (a_r + a_o)).detach().requires_grad_(True)
gs = torch.autograd.grad(vc(dx, dy, s_star, m), s_star)[0]
inside = s_star.detach() > -13.8          # the clamp zeroes the obs-term gradient below the floor
r = gs[inside].abs().max().item() * N
print(f"[3] stationary point: max |dJ/ds|*N {r:.2e} on {inside.float().mean().item():.1%} of points")
ok &= r < 1e-8

# 4. full solve on GPU: finite, observed-cell sigma differs from aug0, gradient flows
x0 = torch.randn(1, C, T, H, W, device=d) * 0.1
y = torch.randn(1, C, T, H, W, device=d) * 0.1
mk = (torch.rand(1, C, T, H, W, device=d) > 0.5).float()
out = {}
for on in (False, True):
    torch.manual_seed(1)
    S = GradSolver(GENN(n_channels=C).to(d), n_channels=C, dT=T, n_iter=5, hidden_ch=64,
                   predict_var=True, augmented_var=True, obs_nll=on).to(d)
    with torch.no_grad():                      # move off the zero-ish init so the update is not trivial
        S.grad_net.out.weight.normal_(0, 0.1)
    xr, v = S(x0.clone(), y, mk, return_var=True)
    S.zero_grad(); (v.mean() + xr.mean()).backward()
    ga = S.var_cost.alpha_obs.grad
    out[on] = (v.detach(), torch.isfinite(xr).all().item() and torch.isfinite(v).all().item())
    print(f"[4] obs_nll={on!s:5s} finite={out[on][1]}  sigma^2 obs {v[mk > 0].mean().item():.4f}  "
          f"blind {v[mk == 0].mean().item():.4f}  |grad alpha_obs| {0.0 if ga is None else ga.abs().item():.2e}")
    ok &= out[on][1]
ok &= not torch.equal(out[False][0], out[True][0])

# 5. model_io rebuilds obs_nll from args (it is not visible in the weights)
S = GradSolver(GENN(n_channels=C, hidden=32, kt=3), n_channels=C, dT=T, n_iter=5, hidden_ch=64,
               predict_var=True, augmented_var=True, obs_nll=True)
from crowdcore import config
P = config.CFG["prior"]
args = dict(hidden=32, kt=3, kh=3, kw=3, n_phi_layers=P["n_phi_layers"], dT=T, n_iter=5,
            lstm_hidden=64, var_eps=1e-6, augmented_var=True, obs_nll=True)
with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as td:
    p = os.path.join(td, "ck.pt")
    torch.save({"args": args, "solver": S.state_dict()}, p)
    S2, _, _ = model_io.load_solver(p)
    args.pop("obs_nll"); torch.save({"args": args, "solver": S.state_dict()}, p)
    S3, _, _ = model_io.load_solver(p)
print(f"[5] model_io: obs_nll ckpt -> {S2.var_cost.obs_nll}, legacy aug ckpt -> {S3.var_cost.obs_nll}")
ok &= S2.var_cost.obs_nll and not S3.var_cost.obs_nll and S2.augmented_var

print("ALL_OK" if ok else "FAILED")
