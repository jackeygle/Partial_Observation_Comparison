# Peak training VRAM vs window length dT, one optimiser step on random tensors.
# Run from the repo root: source sbatch/_env.sh && python3 -m methods.varnet.scratch.prof_vram_dT
import json, os, torch
from crowdcore import config
from methods.varnet.prior_model import GENN
from methods.varnet.variational_solver import GradSolver
from methods.varnet.losses import compute_loss

d = torch.device("cuda")
P = config.CFG["prior"]
C, H, W, N_ITER = 4, 36, 12, 20
ARMS = {  # name: (loss, augmented_var, obs_nll)
    "mse5": ("supervised", False, False),
    "augobs": ("nll", True, True),
}
DTS = [30, 50, 100, 200]
BATCHES = [1, 4, 8, 16, 32]
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"[card] {torch.cuda.get_device_name(0)}  {total:.1f} GiB", flush=True)
res = []
for arm, (loss_kind, aug, obs) in ARMS.items():
    for dT in DTS:
        for B in BATCHES:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            rec = {"arm": arm, "dT": dT, "batch": B}
            try:
                phi = GENN(n_channels=C, hidden=96, kt=5, kh=P["kh"], kw=P["kw"],
                           n_phi_layers=P["n_phi_layers"], two_scale=P["two_scale"], scale=P["scale"])
                S = GradSolver(phi, n_channels=C, dT=dT, n_iter=N_ITER,
                               hidden_ch=config.get("solver", "lstm_hidden"),
                               predict_var=loss_kind == "nll", var_eps=1e-6,
                               augmented_var=aug, obs_nll=obs).to(d)
                opt = torch.optim.Adam(S.parameters(), lr=1e-3)
                base = torch.cuda.memory_allocated() / 2**30
                X = torch.randn(B, C, dT, H, W, device=d) * 0.2
                Y = X + torch.randn_like(X) * 0.05
                M = (torch.rand_like(X) > 0.5).float()
                X0 = Y * M
                for _ in range(2):  # second step includes Adam state
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if loss_kind == "nll":
                            xr, var = S(X0.clone(), Y * M, M, return_var=True)
                        else:
                            xr, var = S(X0.clone(), Y * M, M), None
                        loss = compute_loss(loss_kind, xr.float(), X, Y, M, phi,
                                            var=None if var is None else var.float(), nll_beta=0.0)
                    loss.backward(); opt.step()
                torch.cuda.synchronize()
                rec["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                rec["static_gib"] = base
                rec["ok"] = True
            except torch.OutOfMemoryError:
                rec["ok"] = False
            print(json.dumps(rec), flush=True)
            res.append(rec)
            for v in ("S", "opt", "X", "Y", "M", "X0", "xr", "var", "loss", "phi"):
                globals().pop(v, None)
            torch.cuda.empty_cache()
            if not rec["ok"]:
                break  # larger batches at this dT will also OOM
out = os.path.join(os.path.dirname(__file__), "prof_vram_dT.json")
json.dump({"card": torch.cuda.get_device_name(0), "total_gib": total, "n_iter": N_ITER,
           "results": res}, open(out, "w"), indent=1)
print("[done]", out)
