# dT=200 training step on the card this runs on: peak VRAM and s/step per arm x precision x batch.
# Run from the repo root: source sbatch/_env.sh && python3 -m methods.varnet.scratch.prof_v100_batch
import json, os, time, torch
from contextlib import nullcontext
from crowdcore import config
from methods.varnet.prior_model import GENN
from methods.varnet.variational_solver import GradSolver
from methods.varnet.losses import compute_loss

d = torch.device("cuda")
P = config.CFG["prior"]
C, H, W, dT, N_ITER = 4, 36, 12, 200, 20
ARMS = {"mse5": ("supervised", False, False), "vsb0": ("nll", False, False),
        "augobs": ("nll", True, True)}
PREC = {"bf16": torch.bfloat16, "fp32": None}
BATCHES = [4, 8, 12]
prop = torch.cuda.get_device_properties(0)
print(f"[card] {prop.name}  {prop.total_memory / 2**30:.1f} GiB  "
      f"bf16_native={torch.cuda.is_bf16_supported(including_emulation=False)}", flush=True)
res = []
for arm, (loss_kind, aug, obs) in ARMS.items():
    for pname, pdt in PREC.items():
        for B in BATCHES:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            rec = {"arm": arm, "prec": pname, "batch": B}
            try:
                torch.manual_seed(0)
                phi = GENN(n_channels=C, hidden=96, kt=5, kh=P["kh"], kw=P["kw"],
                           n_phi_layers=P["n_phi_layers"], two_scale=P["two_scale"], scale=P["scale"])
                S = GradSolver(phi, n_channels=C, dT=dT, n_iter=N_ITER,
                               hidden_ch=config.get("solver", "lstm_hidden"),
                               predict_var=loss_kind == "nll", var_eps=1e-6,
                               augmented_var=aug, obs_nll=obs).to(d)
                opt = torch.optim.Adam(S.parameters(), lr=1e-3)
                X = torch.randn(B, C, dT, H, W, device=d) * 0.2
                M = (torch.rand_like(X) > 0.5).float()
                Y = (X + torch.randn_like(X) * 0.05) * M
                times, finite = [], True
                for step in range(3):
                    torch.cuda.synchronize(); t0 = time.time()
                    opt.zero_grad(set_to_none=True)
                    ctx = torch.autocast("cuda", dtype=pdt) if pdt else nullcontext()
                    with ctx:
                        if loss_kind == "nll":
                            xr, var = S(Y.clone(), Y, M, return_var=True)
                        else:
                            xr, var = S(Y.clone(), Y, M), None
                        loss = compute_loss(loss_kind, xr.float(), X, Y, M, phi,
                                            var=None if var is None else var.float(), nll_beta=0.0)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0)
                    opt.step()
                    torch.cuda.synchronize()
                    finite &= bool(torch.isfinite(loss))
                    if step > 0:
                        times.append(time.time() - t0)
                rec.update(ok=True, finite=finite, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                           reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                           s_per_step=sum(times) / len(times))
            except torch.OutOfMemoryError:
                rec["ok"] = False
            except Exception as e:  # e.g. an unsupported bf16 kernel on this card
                rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
            print(json.dumps(rec), flush=True)
            res.append(rec)
            for v in ("S", "opt", "X", "Y", "M", "xr", "var", "loss", "phi"):
                globals().pop(v, None)
            torch.cuda.empty_cache()
            if not rec["ok"]:
                break
out = os.path.join(os.path.dirname(__file__), "prof_v100_batch.json")
json.dump({"card": prop.name, "dT": dT, "n_iter": N_ITER, "results": res}, open(out, "w"), indent=1)
print("[done]", out)
