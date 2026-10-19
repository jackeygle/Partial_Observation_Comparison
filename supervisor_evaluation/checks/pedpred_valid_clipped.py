"""Validation loss of saved PedPred3 epochs, as logged (raw forecast) and with the forecast density
clamped to the EnKF's bound 5 (the only difference), plus the lead-1 clipped RMSE the EnKF sees.

    DATA_VARIANT=all92 python3 -m supervisor_evaluation.checks.pedpred_valid_clipped CKPT...
"""
import sys, torch
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import load_pedpred3, mean_forecast, original_loss, surrogate_mean

dev = torch.device("cuda")
valid = PairFrames("valid", dev, 0, 0, input_frames=5, output_frames=5)
print(f"[data] {valid.n_pairs:,} pairs", flush=True)
for ck in sys.argv[1:]:
    m = load_pedpred3(ck, device=dev).eval()
    raw = clip = 0.0; n = 0; se = 0.0; ne = 0
    with torch.no_grad():
        for i in range(0, valid.n_pairs, 1024):
            x, y = valid.batch(torch.arange(i, min(i + 1024, valid.n_pairs), device=dev))
            mu = mean_forecast(m, x, horizon=5)
            raw += float(original_loss(mu, y)) * len(x)
            # the EnKF's upper bounds (density 5, |velocity| 5, variance 2); no lower clamp, the
            # NLL takes the log of the variance
            mu_c = torch.cat([mu[:, :, :1].clamp(max=5.0), mu[:, :, 1:3].clamp(-5.0, 5.0),
                              mu[:, :, 3:].clamp(max=2.0)], dim=2)
            clip += float(original_loss(mu_c, y)) * len(x); n += len(x)
            mu1 = surrogate_mean(m, x, horizon=1)[:, 0]
            se += float(((mu1 - y[:, 0]) ** 2).double().sum()); ne += mu1.numel()
    print(f"{ck}  valid_raw {raw/n:.6f}  valid_clipped {clip/n:.6f}  enkf_lead1_rmse {(se/ne)**.5:.5f}", flush=True)
