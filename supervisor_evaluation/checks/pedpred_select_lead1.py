"""Route 2: re-score every saved PedPred3 epoch on the 14 validation days with the forecast the
EnKF actually uses -- one step ahead (horizon 1, no feedback of the model's own forecast),
clamped to the EnKF's upper bounds (density 5, |velocity| 5, variance 2) -- by the training loss
(weighted NLL) and the RMSE of surrogate_mean. The logged 5-step validation loss is dominated by
a few tracking-dropout frames on 2013-06-16 (and turns NaN once a blown-up step is fed back).
Only scorable target frames are used (crowdcore.frame_reliability: under DATA_VARIANT=all92 the
tracker-gap blocks and the block after each are left out).

    DATA_VARIANT=all92 python3 -m supervisor_evaluation.checks.pedpred_select_lead1 OUT.json CKPT...
"""
import json, math, sys, torch
from crowdcore import frame_reliability as fr
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import load_pedpred3, mean_forecast, original_loss, surrogate_mean

dev = torch.device("cuda")
valid = PairFrames("valid", dev, 0, 0, input_frames=5, output_frames=1)
keep = torch.nonzero(valid.scorable_pairs()).flatten()
print(f"[data] {fr.describe()}: {len(keep):,} of {valid.n_pairs:,} one-step pairs scored", flush=True)
out, rows = sys.argv[1], []
for ck in sys.argv[2:]:
    m = load_pedpred3(ck, device=dev).eval()
    tot = 0.0; n = 0; se = 0.0; ne = 0; nonfinite = 0
    with torch.no_grad():
        for i in range(0, len(keep), 2048):
            x, y = valid.batch(keep[i:i + 2048])
            mu = mean_forecast(m, x, horizon=1)
            nonfinite += int((~torch.isfinite(mu)).flatten(1).any(1).sum())
            mu = torch.cat([mu[:, :, :1].clamp(max=5.0), mu[:, :, 1:3].clamp(-5.0, 5.0),
                            mu[:, :, 3:].clamp(max=2.0)], dim=2)
            tot += float(original_loss(mu, y)) * len(x); n += len(x)
            s = surrogate_mean(m, x, horizon=1)
            se += float(((s - y) ** 2).double().sum()); ne += s.numel()
    r = {"ckpt": ck, "lead1_loss_clipped": tot / n, "lead1_rmse_enkf": math.sqrt(se / ne),
         "nonfinite_samples": nonfinite, "scored_pairs": int(len(keep)), "gap_mask": fr.ENABLED}
    rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open(out, "w"), indent=1)
best = min((r for r in rows if math.isfinite(r["lead1_loss_clipped"])), key=lambda r: r["lead1_loss_clipped"])
print("[best by lead-1 clipped loss]", json.dumps(best), flush=True)
