"""Where does PedPred3's validation density blow up? (route 2, all 92 days)

    DATA_VARIANT=all92 python3 -m supervisor_evaluation.checks.diag_pedpred_density CKPT [CKPT ...]

For each checkpoint, on the 14 validation days (truth history, 5 -> 5 frames, as in training):
per lead, the share of samples whose forecast density exceeds 5 (the EnKF's clip bound) and 1e3
anywhere on the grid; the lead-1 RMSE of the clipped mean the EnKF actually propagates
(surrogate_mean); per weekday/day counts; and what the exploding samples look like (people in
the last input frame, position in the day, max of the raw log-density output).
"""
import datetime, json, os, sys
import h5py, numpy as np, torch
from crowdcore import observation_model as om
from methods.enkf.lcskf.dynamics.data import PairFrames
from methods.enkf.lcskf.dynamics.model import load_pedpred3, raw_forecast, surrogate_mean

dev = torch.device("cuda")
valid = PairFrames("valid", dev, 0, 0, input_frames=5, output_frames=5)
files = om.split_files("valid")
lengths = [h5py.File(f, "r")["grid"].shape[0] for f in files]
starts = np.r_[0, np.cumsum(lengths)[:-1]]
days = [os.path.basename(f)[4:12] for f in files]
wd = ["Wed" if datetime.date(int(d[:4]), int(d[4:6]), int(d[6:])).weekday() == 2 else "Sun" for d in days]
tgt = valid.index.cpu().numpy()
day_of = np.searchsorted(starts, tgt, side="right") - 1
pos_in_day = (tgt - starts[day_of]) / np.asarray(lengths)[day_of]
print(f"[data] {valid.n_pairs:,} validation pairs, days {list(zip(days, wd))}", flush=True)

for ck in sys.argv[1:]:
    m = load_pedpred3(ck, device=dev).eval()
    over5 = np.zeros((valid.n_pairs, 5), bool); over1k = np.zeros((valid.n_pairs, 5), bool)
    rawmax = np.zeros((valid.n_pairs, 5), np.float32); people = np.zeros(valid.n_pairs, np.float32)
    se1 = 0.0; n1 = 0
    with torch.no_grad():
        for i in range(0, valid.n_pairs, 1024):
            rows = torch.arange(i, min(i + 1024, valid.n_pairs), device=dev)
            x, y = valid.batch(rows)
            raw = raw_forecast(m, x, horizon=5)                 # (B,5,4,H,W), ch0 = log density
            ld = raw[:, :, 0].flatten(2).amax(2)                # (B,5)
            rawmax[i:i + len(rows)] = ld.float().cpu().numpy()
            over5[i:i + len(rows)] = (ld > np.log(5.0)).cpu().numpy()
            over1k[i:i + len(rows)] = (ld > np.log(1e3)).cpu().numpy()
            people[i:i + len(rows)] = x[:, -1, 0].flatten(1).sum(1).cpu().numpy()
            mu1 = surrogate_mean(m, x, horizon=1)[:, 0]
            se1 += float(((mu1 - y[:, 0]) ** 2).double().sum()); n1 += mu1.numel()
    bad = over1k.any(1)
    print(f"\n=== {ck}", flush=True)
    print("share of samples with forecast density > 5, per lead: "
          + " ".join(f"{v*100:.3f}%" for v in over5.mean(0)), flush=True)
    print("share of samples with forecast density > 1e3, per lead: "
          + " ".join(f"{v*100:.4f}%" for v in over1k.mean(0)), flush=True)
    print(f"EnKF-relevant lead-1 clipped RMSE (all channels): {np.sqrt(se1 / n1):.5f}", flush=True)
    per_day = {f"{days[k]} {wd[k]}": int(bad[day_of == k].sum()) for k in range(len(days))}
    print("samples with density > 1e3 at any lead, per day: " + json.dumps(per_day), flush=True)
    if bad.any():
        first = over1k[bad].argmax(1)
        print(f"  first lead at which they exceed 1e3: {np.bincount(first, minlength=5).tolist()}", flush=True)
        print(f"  people in last input frame: exploding median {np.median(people[bad]):.2f} "
              f"(all samples median {np.median(people):.2f}); exploding share with < 1 person "
              f"{(people[bad] < 1).mean()*100:.1f}% (all {(people < 1).mean()*100:.1f}%)", flush=True)
        print(f"  position in the day (0=start,1=end): exploding quartiles "
              f"{np.round(np.quantile(pos_in_day[bad], [.1,.25,.5,.75,.9]), 3).tolist()}", flush=True)
        print(f"  max raw log-density: {float(rawmax[bad].max()):.1f} (normal samples p99.9 "
              f"{float(np.quantile(rawmax[~bad], .999)):.2f})", flush=True)
