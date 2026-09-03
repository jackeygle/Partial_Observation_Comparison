"""Where does the time go in the CURRENT (optimised) EnKF? Same node/cores as the benchmark."""
import cProfile, io, os, pstats, sys, time
import numpy as np, torch
ROOT = "/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "enkf_opt"))
from pedpred.ENKF import LocalizedEnsembleKalmanFilter
from pedpred.utils import load_model

H, W, F = 36, 12, 4; TOTAL, SD = H*W, F*H*W
PROC = (0.02829307, 0.31263075, 0.12325809, 0.41680932)
model = load_model(f"{ROOT}/enkf_opt/apt-ibex_train_model_28D.pth", torch.device("cpu"))
z = np.load(f"{ROOT}/check_outputs/enkf_k1_full/obs_atc-20130811.npz")
Y, Om, std = z["Y"], z["Omega"], z["obs_std"]
N = 60

def C_of(cells):
    C = np.zeros((F*len(cells), SD))
    for i, (r, c) in enumerate(cells):
        for k in range(F): C[i*F+k, k*TOTAL+r*W+c] = 1.0
    return C
pre = []
for t in range(N):
    cells = list(zip(*np.where(Om[t])))
    C = C_of(cells) if cells else None
    pre.append((C, C @ Y[t].reshape(-1) if C is not None else None))

def run():
    f = LocalizedEnsembleKalmanFilter(grid_size=(H,W), state_shape=(F,H,W), ensemble_size=100,
        proc_noise_std=PROC, obs_noise_std=tuple(float(s) for s in std),
        init_perturb_std=(0.2290,1.2660,0.3429,0.0259), inflation=1.02, localization_radius=7)
    rng = np.random.RandomState(0); X0 = np.zeros((100, SD))
    for i, s in enumerate(PROC): X0[:, i*TOTAL:(i+1)*TOTAL] = rng.normal(0, s, size=(100, TOTAL))
    f.X = X0
    for C, y in pre:
        if C is not None: f.step(C, y, model=model)
        else: f.X = f._clip_bounds(f.forecast(model))

run()                                                  # warm-up
t0 = time.perf_counter(); run(); wall = time.perf_counter() - t0
print(f"  未插桩实测: {N} 帧 {wall:.2f}s = {wall/N*1000:.1f} ms/帧")
print(f"  硬件: {len(os.sched_getaffinity(0))} cores, node={os.environ.get('SLURMD_NODENAME','')}\n")

pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
st = pstats.Stats(pr); tot = st.total_tt
print(f"  cProfile 总计 {tot:.2f}s(插桩有额外开销,看相对占比)")
print(f"  {'函数':52s} {'cumtime':>9} {'占比':>7} {'调用':>7}")
print("  " + "-"*80)
rows = []
for (fn, ln, name), (cc, nc, tt, ct, cal) in st.stats.items():
    short = f"{os.path.basename(fn)}:{name}"
    rows.append((ct, short, nc))
for ct, short, nc in sorted(rows, reverse=True)[:16]:
    if any(s in short for s in ("prof_opt", "<built-in method builtins.exec", "run")): continue
    print(f"  {short:52s} {ct:>9.2f} {ct/tot*100:>6.1f}% {nc:>7}")
