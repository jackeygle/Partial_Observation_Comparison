"""End-to-end check: run the SAME frames through enkf_lab (pristine) and enkf_opt
(vectorised) and require the estimates to be bit-identical."""
import sys, time, numpy as np, importlib, torch
TP = "/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf"

def run(root, n_frames):
    for m in [k for k in list(sys.modules) if k.startswith(("pedpred", "tools"))]:
        del sys.modules[m]
    sys.path.insert(0, root)
    E = importlib.import_module("pedpred.ENKF")
    U = importlib.import_module("pedpred.utils")
    model = U.load_model(f"{root}/apt-ibex_train_model_28D.pth", torch.device("cpu"))
    sys.path.remove(root)

    z = np.load(f"{TP}/check_outputs/enkf_k1_full/obs_atc-20130811.npz")
    Y, Om = z["Y"], z["Omega"]
    H, W, F = 36, 12, 4
    f = E.LocalizedEnsembleKalmanFilter(grid_size=(H, W), state_shape=(F, H, W),
                                        ensemble_size=100, localization_radius=7, seed=0)
    f.initialize(np.zeros((F, H, W)))                      # identical deterministic init
    Est = np.zeros((n_frames, F, H, W))
    t0 = time.perf_counter()
    for t in range(n_frames):
        cells = list(zip(*np.where(Om[t])))
        rows = []
        for (r, c) in cells:
            for k in range(F):
                e = np.zeros(F * H * W); e[k * H * W + r * W + c] = 1.0; rows.append(e)
        C = np.array(rows)
        Est[t] = f.step(C, C @ Y[t].reshape(-1), model=model)
    return Est, time.perf_counter() - t0

N = 12
a, ta = run(f"{TP}/enkf_lab", N)
b, tb = run(f"{TP}/enkf_opt", N)
print(f"  enkf_lab (原版)   {N} 帧  {ta:6.1f}s   {ta/N:.2f} s/帧")
print(f"  enkf_opt (向量化) {N} 帧  {tb:6.1f}s   {tb/N:.2f} s/帧   提速 {ta/tb:.0f}×")
print(f"\n  估计值逐位相同: {'✅ 是' if np.array_equal(a, b) else '❌ 否  maxdiff=' + str(np.abs(a-b).max())}")
