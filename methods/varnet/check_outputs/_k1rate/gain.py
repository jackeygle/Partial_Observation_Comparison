"""Direct measurement: assimilating one frame of observations, how much does it actually change the state?"""
import sys, numpy as np, torch
PO="/scratch/work/zhangx29/Partial_observation"; sys.path.insert(0,PO)
from pedpred.ENKF import LocalizedEnsembleKalmanFilter
from pedpred.utils import load_model
H,W,F=36,12,4; TOTAL,SD=H*W,F*H*W
PROC=(0.02829307,0.31263075,0.12325809,0.41680932)
model=load_model(f"{PO}/apt-ibex_train_model_28D.pth", torch.device("cpu"))
z=np.load("check_outputs/enkf_k4_full/obs_atc-20130811.npz")
Y,Om,std,Xt=z["Y"],z["Omega"],z["obs_std"],z["X_true"]

f=LocalizedEnsembleKalmanFilter(grid_size=(H,W),state_shape=(F,H,W),ensemble_size=100,
    proc_noise_std=PROC,obs_noise_std=tuple(float(s) for s in std),
    init_perturb_std=(0.2290,1.2660,0.3429,0.0259),inflation=1.02,localization_radius=7)
rng=np.random.RandomState(0); X0=np.zeros((100,SD))
for i,s in enumerate(PROC): X0[:,i*TOTAL:(i+1)*TOTAL]=rng.normal(0,s,size=(100,TOTAL))
f.X=X0
def C_of(cells):
    C=np.zeros((F*len(cells),SD))
    for i,(r,c) in enumerate(cells):
        for k in range(F): C[i*F+k,k*TOTAL+r*W+c]=1.0
    return C

# run 500 frames first to let the filter reach steady state
for t in range(500):
    cells=list(zip(*np.where(Om[t])))
    if cells:
        C=C_of(cells); f.step(C,C@Y[t].reshape(-1),model=model)
    else:
        f.X=f._clip_bounds(f.forecast(model))

t=500
while not Om[t].any(): t+=1
cells=list(zip(*np.where(Om[t]))); C=C_of(cells); y=C@Y[t].reshape(-1)
Xsave=f.X.copy()
# (a) forecast only
Xf=f.forecast(model); est_fore=f._clip_bounds(Xf).mean(0)
# (b) forecast + assimilate
f.X=Xsave; est_anal=f.step(C,y,model=model).reshape(-1)

d=np.abs(est_anal-est_fore)
mm=np.zeros((F,H,W),bool)
for (r,c) in cells: mm[:,r,c]=True
mm=mm.reshape(-1)
print(f"  observed cells this frame: {len(cells)}")
print(f"  ensemble spread          : {f.get_std().mean():.5f}")
print()
print(f"  how much assimilation changed the state (|post-analysis - forecast only|):")
print(f"    observed cells   : {d[mm].mean():.6f}")
print(f"    unobserved cells : {d[~mm].mean():.6f}")
print()
print(f"  for reference, the scale of this frame's state itself:")
print(f"    |forecast-only value|     : {np.abs(est_fore[mm]).mean():.6f}")
print(f"    forecast vs. truth        : {np.abs(est_fore[mm]-Xt[t].reshape(-1)[mm]).mean():.6f}")
print(f"    observation vs. truth     : {np.abs(Y[t].reshape(-1)[mm]-Xt[t].reshape(-1)[mm]).mean():.6f}")
print()
print(f"  -> assimilation's correction / forecast's error = {d[mm].mean()/np.abs(est_fore[mm]-Xt[t].reshape(-1)[mm]).mean()*100:.2f}%")
