"""直接测:同化一帧观测,到底把状态改变了多少?"""
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

# 先跑 500 帧让滤波器进入稳态
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
# (a) 只预报
Xf=f.forecast(model); est_fore=f._clip_bounds(Xf).mean(0)
# (b) 预报 + 同化
f.X=Xsave; est_anal=f.step(C,y,model=model).reshape(-1)

d=np.abs(est_anal-est_fore)
mm=np.zeros((F,H,W),bool)
for (r,c) in cells: mm[:,r,c]=True
mm=mm.reshape(-1)
print(f"  该帧观测格子数: {len(cells)}")
print(f"  集合 spread    : {f.get_std().mean():.5f}")
print()
print(f"  同化把状态改变了多少(|分析后 − 纯预报|):")
print(f"    被观测的格子 : {d[mm].mean():.6f}")
print(f"    未观测的格子 : {d[~mm].mean():.6f}")
print()
print(f"  作为参照,该帧状态本身的量级:")
print(f"    |纯预报值|   : {np.abs(est_fore[mm]).mean():.6f}")
print(f"    预报离真值   : {np.abs(est_fore[mm]-Xt[t].reshape(-1)[mm]).mean():.6f}")
print(f"    观测离真值   : {np.abs(Y[t].reshape(-1)[mm]-Xt[t].reshape(-1)[mm]).mean():.6f}")
print()
print(f"  → 同化的修正量 / 预报的误差 = {d[mm].mean()/np.abs(est_fore[mm]-Xt[t].reshape(-1)[mm]).mean()*100:.2f}%")
