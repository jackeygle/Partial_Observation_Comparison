import os, sys
from crowdcore import observation_model as om
# Scripts in checks/ import source modules from the project root; the root must
# be inserted at the front (so this directory's losses.py takes priority),
# 4dvarnet_enkf can only be appended (it also has a losses.py, and inserting it
# at the front would shadow this directory's).
import numpy as np, h5py, glob
from crowdcore import navigation as nav
# grid_cache location comes from crowdcore/config.yaml's data.root, not a hardcoded
# absolute path -- otherwise this script silently ignores a repointed data root.
fps=sorted(glob.glob(os.path.join(om.GRID_CACHE, 'atc-*_corridor_1.0s.h5')))
Xf=h5py.File(fps[0],'r')['grid']; valid=nav.build_valid_mask_from_config(Xf[:])
X=Xf[20000:32000]                        # 12000 s, a busy daytime segment
rho=X[:,0][:,valid]; vx=X[:,1][:,valid]
occ=rho>0
print('busy segment: mean density %.4f, occupancy %.1f%%'%(rho.mean(),100*occ.mean()))
def ac(a,lags):
    a=a-a.mean(0); s=a.std(0)+1e-12; out=[]
    for L in lags:
        out.append(float(np.mean((a[:-L]*a[L:]).mean(0)/(s*s)))) if L else out.append(1.0)
    return out
lags=[0,1,2,4,8,16,32,64,128,256,512]
print('lag(s):      ', '  '.join('%5d'%l for l in lags))
print('AC density:  ', '  '.join('%5.2f'%v for v in ac(rho,lags)))
print('AC vx:       ', '  '.join('%5.2f'%v for v in ac(vx,lags)))
# Only look at vx's autocorrelation on cells with people (empty-cell vx=0 is a
# placeholder, which would artificially inflate the correlation)
vxo=np.where(occ,vx,np.nan)
def acn(a,lags):
    m=np.nanmean(a,0); a=a-m; s=np.nanstd(a,0)+1e-12; out=[]
    for L in lags:
        if L==0: out.append(1.0); continue
        p=a[:-L]*a[L:]; out.append(float(np.nanmean(np.nanmean(p,0)/(s*s))))
    return out
print('AC vx|occ:   ', '  '.join('%5.2f'%v for v in acn(vxo,lags)))
# Persistence baseline: use frame t-L as the prediction for frame t, error vs. using the time mean as the prediction
for L in (1,4,16,61,256):
    e=((rho[:-L]-rho[L:])**2).mean(); e0=((rho-rho.mean(0))**2).mean()
    ev=((vx[:-L]-vx[L:])**2).mean(); ev0=((vx-vx.mean(0))**2).mean()
    print('  persistence lag %4ds: density MSE %.5f (clim %.5f)  vx MSE %.4f (clim %.4f)'%(L,e,e0,ev,ev0))
