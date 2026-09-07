import os, sys
# Scripts in checks/ import source modules from the project root; the root must
# be inserted at the front (so this directory's losses.py takes priority),
# 4dvarnet_enkf can only be appended (it also has a losses.py, and inserting it
# at the front would shadow this directory's).
import numpy as np, h5py, glob
from crowdcore import observation_model as d
from crowdcore import navigation as nav
fps=sorted(glob.glob('/scratch/work/zhangx29/data/grid_cache/atc-*_corridor_1.0s.h5'))
Xf=h5py.File(fps[0],'r')['grid']; valid=nav.build_valid_mask_from_config(Xf[:])
X=Xf[20000:26000]                                  # 6000 s, a daytime segment
out=d.generate_observations(X,7,3,add_noise=False,seed=0,valid_mask=valid)
Om=out['Omega']; T=Om.shape[0]
# age of last observation, per (t,cell)
age=np.full(Om.shape,10**6,dtype=np.int32); last=np.full(valid.shape,-10**6,dtype=np.int64)
for t in range(T):
    last=np.where(Om[t],t,last); age[t]=t-last
A=age[:,valid]; O=Om[:,valid]
un=~O
print('unobserved (t,cell) share: %.1f%%'%(100*un.mean()))
a=A[un]; a=a[a<10**5]
print('age of last obs at UNOBSERVED cells (s): median %d  p25 %d  p75 %d  p90 %d  p95 %d'%(
    np.median(a),*np.percentile(a,[25,75,90,95]).astype(int)))
for thr in (2,4,8,16,32,64):
    print('   share of unobserved cells whose last obs is <= %2ds old : %.1f%%'%(thr,100*(a<=thr).mean()))
