import os, sys
# checks/ 里的脚本从项目根导入源码模块；根要插在最前(本目录的 losses.py 优先)，
# 4dvarnet_enkf 只能 append(它也有 losses.py，插到最前会把本目录的顶掉)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# checks/ 里的脚本从项目根导入源码模块；根要插在最前(本目录的 losses.py 优先)，
# 4dvarnet_enkf 只能 append(它也有 losses.py，插到最前会把本目录的顶掉)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
import numpy as np, h5py, glob, observation_model as d, navigation as nav
fps=sorted(glob.glob('/scratch/work/zhangx29/data/grid_cache/atc-*_corridor_1.0s.h5'))
Xf=h5py.File(fps[0],'r')['grid']
X=Xf[:4000]
valid=nav.build_valid_mask_from_config(Xf[:])
print('walkable cells: %d/432 (%.1f%%)'%(valid.sum(),100*valid.mean()))
out=d.generate_observations(X, sensing_range=7, num_agents=3, add_noise=False,
                            seed=0, valid_mask=valid)
Om=out['Omega']; T=Om.shape[0]
print('per-frame coverage of walkable cells: %.1f%%'%(100*Om[:,valid].mean()))
gaps=[]; never=0
for i,j in zip(*np.where(valid)):
    t=np.flatnonzero(Om[:,i,j])
    if len(t)==0: never+=1
    elif len(t)>1: gaps.append(np.diff(t))
g=np.concatenate(gaps)
print('revisit gap (s): median %d  mean %.0f  p75 %d  p90 %d  p95 %d  p99 %d  max %d'%(
    np.median(g), g.mean(), *np.percentile(g,[75,90,95,99]).astype(int), g.max()))
print('walkable cells never observed in 4000 s: %d/%d'%(never, valid.sum()))
for W in (3,9,27,61,121,301,601,1201):
    s=[Om[t:t+W][:,valid].any(0).mean() for t in range(0,T-W,max(1,W//2))]
    print('  window %5d s -> P(cell seen >=1 time) = %.3f'%(W,np.mean(s)))
