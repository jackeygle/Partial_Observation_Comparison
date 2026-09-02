import os, sys
# checks/ 里的脚本从项目根导入源码模块；根要插在最前(本目录的 losses.py 优先)，
# 4dvarnet_enkf 只能 append(它也有 losses.py，插到最前会把本目录的顶掉)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# checks/ 里的脚本从项目根导入源码模块；根要插在最前(本目录的 losses.py 优先)，
# 4dvarnet_enkf 只能 append(它也有 losses.py，插到最前会把本目录的顶掉)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf")
import numpy as np, h5py, glob, navigation as nav
fps=sorted(glob.glob('/scratch/work/zhangx29/data/grid_cache/atc-*_corridor_1.0s.h5'))
Xf=h5py.File(fps[0],'r')['grid']; valid=nav.build_valid_mask_from_config(Xf[:])
X=Xf[20000:32000]                        # 12000 s, 白天繁忙段
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
# 只在有人的格子上看 vx 的自相关(空格子 vx=0 是占位符,会人为拉高相关)
vxo=np.where(occ,vx,np.nan)
def acn(a,lags):
    m=np.nanmean(a,0); a=a-m; s=np.nanstd(a,0)+1e-12; out=[]
    for L in lags:
        if L==0: out.append(1.0); continue
        p=a[:-L]*a[L:]; out.append(float(np.nanmean(np.nanmean(p,0)/(s*s))))
    return out
print('AC vx|occ:   ', '  '.join('%5.2f'%v for v in acn(vxo,lags)))
# 持续性基线:用 t-L 帧当 t 帧的预测,误差 vs 用时间均值当预测
for L in (1,4,16,61,256):
    e=((rho[:-L]-rho[L:])**2).mean(); e0=((rho-rho.mean(0))**2).mean()
    ev=((vx[:-L]-vx[L:])**2).mean(); ev0=((vx-vx.mean(0))**2).mean()
    print('  persistence lag %4ds: density MSE %.5f (clim %.5f)  vx MSE %.4f (clim %.4f)'%(L,e,e0,ev,ev0))
