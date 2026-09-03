import sys, os, torch
R="/scratch/work/zhangx29/Thesis_Project/4dvarnet_enkf"
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,"checks"))
from model_io import load_solver
d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for ck in ["runs/varnet_vsb0_s0/varnet_best.pt","runs/varnet_vrb0_s0/varnet_best.pt",
           "runs/varnet_ml5_s0/varnet_best.pt"]:
    S,a,_ = load_solver(os.path.join(R,ck), d)
    ov = S.grad_net.out_var
    print(f"OK {ck:40s} out_var={'None' if ov is None else tuple(ov.weight.shape)} "
          f"dropout={S.grad_net.dropout.p}")
