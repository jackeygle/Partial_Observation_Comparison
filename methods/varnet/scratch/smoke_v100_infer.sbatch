#!/bin/bash
#SBATCH --job-name=smoke_v100_infer
#SBATCH --partition=gpu-v100-16g,gpu-v100-32g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Does select_checkpoint.py's INFERENCE path (not training) fit on a V100?
#
# Training OOM'd on both 16GB and 32GB V100 at n_iter=20 because
# variational_solver.py's create_graph=self.training builds a second-order graph
# for backprop-through-the-optimizer. select_checkpoint.py calls sol.eval() before
# scoring, so self.training=False -> create_graph=False -> only a first-order
# graph, and its --batch default is 8, not training's 32. Both should cut memory
# a lot; this checks by how much, on the exact code path (model_io.load_solver +
# solver.eval() + forward at n_iter=20, dT=200).
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(f'[card] {p.name}  {p.total_memory/1e9:.1f} GB')"

python3 -u -c "
import torch, numpy as np
from methods.varnet.checks.model_io import load_solver

dev = 'cuda'
sol, a, ck = load_solver('runs/varnet_mse5_h96_s0/ckpt_00080.pt', dev)
print(f'[model] n_iter={sol.n_iter}  hidden={a[\"hidden\"]}  dT={a[\"dT\"]}')
sol.eval()

B, C, T, H, W = 8, 4, 200, 36, 12
x0 = torch.randn(B, C, T, H, W, device=dev)
y  = torch.randn(B, C, T, H, W, device=dev)
m  = (torch.rand(B, C, T, H, W, device=dev) > 0.5).float()

torch.cuda.reset_peak_memory_stats()
with torch.enable_grad():
    for i in range(3):
        out = sol(x0, y, m)
        torch.cuda.synchronize()
        print(f'  step {i} ok, peak so far: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
print(f'[smoke] PASS -- peak VRAM (inference, batch=8, n_iter={sol.n_iter}): {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
"
