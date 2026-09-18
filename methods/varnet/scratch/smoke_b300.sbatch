#!/bin/bash
#SBATCH --job-name=smoke_b300
#SBATCH --partition=gpu-b300-288g-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/work/zhangx29/Thesis_Project/methods/varnet/runs/slurm_%x_%j.out
# Does scicomp-pytorch-env/2026.1 actually run on the B300 (Blackwell, sm_103)?
# 288 GB VRAM makes memory a non-issue; the question is purely whether the CUDA build
# has kernels for this architecture. Runs all three arms at the real n_iter=20 config
# for a few real steps -- if they print [epoch 0] lines, it works.
source /etc/profile 2>/dev/null
source /scratch/work/zhangx29/Thesis_Project/sbatch/_env.sh
cd /scratch/work/zhangx29/Thesis_Project/methods/varnet

python3 -c "import torch; p=torch.cuda.get_device_properties(0); print(f'[card] {p.name}  cc={p.major}.{p.minor}  {p.total_memory/1e9:.0f} GB'); print('[torch]', torch.__version__, 'cuda', torch.version.cuda)"
python3 -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=(x@x).sum(); y.backward() if x.requires_grad else None; torch.cuda.synchronize(); print('[matmul] ok', float(y))"
D=scratch/smoke_b300; rm -rf "$D"; mkdir -p "$D"
run () { TAG=$1; shift; echo; echo "==== $TAG ===="
    python3 -u -m methods.varnet.train --obs-every-k 1 --init-seed 0 --data-seed 0 \
        --hidden 96 --kt 5 --days 2 --dT 200 --epochs 1 --batch 32 --amp --steps 8 \
        --iter-schedule '0:20:3e-4' --clip-grad 1.0 --outdir "$D/$TAG" "$@" 2>&1 \
      | grep -vE "^\[data\]|^\[schedule\] ep" | tail -6
}
run mse5 --loss supervised
run vsb0 --loss nll --nll-beta 0.0 --var-eps 1e-6
run aug0 --loss nll --nll-beta 0.0 --var-eps 1e-6 --augmented-var
echo; echo "[smoke] b300 test done"
