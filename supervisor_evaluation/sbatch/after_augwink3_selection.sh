#!/bin/bash
# Route 1: once the aug. var. selection (final_augwink3) has written status.json, package the six-
# method models, prepare outputs/full_6m_allwalk (the five models' accuracy files copied from the
# five-method evaluation, the EnKF run linked) and submit the evaluation and the gpu9 timing.
set -euo pipefail
cd /scratch/work/zhangx29/Thesis_Project
source sbatch/_env.sh >/dev/null 2>&1
SEL=supervisor_evaluation/outputs/dev/varnet_augwink3_200_20261012
until [ -f $SEL/status.json ]; do
    if [ -z "$(squeue -u $USER -h -n final_augwink3.sbatch,varnet_augwink3_final)" ]; then
        sleep 60; [ -f $SEL/status.json ] || { echo "selection jobs gone without status.json"; exit 1; }
    fi
    sleep 120
done
python3 -c "import json; d=json.load(open('$SEL/selection.json')); s=d['selected']; print('[selected]', s['epoch'], s['checkpoint'], 'valid all-walkable RMSE', round(s['mse']**0.5,5))"
export EVAL_VARIANT=six_allwalk DINCAE_VAR_TRANSFORM=none
python3 supervisor_evaluation/evaluate.py prepare
F=supervisor_evaluation/outputs/full_5m_allwalk_gaussdincae_20261011
O=supervisor_evaluation/outputs/full_6m_allwalk
mkdir -p $O/raw
cp -p $F/raw/senseiver_a_accuracy.json $F/raw/senseiver_g_accuracy.json $F/raw/varnet_mse_accuracy.json $O/raw/
cp -rp $F/raw/dincae_accuracy $O/raw/
for l in enkf enkf_exports enkf_observations; do [ -e $O/raw/$l ] || ln -s ../../full/raw/$l $O/raw/$l; done
ls -la $O/raw
sbatch supervisor_evaluation/sbatch/full_6m_allwalk.sbatch
sbatch supervisor_evaluation/sbatch/bench_6m_allwalk.sbatch
