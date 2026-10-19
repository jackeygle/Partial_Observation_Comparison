"""Run the approved no-training evaluation in resumable stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "supervisor_evaluation/outputs/dev/fold_adoption_20261009/state.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["all", "front", "uncertainty", "post"], default="all")
    a = ap.parse_args()
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.setdefault("completed_steps", [])
    state["new_training_performed"] = False
    state["training_budget_completed"] = False
    state["active_evaluation_job"] = os.environ.get("SLURM_JOB_ID")

    def save():
        STATE.parent.mkdir(parents=True, exist_ok=True)
        temp = STATE.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(STATE)

    def step(name, command):
        if name in state["completed_steps"]:
            print(f"[resume] {name} already completed", flush=True)
            return
        state.update(phase=name, error=None)
        save()
        print(f"[stage start] {name}", flush=True)
        start = time.monotonic()
        try:
            subprocess.run([sys.executable, "-u", *command], cwd=ROOT, check=True)
        except Exception as exc:
            state["error"] = str(exc)
            save()
            raise
        state["completed_steps"].append(name)
        state.setdefault("stage_seconds", {})[name] = time.monotonic() - start
        save()
        print(f"[stage complete] {name}: {state['stage_seconds'][name]:.1f}s", flush=True)

    common = ["--data-root", "/scratch/work/zhangx29/data", "--output-dir", "supervisor_evaluation/outputs/full"]
    if a.phase in ["all", "front"]:
        step("validation_selection", ["-m", "supervisor_evaluation.checks.select_saved_fold"])
        step("corrected_model_packaging", ["-m", "supervisor_evaluation.checks.adopt_fold_models"])
        step("accuracy", ["supervisor_evaluation/evaluate.py", "full", "--methods",
                          "varnet_mse", "varnet_aughead", *common])
    if a.phase in ["all", "uncertainty"]:
        if "corrected_model_packaging" not in state["completed_steps"]:
            raise RuntimeError("corrected models must be packaged before uncertainty evaluation")
        step("uncertainty", ["supervisor_evaluation/evaluate.py", "uncertainty", *common])
    if a.phase in ["all", "post"]:
        if not {"accuracy", "uncertainty"}.issubset(state["completed_steps"]):
            raise RuntimeError("accuracy and uncertainty must finish before final figures and breakdown")
        step("benchmark", ["supervisor_evaluation/evaluate.py", "benchmark", *common])
        step("images", ["supervisor_evaluation/evaluate.py", "images", "--n-images", "100", *common])
        step("breakdown", ["supervisor_evaluation/breakdown.py", *common])
        state.update(phase="evaluation_complete", thesis_update_pending=True)
        save()


if __name__ == "__main__":
    main()
