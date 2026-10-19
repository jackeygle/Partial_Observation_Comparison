"""Archive the legacy comparison and package the validation-selected corrected models."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from supervisor_evaluation import evaluate as ev


def main():
    queued = subprocess.check_output(["squeue", "-h", "-u", "zhangx29", "-o", "%100j"], text=True)
    if any(line.strip() in {"varnet_mse5_fold_s3", "varnet_aughead_fold_s0"} for line in queued.splitlines()):
        raise RuntimeError("no training may remain active while adopting frozen snapshots")
    # Resolve and verify every source before changing the reported artifacts.
    sources = ev.source_files()
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing final artifacts: " + ", ".join(missing))
    for key in ("varnet_mse", "varnet_aughead"):
        selection = json.loads((ev.ROOT / ev.FINAL_CONFIG[key]["run_dir"] / "select_valid.json").read_text())
        if len(selection["days"]) != 7 or selection.get("candidate_rule") != "all_saved_corrected_K20":
            raise ValueError(f"{key}: expected full seven-day selection of existing corrected K20 snapshots")
    outputs = ev.HERE / "outputs"
    current = outputs / "full"
    archive = outputs / "full_legacy_before_ct_20261009"
    if archive.exists():
        raise FileExistsError(f"{archive}: refusing to overwrite an existing archive")
    unchanged = ["senseiver_a_accuracy.json", "senseiver_g_accuracy.json", "dincae_accuracy",
                 "enkf", "enkf_exports", "enkf_observations"]
    for item in unchanged:
        if not (current / "raw" / item).exists():
            raise FileNotFoundError(f"missing unchanged evaluation artifact: {item}")
    model_archive = ev.MODELS / "archive_legacy_20261009"
    model_archive.mkdir(exist_ok=False)
    for name in ["manifest.json", "varnet_mse_pd_s3_epoch80.pt", "varnet_aughead_obs_s0.pt"]:
        shutil.copy2(ev.MODELS / name, model_archive / name)
    current.rename(archive)
    (current / "raw").mkdir(parents=True)
    # Read-only reuse of the unchanged results; do not duplicate 8.6 GB of exports.
    for item in unchanged:
        (current / "raw" / item).symlink_to(archive / "raw" / item,
                                           target_is_directory=(archive / "raw" / item).is_dir())
    ev.prepare_models()
    verification = ev.verify(Path("/scratch/work/zhangx29/data"), True)
    ev.write_verification(verification, current)
    if not verification["ok"]:
        raise RuntimeError("corrected model package verification failed; evaluation must not start")
    selections = {key: json.loads((ev.ROOT / ev.FINAL_CONFIG[key]["run_dir"] /
                                  "select_valid.json").read_text())
                  for key in ("varnet_mse", "varnet_aughead")}
    (current / "corrected_model_selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    shutil.copy2(ev.MODELS / "manifest.json", current / "model_manifest.json")
    print(f"[adopt] archived {archive}; corrected final package verified", flush=True)


if __name__ == "__main__":
    main()
