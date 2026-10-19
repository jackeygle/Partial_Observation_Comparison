"""Freeze and validation-select all existing corrected K=20 snapshots, without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch

from crowdcore import navigation as nav
from crowdcore import observation_model as om
from methods.varnet.checks.model_io import load_solver
from methods.varnet.checks.select_checkpoint import score_day

ROOT = Path(__file__).resolve().parents[2]
RULE = "all_saved_corrected_K20"
RUNS = {"plain": "varnet_mse5_h96_pdfold_s3", "augmented": "varnet_aughead_obs_h96_fold_s0"}
OUT = ROOT / "supervisor_evaluation/outputs/dev/fold_adoption_20261009"
DIAGNOSTICS = ROOT / "supervisor_evaluation/outputs/dev/fold_diagnostics_20261009"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze():
    if (OUT / "candidate_plan.json").exists():
        return json.loads((OUT / "candidate_plan.json").read_text())
    audit = json.loads((DIAGNOSTICS / "checkpoint_audit.json").read_text())
    precrash = json.loads((DIAGNOSTICS / "precrash_full_validation_diagnostic.json").read_text())
    files = [Path(p).name for p in om.split_files("valid")]
    if len(files) != 7:
        raise ValueError("expected all seven validation days")
    cached = {}
    for label in ("plain_ct", "aug_ct"):
        selection = audit["selections"][label]
        if selection["split"] != "valid" or selection["days"] != files or selection["n_iter"] != 20:
            raise ValueError("cached selection protocol does not match")
        for epoch, mse in selection["mse_by_epoch"].items():
            run = "plain" if label == "plain_ct" else "augmented"
            path = ROOT / "methods/varnet/runs" / RUNS[run] / f"ckpt_{int(epoch):05d}.pt"
            matching = [m for m in audit["checkpoints"].values()
                        if m["path"] == str(path.relative_to(ROOT)) and m["sha256"] == sha(path)]
            if len(matching) != 1:
                raise ValueError(f"cached checkpoint hash is not uniquely verified: {path}")
            cached[(run, int(epoch))] = {"mse": mse, "sha256": matching[0]["sha256"],
                                       "source": str(DIAGNOSTICS / "checkpoint_audit.json")}
    if [x["day"] for x in precrash["per_day"]] != files or precrash["batch"] != 8:
        raise ValueError("precrash score does not cover the same validation protocol")
    cached[("plain", precrash["epoch"])] = {"mse": precrash["mse"], "sha256": precrash["sha256"],
                                           "source": str(DIAGNOSTICS / "precrash_full_validation_diagnostic.json")}
    plan = {"candidate_rule": RULE, "split": "valid", "days": files, "batch": 8,
            "n_iter": 20, "training_stopped_by_user": True, "arms": {}}
    for arm, run in RUNS.items():
        directory = ROOT / "methods/varnet/runs" / run
        sources = sorted(directory.glob("ckpt_*.pt"))
        sources += [directory / "varnet_best.pt", directory / "varnet_last.pt"]
        candidates = {}
        for source in sources:
            ck = torch.load(source, map_location="cpu", weights_only=True)
            if ck["args"].get("fold_axes") != "ct":
                raise ValueError(f"legacy checkpoint in corrected run: {source}")
            if ck.get("n_iter_eff") != 20:
                continue
            epoch = int(ck["epoch"])
            if epoch < 75:
                raise ValueError("K20 checkpoint precedes the final curriculum stage")
            if epoch in candidates:
                original = torch.load(directory / candidates[epoch]["checkpoint"], map_location="cpu", weights_only=True)
                if set(original["solver"]) != set(ck["solver"]) or any(
                    not torch.equal(original["solver"][k], ck["solver"][k]) for k in ck["solver"]):
                    raise ValueError(f"different saved weights share epoch {epoch}")
                candidates[epoch]["aliases"].append(source.name)
                continue
            frozen = directory / "frozen_candidates_20261009" / f"epoch_{epoch:05d}.pt"
            frozen.parent.mkdir(exist_ok=True)
            fingerprint = sha(source)
            if frozen.exists() and sha(frozen) != fingerprint:
                raise ValueError(f"refusing to change frozen candidate {frozen}")
            if not frozen.exists():
                shutil.copy2(source, frozen)
            candidate = {"epoch": epoch, "checkpoint": str(frozen.relative_to(directory)),
                         "sha256": fingerprint, "aliases": [source.name], "cached_score": None}
            cache = cached.get((arm, epoch))
            if cache and cache["sha256"] == fingerprint:
                candidate["cached_score"] = cache
            candidates[epoch] = candidate
        expected = [76, 80, 90] if arm == "plain" else [80, 88]
        if sorted(candidates) != expected:
            raise ValueError(f"{arm}: saved candidate set changed from the approved plan: {sorted(candidates)}")
        plan["arms"][arm] = {"run_dir": str(directory), "candidates": list(sorted(candidates.values(), key=lambda c:c["epoch"]))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze-only", action="store_true")
    a = ap.parse_args()
    plan = freeze()
    if a.freeze_only:
        print(json.dumps(plan, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("validation scoring requires CUDA; no CPU fallback")
    walk = nav.build_valid_mask_from_config()
    files = om.split_files("valid")
    results = {}
    for arm, config in plan["arms"].items():
        run = Path(config["run_dir"])
        scores, provenance = {}, {}
        pending, args_by_path = {}, {}
        for candidate in config["candidates"]:
            path = run / candidate["checkpoint"]
            if sha(path) != candidate["sha256"]:
                raise ValueError(f"frozen checkpoint changed: {path}")
            epoch = candidate["epoch"]
            cache = candidate["cached_score"]
            if cache:
                scores[epoch] = cache["mse"]
                provenance[str(epoch)] = {"kind": "reused_verified_validation_score", "source": cache["source"]}
                print(f"[select] {arm} epoch {epoch}: verified cached MSE={scores[epoch]:.8f}", flush=True)
            else:
                solver, model_args, _ = load_solver(path, "cuda")
                pending[epoch] = solver
                args_by_path[str(path)] = model_args
        if pending:
            se, count = {ep: 0.0 for ep in pending}, {ep: 0 for ep in pending}
            day_results = {ep: [] for ep in pending}
            for fp in files:
                for epoch, (s, n) in score_day(pending, fp, args_by_path, walk, "cuda", plan["batch"]).items():
                    se[epoch] += s; count[epoch] += n
                    day_results[epoch].append({"day": Path(fp).name, "squared_error_sum": s, "scored_count": n})
                    print(f"[select] {arm} epoch {epoch} {Path(fp).name}: MSE={s/n:.8f}", flush=True)
            for epoch in pending:
                scores[epoch] = se[epoch] / count[epoch]
                provenance[str(epoch)] = {"kind": "full_validation_scoring", "per_day": day_results[epoch]}
        if not all(math.isfinite(s) for s in scores.values()):
            raise ValueError("nonfinite validation score")
        winner = min(scores, key=lambda ep: (scores[ep], ep))
        chosen = next(c for c in config["candidates"] if c["epoch"] == winner)
        selection = {"run_dir": str(run), "candidate_rule": RULE, "split": "valid",
                     "min_epoch": 75, "n_iter": 20, "batch": plan["batch"], "days": plan["days"],
                     "scope": "walkable (blind cells inside the walkable region, clipped)",
                     "candidates": config["candidates"], "score_provenance": provenance,
                     "mse_by_epoch": {str(ep): mse for ep, mse in scores.items()},
                     "selected_epoch": winner, "selected_mse": scores[winner],
                     "selected_ckpt": chosen["checkpoint"], "checkpoint_sha256": chosen["sha256"],
                     "training_stopped_by_user": True,
                     "gpu_for_new_scores": torch.cuda.get_device_name(0)}
        old = run / "select_valid.json"
        backup = run / "select_valid_grid_before_saved_selection_20261009.json"
        if old.exists() and not backup.exists():
            shutil.copy2(old, backup)
        temporary = run / "select_valid.tmp"
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        temporary.replace(old)
        results[arm] = selection
        print(f"[selected] {arm} epoch {winner}: RMSE={math.sqrt(scores[winner]):.6f}", flush=True)
        pending.clear()
        torch.cuda.empty_cache()
    (OUT / "validation_selection.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
