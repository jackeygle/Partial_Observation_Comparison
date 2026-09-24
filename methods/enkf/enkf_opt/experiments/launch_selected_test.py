"""After validation finishes, lock candidates and launch their held-out test runs."""
from __future__ import annotations
import json, os, subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "outputs", "validation_sweep_summary.json")
OUT = os.path.join(HERE, "outputs", "selected_test")
SBATCH = os.path.join(HERE, "submit_eval_structured_q_gpu.sbatch")
TEST_DAYS = ("atc-20130811", "atc-20130818", "atc-20130825", "atc-20130901",
             "atc-20130915", "atc-20130922", "atc-20130929")
ARGS = {
    "g1": ["--noise-kind", "gaussian", "--scale", "1"],
    "r1": ["--noise-kind", "residual", "--scale", "1"],
    "r125": ["--noise-kind", "residual", "--scale", "1.25"],
    "r15": ["--noise-kind", "residual", "--scale", "1.5"],
    "r175": ["--noise-kind", "residual", "--scale", "1.75"],
    "r2": ["--noise-kind", "residual", "--scale", "2"],
    "r225": ["--noise-kind", "residual", "--scale", "2.25"],
    "r25": ["--noise-kind", "residual", "--scale", "2.5"],
    "r15_rt25": ["--noise-kind", "residual", "--scale", "1.5", "--rtps", ".25"],
    "r15_rt50": ["--noise-kind", "residual", "--scale", "1.5", "--rtps", ".5"],
    "r15_rho50": ["--noise-kind", "residual", "--scale", "1.5", "--temporal-rho", ".5"],
    "r15_rho80": ["--noise-kind", "residual", "--scale", "1.5", "--temporal-rho", ".8"],
}


def sbatch(*args):
    return subprocess.check_output(["sbatch", "--parsable", *args], text=True).strip().split(";")[0]


def main():
    summary = json.load(open(SUMMARY))
    locked = set(summary["selection"].values()) | {"g1", "r1", "r15"}
    unknown = locked - ARGS.keys()
    if unknown:
        raise RuntimeError(f"no command mapping for {unknown}")
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for day in TEST_DAYS:
        for config in sorted(locked):
            out = os.path.join(OUT, f"{day}_{config}.json")
            jid = sbatch(f"--job-name=t_{config}_{day[4:]}", SBATCH,
                         "--day", day, "--frames", "100000", "--warmup", "500",
                         "--out", out, *ARGS[config])
            jobs.append(jid)
            print("submitted", jid, day, config, flush=True)
    deps = ":".join(jobs)
    test_summary = os.path.join(HERE, "outputs", "selected_test_summary.json")
    summary_cmd = (f"source {ROOT}/sbatch/_env.sh; cd {ROOT}; "
                   "python3 -u -m methods.enkf.enkf_opt.experiments.summarize_validation_sweep "
                   f"--root {OUT} --out {test_summary}")
    summary_job = sbatch("--job-name=seltest_sum", "--partition=batch-bdw,batch-csl,batch-hsw,batch-skl",
                         "--cpus-per-task=1", "--mem=4G", "--time=00:10:00",
                         f"--dependency=afterok:{deps}",
                         f"--output={OUT}/summary_%j.out", f"--wrap={summary_cmd}")
    report = os.path.join(HERE, "outputs", "optimization_report.md")
    report_cmd = (f"source {ROOT}/sbatch/_env.sh; cd {ROOT}; "
                  "python3 -u -m methods.enkf.enkf_opt.experiments.write_optimization_report "
                  f"--out {report}")
    report_job = sbatch("--job-name=enkf_report", "--partition=batch-bdw,batch-csl,batch-hsw,batch-skl",
                        "--cpus-per-task=1", "--mem=4G", "--time=00:10:00",
                        f"--dependency=afterok:{summary_job}",
                        f"--output={OUT}/report_%j.out", f"--wrap={report_cmd}")
    manifest = {"locked_configs": sorted(locked), "selection_on_validation": summary["selection"],
                "test_jobs": jobs, "summary_job": summary_job, "report_job": report_job}
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
