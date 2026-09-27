"""Run P5-E8-CALIB once: python3 scripts/run_calib.py --config configs/p5_e8_calib_v3.json"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat.calib import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="override output dir (dev smoke only)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out_dir = args.out or os.path.join(ROOT, "results", "raw", cfg["run_name"])
    if args.out is None and os.path.exists(os.path.join(out_dir, "summary.json")):
        sys.exit(f"{out_dir} already has results; the calibration is run once (stop rule)")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "configs", "scripts"], cwd=ROOT).decode().strip())
    summ = run(cfg, out_dir)
    man = {"run_id": cfg["run_name"], "experiment_id": cfg["experiment_id"], "config_path": os.path.relpath(args.config, ROOT),
           "started_utc": started, "git_commit_at_run": commit, "code_dirty_at_run": dirty,
           "wall_seconds": summ["wall_seconds"], "cpu_seconds_tasks": summ["cpu_seconds_tasks"],
           "tasks": [summ["n_tasks_done"], summ["n_tasks_planned"]], "timed_out": summ["timed_out"],
           "workers": cfg["design"]["workers"], "python": sys.version.split()[0]}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))
    print(json.dumps(summ["gate"], indent=1))


if __name__ == "__main__":
    main()
