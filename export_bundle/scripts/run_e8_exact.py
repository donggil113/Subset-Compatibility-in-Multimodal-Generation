"""Run P5-E8-EXACT (small exact randomisation verification).

Usage: python3 scripts/run_e8_exact.py --config configs/p5_e8_exact_v1.json
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat.e8 import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config = json.load(open(args.config))
    out_dir = os.path.join(ROOT, "results", "raw", config["run_name"])
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "configs", "scripts"], cwd=ROOT).decode().strip())
    res = run(config, out_dir)
    man = {"run_id": config["run_name"], "experiment_id": config["experiment_id"], "config_path": os.path.relpath(args.config, ROOT),
           "started_utc": started, "git_commit_at_run": commit, "code_dirty_at_run": dirty,
           "total_seconds": res["total_seconds"], "budget_seconds": config["budget"]["max_seconds"], "status": res["status"],
           "python": sys.version.split()[0], "note": "pure Python, single process"}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
