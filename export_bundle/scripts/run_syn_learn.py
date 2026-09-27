"""Run P5-SYN-LEARN-01 once: python3 scripts/run_syn_learn.py --config configs/p5_syn_learn_01.json --split test"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat.syn_learn import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--out", default=None, help="override output dir (dev smoke only)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    out_dir = args.out or os.path.join(ROOT, "results", "raw", f"{cfg['run_name']}_{args.split}")
    if args.out is None and os.path.exists(os.path.join(out_dir, "summary.json")):
        sys.exit(f"{out_dir} already has results; this experiment is run once (stop rule)")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "configs", "scripts"], cwd=ROOT).decode().strip())
    t0 = time.time()
    s = run(cfg, out_dir, args.split)
    man = {"run_id": f"{cfg['run_name']}_{args.split}", "experiment_id": cfg["experiment_id"], "split": args.split,
           "config_path": os.path.relpath(args.config, ROOT), "started_utc": started, "git_commit_at_run": commit,
           "code_dirty_at_run": dirty, "wall_seconds": time.time() - t0, "processes": 1, "threads": 1,
           "python": sys.version.split()[0], "note": "single process; CPU-seconds approximately equal wall-seconds"}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
