"""Run the P5 FIRST RUN (3-variable Gaussian compatibility probe).

Usage:
    python3 scripts/run_first_run.py --config configs/p5_first_run.json --split dev
    python3 scripts/run_first_run.py --config configs/p5_first_run.json --split test
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat.experiment import run  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL).decode()
        return bool(out.strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    run_id = args.run_id or f"{config['run_name']}_{args.split}"
    out_dir = os.path.join(ROOT, "results", "raw", run_id)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = run(config, args.split, out_dir)
    manifest.update({
        "run_id": run_id,
        "config_path": os.path.relpath(os.path.abspath(args.config), ROOT),
        "started_utc": started,
        "git_commit_at_run": git_commit(),
        "git_worktree_dirty_at_run": git_dirty(),
        "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "note": "Pure-Python single-process run; no numpy/torch, no GPU, no downloads.",
    })
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in ("run_id", "total_seconds", "budget_seconds")}, indent=1))
    for eid, e in manifest["experiments"].items():
        print(f"{eid:28s} {e['status']:10s} {e.get('seconds', 0):7.2f}s {e.get('error', '')}")


if __name__ == "__main__":
    main()
