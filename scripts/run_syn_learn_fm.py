"""P5-SYN-LEARN-01 flow-matching arms (round 6; CPU, 1 thread, budget-guarded).

Smoke (train-only timing, seed 999, scratch output):
  .venv-p5/bin/python scripts/run_syn_learn_fm.py --smoke --out <scratch>
Single pre-registered run (refuses to overwrite results):
  .venv-p5/bin/python scripts/run_syn_learn_fm.py --cpu-spent-before <seconds already spent on fixtures and smoke>
"""

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from p5compat import syn_learn_fm as fm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "p5_syn_learn_01.json"))
    ap.add_argument("--amendment", default=os.path.join(ROOT, "configs", "p5_syn_learn_01_fm_amendment_a2.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="output root (required for --smoke; results/raw otherwise)")
    ap.add_argument("--cpu-spent-before", type=float, default=0.0, help="CPU-s already charged to the research budget")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    a2 = json.load(open(args.amendment))
    if args.smoke:
        if not args.out:
            sys.exit("--smoke needs --out (scratch)")
        out = fm.smoke(cfg, a2, args.out)
        print(json.dumps({"estimate_cpu_seconds": out["estimate_cpu_seconds"], "smoke_cost": out["smoke_cost"]}, indent=1))
        return
    out_root = args.out or os.path.join(ROOT, "results", "raw")
    if os.path.exists(os.path.join(out_root, "p5_syn_learn_01_fm_test", "summary.json")):
        sys.exit("flow-matching test results exist; this experiment is run once (stop rule)")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "configs", "scripts"], cwd=ROOT).decode().strip())
    budget = fm.Budget(a2["budget"]["cpu_cap_seconds"], a2["budget"]["wall_cap_seconds"], args.cpu_spent_before)
    status = fm.run(cfg, a2, out_root, budget)
    env = json.load(open(os.path.join(ROOT, "results", "env", "torch_env.json")))
    man = {"run_id": "p5_syn_learn_01_fm", "experiment_id": cfg["experiment_id"], "config_path": os.path.relpath(args.config, ROOT),
           "amendment_path": os.path.relpath(args.amendment, ROOT), "started_utc": started, "git_commit_at_run": commit,
           "code_dirty_at_run": dirty, "processes": 1, "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
           "python": sys.version.split()[0], "torch": torch.__version__, "torch_wheel_sha256": env["torch_wheel_sha256"],
           "numpy": env["numpy_version"], "device": str(torch.zeros(1).device), "cpu_spent_before_seconds": args.cpu_spent_before,
           "caps": a2["budget"], **status}
    os.makedirs(os.path.join(out_root, "p5_syn_learn_01_fm_train"), exist_ok=True)
    with open(os.path.join(out_root, "p5_syn_learn_01_fm_train", "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps({k: man[k] for k in ("status", "stages", "budget", "labels") if k in man}, indent=1, default=str))


if __name__ == "__main__":
    main()
