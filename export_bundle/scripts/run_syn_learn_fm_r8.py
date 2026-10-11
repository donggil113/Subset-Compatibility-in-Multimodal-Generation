"""R8-P5-CPU-BUDGET auto-runner (bounded; stages fixed in configs/auto_run_r8.json).

  --register        write configs/auto_run_r8.json and the fixed test set before any training (once)
  --smoke --out D   timing smoke (seed 999, train rows only)
  (default)         run the allowed stages in order, skipping stages already completed with the same
                    input hash (results/raw/p5_syn_learn_01_fm_r8/auto_state.json), under the caps
"""

import argparse
import hashlib
import json
import os
import random
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

from p5compat import syn_learn_fm_r8 as r8mod  # noqa: E402
from p5compat.syn_learn import _key, draw, source  # noqa: E402
from p5compat.seeds import seed_v2  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p5_syn_learn_01.json")
R7 = os.path.join(ROOT, "configs", "p5_syn_learn_01_fm_replicates_r7.json")
R8 = os.path.join(ROOT, "configs", "auto_run_r8.json")
OUT = os.path.join(ROOT, "results", "raw", "p5_syn_learn_01_fm_r8")
CODE = ["src/p5compat/fm_adapter.py", "src/p5compat/syn_learn_fm.py", "src/p5compat/syn_learn_fm_r7.py", "src/p5compat/syn_learn_fm_r8.py",
        "src/p5compat/gmm.py", "src/p5compat/permutation.py", "src/p5compat/metrics.py", "src/p5compat/seeds.py", "src/p5compat/stats.py",
        "scripts/run_syn_learn_fm_r8.py", "tests/test_fm_adapter.py", "tests/test_fm_r7.py", "tests/test_fm_r8.py"]


def file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def code_hashes():
    return {p: file_sha(os.path.join(ROOT, p)) for p in CODE if os.path.exists(os.path.join(ROOT, p))}


def register():
    if os.path.exists(R8):
        sys.exit(f"{R8} exists; registration is done once")
    cfg, r7 = json.load(open(CFG)), json.load(open(R7))
    truth = source(cfg)
    branch_key = "test_r8"
    rows = draw(truth, 200, random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=branch_key, level="-"))))
    pilot = draw(truth, cfg["splits"]["n_test"], random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch="test", level="-"))))
    r7rows = draw(truth, r7["test_set"]["n"], random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=r7["test_set"]["branch_key"], level="-"))))
    xs = {round(r["x"], 12) for r in rows}
    ov_pilot = len(xs & {round(r["x"], 12) for r in pilot}); ov_r7 = len(xs & {round(r["x"], 12) for r in r7rows})
    est = json.load(open(os.path.join(ROOT, "results", "estimator_check", "r8_ed_u_check.json")))
    r8 = {
        "run_id": "R8-P5-CPU-BUDGET", "experiment_id": "P5-SYN-LEARN-01-R8", "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "REGISTERED_NOT_RUN", "split_tag": "test_r8",
        "question": "does the measured compatibility/quality trade-off between independently trained conditionals and one shared conditional network remain when the two arms get the same process-CPU training budget (R7 matched update counts, not CPU)? Not a test of joint coherence of the shared network.",
        "allowed_stages": ["estimator_check (done before registration)", "fixtures", "smoke", "train", "test", "analysis", "assets_and_manuscript (operator)", "pdf_build_and_inspection (operator)", "delivery (operator)"],
        "inputs": {"base_config": {"path": "configs/p5_syn_learn_01.json", "sha256": file_sha(CFG)}, "r7_config": {"path": "configs/p5_syn_learn_01_fm_replicates_r7.json", "sha256": file_sha(R7)},
                   "estimator_check": {"path": "results/estimator_check/r8_ed_u_check.json", "sha256": file_sha(os.path.join(ROOT, "results", "estimator_check", "r8_ed_u_check.json")), "verdict": est["verdict"][:80]},
                   "code_hashes": code_hashes(), "torch_env": "results/env/torch_env.json"},
        "replicate_ids": r7["replicate_ids"], "replicates": r7["replicates"],
        "seeds_note": "initialization, minibatch-stream and dev-loss-noise seeds are the R7 integers, reused unchanged; only the stopping rule differs (CPU budget instead of 20000 updates)",
        "training": {"cpu_budget_seconds_per_arm": 60.0, "unit": "process CPU seconds of this single-thread process measured with time.process_time(), per arm: the whole independent ensemble (4 networks, balanced round-robin) and the whole shared network",
                     "stop": "after the first full update at which the arm's CPU use >= 60.0 s (overshoot recorded); final iterate only; no checkpoint selection",
                     "warmup": "20 updates on a throwaway model of the same class (seed 999) before each arm type; identical for both arms; not charged to the arm budget; recorded",
                     "max_updates_guard": 200000, "matching": "approximate OS-level CPU matching, not exact FLOP matching",
                     "fixed_as_before": ["architecture and role masks", "loss", "Adam lr 1e-3", "batch 256", "training rows", "standardisation"]},
        "data": {"train": "unchanged", "dev": "unchanged rows; used only for the end-of-training held-out loss diagnostic"},
        "test_set": {"branch_key": branch_key, "n": 200, "rows_sha256": r8mod.rows_digest(rows), "x_overlap_with_pilot_test": ov_pilot, "x_overlap_with_r7_test": ov_r7,
                     "disjoint": ov_pilot == 0 and ov_r7 == 0, "rows_file": "results/raw/p5_syn_learn_01_fm_r8/test_examples.json"},
        "rng": "seed_v2 hash over {experiment, joint, split='test_r8', replicate, arm, pattern, example, branch, sample_id, purpose}; step count excluded so 128/512 share initial noise; direct and sequential on disjoint streams; fresh y per z sample",
        "solver": {"primary": 128, "sensitivity": [512], "integrator": "Euler", "cfg": "off", "base": "N(0, I)", "rule": "fixed before the run; no permutation test at 512; nothing selected by outcome"},
        "evaluation": {"N": 200, "M": 64, "B": 199, "n_boot": 2000, "test": "Test P at 128 (mean ED_V, upper tail)",
                       "primary_endpoint": "Theta_128 = mean_i mean_r [ED_U(shared)_ri - ED_U(independent)_ri] at the target level; cluster bootstrap over examples carrying all pairs/arms/solvers; per-replicate contrasts and range shown; 4 x 200 not pooled",
                       "families": {"target": "8 target tests at 128, Holm", "joint": "8 joint tests at 128, Holm, separate secondary"},
                       "quality": "direct and sequential sample ED_U to the true conditional, sd ratios, mean shifts; cost: updates reached, exposure, parameters, CPU and wall",
                       "no_ranking_by_p": True, "no_clipping": True, "comparison_with_R7": "descriptive only (different test sets; design seen before); not an independent causal or equal-everything comparison"},
        "stop_conditions": ["leakage or split mismatch", "test-set digest mismatch", "non-finite losses or samples (arm labelled INCONCLUSIVE; run continues)", "resource caps", "data-loss risk"],
        "not_stop_conditions": ["null, mixed or unfavourable results", "p > 0.05", "intervals containing 0", "MIXED_REPLICATION"],
        "bug_policy": "an engineering bug keeps the original result as INVALID with its scope; one minimal fix and one re-run of the affected stage at most; protocol/estimand changes stop the run",
        "caps": {"cpu_seconds": 5400, "wall_seconds": 7200, "threads": 1, "ram_gib": 16, "includes": ["estimator check", "fixtures", "smoke", "8 fits", "evaluation", "permutation", "bootstrap"]},
        "evidence_hashes_completed": {"estimator_check": file_sha(os.path.join(ROOT, "results", "estimator_check", "r8_ed_u_check.json"))},
        "next_command": "OMP_NUM_THREADS=1 .venv-p5/bin/python scripts/run_syn_learn_fm_r8.py --cpu-spent-before <seconds>",
        "forbidden": ["new loss/model/mixture/dataset", "extra seeds", "threshold changes", "test replacement", "removing arms", "GPU", "downloads"],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "test_examples.json"), "w") as f:
        json.dump({"branch_key": branch_key, "rows_sha256": r8["test_set"]["rows_sha256"], "rows": rows}, f)
    with open(R8, "w") as f:
        json.dump(r8, f, indent=2)
    print(json.dumps({"test_rows_sha256": r8["test_set"]["rows_sha256"], "overlap_pilot": ov_pilot, "overlap_r7": ov_r7}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu-spent-before", type=float, default=0.0)
    args = ap.parse_args()
    if args.register:
        register(); return
    cfg, r8 = json.load(open(CFG)), json.load(open(R8))
    if args.smoke:
        if not args.out:
            sys.exit("--smoke needs --out (scratch)")
        out = r8mod.smoke(cfg, r8, args.out)
        print(json.dumps({"estimate_cpu_seconds": out["estimate_cpu_seconds"], "smoke_cost": out["smoke_cost"], "arms": out["arms"]}, indent=1)); return
    out_root = args.out or OUT
    state_path = os.path.join(out_root, "auto_state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"stages": {}}
    inputs_hash = hashlib.sha256(json.dumps({"config": file_sha(R8), "code": code_hashes()}, sort_keys=True).encode()).hexdigest()
    if state["stages"].get("train_and_test", {}).get("status") == "COMPLETE" and state["stages"]["train_and_test"].get("inputs_hash") == inputs_hash:
        print("train_and_test already COMPLETE with identical inputs; skipping (stop rule)"); return
    for prior in ("training.json", "summary.json", "checkpoints", "test"):
        if os.path.exists(os.path.join(out_root, prior)):
            sys.exit(f"{prior} exists under {out_root}: the run is done once; a partial run is kept (stop rule)")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "configs", "scripts", "tests"], cwd=ROOT).decode().strip())
    env = json.load(open(os.path.join(ROOT, "results", "env", "torch_env.json")))
    man = {"run_id": r8["run_id"], "experiment_id": r8["experiment_id"], "started_utc": started, "git_commit_at_run": commit, "code_dirty_at_run": dirty,
           "code_hashes_match_registration": code_hashes() == r8["inputs"]["code_hashes"], "config_sha256": file_sha(R8), "inputs_hash": inputs_hash,
           "pid": os.getpid(), "processes": 1, "threads": torch.get_num_threads(), "python": sys.version.split()[0], "torch": torch.__version__,
           "torch_wheel_sha256": env["torch_wheel_sha256"], "numpy": env["numpy_version"], "device": str(torch.zeros(1).device),
           "cpu_spent_before_seconds": args.cpu_spent_before, "caps": r8["caps"]}
    os.makedirs(out_root, exist_ok=True)
    budget = r8mod.Budget(r8["caps"]["cpu_seconds"], r8["caps"]["wall_seconds"], args.cpu_spent_before)
    try:
        status = r8mod.run(cfg, r8, out_root, budget)
    except BaseException as e:
        man.update({"status": "FAILED", "error": repr(e), "budget": budget.snapshot()})
        json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=1)
        state["stages"]["train_and_test"] = {"status": "FAILED", "inputs_hash": inputs_hash, "error": repr(e)}
        json.dump(state, open(state_path, "w"), indent=1)
        raise
    man.update(status)
    json.dump(man, open(os.path.join(out_root, "manifest.json"), "w"), indent=1)
    state["stages"]["train_and_test"] = {"status": status["status"], "inputs_hash": inputs_hash, "evidence_sha256": {f: file_sha(os.path.join(out_root, f)) for f in ("summary.json", "training.json", "manifest.json")},
                                         "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    state["next"] = "operator stages: python3 scripts/make_paper_assets.py; manuscript edit; PDF build with -halt-on-error -no-shell-escape; PyMuPDF inspection; delivery"
    json.dump(state, open(state_path, "w"), indent=1)
    print(json.dumps({k: man[k] for k in ("status", "stages", "budget") if k in man}, indent=1, default=str))


if __name__ == "__main__":
    main()
