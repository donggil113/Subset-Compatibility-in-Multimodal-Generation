"""P5-SYN-LEARN-01 round 7: four fixed training replicates with solver sensitivity.

  --register   write configs/p5_syn_learn_01_fm_replicates_r7.json (seeds, keys, the fixed
               test set digest) before any training; refuses to overwrite.
  --smoke      train-only timing smoke (seed 999) into a scratch dir; no result is looked at.
  (default)    the single registered run into results/raw/p5_syn_learn_01_fm_r7/.
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

from p5compat import syn_learn_fm_r7 as r7mod  # noqa: E402
from p5compat.syn_learn import _key, draw, source  # noqa: E402
from p5compat.seeds import seed_v2  # noqa: E402

CFG = os.path.join(ROOT, "configs", "p5_syn_learn_01.json")
R7 = os.path.join(ROOT, "configs", "p5_syn_learn_01_fm_replicates_r7.json")
OUT = os.path.join(ROOT, "results", "raw", "p5_syn_learn_01_fm_r7")


def file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def register():
    if os.path.exists(R7):
        sys.exit(f"{R7} exists; registration is done once")
    cfg = json.load(open(CFG))
    truth = source(cfg)
    branch_key = "test_r7"
    rng = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=branch_key, level="-")))
    test_rows = draw(truth, 200, rng)
    rng_p = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch="test", level="-")))
    pilot_rows = draw(truth, cfg["splits"]["n_test"], rng_p)
    overlap = len({round(r["x"], 12) for r in test_rows} & {round(r["x"], 12) for r in pilot_rows})
    reps = ["R1", "R2", "R3", "R4"]
    seeds = {rep: {arm: {"model_init": 7000 + 100 * (k + 1) + (1 if arm.startswith("INDEP") else 2),
                         "train_stream": 7000 + 100 * (k + 1) + (11 if arm.startswith("INDEP") else 12),
                         "dev_loss_noise": 7000 + 100 * (k + 1) + (21 if arm.startswith("INDEP") else 22)}
                   for arm in r7mod.ARMS} for k, rep in enumerate(reps)}
    pilot_tr = json.load(open(os.path.join(ROOT, "results", "raw", "p5_syn_learn_01_fm_train", "training.json")))
    pilot_seeds = {arm: pilot_tr[arm]["seeds"] for arm in r7mod.ARMS}
    r7 = {
        "experiment_id": "P5-SYN-LEARN-01-R7", "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "REGISTERED_NOT_RUN",
        "base_config": {"path": "configs/p5_syn_learn_01.json", "file_sha256": file_sha(CFG), "unchanged": True},
        "amendments_unchanged": {"a1": file_sha(os.path.join(ROOT, "configs", "p5_syn_learn_01_fm_amendment_a1.json")),
                                 "a2": file_sha(os.path.join(ROOT, "configs", "p5_syn_learn_01_fm_amendment_a2.json"))},
        "code_hashes_at_registration": {p: file_sha(os.path.join(ROOT, p)) for p in ("src/p5compat/fm_adapter.py", "src/p5compat/syn_learn_fm.py",
                                                                                       "src/p5compat/syn_learn_fm_r7.py", "src/p5compat/gmm.py", "src/p5compat/permutation.py",
                                                                                       "src/p5compat/metrics.py", "src/p5compat/seeds.py", "tests/test_fm_adapter.py", "tests/test_fm_r7.py")},
        "pilot": {"status": "DEVELOPMENT_NEURAL_PILOT (round 6, commit 7e25f68); preserved; not re-scored; its model seeds are excluded below",
                  "pilot_model_seeds": pilot_seeds,
                  "pilot_streams": {"sampling_keys": "seed_v2(seed, experiment, joint, condition=<arm>, level=<target|joint>, example=i, replicate=0, branch=<direct|sequential>) -> random.Random -> getrandbits(63) -> torch.Generator (low 32 bits used): no split field, so dev example i and test example i drew identical initial noise; no step-count field, so 128 and 32 shared noise",
                                    "permutation_keys": "... branch='permutation', solver=<steps>: no split field",
                                    "dev_uses": "dev-loss diagnostic at 90/100% (fixed noise key branch='dev_loss'); labels UNDERTRAINING_FLAG, SOLVER_SENSITIVE (dev 128 vs 32), non-finite; a descriptive dev Test P was computed and seen before the test stage; the checkpoint was the final iterate and no score-based selection was performed; the independence of that reading from the test outcome is not assumed, only recorded"}},
        "replicate_ids": reps, "replicates": seeds,
        "rng": {"model_init": "torch.manual_seed(model_init) before constructing the networks (integer per replicate and arm)",
                "train_stream": "fa.train(seed=train_stream): torch.Generator for (x0, t) noise and random.Random for minibatch indices and the shared arm's pattern choice",
                "dev_loss_noise": "fixed (x0, t) for the held-out dev loss at 18000 and 20000 updates",
                "sampling": "stable hash seed_v2 over {experiment, joint, split, replicate, arm, pattern, example (conditioning id), branch, sample_id, purpose='sampling:<level>'}; one stdlib Gaussian stream per sample; the solver step count is deliberately absent so 128/256/512 share every sample's initial noise; direct and sequential never share a stream; a fresh y is drawn for every z sample of the sequential branch",
                "permutation": "stable hash with purpose='permutation', split, replicate, arm, level, steps", "bootstrap": "purpose='bootstrap'"},
        "data": {"train": "unchanged (2000 rows, base data key)", "dev": "unchanged rows (200); sampling keys now carry split='dev'",
                 "standardisation": "y, z divided by the train-split sds (unchanged)"},
        "test_set": {"branch_key": branch_key, "n": 200, "rows_sha256": r7mod.rows_digest(test_rows),
                     "disjoint_from_pilot_test": overlap == 0, "x_overlap_count_with_pilot_test": overlap,
                     "rows_file": "results/raw/p5_syn_learn_01_fm_r7/test_examples.json (written at registration)"},
        "training": {"independent": "4 networks x 5000 updates = 20000", "shared": "20000 updates", "checkpoint": "final iterate only",
                     "fixed_as_A2": ["architecture", "batch 256", "loss", "Adam lr 1e-3, no schedule"], "not_equal_compute": True},
        "solver": {"primary": 128, "sensitivity": [256, 512], "integrator": "Euler", "cfg": "off", "base": "N(0, I)",
                   "rule": "the primary is fixed; 256/512 are numerical sensitivity endpoints with the same gap and quality estimators and paired differences; no permutation test is run at 256/512; nothing is selected by any outcome"},
        "evaluation": {"N": 200, "M": 64, "B": 199, "n_boot": 2000, "test_statistic": "mean ED_V, upper tail (Test P), at 128 only",
                       "families": {"target_auxiliary": "8 target tests (4 replicates x 2 arms) at 128, Holm; smallest attainable p 0.005 vs first threshold 0.05/8 = 0.00625",
                                    "joint_secondary": "8 joint tests at 128, Holm, separate; the round-6 joint p = 0.035 stays unadjusted and is not re-scored"},
                       "primary_effect": "Theta_128 = mean_i mean_r d_ri(128), d_ri(h) = ED_U(shared)_ri - ED_U(independent)_ri; conditional on the four fit-pairs; per-replicate means shown side by side; example-level paired CIs; bootstrap resamples examples carrying all replicates/arms/steps; 4 x 200 never pooled as 800",
                       "quality": "per example: unbiased sample ED to the true conditional for the direct and the sequential branch (same population estimand as the closed-form control distance; different estimator and, for this round, different test examples), sd ratios and mean shifts for both branches",
                       "no_ranking_by_p": True, "no_clipping": True},
        "labels": {"undertraining_threshold": 0.02, "undertraining_role": "diagnostic only; final iterate kept",
                   "inconclusive": "MODEL_FIT_OR_NUMERICS_INCONCLUSIVE only for non-finite losses or samples",
                   "numerics": "NUMERICS_NOT_SEPARATED stays whenever the paired 256-128 or 512-128 difference of the gap estimate has a 95% CI excluding 0; agreement is not an ODE-limit proof",
                   "outcome_labels": "SINGLE_PROBE_MULTI_INITIALIZATION_EVIDENCE if all four replicate means of d(128) share the sign of Theta_128 and their pointwise CIs exclude 0 in the same direction; MIXED_REPLICATION otherwise; both conditional on this probe and these fits"},
        "budget": {"cpu_cap_seconds": 3600, "wall_cap_seconds": 4500, "threads": 1, "ram_gib": 8,
                   "includes": ["fixtures", "timing smoke", "training", "generation", "permutation", "bootstrap", "failures"],
                   "blocked": "BLOCKED_BUDGET before any result if the smoke estimate exceeds the remaining cap; N, M and seeds are not reduced",
                   "incomplete": "INCOMPLETE_BUDGET with saved checkpoints and completed replicates; no seed substitution or arm selection"},
        "order": ["fixtures on the current snapshot", "registration (this file)", "commit", "timing smoke", "per replicate: train both arms -> dev (primary level, no test statistic) -> labels -> test (128 with Test P; 256; 512)",
                  "analysis", "nothing changed after any p-value or effect is seen"],
        "forbidden": ["new dataset", "GMM re-fit", "weights download", "GPU", "new architecture/loss", "optimizer/width sweep", "extra seeds"],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "test_examples.json"), "w") as f:
        json.dump({"branch_key": branch_key, "rows_sha256": r7["test_set"]["rows_sha256"], "rows": test_rows}, f)
    with open(R7, "w") as f:
        json.dump(r7, f, indent=2)
    print(json.dumps({"test_rows_sha256": r7["test_set"]["rows_sha256"], "overlap_with_pilot_test": overlap, "replicates": seeds}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu-spent-before", type=float, default=0.0)
    args = ap.parse_args()
    if args.register:
        register()
        return
    cfg, r7 = json.load(open(CFG)), json.load(open(R7))
    if args.smoke:
        if not args.out:
            sys.exit("--smoke needs --out (scratch)")
        out = r7mod.smoke(cfg, r7, args.out)
        print(json.dumps({"estimate_cpu_seconds": out["estimate_cpu_seconds"], "smoke_cost": out["smoke_cost"], "arms": out["arms"]}, indent=1))
        return
    out_root = args.out or OUT
    for prior in ("training.json", "manifest.json", "summary.json", "checkpoints", "dev", "test"):
        if os.path.exists(os.path.join(out_root, prior)):
            sys.exit(f"{prior} exists under {out_root}: this experiment is run once, and a partial run is kept (stop rule)")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "src", "configs", "scripts", "tests"], cwd=ROOT).decode().strip())
    env = json.load(open(os.path.join(ROOT, "results", "env", "torch_env.json")))
    code_now = {p: file_sha(os.path.join(ROOT, p)) for p in r7["code_hashes_at_registration"]}
    man = {"run_id": "p5_syn_learn_01_fm_r7", "experiment_id": r7["experiment_id"], "started_utc": started, "git_commit_at_run": commit,
           "code_dirty_at_run": dirty, "code_hashes_match_registration": code_now == r7["code_hashes_at_registration"],
           "config_sha256": file_sha(R7), "processes": 1, "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
           "python": sys.version.split()[0], "torch": torch.__version__, "torch_wheel_sha256": env["torch_wheel_sha256"], "numpy": env["numpy_version"],
           "device": str(torch.zeros(1).device), "cpu_spent_before_seconds": args.cpu_spent_before, "caps": r7["budget"]}
    os.makedirs(out_root, exist_ok=True)
    budget = r7mod.Budget(r7["budget"]["cpu_cap_seconds"], r7["budget"]["wall_cap_seconds"], args.cpu_spent_before)
    try:
        status = r7mod.run(cfg, r7, out_root, budget)
    except BaseException as e:
        man.update({"status": "FAILED", "error": repr(e), "budget": budget.snapshot()})
        with open(os.path.join(out_root, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        raise
    man.update(status)
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps({k: man[k] for k in ("status", "replicates", "budget") if k in man}, indent=1, default=str))


if __name__ == "__main__":
    main()
