"""R8-P5-CPU-BUDGET: the two flow arms retrained from step 0 with the R1-R4
initialization seeds under an equal process-CPU training budget (60.0 CPU-s per
arm: the whole independent ensemble, round-robin over its four networks, and
the whole shared network), then evaluated on a new test set with Euler 128
(primary) and 512 (sensitivity). Contract: configs/auto_run_r8.json.

Reuses the round-7 keyed randomness (src/p5compat/syn_learn_fm_r7.py) with the
split tags 'test_r8' / 'dev_r8' so that no R7 stream is reused. Approximate
OS-level CPU matching, not exact FLOP matching.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Dict, List

from . import fm_adapter as fa
from . import metrics as mt
from . import stats as st
from .seeds import seed_v2
from .syn_learn import _key, draw, holm, source
from .syn_learn_fm import Budget, BudgetExceeded, Stage, design, sha256_file
from . import syn_learn_fm_r7 as r7

ARMS = r7.ARMS
PATTERNS = r7.PATTERNS
LEVELS = r7.LEVELS


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def data(cfg, r8):
    truth = source(cfg)
    sp_ = cfg["splits"]
    rows = {}
    for split in ("train", "dev"):
        rng = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=split, level="-")))
        rows[split] = draw(truth, sp_["n_" + split], rng)
    rng = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=r8["test_set"]["branch_key"], level="-")))
    rows["test"] = draw(truth, r8["test_set"]["n"], rng)
    scale = {"y": mt.sd([r["y"] for r in rows["train"]]), "z": mt.sd([r["z"] for r in rows["train"]])}
    return truth, rows, scale


# ---------------------------------------------------------------------------
# CPU-budgeted training
# ---------------------------------------------------------------------------


def warmup(arm: str, cfg, rows, n_updates: int = 20, seed: int = 999) -> Dict[str, float]:
    """First-run costs (allocator, kernels) on a throwaway model of the same class; identical for both arms;
    not charged to the arm's training budget (recorded separately)."""
    d = design(cfg, arm)
    m = r7.build_model(arm, cfg, seed)
    stage = Stage()
    fa.train(m, rows, PATTERNS, n_updates, d["batch"], d["lr"], seed, log_every=n_updates)
    return stage.done()


def train_cpu_budget(arm: str, rep: str, cfg, r8, train_rows, dev_rows, budget: Budget, ckpt_dir: str) -> Dict[str, Any]:
    d = design(cfg, arm)
    seeds = r8["replicates"][rep][arm]
    cap = float(r8["training"]["cpu_budget_seconds_per_arm"])
    model = r7.build_model(arm, cfg, seeds["model_init"])
    c0 = time.process_time()

    def check(step):  # called after every full update: stop at the first update boundary past the budget
        if time.process_time() - c0 >= cap:
            return f"cpu budget {cap} s reached at update {step}"
        if budget.reason():
            return budget.reason()
        return None

    stage = Stage()
    info = fa.train(model, train_rows, PATTERNS, r8["training"]["max_updates_guard"], d["batch"], d["lr"], seeds["train_stream"],
                    log_every=500, check=check, check_every=1)
    cpu_used = time.process_time() - c0
    info["timing"] = stage.done()
    info["cpu_budget_seconds"] = cap
    info["cpu_used_seconds"] = cpu_used
    info["overshoot_seconds"] = cpu_used - cap
    info["stopped_at_full_update_boundary"] = True
    info["budget_reached"] = bool(info.get("stopped_early")) and "cpu budget" in str(info.get("stopped_early"))
    info["dev_fm_loss_fixed_noise_at_end"] = {p: fa.held_out_fm_loss(model, p, dev_rows, seeds["dev_loss_noise"]) for p in PATTERNS}
    info["seeds"] = seeds
    info["design"] = d
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{rep}_{arm}.pt")
    fa.torch.save({"arm": arm, "replicate": rep, "state_dict": model.state_dict(), "updates": info["updates"],
                   "cpu_budget_seconds": cap, "torch_version": fa.torch.__version__}, path)
    info["checkpoint"] = {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(ckpt_dir))), "sha256": sha256_file(path),
                          "bytes": os.path.getsize(path), "final_iterate": True}
    info["complete"] = info["budget_reached"] and info["all_losses_finite"]
    return model, info


# ---------------------------------------------------------------------------
# Evaluation and analysis
# ---------------------------------------------------------------------------


def evaluate_test(rep: str, arms: Dict[str, Any], cfg, r8, rows, scale, truth, budget: Budget, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    m, b_perm = r8["evaluation"]["M"], r8["evaluation"]["B"]
    prim = r8["solver"]["primary"]
    blocks, per_all = {}, {}
    for arm, model in arms.items():
        for level in LEVELS:
            for steps in [prim] + list(r8["solver"]["sensitivity"]):
                budget.check()
                do_test = steps == prim
                blk, per, samples = r7.evaluate_block(cfg["seed"], rep, arm, model, r8["split_tag"], level, steps, rows, scale, truth,
                                                      do_test, b_perm, m, budget, keep_samples=True)
                k = f"{rep}|{arm}|{level}|{steps}"
                blocks[k], per_all[k] = blk, per
                r7.write_jsonl(os.path.join(out_dir, "blocks.jsonl"), [blk])
                r7.write_jsonl(os.path.join(out_dir, "per_example.jsonl"), per)
                with gzip.open(os.path.join(out_dir, "samples.jsonl.gz"), "at") as f:
                    for smp in samples:
                        f.write(json.dumps({"replicate": rep, "arm": arm, "level": level, "steps": steps, **smp}) + "\n")
    return blocks, per_all


def analysis(per_all: Dict[str, List[Dict]], reps: List[str], prim: int, sens: int, run_seed: int, n_boot: int, split_tag: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"definition": {
        "d_ri(h)": "ED_U(shared direct, shared sequential)_ri - ED_U(independent direct, independent sequential)_ri; negative = shared smaller gap",
        "Theta_h": "mean over examples of the mean over the four fit-pairs of d_ri(h); conditional on these fits; not a population effect",
        "intervals": "per replicate: normal 95% CI over examples (pointwise); overall: normal CI over the per-example replicate means; cluster bootstrap resampling examples and carrying every pair, arm and solver value of an example (2000 resamples)",
        "not_pooled": "4 x 200 values are not 800 independent observations; no clipping of negative ED_U"}}
    for level in LEVELS:
        lv = {}
        for h in (prim, sens):
            d = {}
            for rep in reps:
                a = r7._by_example(per_all[f"{rep}|SHARED_CONDITIONAL_FM|{level}|{h}"], "ed_u")
                b = r7._by_example(per_all[f"{rep}|INDEPENDENT_CONDITIONAL_FM|{level}|{h}"], "ed_u")
                d[rep] = {i: a[i] - b[i] for i in sorted(set(a) & set(b))}
            common = sorted(set.intersection(*[set(v) for v in d.values()]))
            m_i = {i: sum(d[rep][i] for rep in reps) / len(reps) for i in common}
            per_rep = {rep: dict(st.mean_ci([d[rep][i] for i in common]), n=len(common)) for rep in reps}
            overall = st.mean_ci([m_i[i] for i in common])
            rng_b = random.Random(r7.key_seed(run_seed, split=split_tag, replicate="all", condition="between_arms", level=level, example="all",
                                              branch="bootstrap", purpose="bootstrap", steps=h))
            n = len(common); boots = []
            for _ in range(n_boot):
                s = 0.0
                for _ in range(n):
                    s += m_i[common[rng_b.randrange(n)]]
                boots.append(s / n)
            boots.sort()
            lo_i, hi_i = int(math.floor(0.025 * n_boot)), int(math.ceil(0.975 * n_boot)) - 1
            means = [v["mean"] for v in per_rep.values()]
            lv[str(h)] = {"per_replicate_mean_d": per_rep, "Theta": overall, "n_examples": n,
                          "bootstrap_percentile_95": {"lo": boots[max(lo_i, 0)], "hi": boots[min(hi_i, n_boot - 1)], "n_boot": n_boot},
                          "replicate_means_sorted": sorted(round(v, 6) for v in means), "replicate_range": [min(means), max(means)],
                          "sign_reversal_across_replicates": (min(means) < 0) != (max(means) < 0),
                          "between_replicate_sd_of_means": st.mean_ci(means)["sd"]}
        out[level] = lv
    sensd = {}
    for rep in reps:
        for arm in ARMS:
            for level in LEVELS:
                qkey = "q_direct_truth_ed_u" if level == "target" else "q_direct_truth_proj_ed_u"
                e = r7.paired(r7._by_example(per_all[f"{rep}|{arm}|{level}|{sens}"], "ed_u"), r7._by_example(per_all[f"{rep}|{arm}|{level}|{prim}"], "ed_u"))
                q = r7.paired(r7._by_example(per_all[f"{rep}|{arm}|{level}|{sens}"], qkey), r7._by_example(per_all[f"{rep}|{arm}|{level}|{prim}"], qkey))
                sensd[f"{rep}|{arm}|{level}"] = {f"{sens}_minus_{prim}": {"gap_ED_U": e, "quality_direct": q,
                                                                          "SOLVER_SENSITIVE": e["lo"] is not None and (e["lo"] > 0 or e["hi"] < 0)}}
    out["solver_sensitivity"] = sensd
    out["solver_note"] = "paired per-example differences on shared initial noise; agreement does not prove ODE-limit convergence; results describe the implemented sampler"
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def rows_digest(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def stage_train(cfg, r8, out_root, budget: Budget, rows, scale, truth) -> Dict[str, Any]:
    ckpt_dir = os.path.join(out_root, "checkpoints")
    info_all: Dict[str, Any] = {}
    wu = {arm: warmup(arm, cfg, rows["train"]) for arm in ARMS}  # first-run costs, identical treatment, not charged to the arm budgets
    models: Dict[str, Dict[str, Any]] = {}
    for rep in r8["replicate_ids"]:
        models[rep] = {}
        for arm in ARMS:
            budget.check(margin=float(r8["training"]["cpu_budget_seconds_per_arm"]) + 10.0)
            model, info = train_cpu_budget(arm, rep, cfg, r8, rows["train"], rows["dev"], budget, ckpt_dir)
            info_all[f"{rep}|{arm}"] = info
            models[rep][arm] = model
            with open(os.path.join(out_root, "training.json"), "w") as f:
                json.dump({"warmup_not_charged": wu, **info_all}, f, indent=1)
            if not info["complete"]:
                raise BudgetExceeded(f"{rep} {arm}: training did not reach its CPU budget cleanly ({info.get('stopped_early')}; finite={info['all_losses_finite']})")
    return models, info_all, wu


def run(cfg, r8, out_root: str, budget: Budget) -> Dict[str, Any]:
    truth, rows, scale = data(cfg, r8)
    assert rows_digest(rows["test"]) == r8["test_set"]["rows_sha256"], "test set digest mismatch with the registered config"
    os.makedirs(out_root, exist_ok=True)
    reps = r8["replicate_ids"]
    prim, sens = r8["solver"]["primary"], r8["solver"]["sensitivity"][0]
    status: Dict[str, Any] = {"status": "RUNNING", "stages": {}, "scale_from_train": scale}
    blocks_all, per_all, train_info = {}, {}, {}
    try:
        models, train_info, wu = stage_train(cfg, r8, out_root, budget, rows, scale, truth)
        status["stages"]["train"] = "COMPLETE"
        status["budget_after_train"] = budget.snapshot()
        for rep in reps:
            blocks, per = evaluate_test(rep, models[rep], cfg, r8, rows["test"], scale, truth, budget, os.path.join(out_root, "test"))
            blocks_all.update(blocks); per_all.update(per)
            status["stages"][f"test_{rep}"] = "COMPLETE"
        status["stages"]["test"] = "COMPLETE"
        status["status"] = "COMPLETE"
    except BudgetExceeded as e:
        status["status"] = "INCOMPLETE_BUDGET"
        status["reason"] = str(e)
    done = [rep for rep in reps if status["stages"].get(f"test_{rep}") == "COMPLETE"]
    summary: Dict[str, Any] = {"status": status["status"], "replicates_complete": done, "blocks": blocks_all,
                               "labels": {k: ("MODEL_FIT_OR_NUMERICS_INCONCLUSIVE" if not v["all_losses_finite"] else "EVALUABLE") for k, v in train_info.items()},
                               "test_non_finite": {k: b["n_examples"] - b["n_finite"] for k, b in blocks_all.items()}}
    if done:
        fill = lambda d: {k: (v if v is not None else 1.0) for k, v in d.items()}
        fam_t = {f"{rep}|{arm}": blocks_all[f"{rep}|{arm}|target|{prim}"]["p"] for rep in done for arm in ARMS}
        fam_j = {f"{rep}|{arm}": blocks_all[f"{rep}|{arm}|joint|{prim}"]["p"] for rep in done for arm in ARMS}
        summary["families"] = {"target_auxiliary": {"raw_p": fam_t, "holm_p": holm(fill(fam_t)), "m": 2 * len(reps), "alpha": 0.05,
                                                    "resolution_note": "B = 199: smallest attainable p 0.005 vs first Holm threshold 0.00625"},
                               "joint_secondary": {"raw_p": fam_j, "holm_p": holm(fill(fam_j)), "m": 2 * len(reps), "alpha": 0.05, "note": "separate secondary family"}}
        if len(done) == len(reps):
            summary["analysis"] = analysis(per_all, reps, prim, sens, cfg["seed"], r8["evaluation"]["n_boot"], r8["split_tag"])
        else:
            summary["analysis"] = {"status": "INCOMPLETE: Theta over the four pairs is not computed; completed pairs listed; no subset reported as the planned effect"}
    summary["budget"] = budget.snapshot()
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    status["training"] = {k: {kk: vv for kk, vv in v.items() if kk != "loss_trace"} for k, v in train_info.items()}
    status["budget"] = budget.snapshot()
    return status


def smoke(cfg, r8, out_dir: str, run_seed: int = 999) -> Dict[str, Any]:
    """Timing smoke (seed 999, train rows only): warm-up cost, CPU per update per arm, and a sampling benchmark at 128/512."""
    os.makedirs(out_dir, exist_ok=True)
    budget = Budget(120.0, 600.0)
    truth, rows, scale = data(cfg, r8)
    out: Dict[str, Any] = {"run_seed": run_seed, "arms": {}}
    for arm in ARMS:
        d = design(cfg, arm)
        wu = warmup(arm, cfg, rows["train"], seed=run_seed)
        model = r7.build_model(arm, cfg, run_seed)
        stage = Stage()
        info = fa.train(model, rows["train"], PATTERNS, 300, d["batch"], d["lr"], run_seed, log_every=100, check=lambda s: budget.reason())
        t = stage.done()
        bench = {}
        for steps in [r8["solver"]["primary"]] + list(r8["solver"]["sensitivity"]):
            stage = Stage()
            for i, r in enumerate(rows["train"][:2]):
                ctx = r7.Ctx(run_seed, "smoke_r8", "R0", arm, i, "target")
                r7.draw_direct(model, ctx, r["x"], 64, steps); r7.draw_sequential(model, ctx, r["x"], 64, steps)
            bench[str(steps)] = stage.done()["cpu_seconds"] / 2
        out["arms"][arm] = {"warmup_cpu_seconds": wu["cpu_seconds"], "cpu_per_update": t["cpu_seconds"] / 300,
                            "expected_updates_in_budget": int(float(r8["training"]["cpu_budget_seconds_per_arm"]) / (t["cpu_seconds"] / 300)),
                            "cpu_per_example_target_by_steps": bench, "all_losses_finite": info["all_losses_finite"]}
    n_rep = len(r8["replicate_ids"])
    est_train = n_rep * 2 * float(r8["training"]["cpu_budget_seconds_per_arm"]) + 2 * sum(out["arms"][a]["warmup_cpu_seconds"] for a in ARMS)
    per_ex = sum(sum(out["arms"][a]["cpu_per_example_target_by_steps"].values()) for a in ARMS) * 1.33
    est_sampling = n_rep * 200 * per_ex
    est_perm = n_rep * 2 * (1.1 + 4.3)
    out["estimate_cpu_seconds"] = {"train": est_train, "sampling": est_sampling, "permutation": est_perm, "other": 60.0,
                                   "total": est_train + est_sampling + est_perm + 60.0}
    out["smoke_cost"] = budget.snapshot()
    with open(os.path.join(out_dir, "smoke.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
