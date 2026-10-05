"""P5-SYN-LEARN-01, round 7: four fixed training replicates of the two flow arms
with solver sensitivity (Euler 128 primary; 256 and 512 as pre-fixed numerical
sensitivity endpoints).

Contract: configs/p5_syn_learn_01_fm_replicates_r7.json (written by
`scripts/run_syn_learn_fm_r7.py --register` before any training). The base
config and amendments A1/A2 are unchanged; the round-6 run is preserved as
DEVELOPMENT_NEURAL_PILOT and is not re-scored.

Randomness: every primitive draw is keyed by a stable hash over
(split, replicate, arm, pattern, conditioning example, branch, sample_id,
purpose). The solver step count is deliberately absent from sampling keys, so
128/256/512 share the initial noise of every sample. Direct and sequential
branches never share a primitive stream; the sequential branch draws a fresh
y for every z sample.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence

from . import fm_adapter as fa
from . import gmm as gm
from . import metrics as mt
from . import permutation as pm
from . import stats as st
from .seeds import seed_v2
from .syn_learn import NAMES, _key, draw, holm, source
from .syn_learn_fm import Budget, BudgetExceeded, Stage, design, sample_vs_mixture_ed_u, sha256_file

EXP, JOINT = "P5-SYN-LEARN-01", "mixture3"
ARMS = ("INDEPENDENT_CONDITIONAL_FM", "SHARED_CONDITIONAL_FM")
PATTERNS = list(fa.PATTERNS)
LEVELS = ("target", "joint")


# ---------------------------------------------------------------------------
# Keyed randomness
# ---------------------------------------------------------------------------


def key_seed(run_seed: int, **fields) -> int:
    base = dict(experiment=EXP, joint=JOINT)
    base.update(fields)
    return seed_v2(run_seed, **base)


def noise_matrix(run_seed: int, split: str, rep: str, arm: str, pattern: str, example: int, branch: str, purpose: str,
                 m: int, dim: int):
    """Initial noise for m samples: one stdlib Gaussian stream per sample_id, keyed by all
    identifiers except the solver step count."""
    rows = []
    for j in range(m):
        r = random.Random(key_seed(run_seed, split=split, replicate=rep, condition=arm, pattern=pattern, example=example,
                                   branch=branch, sample_id=j, purpose=purpose))
        rows.append([r.gauss(0.0, 1.0) for _ in range(dim)])
    return fa.torch.tensor(rows, dtype=fa.torch.float32)


def euler(model, pattern: str, x0, gs: Sequence[Sequence[float]], steps: int):
    cond = fa.torch.tensor(gs, dtype=fa.torch.float32)
    x = x0.clone()
    h = 1.0 / steps
    with fa.torch.no_grad():
        for k in range(steps):
            t = fa.torch.full((x.shape[0], 1), k * h)
            x = x + h * model.velocity(pattern, x, t, cond)
    return x.tolist()


class Ctx:
    def __init__(self, run_seed: int, split: str, rep: str, arm: str, example: int, level: str):
        self.run_seed, self.split, self.rep, self.arm, self.example, self.level = run_seed, split, rep, arm, example, level

    def noise(self, pattern: str, branch: str, m: int, dim: int):
        return noise_matrix(self.run_seed, self.split, self.rep, self.arm, pattern, self.example, branch, f"sampling:{self.level}", m, dim)


def draw_direct(model, ctx: Ctx, x: float, m: int, steps: int) -> List[float]:
    return [v[0] for v in euler(model, "z|x", ctx.noise("z|x", "direct", m, 1), [[x]] * m, steps)]


def draw_sequential(model, ctx: Ctx, x: float, m: int, steps: int) -> Dict[str, List[float]]:
    y = [v[0] for v in euler(model, "y|x", ctx.noise("y|x", "sequential", m, 1), [[x]] * m, steps)]  # fresh y per sample
    z = [v[0] for v in euler(model, "z|x,y", ctx.noise("z|x,y", "sequential", m, 1), [[x, yj] for yj in y], steps)]
    return {"y": y, "z": z}


def draw_direct_joint(model, ctx: Ctx, x: float, m: int, steps: int) -> Dict[str, List[float]]:
    out = euler(model, "y,z|x", ctx.noise("y,z|x", "direct_joint", m, 2), [[x]] * m, steps)
    return {"y": [v[0] for v in out], "z": [v[1] for v in out]}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_model(arm: str, cfg, seed: int):
    d = design(cfg, arm)
    cls = fa.IndependentConditionalFM if arm == "INDEPENDENT_CONDITIONAL_FM" else fa.SharedConditionalFM
    return cls(d["mlp_width"], d["mlp_depth"], seed)


def train_pair_arm(arm: str, rep: str, cfg, r7, train_rows, dev_rows, budget: Budget, ckpt_dir: str, n_updates: Optional[int] = None):
    d = design(cfg, arm)
    seeds = r7["replicates"][rep][arm]
    total = n_updates if n_updates is not None else d["total_updates"]
    model = build_model(arm, cfg, seeds["model_init"])
    dev_losses: Dict[int, Dict[str, float]] = {}
    hook_at = sorted({int(round(0.9 * total)), total})

    def hook(step, m):
        dev_losses[step] = {p: fa.held_out_fm_loss(m, p, dev_rows, seeds["dev_loss_noise"]) for p in PATTERNS}

    stage = Stage()
    info = fa.train(model, train_rows, PATTERNS, total, d["batch"], d["lr"], seeds["train_stream"], log_every=1000,
                    hook=hook, hook_at=hook_at, check=lambda s: budget.reason())
    info["timing"] = stage.done()
    info["dev_fm_loss_fixed_noise"] = {str(k): v for k, v in dev_losses.items()}
    if len(dev_losses) == 2:
        k90, k100 = hook_at
        rel = {p: (dev_losses[k90][p] - dev_losses[k100][p]) / dev_losses[k90][p] for p in PATTERNS}
        info["dev_loss_rel_decrease_90_to_100"] = rel
        info["UNDERTRAINING_FLAG"] = any(v > r7["labels"]["undertraining_threshold"] for v in rel.values())
    info["seeds"] = seeds
    info["design"] = d
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{rep}_{arm}.pt")
    fa.torch.save({"arm": arm, "replicate": rep, "state_dict": model.state_dict(), "updates": info["updates"],
                   "updates_planned": info["updates_planned"], "torch_version": fa.torch.__version__}, path)
    info["checkpoint"] = {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(ckpt_dir))), "sha256": sha256_file(path),
                          "bytes": os.path.getsize(path), "final_iterate": info["updates"] == info["updates_planned"]}
    info["complete"] = info["updates"] == total
    return model, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _finite(vals) -> bool:
    return all(math.isfinite(v) for v in vals)


def evaluate_block(run_seed: int, rep: str, arm: str, model, split: str, level: str, steps: int, rows, scale, truth,
                   do_test: bool, b_perm: int, m: int, budget: Budget, keep_samples: bool):
    sy, sz = scale["y"], scale["z"]
    dirs = mt.slice_directions(8)
    per, exs, samples = [], [], []
    stage = Stage()
    for i, ex in enumerate(rows):
        x = ex["x"]
        ctx = Ctx(run_seed, split, rep, arm, i, level)
        rec: Dict[str, Any] = {"replicate": rep, "arm": arm, "split": split, "level": level, "steps": steps, "example": i, "x": x}
        oc = draw_sequential(model, ctx, x, m, steps)
        rec["n_distinct_y_seq"] = len(set(oc["y"]))
        if level == "target":
            za = draw_direct(model, ctx, x, m, steps)
            ok = _finite(za) and _finite(oc["z"]) and _finite(oc["y"])
            rec["finite"] = ok
            if ok:
                a_s, c_s = [v / sz for v in za], [v / sz for v in oc["z"]]
                rec["ed_u"] = mt.energy_distance_1d(a_s, c_s, unbiased=True)
                rec["ed_v"] = mt.energy_distance_1d(a_s, c_s, unbiased=False)
                tmix = gm.scale_mixture(truth.conditional_1d("z", ["x"], [x]), sz)
                rec["q_direct_truth_ed_u"] = sample_vs_mixture_ed_u(a_s, tmix)
                rec["q_seq_truth_ed_u"] = sample_vs_mixture_ed_u(c_s, tmix)
                tm, tv = gm.mix_moments(truth.conditional_1d("z", ["x"], [x]))
                rec["sd_ratio_direct"] = mt.sd(za) / math.sqrt(tv)
                rec["sd_ratio_seq"] = mt.sd(oc["z"]) / math.sqrt(tv)
                rec["mean_direct_minus_truth_std"] = (mt.mean(za) - tm) / math.sqrt(tv)
                rec["mean_seq_minus_truth_std"] = (mt.mean(oc["z"]) - tm) / math.sqrt(tv)
                if do_test:
                    exs.append(pm.PooledExample(a_s, c_s))
                if keep_samples:
                    samples.append({"example": i, "direct_z": [round(v, 6) for v in za], "seq_y": [round(v, 6) for v in oc["y"]],
                                    "seq_z": [round(v, 6) for v in oc["z"]]})
        else:
            oa = draw_direct_joint(model, ctx, x, m, steps)
            ok = _finite(oa["y"]) and _finite(oa["z"]) and _finite(oc["y"]) and _finite(oc["z"])
            rec["finite"] = ok
            if ok:
                pa = [[u / sy, v / sz] for u, v in zip(oa["y"], oa["z"])]
                pc = [[u / sy, v / sz] for u, v in zip(oc["y"], oc["z"])]
                tc = truth.conditional_components(["y", "z"], ["x"], [x])
                edu, edv, qa, qc = 0.0, 0.0, 0.0, 0.0
                for u in dirs:
                    proj_a = [u[0] * p[0] + u[1] * p[1] for p in pa]
                    proj_c = [u[0] * p[0] + u[1] * p[1] for p in pc]
                    edu += mt.energy_distance_1d(proj_a, proj_c, unbiased=True)
                    edv += mt.energy_distance_1d(proj_a, proj_c, unbiased=False)
                    tmix = gm.project_2d(tc, [u[0] / sy, u[1] / sz], [x])
                    qa += sample_vs_mixture_ed_u(proj_a, tmix)
                    qc += sample_vs_mixture_ed_u(proj_c, tmix)
                rec["ed_u"], rec["ed_v"] = edu / len(dirs), edv / len(dirs)
                rec["q_direct_truth_proj_ed_u"], rec["q_seq_truth_proj_ed_u"] = qa / len(dirs), qc / len(dirs)
                if do_test:
                    exs.append(pm.PooledExample(pa, pc, dirs))
                if keep_samples:
                    samples.append({"example": i, "direct_y": [round(v, 6) for v in oa["y"]], "direct_z": [round(v, 6) for v in oa["z"]],
                                    "seq_y": [round(v, 6) for v in oc["y"]], "seq_z": [round(v, 6) for v in oc["z"]]})
        per.append(rec)
        if i % 50 == 49:
            budget.check()
    t_sampling = stage.done()
    res, t_test = None, {"wall_seconds": 0.0, "cpu_seconds": 0.0}
    n_finite = sum(1 for r in per if r["finite"])
    if do_test and n_finite >= 2:
        stage = Stage()
        rng_p = random.Random(key_seed(run_seed, split=split, replicate=rep, condition=arm, level=level, example="all", branch="permutation",
                                       purpose="permutation", steps=steps))
        res = pm.mc_permutation_test(exs, b_perm, rng_p)
        t_test = stage.done()
    fin = [r for r in per if r["finite"]]
    blk = {"replicate": rep, "arm": arm, "split": split, "level": level, "steps": steps, "n_examples": len(rows), "n_finite": n_finite,
           "non_finite_examples": [r["example"] for r in per if not r["finite"]], "M": m, "B": b_perm if do_test else None,
           "p": res["p"] if res else None, "t_obs": res["t_obs"] if res else None, "effect_ED_U": st.mean_ci([r["ed_u"] for r in fin]),
           "min_distinct_intermediates": min(r["n_distinct_y_seq"] for r in per) if per else None,
           "timing_sampling": t_sampling, "timing_test": t_test}
    if level == "target":
        for k in ("q_direct_truth_ed_u", "q_seq_truth_ed_u", "sd_ratio_direct", "sd_ratio_seq", "mean_direct_minus_truth_std", "mean_seq_minus_truth_std"):
            blk[k] = st.mean_ci([r[k] for r in fin])
    else:
        for k in ("q_direct_truth_proj_ed_u", "q_seq_truth_proj_ed_u"):
            blk[k] = st.mean_ci([r[k] for r in fin])
    return blk, per, samples


# ---------------------------------------------------------------------------
# Analysis (all paired at the conditioning example)
# ---------------------------------------------------------------------------


def _by_example(per: List[Dict], key: str) -> Dict[int, float]:
    return {r["example"]: r[key] for r in per if r.get("finite")}


def paired(a: Dict[int, float], b: Dict[int, float]) -> Dict[str, Any]:
    common = sorted(set(a) & set(b))
    out = st.mean_ci([a[i] - b[i] for i in common])
    out["n_pairs"] = len(common)
    return out


def analysis(per_all: Dict[str, List[Dict]], reps: List[str], steps_all: List[int], prim: int, run_seed: int, n_boot: int) -> Dict[str, Any]:
    """Between-arm paired effects per replicate and overall (conditional on the fits), plus solver sensitivity."""
    out: Dict[str, Any] = {"definition": {
        "d_ri(h)": "ED_U(shared direct, shared sequential)_ri - ED_U(independent direct, independent sequential)_ri on the standardised scale; negative = shared smaller gap",
        "Theta_h": "mean over examples i of mean over the four replicates r of d_ri(h); conditional on the four fit-pairs, not a population effect",
        "intervals": "per replicate: normal 95% CI over the 200 examples (pointwise); overall: normal 95% CI over the per-example replicate means m_i, and a percentile bootstrap that resamples examples i and carries all replicates, arms and step counts of i together (2000 resamples)",
        "not_pooled": "4 x 200 values are never treated as 800 independent observations"}}
    for level in LEVELS:
        lv: Dict[str, Any] = {}
        for h in steps_all:
            d: Dict[str, Dict[int, float]] = {}
            for rep in reps:
                a = _by_example(per_all[f"{rep}|SHARED_CONDITIONAL_FM|{level}|{h}"], "ed_u")
                b = _by_example(per_all[f"{rep}|INDEPENDENT_CONDITIONAL_FM|{level}|{h}"], "ed_u")
                d[rep] = {i: a[i] - b[i] for i in sorted(set(a) & set(b))}
            common = sorted(set.intersection(*[set(v) for v in d.values()])) if d else []
            m_i = {i: sum(d[rep][i] for rep in reps) / len(reps) for i in common}
            per_rep = {rep: dict(st.mean_ci([d[rep][i] for i in common]), n=len(common)) for rep in reps}
            overall = st.mean_ci([m_i[i] for i in common])
            rng_b = random.Random(key_seed(run_seed, split="test", replicate="all", condition="between_arms", level=level, example="all",
                                           branch="bootstrap", purpose="bootstrap", steps=h))
            boots = []
            n = len(common)
            for _ in range(n_boot):
                s = 0.0
                for _ in range(n):
                    s += m_i[common[rng_b.randrange(n)]]
                boots.append(s / n)
            boots.sort()
            lo_i, hi_i = int(math.floor(0.025 * n_boot)), int(math.ceil(0.975 * n_boot)) - 1
            lv[str(h)] = {"per_replicate_mean_d": per_rep, "Theta": overall, "n_examples": n,
                          "bootstrap_percentile_95": {"lo": boots[max(lo_i, 0)], "hi": boots[min(hi_i, n_boot - 1)], "n_boot": n_boot},
                          "replicate_means_sorted": sorted(round(v["mean"], 6) for v in per_rep.values()),
                          "sign_reversal_across_replicates": (min(v["mean"] for v in per_rep.values()) < 0) != (max(v["mean"] for v in per_rep.values()) < 0),
                          "between_replicate_sd_of_means": st.mean_ci([v["mean"] for v in per_rep.values()])["sd"]}
        out[level] = lv
    # solver sensitivity: paired per example, per replicate and arm; sensitivity endpoints vs primary
    sens: Dict[str, Any] = {}
    for rep in reps:
        for arm in ARMS:
            for level in LEVELS:
                qkey = "q_direct_truth_ed_u" if level == "target" else "q_direct_truth_proj_ed_u"
                base_e = _by_example(per_all[f"{rep}|{arm}|{level}|{prim}"], "ed_u")
                base_q = _by_example(per_all[f"{rep}|{arm}|{level}|{prim}"], qkey)
                entry = {}
                for h in steps_all:
                    if h == prim:
                        continue
                    e = paired(_by_example(per_all[f"{rep}|{arm}|{level}|{h}"], "ed_u"), base_e)
                    q = paired(_by_example(per_all[f"{rep}|{arm}|{level}|{h}"], qkey), base_q)
                    entry[f"{h}_minus_{prim}"] = {"gap_ED_U": e, "quality_direct": q,
                                                  "SOLVER_SENSITIVE": e["lo"] is not None and (e["lo"] > 0 or e["hi"] < 0)}
                h1, h2 = [h for h in steps_all if h != prim][:2]
                e12 = paired(_by_example(per_all[f"{rep}|{arm}|{level}|{h2}"], "ed_u"), _by_example(per_all[f"{rep}|{arm}|{level}|{h1}"], "ed_u"))
                entry[f"{h2}_minus_{h1}"] = {"gap_ED_U": e12, "SOLVER_SENSITIVE": e12["lo"] is not None and (e12["lo"] > 0 or e12["hi"] < 0)}
                sens[f"{rep}|{arm}|{level}"] = entry
    out["solver_sensitivity"] = sens
    out["solver_note"] = "paired per-example differences with shared initial noise (no sum of independent SEs); agreement is not a convergence proof; results describe the implemented sampler"
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def data(cfg, r7):
    truth = source(cfg)
    sp_ = cfg["splits"]
    rows = {}
    for split in ("train", "dev"):
        rng = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=split, level="-")))
        rows[split] = draw(truth, sp_["n_" + split], rng)
    rng = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=r7["test_set"]["branch_key"], level="-")))
    rows["test"] = draw(truth, r7["test_set"]["n"], rng)
    scale = {"y": mt.sd([r["y"] for r in rows["train"]]), "z": mt.sd([r["z"] for r in rows["train"]])}
    return truth, rows, scale


def rows_digest(rows) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def write_jsonl(path, recs, mode="a"):
    with open(path, mode) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def evaluate_split(rep: str, arms: Dict[str, Any], cfg, r7, split: str, rows, scale, truth, budget: Budget, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    run_seed = cfg["seed"]
    m, b_perm = r7["evaluation"]["M"], r7["evaluation"]["B"]
    prim = r7["solver"]["primary"]
    steps_list = [prim] + list(r7["solver"]["sensitivity"]) if split == "test" else [prim]
    blocks, per_all = {}, {}
    keep = split == "test"
    for arm, model in arms.items():
        for level in LEVELS:
            for steps in steps_list:
                budget.check()
                do_test = (steps == prim) and (split == "test")
                blk, per, samples = evaluate_block(run_seed, rep, arm, model, split, level, steps, rows, scale, truth, do_test, b_perm, m, budget, keep)
                k = f"{rep}|{arm}|{level}|{steps}"
                blocks[k], per_all[k] = blk, per
                write_jsonl(os.path.join(out_dir, "blocks.jsonl"), [blk])
                write_jsonl(os.path.join(out_dir, "per_example.jsonl"), per)
                if samples:
                    with gzip.open(os.path.join(out_dir, "samples.jsonl.gz"), "at") as f:
                        for smp in samples:
                            f.write(json.dumps({"replicate": rep, "arm": arm, "level": level, "steps": steps, **smp}) + "\n")
    return blocks, per_all


def labels_for(rep: str, train_info: Dict[str, Any], dev_blocks: Dict[str, Any], r7) -> Dict[str, Any]:
    out = {}
    for arm in ARMS:
        ti = train_info[f"{rep}|{arm}"]
        non_finite = (not ti["all_losses_finite"]) or any(b["n_finite"] < b["n_examples"] for k, b in dev_blocks.items() if k.startswith(f"{rep}|{arm}|"))
        out[arm] = {"trained_updates": ti["updates"], "complete": ti["complete"], "UNDERTRAINING_FLAG": ti.get("UNDERTRAINING_FLAG"),
                    "dev_loss_rel_decrease_90_to_100": ti.get("dev_loss_rel_decrease_90_to_100"), "non_finite": non_finite,
                    "label": "MODEL_FIT_OR_NUMERICS_INCONCLUSIVE" if non_finite else "EVALUABLE",
                    "note": "written before the test split of this replicate is opened; no dev test statistic is computed; nothing is selected"}
    return out


def run(cfg, r7, out_root: str, budget: Budget) -> Dict[str, Any]:
    truth, rows, scale = data(cfg, r7)
    assert rows_digest(rows["test"]) == r7["test_set"]["rows_sha256"], "test set digest mismatch with the registered config"
    os.makedirs(out_root, exist_ok=True)
    ckpt_dir = os.path.join(out_root, "checkpoints")
    reps = r7["replicate_ids"]
    status: Dict[str, Any] = {"status": "RUNNING", "replicates": {}, "scale_from_train": scale}
    train_info: Dict[str, Any] = {}
    dev_blocks_all: Dict[str, Any] = {}
    test_blocks_all: Dict[str, Any] = {}
    per_all: Dict[str, List[Dict]] = {}
    labels_all: Dict[str, Any] = {}
    try:
        for rep in reps:
            rs: Dict[str, Any] = {"train": "PENDING", "dev": "PENDING", "test": "PENDING"}
            status["replicates"][rep] = rs
            arms: Dict[str, Any] = {}
            for arm in ARMS:
                budget.check(margin=5.0)
                model, info = train_pair_arm(arm, rep, cfg, r7, rows["train"], rows["dev"], budget, ckpt_dir)
                train_info[f"{rep}|{arm}"] = info
                with open(os.path.join(out_root, "training.json"), "w") as f:
                    json.dump(train_info, f, indent=1)
                if not info["complete"]:
                    raise BudgetExceeded(f"{rep} {arm}: {info['updates']}/{info['updates_planned']} updates ({info.get('stopped_early')})")
                arms[arm] = model
            rs["train"] = "COMPLETE"
            dev_blocks, _ = evaluate_split(rep, arms, cfg, r7, "dev", rows["dev"], scale, truth, budget, os.path.join(out_root, "dev"))
            dev_blocks_all.update(dev_blocks)
            labels_all[rep] = labels_for(rep, train_info, dev_blocks, r7)
            with open(os.path.join(out_root, "dev", "labels.json"), "w") as f:
                json.dump(labels_all, f, indent=1)
            rs["dev"] = "COMPLETE"
            test_blocks, per = evaluate_split(rep, arms, cfg, r7, "test", rows["test"], scale, truth, budget, os.path.join(out_root, "test"))
            test_blocks_all.update(test_blocks)
            per_all.update(per)
            rs["test"] = "COMPLETE"
            status["budget_after_" + rep] = budget.snapshot()
        status["status"] = "COMPLETE"
    except BudgetExceeded as e:
        status["status"] = "INCOMPLETE_BUDGET"
        status["reason"] = str(e)
    done_reps = [rep for rep in reps if status["replicates"].get(rep, {}).get("test") == "COMPLETE"]
    prim = r7["solver"]["primary"]
    steps_all = [prim] + list(r7["solver"]["sensitivity"])
    summary: Dict[str, Any] = {"status": status["status"], "replicates_complete": done_reps, "blocks": test_blocks_all, "dev_blocks": dev_blocks_all,
                               "labels": labels_all}
    if done_reps:
        fam_t = {f"{rep}|{arm}": test_blocks_all[f"{rep}|{arm}|target|{prim}"]["p"] for rep in done_reps for arm in ARMS}
        fam_j = {f"{rep}|{arm}": test_blocks_all[f"{rep}|{arm}|joint|{prim}"]["p"] for rep in done_reps for arm in ARMS}
        fill = lambda d: {k: (v if v is not None else 1.0) for k, v in d.items()}
        summary["families"] = {
            "target_auxiliary": {"raw_p": fam_t, "holm_p": holm(fill(fam_t)), "m": len(reps) * 2, "alpha": 0.05,
                                 "resolution_note": f"B = {r7['evaluation']['B']}: the smallest attainable p is 1/{r7['evaluation']['B'] + 1} = {1 / (r7['evaluation']['B'] + 1):.4f}; the first Holm threshold is 0.05/{len(reps) * 2} = {0.05 / (len(reps) * 2):.5f}, so a minimal p can pass the first step but the family cannot resolve below the grid",
                                 "note": "auxiliary family of the eight target tests at the primary solver; the primary effect is Theta_128, not these p-values"},
            "joint_secondary": {"raw_p": fam_j, "holm_p": holm(fill(fam_j)), "m": len(reps) * 2, "alpha": 0.05,
                                "note": "separate secondary family; not mixed with the target family; the round-6 joint p = 0.035 is kept as an unadjusted secondary result and is not re-scored"}}
        if len(done_reps) == len(reps):
            summary["analysis"] = analysis(per_all, reps, steps_all, prim, cfg["seed"], r7["evaluation"]["n_boot"])
        else:
            summary["analysis"] = {"status": "INCOMPLETE: the planned Theta over four replicates is not computed; completed replicates are listed; no subset is reported as the planned effect"}
        summary["test_non_finite"] = {k: b["n_examples"] - b["n_finite"] for k, b in test_blocks_all.items()}
    summary["budget"] = budget.snapshot()
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    status["training"] = {k: {kk: vv for kk, vv in v.items() if kk != "loss_trace"} for k, v in train_info.items()}
    status["budget"] = budget.snapshot()
    return status


def smoke(cfg, r7, out_dir: str, n_updates: int = 300, run_seed: int = 999) -> Dict[str, Any]:
    """Train-only timing smoke (run seed 999; train rows only); no results are looked at."""
    os.makedirs(out_dir, exist_ok=True)
    budget = Budget(90.0, 400.0)
    truth, rows, scale = data(cfg, r7)
    out: Dict[str, Any] = {"run_seed": run_seed, "n_updates_per_arm": n_updates, "arms": {}}
    steps_all = [r7["solver"]["primary"]] + list(r7["solver"]["sensitivity"])
    for arm in ARMS:
        d = design(cfg, arm)
        model = build_model(arm, cfg, run_seed)
        stage = Stage()
        info = fa.train(model, rows["train"], PATTERNS, n_updates, d["batch"], d["lr"], run_seed, log_every=100, check=lambda s: budget.reason())
        t = stage.done()
        bench = {}
        for steps in steps_all:
            stage = Stage()
            for i, r in enumerate(rows["train"][:2]):
                ctx = Ctx(run_seed, "smoke", "R0", arm, i, "target")
                draw_direct(model, ctx, r["x"], 64, steps)
                draw_sequential(model, ctx, r["x"], 64, steps)
            bench[str(steps)] = stage.done()["cpu_seconds"] / 2  # CPU-s per example (direct + sequential, target level)
        out["arms"][arm] = {"cpu_per_update": t["cpu_seconds"] / n_updates, "cpu_per_example_target_by_steps": bench,
                            "all_losses_finite": info["all_losses_finite"], "stopped": info.get("stopped_early")}
    n_rep = len(r7["replicate_ids"])
    est_train = n_rep * sum(design(cfg, a)["total_updates"] * out["arms"][a]["cpu_per_update"] for a in ARMS)
    # per replicate: test = 200 examples x (target + joint ~ 1.33 x target cost) x all step levels; dev = 200 x primary only
    per_ex_all = sum(sum(out["arms"][a]["cpu_per_example_target_by_steps"].values()) for a in ARMS) * 1.33
    per_ex_prim = sum(out["arms"][a]["cpu_per_example_target_by_steps"][str(r7["solver"]["primary"])] for a in ARMS) * 1.33
    est_sampling = n_rep * 200 * (per_ex_all + per_ex_prim)
    est_perm = n_rep * 2 * (1.1 + 4.3)
    est_other = n_rep * 20.0
    out["estimate_cpu_seconds"] = {"train": est_train, "sampling": est_sampling, "permutation": est_perm, "other": est_other,
                                   "total": est_train + est_sampling + est_perm + est_other}
    out["smoke_cost"] = budget.snapshot()
    with open(os.path.join(out_dir, "smoke.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out
