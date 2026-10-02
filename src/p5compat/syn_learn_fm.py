"""P5-SYN-LEARN-01 flow-matching arms (round 6).

Contract: configs/p5_syn_learn_01.json (base), amendments A1 and A2. Trains
INDEPENDENT_CONDITIONAL_FM and SHARED_CONDITIONAL_FM once on the train split,
evaluates the dev split (labels written to disk), then the test split. Both
solver levels (128 primary, 32 secondary) share the same noise draws; the
conditioning example is the unit of every interval. Runs under a CPU/wall
budget guard and reports INCOMPLETE_BUDGET instead of partial results.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import resource
import time
from typing import Any, Dict, List, Optional

from . import fm_adapter as fa
from . import gmm as gm
from . import metrics as mt
from . import permutation as pm
from . import samplers as sp
from . import stats as st
from .analytic import e_abs_normal
from .seeds import seed_v2
from .syn_learn import NAMES, _key, draw, holm, source

ARMS = ("INDEPENDENT_CONDITIONAL_FM", "SHARED_CONDITIONAL_FM")
PATTERNS = list(fa.PATTERNS)
LEVELS = ("target", "joint")


class BudgetExceeded(Exception):
    pass


class Budget:
    """Process CPU seconds (all threads; no children are spawned) and wall seconds against the caps."""

    def __init__(self, cpu_cap: float, wall_cap: float, cpu_spent_before: float = 0.0):
        self.cpu_cap, self.wall_cap, self.before = cpu_cap, wall_cap, cpu_spent_before
        self.w0, self.c0 = time.time(), time.process_time()

    def cpu(self) -> float:
        return self.before + time.process_time() - self.c0

    def wall(self) -> float:
        return time.time() - self.w0

    def reason(self, margin: float = 0.0) -> Optional[str]:
        if self.cpu() + margin > self.cpu_cap:
            return f"cpu {self.cpu():.0f}+{margin:.0f} s > cap {self.cpu_cap:.0f} s"
        if self.wall() + margin > self.wall_cap:
            return f"wall {self.wall():.0f}+{margin:.0f} s > cap {self.wall_cap:.0f} s"
        return None

    def check(self, margin: float = 0.0):
        r = self.reason(margin)
        if r:
            raise BudgetExceeded(r)

    def snapshot(self) -> Dict[str, float]:
        ch = resource.getrusage(resource.RUSAGE_CHILDREN)
        return {"cpu_seconds_total_incl_before": self.cpu(), "cpu_seconds_this_process": time.process_time() - self.c0,
                "wall_seconds": self.wall(), "children_cpu_seconds": ch.ru_utime + ch.ru_stime}


class Stage:
    def __init__(self):
        self.w0, self.c0 = time.time(), time.process_time()

    def done(self) -> Dict[str, float]:
        return {"wall_seconds": time.time() - self.w0, "cpu_seconds": time.process_time() - self.c0}


# ---------------------------------------------------------------------------
# Models and training
# ---------------------------------------------------------------------------


def design(cfg, arm: str) -> Dict[str, Any]:
    """Design fields of an arm. The base config states batch and optimizer only under the
    independent arm; A1 'reused_unchanged' fixes the same Adam lr 1e-3 and batch 256 for both arms."""
    d = dict(cfg["models"][arm]["design"])
    ind = cfg["models"]["INDEPENDENT_CONDITIONAL_FM"]["design"]
    d.setdefault("batch", ind["batch"])
    d.setdefault("optimizer", ind["optimizer"])
    d["lr"] = float(d["optimizer"].split()[-1])
    return d


def build_model(arm: str, cfg: Dict[str, Any], seed: int):
    d = design(cfg, arm)
    cls = fa.IndependentConditionalFM if arm == "INDEPENDENT_CONDITIONAL_FM" else fa.SharedConditionalFM
    return cls(d["mlp_width"], d["mlp_depth"], seed)


def arm_seed(cfg, arm: str, branch: str) -> int:
    return seed_v2(cfg["seed"], **_key(condition=arm, level="-", example="all", replicate=0, branch=branch))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(model, arm: str, info: Dict[str, Any], path: str) -> Dict[str, Any]:
    fa.torch.save({"arm": arm, "state_dict": model.state_dict(), "updates": info["updates"], "updates_planned": info["updates_planned"],
                   "torch_version": fa.torch.__version__}, path)
    return {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(path)))), "sha256": sha256_file(path),
            "bytes": os.path.getsize(path), "updates": info["updates"], "final_iterate": info["updates"] == info["updates_planned"]}


def train_arm(arm: str, cfg, a2, train_rows, dev_rows, budget: Budget, ckpt_dir: str, n_updates: Optional[int] = None,
              run_seed_override: Optional[int] = None) -> Dict[str, Any]:
    """Train one arm for the fixed number of updates; final iterate only. The dev
    flow-matching loss (fixed noise) is recorded at 90% and 100% as a diagnostic."""
    cfg_seed = dict(cfg, seed=run_seed_override if run_seed_override is not None else cfg["seed"])
    d = design(cfg, arm)
    total = n_updates if n_updates is not None else d["total_updates"]
    lr = d["lr"]
    model = build_model(arm, cfg, arm_seed(cfg_seed, arm, "model_init"))
    dev_noise_seed = arm_seed(cfg_seed, arm, "dev_loss")
    dev_losses: Dict[int, Dict[str, float]] = {}
    hook_at = sorted({int(round(0.9 * total)), total})

    def hook(step, m):
        dev_losses[step] = {p: fa.held_out_fm_loss(m, p, dev_rows, dev_noise_seed) for p in PATTERNS}

    def check(step):
        return budget.reason(margin=0.0)

    stage = Stage()
    info = fa.train(model, train_rows, PATTERNS, total, d["batch"], lr, arm_seed(cfg_seed, arm, "train_stream"),
                    log_every=1000, hook=hook, hook_at=hook_at, check=check)
    info["timing"] = stage.done()
    info["dev_fm_loss_fixed_noise"] = {str(k): v for k, v in dev_losses.items()}
    if len(dev_losses) == 2:
        k90, k100 = hook_at
        rel = {p: (dev_losses[k90][p] - dev_losses[k100][p]) / dev_losses[k90][p] for p in PATTERNS}
        info["dev_loss_rel_decrease_90_to_100"] = rel
        info["UNDERTRAINING_FLAG"] = any(v > a2["dev_loss_rule"]["threshold"] for v in rel.values())
    info["seeds"] = {"model_init": arm_seed(cfg_seed, arm, "model_init"), "train_stream": arm_seed(cfg_seed, arm, "train_stream"),
                     "dev_loss_noise": dev_noise_seed}
    info["design"] = d
    os.makedirs(ckpt_dir, exist_ok=True)
    info["checkpoint"] = save_checkpoint(model, arm, info, os.path.join(ckpt_dir, f"{arm}.pt"))
    info["complete"] = info["updates"] == total and info["all_losses_finite"]
    return model, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def branches(model, steps: int):
    direct = sp.Branch("direct", fa.FMSampler(model, "z|x", steps))
    seq = sp.Branch("sequential", fa.FMSampler(model, "z|x,y", steps), fa.FMSampler(model, "y|x", steps))
    djoint = sp.Branch("direct_joint", fa.FMSampler(model, "y,z|x", steps))
    return direct, seq, djoint


def sample_vs_mixture_ed_u(a: List[float], mix) -> float:
    """Unbiased energy distance between samples a and a 1-D Gaussian mixture
    (closed-form cross and mixture terms, U-statistic within-sample term)."""
    n = len(a)
    cross = sum(sum(w * e_abs_normal(v - m, s) for w, m, s in mix) for v in a) / n
    within = 2.0 * mt.sum_abs_pairs(a) / (n * (n - 1))
    return 2.0 * cross - within - gm.mix_e_abs(mix, mix)


def _finite(vals) -> bool:
    return all(math.isfinite(v) for v in vals)


def evaluate_block(arm: str, model, cfg, split: str, level: str, steps: int, rows, scale, truth, budget: Budget,
                   keep_samples: bool):
    """One (arm, split, level, steps) block: per-example statistics and Test P."""
    m, b_perm = 64, 199
    sy, sz = scale["y"], scale["z"]
    dirs = mt.slice_directions(8)
    direct, seq, djoint = branches(model, steps)
    per, exs, samples = [], [], []
    stage = Stage()
    for i, ex in enumerate(rows):
        x = ex["x"]
        e = {"x": x}
        # Sampling keys omit the step count on purpose (A2): both solver levels share the noise.
        rng_a = random.Random(seed_v2(cfg["seed"], **_key(condition=arm, level=level, example=i, replicate=0, branch="direct")))
        rng_c = random.Random(seed_v2(cfg["seed"], **_key(condition=arm, level=level, example=i, replicate=0, branch="sequential")))
        oc = seq.sample(rng_c, e, m)
        rec: Dict[str, Any] = {"arm": arm, "split": split, "level": level, "steps": steps, "example": i, "x": x,
                               "n_distinct_y_seq": len(set(oc["y"]))}
        if level == "target":
            za = direct.sample(rng_a, e, m)["z"]
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
                rec["mean_direct_minus_truth_std"] = (mt.mean(za) - tm) / math.sqrt(tv)
                exs.append(pm.PooledExample(a_s, c_s))
                if keep_samples:
                    samples.append({"example": i, "direct_z": [round(v, 6) for v in za], "seq_y": [round(v, 6) for v in oc["y"]],
                                    "seq_z": [round(v, 6) for v in oc["z"]]})
        else:
            oa = djoint.sample(rng_a, e, m)
            ok = _finite(oa["y"]) and _finite(oa["z"]) and _finite(oc["y"]) and _finite(oc["z"])
            rec["finite"] = ok
            if ok:
                pa = [[u / sy, v / sz] for u, v in zip(oa["y"], oa["z"])]
                pc = [[u / sy, v / sz] for u, v in zip(oc["y"], oc["z"])]
                tc = truth.conditional_components(["y", "z"], ["x"], [x])
                edu, edv, qq = 0.0, 0.0, 0.0
                for u in dirs:
                    proj_a = [u[0] * p[0] + u[1] * p[1] for p in pa]
                    proj_c = [u[0] * p[0] + u[1] * p[1] for p in pc]
                    edu += mt.energy_distance_1d(proj_a, proj_c, unbiased=True)
                    edv += mt.energy_distance_1d(proj_a, proj_c, unbiased=False)
                    qq += sample_vs_mixture_ed_u(proj_a, gm.project_2d(tc, [u[0] / sy, u[1] / sz], [x]))
                rec["ed_u"], rec["ed_v"] = edu / len(dirs), edv / len(dirs)
                rec["q_direct_truth_proj_ed_u"] = qq / len(dirs)
                exs.append(pm.PooledExample(pa, pc, dirs))
                if keep_samples:
                    samples.append({"example": i, "direct_y": [round(v, 6) for v in oa["y"]], "direct_z": [round(v, 6) for v in oa["z"]],
                                    "seq_y": [round(v, 6) for v in oc["y"]], "seq_z": [round(v, 6) for v in oc["z"]]})
        per.append(rec)
        if i % 50 == 49:
            budget.check()
    t_sampling = stage.done()
    stage = Stage()
    n_finite = sum(1 for r in per if r["finite"])
    res = None
    if n_finite >= 2:
        rng_p = random.Random(seed_v2(cfg["seed"], **_key(condition=arm, level=level, example="all", replicate=0,
                                                           branch="permutation", solver=steps)))
        res = pm.mc_permutation_test(exs, b_perm, rng_p)
    t_test = stage.done()
    edu = [r["ed_u"] for r in per if r["finite"]]
    blk = {"arm": arm, "split": split, "level": level, "steps": steps, "n_examples": len(rows), "n_finite": n_finite,
           "non_finite_examples": [r["example"] for r in per if not r["finite"]], "M": m, "B": b_perm,
           "p": res["p"] if res else None, "t_obs": res["t_obs"] if res else None,
           "effect_ED_U": st.mean_ci(edu), "min_distinct_intermediates": min(r["n_distinct_y_seq"] for r in per),
           "timing_sampling": t_sampling, "timing_test": t_test}
    if level == "target":
        blk["quality_direct_ED_U_to_truth"] = st.mean_ci([r["q_direct_truth_ed_u"] for r in per if r["finite"]])
        blk["quality_seq_ED_U_to_truth"] = st.mean_ci([r["q_seq_truth_ed_u"] for r in per if r["finite"]])
        blk["sd_ratio_direct"] = st.mean_ci([r["sd_ratio_direct"] for r in per if r["finite"]])
        blk["mean_shift_direct_std"] = st.mean_ci([r["mean_direct_minus_truth_std"] for r in per if r["finite"]])
    else:
        blk["quality_direct_projected_ED_U_to_truth"] = st.mean_ci([r["q_direct_truth_proj_ed_u"] for r in per if r["finite"]])
    return blk, per, samples


def paired_diff(per_a: List[Dict], per_b: List[Dict], key: str) -> Dict[str, Any]:
    """Paired per-example differences a - b over examples finite in both."""
    da = {r["example"]: r[key] for r in per_a if r.get("finite")}
    db = {r["example"]: r[key] for r in per_b if r.get("finite")}
    common = sorted(set(da) & set(db))
    d = [da[i] - db[i] for i in common]
    out = st.mean_ci(d)
    out["n_pairs"] = len(common)
    return out


def bootstrap_mean_ci(d: List[float], rng: random.Random, n_boot: int = 2000, level: float = 0.95) -> Dict[str, Any]:
    n = len(d)
    if n < 2:
        return {"n_boot": n_boot, "lo": None, "hi": None}
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_i, hi_i = int(math.floor((1 - level) / 2 * n_boot)), int(math.ceil((1 + level) / 2 * n_boot)) - 1
    return {"n_boot": n_boot, "lo": means[max(lo_i, 0)], "hi": means[min(hi_i, n_boot - 1)], "method": "percentile over examples"}


def evaluate_split(arms: Dict[str, Any], cfg, a2, split: str, rows, scale, truth, budget: Budget, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    levels_steps = [(lv, s) for lv in LEVELS for s in (a2["solver_sensitivity"]["levels"]["primary"], a2["solver_sensitivity"]["levels"]["secondary"])]
    blocks: Dict[str, Dict[str, Any]] = {}
    per_all: Dict[str, List[Dict]] = {}
    keep = split == "test"
    with open(os.path.join(out_dir, "per_example.jsonl"), "w") as fper, \
            (gzip.open(os.path.join(out_dir, "samples.jsonl.gz"), "wt") if keep else open(os.devnull, "w")) as fsam:
        for arm, model in arms.items():
            for level, steps in levels_steps:
                budget.check()
                blk, per, samples = evaluate_block(arm, model, cfg, split, level, steps, rows, scale, truth, budget, keep)
                blocks[f"{arm}|{level}|{steps}"] = blk
                per_all[f"{arm}|{level}|{steps}"] = per
                for r in per:
                    fper.write(json.dumps(r) + "\n")
                for smp in samples:
                    fsam.write(json.dumps({"arm": arm, "level": level, "steps": steps, **smp}) + "\n")
    prim, sec = a2["solver_sensitivity"]["levels"]["primary"], a2["solver_sensitivity"]["levels"]["secondary"]
    solver = {}
    for arm in arms:
        for level in LEVELS:
            dd = paired_diff(per_all[f"{arm}|{level}|{prim}"], per_all[f"{arm}|{level}|{sec}"], "ed_u")
            qkey = "q_direct_truth_ed_u" if level == "target" else "q_direct_truth_proj_ed_u"
            dq = paired_diff(per_all[f"{arm}|{level}|{prim}"], per_all[f"{arm}|{level}|{sec}"], qkey)
            sens = dd["lo"] is not None and (dd["lo"] > 0 or dd["hi"] < 0)
            solver[f"{arm}|{level}"] = {"paired_ED_U_fine_minus_coarse": dd, "paired_quality_fine_minus_coarse": dq,
                                        "SOLVER_SENSITIVE": sens, "numerics_label": "NUMERICS_NOT_SEPARATED" if sens else "not flagged"}
    between = {}
    if len(arms) == 2:
        sh, ind = "SHARED_CONDITIONAL_FM", "INDEPENDENT_CONDITIONAL_FM"
        for level in LEVELS:
            for steps in (prim, sec):
                pa, pb = per_all[f"{sh}|{level}|{steps}"], per_all[f"{ind}|{level}|{steps}"]
                dd = paired_diff(pa, pb, "ed_u")
                da = {r["example"]: r["ed_u"] for r in pa if r.get("finite")}
                db = {r["example"]: r["ed_u"] for r in pb if r.get("finite")}
                d = [da[i] - db[i] for i in sorted(set(da) & set(db))]
                rng_b = random.Random(seed_v2(cfg["seed"], **_key(condition="between_arms", level=level, example="all", replicate=0,
                                                                   branch="bootstrap", solver=steps, split=split)))
                between[f"{level}|{steps}"] = {"effect": "ED_U(SHARED) - ED_U(INDEPENDENT) per example; negative = shared smaller gap",
                                               "paired_normal": dd, "bootstrap": bootstrap_mean_ci(d, rng_b, a2["evaluation"]["between_arm_effect"]["n_boot"]),
                                               "primary": level == "target" and steps == prim and split == "test"}
    fam = {arm: blocks[f"{arm}|target|{prim}"]["p"] for arm in arms}
    fam_ok = {k: v for k, v in fam.items() if v is not None}
    summary = {"split": split, "blocks": blocks, "solver_sensitivity": solver, "between_arms": between,
               "primary_family": {"tests": sorted(fam_ok), "raw_p": fam_ok, "holm_p": holm(fam_ok) if fam_ok else {}, "alpha": 0.05,
                                  "definition": f"FM target-level Test P at {prim} steps on this split; Holm; GMM controls excluded",
                                  "is_primary_split": split == "test"},
               "budget_after": budget.snapshot()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def arm_labels(train_info: Dict[str, Any], dev_summary: Dict[str, Any], arm: str, a2) -> Dict[str, Any]:
    prim = a2["solver_sensitivity"]["levels"]["primary"]
    non_finite = (not train_info["all_losses_finite"]) or any(
        blk["n_finite"] < blk["n_examples"] for k, blk in dev_summary["blocks"].items() if k.startswith(arm + "|"))
    lab = {"arm": arm, "trained_updates": train_info["updates"], "complete": train_info["complete"],
           "UNDERTRAINING_FLAG": train_info.get("UNDERTRAINING_FLAG"), "dev_loss_rel_decrease_90_to_100": train_info.get("dev_loss_rel_decrease_90_to_100"),
           "non_finite": non_finite,
           "dev_SOLVER_SENSITIVE_target": dev_summary["solver_sensitivity"][f"{arm}|target"]["SOLVER_SENSITIVE"],
           "dev_SOLVER_SENSITIVE_joint": dev_summary["solver_sensitivity"][f"{arm}|joint"]["SOLVER_SENSITIVE"],
           "dev_target_p_primary_level": dev_summary["blocks"][f"{arm}|target|{prim}"]["p"],
           "label": "MODEL_FIT_OR_NUMERICS_INCONCLUSIVE" if non_finite else "EVALUABLE",
           "note": "labels fixed before the test split is opened; the dev p-value is descriptive and chooses nothing"}
    return lab


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def data(cfg):
    truth = source(cfg)
    sp_ = cfg["splits"]
    rows = {}
    for split in ("train", "dev", "test"):
        rng = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=split, level="-")))
        rows[split] = draw(truth, sp_["n_" + split], rng)
    scale = {"y": mt.sd([r["y"] for r in rows["train"]]), "z": mt.sd([r["z"] for r in rows["train"]])}
    return truth, rows, scale


def smoke(cfg, a2, out_dir: str, n_updates: int = 500, run_seed: int = 999) -> Dict[str, Any]:
    """Train-only timing smoke (run seed 999; train rows only); estimates the full plan's CPU cost."""
    os.makedirs(out_dir, exist_ok=True)
    budget = Budget(60.0, 300.0)
    truth, rows, scale = data(cfg)
    out: Dict[str, Any] = {"run_seed": run_seed, "n_updates_per_arm": n_updates, "arms": {}}
    for arm in ARMS:
        d = design(cfg, arm)
        model = build_model(arm, cfg, arm_seed(dict(cfg, seed=run_seed), arm, "model_init"))
        stage = Stage()
        info = fa.train(model, rows["train"], PATTERNS, n_updates, d["batch"], d["lr"], run_seed, log_every=100)
        t = stage.done()
        # Sampling benchmark on train conditioning values: 4 examples x 64 samples x 128 steps per pattern.
        stage = Stage()
        n_passes = 0
        for pat in PATTERNS:
            smp = fa.FMSampler(model, pat, 128)
            for r in rows["train"][:4]:
                smp.sample_batch(random.Random(run_seed), [[r[v] for v in fa.PATTERNS[pat][0]]] * 64)
                n_passes += 128
        ts = stage.done()
        out["arms"][arm] = {"cpu_per_update": t["cpu_seconds"] / n_updates, "wall_per_update": t["wall_seconds"] / n_updates,
                            "cpu_per_forward_pass_b64": ts["cpu_seconds"] / n_passes, "n_params": info["n_params"],
                            "all_losses_finite": info["all_losses_finite"], "loss_trace": info["loss_trace"]}
    # Full-plan estimate: training + sampling passes + permutation tests (GMM-run timings) + bootstrap
    est_train = sum(design(cfg, a)["total_updates"] * out["arms"][a]["cpu_per_update"] for a in ARMS)
    passes_per_arm_split = 1200 * (a2["solver_sensitivity"]["levels"]["primary"] + a2["solver_sensitivity"]["levels"]["secondary"])
    est_sampling = sum(2 * passes_per_arm_split * out["arms"][a]["cpu_per_forward_pass_b64"] for a in ARMS)
    est_perm = 2 * 2 * 2 * (1.0 + 3.5)  # arms x splits x steps x (target + joint), from the round-4 GMM timings
    est_other = 30.0  # per-example statistics, projections, bootstrap, dev losses
    out["estimate_cpu_seconds"] = {"train": est_train, "sampling": est_sampling, "permutation": est_perm, "other": est_other,
                                   "total": est_train + est_sampling + est_perm + est_other}
    # Dry run of the evaluation code path on 3 TRAIN conditioning values with tiny step counts (numbers discarded).
    a2_dry = json.loads(json.dumps(a2))
    a2_dry["solver_sensitivity"]["levels"].update({"primary": 4, "secondary": 2})
    a2_dry["evaluation"]["between_arm_effect"]["n_boot"] = 50
    models = {arm: build_model(arm, cfg, arm_seed(dict(cfg, seed=run_seed), arm, "model_init")) for arm in ARMS}
    dry = evaluate_split(models, cfg, a2_dry, "test", rows["train"][:3], scale, truth, budget, os.path.join(out_dir, "dry_eval"))
    fake_train = {"all_losses_finite": True, "updates": 0, "complete": False, "updates_planned": 0}
    out["dry_eval_labels"] = {arm: arm_labels(fake_train, dry, arm, a2_dry) for arm in ARMS}
    out["dry_eval_blocks"] = sorted(dry["blocks"])
    out["smoke_cost"] = budget.snapshot()
    with open(os.path.join(out_dir, "smoke.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def run(cfg, a2, out_root: str, budget: Budget) -> Dict[str, Any]:
    """Train both arms, evaluate dev (labels), then test. Returns the manifest body."""
    truth, rows, scale = data(cfg)
    status: Dict[str, Any] = {"status": "RUNNING", "stages": {}}
    ckpt_dir = os.path.join(out_root, "p5_syn_learn_01_fm_train", "checkpoints")
    train_dir = os.path.join(out_root, "p5_syn_learn_01_fm_train")
    os.makedirs(train_dir, exist_ok=True)
    arms: Dict[str, Any] = {}
    train_info: Dict[str, Any] = {}
    try:
        for arm in ARMS:
            budget.check(margin=5.0)
            model, info = train_arm(arm, cfg, a2, rows["train"], rows["dev"], budget, ckpt_dir)
            train_info[arm] = info
            with open(os.path.join(train_dir, "training.json"), "w") as f:
                json.dump(train_info, f, indent=1)
            if not info["complete"]:
                raise BudgetExceeded(f"{arm}: {info['updates']}/{info['updates_planned']} updates ({info.get('stopped_early')})")
            arms[arm] = model
        status["stages"]["train"] = "COMPLETE"
        dev = evaluate_split(arms, cfg, a2, "dev", rows["dev"], scale, truth, budget, os.path.join(out_root, "p5_syn_learn_01_fm_dev"))
        labels = {arm: arm_labels(train_info[arm], dev, arm, a2) for arm in ARMS}
        with open(os.path.join(out_root, "p5_syn_learn_01_fm_dev", "labels.json"), "w") as f:
            json.dump(labels, f, indent=1)
        status["stages"]["dev"] = "COMPLETE"
        status["labels"] = labels
        test = evaluate_split(arms, cfg, a2, "test", rows["test"], scale, truth, budget, os.path.join(out_root, "p5_syn_learn_01_fm_test"))
        status["stages"]["test"] = "COMPLETE"
        status["status"] = "COMPLETE"
        status["test_primary_family"] = test["primary_family"]
    except BudgetExceeded as e:
        status["status"] = "INCOMPLETE_BUDGET"
        status["reason"] = str(e)
    status["training"] = {a: {k: v for k, v in i.items() if k != "loss_trace"} for a, i in train_info.items()}
    status["budget"] = budget.snapshot()
    status["scale_from_train"] = scale
    return status
