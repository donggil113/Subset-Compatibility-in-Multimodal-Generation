"""P5-SYN-LEARN-01 runner (config: configs/p5_syn_learn_01.json).

Runs the JOINT_COHERENT_CONTROL models (true mixture, EM-fitted mixture) with
Test P on the target (primary) and projected-joint (secondary) endpoints, plus
closed-form quality to the truth. Flow-matching arms are recorded as NOT_RUN
when no tensor library is available.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List

from . import gmm as gm
from . import metrics as mt
from . import permutation as pm
from . import samplers as sp
from . import stats as st
from .seeds import seed_v2

EXP = "P5-SYN-LEARN-01"
NAMES = ["x", "y", "z"]


def _key(**kw):
    base = dict(experiment=EXP, joint="mixture3")
    base.update(kw)
    return base


def source(cfg) -> gm.GMM:
    s = cfg["source_distribution"]
    return gm.GMM.from_sd_corr(NAMES, s["weights"], s["means"], s["sds"], s["corrs"])


def draw(g: gm.GMM, n: int, rng: random.Random) -> List[Dict[str, float]]:
    return [g.sample(rng) for _ in range(n)]


def branches(model: gm.GMM):
    direct = sp.Branch("direct", gm.GMMConditionalSampler(model, ["z"], ["x"]))
    seq = sp.Branch("sequential", gm.GMMConditionalSampler(model, ["z"], ["x", "y"]), gm.GMMConditionalSampler(model, ["y"], ["x"]))
    djoint = sp.Branch("direct_joint", gm.GMMConditionalSampler(model, ["y", "z"], ["x"]))
    return direct, seq, djoint


def evaluate(cfg, name: str, model: gm.GMM, truth: gm.GMM, rows, scale) -> Dict[str, Any]:
    ev = cfg["evaluation"]
    n_ex, m, b_perm = 200, 64, 199
    sy, sz = scale["y"], scale["z"]
    dirs = mt.slice_directions(8)
    direct, seq, djoint = branches(model)
    out = {}
    for level in ("target", "joint"):
        t0 = time.time()
        exs, edu, sdr = [], [], []
        min_distinct = None
        q_t, q_j = [], []
        for i, ex in enumerate(rows[:n_ex]):
            rng_a = random.Random(seed_v2(cfg["seed"], **_key(condition=name, level=level, example=i, replicate=0, branch="direct")))
            rng_c = random.Random(seed_v2(cfg["seed"], **_key(condition=name, level=level, example=i, replicate=0, branch="sequential")))
            e = {"x": ex["x"]}
            oc = seq.sample(rng_c, e, m)
            d = len(set(oc["y"]))
            min_distinct = d if min_distinct is None else min(min_distinct, d)
            x = ex["x"]
            if level == "target":
                za = direct.sample(rng_a, e, m)["z"]
                a_s, c_s = [v / sz for v in za], [v / sz for v in oc["z"]]
                exs.append(pm.PooledExample(a_s, c_s))
                edu.append(mt.energy_distance_1d(a_s, c_s, unbiased=True))
                tm, tv = gm.mix_moments(truth.conditional_1d("z", ["x"], [x]))
                sdr.append(mt.sd(za) / math.sqrt(tv))
                q_t.append(gm.mix_energy_distance(gm.scale_mixture(model.conditional_1d("z", ["x"], [x]), sz),
                                                  gm.scale_mixture(truth.conditional_1d("z", ["x"], [x]), sz)))
            else:
                oa = djoint.sample(rng_a, e, m)
                pa = [[u / sy, v / sz] for u, v in zip(oa["y"], oa["z"])]
                pc = [[u / sy, v / sz] for u, v in zip(oc["y"], oc["z"])]
                exs.append(pm.PooledExample(pa, pc, dirs))
                edu.append(sum(mt.energy_distance_1d([u[0] * p[0] + u[1] * p[1] for p in pa],
                                                     [u[0] * p[0] + u[1] * p[1] for p in pc], unbiased=True) for u in dirs) / len(dirs))
                mc = model.conditional_components(["y", "z"], ["x"], [x])
                tc = truth.conditional_components(["y", "z"], ["x"], [x])
                qq = 0.0
                for u in dirs:
                    us = [u[0] / sy, u[1] / sz]
                    qq += gm.mix_energy_distance(gm.project_2d(mc, us, [x]), gm.project_2d(tc, us, [x]))
                q_j.append(qq / len(dirs))
        rng_p = random.Random(seed_v2(cfg["seed"], **_key(condition=name, level=level, example="all", replicate=0, branch="permutation")))
        t1 = time.time()
        res = pm.mc_permutation_test(exs, b_perm, rng_p)
        t2 = time.time()
        blk = {"p": res["p"], "t_obs": res["t_obs"], "effect_ED_U": st.mean_ci(edu), "n_examples": len(exs), "M": m, "B": b_perm,
               "min_distinct_intermediates": min_distinct, "seconds_sampling": t1 - t0, "seconds_test": t2 - t1}
        if level == "target":
            blk["quality_ED_to_truth"] = st.mean_ci(q_t)
            blk["quality_CRPS_divergence"] = st.mean_ci([v / 2 for v in q_t])
            blk["sd_ratio_direct"] = st.mean_ci(sdr)
        else:
            blk["quality_projected_ED_to_truth"] = st.mean_ci(q_j)
        out[level] = blk
    return out


def holm(ps: Dict[str, float]) -> Dict[str, float]:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for r, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        adj[k] = run
    return adj


def run(cfg: Dict[str, Any], out_dir: str, split: str = "test") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.time()
    truth = source(cfg)
    sp_ = cfg["splits"]
    rng_tr = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch="train", level="-")))
    rng_ev = random.Random(seed_v2(cfg["seed"], **_key(condition="data", example="all", replicate=0, branch=split, level="-")))
    train = draw(truth, sp_["n_train"], rng_tr)
    rows = draw(truth, sp_["n_" + split], rng_ev)
    scale = {"y": mt.sd([r["y"] for r in train]), "z": mt.sd([r["z"] for r in train])}
    timings = {}
    t0 = time.time()
    fcfg = cfg["models"]["FITTED_JOINT_CONTROL"]
    rng_m = random.Random(seed_v2(cfg["seed"], **_key(condition="FITTED_JOINT_CONTROL", example="all", replicate=0, branch="model_init", level="-")))
    fitted, diag = gm.fit_gmm_em([[r[n] for n in NAMES] for r in train], NAMES, fcfg["K_fit"], rng_m,
                                 max_iter=fcfg["em"]["max_iter"], tol=fcfg["em"]["tol"], reg=fcfg["em"]["reg"])
    timings["fit_FITTED_JOINT_CONTROL_seconds"] = time.time() - t0
    heldout = {"true_mean_loglik": sum(truth.logpdf([r[n] for n in NAMES]) for r in rows) / len(rows),
               "fitted_mean_loglik": sum(fitted.logpdf([r[n] for n in NAMES]) for r in rows) / len(rows)}
    results = {}
    for name, model in (("TRUE_JOINT_CONTROL", truth), ("FITTED_JOINT_CONTROL", fitted)):
        t0 = time.time()
        results[name] = evaluate(cfg, name, model, truth, rows, scale)
        timings[f"eval_{name}_seconds"] = time.time() - t0
    for name in ("INDEPENDENT_CONDITIONAL_FM", "SHARED_CONDITIONAL_FM"):
        results[name] = {"status": "NOT_RUN", "blocker": cfg["models"][name]["blocker"]}
    prim = {k: v["target"]["p"] for k, v in results.items() if "target" in v}
    summary = {
        "split": split, "scale_from_train": scale, "fitted_model": {
            "K": fitted.K, "weights": fitted.weights, "means": [c.mu for c in fitted.comps], "em": diag},
        "heldout_loglik": heldout, "results": results,
        "primary_family": {"tests": sorted(prim), "raw_p": prim, "holm_p": holm(prim), "alpha": 0.05},
        "timings": timings, "total_wall_seconds": time.time() - t_all}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
