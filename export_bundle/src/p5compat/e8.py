"""P5-E8-EXACT: small exact randomisation verification of the within-example
label-permutation test (see permutation.py for the design).

Components (all pre-registered in configs/p5_e8_exact_v1.json):
  A  group super-uniformity by full enumeration (implementation check, incl. ties)
  B  Monte Carlo p-value vs exact p-value agreement
  C  unconditional rejection rates with exact p-values under H0 and fixed
     alternatives, plus always-accept / always-reject negative-control detectors
  D  timing smoke of the Monte Carlo test at a small realistic size (no inference)

Every dataset uses fresh conditioning rows and v2 seeds, so datasets are
independent units; permutations are never counted as units.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, List

from . import metrics as mt
from . import permutation as pm
from . import samplers as sp
from . import stats as st
from .experiment import build_condition, exact, make_joint
from .seeds import seed_v2

EXP = "P5-E8-EXACT"


def _rows(joint_obj, n, run_seed, **key):
    rng = random.Random(seed_v2(run_seed, experiment=EXP, example="all", branch="rows", **key))
    rows = []
    for _ in range(n):
        s = joint_obj.sample(rng)
        rows.append({"x": s["x"], "y_obs": s["y"], "z_obs": s["z"]})
    return rows


def make_dataset(joint_obj, tc, cond, n, m, run_seed, jname, component, design, replicate, level, scale):
    """Return list of PooledExample for one independent dataset."""
    ref, cmp_branch, _, _ = build_condition(tc, cond)
    key = dict(joint=jname, condition=cond["id"], replicate=replicate, component=component, design=design)
    rows = _rows(joint_obj, n, run_seed, **key)
    if level == "joint":
        if cond["kind"] == "collapse_both":
            jref = sp.Branch("direct_joint", exact(sp.collapsed(tc.yz_x)))
        else:
            jref = sp.Branch("direct_joint", exact(tc.yz_x))
    exs = []
    for i, ex in enumerate(rows):
        rng_a = random.Random(seed_v2(run_seed, experiment=EXP, example=i, branch="direct", **key))
        rng_c = random.Random(seed_v2(run_seed, experiment=EXP, example=i, branch="sequential", **key))
        oc = cmp_branch.sample(rng_c, ex, m)
        if level == "target":
            za = ref.sample(rng_a, ex, m)["z"]
            exs.append(pm.PooledExample([v / scale["z"] for v in za], [v / scale["z"] for v in oc["z"]]))
        else:
            oa = jref.sample(rng_a, ex, m)
            pa = [[u / scale["y"], v / scale["z"]] for u, v in zip(oa["y"], oa["z"])]
            pc = [[u / scale["y"], v / scale["z"]] for u, v in zip(oc["y"], oc["z"])]
            exs.append(pm.PooledExample(pa, pc, mt.slice_directions(8)))
    return exs


def rate_block(k, n, alpha):
    lo = st.clopper_pearson(k, n, 0.95, "lower")["lo"]
    hi = st.clopper_pearson(k, n, 0.95, "upper")["hi"]
    return {"k": k, "n": n, "rate": k / n if n else None, "cp_lower_95_one_sided": lo, "cp_upper_95_one_sided": hi,
            "excess_flag": lo > alpha, "powered_flag": lo > alpha}


def run(config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    t_all = time.time()
    run_seed = config["seed"]
    alpha = config["alpha"]
    jname = config["joint"]
    joint = make_joint(config["joints"][jname])
    tc = sp.true_conditionals(joint)
    scale = config["standardisation_sd"]
    conds = {c["id"]: c for c in config["conditions"]}
    results: Dict[str, Any] = {}
    raw: List[Dict[str, Any]] = []
    status: Dict[str, str] = {}
    budget = config["budget"]["max_seconds"]

    def over():
        return time.time() - t_all > budget

    # ---- A: group super-uniformity -------------------------------------------------
    t0 = time.time()
    a_res = []
    ca = config["component_A"]
    for lvl, designs in (("target", ca["designs_target"]), ("joint", ca["designs_joint"])):
        for (n, m) in designs:
            for cid in ca["conditions"]:
                reps = ca["datasets_per_design"] if cid != "collapse_both" else ca["tie_datasets"]
                for r in range(reps):
                    exs = make_dataset(joint, tc, conds[cid], n, m, run_seed, jname, "A", f"{n}x{m}", r, lvl, scale)
                    vals, t_obs = pm.exact_null_values(exs)
                    su = pm.group_superuniformity(vals, ca["alphas"])
                    rec = {"component": "A", "level": lvl, "design": [n, m], "cond": cid, "rep": r,
                           "G": su["size"], "all_ok": su["all_ok"], "p_exact": pm.exact_pvalue(vals, t_obs),
                           "counts": {a: su[str(a)]["count"] for a in ca["alphas"]}}
                    a_res.append(rec)
                    raw.append(rec)
    results["A"] = {"n_datasets": len(a_res), "all_ok": all(r["all_ok"] for r in a_res), "seconds": time.time() - t0}
    status["A"] = "PASS" if results["A"]["all_ok"] else "FAIL"

    # ---- B: Monte Carlo vs exact -----------------------------------------------------
    t0 = time.time()
    cb = config["component_B"]
    n, m = cb["design"]
    b_res = []
    for cid in cb["conditions"]:
        for r in range(cb["datasets_per_condition"]):
            exs = make_dataset(joint, tc, conds[cid], n, m, run_seed, jname, "B", f"{n}x{m}", r, "target", scale)
            vals, t_obs = pm.exact_null_values(exs)
            p_ex = pm.exact_pvalue(vals, t_obs)
            rng = random.Random(seed_v2(run_seed, experiment=EXP, joint=jname, condition=cid, example="all",
                                        replicate=r, branch="permutation", component="B"))
            mc = pm.mc_permutation_test(exs, cb["n_perm"], rng)
            bperm = cb["n_perm"] + 1
            tol = 4.0 * math.sqrt(max(p_ex * (1 - p_ex), 1e-12) / bperm) + 1.0 / bperm
            ok = abs(mc["p"] - p_ex) <= tol and mc["p"] >= 1.0 / bperm
            rec = {"component": "B", "cond": cid, "rep": r, "p_exact": p_ex, "p_mc": mc["p"], "tol": tol, "ok": ok}
            b_res.append(rec)
            raw.append(rec)
    results["B"] = {"n_datasets": len(b_res), "all_ok": all(r["ok"] for r in b_res),
                    "min_p_mc": min(r["p_mc"] for r in b_res), "seconds": time.time() - t0}
    status["B"] = "PASS" if results["B"]["all_ok"] else "FAIL"

    # ---- C: unconditional rejection rates with exact p ------------------------------
    t0 = time.time()
    cc = config["component_C"]
    n, m = cc["design"]
    c_blocks = {}
    for item in cc["cells"]:
        if over():
            status["C"] = "PARTIAL"
            c_blocks[f"{item['cond']}|{item['level']}"] = {"status": "NOT_RUN", "reason": "budget"}
            continue
        cid, lvl, reps = item["cond"], item["level"], item["replicates"]
        ps = []
        g_size = None
        for r in range(reps):
            exs = make_dataset(joint, tc, conds[cid], n, m, run_seed, jname, "C", f"{n}x{m}", r, lvl, scale)
            vals, t_obs = pm.exact_null_values(exs)
            g_size = len(vals)
            p = pm.exact_pvalue(vals, t_obs)
            ps.append(p)
            raw.append({"component": "C", "cond": cid, "level": lvl, "rep": r, "p_exact": p})
        k = sum(1 for p in ps if p <= alpha)
        blk = {"cond": cid, "level": lvl, "role": item["role"], "G": g_size,
               "detector": rate_block(k, reps, alpha),
               "always_accept": rate_block(0, reps, alpha),
               "always_reject": rate_block(reps, reps, alpha)}
        if item["role"] == "null":
            blk["verdict"] = "EXCESS_FLAGGED" if blk["detector"]["excess_flag"] else "NO_EXCESS_FLAGGED"
            blk["negative_controls"] = {
                "always_accept": "EXCESS_FLAGGED" if blk["always_accept"]["excess_flag"] else "NO_EXCESS_FLAGGED",
                "always_reject": "EXCESS_FLAGGED" if blk["always_reject"]["excess_flag"] else "NO_EXCESS_FLAGGED"}
        else:
            blk["verdict"] = "POWERED" if blk["detector"]["powered_flag"] else "NOT_POWERED"
            blk["negative_controls"] = {
                "always_accept": "POWERED" if blk["always_accept"]["powered_flag"] else "NOT_POWERED",
                "always_reject": "POWERED" if blk["always_reject"]["powered_flag"] else "NOT_POWERED"}
        blk["expected"] = item.get("expect")
        blk["matches_expectation"] = (None if item.get("expect") in (None, "report")
                                      else blk["verdict"] == item["expect"])
        c_blocks[f"{cid}|{lvl}"] = blk
    results["C"] = {"cells": c_blocks, "seconds": time.time() - t0}
    status.setdefault("C", "COMPLETED")

    # ---- D: timing smoke --------------------------------------------------------------
    t0 = time.time()
    cd = config["component_D"]
    n, m = cd["design"]
    d_res = {}
    for lvl in ("target", "joint"):
        if over():
            d_res[lvl] = {"status": "NOT_RUN", "reason": "budget"}
            continue
        exs = make_dataset(joint, tc, conds["exact_sequential"], n, m, run_seed, jname, "D", f"{n}x{m}", 0, lvl, scale)
        rng = random.Random(seed_v2(run_seed, experiment=EXP, joint=jname, condition="exact_sequential",
                                    example="all", replicate=0, branch="permutation", component="D", level=lvl))
        ts = time.time()
        mc = pm.mc_permutation_test(exs, cd["n_perm"], rng)
        el = time.time() - ts
        d_res[lvl] = {"N": n, "M": m, "B": cd["n_perm"], "seconds": el,
                      "sec_per_example_per_perm": el / (n * (cd["n_perm"] + 1)),
                      "p_not_for_inference": mc["p"]}
    results["D"] = {"timing": d_res, "seconds": time.time() - t0}
    status["D"] = "COMPLETED"

    results["status"] = status
    results["total_seconds"] = time.time() - t_all
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(out_dir, "raw.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    return results
