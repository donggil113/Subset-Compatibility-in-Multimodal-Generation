"""P5-E8-CALIB runner (config: configs/p5_e8_calib_v3.json).

One task = one (cell, replicate): fresh conditioning rows, fresh direct and
sequential samples (seed_v2 keys include the level, so target- and joint-level
cells never share data), one within-example permutation test, and quality
metrics against the truth. Tasks are independent, so the result does not
depend on how they are scheduled over worker processes.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import time
from statistics import NormalDist
from typing import Any, Dict, List

from . import metrics as mt
from . import permutation as pm
from . import samplers as sp
from . import stats as st
from .experiment import build_condition, exact, make_joint
from .seeds import seed_v2

EXP = "P5-E8-CALIB"
_ND = NormalDist()


def crps_gauss(m: float, s: float, y: float) -> float:
    """Closed-form CRPS of N(m, s^2) at y (Gneiting & Raftery 2007)."""
    w = (y - m) / s
    return s * (w * (2 * _ND.cdf(w) - 1) + 2 * _ND.pdf(w) - 1 / math.sqrt(math.pi))


def cell_id(c: Dict[str, Any]) -> str:
    return f"{c['cond']}|{c['level']}|{c['joint']}"


def run_task(args):
    cfg, cell, rep = args
    t0 = time.time()
    run_seed = cfg["seed"]
    jname, level = cell["joint"], cell["level"]
    joint = make_joint(cfg["joints"][jname])
    tc = sp.true_conditionals(joint)
    cond = {"id": cell["cond"], "kind": cell["kind"], "param": cell.get("param")}
    ref, cmp_branch, _, _ = build_condition(tc, cond)
    if level == "joint":
        jref = sp.Branch("direct_joint", exact(sp.collapsed(tc.yz_x) if cell["kind"] == "collapse_both" else tc.yz_x))
    n, m = cfg["design"]["N_examples_per_replicate"], cfg["design"]["M_samples_per_branch"]
    sy, sz = cfg["standardisation_sd"]["y"], cfg["standardisation_sd"]["z"]
    key = dict(experiment=EXP, joint=jname, condition=cell["cond"], level=level, replicate=rep)
    rng_rows = random.Random(seed_v2(run_seed, example="all", branch="rows", **key))
    rows = []
    for _ in range(n):
        s = joint.sample(rng_rows)
        rows.append({"x": s["x"], "y_obs": s["y"], "z_obs": s["z"]})
    tau = math.sqrt(tc.tau2)
    dirs = mt.slice_directions(8) if level == "joint" else None
    exs, ex_crps_c, ex_crps_a, sdr_c, sdr_a = [], [], [], [], []
    min_distinct_intermediates = None
    for i, ex in enumerate(rows):
        rng_a = random.Random(seed_v2(run_seed, example=i, branch="direct", **key))
        rng_c = random.Random(seed_v2(run_seed, example=i, branch="sequential", **key))
        oc = cmp_branch.sample(rng_c, ex, m)
        if oc["y"] is not None:
            d = len(set(oc["y"]))
            min_distinct_intermediates = d if min_distinct_intermediates is None else min(min_distinct_intermediates, d)
        if level == "target":
            za = ref.sample(rng_a, ex, m)["z"]
            exs.append(pm.PooledExample([v / sz for v in za], [v / sz for v in oc["z"]]))
        else:
            oa = jref.sample(rng_a, ex, m)
            za = oa["z"]
            exs.append(pm.PooledExample([[u / sy, v / sz] for u, v in zip(oa["y"], oa["z"])],
                                        [[u / sy, v / sz] for u, v in zip(oc["y"], oc["z"])], dirs))
        m_true = tc.z_x.mean([ex["x"]])[0]
        truth = crps_gauss(m_true / sz, tau / sz, ex["z_obs"] / sz)
        ex_crps_c.append(mt.crps_fair([v / sz for v in oc["z"]], ex["z_obs"] / sz) - truth)
        ex_crps_a.append(mt.crps_fair([v / sz for v in za], ex["z_obs"] / sz) - truth)
        sdr_c.append(mt.sd(oc["z"]) / tau)
        sdr_a.append(mt.sd(za) / tau)
    rng_p = random.Random(seed_v2(run_seed, example="all", branch="permutation", **key))
    res = pm.mc_permutation_test(exs, cfg["design"]["B_permutations"], rng_p)
    qc = st.mean_ci(ex_crps_c)
    qa = st.mean_ci(ex_crps_a)
    return {"cell": cell_id(cell), "role": cell["role"], "rep": rep, "p": res["p"], "t_obs": res["t_obs"],
            "reject": res["p"] <= cfg["gate"]["nominal_alpha_per_test"],
            "excess_crps_seq": qc["mean"], "excess_crps_seq_lo": qc["lo"], "excess_crps_seq_hi": qc["hi"],
            "excess_crps_direct": qa["mean"], "excess_crps_direct_lo": qa["lo"], "excess_crps_direct_hi": qa["hi"],
            "sd_ratio_seq": sum(sdr_c) / n, "sd_ratio_direct": sum(sdr_a) / n,
            "min_distinct_intermediates": min_distinct_intermediates, "seconds": time.time() - t0}


def _rate(k: int, n: int, level: float) -> Dict[str, float]:
    return {"k": k, "n": n, "rate": k / n if n else None,
            "cp_lower": st.clopper_pearson(k, n, level, "lower")["lo"],
            "cp_upper": st.clopper_pearson(k, n, level, "upper")["hi"], "level": level}


def summarise(cfg: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    alpha = cfg["gate"]["nominal_alpha_per_test"]
    by = {}
    for r in results:
        by.setdefault(r["cell"], []).append(r)
    k0 = len(cfg["families"]["null_validity"])
    bonf = 1 - alpha / k0
    cells = {}
    for c in cfg["cells"] + cfg["quality_controls"]:
        cid = cell_id(c)
        rs = sorted(by.get(cid, []), key=lambda r: r["rep"])
        planned = c["replicates"]
        k = sum(1 for r in rs if r["reject"])
        blk = {"role": c["role"], "planned": planned, "done": len(rs),
               "status": "COMPLETED" if len(rs) == planned else ("PARTIAL" if rs else "NOT_RUN"),
               "rate_95": _rate(k, len(rs), 0.95) if rs else None,
               "p_values_min": min((r["p"] for r in rs), default=None),
               "mean_excess_crps_seq": sum(r["excess_crps_seq"] for r in rs) / len(rs) if rs else None,
               "mean_excess_crps_direct": sum(r["excess_crps_direct"] for r in rs) / len(rs) if rs else None,
               "frac_reps_excess_crps_seq_ci_above_0": (sum(1 for r in rs if r["excess_crps_seq_lo"] > 0) / len(rs)) if rs else None,
               "mean_sd_ratio_seq": sum(r["sd_ratio_seq"] for r in rs) / len(rs) if rs else None,
               "mean_sd_ratio_direct": sum(r["sd_ratio_direct"] for r in rs) / len(rs) if rs else None,
               "min_distinct_intermediates": min((r["min_distinct_intermediates"] for r in rs if r["min_distinct_intermediates"] is not None), default=None),
               "cpu_seconds": sum(r["seconds"] for r in rs)}
        if rs and c["role"] == "null":
            blk["rate_bonferroni"] = _rate(k, len(rs), bonf)
            blk["excess_flag"] = blk["rate_bonferroni"]["cp_lower"] > alpha
            if c.get("tolerance"):
                blk["tolerance_pass"] = blk["rate_95"]["cp_upper"] <= 0.10
        if rs and c["role"] == "alternative":
            blk["powered"] = blk["rate_95"]["cp_lower"] > alpha
        if rs and c["role"] == "quality_control":
            blk["compat_test_rejects_any"] = k > 0
            blk["quality_flag_all_reps"] = all(r["excess_crps_seq_lo"] > 0 for r in rs)
        cells[cid] = blk

    def gate_eval(get_k):
        """Evaluate G1-G3 for a detector whose rejection count per cell is get_k(cid, R)."""
        g1 = g2 = g3 = True
        for c in cfg["cells"]:
            cid, R = cell_id(c), c["replicates"]
            k = get_k(cid, R)
            if c["role"] == "null":
                if c.get("tolerance") and st.clopper_pearson(k, R, 0.95, "upper")["hi"] > 0.10:
                    g1 = False
                if st.clopper_pearson(k, R, bonf, "lower")["lo"] > alpha:
                    g2 = False
            elif c["role"] == "alternative":
                if not st.clopper_pearson(k, R, 0.95, "lower")["lo"] > alpha:
                    g3 = False
        return {"G1_primary_tolerance": g1, "G2_null_family_no_excess": g2, "G3_power_family": g3, "gate_pass": g1 and g2 and g3}

    complete = all(cells[cell_id(c)]["status"] == "COMPLETED" for c in cfg["cells"])
    gate = gate_eval(lambda cid, R: cells[cid]["rate_95"]["k"]) if complete else {"gate_pass": None, "reason": "incomplete: NOT_RUN cells"}
    neg = {"always_accept": gate_eval(lambda cid, R: 0), "always_reject": gate_eval(lambda cid, R: R)}
    return {"cells": cells, "gate": gate, "negative_controls": neg, "bonferroni_level_null_family": bonf}


def run(cfg: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(cfg, c, r) for c in cfg["cells"] + cfg["quality_controls"] for r in range(c["replicates"])]
    # joint-level tasks are ~4x slower; submit them first for better load balance
    tasks.sort(key=lambda t: 0 if t[1]["level"] == "joint" else 1)
    t0 = time.time()
    results, timed_out = [], False
    with mp.Pool(cfg["design"]["workers"]) as pool:
        it = pool.imap_unordered(run_task, tasks, chunksize=1)
        with open(os.path.join(out_dir, "raw.jsonl"), "w") as f:
            for _ in range(len(tasks)):
                remaining = cfg["budget"]["max_wall_seconds"] - (time.time() - t0)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    r = it.next(timeout=remaining)
                except mp.TimeoutError:
                    timed_out = True
                    break
                results.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
        if timed_out:
            pool.terminate()
    results.sort(key=lambda r: (r["cell"], r["rep"]))
    with open(os.path.join(out_dir, "raw.jsonl"), "w") as f:  # deterministic order
        for r in results:
            f.write(json.dumps(r) + "\n")
    summ = summarise(cfg, results)
    summ["wall_seconds"] = time.time() - t0
    summ["cpu_seconds_tasks"] = sum(r["seconds"] for r in results)
    summ["timed_out"] = timed_out
    summ["n_tasks_planned"], summ["n_tasks_done"] = len(tasks), len(results)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    return summ
