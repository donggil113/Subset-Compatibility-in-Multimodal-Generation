"""FIRST RUN driver: direct vs intermediate-then-target compatibility probe.

Independent evaluation unit = conditioning example i (a row (x_i, y_i, z_i)
of the test split). Per example we draw M samples from each branch and
compute:

  target level  d_i  = ED(zA_i, zC_i) - ED(zA_i, zB_i)
  joint level   dJ_i = SED((y,z)A_i, (y,z)C_i) - SED((y,z)A_i, (y,z)B_i)

where A, B are two independent sample sets from the reference (direct)
branch and C is the compared branch. Under H0 (C has the reference
distribution) C and B are exchangeable given A, so d_i is symmetric about 0
and the one-sided sign-flip test over examples is exact. No same-index
pairing between generated samples is used for testing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from typing import Any, Dict, List, Optional

from . import metrics as mt
from . import samplers as sp
from . import stats as st
from .diffusion import GuidedPFODESampler
from .gaussian import JointGaussian

# ---------------------------------------------------------------------------
# Seeding and data
# ---------------------------------------------------------------------------


def derive_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def make_joint(spec: Dict[str, Any]) -> JointGaussian:
    return JointGaussian.from_sd_corr(["x", "y", "z"], spec["mean"], spec["sd"], spec["corr"])


def make_split(joint: JointGaussian, n: int, seed: int) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = joint.sample(rng)
        rows.append({"x": s["x"], "y_obs": s["y"], "z_obs": s["z"]})
    return rows


def rows_digest(rows: List[Dict[str, float]]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Condition construction
# ---------------------------------------------------------------------------


def exact(c):
    return sp.ExactSampler(c)


def build_condition(tc: sp.TrueConditionals, cond: Dict[str, Any], diffusion_cfg: Optional[Dict[str, Any]] = None):
    """Return (reference_branch, compared_branch, has_joint_reference, ref_key).

    ref_key names the reference sampler; conditions that share it reuse the
    same reference draws A and B (common random numbers across conditions).
    """
    ref_key = "exact_direct"
    kind = cond["kind"]
    p = cond.get("param")
    ref = sp.Branch("direct", exact(tc.z_x))
    y_x = exact(tc.y_x)
    if kind == "exact_sequential":
        return ref, sp.Branch("sequential", exact(tc.z_xy), y_x), True, ref_key
    if kind == "stage2_mean_shift":
        return ref, sp.Branch("sequential", exact(sp.stage2_mean_shift(tc, p)), y_x), True, ref_key
    if kind == "stage2_target_var_ratio":
        return ref, sp.Branch("sequential", exact(sp.stage2_target_var_ratio(tc, p)), y_x), True, ref_key
    if kind == "stage2_corr_scale":
        return ref, sp.Branch("sequential", exact(sp.stage2_corr_scale(tc, p)), y_x), True, ref_key
    if kind == "stage1_mean_shift":
        return ref, sp.Branch("sequential", exact(tc.z_xy), exact(sp.stage1_mean_shift(tc, p))), True, ref_key
    if kind == "collapse_both":
        cref = sp.Branch("direct", exact(sp.collapsed(tc.z_x)))
        ref_key = "collapsed_direct"
        return cref, sp.Branch("sequential", exact(sp.collapsed(tc.z_xy)), exact(sp.collapsed(tc.y_x))), True, ref_key
    if kind == "collapse_intermediate":
        return ref, sp.Branch("sequential", exact(tc.z_xy), exact(sp.collapsed(tc.y_x))), True, ref_key
    if kind == "observed_intermediate":
        return ref, sp.Branch("observed_int", exact(tc.z_xy)), False, ref_key
    if kind == "chain_drop_x":
        return ref, sp.Branch("chain", exact(tc.z_y), y_x), True, ref_key
    if kind in ("diffusion_sequential", "diffusion_direct_vs_exact"):
        d = dict(diffusion_cfg or {})
        d.update(cond.get("diffusion", {}))
        kw = dict(n_steps=d["n_steps"], solver=d["solver"], guidance=d["guidance"],
                  sigma_min=d["sigma_min"], sigma_max=d["sigma_max"], rho=d["rho"])
        dref = sp.Branch("direct", GuidedPFODESampler(tc.z_x, tc.z, **kw))
        if kind == "diffusion_direct_vs_exact":
            # Reference = exact direct; compared = the numerical direct sampler.
            return ref, dref, False, ref_key
        ref_key = "pfode:" + json.dumps(kw, sort_keys=True)
        s1 = GuidedPFODESampler(tc.y_x, tc.y, **kw)
        s2 = GuidedPFODESampler(tc.z_xy, tc.z, **kw)
        return dref, sp.Branch("sequential", s2, s1), False, ref_key
    raise ValueError(f"unknown condition kind {kind}")


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------


def evaluate_condition(tc, rows, cond, m, run_seed, split, rep, scale, directions, diffusion_cfg=None,
                       keep_rows=True):
    ref, cmp_branch, has_joint, ref_key = build_condition(tc, cond, diffusion_cfg)
    jref = sp.Branch("direct_joint", exact(tc.yz_x))
    sd_y, sd_z = scale["y"], scale["z"]
    per = []
    tau = math.sqrt(tc.tau2)
    for i, ex in enumerate(rows):
        rng_a = random.Random(derive_seed(run_seed, split, rep, i, ref_key, "A"))
        rng_b = random.Random(derive_seed(run_seed, split, rep, i, ref_key, "B"))
        rng_c = random.Random(derive_seed(run_seed, split, rep, i, cond["id"], "C"))
        za = ref.sample(rng_a, ex, m)["z"]
        zb = ref.sample(rng_b, ex, m)["z"]
        out_c = cmp_branch.sample(rng_c, ex, m)
        zc = out_c["z"]
        za_s = [v / sd_z for v in za]
        zb_s = [v / sd_z for v in zb]
        zc_s = [v / sd_z for v in zc]
        ed_c = mt.energy_distance_1d(za_s, zc_s)
        ed_f = mt.energy_distance_1d(za_s, zb_s)
        m_true = tc.z_x.mean([ex["x"]])[0]
        rec = {
            "i": i,
            "ed_cmp": ed_c,
            "ed_floor": ed_f,
            "d": ed_c - ed_f,
            "mean_diff_over_tau": (mt.mean(zc) - mt.mean(za)) / tau,
            "log_sd_ratio": mt.log_sd_ratio(za, zc),
            "crps_ref": mt.crps_fair(za, ex["z_obs"]) / sd_z,
            "crps_cmp": mt.crps_fair(zc, ex["z_obs"]) / sd_z,
            "fid_ref": mt.mean_true_logpdf(za, m_true, tc.tau2),
            "fid_cmp": mt.mean_true_logpdf(zc, m_true, tc.tau2),
            "div_ref": mt.sd(za) / tau,
            "div_cmp": mt.sd(zc) / tau,
            "bias_ref_over_tau": (mt.mean(za) - m_true) / tau,
            "bias_cmp_over_tau": (mt.mean(zc) - m_true) / tau,
            "paired_mse_over_tau2": mt.paired_index_mse(za, zc) / tc.tau2,
        }
        if has_joint and out_c["y"] is not None:
            rng_ja = random.Random(derive_seed(run_seed, split, rep, i, "exact_joint", "A"))
            rng_jb = random.Random(derive_seed(run_seed, split, rep, i, "exact_joint", "B"))
            if cond["kind"] == "collapse_both":
                cj = sp.Branch("direct_joint", exact(sp.collapsed(tc.yz_x)))
                ja, jb = cj.sample(rng_ja, ex, m), cj.sample(rng_jb, ex, m)
            else:
                ja, jb = jref.sample(rng_ja, ex, m), jref.sample(rng_jb, ex, m)
            pa = [[u / sd_y, v / sd_z] for u, v in zip(ja["y"], ja["z"])]
            pb = [[u / sd_y, v / sd_z] for u, v in zip(jb["y"], jb["z"])]
            pc = [[u / sd_y, v / sd_z] for u, v in zip(out_c["y"], out_c["z"])]
            sed_c = mt.sliced_energy_distance_2d(pa, pc, directions)
            sed_f = mt.sliced_energy_distance_2d(pa, pb, directions)
            rec.update({"sed_cmp": sed_c, "sed_floor": sed_f, "dJ": sed_c - sed_f})
        per.append(rec)
    return per


def summarise(per, alpha, n_flips, seed):
    rng = random.Random(seed)
    out: Dict[str, Any] = {"n_examples": len(per)}
    d = [r["d"] for r in per]
    out["target"] = {
        "ed_cmp": st.mean_ci([r["ed_cmp"] for r in per]),
        "ed_floor": st.mean_ci([r["ed_floor"] for r in per]),
        "d": st.mean_ci(d),
        "p_sign_flip": st.sign_flip_pvalue(d, n_flips, rng),
    }
    out["target"]["detected"] = out["target"]["p_sign_flip"] < alpha
    if all("dJ" in r for r in per):
        dj = [r["dJ"] for r in per]
        out["joint"] = {
            "sed_cmp": st.mean_ci([r["sed_cmp"] for r in per]),
            "sed_floor": st.mean_ci([r["sed_floor"] for r in per]),
            "dJ": st.mean_ci(dj),
            "p_sign_flip": st.sign_flip_pvalue(dj, n_flips, rng),
        }
        out["joint"]["detected"] = out["joint"]["p_sign_flip"] < alpha
    else:
        out["joint"] = None
    for k in ("mean_diff_over_tau", "log_sd_ratio", "crps_ref", "crps_cmp", "fid_ref", "fid_cmp",
              "div_ref", "div_cmp", "bias_ref_over_tau", "bias_cmp_over_tau", "paired_mse_over_tau2"):
        out[k] = st.mean_ci([r[k] for r in per])
    out["n_log_sd_ratio_undefined"] = sum(1 for r in per if r["log_sd_ratio"] is None)
    out["crps_diff_cmp_minus_ref"] = st.mean_ci([r["crps_cmp"] - r["crps_ref"] for r in per])
    return out


def expectation_check(summary, expect: Dict[str, str]) -> Dict[str, str]:
    res = {}
    for level, want in expect.items():
        blk = summary.get(level)
        if blk is None:
            res[level] = "NOT_APPLICABLE"
            continue
        if want == "report":
            res[level] = "DESCRIPTIVE"
            continue
        got = "detect" if blk["detected"] else "no_detect"
        res[level] = "MATCH" if got == want else "MISMATCH"
    return res


# ---------------------------------------------------------------------------
# Experiment blocks
# ---------------------------------------------------------------------------


class Budget:
    def __init__(self, seconds: float):
        self.t0 = time.time()
        self.seconds = seconds

    def elapsed(self) -> float:
        return time.time() - self.t0

    def exhausted(self) -> bool:
        return self.elapsed() > self.seconds


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run(config: Dict[str, Any], split: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    budget = Budget(config["budget"]["max_seconds"])
    run_seed = config["seed"]
    alpha = config["alpha"]
    n_flips = config["n_sign_flips"]
    directions = mt.slice_directions(config["joint_slices"])
    manifest: Dict[str, Any] = {
        "split": split,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "experiments": {},
        "data": {},
    }
    summaries: Dict[str, Any] = {}

    joints = {name: make_joint(spec) for name, spec in config["joints"].items()}
    tcs = {name: sp.true_conditionals(j) for name, j in joints.items()}
    data = {}
    for name, joint in joints.items():
        seeds = config["split_seeds"]
        train = make_split(joint, config["n_train"], derive_seed(run_seed, name, "train", seeds["train"]))
        rows = make_split(joint, config["n_" + split], derive_seed(run_seed, name, split, seeds[split]))
        scale = {"y": mt.sd([r["y_obs"] for r in train]), "z": mt.sd([r["z_obs"] for r in train])}
        data[name] = {"rows": rows, "scale": scale}
        manifest["data"][name] = {
            "n_train": len(train), f"n_{split}": len(rows), "train_digest": rows_digest(train),
            f"{split}_digest": rows_digest(rows), "scale_from_train": scale,
            "true": {
                "tau2_z_given_x": tcs[name].tau2, "var_y_given_x": tcs[name].v_y, "b_y_in_z_given_xy": tcs[name].b,
                "var_z_given_xy": tcs[name].z_xy.cov[0][0],
                "partial_corr_xz_given_y": joints[name].partial_corr("x", "z", ["y"]),
                "partial_corr_yz_given_x": joints[name].partial_corr("y", "z", ["x"]),
            },
        }

    for exp in config["experiments"]:
        eid = exp["id"]
        if budget.exhausted():
            manifest["experiments"][eid] = {"status": "NOT_RUN", "reason": "budget exhausted before start"}
            continue
        t_start = time.time()
        jname = exp["joint"]
        tc, rows, scale = tcs[jname], data[jname]["rows"], data[jname]["scale"]
        n_use = exp.get("n_examples", len(rows))
        rows_e = rows[:n_use]
        raw_rows: List[Dict[str, Any]] = []
        exp_sum: Dict[str, Any] = {"joint": jname, "conditions": {}}
        status = "COMPLETED"
        try:
            if exp["type"] == "conditions":
                for cond in exp["conditions"]:
                    if budget.exhausted():
                        exp_sum["conditions"][cond["id"]] = {"status": "NOT_RUN", "reason": "budget"}
                        status = "PARTIAL"
                        continue
                    m = cond.get("m", exp["m"])
                    per = evaluate_condition(tc, rows_e, cond, m, run_seed, split, 0, scale, directions,
                                             exp.get("diffusion"))
                    s = summarise(per, alpha, n_flips, derive_seed(run_seed, eid, cond["id"], "flips"))
                    s["m"] = m
                    s["cond"] = cond
                    s["expectation"] = expectation_check(s, cond.get("expect", {}))
                    exp_sum["conditions"][cond["id"]] = s
                    for r in per:
                        raw_rows.append({"exp": eid, "cond": cond["id"], "m": m, **r})
            elif exp["type"] == "null_replicates":
                cond = {"id": "exact_sequential", "kind": "exact_sequential"}
                reps = []
                for rep in range(1, exp["replicates"] + 1):
                    if budget.exhausted():
                        status = "PARTIAL"
                        break
                    per = evaluate_condition(tc, rows_e, cond, exp["m"], run_seed, split, rep, scale, directions)
                    s = summarise(per, alpha, n_flips, derive_seed(run_seed, eid, rep, "flips"))
                    reps.append({
                        "rep": rep,
                        "p_target": s["target"]["p_sign_flip"],
                        "p_joint": s["joint"]["p_sign_flip"],
                        "z_target": s["target"]["d"]["mean"] / s["target"]["d"]["se"],
                        "z_joint": s["joint"]["dJ"]["mean"] / s["joint"]["dJ"]["se"],
                    })
                k_t = sum(1 for r in reps if r["p_target"] < alpha)
                k_j = sum(1 for r in reps if r["p_joint"] < alpha)
                exp_sum["replicates"] = reps
                exp_sum["fpr_target"] = st.wilson_ci(k_t, len(reps))
                exp_sum["fpr_joint"] = st.wilson_ci(k_j, len(reps))
                exp_sum["z_target_mean_sd"] = st.mean_ci([r["z_target"] for r in reps])
                exp_sum["z_joint_mean_sd"] = st.mean_ci([r["z_joint"] for r in reps])
                raw_rows = [{"exp": eid, **r} for r in reps]
            else:
                raise ValueError(f"unknown experiment type {exp['type']}")
        except Exception as e:  # recorded, never converted into a pass
            status = "FAIL"
            exp_sum["error"] = f"{type(e).__name__}: {e}"
        exp_sum["seconds"] = time.time() - t_start
        summaries[eid] = exp_sum
        raw_path = os.path.join(out_dir, f"{eid}.jsonl")
        _write_jsonl(raw_path, raw_rows)
        manifest["experiments"][eid] = {"status": status, "seconds": exp_sum["seconds"],
                                        "raw": os.path.basename(raw_path), "n_raw_rows": len(raw_rows)}
        if "error" in exp_sum:
            manifest["experiments"][eid]["error"] = exp_sum["error"]

    manifest["total_seconds"] = budget.elapsed()
    manifest["budget_seconds"] = budget.seconds
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=1)
    return manifest
