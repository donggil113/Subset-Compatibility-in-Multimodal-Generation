"""EXPLORATORY re-analysis of the committed v1 test-split results (no new sampling).

Inputs: results/raw/p5_first_run_v1_test/{summary.json, manifest.json, *.jsonl},
configs/p5_first_run.json. The v1 test rows are regenerated deterministically
from the v1 seed (make_split) and checked against the digest in the manifest.

Outputs (results/reanalysis_v1/):
  predicted_vs_observed.json / .md
      closed-form E[d] (and E[dJ]) of the v1 V-statistic contrast for every
      exact-Gaussian and PF-ODE condition vs the observed mean and 95% CI.
  population_gaps.json
      Monte-Carlo-free population energy distances induced by solver / CFG.
  z_null_check.json
      sign-flip randomisation distribution of the studentised mean of d for
      the v1 null conditions (checks the normal reference used for z).

All of this was computed after the v1 results were seen: EXPLORATORY.
"""

import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat import analytic as an  # noqa: E402
from p5compat import metrics as mt  # noqa: E402
from p5compat import samplers as sp  # noqa: E402
from p5compat import stats as st  # noqa: E402
from p5compat.experiment import derive_seed, make_joint, make_split, rows_digest  # noqa: E402

RUN = os.path.join(ROOT, "results", "raw", "p5_first_run_v1_test")
OUT = os.path.join(ROOT, "results", "reanalysis_v1")


def load():
    cfg = json.load(open(os.path.join(ROOT, "configs", "p5_first_run.json")))
    summ = json.load(open(os.path.join(RUN, "summary.json")))
    man = json.load(open(os.path.join(RUN, "manifest.json")))
    return cfg, summ, man


def rows_for(cfg, man, jname):
    joint = make_joint(cfg["joints"][jname])
    rows = make_split(joint, cfg["n_test"], derive_seed(cfg["seed"], jname, "test", cfg["split_seeds"]["test"]))
    assert rows_digest(rows) == man["data"][jname]["test_digest"], "regenerated rows do not match v1 digest"
    return joint, rows


# ---------------------------------------------------------------------------
# Laws of the compared branch C and the second reference set B per example
# ---------------------------------------------------------------------------


def joint_law_seq(stage1, stage2, x):
    """(mean2, cov2) of (y, z) | x for affine-Gaussian stages; stage2 given (x,y) or (y)."""
    t1, t2 = stage1.temperature, stage2.temperature
    m_y = stage1.mean([x])[0]
    v_y = t1 ** 2 * stage1.cov[0][0]
    if stage2.given == ["x", "y"]:
        cx, cy = stage2.coef[0]
        m_z = stage2.intercept[0] + cx * x + cy * m_y
    elif stage2.given == ["y"]:
        cy = stage2.coef[0][0]
        m_z = stage2.intercept[0] + cy * m_y
    else:
        raise ValueError(stage2.given)
    s2 = t2 ** 2 * stage2.cov[0][0]
    cov = [[v_y, cy * v_y], [cy * v_y, cy * cy * v_y + s2]]
    return [m_y, m_z], cov


def exact_condition_laws(tc, cond, ex):
    """Return dict with target laws (mean, sd) for C and B and joint laws for C and B (or None)."""
    x = ex["x"]
    kind, p = cond["kind"], cond.get("param")
    ref_t = (tc.z_x.mean([x])[0], math.sqrt(tc.tau2))
    ref_j = (tc.yz_x.mean([x]), tc.yz_x.cov)
    stage1, stage2 = tc.y_x, tc.z_xy
    if kind == "exact_sequential":
        pass
    elif kind == "stage2_mean_shift":
        stage2 = sp.stage2_mean_shift(tc, p)
    elif kind == "stage2_target_var_ratio":
        stage2 = sp.stage2_target_var_ratio(tc, p)
    elif kind == "stage2_corr_scale":
        stage2 = sp.stage2_corr_scale(tc, p)
    elif kind == "stage1_mean_shift":
        stage1 = sp.stage1_mean_shift(tc, p)
    elif kind == "collapse_intermediate":
        stage1 = sp.collapsed(tc.y_x)
    elif kind == "collapse_both":
        m_t = tc.z_x.mean([x])[0]
        mj = tc.yz_x.mean([x])
        zero = [[0.0, 0.0], [0.0, 0.0]]
        return {"C_t": (m_t, 0.0), "B_t": (m_t, 0.0), "C_j": (mj, zero), "B_j": (mj, zero)}
    elif kind == "chain_drop_x":
        stage2 = tc.z_y
    elif kind == "observed_intermediate":
        m = tc.z_xy.mean([x, ex["y_obs"]])[0]
        return {"C_t": (m, math.sqrt(tc.z_xy.cov[0][0])), "B_t": ref_t, "C_j": None, "B_j": None}
    else:
        raise ValueError(kind)
    mj, cj = joint_law_seq(stage1, stage2, x)
    return {"C_t": (mj[1], math.sqrt(cj[1][1])), "B_t": ref_t, "C_j": (mj, cj), "B_j": ref_j}


def predicted_target(laws, m, sz):
    (mc, sc), (mb, sb) = laws["C_t"], laws["B_t"]
    d_pop = an.energy_distance_gauss(mb / sz, sb / sz, mc / sz, sc / sz)
    return d_pop, an.expected_v1_contrast(d_pop, sc / sz, sb / sz, m)


def predicted_joint(laws, m, sy, sz, dirs):
    if laws["C_j"] is None:
        return None, None
    (mc, cc), (mb, cb) = laws["C_j"], laws["B_j"]

    def std(mean, cov):
        return [mean[0] / sy, mean[1] / sz], [[cov[0][0] / sy ** 2, cov[0][1] / (sy * sz)],
                                              [cov[1][0] / (sy * sz), cov[1][1] / sz ** 2]]

    mcs, ccs = std(mc, cc)
    mbs, cbs = std(mb, cb)
    dp = ev = 0.0
    for u in dirs:
        m1, s1 = an.gaussian_projection(mbs, cbs, u)
        m2, s2 = an.gaussian_projection(mcs, ccs, u)
        d_u = an.energy_distance_gauss(m1, s1, m2, s2)
        dp += d_u
        ev += an.expected_v1_contrast(d_u, s2, s1, m)
    return dp / len(dirs), ev / len(dirs)


def zscore(obs_ci, pred):
    if obs_ci["se"] is None or obs_ci["se"] < 1e-12:
        return None  # degenerate (e.g. collapse_both: d_i = 0 up to round-off)
    return (obs_ci["mean"] - pred) / obs_ci["se"]


def main():
    cfg, summ, man = load()
    os.makedirs(OUT, exist_ok=True)
    dirs = mt.slice_directions(cfg["joint_slices"])
    records, gaps = [], []
    for exp in cfg["experiments"]:
        if exp["type"] != "conditions":
            continue
        eid, jname = exp["id"], exp["joint"]
        joint, rows = rows_for(cfg, man, jname)
        tc = sp.true_conditionals(joint)
        sc = man["data"][jname]["scale_from_train"]
        rows_e = rows[: exp.get("n_examples", len(rows))]
        for cond in exp["conditions"]:
            s = summ[eid]["conditions"][cond["id"]]
            m = s["m"]
            rec = {"exp": eid, "cond": cond["id"], "m": m, "n": len(rows_e)}
            if cond["kind"].startswith("diffusion"):
                d = dict(exp["diffusion"])
                d.update(cond.get("diffusion", {}))
                dpop_list, ed_list, dexact_list = [], [], []
                for ex in rows_e:
                    laws = an.pfode_branch_laws(tc, ex["x"], d["n_steps"], d["solver"], d["guidance"],
                                                d["sigma_min"], d["sigma_max"], d["rho"])
                    if cond["kind"] == "diffusion_sequential":
                        (mb, sb), (mc, scc) = laws["direct"], laws["sequential"]
                    else:
                        (mb, sb), (mc, scc) = laws["exact"], laws["direct"]
                    dp = an.energy_distance_gauss(mb / sc["z"], sb / sc["z"], mc / sc["z"], scc / sc["z"])
                    dpop_list.append(dp)
                    ed_list.append(an.expected_v1_contrast(dp, scc / sc["z"], sb / sc["z"], m))
                    me, se = laws["exact"]
                    dexact_list.append(an.energy_distance_gauss(me / sc["z"], se / sc["z"], mb / sc["z"], sb / sc["z"]))
                rec.update({"pop_D_target": sum(dpop_list) / len(dpop_list),
                            "pred_E_d": sum(ed_list) / len(ed_list), "obs_d": s["target"]["d"],
                            "z_obs_minus_pred": zscore(s["target"]["d"], sum(ed_list) / len(ed_list))})
                gaps.append({"exp": eid, "cond": cond["id"], "n_steps": d["n_steps"], "solver": d["solver"],
                             "guidance": d["guidance"], "kind": cond["kind"],
                             "pop_D_compared_vs_reference": rec["pop_D_target"],
                             "pop_D_reference_vs_exact": sum(dexact_list) / len(dexact_list)})
            else:
                dp_t, ed_t, dp_j, ed_j = [], [], [], []
                for ex in rows_e:
                    laws = exact_condition_laws(tc, cond, ex)
                    a, b = predicted_target(laws, m, sc["z"])
                    dp_t.append(a)
                    ed_t.append(b)
                    a, b = predicted_joint(laws, m, sc["y"], sc["z"], dirs)
                    if a is not None:
                        dp_j.append(a)
                        ed_j.append(b)
                rec.update({"pop_D_target": sum(dp_t) / len(dp_t), "pred_E_d": sum(ed_t) / len(ed_t),
                            "obs_d": s["target"]["d"], "z_obs_minus_pred": zscore(s["target"]["d"], sum(ed_t) / len(ed_t))})
                if dp_j and s.get("joint"):
                    rec.update({"pop_D_joint": sum(dp_j) / len(dp_j), "pred_E_dJ": sum(ed_j) / len(ed_j),
                                "obs_dJ": s["joint"]["dJ"],
                                "zJ_obs_minus_pred": zscore(s["joint"]["dJ"], sum(ed_j) / len(ed_j))})
            records.append(rec)

    # continuous-time limit (exact flow, unguided): isolates the sigma_max prior mismatch
    for jname in ("nonmarkov",):
        joint, rows = rows_for(cfg, man, jname)
        tc = sp.true_conditionals(joint)
        sc = man["data"][jname]["scale_from_train"]
        rows_e = rows[:100]
        dseq, dex = [], []
        for ex in rows_e:
            laws = an.exact_flow_branch_laws(tc, ex["x"])
            (mb, sb), (mc, scc), (me, se) = laws["direct"], laws["sequential"], laws["exact"]
            dseq.append(an.energy_distance_gauss(mb / sc["z"], sb / sc["z"], mc / sc["z"], scc / sc["z"]))
            dex.append(an.energy_distance_gauss(me / sc["z"], se / sc["z"], mb / sc["z"], sb / sc["z"]))
        gaps.append({"exp": "analytic", "cond": "exact_flow_limit", "n_steps": "inf", "solver": "exact",
                     "guidance": 0.0, "kind": "diffusion_sequential",
                     "pop_D_compared_vs_reference": sum(dseq) / len(dseq),
                     "pop_D_reference_vs_exact": sum(dex) / len(dex)})

    # z-null check: sign-flip distribution of the studentised mean for v1 null conditions
    znull = []
    rng = random.Random(20260926)
    null_conds = [("P5-E1-NULL", "exact_sequential"), ("P5-E4-MARKOV", "exact_sequential"),
                  ("P5-E4-MARKOV", "chain_drop_x"), ("P5-E5-MC", "null_m32"), ("P5-E5-MC", "null_m64"),
                  ("P5-E5-MC", "null_m128"), ("P5-E5-MC", "null_m512")]
    n_flip = 4999
    for eid, cid in null_conds:
        d = [json.loads(l)["d"] for l in open(os.path.join(RUN, f"{eid}.jsonl")) if json.loads(l)["cond"] == cid]
        n = len(d)
        mu = sum(d) / n
        sdv = math.sqrt(sum((v - mu) ** 2 for v in d) / (n - 1))
        kurt = sum((v - mu) ** 4 for v in d) / n / sdv ** 4
        skew = sum((v - mu) ** 3 for v in d) / n / sdv ** 3
        t_obs = mu / (sdv / math.sqrt(n))
        ts = []
        for _ in range(n_flip):
            bits = rng.getrandbits(n)
            e = [v if (bits >> k) & 1 else -v for k, v in enumerate(d)]
            me = sum(e) / n
            se = math.sqrt(sum((v - me) ** 2 for v in e) / (n - 1)) / math.sqrt(n)
            ts.append(me / se)
        ts.sort()
        mts = sum(ts) / len(ts)
        sdt = math.sqrt(sum((v - mts) ** 2 for v in ts) / (len(ts) - 1))
        q95 = ts[int(0.95 * len(ts))]
        p_t = (1 + sum(1 for v in ts if v >= t_obs)) / (n_flip + 1)
        znull.append({"exp": eid, "cond": cid, "n": n, "skew_d": skew, "kurtosis_d": kurt, "t_obs": t_obs,
                      "signflip_t_sd": sdt, "signflip_t_q95": q95, "normal_q95": 1.6449,
                      "p_signflip_studentised": p_t,
                      "p_v1_reported": summ[eid]["conditions"][cid]["target"]["p_sign_flip"]})
    reps = summ["P5-E1-NULL-REPS"]["replicates"]
    zs = [r["z_target"] for r in reps]
    zj = [r["z_joint"] for r in reps]

    def sd(v):
        m_ = sum(v) / len(v)
        return math.sqrt(sum((u - m_) ** 2 for u in v) / (len(v) - 1))

    # chi-square(19) quantiles for a 95% CI on sd of 20 replicate z's (tabulated values)
    chi_lo, chi_hi = 8.9065, 32.8523
    rep_block = {"R": len(zs), "sd_z_target": sd(zs), "sd_z_joint": sd(zj),
                 "sd_ci_target": [sd(zs) * math.sqrt(19 / chi_hi), sd(zs) * math.sqrt(19 / chi_lo)],
                 "sd_ci_joint": [sd(zj) * math.sqrt(19 / chi_hi), sd(zj) * math.sqrt(19 / chi_lo)],
                 "rejections_target": sum(1 for r in reps if r["p_target"] < 0.05),
                 "rejections_joint": sum(1 for r in reps if r["p_joint"] < 0.05),
                 "cp_two_sided_target": st.clopper_pearson(sum(1 for r in reps if r["p_target"] < 0.05), len(reps)),
                 "cp_two_sided_joint": st.clopper_pearson(sum(1 for r in reps if r["p_joint"] < 0.05), len(reps)),
                 "note": "replicates share the same 200 conditioning rows; only sampling seeds differ"}

    json.dump(records, open(os.path.join(OUT, "predicted_vs_observed.json"), "w"), indent=1)
    json.dump(gaps, open(os.path.join(OUT, "population_gaps.json"), "w"), indent=1)
    json.dump({"per_condition": znull, "replicates": rep_block}, open(os.path.join(OUT, "z_null_check.json"), "w"), indent=1)

    lines = ["# v1 re-analysis (EXPLORATORY; computed after the v1 results were seen)", "",
             "Closed-form expectation of the v1 contrast d = ED_V(A,C) - ED_V(A,B) vs observed (mean over examples, 95% CI). "
             "pop D = population energy distance (no Monte Carlo). z = (observed - predicted)/SE.", "",
             "| experiment | condition | M | pop D (target) | predicted E[d] | observed d [95% CI] | z | pop D (joint) | predicted E[dJ] | observed dJ [95% CI] | zJ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, nd=4):
        return "NA" if v is None else f"{v:.{nd}f}"

    for r in records:
        o = r["obs_d"]
        oj = r.get("obs_dJ")
        lines.append(
            f"| {r['exp']} | {r['cond']} | {r['m']} | {f(r['pop_D_target'])} | {f(r['pred_E_d'])} | "
            f"{f(o['mean'])} [{f(o['lo'])}, {f(o['hi'])}] | {f(r['z_obs_minus_pred'], 2)} | "
            f"{f(r.get('pop_D_joint'))} | {f(r.get('pred_E_dJ'))} | "
            + ("NA" if oj is None else f"{f(oj['mean'])} [{f(oj['lo'])}, {f(oj['hi'])}]")
            + f" | {f(r.get('zJ_obs_minus_pred'), 2)} |")
    lines += ["", "## Studentised-z null check (sign-flip distribution given |d_i|)", "",
              "| experiment | condition | skew(d) | kurtosis(d) | t_obs | sd of t under sign flips | 95% quantile | p (studentised sign-flip) |",
              "|---|---|---|---|---|---|---|---|"]
    for z in znull:
        lines.append(f"| {z['exp']} | {z['cond']} | {z['skew_d']:.2f} | {z['kurtosis_d']:.2f} | {z['t_obs']:.2f} | "
                     f"{z['signflip_t_sd']:.3f} | {z['signflip_t_q95']:.3f} | {z['p_signflip_studentised']:.3f} |")
    rb = rep_block
    lines += ["", f"Replicates (R={rb['R']}): sd(z_target) = {rb['sd_z_target']:.3f} "
              f"[chi2 95% CI {rb['sd_ci_target'][0]:.2f}, {rb['sd_ci_target'][1]:.2f}]; "
              f"sd(z_joint) = {rb['sd_z_joint']:.3f} [{rb['sd_ci_joint'][0]:.2f}, {rb['sd_ci_joint'][1]:.2f}]; "
              f"rejections target {rb['rejections_target']}/{rb['R']} (Clopper-Pearson 95% "
              f"[{rb['cp_two_sided_target']['lo']:.3f}, {rb['cp_two_sided_target']['hi']:.3f}]), joint "
              f"{rb['rejections_joint']}/{rb['R']} ([{rb['cp_two_sided_joint']['lo']:.3f}, {rb['cp_two_sided_joint']['hi']:.3f}]).", ""]
    lines += ["## Population gaps induced by the numerical sampler alone (exact scores, no Monte Carlo)", "",
              "| experiment | condition | solver | steps | w | pop D(compared, reference) | pop D(reference, exact) |",
              "|---|---|---|---|---|---|---|"]
    for g in gaps:
        lines.append(f"| {g['exp']} | {g['cond']} | {g['solver']} | {g['n_steps']} | {g['guidance']} | "
                     f"{g['pop_D_compared_vs_reference']:.5f} | {g['pop_D_reference_vs_exact']:.5f} |")
    open(os.path.join(OUT, "reanalysis_v1.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
