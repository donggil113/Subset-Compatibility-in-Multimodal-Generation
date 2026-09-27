"""Pre-run contract audit of the P5-E8-CALIB configuration (no sampling of results).

For every cell of a calibration config it computes, in closed form on the
Gaussian probe, what the compared (sequential) branch changes relative to the
reference (direct) branch and relative to the truth:
  * target law  (z | x)              -> population ED on z (standardised)
  * projected joint law ((y, z) | x) -> mean projected ED over the 8 directions
  * truth / quality                  -> ED of each branch to the true law and the
                                        expected CRPS excess over the true conditional
It also computes the operating characteristics of the gate (exact binomial) and
the p-value resolution of the Monte Carlo permutation test against the
multiplicity families.

Usage: python3 scripts/audit_calib_contract.py configs/p5_e8_calib_v2.json [more configs]
Writes results/calib_contract_audit/<config stem>.json and .md
"""

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat import analytic as an  # noqa: E402
from p5compat import metrics as mt  # noqa: E402
from p5compat import samplers as sp  # noqa: E402
from p5compat import stats as st  # noqa: E402
from p5compat.experiment import make_joint  # noqa: E402

JOINTS = {
    "nonmarkov": {"mean": [0.5, -1.0, 2.0], "sd": [1.0, 2.0, 0.5], "corr": [[1.0, 0.6, 0.1], [0.6, 1.0, 0.6], [0.1, 0.6, 1.0]]},
    "markov": {"mean": [0.5, -1.0, 2.0], "sd": [1.0, 2.0, 0.5], "corr": [[1.0, 0.6, 0.36], [0.6, 1.0, 0.6], [0.36, 0.6, 1.0]]},
}
SCALE = {"y": 2.0, "z": 0.5}
KIND = {"exact_sequential": ("exact_sequential", None), "corr_scale_-1": ("stage2_corr_scale", -1.0),
        "chain_drop_x": ("chain_drop_x", None), "mean_shift_0.1": ("stage2_mean_shift", 0.1),
        "var_ratio_1.25": ("stage2_target_var_ratio", 1.25), "collapse_intermediate": ("collapse_intermediate", None),
        "collapse_both": ("collapse_both", None)}


def branch_laws(tc, cond_id, x):
    """Return ((mean_y, mean_z), cov2) for reference and compared branches, plus truth."""
    kind, p = KIND[cond_id]
    true_m, true_c = tc.yz_x.mean([x]), tc.yz_x.cov
    ref = (true_m, true_c)
    s1, s2 = tc.y_x, tc.z_xy
    if kind == "stage2_mean_shift":
        s2 = sp.stage2_mean_shift(tc, p)
    elif kind == "stage2_target_var_ratio":
        s2 = sp.stage2_target_var_ratio(tc, p)
    elif kind == "stage2_corr_scale":
        s2 = sp.stage2_corr_scale(tc, p)
    elif kind == "collapse_intermediate":
        s1 = sp.collapsed(tc.y_x)
    elif kind == "chain_drop_x":
        s2 = tc.z_y
    elif kind == "collapse_both":
        zero = [[0.0, 0.0], [0.0, 0.0]]
        return (true_m, zero), (true_m, zero), (true_m, true_c)
    t1, t2 = s1.temperature, s2.temperature
    m_y = s1.mean([x])[0]
    v_y = t1 ** 2 * s1.cov[0][0]
    if s2.given == ["x", "y"]:
        cx, cy = s2.coef[0]
        m_z = s2.intercept[0] + cx * x + cy * m_y
    else:
        cy = s2.coef[0][0]
        m_z = s2.intercept[0] + cy * m_y
    r2 = t2 ** 2 * s2.cov[0][0]
    cmp_ = ([m_y, m_z], [[v_y, cy * v_y], [cy * v_y, cy * cy * v_y + r2]])
    return ref, cmp_, (true_m, true_c)


def std2(law):
    (m, c) = law
    sy, sz = SCALE["y"], SCALE["z"]
    return [m[0] / sy, m[1] / sz], [[c[0][0] / sy ** 2, c[0][1] / (sy * sz)], [c[1][0] / (sy * sz), c[1][1] / sz ** 2]]


def target_d(l1, l2):
    (m1, c1), (m2, c2) = std2(l1), std2(l2)
    return an.energy_distance_gauss(m1[1], math.sqrt(c1[1][1]), m2[1], math.sqrt(c2[1][1]))


def proj_d(l1, l2, dirs):
    (m1, c1), (m2, c2) = std2(l1), std2(l2)
    tot = 0.0
    for u in dirs:
        a, sa = an.gaussian_projection(m1, c1, u)
        b, sb = an.gaussian_projection(m2, c2, u)
        tot += an.energy_distance_gauss(a, sa, b, sb)
    return tot / len(dirs)


def expected_crps_excess(branch_law, true_law):
    """E_{z ~ truth} CRPS(N(m', s'^2), z) - E CRPS(truth, z), on the standardised z scale."""
    (mb, cb), (mt_, ct) = std2(branch_law), std2(true_law)
    mb, sb = mb[1], math.sqrt(cb[1][1])
    m0, s0 = mt_[1], math.sqrt(ct[1][1])
    crps_b = an.e_abs_normal(mb - m0, math.sqrt(sb * sb + s0 * s0)) - sb / math.sqrt(math.pi)
    crps_t = s0 / math.sqrt(math.pi)
    return crps_b - crps_t


def pmf(k, n, p):
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p))


def prob(n, p, pred):
    return sum(pmf(k, n, p) for k in range(n + 1) if pred(k))


def gate_oc(cfg):
    """Operating characteristics of the gate declared in cfg (v2 or v3 semantics)."""
    alpha = 0.05
    out = {"cells": []}
    v3 = "families" in cfg
    cells = cfg["cells"]
    nulls = [c for c in cells if c["role"] == "null"]
    k0 = len(nulls)
    for c in cells:
        R = c["replicates"]
        rec = {"cell": f"{c['cond']}|{c['level']}|{c['joint']}", "role": c["role"], "R": R}
        if c["role"] == "null":
            if v3:
                lvl = 1 - alpha / k0
                kexc = min(k for k in range(R + 1) if st.clopper_pearson(k, R, lvl, "lower")["lo"] > alpha)
                tol = c.get("tolerance", False)
            else:
                kexc = min(k for k in range(R + 1) if st.clopper_pearson(k, R, 0.95, "lower")["lo"] > alpha)
                tol = True
            rec["excess_iff_k_ge"] = kexc
            rec["P_false_excess_at_size_0.05"] = prob(R, 0.05, lambda k: k >= kexc)
            rec["P_excess_at_size_0.10"] = prob(R, 0.10, lambda k: k >= kexc)
            if tol:
                ktol = max(k for k in range(R + 1) if st.clopper_pearson(k, R, 0.95, "upper")["hi"] <= 0.10)
                rec["tolerance_pass_iff_k_le"] = ktol
                rec["P_false_tolerance_fail_at_size_0.05"] = prob(R, 0.05, lambda k: k > ktol)
                rec["P_tolerance_pass_at_size_0.10"] = prob(R, 0.10, lambda k: k <= ktol)
            rec["P_cell_pass_at_size_0.05"] = prob(R, 0.05, lambda k: k < kexc and (not tol or k <= rec["tolerance_pass_iff_k_le"]))
        elif c["role"] == "alternative":
            kpow = min(k for k in range(R + 1) if st.clopper_pearson(k, R, 0.95, "lower")["lo"] > alpha)
            rec["powered_iff_k_ge"] = kpow
            rec["P_powered_given_power"] = {str(pi): prob(R, pi, lambda k: k >= kpow) for pi in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)}
        out["cells"].append(rec)
    p_null = 1.0
    for r in out["cells"]:
        if r["role"] == "null":
            p_null *= r["P_cell_pass_at_size_0.05"]
    out["P_all_null_cells_pass_if_exactly_valid"] = p_null
    for pi in ("0.2", "0.3", "0.5"):
        pp = 1.0
        for r in out["cells"]:
            if r["role"] == "alternative":
                pp *= r["P_powered_given_power"][pi]
        out[f"P_gate_pass_if_valid_and_each_power_{pi}"] = p_null * pp
    return out


def resolution(cfg):
    B = cfg["design"]["B_permutations"]
    alpha = 0.05
    fam = {}
    for m in (1, 2, 4, 8, 10, 11, 24):
        thr = alpha / m
        kmax = math.floor(thr * (B + 1)) - 1
        fam[str(m)] = {"per_test_threshold": thr, "attainable_size": max(kmax + 1, 0) / (B + 1),
                       "can_reject": kmax >= 0}
    return {"B": B, "p_min": 1 / (B + 1), "p_grid_step": 1 / (B + 1),
            "exact_size_at_alpha_0.05_continuous_T": math.floor(alpha * (B + 1)) / (B + 1),
            "largest_bonferroni_family_that_can_reject": math.floor(alpha * (B + 1)),
            "family_table": fam}


def main():
    out_dir = os.path.join(ROOT, "results", "calib_contract_audit")
    os.makedirs(out_dir, exist_ok=True)
    dirs = mt.slice_directions(8)
    for path in sys.argv[1:]:
        cfg = json.load(open(os.path.join(ROOT, path)))
        rows = []
        cond_ids = sorted({c["cond"] for c in cfg["cells"]} | {c["cond"] for c in cfg.get("quality_controls", [])})
        for cid in cond_ids:
            for jname in sorted({c["joint"] for c in cfg["cells"] + cfg.get("quality_controls", []) if c["cond"] == cid}):
                tc = sp.true_conditionals(make_joint(JOINTS[jname]))
                x = JOINTS[jname]["mean"][0]
                ref, cmp_, truth = branch_laws(tc, cid, x)
                rows.append({
                    "cond": cid, "joint": jname,
                    "target_D_cmp_vs_ref": target_d(ref, cmp_),
                    "joint_proj_D_cmp_vs_ref": proj_d(ref, cmp_, dirs),
                    "target_D_cmp_vs_truth": target_d(cmp_, truth),
                    "target_D_ref_vs_truth": target_d(ref, truth),
                    "joint_proj_D_cmp_vs_truth": proj_d(cmp_, truth, dirs),
                    "cov_yz_ref": ref[1][0][1], "cov_yz_cmp": cmp_[1][0][1],
                    "sd_ratio_cmp_z": math.sqrt(cmp_[1][1][1] / truth[1][1][1]),
                    "exp_crps_excess_cmp": expected_crps_excess(cmp_, truth),
                    "exp_crps_excess_ref": expected_crps_excess(ref, truth),
                })
        res = {"config": path, "contract": rows, "gate_oc": gate_oc(cfg), "p_resolution": resolution(cfg)}
        stem = os.path.splitext(os.path.basename(path))[0]
        json.dump(res, open(os.path.join(out_dir, stem + ".json"), "w"), indent=1)

        def yes(v, tol=1e-12):
            return "changes" if v > tol else "unchanged"

        lines = [f"# Calibration contract audit: `{path}`", "",
                 "Closed form on the Gaussian probe; D values on the standardised scale (y/2, z/0.5). "
                 "Reference = direct branch, compared = sequential branch.", "",
                 "| cell | target law (cmp vs ref) | projected joint law (cmp vs ref) | cmp vs truth (target) | ref vs truth (target) | Cov(y,z|x) ref -> cmp | sd ratio cmp | E CRPS excess cmp / ref |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['cond']} ({r['joint']}) | {yes(r['target_D_cmp_vs_ref'])} (D={r['target_D_cmp_vs_ref']:.5f}) | "
                         f"{yes(r['joint_proj_D_cmp_vs_ref'])} (D={r['joint_proj_D_cmp_vs_ref']:.5f}) | {r['target_D_cmp_vs_truth']:.5f} | "
                         f"{r['target_D_ref_vs_truth']:.5f} | {r['cov_yz_ref']:.3f} -> {r['cov_yz_cmp']:.3f} | {r['sd_ratio_cmp_z']:.3f} | "
                         f"{r['exp_crps_excess_cmp']:.4f} / {r['exp_crps_excess_ref']:.4f} |")
        g = res["gate_oc"]
        lines += ["", "## Gate operating characteristics (exact binomial; per-test size 0.05)", "",
                  "| cell | role | R | criteria | P(false fail or false EXCESS at size 0.05) | P(flag at size 0.10) |", "|---|---|---|---|---|---|"]
        for c in g["cells"]:
            if c["role"] == "null":
                crit = f"EXCESS iff k>={c['excess_iff_k_ge']}" + (f"; tolerance iff k<={c['tolerance_pass_iff_k_le']}" if "tolerance_pass_iff_k_le" in c else "")
                pf = 1 - c["P_cell_pass_at_size_0.05"]
                p10 = c["P_excess_at_size_0.10"] if "P_tolerance_pass_at_size_0.10" not in c else 1 - c["P_tolerance_pass_at_size_0.10"]
                lines.append(f"| {c['cell']} | null | {c['R']} | {crit} | {pf:.3f} | {p10:.3f} |")
            else:
                lines.append(f"| {c['cell']} | alternative | {c['R']} | POWERED iff k>={c['powered_iff_k_ge']} | "
                             f"P(POWERED) at power 0.05/0.2/0.3: {c['P_powered_given_power']['0.05']:.3f}/{c['P_powered_given_power']['0.2']:.3f}/{c['P_powered_given_power']['0.3']:.3f} | |")
        lines += ["", f"P(all null cells pass | exactly valid test) = {g['P_all_null_cells_pass_if_exactly_valid']:.3f}; "
                  f"P(gate pass | valid, each power 0.3) = {g['P_gate_pass_if_valid_and_each_power_0.3']:.3f}.", ""]
        r = res["p_resolution"]
        lines += ["## p-value resolution vs multiplicity", "",
                  f"B = {r['B']}: p takes values in multiples of {r['p_grid_step']:.4f} with minimum {r['p_min']:.4f}; "
                  f"exact size at alpha = 0.05 for a continuous statistic = {r['exact_size_at_alpha_0.05_continuous_T']:.3f}. "
                  f"A Bonferroni family of m tests can reject at all only if m <= {r['largest_bonferroni_family_that_can_reject']}.", "",
                  "| family size m | per-test threshold | attainable size | can reject |", "|---|---|---|---|"]
        for m, v in r["family_table"].items():
            lines.append(f"| {m} | {v['per_test_threshold']:.4f} | {v['attainable_size']:.4f} | {v['can_reject']} |")
        open(os.path.join(out_dir, stem + ".md"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
