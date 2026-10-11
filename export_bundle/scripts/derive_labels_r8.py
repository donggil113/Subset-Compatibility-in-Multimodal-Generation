"""Derived labels for R8-P5-CPU-BUDGET (written after the run from summary.json; nothing in raw/ is changed).

The outcome label applies the R7 rule unchanged (configs/p5_syn_learn_01_fm_replicates_r7.json,
labels.outcome_labels): SINGLE_PROBE_MULTI_INITIALIZATION_EVIDENCE if all four replicate means of
d(128) share the sign of Theta_128 and their pointwise CIs exclude 0 in the same direction;
MIXED_REPLICATION otherwise. The comparison with R7 is descriptive (different test sets, equal
update counts vs equal process CPU; no causal or equal-everything claim).

Usage: python3 scripts/derive_labels_r8.py
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "raw", "p5_syn_learn_01_fm_r8")
R7 = os.path.join(ROOT, "results", "raw", "p5_syn_learn_01_fm_r7")
ARMS = ["INDEPENDENT_CONDITIONAL_FM", "SHARED_CONDITIONAL_FM"]


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    s, tr, man = load(os.path.join(BASE, "summary.json")), load(os.path.join(BASE, "training.json")), load(os.path.join(BASE, "manifest.json"))
    r8 = load(os.path.join(ROOT, "configs", "auto_run_r8.json"))
    r7rule = load(os.path.join(ROOT, "configs", "p5_syn_learn_01_fm_replicates_r7.json"))["labels"]["outcome_labels"]
    prim, sens = r8["solver"]["primary"], r8["solver"]["sensitivity"][0]
    out = {"derived_from": "summary.json (not raw); written after the run; outcome rule = R7 rule reused unchanged",
           "status": s["status"], "replicates_complete": s["replicates_complete"], "outcome_rule": r7rule}
    if s["status"] != "COMPLETE" or "analysis" not in s or "target" not in s.get("analysis", {}):
        out["outcome_label"] = f"NOT_LABELLED ({s['status']}; Theta over the four pairs not computed)"
    else:
        an = s["analysis"]
        th = an["target"][str(prim)]
        per = th["per_replicate_mean_d"]
        theta = th["Theta"]
        sign = theta["mean"] < 0
        same_sign = all((v["mean"] < 0) == sign for v in per.values())
        excl = all((v["hi"] < 0) if sign else (v["lo"] > 0) for v in per.values())
        out.update({
            "per_replicate_d128": per,
            "all_same_sign_as_Theta": same_sign,
            "all_pointwise_CIs_exclude_0_same_direction": excl,
            "outcome_label": "SINGLE_PROBE_MULTI_INITIALIZATION_EVIDENCE" if (same_sign and excl) else "MIXED_REPLICATION",
            "Theta_128": theta, "Theta_128_bootstrap": th["bootstrap_percentile_95"],
            "replicate_range_d128": th["replicate_range"], "sign_reversal_across_replicates": th["sign_reversal_across_replicates"],
            "joint_level": {"per_replicate": {k: round(v["mean"], 6) for k, v in an["joint"][str(prim)]["per_replicate_mean_d"].items()},
                            "Theta": an["joint"][str(prim)]["Theta"], "reversal": an["joint"][str(prim)]["sign_reversal_across_replicates"]},
            "Theta_by_steps_target": {str(h): round(1e3 * an["target"][str(h)]["Theta"]["mean"], 3) for h in (prim, sens)},
            "solver_sensitive_512_vs_128_target": {k: v[f"{sens}_minus_{prim}"]["SOLVER_SENSITIVE"] for k, v in an["solver_sensitivity"].items() if k.endswith("|target")},
        })
        out["n_solver_flags"] = sum(out["solver_sensitive_512_vs_128_target"].values())
        out["numerics_label"] = ("NUMERICS_NOT_SEPARATED for the flagged arms (paired 512-128 differences with CIs excluding 0); "
                                 "agreement elsewhere is not an ODE-limit proof" if out["n_solver_flags"] else
                                 "no paired 512-128 difference detected at this level (not an ODE-limit proof)")
    fam = s.get("families", {})
    out["target_family_rejections_after_holm"] = {k: v for k, v in fam.get("target_auxiliary", {}).get("holm_p", {}).items() if v <= 0.05}
    out["joint_family_rejections_after_holm"] = {k: v for k, v in fam.get("joint_secondary", {}).get("holm_p", {}).items() if v <= 0.05}
    out["labels"] = s.get("labels", {})
    out["training"] = {k: {kk: v[kk] for kk in ("updates", "cpu_used_seconds", "overshoot_seconds", "budget_reached", "all_losses_finite", "pattern_exposure", "complete")
                           if kk in v} for k, v in tr.items() if k != "warmup_not_charged"}
    out["warmup_not_charged"] = tr.get("warmup_not_charged")
    out["undertraining_flags"] = "NOT_DEFINED for CPU-budgeted training (no 90%/100% checkpoints); final iterates kept; dev loss at the end recorded in training.json"
    out["test_sampling_cpu_seconds"] = {k: {"sampling": b["timing_sampling"].get("cpu_seconds"), "test": b.get("timing_test", {}).get("cpu_seconds")}
                                        for k, b in s["blocks"].items() if k.endswith("|target|" + str(prim)) or k.endswith("|target|" + str(sens))}
    out["budget"] = man.get("budget")
    out["manifest_checks"] = {"code_hashes_match_registration": man.get("code_hashes_match_registration"), "code_dirty_at_run": man.get("code_dirty_at_run"),
                              "git_commit_at_run": man.get("git_commit_at_run"), "threads": man.get("threads"), "pid": man.get("pid")}
    # descriptive comparison with R7 (different test sets; equal updates vs equal CPU)
    try:
        d7 = load(os.path.join(R7, "derived_labels.json"))
        out["comparison_with_R7_descriptive"] = {
            "note": r8["evaluation"]["comparison_with_R7"],
            "R7": {"outcome_label": d7["outcome_label"], "Theta_128_x1e3": round(1e3 * d7["Theta_128"]["mean"], 2),
                   "per_replicate_x1e3": {k: round(1e3 * v["mean"], 2) for k, v in d7["per_replicate_d128"].items()}},
            "R8": {"outcome_label": out["outcome_label"],
                   "Theta_128_x1e3": round(1e3 * out["Theta_128"]["mean"], 2) if "Theta_128" in out else None,
                   "per_replicate_x1e3": {k: round(1e3 * v["mean"], 2) for k, v in out.get("per_replicate_d128", {}).items()}},
        }
    except FileNotFoundError:
        out["comparison_with_R7_descriptive"] = "R7 derived labels not found"
    with open(os.path.join(BASE, "derived_labels.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("status", "outcome_label", "n_solver_flags") if k in out}, indent=1))
    if "Theta_128" in out:
        print("Theta_128 x1e3:", round(1e3 * out["Theta_128"]["mean"], 3), [round(1e3 * v["mean"], 3) for v in out["per_replicate_d128"].values()])


if __name__ == "__main__":
    main()
