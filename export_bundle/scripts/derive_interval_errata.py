"""Derived (non-destructive) interval report for committed calibration / E8 results.

Erratum context: earlier documents wrote the pair (one-sided 95% lower bound,
one-sided 95% upper bound) as "one-sided CP95 [lo, hi]". Such a pair is an
equal-tailed two-sided 90% Clopper-Pearson interval. This script reads the
committed summaries (never rewriting them) and reports, per cell, the rejection
count, the one-sided 95% bounds actually used by the gates, the implied two-sided
90% interval, and the two-sided 95% Clopper-Pearson interval.

Usage: python3 scripts/derive_interval_errata.py
Writes results/errata/interval_report.json and .md
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from p5compat import stats as st  # noqa: E402


def row(label, k, n):
    lo1 = st.clopper_pearson(k, n, 0.95, "lower")["lo"]
    hi1 = st.clopper_pearson(k, n, 0.95, "upper")["hi"]
    two95 = st.clopper_pearson(k, n, 0.95, "two")
    two90 = st.clopper_pearson(k, n, 0.90, "two")
    return {"cell": label, "k": k, "n": n, "rate": k / n,
            "one_sided_95_lower": lo1, "one_sided_95_upper": hi1,
            "two_sided_90_equal_tailed": [two90["lo"], two90["hi"]],
            "two_sided_95": [two95["lo"], two95["hi"]]}


def main():
    out = {"source_files": [], "rows": []}
    cal = os.path.join(ROOT, "results", "raw", "p5_e8_calib_v3", "summary.json")
    s = json.load(open(cal))
    out["source_files"].append(os.path.relpath(cal, ROOT))
    for cid, b in s["cells"].items():
        out["rows"].append({"run": "p5_e8_calib_v3", **row(cid, b["rate_95"]["k"], b["rate_95"]["n"])})
    e8 = os.path.join(ROOT, "results", "raw", "p5_e8_exact_v1", "summary.json")
    s8 = json.load(open(e8))
    out["source_files"].append(os.path.relpath(e8, ROOT))
    for cid, b in s8["C"]["cells"].items():
        d = b["detector"]
        out["rows"].append({"run": "p5_e8_exact_v1", **row(cid, d["k"], d["n"])})
    os.makedirs(os.path.join(ROOT, "results", "errata"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "errata", "interval_report.json"), "w"), indent=1)
    lines = ["# Interval erratum report (derived; committed summaries unchanged)", "",
             "A pair of one-sided 95% Clopper-Pearson bounds is an equal-tailed two-sided 90% interval. "
             "Gates used the one-sided bounds exactly as pre-registered; nothing about any gate outcome changes.", "",
             "| run | cell | k/n | rate | one-sided 95% lower | one-sided 95% upper | two-sided 95% CP |",
             "|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append(f"| {r['run']} | {r['cell']} | {r['k']}/{r['n']} | {r['rate']:.4f} | {r['one_sided_95_lower']:.4f} | "
                     f"{r['one_sided_95_upper']:.4f} | [{r['two_sided_95'][0]:.4f}, {r['two_sided_95'][1]:.4f}] |")
    open(os.path.join(ROOT, "results", "errata", "interval_report.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
