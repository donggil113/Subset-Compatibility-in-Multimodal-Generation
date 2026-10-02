"""Assemble export_bundle/ for follow-up review (not an anonymous submission package).

Copies the manuscript source, configs, code, tests, small raw results and
aggregates, and the status/manifest files into export_bundle/ with the same
relative layout as the repository, then writes export_bundle/MANIFEST.sha256.
The hand-written files export_bundle/{README.md, BUILD.md, fetch_style.sh}
are left untouched. Nothing is deleted. No PDF is produced or included.

Excluded on purpose (available in the repository at the recorded commit):
* results/raw/p5_first_run_v1_{dev,test}/*.jsonl (about 5.8 MB of per-example
  rows). Their summary.json and manifest.json are included. Regenerating
  paper/generated/ needs the full repository, because
  scripts/make_paper_assets.py reads the v1 test rows.
* The official ICML 2026 style files (no redistribution licence; fetched by
  export_bundle/fetch_style.sh with a SHA-256 check).
* paper/p5_skeleton.tex (the round-1 skeleton, superseded by main.tex).
* The PDF built in round 6 is copied separately by the build step (paper/main.pdf) when it exists.

Usage: python3 scripts/make_export_bundle.py
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "export_bundle")

PATTERNS = [
    "paper/main.tex", "paper/fig_numerics.tex", "paper/references.bib", "paper/claims.csv", "paper/main.pdf", "paper/build_log_summary.json",
    "paper/generated/*",
    "configs/*.json",
    "src/p5compat/*.py", "scripts/*.py", "tests/*.py",
    "results/compute_ledger.csv", "results/p5_first_run_v1_test_tables.md",
    "results/errata/*", "results/calib_contract_audit/*", "results/reanalysis_v1/*",
    "results/raw/p5_first_run_v1_dev/summary.json", "results/raw/p5_first_run_v1_dev/manifest.json",
    "results/raw/p5_first_run_v1_test/summary.json", "results/raw/p5_first_run_v1_test/manifest.json",
    "results/raw/p5_e8_exact_v1/*", "results/raw/p5_e8_calib_v3/*",
    "results/raw/p5_syn_learn_01_dev/*", "results/raw/p5_syn_learn_01_test/*",
    "results/raw/p5_syn_learn_01_fm_train/*", "results/raw/p5_syn_learn_01_fm_train/checkpoints/*",
    "results/raw/p5_syn_learn_01_fm_dev/*", "results/raw/p5_syn_learn_01_fm_test/*",
    "results/env/*",
    "STATUS.md", "RESEARCH_PACKET.md", "RELATED_WORK.md", "run_manifest.json",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    os.makedirs(OUT, exist_ok=True)
    copied = []
    for pat in PATTERNS:
        matches = sorted(p for p in glob.glob(os.path.join(ROOT, pat)) if os.path.isfile(p))
        if not matches:
            raise SystemExit(f"pattern matched nothing: {pat}")
        for src in matches:
            rel = os.path.relpath(src, ROOT)
            dst = os.path.join(OUT, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    status = git("status", "--porcelain")
    info = {
        "built_from_head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty_at_build": bool(status),
        "note": "files are copied from the working tree; the bundle is committed in the same commit as these sources, so that commit (not built_from_head) is the exact source",
        "pdf_included": False,
        "compile_status": "COMPILE_NOT_RUN",
        "n_files_copied": len(copied),
    }
    with open(os.path.join(OUT, "bundle_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    lines = []
    for dirpath, _, files in os.walk(OUT):
        for name in files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, OUT)
            if rel == "MANIFEST.sha256" or "__pycache__" in rel:
                continue
            lines.append(f"{sha256(full)}  {rel}")
    lines.sort(key=lambda s: s.split("  ", 1)[1])
    with open(os.path.join(OUT, "MANIFEST.sha256"), "w") as f:
        f.write("\n".join(lines) + "\n")
    missing = [h for h in ("README.md", "BUILD.md", "fetch_style.sh") if not os.path.exists(os.path.join(OUT, h))]
    total = sum(os.path.getsize(os.path.join(OUT, l.split("  ", 1)[1])) for l in lines)
    print(f"copied {len(copied)} files; manifest {len(lines)} entries; {total / 1e6:.2f} MB")
    if missing:
        print("WARNING: hand-written files missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
