# P5 export bundle (for follow-up review)

**P5: Do Any-to-Any Generators Define a Coherent Joint Distribution?**
Manuscript title: *Separating Sampling Error from Conditional
Incompatibility in Generative Models* (v3.1: v3 plus round-5 wording corrections).

This bundle is a copy of the repository files needed to review, rebuild and
re-check the round-5 state. It is an internal evidence package. The layout is the same as the repository's, so
the scripts run from this directory.

## What is here

| Path | Content |
|---|---|
| `paper/main.tex`, `paper/fig_numerics.tex`, `paper/references.bib`, `paper/generated/` | Manuscript v3 source. All numbers are generated from committed results. |
| `paper/claims.csv` | 40 claims with status and evidence paths; the core claims are K1–K3 |
| `BUILD.md`, `fetch_style.sh` | Exact build commands. The style is fetched from the official URL and SHA-256 checked. |
| `configs/` | All configs, including the never-run `p5_e8_calib_v2.json`, kept as a record |
| `src/`, `scripts/`, `tests/` | Code (Python 3.11 standard library) and the unit tests |
| `results/raw/p5_syn_learn_01_{dev,test}/` | P5-SYN-LEARN-01 outputs (summary + manifest). Dev is a smoke only. |
| `results/raw/p5_e8_calib_v3/`, `results/raw/p5_e8_exact_v1/` | Raw jsonl + summary + manifest |
| `results/raw/p5_first_run_v1_{dev,test}/` | Summary + manifest only (the per-example rows are in the repository) |
| `results/errata/` | Derived interval report (erratum E2) |
| `results/calib_contract_audit/`, `results/reanalysis_v1/`, `results/compute_ledger.csv` | Audits, re-analysis and cost ledger |
| `STATUS.md`, `RESEARCH_PACKET.md`, `RELATED_WORK.md`, `run_manifest.json` | Status, pre-registrations, errata, literature, run manifest |
| `bundle_info.json`, `MANIFEST.sha256` | Build provenance and file hashes (`sha256sum -c MANIFEST.sha256`) |

## What is not here

- **No PDF.** The manuscript was never compiled (COMPILE_NOT_RUN). See
  `BUILD.md`.
- **No ICML style files.** `icml2026.sty` has no redistribution licence;
  run `sh fetch_style.sh`.
- **No per-example v1 rows** (about 5.8 MB). They are in the repository.
- **No model weights and no datasets.** None exist: all data are synthetic
  and generated from seeds.
- **No neural flow results.** INDEPENDENT_CONDITIONAL_FM and
  SHARED_CONDITIONAL_FM are BLOCKED_DEPENDENCIES: there is no numpy/PyTorch,
  and installation was not approved. `src/p5compat/fm_adapter.py`, with its
  pre-execution fixes D1/D2, is UNTESTED. Its torch fixtures are SKIPPED.
  The pre-run contract is `configs/p5_syn_learn_01_fm_amendment_a1.json`.
  The only learned model with results is the EM-fitted GMM control.
- **No checkpoints.** No neural model was trained.

## Status in one paragraph

Preserved records:

- v1 EH1 **FAIL**;
- calibration v2 **NEVER RUN**;
- calibration v3 gate **PASS** (engineering evidence for this design only;
  not re-run);
- the model-stage STOP record.

Round 4 did three things:

- applied statistical errata E1–E6 (level ≤ α, not exact size; interval
  notation; CRPS vs Test P are different nulls; meaning of the gate PASS;
  naming; §8 timing);
- ran the coherent-joint mixture controls once. They are not flagged at the
  primary endpoint, and the fitted control is coherent but measurably
  wrong;
- produced manuscript v3.

Round 5 added:

- the FM arms, recorded as BLOCKED_DEPENDENCIES;
- the adapter static review;
- the pre-run amendment;
- wording corrections.

No claim about neural any-to-any generators is supported yet.

## Before any other use

- **Not an anonymous submission package.** It contains `STATUS.md`,
  `run_manifest.json` and `scripts/check_tex.py`. `check_tex.py` holds the
  anonymity patterns it searches for. Other files may identify the
  repository. Build a separate package for submission after checking the
  venue rules and third-party rights.
- The repository has no LICENSE file. Decide licensing before any public
  release.
