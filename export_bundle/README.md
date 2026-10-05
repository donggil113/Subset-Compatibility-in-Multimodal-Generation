# P5 export bundle (for follow-up review)

**P5: Do Any-to-Any Generators Define a Coherent Joint Distribution?**
Manuscript title: *Separating Sampling Error from Conditional
Incompatibility in Generative Models* (v5: four training replicates with solver sensitivity; compiled and raster-inspected).

This bundle is a copy of the repository files needed to review, rebuild and
re-check the round-7 state. It is an internal evidence package. The layout is the same as the repository's, so
the scripts run from this directory.

## What is here

| Path | Content |
|---|---|
| `paper/main.tex`, `paper/fig_numerics.tex`, `paper/references.bib`, `paper/generated/` | Manuscript v5 source. All numbers are generated from committed results. |
| `paper/main.pdf`, `paper/build_log_summary.json` | PDF built locally with TeX Live 2026 and the official ICML 2026 style (review mode); 21 pages, conclusion ends on page 8; pages rasterized with PyMuPDF and inspected (the summary lists which pages were viewed and what was fixed) |
| `paper/claims.csv` | 43 claims with status and evidence paths; the core claims are K1–K5 |
| `BUILD.md`, `fetch_style.sh` | Exact build commands. The style is fetched from the official URL and SHA-256 checked. |
| `configs/` | All configs, including the never-run `p5_e8_calib_v2.json`, kept as a record |
| `src/`, `scripts/`, `tests/` | Code (Python 3.11 standard library) and the unit tests |
| `results/raw/p5_syn_learn_01_{dev,test}/` | P5-SYN-LEARN-01 GMM-control outputs (summary + manifest). Dev is a smoke only. |
| `results/raw/p5_syn_learn_01_fm_{train,dev,test}/` | Neural flow arms (round 6): training.json, manifest.json, checkpoints (final iterates, sha256 in training.json), dev labels, summaries, blocks, per-example results, test samples (gzip) |
| `results/raw/p5_syn_learn_01_fm_r7/` | Round-7 replicates: registered test set, training.json, manifest.json, summary.json, derived_labels.json, dev labels/blocks, test blocks and per-example rows, eight checkpoints (sha256 in training.json). `test/samples.jsonl.gz` (8.8 MB) stays in the repository. |
| `results/env/` | Install records (torch wheel hash, pip freeze, TeX log, PyMuPDF), fixture runs, seed preflight, run logs |
| `results/raw/p5_e8_calib_v3/`, `results/raw/p5_e8_exact_v1/` | Raw jsonl + summary + manifest |
| `results/raw/p5_first_run_v1_{dev,test}/` | Summary + manifest only (the per-example rows are in the repository) |
| `results/errata/` | Derived interval report (erratum E2) |
| `results/calib_contract_audit/`, `results/reanalysis_v1/`, `results/compute_ledger.csv` | Audits, re-analysis and cost ledger |
| `STATUS.md`, `RESEARCH_PACKET.md`, `RELATED_WORK.md`, `run_manifest.json` | Status, pre-registrations, errata, literature, run manifest |
| `bundle_info.json`, `MANIFEST.sha256` | Build provenance and file hashes (`sha256sum -c MANIFEST.sha256`) |

## What is not here

- **The PDF was built locally and raster-inspected** (see
  `paper/build_log_summary.json`); to rebuild, see `BUILD.md`.
- **No ICML style files.** `icml2026.sty` has no redistribution licence;
  run `sh fetch_style.sh`.
- **No per-example v1 rows** (about 5.8 MB). They are in the repository.
- **No model weights and no datasets.** None exist: all data are synthetic
  and generated from seeds.
- **No Python environment.** The venv used for the neural pilot is not
  included; `BUILD.md` gives the exact recreation commands and the wheel
  hash (`results/env/torch_env.json`).
- **Checkpoints are included** (two pilot files and eight replicate files,
  about 150 KB each). They are final iterates; no other weights exist.

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

Round 6 (user-delivered limited approvals) added:

- the two neural flow arms, run once (status NEURAL_PILOT_AUTHORIZED_THIS_RUN;
  method validity and novelty UNVERIFIED): the independent arm is flagged at
  the target endpoint (p = 0.005), the shared arm is not (p = 0.425) but is
  flagged at the projected joint (p = 0.035) and is less accurate; paired
  difference −11.65 ×10⁻³ [−16.93, −6.38]; all endpoints solver-sensitive at
  the 10⁻³ level; one model seed per arm; not equal-compute;
- the compiled PDF.

Round 7 added four fixed training replicates on a new test set
(MIXED_REPLICATION: Θ₁₂₈ = −9.70 ×10⁻³ [−12.57, −6.83] with one sign
reversal; shared arm flagged at the joint level in 2 of 4; solver 128/256/512
effects below 10⁻³). No claim about real any-to-any generators is supported.

## Before any other use

- **Not an anonymous submission package.** Identifying material is present
  in `STATUS.md`, `run_manifest.json`, `scripts/check_tex.py` (its anonymity
  patterns name the author, institution and repository), `bundle_info.json`
  (branch name), `results/env/*` and `paper/build_log_summary.json` (local
  paths, session-specific scratch paths), `results/compute_ledger.csv`, the
  `%` header comments of `paper/main.tex` (commit hashes) and possibly other
  files. The typeset PDF itself carries the style's placeholder authors only.
  Build a separate package for submission after checking the venue rules
  and third-party rights; a string scan is not a proof of anonymity.
- The repository has no LICENSE file. Decide licensing before any public
  release.
