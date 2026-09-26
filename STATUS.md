# STATUS — P5: Do Any-to-Any Generators Define a Coherent Joint Distribution?

Last update: 2026-09-26 (round 2: statistical audit, revised test, manuscript v1).
Round 1 (FIRST RUN) is kept unchanged below, under "History".

## Summary (round 2)

| Item | Status |
|---|---|
| v1 gate EH1 | **FAIL, kept.** Not reinterpreted. |
| Model-stage STOP | **In force.** The checker is not cleared for learned models until P5-E8-CALIB passes. |
| EH2–EH6 (v1 controls) | PASS (engineering) |
| Statistical audit of v1 | DONE. RESEARCH_PACKET.md §6; `results/reanalysis_v1/` (EXPLORATORY). |
| v2 seeds (joint / experiment / condition / example / replicate / branch; SHA-256) | IMPLEMENTED (`src/p5compat/seeds.py`). v1 code is left unchanged for reproducibility. |
| Within-example permutation test (Test P) | IMPLEMENTED (`src/p5compat/permutation.py`) |
| P5-E8-EXACT (small exact verification) | RUN, 1.1 s. Components A and B PASS. C: no excess under the nulls; powered only for the 2τ shift. D: timing. |
| P5-E8-CALIB (realistic calibration) | **NOT_RUN.** Needs about 24 CPU-min (request: 40 CPU-min), beyond the approved budget. |
| Learned generators (P5-REAL-01/02) | **NOT_TESTED** |
| Manuscript v1 | WRITTEN: `paper/main.tex`, ICML 2026 anonymous style. **COMPILE_NOT_RUN**; SUBMISSION_READY=false. |

The categories stay separate:

- **Engineering:** EH2–EH6 PASS, EH1 FAIL, E8-EXACT A/B PASS.
- **Toy results:** Gaussian probe only.
- **Real-model results:** none.
- **External utility:** none shown.
- **Novelty:** candidate only. The diagnostic, the definitions, the
  "marginal ⇏ joint" result and CFG distortion are prior work; see
  RELATED_WORK.md §G.

## Round 2: what was done

1. **Statistical audit** (RESEARCH_PACKET.md §6). The single null
   rejection (p = 0.026) and 2/20 replicate rejections do not show that the
   test is invalid. Clopper–Pearson 95% for 2/20 is [0.012, 0.317].
   - **The gate was mis-designed:** it required one α-level test not to
     reject, and it had no power requirement.
   - **The two exact-null rejections are one event:** the v1 seeds omit the
     joint name, so the non-Markov (p = 0.026) and Markov (p = 0.015) nulls
     reused the same noise.
   - **Test S is exact under its assumptions:** d_i is symmetric by
     exchangeability of B and C.
   - **The normal reference for z holds given |d_i|:** under sign flips the
     studentised mean has sd 0.985–1.014 and 95% quantile 1.61–1.69.
   - **Replicate z sd = 1.33** (χ² CI [1.01, 1.95], R = 20). This is weak
     and inconclusive.
2. **Estimator vs closed form (EXPLORATORY).** Closed-form expectations
   E[d] = D + (E|C−C'| − E|B−B'|)/M match the observed v1 means.
   - Target level: |z| ≤ 2.16 over 43 cells.
   - Joint level: |z| ≤ 2.60 over 25 cells.
   - Both are numerical checks, not proofs.
3. **Numerical components (EXPLORATORY; closed-form PF-ODE laws).**
   - **Discretisation.** The Euler direct-vs-sequential gap is 0.00247 at 4
     steps. From about 6 steps it drops below the resolution of 0.00143.
   - **Prior mismatch.** The gap plateaus at 0.00017, which equals the
     exact-flow limit. The plateau is prior mismatch at σ_max, not
     discretisation.
   - **Sampler bias.** Relative to the exact law it persists, e.g.
     D = 0.0095 at 16 steps.
   - **CFG.** CFG alone gives D = 0.031 / 0.155 / 0.677 / 1.909 at
     w = 0.5 / 1 / 2 / 4.
4. **P5-E8-EXACT** (`configs/p5_e8_exact_v1.json`, committed before the
   run; results in `results/raw/p5_e8_exact_v1/`).
   - **A. Full-enumeration super-uniformity:** holds on 55/55 datasets,
     including ties. PASS.
   - **B. Monte Carlo vs exact p:** agree within tolerance on 10/10; the
     minimum p is 0.0025 (never 0). PASS.
   - **C. Exact-p rejection rates at |G| = 8000:**
     - Nulls: 9/200 (one-sided CP [0.024, 0.077]), 4/100 and 1/50. No
       excess flagged.
     - 2τ shift: 42/50, POWERED.
     - Variance ×4, joint correlation flip and collapse-intermediate:
       NOT_POWERED at this tiny size.
     - Negative controls behave as intended: always-accept is NOT_POWERED,
       and always-reject is flagged for excess.
   - **D. Timing:** used for the E8-CALIB estimate.
5. **Literature, second pass.** Full-text sections of 2608.06004,
   UniDiffuser, MLD and GMM were read directly (RELATED_WORK.md §F).
   Consequence: our former "Prop. 2" (target ⇏ joint) is Prop. 2 of
   2608.06004 and was removed from the contributions. Two further close
   works named there, Yalavarthi et al. (2026) and Young (2026), are
   NOT_CHECKED.
6. **Manuscript v1** (`paper/main.tex`, `paper/fig_numerics.tex`,
   `paper/generated/*`, `paper/references.bib`, `paper/claims.csv`).
   - All sections are written: Abstract, Introduction, Related Work,
     Problem Setup, Method, Experiments, Limitations, Conclusion, Impact
     Statement, Appendix.
   - Tables, figure data and all numbers are generated from raw by
     `scripts/make_paper_assets.py`.
   - Open TODOs: P5-REAL-01 (twice), P5-E8-CALIB and P5-REAL-02.
   - `scripts/check_tex.py` passes. It is a static check, not a compile:
     33/33 citations, all references and macros defined, environments and
     braces balanced, no anonymisation patterns.
   - Page count, layout and figure rendering are NOT verified. The body is
     about 5000 words including tables.
7. **Re-aggregation fix.** The collapse_both ΔCRPS in the paper table is now
   computed against the exact direct sampler, pairing examples from raw.
   The v1 summary compared it with the collapsed reference, which is
   trivially 0. The raw data are unchanged.

## Manuscript flags

- TARGET_YEAR = 2027. TEMPLATE_YEAR = 2026: the official ICML 2026 style,
  anonymous review mode. No official ICML 2027 style was found on
  2026-09-26.
- SUBMISSION_READY = false.
- COMPILE_STATUS = COMPILE_NOT_RUN: no pdflatex, latexmk or tectonic in this
  environment.
- The style kit (icml2026.zip, official URL) was read in the session
  scratchpad only to confirm the macro names. It is not vendored in the
  repo.
- The file `paper/p5_skeleton.tex` (round 1) is superseded by
  `paper/main.tex` and kept as history.

## Compute (all CPU, single process; `results/compute_ledger.csv`)

- **Sampling runs including smoke runs: about 122.7 s in total.**
  - round 1: dev 18.7 + 19.0 s, test 80.9 s, timing about 3 s;
  - round 2: E8-EXACT 1.1 s, E8 smoke 0.04 s.
- **Analysis and unit tests:** about 17 s, with no new sampling.
- **Budget interpretation.** It is ambiguous whether the approved "first
  verification: 120 s synthetic" is a per-run or a total cap. Every
  individual run stayed under 120 s. In total the cap is about 3 s over if
  the smoke runs are counted. Round 2 therefore ran only the small exact
  verification (1.1 s).

## Known issues (additions in round 2)

- v1 known issues 1–6 below still apply.
- During the literature pass, three arXiv HTML files were written to the
  repo root by mistake. They were moved to the scratchpad immediately and
  never committed.
- The subagent bibliography pass left PARTIAL entries (venue or pages
  unverified):
  - Klötergens 2026, Kim 2026, MODUS, Young (arXiv only);
  - Bradley & Nakkiran and Ho & Salimans (workshop only);
  - MMVAE and MVAE (pages missing).
- The E8-CALIB config was re-serialised when its declared resource-estimate
  field was filled. Design fields are unchanged.

## Next decision experiment (one)

**P5-E8-CALIB** (`configs/p5_e8_calib_v2.json`, frozen). This is the only
step that can lift the model-stage STOP.

- **Design:** within-example permutation test, N = 200, M = 64, B = 199.
- **Primary endpoint:** the exact-null target rejection rate over R = 200
  replicates with fresh rows.
- **Secondary null cells:**
  - the joint null (R = 200);
  - the target-level correlation null (R = 100);
  - the Markov chain null (R = 100).
- **Four fixed alternatives, R = 50 each:**
  - a 0.1τ shift;
  - variance ×1.25;
  - collapse-intermediate;
  - the joint correlation flip.
- **Gate:** every null cell has no excess flag and a one-sided CP upper
  bound ≤ 0.10, AND all four alternatives are POWERED. Passing does not
  prove size 0.05.
- **If it fails:** report FAIL and do not change estimator, seeds or
  thresholds.
- **Unapproved resource needed:** about 40 CPU-minutes of pure-Python CPU (2
  processes, about 20 min wall), or approval to install numpy (not
  requested).

---

# History: round 1 (FIRST RUN), unchanged

Last update: 2026-09-26. Pre-registration: RESEARCH_PACKET.md §2, committed in
51ed9f6 before the test split was run. Raw data: `results/raw/`. Rendered
tables: `results/p5_first_run_v1_test_tables.md`.

## Summary

| Item | Status |
|---|---|
| Prior work in this repo | None: empty repo, no STATUS / STOP / ARCHIVE, nothing overridden. |
| FIRST RUN, test split | Executed. 9/9 blocks COMPLETED in 80.9 s (budget 120 s), CPU, single process, stdlib only. |
| Engineering: checker validity | EH2–EH6 PASS. **EH1 FAIL** against its pre-registered criterion. Null calibration is **not established**. |
| Stop rule (RESEARCH_PACKET §2) | **Triggered by EH1.** The checker is not cleared for use on learned models. |
| Scientific question on learned generators | **NOT_TESTED.** No learned model exists yet. |
| Model phase: encoders, latent generator, baselines, consistency loss | NOT_RUN |
| Full-text reading of closest papers | NOT_DONE. Abstract-level only; see RELATED_WORK.md. |

## Runs

| run_id | split | code | runtime | blocks |
|---|---|---|---|---|
| p5_first_run_v1_dev | dev (n=40) | Same code as 51ed9f6, run before the first commit | 19.0 s | 9/9 COMPLETED. Pipeline smoke only. It exposed a degenerate-sd bug, which was fixed before the commit. |
| p5_first_run_v1_test | test (n=200; diffusion blocks use the first 100 rows) | 51ed9f6. The manifest marks the tree dirty only because of the run's own output folder; src / configs / tests are identical to 51ed9f6. | 80.9 s | 9/9 COMPLETED |

Tests: `python3 -m unittest discover -s tests` passes all 27. These are
numerical checks, not proofs.

## Engineering results: checker validity (test split, α = 0.05, one-sided sign-flip test over examples)

Primary statistic: d = ED(A,C) − ED(A,B), mean over examples with 95% CI.
The noise floor ED(A,B) at M = 256 is 0.0082; theory gives 0.0088.

| ID | Result | Verdict |
|---|---|---|
| EH1 null | P5-E1-NULL exact direct vs exact sequential: d = 0.0012 [0.0000, 0.0024], p = 0.026 → **detected** at target level. Joint dJ = 0.0006, p = 0.057, not detected. Replicates (R = 20): target FPR 2/20 = 0.10, Wilson [0.028, 0.301]; joint 1/20 = 0.05, [0.009, 0.236]. The z statistic over replicates has sd 1.33 (target) and 0.88 (joint). The P5-E4-MARKOV exact null also rejects (p = 0.015), but it shares random numbers with E1-NULL (Known issue 1), so it is not independent evidence. | **FAIL** (pre-registered criterion). Calibration inconclusive. |
| EH2 mean / var breaks | Detected at the target level: mean shift 0.1τ (d = 0.0044), 0.2τ, 0.5τ; variance ratio 0.8, 1.25, 1.5. Also detected: mean 0.05τ (p = 0.006, descriptive). Variance ratio 1.1 is not detected at target level (p = 0.16) but is detected at joint level (p = 0.017). | PASS |
| EH3 correlation break, exact target marginal | corr scale −1 / 0 / 0.5: target p = 0.46 / 0.62 / 0.28, not detected. Joint dJ = 0.066 / 0.018 / 0.007, all p = 0.001, detected. | PASS |
| EH4 collapse | collapse_both: d = 0 exactly (p = 1) at both levels, so consistency "passes". CRPS is 0.8145 vs 0.5753 for exact direct, 41.6% worse (theory 41.4%). Diversity 0. Oracle fidelity −0.221 vs −0.723: collapse looks *better*. Paired-index MSE/τ² is 0.000 vs 2.016 for exact. collapse_intermediate is detected (d = 0.023, dJ = 0.112), with diversity 0.73 and paired MSE 1.54 (the invalid metric "improves"). | PASS |
| EH5 observed intermediate | Detected: d = 0.321. CRPS is better than direct by −0.180 [−0.229, −0.131], i.e. information leakage (Prop. 4). | PASS |
| EH6 chain z\|y | nonmarkov: detected, d = 0.039. markov: not detected, p = 0.67. | PASS |

## Numerical-effect results (descriptive; exact scores; not novel in principle)

- **Monte Carlo (P5-E5).**
  - The noise floor follows 2E|X−X′|/M:

    | M | observed | theory |
    |---|---|---|
    | 32 | 0.0665 | 0.0706 |
    | 64 | 0.0351 | 0.0353 |
    | 128 | 0.0160 | 0.0177 |
    | 256 | 0.0082 | 0.0088 |
    | 512 | 0.0043 | 0.0044 |

  - A 0.1τ mean shift is detected at every M from 32 to 512 with N = 200. At M = 512, d = 0.0051 [0.0041, 0.0062] vs a population ED of 0.0057.
  - None of the 4 extra null conditions (M = 32 to 512) reject.
- **Solver (P5-E6, PF-ODE, EDM schedule, N = 100).**
  - Euler 2 steps: both branches nearly collapse (diversity 0.010 / 0.016) and d is tiny (0.0008) but detected. With 4 steps it is still detected (d = 0.0012, p = 0.014). From 8 to 64 steps nothing is detected.
  - The direct sampler is biased relative to the truth even when nothing is detected: at 16 steps its diversity is 0.83 (vs exact, d = 0.0096, detected); at 64 steps its log sd ratio is −0.042 [−0.055, −0.030], although the ED test does not reject (p = 0.11).
  - So in this toy most of the solver error is common-mode and cancels in the direct-vs-sequential comparison. Consistency does not certify correctness.
  - Heun with 4 steps is unstable: diversity 4.35 / 3.59, d = 0.041. Heun with 16 steps is not detected.
- **CFG (P5-E7, Euler 64, N = 100).**
  - Direct vs sequential is detected for every w > 0, with d = 0.031 / 0.160 / 0.675 / 1.894 at w = 0.5 / 1 / 2 / 4. The noise floor is about 0.008.
  - The per-example discrepancy is large even though the example-averaged mean difference has a CI containing 0. This is why the unit must be the conditioning example.
  - Guided direct vs exact: d = 0.0057 at w = 1 and 0.085 at w = 4. The sequential branch is far further off, because guidance is applied at both stages.
  - With guidance on, a direct-vs-sequential gap cannot be attributed to the model.

## Scientific status

- The central question — whether learned any-to-any generators have a
  direct-vs-sequential gap beyond numerics, and whether it can be reduced
  without quality loss — is **NOT_TESTED**.
- The FIRST RUN supports only toy-level statements, which are elementary or
  already known:
  - the identity holds for exact samplers;
  - a target-only test is blind to joint breaks;
  - collapse passes consistency;
  - observed intermediates leak information;
  - CFG and few-step solvers create gaps.
- Toy-specific observations, **not** claimed to generalise:
  - moderate solver error cancels between the branches;
  - the CFG gap is magnified in the sequential branch.
- Novelty caveat: the direct-vs-marginalised diagnostic itself is already
  published for tabular foundation models (arXiv 2608.06004; ABSTRACT_ONLY).
  See RELATED_WORK.md.

## Known issues (none of them were used to change a verdict)

1. **Seed design.** Sample seeds do not include the joint name. Same-named
   conditions on `nonmarkov` and `markov` therefore reuse the same noise:
   `exact_sequential` and `chain_drop_x` in E1 vs E4-MARKOV. They are not
   independent replications. Fix for v2: add the joint name to
   `derive_seed`. The test run was not repeated.
2. Within one run, conditions share the reference draws A and B (common
   random numbers, by design), and the E5 sets at different M are nested
   prefixes. Results across conditions are correlated.
3. The EH1 sub-criterion "the single null test is not detected" was
   mis-specified. An exact α-level test fails it 5% of the time. The verdict
   stays FAIL; this is noted post hoc and was not used to reclassify.
4. The calibration evidence is weak either way:
   - R = 20 replicates is too few;
   - the replicate z has sd 1.33, whose chi-square CI of about [1.01, 1.95]
     sits just above 1;
   - in E1-NULL the sample skewness of per-example d is 0.46, although d is
     symmetric under H0 by construction.
5. `scripts/make_tables.py` rendered the replicate table empty. This was a
   rendering bug and was fixed after the run. The raw data and summary.json
   are unchanged.
6. Environment:
   - no numpy, so everything is pure Python;
   - the web-fetch tool cached one public PDF (Kuo & Wang) outside the repo
     as a side effect; it was not used;
   - the repository has no LICENSE file.

## Next decision experiment (one)

**P5-E8-CALIB — null calibration of the checker.** This must happen before
any model is evaluated, because the stop rule currently blocks model use.

- **Seeds:** v2 seeds, with the joint name included.
- **Replicates:** R = 200 independent null replicates of exact direct vs
  exact sequential, each with fresh conditioning rows.
- **Design:** N = 200, M = 256, both levels. Also add the U-statistic variant
  of d as a pre-registered secondary.
- **Pass:** target and joint FPR Wilson upper bound < 0.10 with 0.05 inside
  or above the CI, and a replicate-z sd CI that covers 1.
- **Fail:** replace the aggregate test, for example with a per-example
  permutation test, then re-calibrate.
- **Cost:** about 300 s single-process, or about 150 s on 2 processes. This
  is above the 120 s first-run cap and needs approval. R ≈ 120 on 2
  processes fits inside the cap, at lower precision.
