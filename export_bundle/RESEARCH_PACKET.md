# RESEARCH_PACKET — P5: Do Any-to-Any Generators Define a Coherent Joint Distribution?

Status labels: PROVED (complete elementary proof given here), CONJECTURE / UNPROVED
(no proof), NOT_RUN (no experiment executed), ABSTRACT_ONLY (source read at
abstract level only). Engineering validity and scientific support are reported
separately (see STATUS.md).

## 0. Provenance and scope

- Repository state at the start of this work (2026-09-26): empty git repo, no
  commits, no CLAUDE.md / STATUS.md / RESEARCH_PACKET.md, no remote branches.
  There are no earlier Work IDs; P5 is **not** mapped to any existing Work ID.
  Experiment IDs below (`P5-E*`) are new.
- Environment: Python 3.11.15 standard library only. numpy / scipy / torch /
  pytest are not installed and were **not** installed (not approved). CPU only,
  single process (within the 2-thread limit), 120 s budget per synthetic run.
- External data / models / code used in FIRST RUN: **none**. All data are
  synthetic draws from Gaussians defined in `configs/p5_first_run.json`. All
  code under `src/`, `scripts/`, `tests/` is new in this repo (the repository
  has no licence file yet — decision for the owner).

## 1. Question, objects, and what is not new

**Question.** Under the same observed condition x, is direct generation
z ~ q_d(z|x) distributionally compatible with intermediate-then-target
generation y ~ q(y|x), z ~ q(z|x,y)? If a residual gap exists in learned
any-to-any generators, can it be reduced without quality loss?

**Not new (do not claim):** the marginalisation identity
p(z|x) = ∫ p(z|x,y) p(y|x) dy; any-to-any generation; the notion of
compatible conditionals (Arnold & Press 1989); the fact that learned
conditionals from one network can be mutually incompatible (dependency
networks, MLMs, AO-ARMs, tabular foundation models); marginalisation /
path-independence regularisers as a general idea (GMM, PiFM); "CFG does not
sample the intended conditional"; and — closest — the exact
direct-vs-marginalised-conditional diagnostic, already defined and measured for
tabular foundation models (arXiv 2608.06004). See RELATED_WORK.md.

**Objects.**

- Sequential target law: q_s(z|x) := ∫ q(z|x,y) q(y|x) dy.
- Target-level gap: Δ_T(x) := D(q_d(·|x), q_s(·|x)).
- Joint-level gap (when the model also has a direct joint sampler
  q_d(y,z|x)): Δ_J(x) := D(q_d(·,·|x), q(y|x) q(z|x,y)).
- D = energy distance; estimated from samples with an explicitly measured Monte
  Carlo noise floor (below). Independent evaluation unit = conditioning example.

**Proposition 1 (PROVED, elementary).** For the triple
(q(y|x), q(z|x,y), q_d(z|x)) there exists a joint r(y,z|x) whose y|x marginal is
q(y|x), whose z|x,y conditional is q(z|x,y) and whose z|x marginal is q_d(z|x)
iff q_s(·|x) = q_d(·|x).
*Proof.* (⇐) r := q(y|x) q(z|x,y) has the first two properties by construction
and z-marginal q_s = q_d. (⇒) any r with the first two properties equals
q(y|x) q(z|x,y) on {q(y|x) > 0}, so its z-marginal is q_s, which must equal q_d. ∎
So for this triple, target-level equality is exactly compatibility.

**Proposition 2 (PROVED, by construction).** Δ_T ≡ 0 does not imply Δ_J ≡ 0.
In the Gaussian probe, replacing the stage-2 coefficient b on (y − m_{y|x}) by c·b
and the residual variance by τ² − c²b²v_y (τ² = Var(z|x), v_y = Var(y|x))
leaves q_s(z|x) = p(z|x) exactly for every c, while Cov(y,z|x) becomes c·b·v_y
instead of b·v_y (`src/p5compat/samplers.py::stage2_corr_scale`; algebra checked
in `tests/test_gaussian.py`). A target-only checker is therefore blind to such
joint incoherence by construction.

**Proposition 3 (PROVED, elementary).** If every branch is deterministic and
returns the conditional mean (collapse), q_d = q_s = point mass at E[z|x], so
Δ_T = Δ_J = 0. For a Gaussian truth N(m, σ²) the expected CRPS of the point
mass is σ√(2/π) ≈ 0.798σ vs σ/√π ≈ 0.564σ for the true conditional. Hence
consistency must be reported together with a proper score and diversity.
Paired same-index MSE between two branches is 2·Var for independent exact
samplers and 0 for collapsed ones, so it is not a compatibility metric.

**Proposition 4 (PROVED, propriety argument).** Feeding the *observed*
intermediate y_obs (drawn jointly with z_obs) into q(z|x,y) gives, for a
strictly proper score S and the true conditionals,
E[S(p(·|x,y_obs), z_obs)] ≤ E[S(p(·|x), z_obs)]: conditional on (x, y_obs),
z_obs ~ p(·|x,y_obs), so by propriety p(·|x,y_obs) is optimal and in particular
no worse than p(·|x); average over y_obs. The observed-intermediate branch thus
wins on the proper score through information leakage while being a different
per-example distribution; it must not be used as the "sequential" branch in a
compatibility test.

**Known fact.** Chain generation z ~ q(z|y) (x dropped) is compatible with
p(z|x) iff z ⟂ x | y (Gaussian: partial correlation ρ_{xz·y} = 0). A mismatch is
then expected and is not generator incoherence.

## 2. FIRST RUN pre-registration (fixed before the test split was run)

Config: `configs/p5_first_run.json` (commit containing this file precedes the
test run). Design:

- Joints: `nonmarkov` (ρ_xy = 0.6, ρ_yz = 0.6, ρ_xz = 0.1; ρ_{xz·y} ≈ −0.41) and
  `markov` (ρ_xz = 0.36 = ρ_xy·ρ_yz). Means (0.5, −1, 2), sds (1, 2, 0.5).
- Splits (independent draws, separate seeds, unit = row): train n=2000 (only
  used for the standardising sds of y and z), dev n=40 (pipeline smoke only),
  test n=200 (reported). Diffusion experiments use the first 100 test rows.
- Per example i and branch: M = 256 samples (E5 varies M). A, B = two
  independent sets from the reference direct sampler, C = compared branch.
- **Primary statistic:** d_i = ED(A_i, C_i) − ED(A_i, B_i) (target z,
  standardised). Under H0, C_i and B_i are exchangeable given A_i, so d_i is
  symmetric about 0; decision = one-sided sign-flip test over examples,
  999 flips, α = 0.05. Effect size = mean d with normal 95% CI over examples.
  ED(A,B) itself is the reported Monte Carlo noise floor.
- **Joint statistic:** same with sliced energy distance over 8 fixed directions
  on standardised (y, z).
- **Quality metrics** per branch vs held-out z_obs: fair CRPS (proper score);
  oracle fidelity = mean true log density of samples (collapse-gameable);
  oracle diversity = sample sd / true sd; paired-index MSE (reported only as a
  demonstration that it is invalid).
- Budget 120 s wall-clock; if exceeded, remaining experiments are recorded as
  NOT_RUN (never as pass).

Engineering hypotheses (checker validity; NOT scientific claims):

| ID | Prediction | Pass criterion |
|---|---|---|
| EH1 | exact direct vs exact sequential is not flagged | P5-E1-NULL not detected at both levels; P5-E1-NULL-REPS FPR Wilson CI contains 0.05 or lies below it (R = 20 replicates) |
| EH2 | broken mean / variance is flagged | mean shift ≥ 0.1 τ and variance ratio ∈ {0.8, 1.25, 1.5} detected at target level |
| EH3 | correlation break with exact target marginal is invisible at target level and visible at joint level | corr_scale ∈ {−1, 0, 0.5}: target not detected, joint detected |
| EH4 | collapse passes consistency but fails proper score / diversity | collapse_both: d = 0, CRPS worse than exact direct, diversity 0; collapse_intermediate: detected |
| EH5 | observed intermediate is a category error | detected at target level and CRPS better than direct (Prop. 4) |
| EH6 | chain generation flagged iff non-Markov | detected on `nonmarkov`, not on `markov` |

Numerical-effect items (descriptive; expected from known theory, not novel):
E6 solver (exact score, PF-ODE Euler 2–64 steps, Heun 4/16), E7 CFG
(w ∈ {0.5, 1, 2, 4}, unconditional = target marginal), E5 Monte Carlo
(M ∈ {32, 64, 128, 256, 512}).

FIRST RUN stop rule: if EH1 fails, or the large effects in EH2 (mean 0.5,
variance 1.5) are missed, the checker is invalid and must not be used on models.

## 3. P5 decision rules for later phases (fixed now)

- **No new phenomenon** if, for a learned generator, the direct-vs-sequential
  gap at w = 0 with a converged solver and large M is within the noise floor,
  or if the gap is fully explained by the solver / CFG / Monte Carlo
  components measured as in E5–E7.
- **No method gain** if a reduction of the gap comes with a worse proper score
  or lower diversity (collapse), or is matched by the same-extra-compute
  baseline or by the shared-joint baseline.

## 4. Model-phase plan (all NOT_RUN)

- Architecture: frozen per-modality encoders/decoders + small conditional
  latent generator with per-modality masks/times (MLD / UniDiffuser-style).
- Baselines, strongest and simplest first:
  - B-shared-joint: one joint latent score model; all conditionals derived
    from it (UniDiffuser / MLD multi-time). This is the strongest direct
    baseline and the closest prior design.
  - B-simplest: define the direct output as the Monte Carlo sequential
    output (no new loss) — measures whether consistency is free by
    construction and at what compute.
  - B-extra-compute: the base model given exactly the extra training
    FLOPs / sampling NFEs that the method uses.
  - M-candidate: base loss + λ · sample-based discrepancy (energy distance /
    MMD between direct and sequential latent samples). Closest prior loss:
    Generative Marginalization Models (ICML 2024, discrete).
- Metrics: Δ_T / Δ_J with noise floor (latent and feature space), proper
  score (energy score in latent space; NLL where tractable), fidelity and
  diversity (precision / recall-type), guidance off for the primary
  analysis, CFG / solver / M analysed separately.
- Dataset: NOT_DECIDED (needs ≥ 3 paired modalities, licence and version
  recorded; splits by conditioning example / subject).

## 5. Post-hoc notes after the test run (added 2026-09-26; do not alter §2)

- Outcome summary and verdicts: STATUS.md. EH1 failed its pre-registered
  criterion; the stop rule in §2 is in force.
- The EH1 sub-criterion "P5-E1-NULL not detected" is mis-specified (an exact
  α-level test fails it with probability α). Recorded, not used to change the
  verdict; P5-E8-CALIB replaces it with a replicate-based calibration criterion.
- Seed-design flaw: seeds omit the joint name, so same-named conditions on
  the two joints share noise (see STATUS.md, Known issue 1).

## 6. Statistical audit of the v1 checker (2026-09-26; post hoc, v1 verdicts unchanged)

Scope: the v1 estimator, test and gate as committed in 51ed9f6 / 0f38af4.
Numerical checks below come from `scripts/reanalyse_v1.py`
(`results/reanalysis_v1/`), which reads only committed raw results and
regenerates the v1 test rows from their seed (digest-checked). They are
EXPLORATORY, because the v1 results had already been seen, and they are
numerical checks, not proofs.

### 6.1 Unit and dependence structure

- **Independent unit:** the conditioning example (one test-split row).
  Rows are i.i.d. and seeded per example. Generated samples, sign flips and
  permutations are not units.
- **Within an example:** the reference sets A, B and the compared set C are
  independent. Inside each set, samples are i.i.d.:
  - a fresh intermediate ŷ_j is drawn for every sequential sample;
  - every PF-ODE sample has its own initial noise.
- **Common random numbers across conditions:**
  - A, B (and the joint references A_J, B_J) are shared by all conditions of
    a run;
  - the P5-E5 sets at different M are nested prefixes.
  - So p-values across conditions are dependent. This is not a validity
    problem for any single test.
- **Shared noise across joints:** v1 seeds omit the joint and experiment
  names. `exact_sequential` and `chain_drop_x` on `nonmarkov` and `markov`
  therefore reuse identical noise. The two exact-null rejections
  (p = 0.026 and 0.015; studentised z = 2.00 and 2.16) are one event, not
  two.
- **Target and joint tests of one condition** share C, so they are
  dependent.
- **P5-E1-NULL-REPS replicates** share the same 200 rows and differ only in
  sampling seeds. Their rejection rate is conditional on the rows. This is
  valid under H0 because the per-example null holds at every x.

### 6.2 Estimands and estimators

- **V-statistic.** For M samples per set,
  E[ED_V(P_M, Q_M)] = D(P,Q) + (E|X−X'| + E|Y−Y'|)/M. So the v1 contrast has
  E[d] = D + (E|C−C'| − E|B−B'|)/M.
  - It is exactly 0 under H0.
  - It is biased under alternatives whose spread differs from the
    reference, and the bias is negative for under-dispersed C.
  - Closed-form predictions against the observed mean d:
    - Target level, 43 non-degenerate cells: |z| ≤ 2.16. Two cells exceed
      2, and they are the shared-noise exact-null pair E1-NULL / E4-MARKOV.
    - Joint level, 25 cells: |z| ≤ 2.60. Two cells exceed 2:
      collapse_intermediate and corr_scale_0.5.
    - Examples: var_ratio_0.8 predicted 0.0029 vs population D 0.0033;
      collapse_intermediate predicted 0.0216 vs D 0.0228.
    - (Corrected after commit 4b635f9, which misstated the cell count as 37
      and the bound as "about 2 SE".)
- **U-statistic.** ED_U(A,C) is unbiased for D with null mean 0 and needs no
  B set. It cannot be recomputed from v1 raw because samples were not
  stored: NOT_RUN for v1.
- **Studentisation.** The v1 decision used the sign-flip p-value of the
  *unstudentised* sum of d_i. z = mean/SE with a normal reference was only
  reported. Direct check of the reference: the sign-flip distribution of the
  studentised mean, given the observed |d_i|, has sd 0.985–1.014 and 95%
  quantile 1.61–1.69 (normal: 1.645) in all 7 v1 null cells, although d_i has
  kurtosis 4.3–7.6. At N = 200 the normal reference is adequate *given |d|*.
  The replicate z sd of 1.33 (χ² 95% CI [1.01, 1.95], R = 20) is weak
  evidence that cannot be separated from chance at this R.

### 6.3 Validity of the v1 test versus the v1 gate

- **The test.** Under H0, C_i and B_i are independent i.i.d. sets from the
  same law and independent of A_i. So (A_i, B_i, C_i) and (A_i, C_i, B_i)
  have the same distribution, d_i is symmetric about 0, and the rows are
  independent. The sign-flip test (a whole-set swap of B and C, two
  allocations per example) is therefore exact under these assumptions; this
  is standard randomisation logic, not a new result.
  - The single rejection at p = 0.026 does not show invalidity.
  - Nor do 2/20 replicate rejections (Clopper–Pearson 95% [0.012, 0.317]).
  - Its weaknesses are resolution and power: it spends a third sample set B,
    whose noise enters d, and it has only two allocations per example.
- **The gate (EH1)** was mis-specified independently of the test:
  - It required one α-level test not to reject, which fails with
    probability α even for an exact test.
  - It asked a 20-replicate FPR interval to "contain 0.05 or lie below it".
  - It had no power requirement, so an always-accept detector passes EH1.
  - **EH1 FAIL and the model-stage STOP remain recorded as issued.** They are
    not reinterpreted as PASS.

### 6.4 Target vs joint vs factorisation consistency

- Δ_T tests the compatibility of the triple (q(y|x), q(z|x,y), q_d(z|x))
  (Prop. 1).
- Δ_J compares the sequential pair (ŷ, z) with a *direct joint sampler*
  q_d(y,z|x).
- Factorisation consistency across orders, q(y|x)q(z|x,y) vs
  q(z|x)q(y|x,z), is a third estimand (C2 of arXiv 2608.06004). It is not
  measured here: NOT_RUN.
- None of these checks establishes that one global joint exists for all
  conditionals of an any-to-any model. Each checks one relation.

### 6.5 Numerical components isolated analytically (EXPLORATORY)

With exact scores, the PF-ODE (Euler or Heun, with or without CFG) is affine
in its inputs, so its output law is Gaussian and known in closed form
(`analytic.py`). Monte-Carlo-free population gaps on the standardised
scale:

- **Discretisation.** Direct vs sequential D is 0.00075, 0.00247 and 0.00026
  at Euler 2, 4 and 8 steps. From 16 to 64 steps it is 0.00016–0.00017.
- **Prior mismatch at σ_max.** The continuous-time limit with the standard
  N(0, σ_max²) initialisation gives D = 0.00017. This explains the 16–64
  step plateau: a third numerical component, not discretisation.
- **CFG.** D = 0.031 / 0.155 / 0.677 / 1.909 at w = 0.5 / 1 / 2 / 4.
- **Observed v1 d values** agree with these predictions (|z| ≤ 1.84).
- **"Not detected" is not "consistent."** At Euler ≥ 8 steps the population
  gap is nonzero but about 10× below the v1 detection resolution (SE of mean
  d ≈ 0.0006–0.0008).

### 6.6 Revised test (P5-E8)

- **Stratification:** labels are relabelled within each example; the 2M
  pooled samples are exchangeable under H0_i if both branches are i.i.d.
  inside and independent.
- **Statistic:** T = mean_i ED_V(a_i, c_i).
  - For equal group sizes, the V- and U-versions give identical
    permutation p-values. Proof: given the pooled multiset,
    S_aa + S_cc + S_ac is fixed, so both are increasing affine functions of
    S_ac, with slopes that depend only on M.
  - Vectors (y, z) are relabelled as units.
- **p-values:**
  - Monte Carlo: p = (1 + #{T_b ≥ T_obs − tol})/(B + 1), which is never 0.
  - Ties count as extreme.
  - Exact enumeration of the product group is used for small designs.
- **Invalid designs (not used):**
  - shared intermediates across target samples;
  - common random numbers between the branches;
  - relabelling across examples.
- **Pre-registration:**
  - `configs/p5_e8_exact_v1.json`: small exact verification, run in this
    session.
  - `configs/p5_e8_calib_v2.json`: realistic calibration and power, NOT_RUN.
  - Both were committed before any E8 result.

## 7. P5-E8 outcomes (post-run record, 2026-09-26; §6.6 design unchanged)

- **P5-E8-EXACT** ran at commit 4b635f9, clean code tree, in 1.1 s.
  - A: 55/55 datasets pass the super-uniformity check. PASS.
  - B: 10/10 agree between Monte Carlo and exact p. PASS.
  - C, null cells: 9/200, 4/100 and 1/50, all without an excess flag. This
    matches the pre-registered expectation NO_EXCESS_FLAGGED in all three
    cells.
  - C, alternatives: the 2τ shift is POWERED (42/50), matching its
    expectation. The variance ×4, correlation-flip (joint) and
    collapse-intermediate cells are NOT_POWERED at N = 3, M = 3. They were
    pre-registered as "report".
  - Negative-control rules fail as intended.
- These checks verify the implementation of a textbook principle on tiny
  designs. They do **not** calibrate the checker at realistic size.
- **P5-E8-CALIB: NOT_RUN.** The resource estimate (about 24 CPU-min) was
  filled from the E8-EXACT timing, as the config declared in advance.
- The model-stage STOP stands until P5-E8-CALIB passes.

## 8. Calibration contract audit and v3 pre-registration (decisions committed pre-run in 9ba6ad9 at 00:00:16Z; this text written during the run, committed 00:04:33Z; see §10 errata)

Scope: `configs/p5_e8_calib_v2.json` was never run and is preserved.
`scripts/audit_calib_contract.py` (closed form, no sampling) writes
`results/calib_contract_audit/`. The v1 EH1 FAIL and the model-stage STOP are
unchanged.

### 8.1 What each fixed alternative changes

The table below is closed form and uses the standardised scale. "Seq." is
the sequential branch.

| Cell | Target law (seq. vs direct) | Projected joint law | Truth / quality |
|---|---|---|---|
| mean_shift_0.1 (target) | changes (D = 0.0056) | changes (0.0033) | seq. biased by 0.1τ; E ΔCRPS 0.0028 |
| var_ratio_1.25 (target) | changes (0.0037) | changes (0.0023) | seq. over-dispersed (sd ratio 1.118); E ΔCRPS 0.0018 |
| collapse_intermediate (target) | changes (0.0227) | changes (0.110) | seq. under-dispersed (0.735); Cov(y,z\|x) 0.54 → 0; E ΔCRPS 0.0113 |
| corr_scale_−1 (joint) | **unchanged** (target null) | changes (0.065) | target exactly the truth; Cov flips 0.54 → −0.54 |

Notes on the alternatives:

- collapse_intermediate is a one-sided collapse. The two branch laws
  differ, so it is a legitimate compatibility alternative.
- Both-branch collapse (collapse_both) is a compatibility null: both
  branches are the same point mass. It is at the same time a quality
  failure, with E ΔCRPS 0.233 for both branches. It was not in the v2 power
  gate. v3 adds it as a separate quality-control family.
- The corr_scale_−1 perturbation is a null for the target test and an
  alternative for the joint test. It enters as two cells with different
  roles.

### 8.2 Sampling contract (checked in code and tests)

Checked:

- The direct branch draws M i.i.d. samples.
- Each sequential sample draws a fresh intermediate. `min_distinct_intermediates`
  = M is recorded per task.
- Branches use separate `seed_v2` streams, so there are no common random
  numbers.
- Permutations happen within an example only. Vectors are relabelled as
  units; a test rebuilds the vector groups and compares.

Found (E8-EXACT, round 2): `e8.make_dataset` omitted the level from its
seed key. Target- and joint-level cells of one condition and replicate
therefore shared rows and samples, so E8-EXACT C's target and joint cells
are not independent evidence. No verdict changes. The calibration keys
include the level (`tests/test_calib_contract.py`).

### 8.3 Identification

- The target test is exact under equality of the z-laws.
- The joint test is exact under full joint equality but only has power
  against differences in the 8 projected laws. It cannot certify joint
  equality.
- Neither test addresses correctness relative to the truth, nor the
  existence of a coherent joint for all conditionals.

### 8.4 Gate operating characteristics

All probabilities below are exact binomial, for a test whose size is
exactly 0.05.

**v2 gate:**

- Its tolerance "CP upper ≤ 0.10" fails a valid test with probability 0.56
  at R = 100 and 0.20 at R = 200.
- P(all v2 null cells pass | exactly valid) = 0.121. The v2 gate was not a
  usable contract.

**v3 gate:**

- **Primary null**, R = 400, with a tolerance criterion:
  - false fail 0.019;
  - detection at size 0.10: 0.964.
- **Null family (4 cells)**, EXCESS flags at the Bonferroni level
  1 − 0.05/4:
  - false EXCESS ≈ 0.011 per R = 100 cell;
  - detection at size 0.10: only 0.42 per secondary cell, which is a stated
    limitation.
- **Power:** each of 4 cells has R = 50 and is POWERED iff k ≥ 6:
  - P(POWERED | power 0.2) = 0.952;
  - P(POWERED | power 0.05) = 0.038.
- **Whole gate:**
  - P(all null cells pass | valid) = 0.948;
  - P(gate pass | valid, each power 0.3) = 0.945.

### 8.5 p-value resolution vs multiplicity

- B = 199 puts p on a grid of 0.005, and the size at α = 0.05 is exactly
  0.050.
- The actual families differ by level:
  - per replicate: one test, no adjustment;
  - gate: 4 null cells with Bonferroni EXCESS, and a conjunction of 4
    power cells;
  - later model evaluations: a Bonferroni family with m > 10 cannot reject
    at all with B = 199.

### 8.6 Computation

- **Array-at-a-time backend** (stdlib; numpy is not installed): identical
  to the loop backend on every allocation of the exact fixtures, including
  the joint case and ties. It gives identical exact and Monte Carlo
  p-values under the same RNG.
- **Speed:** it is about 0.6× the speed of the loop backend, because
  `rng.sample` dominates the cost. It is therefore unused.
- **Parallelism:** the only speed-up is 2 worker processes over independent
  tasks. Results do not depend on scheduling, because seeds are fixed per
  task.

## 9. P5-E8-CALIB v3 outcome (single run, 2026-09-27; post-run record)

- **Run.** Commit 9ba6ad9, clean code tree, 2 processes, 1192.8 s wall /
  2381 CPU-s, 920/920 tasks, no timeout. The run is not repeated.
- **Null validity.**
  - Primary: 16/400 (one-sided 95% CP [0.025, 0.060]), within tolerance.
  - Joint exact null: 5/100.
  - Target correlation null: 5/100.
  - Markov chain null: 4/100.
  - No Bonferroni excess flag.
- **Power.** All four POWERED:
  - 0.1τ shift: 45/50;
  - variance ratio 1.25: 26/50;
  - one-sided collapse: 50/50;
  - joint correlation flip: 50/50.
- **Quality control.** collapse_both was rejected 0/10 at the target level
  and 0/10 at the joint level. The quality flag (CRPS excess CI > 0 and sd
  ratio 0) holds in 10/10 replicates at each level.
- **Negative controls.** Always-accept fails G3; always-reject fails G1 and
  G2.
- **Gate:** PASS.
- **Interpretation (engineering, Gaussian probe, this design only).**
  - The implementation shows no excess rejection at the tested nulls and has
    power at the tested alternatives.
  - Size 0.05 is not proven by this.
  - Distributional correctness and coherent-joint existence are not
    addressed.
  - Validity for learned generators rests on A1–A3 holding for their
    sampling pipelines, which is unchecked.
- **Descriptive.** The per-replicate CRPS excess CI excludes 0 in only 12%
  (shift) and 14% (variance) of replicates, versus 90% / 52% rejections by
  the compatibility test. At N = 200 the proper score is far less sensitive
  to these small perturbations.
- **Records.** The v1 EH1 FAIL and the model-stage STOP stay on record. The
  STOP's lifting condition, "P5-E8-CALIB passes", is now met for the
  Gaussian probe. The model stage has not started.


## 10. Errata (2026-09-27, round 4; no re-run, configs and raw unchanged)

- **E1. Size statement.**
  - **Wrong.** "B = 199 ⇒ size at α = 0.05 is exactly 0.05" (§8.5; STATUS
    round 3; manuscript v2; claim C30).
  - **Correct.** B = 199 gives a p-value grid of step 1/200. With the
    identity counted (+1) and ties counted as ≥, the Monte Carlo
    permutation p-value is super-uniform under H0, i.e.
    P(p ≤ α) ≤ α (Phipson & Smyth 2010). This holds when:
    - the pooled samples of each example are exchangeable (A1–A3);
    - the statistic is fixed in advance;
    - the B allocations are drawn i.i.d. uniformly from the product group
      of within-example relabellings.
  - Equality P(p ≤ α) = ⌊α(B+1)⌋/(B+1) needs in addition that
    (T_obs, T_1..T_B) have no ties almost surely. That fails in general:
    - the group is finite;
    - the per-example ED_V is invariant under complementing the allocation;
    - tied data occur, e.g. collapse gives T constant, so p = 1.
  - **Code (`permutation.mc_permutation_test`):**
    - it satisfies the level ≤ α conditions: `random_mask` draws a uniform
      M-subset per example per replicate independently; the observed
      allocation is counted; ties use `>= T_obs - tol`;
    - it does not guarantee exact size.
  - **Consequence for the OC numbers.** The gate operating characteristics
    were computed at a null rejection rate of 0.05. That is the least
    favourable rate for a valid (level ≤ 0.05) test, so the stated
    false-fail probabilities are upper bounds for valid tests. "Exactly
    valid test" in §8.4 means "rejection rate equal to 0.05".
- **E2. Interval notation.**
  - **Wrong.** "one-sided 95% CP [0.025, 0.060]" for 16/400, and similarly
    elsewhere.
  - **Correct.**
    - Each number is a one-sided 95% bound: lower 0.0252, upper 0.0601.
    - Together they form an equal-tailed two-sided 90% interval.
    - The two-sided 95% Clopper–Pearson interval is [0.0230, 0.0641].
  - The gate used the one-sided upper bound (≤ 0.10) and the
    Bonferroni-level one-sided lower bound exactly as registered, so no gate
    outcome changes.
  - Derived report for all cells: `results/errata/interval_report.md`
    (`scripts/derive_interval_errata.py`).
- **E3. CRPS vs Test P.**
  - **Wrong.** Round-3 text compared "CRPS CI > 0 in 12% vs Test P 90%" as
    if it were a power comparison.
  - **Correct.** These are detection rates for different nulls:
    - the per-replicate CRPS CI tests whether the mean CRPS excess of the
      sequential branch over the *truth* is 0 (a quality null, using one
      held-out observation per example);
    - Test P tests equality of the *direct and sequential branch laws*.
  - Neither is more powerful for the other's hypothesis.
- **E4. Meaning of the Gaussian gate PASS.** It is engineering evidence
  about this implementation and design (N = 200, M = 64, B = 199, the
  Gaussian probe). The distribution-free validity of Test P comes from
  exchangeability and a fixed statistic. It does not guarantee an empirical
  false-positive rate of 0.05 on a new dataset or model, where A1–A3 must be
  checked for the sampling pipeline.
- **E5. Naming.** "Shared-joint baseline" (§3, §4; RELATED_WORK; STATUS;
  manuscript) is renamed **SHARED_CONDITIONAL_NET**: one multi-time or
  mask-conditioned network used for several conditionals. Sharing
  parameters does not make its conditionals those of one normalized joint.
  **JOINT_COHERENT_CONTROL** is reserved for models with a real normalized
  density whose conditionals are computed analytically from it.
- **E6. Timing of §8.** The contract decisions (config v3, audit outputs)
  were committed before the run. The §8 prose was written during the run.
  Round-3 manuscript edits committed during the run were result-independent
  but are not "pre-run".

## 11. P5-SYN-LEARN-01 outcome (single run, 2026-09-27; post-run record)

Config: `configs/p5_syn_learn_01.json`, committed pre-run in 161a432.
Nothing was changed after the run: data, K_fit, EM settings, N, M, B and
seeds are all as registered.

Runs:

- dev smoke: `results/raw/p5_syn_learn_01_dev/`, pipeline only, not
  reported;
- test: `results/raw/p5_syn_learn_01_test/`.

### 11.1 Results (test split)

- **TRUE_JOINT_CONTROL** (exact source mixture):
  - target Test P p = 0.075; Δ̂_T = 0.0027 [−0.0014, 0.0068];
  - projected-joint p = 0.035; Δ̂_J^U = 0.0021 [−0.0017, 0.0058];
  - quality to truth: exactly 0;
  - sd ratio of the direct branch 0.989 [0.976, 1.002].
- **FITTED_JOINT_CONTROL** (EM on train, K_fit = 2 < 3, declared
  misspecified; 37 iterations, converged, monotone):
  - target p = 0.460; Δ̂_T = 0.0002 [−0.0030, 0.0035];
  - projected-joint p = 0.250;
  - quality ED to truth 0.0301 [0.0253, 0.0349] (CRPS divergence 0.0151);
  - projected-joint ED 0.0210 [0.0181, 0.0238];
  - sd ratio 1.056 [1.031, 1.080];
  - held-out mean log-lik −3.623, against −3.390 for the source.
- **Primary family** (target tests of the models run): Holm-adjusted
  p = 0.15 (TRUE) and 0.46 (FITTED).
- **INDEPENDENT_CONDITIONAL_FM and SHARED_CONDITIONAL_FM:** NOT_RUN. There
  is no tensor library, and installation was not approved. The adapter
  `src/p5compat/fm_adapter.py` is UNTESTED.

### 11.2 Decision table

| Question | Outcome |
|---|---|
| Coherence vs quality to truth | Both controls are coherent by construction and are not flagged at the primary endpoint. FITTED is clearly worse in quality. **Consistency ≠ correctness.** |
| Separate vs shared learner gap, quality, cost | NOT_RUN (blocker above) |
| Gap explained by solver refinement vs remaining | Not applicable to the controls (exact sampling). NOT_RUN for learners. |
| Non-detection vs equality / correctness / global joint | For the controls, equality and a global joint hold by construction (a proof, not the test). Correctness fails for FITTED. For the neural flow learners (NOT_RUN) none of the three is established. *[Round 5: "learned models" narrowed; FITTED is an EM-learned model.]* |

### 11.3 Notes

- **TRUE joint p = 0.035.**
  - This is a rejection of a null that holds exactly, so it is a false
    positive.
  - Four control tests were run with independent streams given the test
    rows, so P(any p ≤ 0.05) ≤ 1 − 0.95⁴ ≈ 0.19. *[Round 5 correction:
    the independence was not proven. The bound used is the union bound
    ≤ 4 × 0.05 = 0.20.]*
  - The declared checks were done:
    - fixture tests pass (including the quadrature marginalization
      identity);
    - min distinct intermediates = 64;
    - direct and sequential seed keys differ.
  - It is not a FAIL, and it is not evidence of incompatibility.
- **Dirty flag.**
  - Both manifests record `code_dirty_at_run: true`.
  - `git diff 161a432 -- src configs tests scripts/run_syn_learn.py` is
    empty.
  - The dirt was documentation and the untracked
    `scripts/derive_interval_errata.py`, which the runner does not import.
  - The raw manifests are not rewritten.
- **Timing.** The dev smoke started 1 s after the pre-registration commit,
  and the test run 34 s after the dev smoke. For the GMM controls no
  solver-level choice exists, so dev informed nothing but the pipeline.
- **Scope.** This is one data seed, one model seed and one fitted control,
  so it is a pilot. It does not test any any-to-any generator.

## 12. Flow-matching arms: pre-execution amendment A1 (round 5; no FM run)

**Status: BLOCKED_DEPENDENCIES.** No torch or numpy is available, and no
installation was approved. No flow-matching code has been executed, and no
FM output exists.

The amendment is `configs/p5_syn_learn_01_fm_amendment_a1.json`. It leaves
`configs/p5_syn_learn_01.json` unchanged and fixes the fields the base left
open, before any execution:

- **Checkpoint:** the final iterate; no early stopping or selection.
- **Solver:** 128 steps is primary, fixed a priori; 32 steps is a
  sensitivity check.
- **MODEL_FIT_OR_NUMERICS_INCONCLUSIVE label:** any of
  - non-finite values;
  - dev-loss decrease of more than 2% between 90% and 100% of updates;
  - a solver-level disagreement larger than 1.645 combined SEs.
- **Primary family:** the two FM target Test P results, Holm.
- **Primary comparison:** the paired per-example ED_U difference between the
  arms, on the same 200 test examples. It is conditional on one model seed
  per arm.
- **Cap:** 1800 CPU-s, one thread. Otherwise INCOMPLETE_BUDGET.

Static review of `src/p5compat/fm_adapter.py` (original sha256 at ca0c350:
`abcc2ef3…`):

- **D1:** shared-network role ambiguity; fixed with a target mask.
- **D2:** one optimizer over the independent MLPs; fixed with per-network
  optimizers.

Everything else checked is correct: dimensions, path direction, velocity
sign, Euler direction, fresh intermediates, branch streams, and no batch
coupling.

Budget:

- 20 000 updates per arm (independent: 4 × 5000);
- parameters 34 757 vs 34 819 (by formula);
- about 4× training and sampling FLOPs for the shared arm, so the arms are
  not equal-compute.

Controls are reused from the round-4 run. Their per-example values were not
stored, so no FM-vs-control paired comparison is planned.

## 13. Neural flow arms: first execution (round 6, 2026-10-02)

**Authorization.** The user delivered a limited approval block in the
round-6 directive: a CPU-only torch install into an isolated venv, numpy from
PyPI, execution within 1 thread / 1800 CPU-s / 2400 wall-s, and a minimal
local TeX install for the PDF. Status of this stage:
NEURAL_PILOT_AUTHORIZED_THIS_RUN. The validity and novelty of the method
remain UNVERIFIED; the model-stage STOP record and the Gaussian calibration
PASS are unchanged, and the PASS is not a neural-performance PASS.

**Environment (results/env/torch_env.json).** Python 3.11.15, Linux x86_64,
torch 2.14.1+cpu (wheel
`torch-2.14.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl`, 196.2 MB, sha256
`5e38154c…34f37`, from https://download.pytorch.org/whl/cpu), numpy 2.4.6
(PyPI). CPU device and one thread verified with a real tensor. The venv
(985 MB) is not committed. Install wall time 58 s.

**Fixtures (results/env/fm_fixtures_run.json).** 12 torch fixtures pass
(2.1 CPU-s): roles distinguishable for all patterns including value 0 and
t = 0; observed and target roles disjoint; only target dims in loss and
velocity; conditioning holds exactly the given variables; no leakage of
unobserved variables (loss invariant to them); path t = 0 noise / t = 1
data, velocity x1 − x0; Euler forward; per-network parameter and
optimizer-state isolation; finite forward/backward and real updates on
small train-only batches (not a quality claim).

**D2 restated.** One Adam over the four independent MLPs is not an error in
general; the earlier contract relied on the None-vs-zero gradient
distinction. Per-network optimizers make the isolation explicit.

**Amendment A2** (`configs/p5_syn_learn_01_fm_amendment_a2.json`, registered
before any training; addenda added after the static review, still before
training): the dev-loss rule is a diagnostic only (checkpoint = final
iterate); 32 vs 128 steps compared as a paired per-example difference with
shared noise (sampling keys omit the step count); the A1 "1.645 × combined
SE" rule withdrawn; primary 128, secondary 32 regardless of outcome;
between-arm effect ED_U(SHARED) − ED_U(INDEPENDENT), example-paired,
samples unpaired; Holm over the fixed family of two FM target tests; no
log-likelihood for the flow arms; INCONCLUSIVE only for non-finite values;
INCOMPLETE_BUDGET only when the cap stops a stage.

**Static review before training.** A read-only adversarial review (five
lenses, two skeptics per finding) confirmed two major defects, fixed before
the run: a non-finite loss would have been reported as INCOMPLETE_BUDGET and
aborted the run; the stop-rule guard checked only the test summary. Minor
fixes: Holm over the fixed family; label after test for non-finite test
samples; manifest on failure; smoke budget; per-block summaries. Documented
without code change: dev and test share sampling and permutation streams by
the registered key scheme. The installed torch CPU generator uses the low 32
bits of a seed; all 2406 effective seeds of the run were checked distinct
(`results/env/seed_preflight.json`).

**Smoke.** Train-only timing smoke with run seed 999 (two attempts, 6.9 +
7.1 CPU-s; the first failed on a config-field bug fixed before anything
else ran): estimate 481 CPU-s for the full plan against 1782 CPU-s
remaining; the plan was run unchanged.

**Order actually followed.** fixtures → smoke → A2 addenda → pre-run commit
7e25f68 → training (INDEPENDENT, then SHARED) → dev evaluation and labels
written → test evaluation → nothing changed afterwards.

### 13.1 Outcome (test split; single run; commit 7e25f68, clean tree)

Raw: `results/raw/p5_syn_learn_01_fm_{train,dev,test}/` (training.json,
manifest.json, checkpoints with sha256, labels.json, summary.json,
blocks.jsonl, per_example.jsonl, samples.jsonl.gz).

| Arm | steps | p_T | p_J^U | Δ̂_T ×10³ [95% CI] | Δ̂_J^U ×10³ | D̂_U(direct, truth) ×10³ | D̂_U(seq, truth) ×10³ | sd ratio |
|---|---|---|---|---|---|---|---|---|
| INDEPENDENT | 128 (primary) | **0.005** (Holm 0.01) | 0.005 | 11.90 [7.19, 16.61] | 11.83 [7.01, 16.65] | 1.91 [0.02, 3.81] | 9.84 [6.95, 12.74] | 0.980 |
| INDEPENDENT | 32 | 0.005 | 0.005 | 10.80 [6.26, 15.33] | 11.27 [6.63, 15.91] | 2.19 | 10.18 | 0.952 |
| SHARED | 128 (primary) | 0.425 (Holm 0.425) | 0.035 | 0.24 [−2.75, 3.24] | 2.52 [0.28, 4.76] | 5.51 [3.52, 7.51] | 2.39 [0.40, 4.38] | 0.932 |
| SHARED | 32 | 0.295 | 0.010 | 0.59 [−2.35, 3.54] | 2.77 [0.59, 4.96] | 7.00 | 3.28 | 0.905 |

- **Primary family (Holm, 2 tests at 128 steps):** INDEPENDENT 0.01 (rejected),
  SHARED 0.425 (not flagged).
- **Between-arm paired effect** ED_U(SHARED) − ED_U(INDEPENDENT), same 200
  test examples, samples unpaired: target/128 **−11.65 ×10⁻³ [−16.93, −6.38]**
  (bootstrap [−16.86, −6.30]); target/32 −10.20 [−15.32, −5.09]; joint/128
  −9.30 [−14.43, −4.18]. Negative = shared arm has the smaller gap.
- **Solver sensitivity (paired, shared noise):** all four (arm × level)
  endpoints flagged on test: INDEPENDENT target +1.10 [0.79, 1.41], joint
  +0.55 [0.31, 0.80]; SHARED target −0.35 [−0.64, −0.06], joint −0.25
  [−0.42, −0.08] (×10⁻³, 128 minus 32). Quality shifts: SHARED direct
  −1.48 [−1.76, −1.21] (finer steps more accurate). By the pre-registered
  rule the numerical and learned components are NUMERICS_NOT_SEPARATED at
  the 10⁻³ level; the between-arm difference is an order of magnitude
  larger and has the same sign at both levels.
- **Dev labels (written before test):** both EVALUABLE; UNDERTRAINING_FLAG
  false (max relative dev-loss decrease 1.04%); dev target p: INDEPENDENT
  0.005, SHARED 0.585; SOLVER_SENSITIVE on dev: INDEPENDENT yes, SHARED no.
  Label after test: both EVALUABLE (no non-finite samples).
- **Cost:** INDEPENDENT 34 757 params, 20 000 updates (5000 per network),
  31.9 CPU-s training, 13.5 CPU-s test sampling; SHARED 34 819 params,
  20 000 updates (exposure 5052/4985/4953/5010), 46.7 CPU-s training,
  35.4 CPU-s test sampling. Not equal-compute. Whole run 220.2 CPU-s,
  226.7 wall-s, 1 thread; budget total 238.3 / 1800 CPU-s.
- **Checkpoints:** INDEPENDENT sha256 `e2e2a5c89188…` (150 229 B); SHARED
  `4f6154f5c426…` (143 185 B); final iterates.

### 13.2 Reading

- **Separate conditionals are detectably incompatible on this probe.** The
  independent arm's direct q(z|x) is close to the truth (1.9 ×10⁻³) while
  its sequential route is not (9.8 ×10⁻³): the gap is error accumulated over
  two learned conditionals.
- **The shared network has the smaller target gap, at worse direct quality,
  and its joint endpoint is flagged.** Non-detection at the target is not
  equality, and parameter sharing defines no joint (C38 NOT_ESTABLISHED).
  The joint-level rejection at target-level non-detection is the direction
  of Klötergens et al. Prop. 2 once more, not a new fact.
- **p-values are not scores.** The decisive quantity is the paired
  difference with its interval; one rejecting arm and one non-rejecting arm
  alone would not show a difference.
- **What this is not:** evidence about real any-to-any generators; a
  seed-replicated effect; an equal-compute comparison; a statement about
  the ideal continuous-time flow (the tests concern the fixed-solver output
  laws).
- **Decision table update:** "separate vs shared" is now measured once;
  "gap explained by solver" is flagged (not separated at 10⁻³);
  "non-detection vs equality / correctness / global joint" unchanged.
