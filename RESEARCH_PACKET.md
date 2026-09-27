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

## 8. Calibration contract audit and v3 pre-registration (2026-09-27, before the run)

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
