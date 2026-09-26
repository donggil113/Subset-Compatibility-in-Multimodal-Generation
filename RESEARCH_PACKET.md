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
