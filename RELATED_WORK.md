# RELATED_WORK — P5

Check date: 2026-09-26. Method: web search and page fetches (read-only), no
file downloads on purpose. One side effect: while fetching the Kuo & Wang PDF,
the fetch tool cached the binary under the session's tool-results folder
outside this repo. It was not opened or used.

Access levels, most to least direct:

- **ABSTRACT_ONLY (verified)**: I fetched the arXiv abstract page myself and
  checked the title, authors, date and abstract.
- **SUMMARY_FULL_TEXT (subagent)**: a delegated search agent fetched the
  full-text HTML and got a tool-written summary. Nobody read the paper line by
  line, so treat its details as unverified until the paper is read.
- **ABSTRACT_ONLY (subagent)**: the subagent saw only the abstract.
- **SNIPPET_ONLY**: only a search-result snippet was seen.

**No paper here was read in full.** The column "Direct-vs-seq test?" asks
whether the paper measures whether direct and intermediate-then-target
generation give the same *distribution*.

## A. Closest prior work (the direct-vs-marginalised diagnostic already exists)

| Work | Access | What it establishes | Direct-vs-seq test? | Remaining difference to P5 |
|---|---|---|---|---|
| Klötergens, Yalavarthi, Schmidt-Thieme, Hanika. *Do Tabular Foundation Models Agree with Themselves?* arXiv 2608.06004 (6 Aug 2026) | FULL_TEXT_SECTIONS_READ (second pass; see §F) | Defines *marginalization consistency* (marginalised conditionals must equal directly predicted marginals) and *factorization consistency*. Every TFM they evaluate violates both. | **YES**, for tabular predictors. The subagent's summary says total-variation distance on TabPFN / TabICL / TabDPT / TabFM; this is not verified. | Tabular prediction, not multimodal generation. Per the summary it is diagnostic only, with no fix proposed. **P5's definition of Δ_T is essentially their C1**: we cannot claim the diagnostic itself. |
| Kim. *Path-Dependent Denoising: A Non-Conservative Field Perspective on Order Collapse in Diffusion Language Models.* arXiv 2605.09303 (10 May 2026) | ABSTRACT_ONLY (verified) | Order-induced pseudo-joints, and a local "circulation" that is zero under compatible conditionals, for diffusion LMs. | Order dependence, not direct-vs-intermediate. | Text only. The subagent reported a proposed regularizer; the abstract does not mention one, so this is UNVERIFIED. |
| Téllez et al. *Path-independent Flow Matching for Multi-parameter Generative Dynamics.* arXiv 2605.13487 | SUMMARY_FULL_TEXT (subagent) | In multi-parameter flows the final distribution can depend on integration order. Measured with W2 and reduced with a commutativity regularizer. | No, it concerns path order. | Not about multimodal conditionals. |
| Liu, Ramadge, Adams. *Generative Marginalization Models.* ICML 2024 (PMLR 235), arXiv 2310.12920 | FULL_TEXT_SECTIONS_READ (§F) | A marginalization self-consistency loss (squared log-space) for discrete models. | Partial: it checks marginal estimates, not sampling-based direct-vs-chain. | Discrete data. The closest prior *loss* for P5's method candidate. |

## B. Any-to-any / multimodal diffusion

| Work | Access | What it establishes | Direct-vs-seq test? | Remaining difference |
|---|---|---|---|---|
| Bao et al. *One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale* (UniDiffuser). ICML 2023 (PMLR 202), arXiv 2303.06555 | FULL_TEXT_SECTIONS_READ (§F) | Predicts the noise of all modalities with a separate timestep per modality. Marginal, conditional and joint are special cases of the timesteps. | NO. Chained / Gibbs-like sampling is shown qualitatively; the metrics are FID and CLIP. | This is the natural **shared-conditional-network baseline (SHARED_CONDITIONAL_NET; renamed from "shared-joint", erratum E5)**. Its conditionals come from one network, but sharing parameters does not make them conditionals of one normalized joint. |
| Bounoua, Franzese, Michiardi. *Multi-modal Latent Diffusion.* arXiv 2306.04445; Entropy 26(4):320, 2024 (metadata verified via Crossref) | FULL_TEXT_SECTIONS_READ (§F) | Frozen deterministic per-modality autoencoders plus a masked multi-time latent score model. Coherence is judged by classifiers. | NO. Only single-step subset→missing generation is evaluated. | Closest to the P5 model template (frozen enc/dec + small latent generator). |
| Li et al. *OmniFlow.* CVPR 2025, arXiv 2412.01169 | SUMMARY_FULL_TEXT (subagent) | Multi-modal rectified flow with one time per modality. | NO | — |
| Rojas et al. *Diffuse Everything.* ICML 2025, arXiv 2506.07903 | ABSTRACT_ONLY (subagent) | A noise schedule per modality. | NO | — |
| Mizrahi et al. *4M.* NeurIPS 2023, arXiv 2312.06647 | SUMMARY_FULL_TEXT (subagent) | Chained generation: finished modalities are fed back in as conditions for "self-consistency". | NO, qualitative only. | Chaining is used, not tested distributionally. |
| Ye et al. *MODUS: Decoder-Only Any-to-Any Modeling of Diverse Modalities.* arXiv 2607.25948 (28 Jul 2026) | ABSTRACT_ONLY (verified); numbers from SUMMARY_FULL_TEXT (subagent) | Supports chained generation through intermediate modalities. The subagent reports NYUv2 surface-normal error for chained vs independent generation. | NO. Task quality only, not distributional equality. | Evidence that chaining changes outputs. Not a compatibility test. |
| Chung et al. *Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?* ACL 2025, arXiv 2505.24211 | ABSTRACT_ONLY (verified) | Three criteria: cyclic consistency, forward equivariance and conjugated equivariance. Per-sample, VQA-based. | NO. Round-trip and equivariance, per sample. | Not distributional, not direct-vs-intermediate. |
| CoDi (arXiv 2305.11846), NExT-GPT (arXiv 2309.05519), Versatile Diffusion (arXiv 2211.08332) | ABSTRACT_ONLY (subagent) | Any-to-any systems. | No evidence of such a test. | — |

## C. Classical conditional compatibility / pseudo-Gibbs

| Work | Access | Known result |
|---|---|---|
| Arnold & Press. *Compatible conditional distributions.* JASA 84(405):152–156, 1989 | SNIPPET_ONLY | Necessary and sufficient conditions for a joint to exist with the given conditionals, and KL-based measures of incompatibility. |
| Arnold, Castillo, Sarabia. *Conditional Specification of Statistical Models.* Springer 1999 | SNIPPET_ONLY | Book-length treatment, including Gaussian conditional specification. |
| Hobert & Casella. JCGS 7(1):42–60, 1998 | SNIPPET_ONLY | Compatible conditionals can imply an improper joint, and then Gibbs sampling has no proper stationary law. |
| Heckerman et al. *Dependency networks.* JMLR 1, 2000 | SUMMARY_FULL_TEXT (subagent; two sections) | Learned local conditionals are generally inconsistent, and pseudo-Gibbs sampling is used as a heuristic. |
| Chen & Ip. J. Stat. Comput. Simul. 85:3266–3275, 2015 | SUMMARY_FULL_TEXT (subagent) | With incompatible conditionals, the stationary law of pseudo-Gibbs depends on the scan order. |
| Liu, Gelman, Hill, Su, Kropko. Biometrika 101(1):155–173, 2014 (arXiv 1012.2902) | ABSTRACT_ONLY (subagent) | Stationary distribution of iterative imputation (MICE) under compatible and incompatible models. |
| Wang & Kuo, JMVA 2010; Kuo & Wang, AISM 71:93–105, 2019 | SNIPPET_ONLY / ABSTRACT_ONLY (subagent) | Compatibility checks for discrete conditionals, and pseudo-Gibbs convergence. |
| Young et al. *Inconsistencies in Masked Language Models.* arXiv 2301.00068 | SUMMARY_FULL_TEXT (subagent) | MLM conditionals from different masks cannot all come from one joint. |

## D. Multimodal VAEs (coherence = classifier agreement)

These are MVAE (arXiv 1802.05335), MMVAE (arXiv 1911.03393), MoPoE (ICLR 2021,
arXiv 2105.02470) and Daunhawer et al. (ICLR 2022, arXiv 2110.04121). Access
was SNIPPET_ONLY or ABSTRACT_ONLY (subagent). In all of them, coherence means
that a classifier gives the same label to the generated and the conditioning
modality. This is a per-sample semantic check, not a distributional
direct-vs-seq test.

## E. Guidance confound

| Work | Access | Relevance |
|---|---|---|
| Chidambaram et al. *What does guidance do?* NeurIPS 2024, arXiv 2409.13074 | ABSTRACT_ONLY (subagent) | Guidance does not sample the tilted distribution. |
| Bradley & Nakkiran. *Classifier-Free Guidance is a Predictor-Corrector.* arXiv 2408.09000 (venue UNVERIFIED) | ABSTRACT_ONLY (subagent) | Under CFG, neither DDPM nor DDIM yields p(x\|c)^γ p(x)^{1−γ}. |

So a direct-vs-seq gap measured with guidance on cannot be attributed to the
model. P5-E7 measures this confound with exact scores.

## Gap assessment (cautious; the search was not exhaustive)

**Already known:**

- compatibility as a concept
- incompatibility of learned conditionals
- the direct-vs-marginalised diagnostic itself (tabular: 2608.06004)
- consistency / commutativity regularisers as an idea
- chaining used in any-to-any models
- CFG distortion

**Not found in this check:**

- A distributional test of q_d(z|x) against ∫q(z|x,y)q(y|x)dy for continuous
  any-to-any multimodal diffusion or flow generators (UniDiffuser, MLD,
  OmniFlow, 4M-style) that controls the solver, guidance and Monte Carlo
  error, and uses the conditioning example as the unit.
- A method that reduces such a gap, with proper score, fidelity and diversity
  reported alongside it, against shared-conditional-network and extra-compute baselines (renamed from "shared-joint"; erratum E5 in RESEARCH_PACKET §10).

Any novelty claim for P5 must be restricted to these two points. It stays
provisional until the closest papers (2608.06004, UniDiffuser, MLD, GMM) are
read in full.

## F. Full-text checks (second pass, 2026-09-26)

**Method.** The arXiv HTML full text was fetched with curl and converted to
plain text in the session scratchpad; nothing was saved in this repo. The
sections listed below were read directly, not through a summariser. Sections
not listed were not read. Bibliographic metadata for every cited work was
checked separately; `paper/references.bib` records the verification URL of
each entry in its `x-verified` field.

### Klötergens et al., arXiv 2608.06004

Read: §3.1–3.3, §4.1, the regression paragraph of §4.2, §4.3, the related
work on Kolmogorov consistency and on compatibility, the conclusion, and the
first paragraph of Appendices A and C.

- **Definition 1 (C1, marginalization consistency).**
  p̂(a|x; D^A_{−B}) = ∫ p̂(a|b,x; D^A) p̂(b|x; D^B_{−A}) db. This is our Δ_T = 0
  with a = target and b = intermediate.
- **Definition 2 (C2, factorization consistency).** The two chain-rule orders
  must agree.
- **Proposition 1.** C2 ⇒ C1.
- **Proposition 2.** C1 ⇏ C2, via a binary counterexample in Appendix A.
  **Our "Proposition 2" is the same kind of statement** (a target-level
  identity does not fix the joint). Our Gaussian corr-scale construction is
  only a continuous instance of it, so it is **not a contribution**.
- **Measurement.** Total variation computed from the models' explicit
  predictive heads:
  - classification: exact finite sums;
  - regression: deterministic quadrature with K = 1000 equal-mass atoms and
    a 20-cell grid, a TV lower bound; the factorization check uses a
    128×128 grid.
  - Appendix C states that "there is no sampling anywhere in the pipeline".
- **Setup.** TabPFNv2/v3, TabICLv1/v2, TabDPT and TabFM on OpenML datasets,
  with 5-fold cross-testing. Every model violates C1 and C2.
- **Difference to P5.**
  - Their models expose densities or quantiles, so no sampling error arises.
  - P5 targets generators that expose samples only, such as diffusion or
    flow models. There, Monte Carlo, solver and guidance error must be
    separated from incompatibility, and a sample-based test is needed.
  - The consistency definition itself is theirs (and older).
- **Also cited there, not checked here:**
  - Yalavarthi et al. (2026), marginalization consistency in irregular
    time-series forecasting;
  - Young (2026), a conditioning-consistency gap for conditional neural
    processes.

  Both are NOT_CHECKED and are possible further close prior work.

### UniDiffuser, arXiv 2303.06555

Read: §3.1 (Eq. 5), §3.2, §6.1, §6.3.

- **Objective.** A joint noise-prediction network trained with independent
  per-modality timesteps. t^y = T gives the marginal, t^y = 0 the
  conditional, and t^x = t^y the joint.
- **CFG.** CFG "for free" uses the model at t^y = T (the marginal) as the
  unconditional model. This is the convention used in P5-E7.
- **Evaluation.** FID and CLIP score. DPM-Solver with 50 steps.
- **§6.3.** Data variation (image→text→image) and blocked Gibbs sampling are
  shown only as qualitative samples.
- **Direct-vs-sequential test.** None.

### Multi-modal Latent Diffusion, arXiv 2306.04445

Read: §4.1 (multi-time diffusion; training and conditional generation) and
the §5 evaluation-metrics paragraph.

- **Model.** Independently trained deterministic unimodal autoencoders and a
  masked score network with a multi-time vector τ.
- **Training.** The conditioning subset A₂ is drawn from ν with
  ν(∅) = d at each step.
- **Sampling.** Euler–Maruyama.
- **Evaluation.**
  - Coherence: pre-trained classifiers, following Shi et al., Sutter et al.
    and Palumbo et al.
  - Quality: FID / FAD.
  - Averages are over 5 seeds.
- **Direct-vs-sequential test.** None.

### Generative Marginalization Models, arXiv 2310.12920

Read: §3 (Eqs. 4, 5, 7 and the ConsistencyError) and the conclusion.

- **Loss.** Marginalization self-consistency is enforced as the squared
  log-space error of single-step constraints
  p_θ(x_{σ(<d)}) p_φ(x_{σ(d)} | x_{σ(<d)}) ≈ p_θ(x_{σ(≤d)}).
- **Scope.** The method is for "high-dimensional discrete data", and it
  needs explicit likelihoods.
- **Relation to P5.** This is the closest prior *training-time* consistency
  loss. It does not transfer directly to sample-only continuous generators.

## G. Revised novelty accounting (supersedes the gap assessment above where they differ)

**Not ours:**

- the consistency definitions (C1/C2, compatibility);
- the fact that marginal-level consistency does not imply joint-level
  consistency;
- the propriety argument against observed intermediates;
- CFG distortion;
- permutation-test validity;
- energy distance and its bias;
- ED-based two-sample tests;
- consistency losses as an idea.

**Candidate contributions, each still to be defended:**

1. A sampling design for generators that expose samples only. It uses
   within-example exchangeable relabelling with a fresh intermediate per
   sample, with the conditioning example as the unit.
2. An error decomposition that separates Monte Carlo, discretisation, prior
   mismatch and CFG components from incompatibility. It is validated in
   closed form on a Gaussian probe.
3. \todo{P5-REAL-01}: measurements on learned multimodal generators — NOT_RUN.

**Round 6 note (2026-10-02).** A synthetic, single-seed neural pilot ran on
the three-variable mixture probe (RESEARCH_PACKET §13): separately trained
conditional flow learners were detectably incompatible, a shared conditional
network was not flagged at the target level but was at the projected joint,
and was less accurate. This is a measurement on a probe, not on a learned
multimodal generator, so item 3 (P5-REAL-01) stays NOT_RUN and the novelty
accounting above is unchanged (UNVERIFIED). The direction "target-level
non-detection with a joint-level rejection" is consistent with Klötergens et
al. Prop. 2 and is not claimed as new.
