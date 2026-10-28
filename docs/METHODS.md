# Methods

This document describes the simulator, the models, the uncertainty propagation, the robust
optimiser, active learning and the **evaluation protocol**. Section 8 (protocol) was
written and committed *before* the final experiments were run (see the git history:
commit "docs: pre-register evaluation protocol"). A pilot run with 8 seeds of the
optimisation experiment was performed beforehand to check that the pipeline works and how
long it takes; it is disclosed in §8.6.

Everything numerical in this repository is **synthetic** unless explicitly stated
otherwise. The only external data is the optional Kaggle dataset (§7).

## 1. Problem statement

A feedstock *batch* has composition c ∈ Δ^(D−1) (mass fractions of D = 5 components: PE,
PET, PA, fibre, contaminant) and moisture m. Print settings x ∈ X ⊂ R^8 (nozzle and bed
temperature, speed, layer height, infill, raster angle, flow, fan). A print job succeeds
with probability π(c, m, x); a successful part has tensile strength Y(c, m, x). The next
batch is not characterised before printing: c ~ F, the *feedstock distribution*, modelled
as Dirichlet(s·μ). We want settings x that are good for the whole distribution F, not
just for its mean μ, and we want to know how confident we can be.

## 2. Ground-truth simulator (synthetic world)

Implemented in `src/robustprint/physics/`. All coefficients are listed in
`SimulatorConfig` and `ProcessConstants`; their provenance is in DECISIONS.md
(ADR-007 – ADR-012). The structure, step by step:

1. **Melting.** Polymer i is unmelted in proportion σ((T_min,i − T)/6 °C), σ = logistic.
   Unmelted polymer behaves as an inclusion.
2. **Volume fractions** from mass fractions and densities: fibre V_f, inclusions V_p
   (contaminants + unmelted polymer), molten-matrix weights w.
3. **Blend strength** σ_m = (Σ w_i σ_i)(1 − κ I(w)), I(w) = Σ_{i≠j} χ_ij w_i w_j, κ = 0.5.
   Blend modulus: Hill average (mean of Voigt and Reuss) × (1 − 0.25 I).
4. **Fibres.** Strength: σ = (η σ_f V_f + σ_m (1 − V_f)) exp(−(V_f/0.2)²), η = 0.2.
   Modulus: Halpin–Tsai, E = E_m (1 + ξηV_f)/(1 − ηV_f), η = (E_f/E_m − 1)/(E_f/E_m + ξ), ξ = 4.
5. **Inclusions.** Strength × (1 − 1.21 V_p^(2/3)); modulus × (1 − V_p)².
6. **Chemistry.** Degradation d = Σ_i c_i softplus((T − T_deg,i)/10 °C); hydrolysis
   h = m · (1.5 c_PET + 1.5 c_PA) σ((T − 240)/10). Strength × exp(−d − h).
7. **Extrusion.** Commanded flow Q = v·h·w·flow; melt-rate limit
   Q_max = 9 exp((T − 220)/60)/(1 + 6V_f + 3V_unmelted) mm³/s; delivered fraction
   = flow × softmin(1, Q_max/Q).
8. **Porosity** p = 0.03 + 0.08 (h/0.4)² + 0.9·softplus-underextrusion + 0.03 m + 0.05 d.
9. **Bonding.** Substrate temperature from the envelope (bed-warmed) and the layer time;
   interface time above T_crit, t_above = τ ln((T − T_s)/(T_crit − T_s)) with
   τ = 1.5 s (h/0.2)/(1 + 2 fan); Arrhenius reptation time t_rep; β = softmin(1, (t_above/t_rep)^(1/4)).
10. **Part strength** Y = σ · e^(−d−h) · (0.55 + 0.4β) · (1 − p)^1.5 · TsaiHill(θ; 0.75β, 0.5β)
    · (0.35 + 0.65·infill) · e^(−4·over) · e^b, with batch effect b ~ N(0, 0.05²).
11. **Failure.** π = (1 − σ(z_clog))(1 − σ(z_warp)); z_clog increases with fibre, contaminant,
    unmelted fraction, flow above the melt limit, degradation and fibre × flow; z_warp
    increases with PE fraction and PE × fan and decreases with bed temperature.
12. **Measurement.** Specimen strength = Y · exp(ε), ε ~ N(−cv²/2, cv²) with
    cv = 0.04 + 0.3 c_contaminant + 0.2 c_fibre + 0.3 p (heteroscedastic). Reported
    composition = Dirichlet(300 · c_true) (analysis error).

Feedstocks: `iss_nominal` μ = (0.45, 0.25, 0.15, 0.08, 0.07), s = 35;
`iss_shifted` (out-of-distribution) μ = (0.30, 0.30, 0.12, 0.16, 0.12), s = 35. Moisture
is log-normal with median 0.3 % and log-sd 0.6.

## 3. Data

A campaign has B batches × 4 settings per batch × 3 replicates. Settings follow one
scrambled Sobol design over the search box. Each row is one job (ADR-003). Failed jobs
have no strength. Synthetic rows are labelled (ADR-004).

## 4. Models

Inputs: ilr(c) (4), settings scaled to [0, 1] (8), log moisture (1). Target: log strength
of the job mean (ADR-013).

| Name | Description |
|---|---|
| `physics_rom` | log y = a + b log Σ c_i σ_i (composition only), OLS with parameter uncertainty |
| `ridge_interactions` | Bayesian ridge on standardised features with all pairwise products |
| `gbm_quantile` | Histogram gradient boosting: mean model + 5 % / 95 % quantile models |
| `gp` | GP, constant × ARD Matérn-5/2 + white noise (noise floor 3 % CV) |
| `het_gp` | Most-likely heteroscedastic GP (Kersting et al. 2007), 2 iterations |
| `hybrid_gp` | GP on the residual of a simplified physics prior (ROM × infill section × Tsai–Hill) |
| `deep_ensemble` | 5 MLPs (2×64, SiLU) with mean/log-variance heads, Gaussian NLL, early stopping |
| `mc_dropout` | One such MLP with dropout 0.1, 50 stochastic passes |

Predictive variance is split into epistemic (GP latent variance; ensemble disagreement;
dropout variance) and aleatoric (noise). GBM reports total variance only.

**Conformal prediction.** Split conformal with normalised scores |y − μ|/σ around `gp`,
`hybrid_gp` and `deep_ensemble`, and CQR around GBM quantiles, at α = 0.1, calibrating on
25 % of training *batches* (Lei et al. 2018; Romano et al. 2019).

**Success classifier.** `gbm_isotonic` (gradient boosting + isotonic calibration, 3-fold)
is the default; `logistic` (L2, pairwise interactions, C = 0.03) is the comparison.

**Single-part correction.** Training targets are means of n̄ replicates; a deployed part
is a single specimen. Aleatoric variance is increased by s²_spec(1 − 1/n̄), where s²_spec is
the pooled within-job variance of log strength estimated from the replicate standard
deviations.

## 5. Uncertainty propagation

For settings x and feedstock F, with K draws (c_k, m_k) ~ F:

E[log Y] ≈ (1/K) Σ μ_k, and Var[log Y] = Var_k[μ_k] (input) + E_k[s²_epi,k] (epistemic) +
E_k[s²_ale,k] (aleatoric).

The unscented transform uses the exact ilr mean and covariance of the Dirichlet
(E[log c_i] = ψ(α_i) − ψ(α_0), Cov[log c_i, log c_j] = ψ′(α_i)δ_ij − ψ′(α_0), projected on the
Helmert basis), plus log moisture, with 2n + 1 = 11 sigma points (κ = 0).

## 6. Robust optimisation

Common random numbers: K composition draws and S = 16 standard normals per draw are fixed.
For candidate x, strength samples exp(μ_k + s_k z_ks) are weighted by the predicted success
probability π_k (conditioning on a part that prints). Objectives (all maximised):

| Objective | Definition | Constraint(s) |
|---|---|---|
| nominal | predicted mean strength at μ (mean composition, median moisture) | π(μ, x) ≥ 0.9 |
| expected | mixture mean strength | E_k[π_k] ≥ 0.9 |
| quantile | weighted 5 % quantile of the strength mixture | E_k[π_k] ≥ 0.9 |
| cvar | weighted CVaR at α = 0.1 (mean of the worst 10 %; Rockafellar & Uryasev 2000) | E_k[π_k] ≥ 0.9 |
| chance | mixture mean strength | E_k[π_k] ≥ 0.9 and P(Y ≥ 5 MPa) ≥ 0.9 |

Constraints are penalties of 100 MPa per unit of probability shortfall. The search box is
intersected with a safe nozzle window for the dominant polymer (≥ its full-melt temperature
and ≤ its degradation onset − 10 °C). Optimiser: 64 scrambled-Sobol points, then CMA-ES
(population 16) from the best one, total budget 800 evaluations (ADR-001).

**Multi-objective.** ε-constraint: maximise CVaR subject to print-time proxy ≤ t for a grid
of budgets t; energy proxy is reported for each solution.

## 7. Public data (optional)

Kaggle "3D Printer Dataset for Mechanical Engineers" (50 rows, PLA/ABS, licence unknown, not
redistributed). Used for (a) a trend check: Spearman correlation of tensile strength with
each setting in the data, compared with the direction of the simulator's effect over the
same range, and (b) a secondary benchmark: repeated 5-fold CV (10 repeats) of ridge, GBM,
GP and deep ensemble on log tensile strength. It cannot test composition effects.

## 8. Evaluation protocol (pre-registered)

### 8.1 Experiments

- **E1 — robust vs nominal (central claim).** 30 seeds. Per seed: a fresh training campaign
  of 60 batches × 4 settings × 3 replicates from `iss_nominal`; model `hybrid_gp` +
  `gbm_isotonic`; each of the 5 objectives recommends one setting; each setting is printed
  once on each of the same 2 000 fresh batches (paired, never seen in training) using the
  ground truth. Oracle settings (objectives optimised on the true simulator with 256
  composition draws, budget 1 500) are deployed on the same batches.
- **E1b — ablations and stress tests.** 15 seeds each, objectives nominal / expected / cvar:
  (a) model `deep_ensemble` (5 members); (b) model `gp` (no physics prior); (c) 30 and 120
  training batches; (d) no composition propagation (CVaR at the mean composition, noise
  only); (e) no epistemic variance in the objectives; (f) optimiser told the wrong spread
  (concentration × 0.25 and × 4); (g) feedstock distribution estimated from the training
  batches; (h) twice the true composition spread; (i) label noise × 2; (j) 5 % outliers
  (strength × U(0.3, 0.6)); (k) deployment on `iss_shifted` while the optimiser still
  believes `iss_nominal`; (l) ensemble sizes 3 and 10.
- **E2 — model comparison.** 3 seeds × training sizes {15, 30, 60, 120} batches; 5-fold
  grouped CV by batch on the training campaign, plus a new-batch test set and a shifted
  (OOD) test set of 30 batches each. All 8 models; conformal wrappers; both classifiers.
- **E3 — active learning.** 20 seeds; 20 initial jobs (5 batches × 4 Sobol settings), then 8
  rounds of 4 jobs chosen by: random, Sobol (one space-filling sequence), max-variance,
  targeted IVR, cost-aware IVR. After each round the CVaR recommendation is scored on the
  ground truth. Metric: regret of the true penalised CVaR score against the best known value.
- **E4 — propagation.** Monte Carlo with K ∈ {16, 64, 256, 1024} and the unscented transform,
  against Monte Carlo with K = 20 000, at 20 random settings and the recommended settings;
  error in predictive mean and standard deviation of log strength; wall time.
- **E5 — sensitivity.** Sobol first-order and total indices (Saltelli/Jansen estimators,
  2 048 base samples) of the ground-truth log strength and success probability with respect
  to the 8 settings, the composition (as one group), moisture and the batch effect; plus
  first-order indices of each component's pivot log-ratio coordinate estimated by binning.
- **E6 — Pareto.** ε-constraint fronts (CVaR vs print time, energy reported) for one fitted
  model; each point deployed on 2 000 fresh batches.

### 8.2 Metrics

Regression (on log strength): RMSE (also in MPa), MAE (MPa), Gaussian NLL, CRPS, quantile
calibration error (mean |observed − nominal| over levels 0.05…0.95), 90 % coverage and
width. Classification: Brier score, log loss, ECE (10 bins), accuracy. Deployment
(per setting, over fresh batches): failure rate, mean / 5 % quantile / CVaR₁₀ of strength
of successful parts, expected utility (failed part = 0 MPa), fraction of parts that print
and reach 5 MPa, regret of expected utility against the oracle-expected setting.

### 8.3 Hypotheses (E1)

- **H1:** composition-aware objectives (expected, quantile, cvar, chance) have a lower
  realised failure rate than nominal.
- **H2:** quantile and cvar have a higher realised 5 % quantile of strength than nominal.
- **H3 (two-sided, no directional prediction):** expected utility differs from nominal.

### 8.4 Statistical tests

Per E1 variant: paired two-sided Wilcoxon signed-rank tests of each method against nominal
for each of the 6 deployment metrics (24 tests), Holm-adjusted within the family; α = 0.05
on adjusted p-values. Effect sizes: mean paired difference with 95 % percentile-bootstrap CI
(5 000 resamples) and matched-pairs rank-biserial correlation. For E1b, E2, E3 we report
bootstrap CIs and treat differences descriptively (no significance claims) unless a test is
explicitly reported with its correction.

### 8.5 Seeds and reproducibility

All seeds derive from the config seed `20261028` through `numpy.random.SeedSequence`. Each
task is deterministic given its seed (single-threaded BLAS and PyTorch in every worker).
Raw per-seed results are saved as CSV/JSON under `results/<config>/`.

### 8.6 Disclosure of the pilot

Before this protocol was written, an 8-seed pilot of E1 (different base seed) was run to
check the pipeline and its runtime. In that pilot the composition-aware objectives had
lower failure rates than nominal, and no clear difference in expected utility. The
simulator was frozen before the pilot (ADR-012); after the pilot only runtime settings
(thread pinning) were changed.

### 8.7 Deviations from the protocol (recorded after the runs)

1. **E3 crash and fallback.** In the first full run, one E3 seed had only 4 successful
   jobs among its 20 initial jobs, too few to fit a model (minimum 5), and the stage
   crashed. We added a fallback that was not in the protocol: until 5 jobs have succeeded,
   the "recommendation" is the default settings and new jobs are chosen at random, for
   every strategy. Rows carry a `has_model` flag.
2. **Final results come from a clean re-run.** After fixing the crash, and after a speed-up
   that does not change any number (the optimiser no longer computes statistics an
   objective does not use), the full configuration was re-run from an empty folder at commit
   `27e1a57` (`results/full/manifest.json`). All E1, E1b, E2, E5, E6 and E7 files of that
   run are byte-identical to the first run's, E3–E4 match the separate re-run of those
   stages, and the only differences are wall-clock timings.
3. **Additional exploratory analyses (not pre-registered):** paired tests of the oracle
   settings against the oracle-nominal setting (`e1_optimisation/tests_oracle_exploratory.csv`,
   where only deployment batches vary between seeds); descriptive paired differences of
   each ablation against the main run on the same seeds (`e1b_ablations/vs_main_paired.csv`);
   and the ground-truth variance decomposition at the recommended settings (E4).
