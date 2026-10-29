# Architecture decision records

Short records: problem, options, choice, reason. Numbered in the order they were made.
"ASSUMPTION" marks a value or modelling choice that is not backed by a reference in
[`REFERENCES.md`](REFERENCES.md).

## ADR-001 — Language, stack and hardware target
- **Problem:** must run on a laptop CPU, reproducibly, in < 20 min (full) / < 3 min (fast).
- **Options:** BoTorch/GPyTorch stack; scikit-learn + PyTorch + CMA-ES; pure NumPy.
- **Choice:** NumPy/SciPy/pandas, scikit-learn (GPs, gradient boosting, ridge), PyTorch CPU
  (deep ensembles, MC dropout), `cma` (CMA-ES). Python ≥ 3.11 (developed on 3.11).
- **Reason:** BoTorch's strength is gradient-based acquisition optimisation for GP models.
  Our robust objectives are Monte-Carlo estimates over a composition distribution and must
  work for *any* model (ensembles, GBMs), so a model-agnostic derivative-free optimiser
  (CMA-ES on a common-random-numbers MC objective, as in sample-average approximation) is
  simpler, transparent and fast enough at 8 decision variables. scikit-learn's GP is
  sufficient for a few hundred training points. Fewer heavy dependencies also make CI faster.

## ADR-002 — Data validation with pandas checks, configs with pydantic
- **Problem:** "enforce the schema with pydantic or pandera" and produce human-readable errors.
- **Choice:** table validation is a set of vectorised pandas checks
  (`robustprint.data.validation`) that report every offending row with its CSV row number,
  experiment id, value and the allowed range. Experiment configs are pydantic models.
- **Reason:** row-wise pydantic models give generic messages and are slow for large files;
  pandera adds a dependency for little gain. Our checks are covered by unit tests.

## ADR-003 — One row per print job, replicates aggregated
- **Problem:** real labs print several specimens per setting.
- **Choice:** one CSV row = one print job (one batch, one setting) with `n_replicates`
  specimens; strength is the replicate mean, optionally with `tensile_strength_sd_MPa`.
  The regression likelihood uses noise variance σ²/n.
- **Reason:** matches how results are usually tabulated; per-specimen data can still be
  entered with `n_replicates = 1`.

## ADR-004 — Every synthetic row is labelled
- **Choice:** mandatory `data_origin` column (`synthetic` | `experimental`) and a
  `source_description` beginning with "SYNTHETIC" for generated data. The validator warns
  when a file mixes both.
- **Reason:** non-negotiable honesty rule.

## ADR-005 — Compositions enter models through the ilr transform
- **Problem:** fractions are constrained to the simplex; Euclidean kernels on raw fractions
  treat 0.01→0.02 like 0.50→0.51.
- **Options:** raw fractions; additive log-ratio (alr); centred log-ratio (clr); ilr.
- **Choice:** ilr with a Helmert basis (Egozcue et al. 2003), with multiplicative zero
  replacement at 1e-4.
- **Reason:** ilr is an isometry onto R^(D−1) with a non-singular covariance (unlike clr) and
  is symmetric in the parts up to rotation (unlike alr). With ARD kernels, a rotation of the
  coordinates matters, which is a known limitation; we keep the Helmert ordering fixed.

## ADR-006 — Two kinds of composition uncertainty
- **Problem:** "composition known only with uncertainty" can mean (a) the analysis of a
  characterised batch is noisy, or (b) the *next* batch has not been characterised and is
  a draw from the feedstock distribution.
- **Choice:** both are represented. (a) is the per-batch `composition_concentration` (or
  `sd_<component>` columns); the simulator reports compositions with Dirichlet(300·c_true)
  error. (b) is a `Feedstock` (Dirichlet mean + concentration), which can be estimated from
  the batches seen so far by maximum likelihood (`estimate_feedstock`). The optimiser
  integrates over (b).
- **Reason:** (b) is the decision-relevant uncertainty: settings are chosen before the next
  batch is known.

## ADR-007 — Material property values
- **Problem:** the simulator needs bulk properties; we could not verify one open,
  peer-reviewed table covering all components.
- **Choice:** moduli and strengths of PE, PET, PA and the aramid modulus fall inside
  Engineering ToolBox ranges ([ETB-MODULUS], [ETB-POLYMER]). The following are ASSUMPTIONS:
  PET and fibre density; printed-specimen elongations; all processing temperatures
  (`t_min_C`, `t_crit_C`, `t_deg_C`), chosen as melting point plus a margin typical of FDM
  practice for PE/PA/PET and a low degradation onset (235 °C) for foam/cellulose residues;
  the effective fibre strength (400 MPa, far below aramid filament strength, representing
  short, poorly bonded fibres); contaminant modulus/strength.
- **Reason:** the simulator only needs the right *scale and structure*; this is stated as a
  limitation. The numbers are in `src/robustprint/physics/materials.py` with source tags.

## ADR-008 — Blend immiscibility penalty
- **Choice:** σ_blend = (Σ w_i σ_i)(1 − κ I(w)), I(w) = Σ_{i≠j} χ_ij w_i w_j with
  χ(PE,PET) = 1.0, χ(PE,PA) = 0.9, χ(PET,PA) = 0.5 and κ = 0.5 (ASSUMPTIONS). Modulus uses a
  Voigt/Reuss (Hill) average with a weaker penalty (κ_E = 0.25); elongation is penalised most.
- **Reason:** immiscible blends from mixed recycling show poor phase adhesion and reduced
  properties ([PPPE-BLENDS]); strength and ductility suffer more than stiffness. The
  quadratic interaction form is the simplest symmetric choice that is zero for pure polymers.

## ADR-009 — Fibre and inclusion models
- **Choice:** modulus by Halpin–Tsai with ξ = 4 (ASSUMPTION, random short fibres); strength
  by a modified rule of mixtures with efficiency 0.2 and a smooth agglomeration penalty
  exp(−(V_f/0.2)²) (ASSUMPTION); non-adhering inclusions (contaminants and unmelted polymer)
  reduce strength by 1 − 1.21 φ^(2/3) (form attributed to Nicolais & Narkis; coefficient
  treated as ASSUMPTION) and modulus by (1 − φ)² (ASSUMPTION).
- **Reason:** standard, citable functional forms where available; fibres are roughly
  strength-neutral but stiffening at low loading, which is the usual experience with short,
  poorly bonded natural/aramid fibres in recycled matrices.

## ADR-010 — Process model
- **Choice:** interlayer bond degree β from the time above a critical sintering temperature
  with the t^(1/4) healing law ([SUN-2008], [WOOL-1981]); an Arrhenius reptation time
  (60 kJ/mol, 0.4 s at 230 °C — ASSUMPTIONS); a hot-end melt-rate limit of 9 mm³/s at 220 °C
  growing e-fold per 60 °C and reduced by fibres and unmelted particles (ASSUMPTION);
  porosity from layer height, under-extrusion, moisture and degassing (ASSUMPTION); raster
  angle via Tsai–Hill with transverse/shear strengths proportional to β ([AHN-2002] for the
  qualitative trend); load-bearing section 0.35 + 0.65·infill (ASSUMPTION).
- **Reason:** each mechanism is a documented FDM trend; coefficients are set to give
  plausible magnitudes, not fitted to data.

## ADR-011 — Failure model
- **Choice:** P(success) = (1 − P(clog))(1 − P(warp)) with logistic models. Clogging rises
  with fibre and contaminant fractions, unmelted polymer (low temperature), commanded flow
  above the melt-rate limit, degradation, and a fibre × volumetric-flow interaction.
  Warping rises with PE fraction and fan speed and falls with bed temperature.
- **Reason:** these are the two failure modes most often reported for recycled and
  fibre-filled filament; the mission asked for clogging risk rising with fibre/contaminant
  fraction and low temperature. All coefficients are ASSUMPTIONS.

## ADR-012 — Simulator revision after a first optimum check (researcher degree of freedom)
- **Problem:** In simulator v0, the optimal settings were nearly the same for every batch:
  composition scaled strength but barely changed *which* settings were best. The nominal,
  expected and 5 %-quantile optima of the true function coincided, so robust optimisation
  could not matter by construction.
- **Choice:** before running any model or optimiser experiment, we (1) added the
  fibre × volumetric-flow clogging term (fibre bridging in the nozzle becomes more likely at
  high flow; ASSUMPTION), (2) made the melt-rate limit less temperature-sensitive (e-fold per
  60 °C instead of 35 °C, giving ~15 mm³/s at 270 °C, a more typical desktop value), and (3)
  reduced the unmelted-polymer clogging coefficient (20 → 12) and PE warping (4 → 3), with
  the nozzle window narrowed to 210–290 °C so a space-filling campaign has ~43 % print
  success instead of ~26 %. The simulator was then frozen (v1.0).
- **Reason / consequence:** this is a modelling choice made *knowing* that it creates
  composition × process interactions; any benefit of robust optimisation shown on
  synthetic data depends on such interactions existing in real feedstock. We state this in
  LIMITATIONS.md and the README. Real data must decide whether they exist.

## ADR-013 — Model log strength
- **Choice:** every strength model predicts the log of the job-mean tensile strength with a
  Gaussian predictive; metrics like NLL, CRPS and calibration are on the log scale, RMSE/MAE
  also in MPa.
- **Reason:** strength is positive and its scatter is roughly proportional to its level
  (coefficient-of-variation noise), so a Gaussian on log strength is homoscedastic to first
  order and quantiles transform exactly back to MPa.

## ADR-014 — GP noise floor
- **Problem:** in the first benchmark the ARD GP learned a noise variance of ~2e-5 (log
  scale), interpolated the data and under-covered (74 % at nominal 90 %).
- **Choice:** lower bound on the white-noise variance equal to a 3 % coefficient of
  variation (ASSUMPTION: tensile tests rarely repeat better than that).
- **Reason:** physically motivated regularisation; the remaining under-coverage is reported,
  not tuned away. Conformal wrappers are the recommended fix for coverage.

## ADR-015 — Early stopping for NLL-trained networks
- **Problem:** deep ensembles and MC dropout trained on the Gaussian NLL were badly
  overconfident (NLL 4–7, 90 % coverage ~0.45–0.57) on ~70 training points.
- **Options:** stronger weight decay; fewer epochs; β-NLL; early stopping.
- **Choice:** each member holds out a random 20 % of its training rows and keeps the state
  with the best validation NLL (checked every 10 epochs).
- **Reason:** weight decay and shorter training did not help (tested, see PROGRESS.md);
  early stopping brought coverage to ~0.90. It also gives each member a different subset
  (implicit bagging). Architecture and learning rate were not tuned further.

## ADR-016 — Constraints, conditioning on success and the safe window
- **Choice:** all objectives share a success constraint E[π] ≥ 0.9 (nominal: π at the mean
  composition), implemented as a penalty of 100 MPa per unit probability shortfall. Strength
  samples are weighted by the predicted success probability (strength is only defined for
  parts that print). The nozzle window is limited to [T_min, T_deg − 10 °C] of the dominant
  polymer (ASSUMPTION: 10 °C margin).
- **Reason:** a 5 % quantile of "utility with failures = 0" is identically 0 whenever more
  than 5 % of parts fail, which makes the objective flat and uninformative. Separating
  strength-given-success from the failure constraint keeps both visible; the realised
  expected utility (failure = 0 MPa) is still reported.

## ADR-017 — Experiment cost model for cost-aware active learning
- **Choice:** cost = 1 (setup: blending, loading, testing) + relative print time.
- **Reason:** simplest model that makes slow prints (low speed, thin layers, full infill)
  more expensive. ASSUMPTION; a lab should replace it with its own costs.

## ADR-018 — Pareto front by ε-constraint
- **Choice:** maximise CVaR subject to print time ≤ t for a grid of t (penalty method);
  energy is reported per solution rather than optimised.
- **Reason:** print time and energy are deterministic, known functions of the settings, so
  a sweep of a 1-D constraint gives an interpretable front with few optimiser runs;
  scalarised 3-objective search would add little for a researcher choosing a setting.

## ADR-019 — Main model for the optimiser
- **Choice:** `hybrid_gp` (physics-residual GP) + `gbm_isotonic` success classifier,
  fixed in the pre-registered protocol; `deep_ensemble` and plain `gp` are ablations.
- **Reason:** in the pre-protocol benchmark it had among the best NLL/CRPS, it separates
  epistemic from aleatoric variance, and its closed-form posterior covariance enables exact
  batch active learning. The choice was made before the final runs.

## ADR-020 — Browser Virtual Lab as a TypeScript re-implementation
- **Problem:** the study should be explorable without installing Python, and a lab should be
  able to try its own CSV in the browser without uploading it anywhere.
- **Options:** Pyodide (Python + scikit-learn in WebAssembly, ~30 MB download, slow start);
  a server (hosting cost, data leaves the browser); a TypeScript port of the parts needed.
- **Choice:** port the simulator, compositional transforms, the physics-residual GP, the
  `gbm_isotonic` success classifier, the robust objectives, CMA-ES and the batch
  active-learning strategies to TypeScript (`web/src/engine`), run them in a Web Worker, and
  ship a single self-contained HTML file. Deep ensembles, MC dropout and conformal wrappers
  are not ported; the Study step shows their pre-computed results instead.
- **Verification:** golden files written by `scripts/export_web_golden.py` from the Python
  code. The simulator agrees to 1e-9 relative on 400 random rows, the ilr transform to
  1e-10 and Dirichlet ilr moments to 1e-8, GP predictions with fixed hyperparameters to
  1e-7, isotonic regression exactly and raw boosted scores to ~0.04 in log-odds on average (float32 histograms and tie-breaking in
  scikit-learn). Hyperparameter fitting (Adam vs L-BFGS) and the classifier's CV folds differ,
  so fitted models agree statistically, not bit for bit: the calibrated classifier's error
  against the true P(success) is 1 % lower than scikit-learn's on the golden data set.
- **Consequence:** numbers produced in the browser are a second implementation of the same
  method, not the study. Only the Study step reports the pre-registered results, and it
  reads them from `results/full`.
- **Found while porting:** an earlier browser version used the logistic classifier for speed;
  it invented a raster-angle effect on print failure (none exists in the simulator), which
  pushed learned-model recommendations to 90°. Search budgets below the study's (300–900
  evaluations) also left CMA-ES unconverged. Both were fixed before release (boosted trees;
  800 / 1 500 / 3 000 evaluations).
