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
