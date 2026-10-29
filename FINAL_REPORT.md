# RobustPrint — final report (v1.1.0)

All results below are **synthetic** (RobustPrint simulator) unless marked "public data".
Every number comes from `results/full/`, produced by `robustprint reproduce --full`
(25 min 2 s on the 4-core CPU container used for v1.1.0; see `results/full/manifest.json`).
That container was about 1.6× slower than the one used for v1.0.0 (16 min 10 s): on it the
v1.0.0 and v1.1.0 code ran the fast configuration's E1–E3 in the same time (61 s vs 60 s),
so the fix did not slow the code down, but the 20-minute target is only met on a machine
like the first. A second clean run reproduced every result file byte-for-byte apart from
wall-clock timings.
`python scripts/readme_numbers.py` prints them all at full precision.

**v1.1.0 replaces every v1.0.0 number.** The v1.0.0 data generator drew print success
separately for each replicate specimen and averaged failed specimens into the job's
strength as 0 MPa, biasing about 46 % of successful job means downward. The bug was found
while porting the generator to the browser. It was fixed with a regression test, and all
experiments were re-run with the pre-registered protocol unchanged (METHODS §8.7). Several
conclusions reversed; §3 lists which.

## 1. What was built

- **A documented data layer:** CSV schema for real experiments (`data/schema.md`), a
  validator with row-level error messages, loaders, maximum-likelihood estimation of the
  feedstock distribution from observed batches, and a starter experimental design for a
  first lab campaign (`robustprint starter-design`).
- **A transparent ground-truth simulator** of ISS-like mixed waste (PE, PET, PA, fibre,
  contaminant). It includes blend immiscibility, Halpin–Tsai fibres, inclusion weakening,
  thermal interlayer healing (t^¼), melt-rate limits, porosity, degradation, hydrolysis,
  clogging, warping, batch effects and heteroscedastic noise. Each coefficient is either
  cited or flagged as an assumption (`docs/DECISIONS.md`).
- **Eight strength models** (physics rule of mixtures, Bayesian ridge, quantile GBM, ARD GP,
  heteroscedastic GP, physics-residual hybrid GP, deep ensemble, MC dropout), conformal
  and CQR wrappers, and two calibrated print-success classifiers.
- **Uncertainty propagation** over the feedstock distribution (Monte Carlo and unscented
  transform) with an input / epistemic / aleatoric variance decomposition, plus grouped
  Sobol sensitivity analysis.
- **A robust optimiser** with five objectives (nominal, expected, 5 % quantile, CVaR,
  chance-constrained) under a success constraint and a safe nozzle window, using
  CMA-ES on a common-random-numbers Monte-Carlo objective, plus ε-constraint Pareto
  fronts (strength vs print time, energy).
- **Active learning:** greedy batch max-variance, targeted integrated-variance reduction
  and a cost-aware variant, with a printable next-experiments table.
- **Evaluation:** a pre-registered protocol (`docs/METHODS.md` §8), 30-seed deployment
  experiment against a ground-truth oracle, 15 ablations and stress tests, model
  comparison and learning curves, active-learning simulation, and an optional public-data
  check.
- **The RobustPrint Virtual Lab** (`web/`, new in v1.1.0): a single-file browser app with
  a TypeScript re-implementation of the simulator, the hybrid GP, the boosted-tree success
  classifier, the objectives, CMA-ES and the acquisition strategies, tested against golden
  files from the Python code (ADR-020). Six steps: bench, campaign (or CSV import),
  model, optimise & deploy, next experiments, and the study results.
- **Interfaces and engineering:** a typer CLI (`validate`, `simulate`, `train`, `predict`,
  `optimize`, `recommend-experiments`, `starter-design`, `reproduce`, `figures`,
  `public-data`), a Streamlit dashboard and the Virtual Lab. 101 Python tests (unit,
  property-based with Hypothesis, integration, dashboard; 98 % line coverage of
  `src/` excluding plotting and the CLI module), 20 TypeScript parity and unit tests and a
  Playwright end-to-end test. ruff, mypy, strict TypeScript, GitHub Actions CI, a GitHub
  Pages workflow, pre-commit, Makefile and pinned lock files.

## 2. Headline results

| Claim | Number | File |
|---|---|---|
| H1: composition-aware objectives fail less (learned model) | 13.1–14.2 % vs 19.5 % for nominal; −5.3 to −6.4 points, all Holm p ≤ 0.014 | `e1_optimisation/summary.csv`, `tests.csv` |
| H2 contradicted: weakest good parts | 5 % quantile 4.18–4.26 vs 4.71 MPa for nominal (expected, quantile, CVaR; Holm p ≤ 0.004) | `e1_optimisation/tests.csv` |
| H3: expected utility | chance-constrained +0.67 MPa [0.37, 0.99] (Holm p = 0.007); others no difference | `e1_optimisation/tests.csv` |
| Oracle (true function): composition-aware vs nominal | failure 9.6 % vs 17.6 %; expected utility 8.52 vs 8.05 MPa (exploratory) | `e1_optimisation/tests_oracle_exploratory.csv` |
| Regret of learned-model recommendations | 1.14 (chance) to 2.06 (CVaR) MPa of expected utility | `e1_optimisation/summary.csv` |
| Where the uncertainty is | composition share of predictive variance: hybrid GP 65–69 %, ground truth 82–84 %, deep ensemble 29–32 % | `e4_propagation/decomposition.csv` |
| Propagation is what buys the failure reduction | without it, expected and CVaR fail 7.6 and 6.6 points more (descriptive, paired vs main run) | `e1b_ablations/vs_main_paired.csv` |
| Best probabilistic models (120 batches) | GP and hybrid GP: CRPS 0.092, NLL −0.39; coverage 0.85–0.86 (hybrid with conformal: 0.88) | `e2_models/metrics.csv`, `conformal.csv` |
| Physics prior | no gain at 120 batches (CRPS 0.0922 vs 0.0920), worse at 15 (0.336 vs 0.264) | `e2_models/metrics.csv` |
| Active learning | no strategy better than random after 52 jobs; cost-aware worse by 1.38 MPa [0.60, 2.19] | `e3_active_learning/per_round.csv` |
| Unscented transform | 6 % SD error on the hybrid GP, 26 % on the ground truth | `e4_propagation/errors.csv` |
| Sensitivity | nozzle temperature total index 0.58 (strength), 0.74 (success); composition 0.25 / 0.29 | `e5_sensitivity/global.csv` |
| Public data (Kaggle PLA/ABS) | CRPS 0.206–0.212 for all models; simulator directions agree for 3 of 6 settings | `e7_public_benchmark.csv`, `e7_public_trends.csv` |

## 3. What changed from v1.0.0, what worked and what did not

**Reversed or changed by the correction**

- H1 (fewer failures) is now supported; in v1.0.0 no difference survived correction.
- The chance-constrained objective now raises expected utility significantly.
- The v1.0.0 diagnosis that the model "mistook its lack of knowledge for noise" (model
  noise 0.36 vs 0.015 in log variance) was mostly an artefact of the bug: the corrected
  model attributes 65–69 % of its variance to composition.
- The physics prior no longer helps with little data (v1.0.0: CRPS 0.377 vs 0.488 at 15
  batches; now 0.336 vs 0.264, the other way round).
- Cost-aware active learning is now significantly worse than random (v1.0.0: no difference).

**Worked**

- The pipeline end to end: real-data schema → validated arrays → models → propagation →
  robust recommendation → next-experiment table, reproducible bit-for-bit given the seed.
- Composition-aware objectives cut failures on unseen batches by 27–33 % with the learned
  model and by 45 % with the perfect model, and the ablation without propagation lost that gain.
- The chance-constrained objective improved failures and expected utility together.
- Early stopping turned overconfident NLL-trained networks into well-covering ones (deep
  ensemble 90 % coverage 0.94). Conformal wrapping restored nominal coverage at 30–60
  training batches.
- Porting to the browser found the generator bug: an independent re-implementation is a
  strong test.

**Did not work**

- Quantile and CVaR objectives did not protect the weakest parts: lower-tail strength went
  down, not up.
- Model-based active learning: no improvement over random or Sobol within 52 jobs; regret
  did not fall; cost-aware selection was worse.
- The physics prior; the heteroscedastic GP (worse NLL than the plain GP); the unscented
  transform on the cliff-like ground truth; deep ensembles as optimiser models (worse
  recommendations despite the best coverage); conformal intervals at 120 batches and under
  shift; everything under out-of-distribution feedstock (54–66 % failures).
- The v1.0.0 test suite: it checked shapes, ranges and determinism but not that the
  generator was unbiased.

## 4. Known limitations

The simulator is not validated against any real recycled feedstock, and its coefficients
are partly assumptions. The benefit of robust optimisation in it depends on an interaction
we added (ADR-012). Models are weak below about 100 successful prints in 13 dimensions.
Conformal guarantees fail under feedstock shift. The browser lab is a second
implementation, not the study. Nothing here is validated for flight or safety-critical use.
Full list: `docs/LIMITATIONS.md`.

## 5. The three most valuable next steps with real lab data

1. **Run the starter design** (`data/examples/starter_design.csv`: 20 jobs, 60 specimens,
   3 batches with repeated settings) and measure composition with sub-sample repeats. The
   repeats directly estimate batch-to-batch variance, the quantity the whole method
   depends on. Then check whether settings × composition interactions exist at all:
   does the best nozzle temperature or speed differ between batches?
2. **Test the failure result first.** The clearest synthetic effect is on failed prints,
   not on strength. Print the nominal and the chance-constrained recommendation on the same
   new batches and count failures; it needs far fewer specimens than a strength comparison.
3. **Validate predictions prospectively.** Compare measured strengths and failures on new
   batches with the predicted 90 % intervals and success probabilities. That is the real
   version of experiment E1, and the only test that counts.

## 6. Talking points (each grounded in a result here)

- "I pre-registered my evaluation. Planning for the feedstock distribution cut failed
  prints on unseen batches by roughly 30 %, but the tail-risk objectives did not make the
  weakest parts stronger; that hypothesis was contradicted." (E1.)
- "An ablation shows where the gain comes from: switch off composition propagation and it
  disappears." (E1b.)
- "I found a bug in my own data generator by re-implementing it for the browser. I fixed
  it, added a regression test, re-ran everything with the same protocol and reported which
  conclusions reversed." (CHANGELOG, METHODS §8.7.)
- "Good uncertainty calibration in cross-validation did not guarantee good decisions: the
  best-covering model gave the worst recommendations." (E1b deep ensemble.)
- "Every synthetic result is traceable to a file and a command, and the whole method also
  runs in a browser tab for anyone to try." (`results/full/`, `web/`.)
