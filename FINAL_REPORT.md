# RobustPrint — final report (v1.0.0)

All results below are **synthetic** (RobustPrint simulator) unless marked "public data".
Every number comes from `results/full/`, produced by `robustprint reproduce --full`
(16 min 10 s on a 4-core CPU; commit `27e1a57`, see `results/full/manifest.json`).
`python scripts/readme_numbers.py` prints them all at full precision.

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
- **Interfaces and engineering:** a typer CLI (`validate`, `simulate`, `train`, `predict`,
  `optimize`, `recommend-experiments`, `starter-design`, `reproduce`, `figures`,
  `public-data`) and a Streamlit dashboard. 100 tests (unit, property-based with Hypothesis,
  integration, dashboard) with 97 % line coverage of `src/` (excluding plotting and the
  CLI module, which the integration tests exercise). ruff, mypy, GitHub Actions CI,
  pre-commit, Makefile and a pinned lock file.

## 2. Headline results

| Claim | Number | File |
|---|---|---|
| Learned-model robust objectives vs nominal: failure rate | 11.9 % (nominal) vs 9.9 % (CVaR); no Holm-significant difference (all adjusted p = 1.0) | `e1_optimisation/summary.csv`, `tests.csv` |
| Learned-model CVaR vs nominal: 5th-percentile strength | 3.67 vs 4.02 MPa (difference −0.34 [−0.61, −0.10]; Holm p = 1.0) | `e1_optimisation/tests.csv` |
| Learned-model expected utility | 6.15–6.66 MPa across objectives (nominal 6.53) | `e1_optimisation/summary.csv` |
| Oracle (true function): composition-aware vs nominal | failure 9.6 % vs 17.6 %; expected utility 8.52 vs 8.05 MPa (exploratory) | `e1_optimisation/tests_oracle_exploratory.csv` |
| Regret of learned-model recommendations | 1.87–2.37 MPa of expected utility | `e1_optimisation/summary.csv` |
| Why: noise vs truth at recommended settings | model noise 0.36 vs true 0.015 (log variance); feedstock share 19–22 % vs 84 % | `e4_propagation/decomposition.csv` |
| Only significant ablation | 2× composition spread: expected objective failure 16.7 % → 13.1 % (Holm p = 0.041) | `e1b_ablations/tests.csv` |
| Best probabilistic model (120 batches) | hybrid GP: CRPS 0.225, NLL 0.53; coverage 0.86 (0.93 with conformal) | `e2_models/metrics.csv`, `conformal.csv` |
| Physics prior and data efficiency | CRPS 0.377 vs 0.488 (hybrid vs plain GP) at 15 batches; equal at 120 | `e2_models/metrics.csv` |
| Active learning | no strategy distinguishable from random after 52 jobs | `e3_active_learning/per_round.csv` |
| Unscented transform | 2.5 % SD error on the hybrid GP, 26 % on the ground truth | `e4_propagation/errors.csv` |
| Sensitivity | nozzle temperature total index 0.58 (strength), 0.74 (success); composition 0.25 / 0.29 | `e5_sensitivity/global.csv` |
| Public data (Kaggle PLA/ABS) | CRPS 0.206–0.212 for all models; simulator directions agree for 3 of 6 settings | `e7_public_benchmark.csv`, `e7_public_trends.csv` |

## 3. What worked and what did not

**Worked**

- The pipeline end to end: real-data schema → validated arrays → models → propagation →
  robust recommendation → next-experiment table, all reproducible bit-for-bit (a clean
  re-run reproduced every result file byte-for-byte, apart from wall-clock timings).
- The physics-residual GP was the most accurate probabilistic model and clearly more
  data-efficient than the plain GP at small sample sizes.
- Early stopping turned overconfident NLL-trained networks into well-covering ones
  (deep ensemble 90 % coverage 0.92). Conformal wrapping restored nominal coverage in
  distribution.
- The oracle experiment showed what composition-aware optimisation *can* gain in this
  simulator: nearly half the print failures.
- The variance decomposition diagnosed *why* the learned-model pipeline did not realise that gain.

**Did not work**

- Robust objectives with a learned model: no confirmed benefit over nominal, and the
  quantile/CVaR objectives made the weakest parts slightly weaker.
- Model-based active learning: no detectable improvement over random or Sobol within 52 jobs.
- The unscented transform on the cliff-like ground truth; the heteroscedastic GP (worse
  NLL than the plain GP); deep ensembles as optimiser models (much worse recommendations
  despite good coverage); everything under out-of-distribution feedstock (45–48 % failures).
- The first simulator version had no composition × process interaction, so robust
  optimisation could not matter. The interaction we then added is a modelling choice
  (ADR-012).

## 4. Known limitations

The simulator is not validated against any real recycled feedstock, and its coefficients
are partly assumptions. The benefit of robust optimisation in it depends on an interaction
we added. Models are weak below about 100 successful prints in 13 dimensions. Conformal
guarantees fail under feedstock shift. Nothing here is validated for flight or
safety-critical use. Full list: `docs/LIMITATIONS.md`.

## 5. The three most valuable next steps with real lab data

1. **Run the starter design** (`data/examples/starter_design.csv`: 20 jobs, 60 specimens,
   3 batches with repeated settings) and measure composition with sub-sample repeats. The
   repeats directly estimate batch-to-batch variance, the quantity the whole method
   depends on. Then check whether settings × composition interactions exist at all:
   does the best nozzle temperature or speed differ between batches?
2. **Reduce model error before relying on robust objectives.** The synthetic results say
   the bottleneck is model error, not the robust formulation. Spend early experiments on
   space-filling coverage of the 3–4 settings that matter most (nozzle temperature first),
   and narrow the search box to the printable region, to cut the effective dimension.
3. **Validate predictions prospectively.** Print the recommended CVaR setting and the
   nominal setting on at least two *new* batches, and compare with the predicted 90 %
   intervals and success probabilities. That is the real version of experiment E1, and the
   only test that counts.

## 6. Talking points (each grounded in a result here)

- "I pre-registered my evaluation and reported a negative result: with a model trained on
  about 100 successful prints, robust optimisation did not beat the naive approach, even
  though it nearly halves failures when the model is perfect." (E1, oracle comparison.)
- "I found out why: the model mistook its own lack of knowledge for random noise, so it
  could not see the composition uncertainty it was supposed to be robust to." (E4.)
- "Adding simple physics as a prior made the model much better with little data, which is
  the regime a student lab is in." (E2 learning curves.)
- "Good uncertainty calibration in cross-validation did not guarantee good decisions: the
  best-covering model gave the worst recommendations." (E1b deep ensemble.)
- "Every synthetic result is traceable to a file and a command, the whole study reproduces
  bit-for-bit in about 16 minutes on a laptop-class CPU, and the tool is ready for real
  data through a validated CSV schema." (`results/full/`, `manifest.json`, `data/schema.md`.)
