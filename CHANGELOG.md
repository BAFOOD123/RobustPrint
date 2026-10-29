# Changelog

## v1.1.0 — 2026-10-29

**Corrected results.** Every v1.0.0 number is superseded.

- **Fixed (data generator):** v1.0.0 drew print success separately for each replicate
  specimen and averaged failed specimens into a job's strength as 0 MPa, biasing about
  46 % of successful job means downward (log-ratio SD 0.41 instead of about 0.08). A job
  now succeeds or fails as a whole and its replicates are noisy specimens of the same
  part (`physics.simulator.print_job`); a regression test checks that job means are
  unbiased. Found while porting the generator to TypeScript. All experiments were re-run
  with the pre-registered protocol unchanged (METHODS §8.7).
- **Changed conclusions:** composition-aware objectives now significantly reduce failures
  on unseen batches (H1 supported, −5.3 to −6.4 points); quantile and CVaR lower the
  5th-percentile strength of good parts (H2 contradicted); the chance-constrained
  objective raises expected utility (+0.67 MPa). The v1.0.0 "inflated noise" diagnosis
  was an artefact of the bug. The physics prior no longer helps with little data.
  Cost-aware active learning is worse than random.
- **Added: RobustPrint Virtual Lab** (`web/`): a single-file browser simulation
  environment with a TypeScript re-implementation of the simulator, hybrid GP,
  boosted-tree success classifier, robust objectives, CMA-ES and acquisition strategies;
  golden-file parity tests against the Python code (`scripts/export_web_golden.py`),
  exported study results (`scripts/export_web_data.py`), a Playwright end-to-end test, a
  CI job, a GitHub Pages workflow and ADR-020.
- Docs: README, FINAL_REPORT, MODEL_CARD, LIMITATIONS and REAL_DATA_GUIDE rewritten for
  the corrected results.

## v1.0.0 — 2026-10-28

First release.

- Data: CSV schema for real experiments with row-level validation; synthetic campaign
  generator (every row labelled `synthetic`); feedstock-distribution estimation; starter
  experimental design; optional loader for the Kaggle FDM dataset (not redistributed).
- Physics: compositional transforms (ilr), mixing models, process models and a frozen
  ground-truth simulator (v1.0) of ISS-like mixed waste.
- Models: physics rule of mixtures, Bayesian ridge, quantile GBM, ARD GP, heteroscedastic
  GP, physics-residual hybrid GP, deep ensemble, MC dropout; split conformal and CQR;
  calibrated print-success classifiers.
- Uncertainty: Monte-Carlo and unscented propagation over the feedstock distribution,
  variance decomposition, grouped Sobol sensitivity.
- Optimisation: nominal, expected, quantile, CVaR and chance-constrained objectives under a
  success constraint; CMA-ES; ε-constraint Pareto fronts.
- Active learning: max-variance, targeted IVR, cost-aware IVR; next-experiments table.
- Evaluation: pre-registered protocol; full results in `results/full/` (synthetic).
  Headline: with a learned model, robust objectives did not significantly beat nominal;
  with the true function they nearly halved print failures.
- CLI, Streamlit dashboard, 100 tests (97 % coverage), CI, pre-commit, Makefile, lock file.
