# Changelog

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
