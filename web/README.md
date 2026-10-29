# RobustPrint Virtual Lab

A browser simulation environment for the RobustPrint method. Everything runs on your
machine; nothing is uploaded. **All printing here happens in a simulator with assumed
physics (synthetic data). Not validated against real recycled material, and not for flight
or safety-critical use.**

## The six steps

| Step | What you do | What runs |
|---|---|---|
| 1 Bench | Set the feedstock distribution and the print settings in the left rail | Ground-truth simulator: the average batch (with a breakdown of where strength is lost), 48 random batches printed once each, and response curves over 160 batches |
| 2 Campaign | Plan a training campaign, or import your own CSV ([schema](../data/schema.md)) | Space-filling design, printed in the simulator with job-level success and 3 specimens per job; CSV validation with row-level messages |
| 3 Model | Train the learned model | Physics-residual GP (Adam on the exact log marginal likelihood) + boosted-tree success classifier with isotonic calibration; scored on held-out batches, then refitted on all |
| 4 Optimise | Pick objectives and a predictor (learned, perfect, or both) | Monte Carlo over the feedstock with common random numbers, CMA-ES inside the safe nozzle window, then deployment on 300–2 000 fresh batches (optionally from a shifted stream) |
| 5 Next experiments | Propose jobs, print them, retrain, repeat | Greedy batch selection with GP fantasy updates (targeted / cost-aware variance reduction, max variance, space filling, random); true regret of each round's CVaR recommendation |
| 6 Study results | Read the pre-registered results | Pre-computed Python results from `results/full` (exported by `scripts/export_web_data.py`) |

With imported data the simulator's hidden truth is unknown, so the steps that score
against it (deployment, regret) are only offered for synthetic campaigns.

## Build and test

```bash
npm ci
npm test              # parity and unit tests (vitest)
npm run build         # dist/index.html: one self-contained file, also works from disk
npm run dev           # development server
npm run build:artifact  # dist-artifact/artifact.html for claude.ai (React from cdnjs)
# end-to-end smoke test (needs Playwright with Chromium):
node tests/e2e.mjs dist/index.html [screenshot-dir]
```

## How faithful is it?

The TypeScript engine in `src/engine` is a second implementation of the Python package,
not a wrapper. `tests/` compares it with golden files written by
`scripts/export_web_golden.py`:

| Component | Agreement with the Python code |
|---|---|
| Simulator (all outputs, 400 random rows) | 1e-9 relative |
| ilr transform and inverse | 1e-10 |
| Dirichlet moments in ilr space | 1e-8 |
| Features and physics prior | 1e-9 |
| GP predictions at fixed hyperparameters | 1e-7 |
| GP log marginal likelihood and its gradient | 1e-6 (finite-difference check of the gradient) |
| Isotonic regression | exact (1e-10) |
| Raw boosted-tree scores vs `HistGradientBoostingClassifier` | 0.04 log-odds mean absolute difference |
| Calibrated success probabilities | error vs the true P(success) within 25 % of scikit-learn's (measured: 1 % lower) |

Hyperparameter fitting (Adam with restarts vs L-BFGS) and the classifier's CV folds
differ, so models trained in the browser agree with the Python ones statistically, not
bit for bit. Not ported: deep ensembles, MC dropout, the heteroscedastic GP, conformal
wrappers and Sobol analysis. See [ADR-020](../docs/DECISIONS.md).

## Layout

```
src/engine/    simulator, compositions, GP, boosted trees, objectives, CMA-ES, acquisition, CSV
src/worker/    Web Worker protocol and the compute behind each step
src/views/     the six steps; src/ui/ the rail and controls; src/charts/ SVG charts
src/data/      study.json (exported results; synthetic)
src/umd/       React shims for the Artifact build only
tests/         vitest parity/unit tests, golden files, Playwright smoke test
```
