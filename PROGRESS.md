# PROGRESS log

Timestamped log of work on RobustPrint. Times are UTC (container clock).

## 2026-10-28 — session start
- Repository was empty (no commits). Branch: `claude/robustprint-uncertainty-optimization-momgzw`.
- Environment: 4 CPU cores, 15 GB RAM, no GPU. Python 3.11/3.12/3.13 available; `uv`, `ruff`, `mypy` on PATH.
- Plan: follow mission phases in order (research → data → models → optimizer/AL → eval → CLI/tests/CI → docs → audit).

## 2026-10-28 ~02:45 — Phase 1 done, Phase 2 in progress
- Verified references (docs/REFERENCES.md). Kaggle "3D Printer Dataset for Mechanical Engineers"
  found (50 rows, PLA/ABS); licence "Unknown" → not redistributed; optional download with checksum.
- Could not find a peer-reviewed open table of bulk properties for all components; used Engineering
  ToolBox ranges and flagged everything else as ASSUMPTION (ADR-007).
- Built physics: compositional transforms (ilr), mixing models (ROM, inverse ROM, Halpin–Tsai,
  immiscibility penalty, inclusion factor), process models (bonding via t^(1/4) healing, extrusion
  limit, porosity, Tsai–Hill raster effect), ground-truth simulator, synthetic data generator.
- IMPORTANT (researcher degree of freedom, logged in ADR-012): simulator v0 had no
  composition×process interaction, so on the *true* function nominal and robust optima coincided.
  Added a fibre×volumetric-flow clogging term (physically motivated) and made the melt-rate limit
  less temperature-sensitive. Also moderated unmelted-polymer clogging so a space-filling design has
  ~43 % print success (was ~26 %). Simulator frozen after this point (v1.0).
- Next: schema doc, tests for physics/data, then models.

## 2026-10-28 ~03:10 — Phase 3/4 implementation
- Models implemented; first CV benchmark (synthetic): deep ensemble/MC dropout badly overconfident
  (NLL 4-7) until early stopping on a validation split was added (ADR-015); GP learned ~zero noise
  and under-covered → 3 % CV noise floor (ADR-014). Logistic poly-2 success model worse than base
  rate at C=1 → C=0.03; GBM+isotonic remains better and is the default.
- Robust objectives (MC + CRN), CMA-ES optimiser with safe nozzle window, deployment evaluator,
  oracle predictor, active-learning acquisitions (max-variance, targeted IVR, cost-aware IVR).
- Pilot E1 (8 seeds): robust objectives lower failure rate (~0.15 → ~0.10); no clear change in
  expected utility; model-based recommendations far from oracle (regret ~1.6-2.0 MPa).
- Found worker thread oversubscription (load 15 on 4 cores) → threadpoolctl pinning.
- Pre-registered protocol in docs/METHODS.md §8 before final runs.

## 2026-10-28 03:30 — full run launched; docs drafted
- Full pre-registered run (`robustprint reproduce --full`) started 03:28 UTC.
- Fast pipeline: ~2.3 min (before batching ablations into one pool).
- Lint (ruff), mypy and 99 tests pass; coverage 97 % (viz/cli excluded from coverage source).
- Drafted LIMITATIONS, MODEL_CARD, REAL_DATA_GUIDE, ADR-013..019, starter-design command.
- MUST VERIFY against full results before release (drafted from pilot/fast runs):
  (1) REAL_DATA_GUIDE §5 claim that model-based AL did not beat random/Sobol;
  (2) MODEL_CARD "known behaviour" bullets; (3) LIMITATIONS: strength range 2-12 MPa,
  UT inaccuracy, GP under-coverage, conformal OOD coverage loss.

## 2026-10-28 04:05 — results in, README drafted
- First full run: E2 219 s, E1 159 s, E1b 614 s; E3 crashed (seed with only 4 successful initial
  jobs). Added fallback (defaults + random until 5 successes), documented as deviation §8.7.
- Profiling: success-classifier predict_proba was ~53 % of optimiser time, half of it for
  nominal statistics not needed by other objectives. Computing only what each objective needs gave
  bit-identical E1 results (diffed) and 2.7x speed-up. Clean full rerun launched 03:59 to get one
  consistent manifest and to check reproducibility against run 1.
- Central result (synthetic): with a learned model, robust objectives are NOT significantly better
  than nominal (all Holm p = 1); CVaR/quantile even lower q05 strength. With the oracle they halve
  failures (17.6 % → 9.6 %). Explanation found in E4: learned model's noise variance ~24x truth,
  composition share 19-22 % vs 84 % in truth.
- AL: no strategy distinguishable from random within 52 jobs. UT inaccurate on ground truth.
- README written from run-1 numbers; to be verified against clean run (scripts/readme_numbers.py).

## 2026-10-28 04:35 — final audit
- Clean full run (commit 27e1a57): 16 min 10 s; all result files byte-identical to run 1 except
  timings and two files run 1 had computed from rounded CSVs (6th significant digit).
- README numbers verified against results/full with scripts/readme_numbers.py; corrected 5
  roundings, the E7 claim (deep ensemble did not have best CRPS), "quadrupling" → "doubling",
  "best-calibrated" → "best coverage", "only significant effect" → two effects in one variant,
  E5 setting description, and the overclaim "calibrated uncertainty".
- Found and fixed a mislabelled sensitivity figure (shared y-axis with per-panel sort order).
- Fresh clone: make setup OK, make test: 100 passed, coverage 96.75 %; make fast: see below.
- make fast in the fresh clone: 83 s (E7 skipped: public data not downloaded). setup+test+fast: 2 min 38 s.
- Tagged v1.0.0.

## 2026-10-29 — v1.1.0: Virtual Lab and a corrected study
- Goal from the user: a web simulator that uses the project's data as an environment for
  simulation. Plan: port the engine to TypeScript (Vite + React + strict TS, Web Worker),
  verify against golden files from Python, then build the UI around the lab loop.
- **Bug found while porting:** the Python generator drew success per replicate and averaged
  failed replicates as 0 MPa (≈46 % of good job means biased; log-ratio SD 0.41 vs ~0.08).
  Fixed with `print_job` + regression test (commit 28a4e0e); full study re-run.
- Parity: simulator 1e-9 rel. (400 rows), ilr 1e-10, GP predictions 1e-7, LML gradient by
  finite differences, isotonic exact, raw boosted scores 0.04 log-odds.
- Browser findings, fixed before release: (1) the poly-2 logistic success model invented a
  raster-angle effect (P(success) 0.67 vs true 0.95 at 0°), pushing learned recommendations
  to 90°; replaced by a port of gbm_isotonic (excess error vs true P 0.045 vs 0.077 at 40
  batches). (2) CMA-ES budgets of 300–900 left raster/speed unconverged; now 800/1500/3000,
  matching the study. (3) Module workers from data: URLs fail from file://; classic worker.
  (4) Regret reference aligned with E3 (oracle CVaR optimum on a fixed truth sample, signed).
- Corrected results reverse several v1.0.0 conclusions (CHANGELOG). The first corrected
  run's timings were inflated by concurrent browser tests; a clean re-run gives the
  reported timing and checks byte-identical outputs.
