# PROGRESS log

Timestamped log of work on RobustPrint. Times are UTC (container clock).

## 2026-10-28 — session start
- Repository was empty (no commits). Branch: `claude/robustprint-uncertainty-optimization-momgzw`.
- Environment: 4 CPU cores, 15 GB RAM, no GPU. Python 3.11/3.12/3.13 available; `uv`, `ruff`, `mypy` on PATH.
- Plan: follow mission phases in order (research → data → models → optimizer/AL → eval → CLI/tests/CI → docs → audit).

## 2026-10-28 ~03:10 — Phase 1 done, Phase 2 in progress
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

## 2026-10-28 ~04:10 — Phase 3/4 implementation
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
