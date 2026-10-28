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
