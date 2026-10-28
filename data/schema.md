# Experiment CSV schema

One row = **one print job**: one feedstock batch printed with one set of settings, producing
`n_replicates` tensile specimens. Rows of the same batch repeat the batch columns.
Validate a file with:

```bash
robustprint validate my_experiments.csv
```

The validator lists every problem with its row number, `experiment_id`, the offending value
and the allowed range. A complete example (synthetic values, format only) is in
[`examples/example_experiments.csv`](examples/example_experiments.csv).

## Identification

| Column | Type | Required | Notes |
|---|---|---|---|
| `experiment_id` | string | yes | Unique per row. |
| `data_origin` | `experimental` \| `synthetic` | yes | Use `experimental` only for real measurements. Files that mix both produce a warning. |

## Batch-level columns

| Column | Type | Required | Valid range | Notes |
|---|---|---|---|---|
| `batch_id` | string | yes | | Batches define the cross-validation groups. Never reuse an id for a different batch. |
| `date` | date `YYYY-MM-DD` | yes | | Date the batch was prepared. |
| `source_description` | string | yes | | Free text, e.g. "shredded food packaging, lot 3". |
| `frac_pe` | float | yes | 0–1 | Mass fraction of polyethylene (films, bags). |
| `frac_pet` | float | yes | 0–1 | Mass fraction of PET. |
| `frac_pa` | float | yes | 0–1 | Mass fraction of polyamide / nylon. |
| `frac_fiber` | float | yes | 0–1 | Mass fraction of fibres (aramid, cloth). |
| `frac_contaminant` | float | yes | 0–1 | Everything else (foam, cellulose, residues). |
| `sd_<component>` | float | no | 0–0.5 | Standard deviation of each fraction (analysis uncertainty), e.g. `sd_pe`. |
| `composition_concentration` | float | no | ≥ 1 | Alternative to `sd_*`: Dirichlet concentration of the analysis error (higher = more certain). |
| `moisture_pct` | float | no | 0–20 | Moisture content (mass %). Missing values are imputed with the median of the file (or 0.3 % if none is given). |
| `melt_flow_index_g_10min` | float | no | 0–500 | Optional, currently not used by the models. |
| `particle_size_mm` | float | no | 0–50 | Optional, currently not used by the models. |

The five fractions must sum to 1 (± 0.01). If you recorded percentages, divide by 100. Zeros
are allowed; they are replaced by 1e-4 before log-ratio transforms.

The component list comes from the material library in
`src/robustprint/physics/materials.py`. To use other components (for example a virgin PLA
carrier), add a `Component` there; the schema's `frac_*` columns follow automatically.

## Print-level columns (all required)

| Column | Unit | Search range | Physically valid range |
|---|---|---|---|
| `nozzle_temp_C` | °C | 210–290 | 150–350 |
| `bed_temp_C` | °C | 40–110 | 15–150 |
| `print_speed_mm_s` | mm/s | 10–80 | 1–300 |
| `layer_height_mm` | mm | 0.10–0.35 | 0.02–1.0 |
| `infill_pct` | % | 20–100 | 0–100 |
| `raster_angle_deg` | degrees from load axis | 0–90 | −90–90 |
| `flow_rate_pct` | % | 85–120 | 50–200 |
| `cooling_fan_pct` | % | 0–100 | 0–100 |

Values outside the physically valid range are errors. Values outside the search range are
accepted with a warning (models extrapolate there).

## Outcome-level columns

| Column | Type | Required | Valid range | Notes |
|---|---|---|---|---|
| `print_success` | bool | yes | true/false (also yes/no, 1/0) | False if the job could not produce testable specimens. |
| `n_replicates` | int | yes | ≥ 1 | Specimens tested. |
| `tensile_strength_MPa` | float | yes (may be empty) | 0–500 | Mean ultimate tensile strength over replicates. Leave empty for failed jobs. |
| `tensile_strength_sd_MPa` | float | no | 0–200 | Sample standard deviation over replicates. |
| `youngs_modulus_GPa` | float | no | 0–100 | Mean Young's modulus. |
| `elongation_pct` | float | no | 0–1000 | Mean elongation at break. |
| `defect_notes` | string | no | | Free text (e.g. "warped corner", "nozzle clog"). |

Rows with `print_success = true` and no strength are used for the success model only.
Rows with `print_success = false` are used for the success model only; any strength given
is ignored (with a warning).
