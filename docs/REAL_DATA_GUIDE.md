# Using RobustPrint with real experiments

This guide is for the researcher who will replace the synthetic data with measurements.
Nothing in the repository has been validated against real recycled feedstock yet; the
first real campaign is what will show whether the models and the robust-optimisation idea
are useful for your material.

## 1. Before printing: decide what you will record

Use the CSV schema in [`data/schema.md`](../data/schema.md). The minimum per print job:

- `experiment_id`, `data_origin = experimental`, `batch_id`, `date`, `source_description`
- the five composition fractions of the batch (`frac_pe`, `frac_pet`, `frac_pa`,
  `frac_fiber`, `frac_contaminant`), summing to 1
- all eight print settings (even the ones you did not change)
- `print_success`, `n_replicates`, and for successful jobs `tensile_strength_MPa`
  (mean of the specimens) and preferably `tensile_strength_sd_MPa`

Strongly recommended: `moisture_pct` (weigh a sample before and after drying), an
uncertainty for the composition (`sd_<component>` from repeated sorting/weighing, or
`composition_concentration`), and `defect_notes` describing failures ("clogged after
3 layers", "corner lifted").

If your waste stream has different components (for example a virgin PLA carrier), add
them to `src/robustprint/physics/materials.py`; the schema columns follow automatically.
The property values there only matter for the physics baseline and the hybrid model's prior.

### Composition: how to measure it

Sort and weigh a representative sample of each shredded batch by material class (visual
or float/sink separation, FTIR if available). Repeat the sorting on 2–3 sub-samples: the
spread between sub-samples is your `sd_<component>`. Measurement error enters the models
as noise, so a rough but honest uncertainty is better than a precise-looking guess.

### Testing

Print tensile specimens following one standard throughout (for example ASTM D638 Type IV
or ISO 527-2 1A), the same orientation and the same number of perimeters. Test at the same
crosshead speed and condition specimens the same way. Record every printed job, including
failures: failures are what the success model learns from.

## 2. A minimal first campaign

```bash
robustprint starter-design --out starter_design.csv
```

This writes 20 print jobs (60 specimens). The pre-generated version is
[`data/examples/starter_design.csv`](../data/examples/starter_design.csv):

- **Batch 1:** 12 settings from a maximin Latin hypercube over the four settings that
  govern interlayer bonding and extrusion: nozzle temperature (240–290 °C), print speed
  (15–60 mm/s), layer height (0.12–0.30 mm) and fan (0–60 %), rounded to steps your
  printer can set. Bed 100 °C, infill 100 %, raster 0°, flow 100 % are held fixed.
- **Batches 2 and 3:** repeat 4 of those settings. Repeats on new batches separate
  batch-to-batch variation from setting effects, which is exactly the uncertainty the
  robust optimiser needs.
- **3 replicates** per job.

Adjust the temperature range to your material. Narrow it if the lower end cannot melt
your dominant polymer, or if the upper end smokes.

## 3. Import, validate and train

```bash
robustprint validate my_experiments.csv        # fix every ERROR it lists
robustprint train my_experiments.csv --model hybrid_gp --out models/mine.pkl
```

With fewer than ~30 successful jobs, treat every prediction as a rough guide. The
synthetic learning curves (`results/full/figures/learning_curves.png`) suggest errors keep
falling up to the largest size tested. The model file is a Python pickle: only load files
you created yourself.

## 4. Ask for predictions and settings

```bash
# predictive distribution for the next, uncharacterised batch
robustprint predict models/mine.pkl --from-data my_experiments.csv \
    --settings "nozzle_temp_C=265,print_speed_mm_s=20,layer_height_mm=0.16"

# robust settings (CVaR of strength, with P(success) >= 0.9)
robustprint optimize models/mine.pkl --from-data my_experiments.csv --objective cvar
```

`--from-data` estimates the feedstock distribution (mean composition and batch-to-batch
spread) from the batches in your file by maximum likelihood. With only 2–3 batches that
estimate is itself very uncertain. Consider passing `--concentration` explicitly, using a
*smaller* value (more spread) to stay on the conservative side.

How to read the output:

- `median_strength_MPa` and `interval90_MPa`: the strength of one part printed from the
  next batch, accounting for composition, model and noise uncertainty.
- `variance_shares`: where that uncertainty comes from. A large **epistemic** share means
  more data would help. A large **input** share means the feedstock itself is the problem
  (sorting or blending to tighten the composition would help more than more experiments).
- `predicted.p_success`: the model's estimate of the fraction of prints that will succeed.

## 5. Choose the next experiments and iterate

```bash
robustprint recommend-experiments models/mine.pkl --from-data my_experiments.csv \
    --n 4 --strategy targeted_ivr --out next_round
```

This writes `next_round.csv` and `next_round.md` with target compositions, settings,
predicted strength intervals, predicted success and relative cost. Blend as close to the
target composition as you can, but record the composition you *measured*. Print the jobs,
append the rows to your CSV, then retrain and repeat. `cost_aware_ivr` prefers cheaper
(faster) prints. `sobol` gives a model-free space-filling choice if you distrust the model.

In the synthetic active-learning experiment (20 initial jobs, 8 rounds of 4), the
model-based strategies did **not** beat random or Sobol selection, and `cost_aware_ivr`
was clearly *worse* than random: its cheap, fast prints failed more often, so its campaigns
produced fewer usable strength measurements (see README, "What did not work"). Until a
campaign has a few dozen successful jobs, space-filling designs are a sound default; do not
use `cost_aware_ivr` for an early campaign.

To try the whole loop before printing anything, open the browser Virtual Lab (`web/`,
README §3). It can also import your CSV for validation, model training and recommendations;
the file stays in your browser.

## 6. Before you trust a recommendation

1. Print a confirmation run at the recommended settings on at least two *new* batches.
2. Compare the measured strengths with `interval90_MPa`. If they often fall outside, the
   model is miscalibrated for your material; add data or use conformal intervals.
3. Never use these settings for flight hardware or anything safety-critical without
   qualification testing under the relevant standards (see [LIMITATIONS.md](LIMITATIONS.md)).
