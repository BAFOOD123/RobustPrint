# Model card: RobustPrint strength and print-success models

## Model details

- **What:** a probabilistic regressor for the log tensile strength of FDM-printed parts
  from recycled, mixed feedstock, plus a classifier for print success. The default is a
  physics-residual Gaussian process (`hybrid_gp`) with a gradient-boosting success
  classifier with isotonic calibration (`gbm_isotonic`). Alternatives: `gp`, `het_gp`,
  `deep_ensemble`, `mc_dropout`, `gbm_quantile`, `ridge_interactions`, `physics_rom`.
- **Inputs:** composition as isometric log-ratio coordinates of 5 mass fractions
  (PE, PET, PA, fibre, contaminant), 8 print settings scaled to the search box, and log
  moisture.
- **Outputs:** Gaussian on log strength, split into epistemic and aleatoric variance;
  P(success). Through Monte Carlo over a feedstock distribution, also the predictive
  distribution for a part from the *next* batch and its variance decomposition.
- **Version:** 1.0.0. Licence: MIT. Code: this repository.

## Intended use

- Planning experiments on recycled feedstock: which settings to try first, which
  experiments are most informative, how much of the uncertainty is due to the feedstock
  versus the model.
- Teaching and research on uncertainty-aware process optimisation.
- Use with the researcher's own data in the documented CSV schema, retrained per material
  system.

## Out-of-scope use

- Certifying, qualifying or accepting parts for flight, crewed environments or any
  safety-critical application.
- Predicting the properties of a material system it was not trained on. The default
  trained models (in the dashboard and in `results/`) are trained on **synthetic data**
  and say nothing about any real material.
- Extrapolating outside the search box (`data/schema.md`) or to feedstocks unlike the
  training batches (see the OOD results).

## Training data

- Default and all reported experiments: synthetic campaigns from the RobustPrint
  simulator (`iss_nominal` feedstock; 15–120 batches × 4 settings × 3 replicates). Every
  row is labelled `data_origin = synthetic`.
- Optional secondary benchmark: the Kaggle "3D Printer Dataset for Mechanical Engineers"
  (50 rows, PLA/ABS, licence unknown, not redistributed).

## Evaluation

Protocol: [`METHODS.md`](METHODS.md) §8 (pre-registered). Metrics: RMSE (MPa), NLL and
CRPS (log scale), quantile calibration error, 90 % coverage and width; Brier score, log
loss, ECE for the classifier; grouped cross-validation by batch, new-batch and shifted
(OOD) test sets. The README reports the numbers with links to the result files. All of
them are synthetic.

Known behaviour (synthetic):

- The GP family under-covers on new batches without conformal calibration. The early-stopped
  deep ensemble is close to nominal coverage in distribution.
- All models degrade under feedstock shift; normalised conformal intervals lose their
  coverage there.
- The fitted models attribute much more of the predictive variance to noise than the
  ground truth does, and less to composition (`results/full/e4_propagation/decomposition.csv`).

## Ethical and practical considerations

- **Over-trust:** a confident-looking interval from a model trained on 20–50 jobs can be
  wrong. Confirmation prints on new batches are required before relying on any setting.
- **Synthetic-to-real gap:** results from the simulator must never be presented as
  experimental findings. Figures carry a "SYNTHETIC DATA" stamp, result files a header.
- **Safety:** printing mixed waste can release fumes, especially at the high temperatures
  that PET-containing feedstock needs and when contaminants degrade. Print in a ventilated
  enclosure and follow your lab's safety rules.
- **Data:** no personal data is used or stored.

## Caveats and recommendations

Retrain on real data before any use. Report the composition uncertainty honestly. Prefer
conformal intervals for coverage claims. Re-check calibration whenever the feedstock
source changes.
