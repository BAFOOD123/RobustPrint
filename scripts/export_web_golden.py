"""Export golden test vectors from the Python reference implementation for the web engine.

Usage: python scripts/export_web_golden.py  (writes web/tests/golden/*.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robustprint import space
from robustprint.physics import process
from robustprint.physics.composition import dirichlet_ilr_moments, ilr, ilr_inverse
from robustprint.physics.simulator import ISS_NOMINAL, true_outcomes

OUT = Path(__file__).resolve().parents[1] / "web" / "tests" / "golden"
KEYS = ["strength_MPa", "modulus_GPa", "elongation_pct", "p_success", "p_clog", "p_warp", "noise_cv", "bond",
        "porosity", "extrusion", "immiscibility", "degradation", "unmelted", "fiber_volume", "inclusion_volume"]


def main() -> None:
    """Write simulator and composition golden vectors."""
    rng = np.random.default_rng(20261028)
    n = 400
    comp = rng.dirichlet(np.full(5, 0.8), size=n)
    comp[:100] = rng.dirichlet(35 * ISS_NOMINAL.mean_array(), size=100)
    comp = np.maximum(comp, 1e-6)
    comp /= comp.sum(1, keepdims=True)
    params = space.from_unit(rng.random((n, 8)))
    moisture = np.exp(rng.normal(np.log(0.3), 0.8, n))
    effect = rng.normal(0, 0.05, n)
    out = true_outcomes(comp, params, moisture, effect)
    rows = [{"composition": comp[i].tolist(), "params": params[i].tolist(), "moisture": float(moisture[i]),
             "batch_effect": float(effect[i]), **{k: float(out[k][i]) for k in KEYS},
             "print_time": float(process.print_time_proxy(params[i:i + 1, 2], params[i:i + 1, 3], params[i:i + 1, 4])[0]),
             "energy": float(process.energy_proxy(params[i:i + 1, 0], params[i:i + 1, 1], params[i:i + 1, 2],
                                                  params[i:i + 1, 3], params[i:i + 1, 4])[0])}
            for i in range(n)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "simulator.json").write_text(json.dumps(rows))
    z = ilr(comp[:50])
    alpha = 35 * ISS_NOMINAL.mean_array()
    m, c = dirichlet_ilr_moments(alpha)
    (OUT / "composition.json").write_text(json.dumps({
        "composition": comp[:50].tolist(), "ilr": z.tolist(), "ilr_inverse": ilr_inverse(z).tolist(),
        "alpha": alpha.tolist(), "ilr_mean": m.tolist(), "ilr_cov": c.tolist()}))
    export_gp()
    print(f"wrote {OUT}")


def export_gp() -> None:
    """Fit the Python hybrid GP on a small campaign and export data, hyperparameters, predictions."""
    from robustprint.eval.common import campaign
    from robustprint.models.features import FeatureBuilder
    from robustprint.models.registry import physics_prior_on_features
    from robustprint.models.gp import GPRegressor

    fb = FeatureBuilder()
    arrays = campaign(ISS_NOMINAL, 30, 4, seed=11)
    X = fb.transform(arrays.composition, arrays.params, arrays.moisture)
    ok = arrays.strength_rows
    Xs, y = X[ok], np.log(arrays.strength[ok])
    resid = y - physics_prior_on_features(Xs, fb)
    gp = GPRegressor(n_restarts=2, random_state=0).fit(Xs, resid)
    test = campaign(ISS_NOMINAL, 5, 4, seed=12)
    Xt = fb.transform(test.composition, test.params, test.moisture)
    pred = gp.predict(Xt)
    k = gp.gp_.kernel_
    (OUT / "gp.json").write_text(json.dumps({
        "composition": arrays.composition[ok].tolist(), "params": arrays.params[ok].tolist(),
        "moisture": arrays.moisture[ok].tolist(), "log_strength": y.tolist(),
        "features": Xs.tolist(), "prior": physics_prior_on_features(Xs, fb).tolist(),
        "test_features": Xt.tolist(), "test_prior": physics_prior_on_features(Xt, fb).tolist(),
        "amp": float(k.k1.k1.constant_value), "length_scales": k.k1.k2.length_scale.tolist(),
        "noise": float(k.k2.noise_level), "lml": float(gp.gp_.log_marginal_likelihood_value_),
        "pred_mean": pred.mean.tolist(), "pred_var_epistemic": pred.var_epistemic.tolist(),
        "pred_var_aleatoric": pred.var_aleatoric.tolist()}))


if __name__ == "__main__":
    main()
