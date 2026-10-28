"""RobustPrint dashboard: ``streamlit run app/streamlit_app.py``.

Sliders set the feedstock composition and its uncertainty; the page shows the predictive
distribution of strength at chosen settings, robust recommended settings and a table of
next experiments. By default the model is trained on a SYNTHETIC campaign; upload your own
CSV (data/schema.md) to use real data.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from robustprint import space
from robustprint.active.planner import experiments_table, select_experiments
from robustprint.data.loaders import to_arrays
from robustprint.data.synthetic import generate_dataset
from robustprint.data.validation import validate_frame
from robustprint.optimize.objectives import OBJECTIVES, RobustProblem
from robustprint.optimize.robust import recommend
from robustprint.physics.composition import closure
from robustprint.physics.materials import DEFAULT_LIBRARY
from robustprint.physics.simulator import ISS_NOMINAL, Feedstock
from robustprint.pipeline import train_predictor
from robustprint.uncertainty.propagation import ModelPredictor, monte_carlo, sample_feedstock

st.set_page_config(page_title="RobustPrint", layout="wide")


@st.cache_resource(show_spinner="Training on a synthetic campaign...")
def synthetic_predictor(seed: int) -> ModelPredictor:
    """Model trained on a SYNTHETIC campaign (60 batches x 4 settings x 3 replicates)."""
    frame, _ = generate_dataset(ISS_NOMINAL, 60, 4, np.random.default_rng(seed))
    return train_predictor(to_arrays(frame), "hybrid_gp", seed=seed)


@st.cache_resource(show_spinner="Training on your data...")
def uploaded_predictor(payload: bytes) -> tuple[ModelPredictor | None, str]:
    """Model trained on an uploaded CSV, or None with the validation report."""
    frame, report = validate_frame(pd.read_csv(io.BytesIO(payload)))
    if not report.ok:
        return None, report.summary()
    return train_predictor(to_arrays(frame), "hybrid_gp", seed=0), report.summary()


st.title("RobustPrint")
st.caption(
    "Uncertainty-aware print settings for recycled, variable-composition feedstock. "
    "Not validated for flight or safety-critical use."
)

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Your experiment CSV (optional)", type="csv")
    if upload is not None:
        predictor, text = uploaded_predictor(upload.getvalue())
        if predictor is None:
            st.error("Your file did not validate:")
            st.code(text)
            st.stop()
        source = "your uploaded data"
    else:
        predictor = synthetic_predictor(0)
        source = "SYNTHETIC data (simulator)"
    st.info(f"Model trained on {source}.")

    st.header("Feedstock (next batch)")
    raw = [
        st.slider(f"{c.name}: {c.description}", 0.0, 1.0, float(m), 0.01)
        for c, m in zip(DEFAULT_LIBRARY.components, ISS_NOMINAL.mean, strict=True)
    ]
    mean = closure(np.maximum(np.array(raw), 1e-3))
    concentration = st.slider(
        "Batch-to-batch consistency (Dirichlet concentration)",
        5.0,
        300.0,
        35.0,
        5.0,
        help="Higher means batches are more alike; 35 gives about ±8 % spread on PE.",
    )
    moisture = st.slider("Median moisture (%)", 0.05, 2.0, 0.3, 0.05)
    feedstock = Feedstock("dashboard", tuple(mean.tolist()), concentration, moisture)
    st.caption(
        "Normalised mean: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(DEFAULT_LIBRARY.names, mean, strict=True))
    )

st.subheader("Print settings")
cols = st.columns(4)
x = np.array(
    [
        cols[i % 4].slider(
            p.name, p.low, p.high, float(np.clip(p.default if p.name != "nozzle_temp_C" else 265.0, p.low, p.high))
        )
        for i, p in enumerate(space.PARAMETERS)
    ]
)

samples = sample_feedstock(feedstock, 2048, np.random.default_rng(1))
pred, p_success = predictor.predict(samples.composition, samples.moisture, np.tile(x, (len(samples), 1)))
draws = np.exp(pred.mean[:, None] + pred.std[:, None] * np.random.default_rng(2).standard_normal((len(samples), 8)))
res = monte_carlo(predictor, x, samples)

left, right = st.columns([3, 2])
with left:
    st.subheader("Predicted strength of a part from the next batch")
    hist, edges = np.histogram(draws.ravel(), bins=40)
    st.bar_chart(pd.DataFrame({"parts": hist}, index=np.round(0.5 * (edges[1:] + edges[:-1]), 2)))
    st.caption(f"Model trained on {source}. Histogram over composition, model and noise uncertainty.")
with right:
    st.metric("Median strength (MPa)", f"{np.exp(res.mean):.2f}")
    lo, hi = np.quantile(draws, [0.05, 0.95])
    st.metric("90 % interval (MPa)", f"{lo:.2f} – {hi:.2f}")
    st.metric("Predicted print success", f"{100 * res.p_success:.0f} %")
    shares = res.shares()
    st.write(pd.DataFrame({"share of variance": shares}).style.format("{:.0%}"))

st.subheader("Robust recommendation")
objective = st.selectbox("Objective", OBJECTIVES, index=OBJECTIVES.index("cvar"))
if st.button("Optimise settings"):
    with st.spinner("Optimising (CMA-ES over a Monte-Carlo objective)..."):
        problem = RobustProblem(predictor, feedstock, seed=0)
        rec = recommend(problem, objective, budget=600, seed=0)
        stats = problem.statistics(rec.params[None, :])
    st.session_state["rec"] = rec.params
    st.dataframe(pd.DataFrame([space.as_dict(rec.params)]).round(2), hide_index=True)
    st.write({k: round(float(v[0]), 3) for k, v in stats.items()})

st.subheader("Next experiments")
strategy = st.selectbox("Strategy", ["targeted_ivr", "cost_aware_ivr", "max_variance", "sobol", "random"])
n_next = st.number_input("Number of print jobs", 1, 12, 4)
if st.button("Propose experiments"):
    rec_params = st.session_state.get("rec")
    if rec_params is None:
        rec_params = recommend(RobustProblem(predictor, feedstock, seed=0), "cvar", budget=400, seed=0).params
    comp, moist, params, scores = select_experiments(
        predictor, feedstock, int(n_next), strategy, np.random.default_rng(0), recommended=rec_params, n_candidates=256
    )
    table = experiments_table(predictor, comp, moist, params, scores, strategy)
    st.dataframe(table, hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False), "next_experiments.csv", "text/csv")
