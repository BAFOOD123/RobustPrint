/** Model: train the physics-residual GP + success classifier in the worker and score it on held-out batches. */

import { useState } from "react";
import { HBarChart, LineChart, ScatterChart } from "../charts/charts";
import { fmt } from "../charts/kit";
import { useApp } from "../state";
import { Empty, mpa, pct, Progress, Readout } from "../ui/controls";
import { run } from "../worker/client";
import type { TrainResult } from "../worker/protocol";

export function useTrainer() {
  const { state, dispatch } = useApp();
  const [busy, setBusy] = useState<{ fraction: number; label: string } | null>(null);
  const [error, setError] = useState("");
  const train = async (jobs = state.jobs) => {
    setError("");
    setBusy({ fraction: 0, label: "Starting" });
    try {
      const result = await run<TrainResult>({ kind: "train", jobs, seed: state.seed }, (fraction, label) => setBusy({ fraction, label }));
      dispatch({ type: "trained", result, nJobs: jobs.length });
      return result;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    } finally {
      setBusy(null);
    }
  };
  return { train, busy, error };
}

function Results({ r, synthetic }: { r: TrainResult; synthetic: boolean }) {
  const [showTruth, setShowTruth] = useState(false);
  const h = r.holdout;
  const covOk = h ? h.coverage90 >= 0.85 && h.coverage90 <= 0.95 : false;
  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h3>Scored on batches it never saw</h3>
          {h && <p>{h.nTestBatches} held-out batches ({h.nTest} good jobs); model refitted on {h.nTrainBatches} batches</p>}
        </div>
        {h ? (
          <div className="readouts">
            <Readout label="RMSE" value={mpa(h.rmseMPa)} unit="MPa" />
            <Readout label="CRPS (log strength)" value={fmt(h.crps)} sub="proper score; lower is better" />
            <Readout label="90 % interval coverage" value={pct(h.coverage90, 0)} sub={<span className={`pill ${covOk ? "good" : "warn"}`}>{covOk ? "close to nominal" : h.coverage90 < 0.85 ? "overconfident" : "conservative"}</span>} />
            <Readout label="Print-success Brier" value={h.brier === null ? "–" : fmt(h.brier)} sub={h.baseRate !== null ? `base rate ${pct(h.baseRate, 0)}; guessing it scores ${fmt(h.baseRate * (1 - h.baseRate))}` : undefined} />
          </div>
        ) : (
          <p className="small muted">Too few batches for a hold-out split (needs 4 or more with at least 5 good jobs in training).</p>
        )}
      </section>
      {h && h.parity.length > 0 && (
        <div className="grid-2">
          <section className="panel">
            <div className="panel-head">
              <h3>Predicted vs measured</h3>
              {synthetic && (
                <label className="checks" htmlFor="parity-truth">
                  <span>
                    <input id="parity-truth" type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} /> against the hidden truth
                  </span>
                </label>
              )}
            </div>
            <ScatterChart
              ariaLabel="Predicted versus measured strength on held-out batches"
              diagonal
              xLabel={showTruth ? "true mean strength (MPa)" : "measured job strength (MPa)"}
              yLabel="predicted (MPa), 90 % interval"
              points={h.parity.map((p) => {
                const xv = showTruth && p.truth !== null ? p.truth : p.measured;
                const inside = xv >= p.lo && xv <= p.hi;
                return {
                  x: xv,
                  y: p.predicted,
                  lo: p.lo,
                  hi: p.hi,
                  color: inside ? "var(--s1)" : "var(--s2)",
                  tip: (
                    <>
                      <div className="tip-title">Batch {p.batchId}</div>
                      <div className="tip-row">measured {mpa(p.measured)} MPa{p.truth !== null ? ` · truth ${mpa(p.truth)}` : ""}</div>
                      <div className="tip-row">predicted {mpa(p.predicted)} ({mpa(p.lo)}–{mpa(p.hi)})</div>
                    </>
                  ),
                };
              })}
            />
            <p className="small muted">Orange points fall outside their 90 % interval.</p>
          </section>
          <section className="panel">
            <div className="panel-head">
              <h3>Calibration</h3>
            </div>
            <LineChart
              ariaLabel="Quantile calibration on held-out batches"
              series={[
                { name: "observed", color: "var(--s1)", points: h.calibration.map((c) => ({ x: c.level, y: c.observed })) },
                { name: "ideal", color: "var(--ink-3)", dashed: true, points: h.calibration.map((c) => ({ x: c.level, y: c.level })) },
              ]}
              xLabel="predicted quantile level"
              yLabel="observed frequency"
              xDomain={[0, 1]}
              yDomain={[0, 1]}
            />
            <p className="small muted">Points above the dashed line at low levels and below it at high levels mean the intervals are too narrow.</p>
          </section>
        </div>
      )}
      <div className="grid-2">
        <section className="panel">
          <div className="panel-head">
            <h3>What the model reacts to</h3>
            <p>inverse ARD length scale, larger = more sensitive</p>
          </div>
          <HBarChart ariaLabel="Learned input relevance" rows={[...r.relevance].sort((a, b) => b.value - a.value).map((x) => ({ label: x.name, value: x.value }))} />
        </section>
        <section className="panel">
          <div className="panel-head">
            <h3>Noise the model believes in</h3>
          </div>
          <div className="readouts">
            <Readout label="Model noise SD (log)" value={fmt(Math.sqrt(r.summary.noiseVariance))} sub="unexplained job-to-job scatter" />
            {r.noiseCheck && <Readout label="True job-to-job SD (log)" value={fmt(Math.sqrt(r.noiseCheck.trueNoiseVar))} sub="specimen noise / √n + batch effect" />}
          </div>
          {r.noiseCheck && (
            <p className="small">
              {r.noiseCheck.modelNoiseVar > 2 * r.noiseCheck.trueNoiseVar ? (
                <span className="callout warn" style={{ display: "block" }}>
                  The model's noise is {fmt(r.noiseCheck.modelNoiseVar / r.noiseCheck.trueNoiseVar, 2)}× the true noise variance: it treats part of what it has not learned yet as random scatter. Robust objectives then hedge against that scatter instead of against the feedstock.
                </span>
              ) : (
                <span className="callout" style={{ display: "block" }}>The model's noise estimate is close to the true scatter.</span>
              )}
            </p>
          )}
          <p className="small muted">
            Trained on {r.summary.nStrength} good jobs of {r.summary.nJobs} in {r.seconds.toFixed(1)} s · log marginal likelihood {fmt(r.summary.lml)}
          </p>
        </section>
      </div>
    </>
  );
}

export function ModelView() {
  const { state } = useApp();
  const { train, busy, error } = useTrainer();
  const nGood = state.jobs.filter((j) => j.success && j.strengthMPa > 0).length;
  const stale = state.train && state.trainedOn !== state.jobs.length;
  return (
    <>
      <header className="view-head">
        <h2>Learn from the campaign</h2>
        <p>
          A Gaussian process learns the gap between a simple physics estimate (rule of mixtures × infill × raster angle) and
          the measured log strength; gradient-boosted trees with isotonic calibration learn which prints fail. Both run in your browser.
        </p>
      </header>
      {state.jobs.length === 0 ? (
        <Empty title="No data to learn from">
          <span>Run a campaign or import data on the Campaign step first.</span>
          <a className="button" href="#campaign">Go to Campaign</a>
        </Empty>
      ) : (
        <section className="panel">
          <div className="actions">
            <button type="button" className="primary" onClick={() => train()} disabled={!!busy || nGood < 5}>
              {state.train ? "Retrain" : "Train model"}
            </button>
            <span className="small muted">
              {state.jobs.length} jobs, {nGood} with a strength{nGood < 5 ? " (need at least 5)" : ""}
            </span>
            {stale && <span className="pill warn">data changed since training</span>}
          </div>
          {busy && <Progress fraction={busy.fraction} label={busy.label} />}
          {error && <div className="callout bad">{error}</div>}
        </section>
      )}
      {state.train && <Results r={state.train} synthetic={state.dataSource === "synthetic"} />}
    </>
  );
}
