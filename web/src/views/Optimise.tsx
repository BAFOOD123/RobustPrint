/** Optimise & deploy: recommend settings with each objective and print them on fresh, unseen batches. */

import { useState } from "react";
import { BarChart, type BarDatum, Histogram } from "../charts/charts";
import { fmt } from "../charts/kit";
import { ISS_SHIFTED } from "../engine/materials";
import { OBJECTIVE_LABELS, OBJECTIVES, type Objective } from "../engine/objectives";
import { PARAMETERS } from "../engine/space";
import { useApp } from "../state";
import { Empty, mpa, pct, Progress } from "../ui/controls";
import { run } from "../worker/client";
import type { OptimiseResult, Recommendation } from "../worker/protocol";

const OBJ_COLOR: Record<Objective, string> = { nominal: "var(--s1)", expected: "var(--s2)", quantile: "var(--s3)", cvar: "var(--s4)", chance: "var(--s5)" };
const SHORT: Record<Objective, string> = { nominal: "Nominal", expected: "Expected", quantile: "5 % quantile", cvar: "CVaR", chance: "Chance" };

function shares(r: Recommendation) {
  const p = r.predicted;
  const tot = p.varInput + p.varEpistemic + p.varAleatoric;
  return { input: p.varInput / tot, epistemic: p.varEpistemic / tot, aleatoric: p.varAleatoric / tot };
}

function RecommendationTable({ recs, onApply, scored }: { recs: Recommendation[]; onApply: (r: Recommendation) => void; scored: boolean }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Objective</th>
            <th>Model</th>
            {PARAMETERS.map((p) => (
              <th key={p.name} title={p.label}>
                {p.label.split(" ")[0]}
              </th>
            ))}
            <th title="Predicted P(success) over the feedstock">Pred. print</th>
            {scored ? (
              <>
                <th>Failed (simulated)</th>
                <th>Weakest 5 % (simulated)</th>
                <th>Expected utility (simulated)</th>
              </>
            ) : (
              <>
                <th>Pred. weakest 5 %</th>
                <th>Pred. CVaR 10 %</th>
              </>
            )}
            <th>Uncertainty from feedstock</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {recs.map((r) => {
            const s = shares(r);
            return (
              <tr key={`${r.predictor}-${r.objective}`}>
                <td>
                  <span className="swatch" style={{ background: OBJ_COLOR[r.objective] }} />
                  {SHORT[r.objective]}
                </td>
                <td>{r.predictor === "oracle" ? "perfect" : "learned"}</td>
                {r.params.map((v, k) => (
                  <td key={k}>{PARAMETERS[k].step < 1 ? v.toFixed(2) : v.toFixed(0)}</td>
                ))}
                <td>{pct(r.predicted.pSuccess, 0)}</td>
                {scored ? (
                  <>
                    <td>{pct(r.deployed.failureRate, 1)}</td>
                    <td>{mpa(r.deployed.q05Strength)}</td>
                    <td>
                      <b>{mpa(r.deployed.expectedUtility)}</b>
                    </td>
                  </>
                ) : (
                  <>
                    <td>{mpa(r.predicted.quantile)}</td>
                    <td>{mpa(r.predicted.cvar)}</td>
                  </>
                )}
                <td>{pct(s.input, 0)}</td>
                <td>
                  <button type="button" onClick={() => onApply(r)} title="Copy these settings to the ticket and look at them on the Bench">
                    Use
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Optimise() {
  const { state, dispatch } = useApp();
  const [objectives, setObjectives] = useState<Objective[]>(["nominal", "expected", "cvar"]);
  const imported = state.dataSource === "uploaded";
  const [which, setWhich] = useState<"model" | "oracle" | "both">(imported ? "model" : state.train ? "both" : "oracle");
  const [budget, setBudget] = useState(1500);
  const [deployBatches, setDeployBatches] = useState(1000);
  const [shifted, setShifted] = useState(false);
  const [busy, setBusy] = useState<{ fraction: number; label: string } | null>(null);
  const [error, setError] = useState("");
  const [histFor, setHistFor] = useState<"model" | "oracle">("model");
  // With imported data the simulator's truth says nothing about the real material: decide with the
  // learned model only, and report its predictions instead of simulated deployment.
  const choice = imported ? "model" : which;
  const needsModel = choice !== "oracle";
  const go = async () => {
    setError("");
    const kinds: ("model" | "oracle")[] = choice === "both" ? ["model", "oracle"] : [choice];
    try {
      const results: Recommendation[] = [];
      let seconds = 0;
      for (const [k, kind] of kinds.entries()) {
        const res = await run<OptimiseResult>(
          { kind: "optimise", objectives, predictor: kind, feedstock: state.feedstock, deployFeedstock: shifted ? ISS_SHIFTED : state.feedstock, budget, nCompositions: 48, deployBatches, seed: state.seed },
          (f, label) => setBusy({ fraction: (k + f) / kinds.length, label: `${kind === "oracle" ? "Perfect model" : "Learned model"}: ${label}` }),
        );
        results.push(...res.recommendations);
        seconds += res.seconds;
      }
      dispatch({ type: "optimised", result: { kind: "optimise", recommendations: results, seconds } });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const recs = state.optimise?.recommendations ?? [];
  const predictors = [...new Set(recs.map((r) => r.predictor))];
  const histPredictor = predictors.includes(histFor) ? histFor : predictors[0];
  const groups = (r: Recommendation) => (r.predictor === "oracle" ? "perfect model" : "learned model");
  const bar = (metric: (r: Recommendation) => number, label: (v: number) => string): BarDatum[] =>
    recs.map((r) => ({ group: groups(r), series: SHORT[r.objective], color: OBJ_COLOR[r.objective], value: metric(r), label: label(metric(r)) }));
  return (
    <>
      <header className="view-head">
        <h2>Optimise and deploy</h2>
        <p>
          Each objective picks print settings for the <em>next</em> batch, whose composition is unknown. The nominal objective
          plans for the average batch; the others plan for the whole distribution on the ticket. The chosen settings are then
          printed once on each of many fresh batches and scored with the simulator's hidden truth.
        </p>
      </header>
      <section className="panel">
        <div className="checks" role="group" aria-label="Objectives">
          {OBJECTIVES.map((o) => (
            <label key={o} htmlFor={`obj-${o}`}>
              <input
                id={`obj-${o}`}
                type="checkbox"
                checked={objectives.includes(o)}
                onChange={(e) => setObjectives((cur) => (e.target.checked ? OBJECTIVES.filter((x) => x === o || cur.includes(x)) : cur.filter((x) => x !== o)))}
              />
              <span className="swatch" style={{ background: OBJ_COLOR[o] }} />
              {OBJECTIVE_LABELS[o]}
            </label>
          ))}
        </div>
        <div className="actions">
          <label className="inline-field" htmlFor="which">
            Decide with
            <select id="which" value={choice} disabled={imported} onChange={(e) => setWhich(e.target.value as typeof which)}>
              <option value="model">the learned model</option>
              <option value="oracle">a perfect model (the hidden truth)</option>
              <option value="both">both, side by side</option>
            </select>
          </label>
          <label className="inline-field" htmlFor="budget">
            Search effort
            <select id="budget" value={budget} onChange={(e) => setBudget(Number(e.target.value))}>
              <option value={800}>quick (800 evaluations)</option>
              <option value={1500}>standard (1 500)</option>
              <option value={3000}>thorough (3 000)</option>
            </select>
          </label>
          <label className="inline-field" htmlFor="deploy-n">
            Fresh batches
            <select id="deploy-n" value={deployBatches} onChange={(e) => setDeployBatches(Number(e.target.value))}>
              <option value={300}>300</option>
              <option value={1000}>1 000</option>
              <option value={2000}>2 000</option>
            </select>
          </label>
          <label className="checks" htmlFor="shifted">
            <span>
              <input id="shifted" type="checkbox" checked={shifted} onChange={(e) => setShifted(e.target.checked)} /> deploy on a shifted stream (stress test)
            </span>
          </label>
          <button type="button" className="primary" onClick={go} disabled={!!busy || objectives.length === 0 || (needsModel && !state.train)}>
            Optimise and deploy
          </button>
        </div>
        <p className="small muted">All objectives require a predicted print success of at least 90 %. Strengths are for parts that print; expected utility counts a failed print as 0 MPa. &ldquo;Simulated&rdquo; columns come from printing each setting once on every fresh batch in the simulator.</p>
        {imported && <div className="callout warn">Imported data: the simulator knows nothing about your material, so recommendations are shown with the learned model's predictions only. Print them to find out how they really do.</div>}
        {needsModel && !state.train && <div className="callout warn">The learned model is not trained yet. Train it on the Model step, or decide with the perfect model.</div>}
        {busy && <Progress fraction={busy.fraction} label={busy.label} />}
        {error && <div className="callout bad">{error}</div>}
      </section>
      {recs.length === 0 ? (
        <Empty title="No recommendations yet">
          <span>Choose objectives and press Optimise and deploy. With the perfect model this takes a few seconds, with the learned model about three seconds per objective.</span>
        </Empty>
      ) : (
        <>
          {!imported && (
            <div className="grid-2">
              <section className="panel">
                <div className="panel-head">
                  <h3>Failed prints on fresh batches</h3>
                </div>
                <BarChart ariaLabel="Failure rate by objective" data={bar((r) => r.deployed.failureRate, (v) => pct(v, 1))} yLabel="failure rate" yFormat={(v) => `${Math.round(v * 100)}%`} />
              </section>
              <section className="panel">
                <div className="panel-head">
                  <h3>Expected utility</h3>
                  <p>mean strength, failed part = 0 MPa</p>
                </div>
                <BarChart ariaLabel="Expected utility by objective" data={bar((r) => r.deployed.expectedUtility, (v) => fmt(v, 3))} yLabel="MPa" />
              </section>
            </div>
          )}
          <section className="panel">
            <div className="panel-head">
              <h3>{imported ? "Recommended settings" : "Recommended settings and what happened in the simulator"}</h3>
              <p>{state.optimise?.seconds.toFixed(1)} s of compute</p>
            </div>
            <RecommendationTable
              recs={recs}
              scored={!imported}
              onApply={(r) => {
                dispatch({ type: "settings", settings: r.params.map((v, k) => Number(v.toFixed(PARAMETERS[k].step < 1 ? 2 : 0))) });
                window.location.hash = "bench";
              }}
            />
          </section>
          {!imported && (
            <section className="panel">
              <div className="panel-head">
                <h3>Strength of the parts that printed</h3>
                {predictors.length > 1 && (
                  <label className="inline-field" htmlFor="hist-for">
                    Settings from
                    <select id="hist-for" value={histPredictor} onChange={(e) => setHistFor(e.target.value as "model" | "oracle")}>
                      <option value="model">the learned model</option>
                      <option value="oracle">the perfect model</option>
                    </select>
                  </label>
                )}
              </div>
              <Histogram
                ariaLabel="Realised strength distributions"
                xLabel="realised tensile strength (MPa)"
                dists={recs.filter((r) => r.predictor === histPredictor).map((r) => ({ name: SHORT[r.objective], color: OBJ_COLOR[r.objective], values: r.deployed.strengths }))}
              />
            </section>
          )}
        </>
      )}
    </>
  );
}
