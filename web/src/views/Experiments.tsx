/** Next experiments: propose informative print jobs, print them in the simulator, retrain, repeat. */

import { useState } from "react";
import { LineChart } from "../charts/charts";
import { STRATEGIES, STRATEGY_LABELS, type Strategy } from "../engine/acquisition";
import { CSV_COLUMNS, printJobs } from "../engine/campaign";
import { closure } from "../engine/composition";
import { COMPONENTS } from "../engine/materials";
import { deriveSeed, Rng } from "../engine/rng";
import { CONFIG, type Batch } from "../engine/simulator";
import { PARAMETERS } from "../engine/space";
import { useApp } from "../state";
import { CompositionBar, Empty, mpa, pct, Progress } from "../ui/controls";
import { run } from "../worker/client";
import type { ProposeResult } from "../worker/protocol";
import { useTrainer } from "./ModelView";

/** Concentration of the error between a target blend and the blend actually made (as in the Python study). */
const BLEND_CONCENTRATION = 200;

export function Experiments() {
  const { state, dispatch } = useApp();
  const { train, busy: training } = useTrainer();
  const [strategy, setStrategy] = useState<Strategy>("targeted_ivr");
  const [n, setN] = useState(4);
  const [busy, setBusy] = useState<{ fraction: number; label: string } | null>(null);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState("");
  const [template, setTemplate] = useState("");
  const synthetic = state.dataSource === "synthetic";

  const propose = async () => {
    setError("");
    try {
      const res = await run<ProposeResult>({ kind: "propose", strategy, n, feedstock: state.feedstock, seed: deriveSeed(state.seed, state.rounds.length) }, (fraction, label) => setBusy({ fraction, label }));
      dispatch({ type: "proposed", result: res });
      const ok = state.jobs.filter((j) => j.success).length;
      if (synthetic && state.rounds.at(-1)?.nJobs !== state.jobs.length) {
        dispatch({ type: "round", round: { round: state.rounds.length, nJobs: state.jobs.length, nSuccess: ok, strategy, ...res.quality, cumulativeCost: state.alCost } });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const printProposals = async () => {
    const proposals = state.proposals;
    if (!proposals) return;
    const rng = new Rng(deriveSeed(state.seed, 77, state.jobs.length));
    const start = state.jobs.length;
    const batches: Batch[] = proposals.map((p, i) => {
      const trueComposition = closure(rng.dirichlet(p.composition.map((v) => Math.max(BLEND_CONCENTRATION * v, 1e-3))).map((v) => Math.max(v, 1e-6)));
      const reportedComposition = closure(rng.dirichlet(trueComposition.map((v) => Math.max(300 * v, 1e-3))).map((v) => Math.max(v, 1e-6)));
      return { id: `AL${String(start + i).padStart(4, "0")}`, trueComposition, reportedComposition, moisture: p.moisture, batchEffect: CONFIG.batchEffectSd * rng.normal() };
    });
    const newJobs = printJobs(batches, proposals.map((p) => p.params), rng, 3, start, "AL");
    dispatch({ type: "addCost", cost: proposals.reduce((s, p) => s + p.cost, 0) });
    dispatch({ type: "appendJobs", jobs: newJobs });
    await train([...state.jobs, ...newJobs]);
  };

  const copyTemplate = async () => {
    const proposals = state.proposals ?? [];
    const lines = [CSV_COLUMNS.join(",")];
    proposals.forEach((p, i) => {
      lines.push([`NEXT-${i + 1}`, "experimental", `NEW-${i + 1}`, "", "planned by RobustPrint", ...p.composition.map((v) => v.toFixed(3)), "", ...p.params.map((v, k) => (PARAMETERS[k].step < 1 ? v.toFixed(2) : v.toFixed(0))), "", "", "", "", "", "3", ""].join(","));
    });
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied("Copied. Fill in the measured composition and results after printing.");
      setTemplate("");
    } catch {
      setCopied("Copying is blocked here; select the text below instead.");
      setTemplate(lines.join("\n"));
    }
  };

  const rounds = state.rounds;
  return (
    <>
      <header className="view-head">
        <h2>Next experiments</h2>
        <p>
          Pick the print jobs that should teach the model the most about the settings it would recommend now. Targeted
          variance reduction spends experiments where the robust (CVaR) recommendation is most uncertain. In the simulator,
          the proposed jobs can be printed straight away, and the true quality of each round's recommendation is tracked.
        </p>
      </header>
      {!state.train ? (
        <Empty title="Needs a trained model">
          <span>Train a model on the Model step first.</span>
          <a className="button" href="#model">Go to Model</a>
        </Empty>
      ) : (
        <section className="panel">
          <div className="actions">
            <label className="inline-field" htmlFor="strategy">
              Strategy
              <select id="strategy" value={strategy} onChange={(e) => setStrategy(e.target.value as Strategy)}>
                {STRATEGIES.map((s) => (
                  <option key={s} value={s}>
                    {STRATEGY_LABELS[s]}
                  </option>
                ))}
              </select>
            </label>
            <label className="inline-field" htmlFor="n-next">
              Jobs
              <input id="n-next" type="number" min={1} max={8} value={n} onChange={(e) => setN(Math.min(Math.max(Number(e.target.value) || 1, 1), 8))} />
            </label>
            <button type="button" className="primary" onClick={propose} disabled={!!busy || !!training}>
              Propose experiments
            </button>
            {state.proposals && synthetic && (
              <button type="button" onClick={printProposals} disabled={!!busy || !!training}>
                Print them and retrain
              </button>
            )}
            {state.proposals && (
              <button type="button" onClick={copyTemplate}>
                Copy as a CSV template
              </button>
            )}
          </div>
          {copied && <span className="small muted">{copied}</span>}
          {template && <textarea readOnly value={template} rows={Math.min(template.split("\n").length + 1, 10)} aria-label="CSV template" onFocus={(e) => e.currentTarget.select()} />}
          {(busy || training) && <Progress fraction={(busy ?? training)!.fraction} label={(busy ?? training)!.label} />}
          {error && <div className="callout bad">{error}</div>}
        </section>
      )}
      {state.proposals && (
        <section className="panel">
          <div className="panel-head">
            <h3>Proposed jobs</h3>
            {state.lastPropose && (
              <p>
                current robust setting: true expected utility {mpa(state.lastPropose.quality.trueExpectedUtility)} MPa, {pct(1 - state.lastPropose.quality.truePSuccess, 1)} failures
              </p>
            )}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th style={{ minWidth: 110 }}>Target blend</th>
                  {PARAMETERS.map((p) => (
                    <th key={p.name}>{p.label.split(" ")[0]}</th>
                  ))}
                  <th>Predicted (MPa)</th>
                  <th>Pred. print</th>
                  <th>Cost</th>
                </tr>
              </thead>
              <tbody>
                {state.proposals.map((p, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td title={COMPONENTS.map((c, k) => `${c.label} ${(p.composition[k] * 100).toFixed(1)}%`).join(", ")}>
                      <CompositionBar composition={p.composition} />
                    </td>
                    {p.params.map((v, k) => (
                      <td key={k}>{PARAMETERS[k].step < 1 ? v.toFixed(2) : v.toFixed(0)}</td>
                    ))}
                    <td>
                      {mpa(p.predMedian)} <span className="muted">({mpa(p.predLo, 1)}–{mpa(p.predHi, 1)})</span>
                    </td>
                    <td>{pct(p.pSuccess, 0)}</td>
                    <td>{p.cost.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">Cost = 1 setup unit + relative print time. Hover a blend for its fractions.</p>
        </section>
      )}
      {synthetic && rounds.length > 1 && (
        <section className="panel">
          <div className="panel-head">
            <h3>Is it learning?</h3>
            <p>true expected utility of the robust recommendation after each round</p>
          </div>
          <LineChart
            ariaLabel="Regret of the robust recommendation by number of jobs"
            series={[{ name: "regret vs perfect model (CVaR setting)", color: "var(--s1)", points: rounds.map((r) => ({ x: r.nJobs, y: r.regretUtility })) }]}
            xLabel="print jobs in the data"
            yLabel="regret (MPa)"
          />
          <p className="small muted">Regret = true expected utility of the perfect model's CVaR setting minus that of the current recommendation (negative when the recommendation does better on expected utility), as in the repository's study. One campaign is noisy: compare strategies on the <a href="#study">Study results</a> step, which averages 20 seeds.</p>
        </section>
      )}
    </>
  );
}
