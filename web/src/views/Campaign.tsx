/** Campaign: run a virtual print campaign or import real data in the repository CSV schema. */

import { useState } from "react";
import { Histogram } from "../charts/charts";
import { generateCampaign, type Job, parseCsv, toCsv } from "../engine/campaign";
import { safeBounds } from "../engine/optimise";
import { PARAMETERS } from "../engine/space";
import { useApp } from "../state";
import { CompositionBar, Empty, mpa, pct } from "../ui/controls";

function JobsTable({ jobs, showTruth }: { jobs: Job[]; showTruth: boolean }) {
  const shown = jobs.slice(0, 400);
  return (
    <div className="table-wrap scroll-y">
      <table>
        <thead>
          <tr>
            <th>Job</th>
            <th>Batch</th>
            <th style={{ minWidth: 90 }}>Composition</th>
            {PARAMETERS.slice(0, 4).map((p) => (
              <th key={p.name}>
                {p.label.split(" ")[0]} ({p.unit})
              </th>
            ))}
            <th>Printed</th>
            <th>Strength (MPa)</th>
            <th>SD</th>
            {showTruth && <th>True mean</th>}
          </tr>
        </thead>
        <tbody>
          {shown.map((j) => (
            <tr key={j.experimentId}>
              <td className="mono">{j.experimentId}</td>
              <td className="mono">{j.batchId}</td>
              <td>
                <CompositionBar composition={j.composition} />
              </td>
              {j.params.slice(0, 4).map((v, k) => (
                <td key={k}>{PARAMETERS[k].step < 1 ? v.toFixed(2) : v.toFixed(0)}</td>
              ))}
              <td>{j.success ? <span className="pill good">yes</span> : <span className="pill bad">failed</span>}</td>
              <td>{j.success ? mpa(j.strengthMPa) : "–"}</td>
              <td>{j.success ? mpa(j.strengthSdMPa) : "–"}</td>
              {showTruth && <td>{j.trueStrengthMPa !== undefined ? mpa(j.trueStrengthMPa) : "–"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      {jobs.length > shown.length && <p className="small muted" style={{ padding: 8 }}>Showing the first {shown.length} of {jobs.length} jobs.</p>}
    </div>
  );
}

function ImportPanel() {
  const { dispatch } = useApp();
  const [text, setText] = useState("");
  const [result, setResult] = useState<ReturnType<typeof parseCsv> | null>(null);
  const check = (t: string) => setResult(parseCsv(t));
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Use your own measurements</h3>
        <p>Columns as in data/schema.md. Nothing leaves your browser.</p>
      </div>
      <div className="actions">
        <label className="button" htmlFor="csv-file">
          Choose a CSV file
        </label>
        <input
          id="csv-file"
          type="file"
          accept=".csv,text/csv"
          className="sr-only"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const t = await f.text();
            setText(t);
            check(t);
          }}
        />
        <span className="muted small">or paste below</span>
      </div>
      <textarea id="csv-paste" aria-label="Paste CSV" value={text} placeholder="experiment_id,data_origin,batch_id,..." onChange={(e) => setText(e.target.value)} />
      <div className="actions">
        <button type="button" onClick={() => check(text)} disabled={!text.trim()}>
          Validate
        </button>
        {result && result.jobs.length > 0 && (
          <button type="button" className="primary" onClick={() => dispatch({ type: "jobs", jobs: result.jobs, source: "uploaded", label: `${result.jobs.length} imported jobs` })}>
            Use {result.jobs.length} valid jobs
          </button>
        )}
      </div>
      {result && (
        <div className="small" aria-live="polite">
          <p>
            {result.jobs.length} valid row(s), {result.errors.length} error(s), {result.warnings.length} warning(s).
          </p>
          {result.errors.slice(0, 12).map((e) => (
            <div key={e} className="callout bad">{e}</div>
          ))}
          {result.warnings.slice(0, 6).map((w) => (
            <div key={w} className="callout warn">{w}</div>
          ))}
        </div>
      )}
    </section>
  );
}

export function Campaign() {
  const { state, dispatch } = useApp();
  const [nBatches, setNBatches] = useState(40);
  const [perBatch, setPerBatch] = useState(4);
  const [reps, setReps] = useState(3);
  const [showTruth, setShowTruth] = useState(false);
  const [showCsv, setShowCsv] = useState(false);
  const [copied, setCopied] = useState("");
  const jobs = state.jobs;
  const ok = jobs.filter((j) => j.success);
  const run = () => {
    const { low, high } = safeBounds(state.feedstock);
    const out = generateCampaign(state.feedstock, nBatches, perBatch, state.seed, { nReplicates: reps, low, high });
    dispatch({ type: "jobs", jobs: out, source: "synthetic", label: `${nBatches} batches × ${perBatch} settings × ${reps} specimens` });
    dispatch({ type: "reseed" });
  };
  const csv = jobs.length ? toCsv(jobs) : "";
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(csv);
      setCopied("Copied to the clipboard.");
    } catch {
      setShowCsv(true);
      setCopied("Copy was blocked here; select the text below instead.");
    }
  };
  return (
    <>
      <header className="view-head">
        <h2>Virtual print campaign</h2>
        <p>
          A simulated lab works through a space-filling design while new feedstock batches arrive. Each job prints a set of
          tensile specimens from one batch; the lab records the batch's measured composition, which carries analysis error.
        </p>
      </header>
      <section className="panel">
        <div className="panel-head">
          <h3>Plan</h3>
          <p>
            {nBatches * perBatch} print jobs, {nBatches * perBatch * reps} specimens, from the stream on the ticket
          </p>
        </div>
        <div className="actions">
          <label className="inline-field" htmlFor="n-batches">
            Batches
            <input id="n-batches" type="number" min={4} max={150} value={nBatches} onChange={(e) => setNBatches(Math.min(Math.max(Number(e.target.value) || 4, 4), 150))} />
          </label>
          <label className="inline-field" htmlFor="per-batch">
            Settings per batch
            <input id="per-batch" type="number" min={1} max={8} value={perBatch} onChange={(e) => setPerBatch(Math.min(Math.max(Number(e.target.value) || 1, 1), 8))} />
          </label>
          <label className="inline-field" htmlFor="reps">
            Specimens per job
            <input id="reps" type="number" min={1} max={5} value={reps} onChange={(e) => setReps(Math.min(Math.max(Number(e.target.value) || 1, 1), 5))} />
          </label>
          <button type="button" className="primary" onClick={run}>
            Run campaign
          </button>
        </div>
        <p className="small muted">Settings follow a Latin hypercube over the search box (nozzle window limited by the dominant polymer). The study in the repository used 60 batches × 4 settings × 3 specimens.</p>
      </section>
      {jobs.length === 0 ? (
        <Empty title="No data yet">
          <span>Run a campaign above, or import your own CSV below. The Model, Optimise and Experiments steps need data.</span>
        </Empty>
      ) : (
        <section className="panel">
          <div className="panel-head">
            <h3>Lab notebook</h3>
            <span className="actions">
              <span className={`pill ${state.dataSource === "uploaded" ? "neutral" : "warn"}`}>{state.dataSource === "uploaded" ? "imported data" : "synthetic"}</span>
              <span className="small muted">{state.dataLabel}</span>
            </span>
          </div>
          <div className="readouts">
            <div className="readout"><span className="label">Print jobs</span><span className="value num">{jobs.length}</span></div>
            <div className="readout"><span className="label">Printed successfully</span><span className="value num">{ok.length}</span><span className="sub">{pct(ok.length / jobs.length, 0)}</span></div>
            <div className="readout"><span className="label">Batches</span><span className="value num">{new Set(jobs.map((j) => j.batchId)).size}</span></div>
            <div className="readout"><span className="label">Median strength</span><span className="value num">{ok.length ? mpa(ok.map((j) => j.strengthMPa).sort((a, b) => a - b)[Math.floor(ok.length / 2)]) : "–"}</span><span className="sub">MPa, good jobs</span></div>
          </div>
          {ok.length > 2 && <Histogram ariaLabel="Distribution of measured job strengths" dists={[{ name: "measured job mean", color: "var(--s1)", values: ok.map((j) => j.strengthMPa) }]} xLabel="measured tensile strength (MPa)" height={170} bins={24} />}
          <div className="actions">
            <button type="button" onClick={copy}>Copy CSV</button>
            <button type="button" onClick={() => setShowCsv((v) => !v)}>{showCsv ? "Hide CSV" : "Show CSV"}</button>
            {jobs.some((j) => j.trueStrengthMPa !== undefined) && (
              <label className="checks" htmlFor="show-truth">
                <span>
                  <input id="show-truth" type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} /> Reveal the hidden true strength
                </span>
              </label>
            )}
            {copied && <span className="small muted" aria-live="polite">{copied}</span>}
          </div>
          {showCsv && <textarea id="csv-out" aria-label="Campaign CSV" readOnly value={csv} onFocus={(e) => e.currentTarget.select()} />}
          <JobsTable jobs={jobs} showTruth={showTruth} />
        </section>
      )}
      <ImportPanel />
    </>
  );
}
