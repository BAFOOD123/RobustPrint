/** Study results: the pre-registered experiments of the repository (synthetic), exported from results/full. */

import { BarChart, type BarDatum, HBarChart, LineChart } from "../charts/charts";
import { fmt, SERIES } from "../charts/kit";
import { study, type Triple } from "../data/study";
import { pct } from "../ui/controls";

const REPO = "https://github.com/BAFOOD123/RobustPrint";
const METHODS = ["nominal", "expected", "quantile", "cvar", "chance"] as const;
const LABEL: Record<string, string> = { nominal: "Nominal", expected: "Expected", quantile: "5 % quantile", cvar: "CVaR", chance: "Chance" };

function methodStats(method: string, metric: string): Triple {
  const m = study.e1.methods[method];
  return m?.[metric] ?? [null, null, null];
}

function bars(metric: string, scale = 1): BarDatum[] {
  const out: BarDatum[] = [];
  for (const [group, prefix] of [["learned model", ""], ["perfect model", "oracle_"]] as const)
    METHODS.forEach((m, i) => {
      const [v, lo, hi] = methodStats(prefix + m, metric);
      if (v === null) return;
      out.push({ group, series: LABEL[m], color: SERIES[i], value: v * scale, lo: (lo ?? v) * scale, hi: (hi ?? v) * scale });
    });
  return out;
}

/** Holm-adjusted p value as text. */
const pText = (p: number | null) => (p === null ? "–" : p < 0.001 ? "<0.001" : p.toPrecision(2));
const signed = (v: number, digits: number) => `${v > 0 ? "+" : ""}${fmt(v, digits)}`;

function testFor(method: string, metric: string) {
  return study.e1.tests.find((t) => t.method === method && t.metric === metric);
}

export function Study() {
  const e2 = study.e2;
  const strategies = Object.keys(study.e3);
  return (
    <>
      <header className="view-head">
        <h2>The study behind the lab</h2>
        <p>
          Pre-registered experiments from the repository, run with the Python implementation (version {study.version}, commit{" "}
          <code>{study.commit.slice(0, 7)}</code>). {study.label}. Every number on this page is read from a file in <code>results/full</code>.
        </p>
      </header>
      <section className="panel">
        <div className="panel-head">
          <h3>E1: robust vs nominal settings on unseen batches</h3>
          <p>
            {study.e1.seeds} seeds · {study.e1.trainJobs} training jobs per seed (about {study.e1.meanSuccessfulJobs} printed) · mean and 95 % CI
          </p>
        </div>
        <div className="grid-2">
          <div>
            <h4 className="small">Failure rate</h4>
            <BarChart ariaLabel="Failure rate by objective, learned and perfect model" data={bars("failure_rate")} yLabel="failure rate" yFormat={(v) => `${Math.round(v * 100)}%`} />
          </div>
          <div>
            <h4 className="small">Expected utility (failed part = 0 MPa)</h4>
            <BarChart ariaLabel="Expected utility by objective, learned and perfect model" data={bars("expected_utility")} yLabel="MPa" />
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Objective (learned model)</th>
                <th>Failure rate</th>
                <th>Δ vs nominal (Holm p)</th>
                <th>Weakest 5 % (MPa)</th>
                <th>Δ vs nominal (Holm p)</th>
                <th>Expected utility (MPa)</th>
                <th>Δ vs nominal (Holm p)</th>
              </tr>
            </thead>
            <tbody>
              {METHODS.map((m) => {
                const cells = (["failure_rate", "q05_strength", "expected_utility"] as const).map((metric) => {
                  const [v] = methodStats(m, metric);
                  const t = testFor(m, metric);
                  const scale = metric === "failure_rate" ? 100 : 1;
                  return { v: v === null ? "–" : metric === "failure_rate" ? pct(v) : fmt(v, 3), d: t ? `${signed((t.diff ?? 0) * scale, 2)}${metric === "failure_rate" ? " pts" : ""} (${pText(t.pHolm)})` : "reference", sig: !!t && t.pHolm !== null && t.pHolm < 0.05 };
                });
                return (
                  <tr key={m}>
                    <td>{LABEL[m]}</td>
                    {cells.map((c, i) => [
                      <td key={`v${i}`}>{c.v}</td>,
                      <td key={`d${i}`}>{c.sig ? <b>{c.d}</b> : c.d}</td>,
                    ])}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="small muted">
          Bold = significant after Holm correction across the 24 pre-registered comparisons. Files:{" "}
          <a href={`${REPO}/blob/main/results/full/e1_optimisation/summary.csv`} target="_blank" rel="noreferrer">summary.csv</a>,{" "}
          <a href={`${REPO}/blob/main/results/full/e1_optimisation/tests.csv`} target="_blank" rel="noreferrer">tests.csv</a>.
        </p>
      </section>
      <div className="grid-2">
        <section className="panel">
          <div className="panel-head">
            <h3>E2: learning curves</h3>
            <p>grouped cross-validation by batch, 3 seeds</p>
          </div>
          <LineChart
            ariaLabel="CRPS versus training batches for each model"
            series={Object.entries(e2.crps).map(([name, vals], i) => ({ name, color: SERIES[i % SERIES.length], points: e2.sizes.map((n, k) => ({ x: n, y: vals[k] ?? NaN })).filter((p) => Number.isFinite(p.y)) }))}
            xLabel="training batches (× 4 jobs)"
            yLabel="CRPS, log strength"
          />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model ({e2.sizes[e2.sizes.length - 1]} batches)</th>
                  <th>RMSE (MPa)</th>
                  <th>NLL</th>
                  <th>CRPS</th>
                  <th>90 % coverage</th>
                </tr>
              </thead>
              <tbody>
                {e2.largest.map((row) => (
                  <tr key={row.model}>
                    <td>{row.model}</td>
                    <td>{fmt(row.rmse ?? NaN, 3)}</td>
                    <td>{fmt(row.nll ?? NaN, 3)}</td>
                    <td>{fmt(row.crps ?? NaN, 3)}</td>
                    <td>{pct(row.coverage ?? NaN, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <h3>E3: active learning</h3>
            <p>regret of expected utility, {study.e3Seeds} seeds, mean</p>
          </div>
          <LineChart
            ariaLabel="Regret versus number of jobs for each experiment-selection strategy"
            series={strategies.map((s, i) => ({ name: s, color: SERIES[i % SERIES.length], points: study.e3[s].map((p) => ({ x: p.nJobs, y: p.mean ?? NaN })) }))}
            xLabel="print jobs"
            yLabel="regret (MPa)"
          />
          <p className="small">
            After 52 jobs, regret relative to random selection (paired, 95 % CI):{" "}
            {study.e3Contrasts.map((c, i) => (
              <span key={c.strategy}>
                {i > 0 && "; "}
                {c.strategy} {signed(c.diff ?? NaN, 2)} [{fmt(c.lo ?? NaN, 2)}, {fmt(c.hi ?? NaN, 2)}]
              </span>
            ))}
            . Positive = worse than random.
          </p>
          <h4 className="small">E5: what drives part strength (total Sobol index, ground truth)</h4>
          <HBarChart
            ariaLabel="Sobol total indices of log strength"
            rows={study.e5
              .filter((x) => x.output === "log_strength")
              .sort((a, b) => (b.total ?? 0) - (a.total ?? 0))
              .map((x) => ({ label: x.input, value: x.total ?? 0 }))}
            format={(v) => fmt(v, 2)}
          />
        </section>
      </div>
      <section className="panel">
        <div className="panel-head">
          <h3>E4: where the uncertainty comes from</h3>
          <p>share of the predictive variance of log strength at two recommended settings</p>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Setting</th>
                <th>Feedstock composition</th>
                <th>Model (epistemic)</th>
                <th>Specimen noise</th>
              </tr>
            </thead>
            <tbody>
              {study.e4.map((row) => (
                <tr key={`${row.model}-${row.setting}`}>
                  <td>{String(row.model).replace("ground_truth", "ground truth")}</td>
                  <td>{row.setting}</td>
                  <td>{pct(Number(row.share_input), 0)}</td>
                  <td>{pct(Number(row.share_epistemic), 0)}</td>
                  <td>{pct(Number(row.share_aleatoric), 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small muted">In the ground truth and in the hybrid GP (the model the optimiser uses), batch-to-batch composition is the largest source; the deep ensemble attributes most of its variance to specimen noise instead.</p>
      </section>
      <section className="panel">
        <div className="panel-head">
          <h3>Read before quoting</h3>
        </div>
        <p className="small">
          The simulator uses simplified physics with assumed coefficients and has not been compared with any real recycled
          feedstock. A composition × process interaction (fibre clogging at high flow) was added before the experiments
          because without one robust optimisation cannot matter. Nothing here is validated for flight or safety-critical use.
          See <a href={`${REPO}/blob/main/docs/LIMITATIONS.md`} target="_blank" rel="noreferrer">LIMITATIONS.md</a>.
        </p>
      </section>
    </>
  );
}
