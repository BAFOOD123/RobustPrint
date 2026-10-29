/** Bench: the ground-truth simulator at the current settings, for the mean batch and for a spread of batches. */

import { useMemo, useState } from "react";
import { LineChart } from "../charts/charts";
import { fmt, Tooltip, type TooltipState, useWidth } from "../charts/kit";
import { closure } from "../engine/composition";
import { COMPONENTS } from "../engine/materials";
import { quantile } from "../engine/math";
import { Rng } from "../engine/rng";
import { printJob, sampleBatches, trueOutcome } from "../engine/simulator";
import { PARAMETERS } from "../engine/space";
import { useApp } from "../state";
import { mpa, pct, Readout, SuccessPill } from "../ui/controls";

const N_STRIP = 48;
const N_CURVE_BATCHES = 160;

function Waterfall({ factors, final }: { factors: ReturnType<typeof trueOutcome>["factors"]; final: number }) {
  const steps: { label: string; note: string; factor: number }[] = [
    { label: "Fibre reinforcement", note: "short fibres, poor bonding", factor: factors.fibre },
    { label: "Inclusions", note: "contaminants and unmelted polymer", factor: factors.inclusions },
    { label: "Degradation + hydrolysis", note: "too hot or too wet", factor: factors.chemistry },
    { label: "Interlayer bonding", note: "bead-to-bead healing", factor: factors.bonding },
    { label: "Porosity", note: "voids and under-extrusion", factor: factors.density },
    { label: "Raster angle", note: "bead direction vs load", factor: factors.raster },
    { label: "Infill", note: "load-bearing section", factor: factors.infill },
    { label: "Over-extrusion", note: "blobs and dimensional error", factor: factors.overExtrusion },
  ];
  let value = factors.blend;
  const rows = [{ label: "Blend strength", note: "rule of mixtures × immiscibility", from: 0, to: value, factor: NaN }];
  for (const s of steps) {
    const next = value * s.factor;
    rows.push({ label: s.label, note: s.note, from: value, to: next, factor: s.factor });
    value = next;
  }
  const max = Math.max(...rows.map((r) => Math.max(r.from, r.to))) * 1.05;
  return (
    <div className="waterfall" role="img" aria-label="How the part strength is built up from the blend strength">
      {rows.map((r) => {
        const lo = Math.min(r.from, r.to);
        const hi = Math.max(r.from, r.to);
        const loss = r.to < r.from;
        return (
          <div key={r.label} className="wf-row">
            <span title={r.note}>
              {r.label}
              <span className="muted small"> {Number.isFinite(r.factor) ? `×${r.factor.toFixed(2)}` : ""}</span>
            </span>
            <span className="wf-track">
              <span className="wf-bar" style={{ left: `${(100 * lo) / max}%`, width: `${Math.max((100 * (hi - lo)) / max, 0.6)}%`, background: Number.isNaN(r.factor) ? "var(--s1)" : loss ? "var(--bad)" : "var(--good)" }} />
            </span>
            <span className="wf-value">{mpa(r.to, 1)} MPa</span>
          </div>
        );
      })}
      <div className="small muted">Part strength for the mean batch: {mpa(final)} MPa (bars are to scale; red = loss, green = gain).</div>
    </div>
  );
}

function BatchStrip({ seed }: { seed: number }) {
  const { state } = useApp();
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<TooltipState | null>(null);
  const data = useMemo(() => {
    const rng = new Rng(seed);
    const batches = sampleBatches(state.feedstock, N_STRIP, rng);
    return batches.map((b) => {
      const t = trueOutcome(b.trueComposition, state.settings, b.moisture, b.batchEffect);
      const job = printJob(b, state.settings, rng, 1, t);
      return { b, t, success: job.success, measured: job.strengthMPa[0] };
    });
  }, [state.feedstock, state.settings, seed]);
  const ok = data.filter((d) => d.success);
  const strengths = ok.map((d) => d.measured);
  return (
    <div className="chart" ref={ref}>
      <div className="batch-strip">
        {data.map((d, i) => (
          <div
            key={d.b.id}
            className={`batch${d.success ? "" : " failed"}`}
            tabIndex={0}
            aria-label={`Batch ${i + 1}: ${d.success ? `printed, ${d.measured.toFixed(1)} MPa` : "print failed"}`}
            onPointerEnter={(e) => {
              const r = (ref.current as HTMLDivElement).getBoundingClientRect();
              const er = e.currentTarget.getBoundingClientRect();
              setTip({
                x: er.left - r.left + er.width / 2,
                y: er.top - r.top,
                content: (
                  <>
                    <div className="tip-title">Batch {i + 1}: {d.success ? `${d.measured.toFixed(2)} MPa` : "print failed"}</div>
                    {COMPONENTS.map((c, k) => (
                      <div key={c.name} className="tip-row">
                        <span className="swatch" style={{ background: `var(--s${k + 1})` }} />
                        {c.label} {(d.b.trueComposition[k] * 100).toFixed(1)}%
                      </div>
                    ))}
                    <div className="tip-row muted">moisture {d.b.moisture.toFixed(2)}% · P(print) {pct(d.t.pSuccess, 0)}</div>
                  </>
                ),
              });
            }}
            onPointerLeave={() => setTip(null)}
          >
            {COMPONENTS.map((c, k) => (
              <span key={c.name} style={{ flex: `${d.b.trueComposition[k]} 1 0`, background: `var(--s${k + 1})` }} />
            ))}
          </div>
        ))}
      </div>
      <Tooltip tip={tip} width={width} />
      <div className="small" style={{ marginTop: 8 }}>
        <span className={`pill ${ok.length / data.length >= 0.9 ? "good" : ok.length / data.length >= 0.75 ? "warn" : "bad"}`}>{data.length - ok.length} of {data.length} prints failed</span>{" "}
        {strengths.length > 0 && (
          <span className="muted">
            Good parts: median {mpa(quantile(strengths, 0.5))} MPa, weakest 5 % below {mpa(quantile(strengths, 0.05))} MPa. Red outline = failed print.
          </span>
        )}
      </div>
    </div>
  );
}

function ResponseCurves() {
  const { state } = useApp();
  const [param, setParam] = useState(0);
  const curves = useMemo(() => {
    const p = PARAMETERS[param];
    const batches = sampleBatches(state.feedstock, N_CURVE_BATCHES, new Rng(7));
    const xs = Array.from({ length: 33 }, (_, i) => p.low + ((p.high - p.low) * i) / 32);
    const strength = xs.map((xv) => {
      const s = state.settings.slice();
      s[param] = xv;
      const vals = batches.map((b) => trueOutcome(b.trueComposition, s, b.moisture, b.batchEffect));
      const st = vals.map((v) => v.strengthMPa);
      return { x: xv, s: { y: quantile(st, 0.5), lo: quantile(st, 0.05), hi: quantile(st, 0.95) }, p: { y: vals.reduce((a, v) => a + v.pSuccess, 0) / vals.length, lo: quantile(vals.map((v) => v.pSuccess), 0.05), hi: quantile(vals.map((v) => v.pSuccess), 0.95) } };
    });
    return { p, strength };
  }, [state.feedstock, state.settings, param]);
  const unit = curves.p.unit;
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Response to one setting</h3>
        <label className="inline-field" htmlFor="curve-param">
          Vary
          <select id="curve-param" value={param} onChange={(e) => setParam(Number(e.target.value))}>
            {PARAMETERS.map((p, i) => (
              <option key={p.name} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="small muted">Other settings stay as on the ticket. Lines are medians over {N_CURVE_BATCHES} simulated batches; bands span the 5th to 95th percentile of batches, i.e. the effect of composition uncertainty alone.</p>
      <div className="grid-2">
        <LineChart
          ariaLabel={`Part strength versus ${curves.p.label}`}
          series={[{ name: "strength (good part)", color: "var(--s1)", band: true, points: curves.strength.map((c) => ({ x: c.x, ...c.s })) }]}
          xLabel={`${curves.p.label} (${unit})`}
          yLabel="strength (MPa)"
          marker={state.settings[param]}
          yFormat={(v) => fmt(v, 3)}
        />
        <LineChart
          ariaLabel={`Print success probability versus ${curves.p.label}`}
          series={[{ name: "P(print succeeds)", color: "var(--s2)", band: true, points: curves.strength.map((c) => ({ x: c.x, ...c.p })) }]}
          xLabel={`${curves.p.label} (${unit})`}
          yLabel="P(print succeeds)"
          marker={state.settings[param]}
          yDomain={[0, 1]}
          yFormat={(v) => `${Math.round(v * 100)}%`}
        />
      </div>
    </div>
  );
}

export function Bench() {
  const { state, dispatch } = useApp();
  const mean = closure(state.feedstock.mean);
  const o = useMemo(() => trueOutcome(mean, state.settings, state.feedstock.moistureMedianPct), [mean.join(), state.settings, state.feedstock.moistureMedianPct]);
  return (
    <>
      <header className="view-head">
        <h2>Print bench</h2>
        <p>
          The simulator's ground truth at the settings on your ticket. The top row is the average batch. Below it, every bar is
          a different batch drawn from the same stream, because recycled feedstock is never the same twice.
        </p>
      </header>
      <section className="panel" aria-label="Outcome for the average batch">
        <div className="panel-head">
          <h3>Average batch</h3>
          <SuccessPill p={o.pSuccess} />
        </div>
        <div className="readouts">
          <Readout label="Tensile strength" value={mpa(o.strengthMPa)} unit="MPa" sub={`specimen scatter ±${pct(o.noiseCv, 0)}`} />
          <Readout label="Young's modulus" value={mpa(o.modulusGPa)} unit="GPa" />
          <Readout label="Elongation at break" value={mpa(o.elongationPct, 1)} unit="%" />
          <Readout label="Clogging risk" value={pct(o.pClog, 0)} sub={`demand ${pct(o.demandRatio, 0)} of melt limit`} />
          <Readout label="Warping risk" value={pct(o.pWarp, 0)} sub="PE shrinkage vs bed" />
          <Readout label="Interlayer bond" value={pct(o.bond, 0)} sub={`porosity ${pct(o.porosity, 1)} · unmelted ${pct(o.unmelted, 1)}`} />
        </div>
      </section>
      <div className="grid-2">
        <section className="panel">
          <div className="panel-head">
            <h3>Where the strength goes</h3>
          </div>
          <Waterfall factors={o.factors} final={o.strengthMPa} />
        </section>
        <section className="panel">
          <div className="panel-head">
            <h3>The next {N_STRIP} batches</h3>
            <button type="button" onClick={() => dispatch({ type: "reseed" })}>
              Draw new batches
            </button>
          </div>
          <p className="small muted">Each bar is one batch's true composition (bottom to top: {COMPONENTS.map((c) => c.label).join(", ")}). One part is printed from each batch at your settings.</p>
          <BatchStrip seed={state.seed} />
        </section>
      </div>
      <ResponseCurves />
    </>
  );
}
