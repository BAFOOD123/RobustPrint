/** The job ticket: feedstock distribution and print settings, used by every view. */

import { closure, dirichletStd } from "../engine/composition";
import { COMPONENTS, FEEDSTOCKS } from "../engine/materials";
import { PARAMETERS } from "../engine/space";
import { useApp } from "../state";
import { CompositionBar, Slider } from "./controls";

const PRESET_LABELS: Record<string, string> = { iss_nominal: "ISS mixed waste (nominal)", iss_shifted: "Shifted stream (more fibre and residue)" };

export function Rail() {
  const { state, dispatch } = useApp();
  const fs = state.feedstock;
  const mean = closure(fs.mean);
  const sd = dirichletStd(mean, fs.concentration);
  const setPart = (i: number, value: number) => {
    const raw = mean.slice();
    raw[i] = Math.max(value, 0.005);
    const others = raw.reduce((s, v, k) => (k === i ? s : s + v), 0);
    const scale = (1 - raw[i]) / others;
    const next = raw.map((v, k) => (k === i ? raw[i] : v * scale));
    dispatch({ type: "feedstock", feedstock: { ...fs, name: "custom", mean: closure(next) } });
  };
  const preset = Object.keys(FEEDSTOCKS).find((k) => FEEDSTOCKS[k].mean.every((v, i) => Math.abs(v - mean[i]) < 1e-6) && FEEDSTOCKS[k].concentration === fs.concentration) ?? "custom";
  return (
    <aside className="rail" aria-label="Feedstock and print settings">
      <section className="rail-section">
        <div className="rail-head">
          <h3>Feedstock</h3>
          <span className="eyebrow">next batch is a draw</span>
        </div>
        <label className="inline-field" htmlFor="preset">
          Stream
          <select
            id="preset"
            value={preset}
            onChange={(e) => {
              const f = FEEDSTOCKS[e.target.value];
              if (f) dispatch({ type: "feedstock", feedstock: f });
            }}
          >
            {Object.keys(FEEDSTOCKS).map((k) => (
              <option key={k} value={k}>
                {PRESET_LABELS[k]}
              </option>
            ))}
            {preset === "custom" && <option value="custom">Custom mix</option>}
          </select>
        </label>
        <CompositionBar composition={mean} />
        {COMPONENTS.map((c, i) => (
          <Slider
            key={c.name}
            id={`frac-${c.name}`}
            label={
              <>
                <span className="swatch" style={{ background: `var(--s${i + 1})` }} />
                {c.label} <span className="muted small">{c.description.toLowerCase()}</span>
              </>
            }
            value={mean[i]}
            min={0.005}
            max={0.9}
            step={0.005}
            onChange={(v) => setPart(i, v)}
            format={(v) => `${(v * 100).toFixed(1)}% ± ${(sd[i] * 100).toFixed(1)}`}
          />
        ))}
        <Slider
          id="concentration"
          label="Batch-to-batch consistency"
          value={fs.concentration}
          min={5}
          max={300}
          step={1}
          onChange={(v) => dispatch({ type: "feedstock", feedstock: { ...fs, concentration: v } })}
          format={(v) => `κ = ${v.toFixed(0)}`}
          hint="Dirichlet concentration. Higher means batches are more alike; the ± above is the batch-to-batch SD."
        />
        <Slider
          id="moisture"
          label="Median moisture"
          value={fs.moistureMedianPct}
          min={0.05}
          max={2}
          step={0.05}
          onChange={(v) => dispatch({ type: "feedstock", feedstock: { ...fs, moistureMedianPct: v } })}
          format={(v) => `${v.toFixed(2)} %`}
        />
      </section>
      <section className="rail-section">
        <div className="rail-head">
          <h3>Print settings</h3>
          <button type="button" className="small" onClick={() => dispatch({ type: "settings", settings: [265, 100, 25, 0.16, 100, 0, 100, 10] })}>
            Reset
          </button>
        </div>
        {PARAMETERS.map((p, i) => (
          <Slider
            key={p.name}
            id={`param-${p.name}`}
            label={p.label}
            value={state.settings[i]}
            min={p.low}
            max={p.high}
            step={p.step}
            onChange={(v) => dispatch({ type: "setting", index: i, value: v })}
            format={(v) => `${p.step < 1 ? v.toFixed(2) : v.toFixed(0)} ${p.unit}`}
          />
        ))}
      </section>
    </aside>
  );
}
